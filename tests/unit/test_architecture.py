"""Executable form of the CLAUDE.md invariants that can be checked structurally.

Every test here scans `src/` rather than exercising behaviour. The distinction
matters: a rule stated in prose is a rule a future refactor can violate without
anything failing, and the invariants below are exactly the ones whose violation
is silent — a leaked test partition still produces plausible metrics, an alias
lookup still returns a model, a swallowed traceback still logs a line.

`tests/unit/test_threshold.py::test_threshold_module_is_leak_free_by_construction`
is the same technique applied to one module; this file generalises it.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import pytest
import yaml
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression

from telco_churn.utils.mlflow import find_dangling_receipts
from telco_churn.utils.paths import get_project_root

_SRC = get_project_root() / "src" / "telco_churn"
_INTEGRATION = get_project_root() / "tests" / "integration"
_ROOT = get_project_root()
_DVC_YAML = _ROOT / "dvc.yaml"

# The modules permitted to bind the sealed test partition (CLAUDE.md,
# "Modelling Invariants": test set touched once). models/sealed_test.py
# (Phase 10a-ii) holds the dataset-agnostic scoring/gate primitives
# extracted out of evaluate.py's own __main__-bearing module; evaluate.py's
# __main__ CLI is now a thin wrapper calling into it for the rare/cold-start
# path.
_TEST_PARTITION_OWNERS = {"models/evaluate.py", "models/sealed_test.py"}


def _src_modules() -> list[Path]:
    """Return every Python module under src/telco_churn, sorted for stable failure output."""
    return sorted(_SRC.rglob("*.py"))


def _rel(path: Path) -> str:
    """Return a module path relative to src/telco_churn, using forward slashes."""
    return path.relative_to(_SRC).as_posix()


def _parse(path: Path) -> ast.Module:
    """Parse a source file into an AST, reading as UTF-8 regardless of platform default."""
    return ast.parse(path.read_text(encoding="utf-8"))


def _is_discard_name(node: ast.expr) -> bool:
    """True when an assignment target is an underscore-prefixed throwaway binding."""
    return isinstance(node, ast.Name) and node.id.startswith("_")


_DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _code_strings(tree: ast.Module) -> list[tuple[int, str]]:
    """Return (lineno, value) for every string literal that is not a docstring.

    Prose that *describes* a forbidden pattern must not trip a rule against it —
    the modules most likely to explain why an alias lookup is banned are exactly
    the modules the rule polices. Comments need no handling: ast drops them.
    """
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


# --------------------------------------------------------------------------
# Modelling invariant: test set touched once
# --------------------------------------------------------------------------


def test_only_evaluate_binds_the_test_partition() -> None:
    """Every `partition()` call outside evaluate.py/sealed_test.py must discard the test half.

    `partition(df)` returns (dev_df, test_df). Today artifacts.py, train/common,
    evaluate.py, and models/sealed_test.py call it — and only evaluate.py/
    models/sealed_test.py may keep the second element. Checking the *binding*
    rather than the import is what makes this precise: the other two
    legitimately import data.split for the dev half, so an import-level rule
    would either fail on them or be too weak to mean anything.
    """
    offenders: list[str] = []
    for path in _src_modules():
        if _rel(path) in _TEST_PARTITION_OWNERS:
            continue
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "partition"
            ):
                continue
            for target in node.targets:
                if isinstance(target, ast.Tuple) and len(target.elts) == 2:
                    if not _is_discard_name(target.elts[1]):
                        offenders.append(f"{_rel(path)}:{node.lineno}")

    assert offenders == [], (
        "Only models/evaluate.py/models/sealed_test.py may bind the test half "
        f"of partition(); these call sites keep it: {offenders}"
    )


# data/split.py's own sealed_test_ids() (Phase 10a-ii) legitimately calls
# test_ids() — it's the customerid-set bookkeeping that keeps "who's in the
# sealed test set" defined in exactly one place, not a bind of the test
# partition's actual feature/label values. models/sealed_test.py does not
# call test_ids() directly — it joins down to sealed_test_ids() instead,
# the same one-choke-point discipline every other partition consumer
# follows — so it needs no entry here despite legitimately touching the test
# partition (see _TEST_PARTITION_OWNERS above for that permission instead).
_TEST_IDS_ALLOWED_CALLERS = {*_TEST_PARTITION_OWNERS, "data/split.py"}


def test_test_ids_is_never_called_outside_evaluate() -> None:
    """`test_ids()` — the direct route to the sealed partition — has no caller but
    evaluate.py and split.py's own sealed_test_ids() accessor.

    split.py defines it and data/__init__.py re-exports it; neither is a call.
    """
    callers: list[str] = []
    for path in _src_modules():
        if _rel(path) in _TEST_IDS_ALLOWED_CALLERS:
            continue
        for node in ast.walk(_parse(path)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "test_ids"
            ):
                callers.append(f"{_rel(path)}:{node.lineno}")

    assert callers == [], f"test_ids() called outside evaluate.py/split.py: {callers}"


# --------------------------------------------------------------------------
# Modelling invariant: evaluate.py resolves by run_id or version, never by alias
# --------------------------------------------------------------------------

_ALIAS_URI = re.compile(r"models:/[^\"'\s]*@")


def test_evaluate_never_resolves_a_model_by_alias() -> None:
    """evaluate.py must not build a `models:/<name>@<alias>` URI.

    An alias is a moving pointer: under Phase 10's monthly retrain `challenger`
    advances every cycle, so resolving through it can evaluate a different model
    than the caller meant. Matching the URI shape rather than the bare word
    'champion' avoids flagging legitimate uses — evaluate.py takes a champion
    *version number* to score the incumbent, which is exactly the compliant form.
    Docstrings are excluded: this module documents the ban, quoting the URI.
    """
    tree = _parse(_SRC / "models" / "evaluate.py")
    hits = [
        f"line {lineno}: {value}"
        for lineno, value in _code_strings(tree)
        if _ALIAS_URI.search(value)
    ]
    assert hits == [], f"evaluate.py resolves a model by alias: {hits}"


# --------------------------------------------------------------------------
# Every compose_config() call site also installs it via activate_config()
# --------------------------------------------------------------------------


def test_every_compose_config_call_site_also_calls_activate_config() -> None:
    """Any module that calls compose_config() must also call activate_config().

    Several modules read their own config via load_config() rather than taking
    cfg as a parameter (features/accessor.py, data/split.py's
    _default_manifest_path) — activate_config() is what lets those readers see
    the same composed, override-applied tree an entry point resolved, rather
    than falling back to config.yaml's on-disk defaults. Forgetting the second
    line is silent: the entry point itself works fine, and only a downstream
    reader sees the wrong path.
    """
    offenders: list[str] = []
    for path in _src_modules():
        tree = _parse(path)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if "compose_config" in calls and "activate_config" not in calls:
            offenders.append(_rel(path))

    assert (
        offenders == []
    ), f"modules call compose_config() without activate_config(): {offenders}"


# --------------------------------------------------------------------------
# cfg.paths.processed_data is read only by its canonical path resolvers
# --------------------------------------------------------------------------

_PROCESSED_DATA_PATH_READERS = {
    "features/accessor.py",
    "data/split.py",
}


def test_processed_data_path_is_read_only_by_its_canonical_resolvers() -> None:
    """cfg.paths.processed_data is resolved only in the modules that own it.

    features/accessor.py and data/split.py are the single accessors for the
    processed-features file and the split manifest respectively — every other
    consumer goes through load_features()/load_split() rather than re-deriving
    the path itself. A new direct read elsewhere is a second, possibly
    diverging path resolution, which is exactly the uncommitted-literal hazard
    this key had before activate_config() existed.
    """
    offenders: list[str] = []
    for path in _src_modules():
        if _rel(path) in _PROCESSED_DATA_PATH_READERS:
            continue
        for node in ast.walk(_parse(path)):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "processed_data"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "paths"
            ):
                offenders.append(f"{_rel(path)}:{node.lineno}")

    assert offenders == [], (
        "cfg.paths.processed_data read outside its canonical resolvers "
        f"{sorted(_PROCESSED_DATA_PATH_READERS)}: {offenders}"
    )


# --------------------------------------------------------------------------
# Code style rules that are mechanical
# --------------------------------------------------------------------------


def test_every_public_module_defines_dunder_all() -> None:
    """CLAUDE.md requires `__all__` in every public module under src/.

    Two exemptions, both meaning 'declares no public surface': `__main__.py`
    entry points, which exist only to be run with `python -m`; and package
    `__init__.py` files that re-export nothing from the project — the empty
    placeholders for phases not yet built (`serving/`, `ui/`, `monitoring/`) and
    the root `__init__.py`, which only resolves `__version__`. A package that
    *does* re-export project symbols, as `data/__init__.py` does, is a real API
    surface and is held to the rule.
    """
    missing: list[str] = []
    for path in _src_modules():
        if path.name == "__main__.py":
            continue
        tree = _parse(path)
        if path.name == "__init__.py" and not any(
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("telco_churn")
            for node in ast.walk(tree)
        ):
            continue
        has_all = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            for node in tree.body
        )
        if not has_all:
            missing.append(_rel(path))

    assert missing == [], f"public modules without __all__: {missing}"


def test_logger_error_inside_except_passes_exc_info() -> None:
    """`logger.error(...)` in an `except` block must attach the traceback.

    Without `exc_info=True` only the message survives, which is precisely the
    situation where a CI or pipeline log is the sole evidence available.
    """
    offenders: list[str] = []
    for path in _src_modules():
        for handler in (
            n for n in ast.walk(_parse(path)) if isinstance(n, ast.ExceptHandler)
        ):
            for node in ast.walk(handler):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "error"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "logger"
                ):
                    continue
                if not any(kw.arg == "exc_info" for kw in node.keywords):
                    offenders.append(f"{_rel(path)}:{node.lineno}")

    assert (
        offenders == []
    ), f"logger.error() inside an except block without exc_info=True: {offenders}"


def test_random_state_literals_are_always_42() -> None:
    """Any hardcoded `random_state=` must be 42.

    Every current call site reads it from config instead, which is stricter than
    this rule — the test guards the case where someone inlines a literal.
    """
    offenders: list[str] = []
    for path in _src_modules():
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "random_state":
                    continue
                if isinstance(kw.value, ast.Constant) and kw.value.value != 42:
                    offenders.append(
                        f"{_rel(path)}:{node.lineno} -> {kw.value.value!r}"
                    )

    assert offenders == [], f"random_state literals other than 42: {offenders}"


# --------------------------------------------------------------------------
# Testing policy
# --------------------------------------------------------------------------

# Waived per CLAUDE.md only when the module's entire __main__ body runs as a
# named subroutine inside another module's subprocess test. Add entries with the
# covering test named, never bare.
_SUBPROCESS_TEST_WAIVERS: dict[str, str] = {}


def _modules_with_main() -> list[Path]:
    """Return src modules containing an `if __name__ == "__main__":` block."""
    found: list[Path] = []
    for path in _src_modules():
        for node in ast.walk(_parse(path)):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            ):
                found.append(path)
                break
    return found


def _dotted(path: Path) -> str:
    """Return the importable dotted name for a src module, as `python -m` takes it."""
    parts = path.relative_to(_SRC.parent).with_suffix("").parts
    if parts[-1] == "__main__":
        parts = parts[:-1]
    return ".".join(parts)


@pytest.mark.parametrize("module_path", _modules_with_main(), ids=_rel)
def test_every_main_module_has_a_subprocess_test(module_path: Path) -> None:
    """Each `__main__` entry point is launched by name from an integration test.

    Direct function calls do not qualify — they skip argparse, OmegaConf
    resolution, dotenv loading and the env-var-to-engine joints that only exist
    at the subprocess boundary.
    """
    dotted = _dotted(module_path)
    if dotted in _SUBPROCESS_TEST_WAIVERS:
        pytest.skip(f"waived: covered by {_SUBPROCESS_TEST_WAIVERS[dotted]}")

    covering = [
        test_file.name
        for test_file in sorted(_INTEGRATION.glob("test_*.py"))
        if dotted in test_file.read_text(encoding="utf-8")
        and "subprocess.run" in test_file.read_text(encoding="utf-8")
    ]
    assert covering, (
        f"no integration test launches `python -m {dotted}` via subprocess.run; "
        "add one, or add a waiver naming the covering test"
    )


# --------------------------------------------------------------------------
# No module imports from a __main__-bearing module, except a named allowance
# --------------------------------------------------------------------------

# Every legitimate cross-import of a __main__-bearing module — the
# data/features stage modules' pre-existing, sanctioned pattern:
# data.split.partition()/features.build's constants already represent
# (test_only_evaluate_binds_the_test_partition polices the test-partition
# half of that pattern separately), plus each other's own __main__ CLI wiring
# (features/build.py's CLI builds SQL features via features/sql_features.py
# first) — unrelated to PR C. data/ingest.py is data/validate.py's only
# remaining importer: ingest() calls validate_raw() itself, since ingest is
# the pipeline's first stage and has no upstream receipt to rely on instead —
# data/split.py and features/build.py no longer re-validate locally, since
# their DVC stages depend on reports/validation_receipt.json for that edge.
# models/ carries no row: calibrate.py, threshold.py, evaluate.py,
# error_analysis.py, and register.py import their shared helpers from
# calibration_metrics.py/artifacts.py/policy_config.py/gate.py/utils, never
# from one another — including calibrate.py <-> register.py, which no longer
# cross-import at all now that register.py's mint step (register_challenger)
# runs as its own CLI, resolving what it needs from an explicit override or
# calibrate.py's reports/calibrate_receipt.json rather than an in-process
# import. models/sealed_test.py (§E6) is a plain, non-__main__-bearing module
# that imports data.split.partition()/sealed_test_ids() and
# features.build.TARGET_COL for its own dataset-agnostic sealed-test scoring
# primitives — evaluate.py no longer imports data.split at all now that
# extraction moved _load_sealed_test_partition() out of it, so
# "models/evaluate.py" moves off data/split.py's row onto sealed_test.py's;
# evaluate.py still imports TARGET_COL directly, so it stays on
# features/build.py's row alongside sealed_test.py.
_MAIN_MODULE_IMPORT_ALLOWANCES: dict[str, set[str]] = {
    "data/split.py": {
        "models/dev_features.py",
        "models/sealed_test.py",
        "models/train/common.py",
    },
    "features/build.py": {
        "models/dev_features.py",
        "models/error_analysis.py",
        "models/evaluate.py",
        "models/sealed_test.py",
        "models/train/candidates.py",
        "models/train/common.py",
        "models/train/feature_audit.py",
        "models/train/feature_selection.py",
        "models/train/log_model.py",
        "models/train/tuning.py",
    },
    "data/validate.py": {"data/ingest.py"},
    "features/sql_features.py": {"features/build.py"},
}


def test_no_module_imports_from_a_dunder_main_bearing_module() -> None:
    """No module under src/ imports a name from a module with a __main__ block, except a named allowance.

    A __main__ block marks a stage entry point — the module a human or `make`
    target runs directly, with its own argparse/compose_config/exit-code
    contract. Importing library code back out of one couples two entry points'
    internals together and is exactly the shape PR C's calibration_metrics.py/
    artifacts.py/policy_config.py/gate.py split existed to undo for
    calibrate.py/threshold.py/evaluate.py/error_analysis.py/register.py. The
    three data/features rows in _MAIN_MODULE_IMPORT_ALLOWANCES are a
    pre-existing, separately-sanctioned pattern (the dev/test-partition and
    raw-data-validation imports) — not a precedent for adding a models/ row.
    """
    main_modules = {_rel(p) for p in _modules_with_main()}
    offenders: list[str] = []
    for path in _src_modules():
        importer = _rel(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("telco_churn.")
            ):
                continue
            parts = node.module.split(".")[1:]  # drop leading "telco_churn"
            target = "/".join(parts) + ".py"
            if target not in main_modules or target == importer:
                continue
            if importer not in _MAIN_MODULE_IMPORT_ALLOWANCES.get(target, set()):
                offenders.append(f"{importer}:{node.lineno} imports from {target}")

    assert offenders == [], (
        "modules import from a __main__-bearing module outside the named "
        f"allowance: {offenders}"
    )


# --------------------------------------------------------------------------
# DVC DAG structural guards — dvc.yaml shared helpers
#
# Phase 8's own params:-coverage guard section (PROJECT_PLAN.md) says to
# "expect the guard to confirm the audit rather than to have prevented the
# need for it" — these walk the real import graph and dvc.yaml text rather
# than re-deriving either from a hand-written list, so a stage row and this
# guard cannot quietly drift apart the way a hand-written list would.
# --------------------------------------------------------------------------


def _dvc_stages() -> dict[str, dict[str, Any]]:
    """Parse dvc.yaml once per call; returns {stage_name: stage_dict}."""
    doc = yaml.safe_load(_DVC_YAML.read_text(encoding="utf-8"))
    stages: dict[str, dict[str, Any]] = doc["stages"]
    return stages


def _stage_entry_relpath(stage_name: str, stage: dict[str, Any]) -> str:
    """Resolve a stage's `cmd: ... -m telco_churn.X.Y` to a src/telco_churn-relative path.

    Handles both a flat module (`telco_churn.data.ingest` -> data/ingest.py)
    and a package run via `__main__.py` (`telco_churn.models.train` ->
    models/train/__main__.py).
    """
    cmd = str(stage["cmd"])
    marker = " -m "
    assert marker in cmd, f"stage {stage_name!r}: cmd has no '-m' entry point: {cmd!r}"
    dotted = cmd.split(marker, 1)[1].strip()
    parts = dotted.split(".")[1:]  # drop leading "telco_churn"
    flat = "/".join(parts) + ".py"
    if (_SRC / flat).exists():
        return flat
    pkg_main = "/".join(parts) + "/__main__.py"
    if (_SRC / pkg_main).exists():
        return pkg_main
    raise AssertionError(
        f"stage {stage_name!r}: cannot resolve an entry module for {cmd!r}"
    )


def _first_party_import_targets(path: Path) -> set[str]:
    """src/telco_churn-relative paths imported via `from telco_churn.X.Y import ...` in path.

    Resolves a package import (`from telco_churn.models import train`) to
    that package's `__init__.py`, since that file's own imports are part of
    the reachable surface too.
    """
    targets: set[str] = set()
    for node in ast.walk(_parse(path)):
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "telco_churn" or node.module.startswith("telco_churn."))
        ):
            continue
        parts = node.module.split(".")[1:]  # drop leading "telco_churn"
        if not parts:
            continue
        flat = "/".join(parts) + ".py"
        pkg_init = "/".join(parts) + "/__init__.py"
        if (_SRC / flat).exists():
            targets.add(flat)
        elif (_SRC / pkg_init).exists():
            targets.add(pkg_init)
    return targets


def _stage_module_closure(stage_name: str, stage: dict[str, Any]) -> set[str]:
    """Transitive first-party import closure reachable from a stage's entry module.

    This is a real Python-import walk, independent of dvc.yaml's own `deps:`
    list — the whole point is to check that list against what the code
    actually imports, not to assume it. A module named in
    _DEPS_COVERAGE_EXEMPT is a walk boundary once reached (its own imports
    are not expanded further) — except the entry module itself, which is
    always fully expanded even if it happens to be in that set
    (features/build.py is both the `features` stage's own entry and every
    *other* stage's TARGET_COL-only import; only the latter is a boundary).
    Applied everywhere this closure is used, including cfg-read collection:
    a stage that reaches build.py solely for a constant should not also
    inherit whatever unrelated cfg.* reads build.py's own other imports
    happen to contain.
    """
    entry = _stage_entry_relpath(stage_name, stage)
    closure: set[str] = {entry}
    stack = list(_first_party_import_targets(_SRC / entry))
    while stack:
        rel = stack.pop()
        if rel in closure:
            continue
        closure.add(rel)
        if rel in _DEPS_COVERAGE_EXEMPT:
            continue
        stack.extend(_first_party_import_targets(_SRC / rel) - closure)
    return closure


def _dvc_declared_deps(stage: dict[str, Any]) -> set[str]:
    """Raw dep path strings a stage declares (directories keep their trailing slash)."""
    return {str(d) for d in stage.get("deps", [])}


def _dvc_declared_params(
    stage: dict[str, Any],
) -> tuple[set[str], dict[str, set[str]]]:
    """Split a stage's params: entries into (whole-file declares, {file: {keys}}).

    A bare string entry (`- configs/costs.yaml`) declares the whole file; a
    dict entry with a key list declares only those keys; a dict entry whose
    value is empty/None also declares the whole file (`- configs/evaluate/default.yaml:`
    with nothing under it — not used today, handled for completeness).
    """
    whole_files: set[str] = set()
    keyed: dict[str, set[str]] = {}
    for entry in stage.get("params", []):
        if isinstance(entry, str):
            whole_files.add(entry)
        elif isinstance(entry, dict):
            for file_path, keys in entry.items():
                if not keys:
                    whole_files.add(str(file_path))
                else:
                    keyed.setdefault(str(file_path), set()).update(str(k) for k in keys)
    return whole_files, keyed


def _stage_declares_path(stage: dict[str, Any], rel_path: str) -> bool:
    """True if rel_path is declared anywhere among a stage's deps: or params: file keys."""
    deps = _dvc_declared_deps(stage)
    if rel_path in deps or f"{rel_path}/" in deps:
        return True
    whole_files, keyed = _dvc_declared_params(stage)
    return rel_path in whole_files or rel_path in keyed


# --------------------------------------------------------------------------
# Item 1 — config files reached by explicit OmegaConf.load(<path>), not cfg
# --------------------------------------------------------------------------

# {function name in models/policy_config.py: the file it loads, resolved from
# configs/config.yaml's own paths.* defaults since the load path is built
# from cfg at runtime, not a literal string this test could read off the AST.
# These reads never go through the composed `cfg` tree (configs/costs.yaml
# and configs/model_promotion.yaml are not in config.yaml's `defaults:`
# list), so test_params_match_reads's cfg.<key> scan structurally cannot see
# them — this is the file-granular guard PROJECT_PLAN.md's stage-8 audit
# calls out as that guard's own stated limitation.
_SIDE_CHANNEL_CONFIG_LOADS: dict[str, str] = {
    "load_costs_config": "configs/costs.yaml",
    "costs_config_hash": "configs/costs.yaml",
    "load_policy_thresholds": "reports/policy/threshold.yaml",
    "load_model_promotion_bars": "configs/model_promotion.yaml",
}

_POLICY_CONFIG_MODULE = "models/policy_config.py"


def test_loaded_configs_are_declared() -> None:
    """Every config file reached via OmegaConf.load(<path>) (never through cfg)
    is declared (deps: or params:) in every stage whose entry module calls
    the function that loads it.

    File-granular by design — key-level completeness for these same files
    inside configs/config.yaml's composed tree is test_params_match_reads's
    job; these three are never in that tree at all. utils/paths.py's own
    OmegaConf.load(configs/config.yaml) is the Hydra composition root every
    stage already reaches through configs/config.yaml's own params: entries
    (or deliberately declares none — features has no params: at all), so
    it is not a side channel and is out of this guard's scope.
    """
    stages = _dvc_stages()
    offenders: list[str] = []
    for stage_name, stage in stages.items():
        closure = _stage_module_closure(stage_name, stage)
        if _POLICY_CONFIG_MODULE not in closure:
            continue
        called_names: set[str] = set()
        for rel in closure:
            if rel == _POLICY_CONFIG_MODULE:
                # Excluded deliberately: policy_config.py's own internal
                # wiring (costs_config_hash calling load_costs_config) would
                # otherwise make every one of these names "called" for any
                # stage that merely imports the module, regardless of
                # whether that stage's own code path uses it.
                continue
            for node in ast.walk(_parse(_SRC / rel)):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called_names.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        called_names.add(node.func.attr)
        for func_name, rel_path in _SIDE_CHANNEL_CONFIG_LOADS.items():
            if func_name in called_names and not _stage_declares_path(stage, rel_path):
                offenders.append(
                    f"{stage_name}: calls {func_name}() but does not declare {rel_path}"
                )

    assert offenders == [], offenders


# --------------------------------------------------------------------------
# Item 2 — cfg.<key> reads vs. dvc.yaml's declared params:, key-granular.
#
# Scope, per PHASE_8_TASKS.md §8's own phasing: only read ⊆ declared is a
# hard-fail here. declared ⊆ read (phantom-param detection) needs alias
# resolution across function boundaries, block expansion, and full
# module-closure collection all working together before it can be trusted
# not to flag a real read as a phantom — left advisory/future work, not
# asserted, exactly as PHASE_8_TASKS.md itself says.
# --------------------------------------------------------------------------

_IDENTITY_TOKEN_SUFFIXES = {
    "run_id",
    "model_version",
    "logged_model_id",
    "eval_run_id",
    "error_analysis_run_id",
}
_EXEMPT_EXACT_CFG_KEYS = {"mlflow.experiment_name"}
_NON_KEY_ATTR_NAMES = {"get", "items", "keys", "values", "pop"}


def _hydra_group_file_map() -> dict[str, str]:
    """{cfg top-level group key: its source YAML file}, from configs/config.yaml's own defaults: list.

    Self-maintaining: reads the same defaults: list Hydra composes from,
    rather than a hand-copied mirror that could silently drift from it.
    Handles the `@` package-override form (`training@logreg: logreg` merges
    configs/training/logreg.yaml under cfg.logreg, not cfg.training).
    """
    doc = yaml.safe_load(
        (_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    )
    mapping: dict[str, str] = {}
    for entry in doc.get("defaults", []):
        if not isinstance(entry, dict):
            continue  # "_self_"
        ((group_spec, filename),) = entry.items()
        group, _, target = group_spec.partition("@")
        target = target or group
        mapping[target] = f"configs/{group}/{filename}.yaml"
    return mapping


def _cfg_key_file_and_key(dotted: str, group_map: dict[str, str]) -> tuple[str, str]:
    """Resolve a cfg.<dotted> read to (source YAML file, key within that file)."""
    head, _, rest = dotted.partition(".")
    if head in group_map:
        return group_map[head], rest
    return "configs/config.yaml", dotted


def _is_env_resolved(file_rel: str, key: str) -> bool:
    """True if key's raw (unresolved) YAML value contains an ${oc.env:...} interpolation."""
    if not key:
        return False
    doc = yaml.safe_load((_ROOT / file_rel).read_text(encoding="utf-8"))
    node: Any = doc
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return isinstance(node, str) and "oc.env:" in node


def _env_resolved_keys(file_rel: str) -> set[str]:
    """Every dotted key in file_rel whose raw YAML value is an ${oc.env:...} interpolation."""
    doc = yaml.safe_load((_ROOT / file_rel).read_text(encoding="utf-8"))

    def walk(node: Any, prefix: str) -> set[str]:
        if isinstance(node, dict):
            found: set[str] = set()
            for k, v in node.items():
                found |= walk(v, f"{prefix}.{k}" if prefix else str(k))
            return found
        if isinstance(node, str) and "oc.env:" in node:
            return {prefix}
        return set()

    return walk(doc, "")


def _is_exempt_cfg_key(dotted: str, file_rel: str, key: str) -> bool:
    """The 4-category exemption list: env-resolved, identity tokens, paths.* (whole prefix), mlflow.experiment_name."""
    if dotted in _EXEMPT_EXACT_CFG_KEYS:
        return True
    if dotted == "paths" or dotted.startswith("paths."):
        return True
    if dotted.split(".")[-1] in _IDENTITY_TOKEN_SUFFIXES:
        return True
    return _is_env_resolved(file_rel, key)


def _attr_chain(node: ast.AST) -> tuple[str, list[str]] | None:
    """If node is an attribute chain rooted at a bare Name, return (root_name, [attrs...]), outermost-first."""
    attrs: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        attrs.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        attrs.reverse()
        return cur.id, attrs
    return None


def _module_cfg_aliases(tree: ast.Module) -> tuple[dict[str, str], set[int]]:
    """{local name: dotted cfg-relative prefix} for every `x = cfg...`-shaped assignment.

    Module-wide rather than per-function-scoped: every function in this
    codebase names its DictConfig parameter `cfg`, and the pervasive
    `group_cfg = cfg.group` pattern (calibrate.py, threshold.py, evaluate.py,
    tuning.py, ...) is what makes a pure `cfg.` scan miss most real reads.
    Fixed-point over assignments so a chain of aliases (`x = cfg.a; y = x.b`)
    resolves fully regardless of source order.

    Also returns the id()s of every assignment RHS node consumed as a pure
    alias target, so _cfg_reads can skip re-recording `ea_cfg = cfg.error_analysis`
    itself as a standalone read of the *whole* error_analysis subtree — the
    leaf reads made later through ea_cfg are what should count, not the
    alias line establishing the shorthand.
    """
    alias_map: dict[str, str] = {}
    alias_rhs_ids: set[int] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                continue
            chain = _attr_chain(node.value)
            if chain is None:
                continue
            root, attrs = chain
            if root == "cfg":
                prefix = ".".join(attrs)
            elif root in alias_map:
                prefix = (
                    ".".join([alias_map[root], *attrs]) if attrs else alias_map[root]
                )
            else:
                continue
            alias_rhs_ids.add(id(node.value))
            target = node.targets[0].id
            if alias_map.get(target) != prefix:
                alias_map[target] = prefix
                changed = True
    return alias_map, alias_rhs_ids


def _is_dunder_main_guard(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
    )


def _cfg_reads(path: Path) -> list[tuple[int, str]]:
    """(lineno, dotted_key) for every resolvable cfg.<...> read in path.

    Resolves direct `cfg.a.b` chains and, via _module_cfg_aliases, the
    codebase's `group_cfg = cfg.group; ...; group_cfg.key` pattern, including
    `group_cfg.get("key", default)`. Does not resolve a cfg subtree passed as
    a bare argument and read through a differently-named parameter inside a
    *different* function/module — the unresolvable-dataflow case
    PROJECT_PLAN.md's own guard-design notes flag as needing the owning file
    declared whole rather than silently passed. That gap is why this test
    only asserts read ⊆ declared, never the reverse.

    `if __name__ == "__main__":` bodies are skipped entirely: that code only
    ever executes for the one stage whose entry point *is* this exact
    module, never for another stage that merely imports a function from it
    (data/split.py's CLI block reads training_setup.test_size/random_seed to
    *build* the split — irrelevant to every stage that only calls
    partition() on an already-built manifest).
    """
    tree = _parse(path)
    alias_map, alias_rhs_ids = _module_cfg_aliases(tree)
    reads: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            if _is_dunder_main_guard(node):
                return
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                chain = _attr_chain(node.func.value)
                if chain is not None:
                    root, attrs = chain
                    prefix = "" if root == "cfg" else alias_map.get(root)
                    if prefix is not None:
                        key = node.args[0].value
                        dotted = f"{prefix}.{key}" if prefix else key
                        reads.append((node.lineno, dotted))
                        return
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if id(node) in alias_rhs_ids:
                return
            chain = _attr_chain(node)
            if chain is not None:
                root, attrs = chain
                if attrs[-1] in _NON_KEY_ATTR_NAMES:
                    return
                prefix = "" if root == "cfg" else alias_map.get(root)
                if prefix is not None:
                    dotted = ".".join([prefix, *attrs]) if prefix else ".".join(attrs)
                    reads.append((node.lineno, dotted))
                return
            self.generic_visit(node)

    Visitor().visit(tree)
    return reads


def test_params_match_reads() -> None:
    """Every resolvable cfg.<key> read in a stage's module closure is declared
    in that stage's dvc.yaml params: (read ⊆ declared, hard-fail).

    Complements test_loaded_configs_are_declared (file-granular, the two
    side-channel-loaded files this key-granular scan structurally cannot
    see since they never enter the composed cfg tree at all).
    """
    group_map = _hydra_group_file_map()
    stages = _dvc_stages()
    offenders: list[str] = []
    for stage_name, stage in stages.items():
        closure = _stage_module_closure(stage_name, stage)
        whole_files, keyed = _dvc_declared_params(stage)
        seen_this_stage: set[tuple[str, str]] = set()
        for rel in sorted(closure):
            for lineno, dotted in _cfg_reads(_SRC / rel):
                file_rel, key = _cfg_key_file_and_key(dotted, group_map)
                if _is_exempt_cfg_key(dotted, file_rel, key):
                    continue
                if file_rel in whole_files:
                    continue
                declared = keyed.get(file_rel, set())
                if any(key == d or key.startswith(d + ".") for d in declared):
                    continue
                marker = (file_rel, key)
                if marker in seen_this_stage:
                    continue
                seen_this_stage.add(marker)
                offenders.append(
                    f"{stage_name}: {rel}:{lineno} reads cfg.{dotted} "
                    f"({file_rel}::{key}) not declared in params:"
                )

    assert offenders == [], offenders


def test_dvc_yaml_never_declares_an_env_resolved_param() -> None:
    """No stage's params: declares a key (or whole file) whose raw YAML value
    is an ${oc.env:...} interpolation, e.g. configs/config.yaml's
    mlflow.tracking_uri or database.url.

    DVC's own YAML parser has no notion of Hydra's ${oc.env:...} syntax, so a
    declared env-resolved key hashes the literal, unresolved interpolation
    string forever -- unchanged regardless of which MLflow/Postgres instance
    is actually live -- silently un-tracking the param while looking handled
    (CLAUDE.md's params: exemptions, category 1). Complements
    test_params_match_reads, which only asserts read subset declared and
    would never flag this: an over-declared env key is never a *missing*
    declaration, so this is the guard for the opposite direction, scoped to
    this one exemption category rather than general phantom-param detection
    (still advisory/future work per that test's own docstring).
    """
    stages = _dvc_stages()
    offenders: list[str] = []
    for stage_name, stage in stages.items():
        whole_files, keyed = _dvc_declared_params(stage)
        for file_rel, keys in keyed.items():
            for key in keys:
                if _is_env_resolved(file_rel, key):
                    offenders.append(
                        f"{stage_name}: declares env-resolved {file_rel}::{key} "
                        "under params: -- DVC hashes the unresolved "
                        "${oc.env:...} literal, never the live value"
                    )
        for file_rel in whole_files:
            for key in sorted(_env_resolved_keys(file_rel)):
                offenders.append(
                    f"{stage_name}: declares {file_rel} whole under params:, "
                    f"which includes env-resolved key {key}"
                )

    assert offenders == [], offenders


# --------------------------------------------------------------------------
# Item 3 — deps: coverage: every module a stage's entry point reaches is
# declared (directly or via a declared parent directory), or named exempt.
# --------------------------------------------------------------------------

# Standing exemptions, each independently justified in PROJECT_PLAN.md's
# stage-by-stage audit:
#  - features/build.py: imported solely for the TARGET_COL constant, whose
#    value is already materialised as a column name in the features parquet
#    (itself a declared dep) — a mismatch raises loudly (KeyError) rather
#    than silently invalidating stale.
#  - models/plots.py: affects only an untracked MLflow figure artifact
#    ("the leaf reason") — no declared out depends on its content.
#  - utils/{db,paths,logging,mlflow}.py: ubiquitous infrastructure, exempted
#    the same way stage 1's "not the whole utils/ folder" carve-out already
#    treats them — utils/stats.py stays *in* scope, since paired_bootstrap_ci
#    can flip a registered/promoted model and is declared per-stage already.
_DEPS_COVERAGE_EXEMPT = {
    "features/build.py",
    "models/plots.py",
    "utils/db.py",
    "utils/paths.py",
    "utils/logging.py",
    "utils/mlflow.py",
}


def test_stage_deps_cover_the_real_import_closure() -> None:
    """Every telco_churn module a stage's entry point transitively imports is
    declared as a dep (directly, or via a declared parent directory), unless
    named in _DEPS_COVERAGE_EXEMPT.

    An undeclared import is a silent missing invalidation edge: change the
    module's behaviour and `dvc repro` reports the stage unaffected.
    """
    stages = _dvc_stages()
    offenders: list[str] = []
    for stage_name, stage in stages.items():
        closure = _stage_module_closure(stage_name, stage)
        deps = _dvc_declared_deps(stage)
        dep_dirs = {d for d in deps if d.endswith("/")}
        for rel in sorted(closure):
            if rel in _DEPS_COVERAGE_EXEMPT:
                continue
            src_dep = f"src/telco_churn/{rel}"
            if src_dep in deps:
                continue
            if any(src_dep.startswith(d) for d in dep_dirs):
                continue
            offenders.append(f"{stage_name}: {rel} reachable but not declared")

    assert offenders == [], offenders


# --------------------------------------------------------------------------
# Item 4 — paths.* correspondence: config paths.* keys that name a
# file-content pointer must match what dvc.yaml actually references, and the
# reports/-subfolder mirror keys must be excluded from both .gitignore and
# .dvcignore verbatim.
# --------------------------------------------------------------------------

# Hand-audited once (same technique as the guards above): for each paths.*
# key that names a concrete file dvc.yaml also references, the literal
# dvc.yaml path(s) that must start with (files: equal) that key's resolved
# value. paths.reports/paths.figures/paths.validation_reports/
# paths.feature_discovery_reports are covered by the mirror-key check below
# instead — they name directories dvc.yaml never references directly.
_PATHS_KEY_DVC_LITERALS: dict[str, list[str]] = {
    "raw_data": ["datasets/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"],
    "processed_data": [
        "datasets/processed/split_manifest.parquet",
        "datasets/processed/reserve_manifest.parquet",
        "datasets/processed/telco_churn_features.parquet",
    ],
    "policy": ["reports/policy/threshold.yaml"],
    "plots": ["reports/plots/decile_lift.csv"],
    "sql_features": ["sql/features/"],
    "costs_config": ["configs/costs.yaml"],
    "model_promotion_config": ["configs/model_promotion.yaml"],
}

# paths.* keys naming a reports/ subfolder that neither DVC nor git may
# track — each must appear, trailing slash included, in both ignore files.
_MIRROR_KEYS = ["figures", "validation_reports", "feature_discovery_reports"]


def _resolved_paths_config() -> dict[str, str]:
    cfg = OmegaConf.load(_ROOT / "configs" / "config.yaml")
    resolved = OmegaConf.to_container(cfg.paths, resolve=True)
    assert isinstance(resolved, dict)
    return {str(k): str(v) for k, v in resolved.items()}


def test_paths_config_matches_dvc_yaml_literals() -> None:
    """configs/config.yaml's paths.* values for file-content pointers agree
    with the literal paths dvc.yaml references for the same files.

    Catches the failure mode where paths.* is repointed but dvc.yaml's own
    literal strings (deps/outs/params file keys, which DVC parses itself and
    cannot follow a Hydra interpolation) are left stale — dvc.yaml would then
    silently hash/track the wrong file.
    """
    paths = _resolved_paths_config()
    offenders: list[str] = []
    for key, literals in _PATHS_KEY_DVC_LITERALS.items():
        resolved = paths[key].rstrip("/")
        for literal in literals:
            matches = literal == resolved or literal.startswith(resolved + "/")
            if not matches:
                offenders.append(
                    f"paths.{key}={paths[key]!r} does not match dvc.yaml literal {literal!r}"
                )

    assert offenders == [], offenders


def test_reports_mirror_keys_excluded_from_git_and_dvc() -> None:
    """Every paths.* reports/-subfolder mirror key is excluded, verbatim and
    with its trailing slash, from both .gitignore and .dvcignore.

    A key present in one but not the other means the folder is tracked by
    exactly one of the two systems — silently, since neither tool errors on
    an asymmetric ignore list.
    """
    paths = _resolved_paths_config()
    gitignore_lines = {
        line.strip()
        for line in (_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    dvcignore_lines = {
        line.strip()
        for line in (_ROOT / ".dvcignore").read_text(encoding="utf-8").splitlines()
    }
    offenders: list[str] = []
    for key in _MIRROR_KEYS:
        entry = paths[key].rstrip("/") + "/"
        if entry not in gitignore_lines:
            offenders.append(f".gitignore is missing {entry!r} (paths.{key})")
        if entry not in dvcignore_lines:
            offenders.append(f".dvcignore is missing {entry!r} (paths.{key})")

    assert offenders == [], offenders


# --------------------------------------------------------------------------
# Item 6 — dvc.yaml invalidation-edge dep-presence: each stage's own upstream
# receipt (the file that carries the "something upstream changed" edge) is
# declared in its deps:.
# --------------------------------------------------------------------------

# {stage: {receipt(s) that stage's own deps: must list}} — the DAG's
# receipt-based invalidation edges, one per adjacent stage pair, per
# PHASE_8_TASKS.md §4's stage-by-stage design.
_EXPECTED_INVALIDATION_EDGES: dict[str, set[str]] = {
    "validate": {"reports/ingest_receipt.json"},
    "split": {"reports/validation_receipt.json"},
    "features": {"reports/validation_receipt.json"},
    "calibrate": {"reports/train_receipt.json"},
    "threshold": {"reports/calibrate_receipt.json"},
    "evaluate": {"reports/calibrate_receipt.json", "reports/policy/threshold.yaml"},
    "error_analysis": {
        "reports/calibrate_receipt.json",
        "reports/test_predictions.parquet",
        "reports/policy/threshold.yaml",
    },
}


def test_dvc_yaml_declares_its_invalidation_edges() -> None:
    """Each stage's deps: contains the upstream receipt(s) that carry its
    invalidation edge from the previous stage.

    A missing receipt dep is exactly the failure PROJECT_PLAN.md's stage-2
    design warns about: an upstream check starts failing (or a hand-edited
    Postgres row invalidates ingest) and `dvc repro` reports the downstream
    stage unaffected because nothing forces it to look.
    """
    stages = _dvc_stages()
    offenders: list[str] = []
    for stage_name, expected in _EXPECTED_INVALIDATION_EDGES.items():
        deps = _dvc_declared_deps(stages[stage_name])
        missing = expected - deps
        if missing:
            offenders.append(
                f"{stage_name}: missing invalidation-edge deps {sorted(missing)}"
            )

    assert offenders == [], offenders


# --------------------------------------------------------------------------
# Item 7 — receipt-resolvability: reports/*_receipt.json identifiers must
# resolve against the live tracking store. Exercises
# utils/mlflow.py::find_dangling_receipts's logic on a throwaway SQLite
# store (matching test_train_tuning.py's tuning_mlflow_uri convention)
# rather than requiring Docker for a unit-test run — the reusable "make
# verify"-callable check itself lives in src/, this only proves it correct.
# --------------------------------------------------------------------------

_REGISTERED_MODEL_NAME = "telco-churn-pipeline"


def test_find_dangling_receipts_flags_unresolvable_identifiers(
    mlflow_test_experiment: Callable[[str], str], tmp_path: Path
) -> None:
    """A receipt pointing at a real run/model is clean; one pointing at an
    identifier the tracking store has never seen (or no longer has, e.g.
    after `docker compose down -v`) is flagged, once per unresolvable key.
    Receipts with no run/model identifiers at all (ingest, validation) are
    silently skipped rather than flagged.
    """
    tracking_uri = mlflow_test_experiment("dangling-receipts-check")

    X = [[0.0], [1.0], [0.0], [1.0]]
    y = [0, 1, 0, 1]
    with mlflow.start_run() as run:
        model_info = mlflow.sklearn.log_model(
            LogisticRegression().fit(X, y), name="model"
        )
        real_run_id = run.info.run_id
        real_logged_model_id = model_info.model_id
    registered = mlflow.register_model(model_info.model_uri, _REGISTERED_MODEL_NAME)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "calibrate_receipt.json").write_text(
        json.dumps(
            {
                "run_id": real_run_id,
                "logged_model_id": real_logged_model_id,
                "model_uri": model_info.model_uri,
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "eval_receipt.json").write_text(
        json.dumps(
            {"model_version": registered.version, "eval_run_id": "not-a-real-run-id"}
        ),
        encoding="utf-8",
    )
    (reports_dir / "error_analysis_receipt.json").write_text(
        json.dumps(
            {"model_version": "999999", "error_analysis_run_id": "also-not-real"}
        ),
        encoding="utf-8",
    )
    (reports_dir / "ingest_receipt.json").write_text(
        json.dumps({"rows_loaded": 7043, "csv_rows": 7043}), encoding="utf-8"
    )

    offenders = find_dangling_receipts(
        reports_dir, tracking_uri, _REGISTERED_MODEL_NAME
    )

    joined = "\n".join(offenders)
    assert "calibrate_receipt.json" not in joined
    assert "ingest_receipt.json" not in joined
    assert "eval_receipt.json: eval_run_id='not-a-real-run-id'" in joined
    assert "error_analysis_receipt.json: model_version='999999'" in joined
    assert (
        "error_analysis_receipt.json: error_analysis_run_id='also-not-real'" in joined
    )
    assert len(offenders) == 3
