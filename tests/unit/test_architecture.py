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
import re
from pathlib import Path

import pytest

from telco_churn.utils.paths import get_project_root

_SRC = get_project_root() / "src" / "telco_churn"
_INTEGRATION = get_project_root() / "tests" / "integration"

# The one module permitted to bind the sealed test partition (CLAUDE.md,
# "Modelling Invariants": test set touched once).
_TEST_PARTITION_OWNER = "models/evaluate.py"


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
    """Every `partition()` call outside evaluate.py must discard the test half.

    `partition(df)` returns (dev_df, test_df). Today artifacts.py, train/common,
    and evaluate.py call it — and only evaluate.py may keep the second element.
    Checking the *binding* rather than the import is what makes this precise:
    the other two legitimately import data.split for the dev half, so an
    import-level rule would either fail on them or be too weak to mean anything.
    """
    offenders: list[str] = []
    for path in _src_modules():
        if _rel(path) == _TEST_PARTITION_OWNER:
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
        "Only models/evaluate.py may bind the test half of partition(); these "
        f"call sites keep it: {offenders}"
    )


def test_test_ids_is_never_called_outside_evaluate() -> None:
    """`test_ids()` — the direct route to the sealed partition — has no caller but evaluate.py.

    split.py defines it and data/__init__.py re-exports it; neither is a call.
    Today the call count across src/ is zero, which is stronger than the rule
    requires — this test pins that so a first caller has to be a deliberate act.
    """
    callers: list[str] = []
    for path in _src_modules():
        if _rel(path) == _TEST_PARTITION_OWNER:
            continue
        for node in ast.walk(_parse(path)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "test_ids"
            ):
                callers.append(f"{_rel(path)}:{node.lineno}")

    assert callers == [], f"test_ids() called outside evaluate.py: {callers}"


# --------------------------------------------------------------------------
# Modelling invariant: evaluate.py resolves by run_id or version, never by alias
# --------------------------------------------------------------------------

_ALIAS_URI = re.compile(r"models:/[^\"'\s]*@")


def test_evaluate_never_resolves_a_model_by_alias() -> None:
    """evaluate.py must not build a `models:/<name>@<alias>` URI.

    An alias is a moving pointer: under Phase 10's weekly retrain `challenger`
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
# (data/split.py's CLI validates via data/validate.py first; features/build.py's
# CLI builds SQL features via features/sql_features.py first, and validates the
# materialized DataFrame via data/validate.py::validate_clean before persisting
# it) — unrelated to PR C. models/ carries no row: calibrate.py, threshold.py, evaluate.py,
# error_analysis.py, and register.py import their shared helpers from
# calibration_metrics.py/artifacts.py/policy_config.py/gate.py/utils, never
# from one another — including calibrate.py <-> register.py, which no longer
# cross-import at all now that register.py's mint step (register_challenger)
# runs as its own CLI, resolving what it needs from an explicit override or
# calibrate.py's reports/calibrate_receipt.json rather than an in-process
# import.
_MAIN_MODULE_IMPORT_ALLOWANCES: dict[str, set[str]] = {
    "data/split.py": {
        "models/dev_features.py",
        "models/evaluate.py",
        "models/train/common.py",
    },
    "features/build.py": {
        "models/dev_features.py",
        "models/error_analysis.py",
        "models/evaluate.py",
        "models/train/candidates.py",
        "models/train/common.py",
        "models/train/feature_audit.py",
        "models/train/feature_selection.py",
        "models/train/log_model.py",
        "models/train/tuning.py",
    },
    "data/validate.py": {"data/ingest.py", "data/split.py", "features/build.py"},
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
