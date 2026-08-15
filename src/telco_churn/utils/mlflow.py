"""Shared MLflow tracking-URI resolution, model-registry lookups, and receipts.

resolve_model_run_id/resolve_logged_model_id live here, rather than in
evaluate.py (their original, single-caller home), because threshold.py's
dev-OOF screen needs them too and neither module owns the other's need for a
plain MLflow-registry lookup. (load_model_promotion_bars used to live here
for the same reason — it now lives in models/policy_config.py instead, which
nothing in the models/__init__.py -> train -> utils.mlflow import chain
touches, so the circular-import workaround that justified parking it here no
longer applies.)

A **receipt** (write_train_receipt/write_calibrate_receipt and their read_*
counterparts) is a small local JSON file a pipeline stage writes describing
what it just produced (identifiers only, never evidence), which the next
stage reads by default when no explicit override is given — the DVC
outs:/deps: handshake between chained stages. This is deliberately unlike
this project's other reports/ rule ("never trust a local reports/ path as
the source of truth for a cross-cycle decision" — register.py,
drift_reference.json): that rule protects evidence that must survive a
champion rollback, where a stale local copy would silently pair a decision
with the wrong model version. A receipt carries no evidence — it's a
convenience default for local/CI/DVC chaining. calibrate_receipt.json is
identifiers-only in the strictest sense — {run_id, logged_model_id,
model_uri}, never a model_version, since calibrate.py performs no registry
write of its own — so resolve_model_identifier's default (neither an
explicit run_id nor model_version given) path turns the receipt's run_id
into a model_version via the same registry lookup its explicit-run_id path
already uses, rather than trusting a stored value; eval_receipt.json/
error_analysis_receipt.json do carry model_version (register.py's own
first-invocation bootstrap pointers), and whatever they resolve to is still
the exact same pair an explicit override would supply. The audit/rollback
path (an explicit run_id or model_version override) never reads a receipt
at all.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tracking
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig
from pandas.errors import Pandas4Warning

from telco_churn.utils.paths import get_project_root

__all__ = [
    "resolve_tracking_uri",
    "resolve_model_run_id",
    "resolve_model_version_from_run_id",
    "resolve_logged_model_id",
    "resolve_model_identifier",
    "resolve_calibrated_run",
    "ensure_experiment_metadata",
    "set_run_description",
    "set_registered_model_description",
    "set_logged_model_description",
    "write_train_receipt",
    "read_train_receipt",
    "write_calibrate_receipt",
    "read_calibrate_receipt",
    "write_eval_receipt",
    "read_eval_receipt",
    "write_error_analysis_receipt",
    "read_error_analysis_receipt",
    "find_dangling_receipts",
    "TRAINING_CYCLE_RUN_DESCRIPTION",
]

_SQLITE_PREFIX = "sqlite:///"
_WINDOWS_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")

_mlflow_warnings_suppressed = False


def _suppress_known_mlflow_warnings() -> None:
    """Silence MLflow/pandas warnings already investigated and confirmed benign.

    Mirrors pyproject.toml's [tool.pytest.ini_options] filterwarnings entries
    verbatim, which only take effect inside a pytest session — dvc repro,
    notebooks, and direct CLI runs never register them, so the same three
    filters are applied here too: mlflow 3.14's dataset digest computation
    still calls DataFrame.map(...).all(0) positionally (Pandas4Warning, scoped
    to mlflow's own module so the project's own pandas code isn't silenced too);
    mlflow.data.from_pandas's source-type registry reports two internally
    duplicated LocalArtifactDatasetSource registrations as ambiguous and
    self-resolves; and infer_signature's integer-column advisory doesn't
    apply here since the committed feature columns are never null at
    training time. Idempotent, so safe to call from every resolve_tracking_uri
    invocation rather than once at process start.
    """
    global _mlflow_warnings_suppressed
    if _mlflow_warnings_suppressed:
        return
    warnings.filterwarnings("ignore", category=Pandas4Warning, module=r"mlflow\..*")
    warnings.filterwarnings(
        "ignore",
        message="The specified dataset source can be interpreted in multiple ways",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Hint: Inferred schema contains integer column",
        category=UserWarning,
    )
    _mlflow_warnings_suppressed = True


_EXPERIMENT_DESCRIPTION = (
    "End-to-end training cycle for the IBM Telco Customer Churn model. "
    "Expand for pipeline stages and the model-selection policy.\n\n"
    "**Pipeline:** candidate comparison (Dummy / LogReg / LightGBM) → "
    "feature selection → Optuna hyperparameter tuning → calibration → "
    "cost-sensitive threshold derivation → sealed-test evaluation → "
    "error analysis → model promotion and registry.\n\n"
    "**Selection policy:** PR-AUC (average precision) is the sole "
    "criterion that can *promote* a model. Three guardrails — recall, "
    "Brier, calibration slope — can *veto* a candidate PR-AUC would "
    "otherwise admit, but none of them can promote one on their own."
)
_EXPERIMENT_TAGS = {
    "project": "telco-churn",
    "dataset": "ibm-telco-churn",
    "env": "development",
}

# Shared verbatim by log_model.py, calibrate.py, and threshold.py — all three
# write onto the same run (calibrate.py/threshold.py reuse it by run_id), so
# this is static, idempotent text describing the run as a whole rather than
# a per-step fragment any one of the three would overwrite the others' with.
TRAINING_CYCLE_RUN_DESCRIPTION = (
    "Training-cycle run. LightGBM's hyperparameters are searched against the "
    "frozen feature set, scored on PR-AUC (tuning.py), and the winning "
    "configuration is refit on the full development set and logged as a "
    "deployable model with its full audit trail in training_manifest.json "
    "(log_model.py). calibrate.py then turns the model's raw scores into "
    "honest probabilities and performs the cycle's single registry "
    "registration. threshold.py turns those probabilities into a decision - "
    "the cost-sensitive cutoff at which a customer is worth contacting - "
    "while re-checking that the calibration is still trustworthy and fair "
    "before anything moves further. All four steps write to this same run. "
    "The result is registered as 'challenger', and - if it clears the "
    "sealed-test gate - 'champion'."
)


def _is_absolute_sqlite_path(path: str) -> bool:
    return path.startswith("/") or bool(_WINDOWS_ABS_PATH_RE.match(path))


def resolve_tracking_uri(uri: str) -> str:
    """Resolve relative MLflow tracking URIs to absolute project-rooted locations.

    Any URI with an explicit scheme (http(s)://, postgresql://, ...) is returned
    unchanged, with one exception: 'sqlite:///<relative-path>' contains '://'
    like any other scheme, but the path after the prefix is CWD-relative, not
    scheme-anchored — the exact failure mode this function exists to prevent for
    a bare 'mlruns' path, re-introduced through a different scheme. A relative
    sqlite path is anchored to get_project_root() and re-assembled as
    'sqlite:///<absolute-posix-path>'; an already-absolute sqlite path (leading
    '/', or a Windows drive like 'C:/...') is left unchanged.

    A bare relative path with no scheme at all (e.g. 'mlruns') is anchored to
    get_project_root() and returned as a file:// URI, not a bare OS path — on
    Windows, str(path) yields 'C:\\...\\mlruns', and MLflow's store registry
    reads urlparse's scheme off that string, which is 'c' for a drive letter,
    not a recognized backend.

    Also registers _suppress_known_mlflow_warnings() as a side effect: every
    training-cycle module calls this function (directly, or transitively via
    ensure_experiment_metadata/resolve_model_run_id/resolve_logged_model_id)
    before its first from_pandas/infer_signature call, making this the one
    choke point that reaches pytest, dvc repro, notebooks, and CLI runs alike.
    """
    _suppress_known_mlflow_warnings()
    if uri.startswith(_SQLITE_PREFIX):
        db_path = uri[len(_SQLITE_PREFIX) :]
        if _is_absolute_sqlite_path(db_path):
            return uri
        return _SQLITE_PREFIX + (get_project_root() / db_path).as_posix()
    if "://" in uri:
        return uri
    return (get_project_root() / uri).as_uri()


def resolve_model_run_id(model_version: str, cfg: DictConfig) -> str:
    """Resolve a registered model version to its run_id — the explicit-version rule.

    Sets the MLflow tracking URI as a side effect, so this is safe to call as
    the first MLflow-touching call in a fresh process.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    registered_model_name = str(cfg.mlflow.registered_model_name)
    client = mlflow.tracking.MlflowClient()
    return str(client.get_model_version(registered_model_name, model_version).run_id)


def resolve_logged_model_id(model_version: str, cfg: DictConfig) -> str:
    """Read the model version's logged_model_id tag to find its LoggedModel entity.

    ModelVersion.model_id does not auto-populate in OSS MLflow 3.14, so this
    tag — set by calibrate.py at registration — is the only supported hop
    from "the version being evaluated/screened" to the LoggedModel that
    metrics attach to via log_metric(..., model_id=..., dataset=...). Shared
    by evaluate.py (sealed-test metrics) and threshold.py (dev-OOF screen
    metrics) to attach to the same LoggedModel.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    registered_model_name = str(cfg.mlflow.registered_model_name)
    client = mlflow.tracking.MlflowClient()
    version = client.get_model_version(registered_model_name, model_version)
    model_id = version.tags.get("logged_model_id")
    if not model_id:
        raise ValueError(
            f"Model version {model_version!r} of {registered_model_name!r} has "
            "no logged_model_id tag — calibrate.py's registration step sets "
            "this; re-run models.calibrate before evaluating."
        )
    return str(model_id)


def resolve_model_version_from_run_id(run_id: str, cfg: DictConfig) -> str:
    """Resolve a run_id to its registered model version — the reverse of resolve_model_run_id.

    Needed for the explicit-run_id override path: a human pinning to a
    specific run still needs a concrete model_version for the
    registry-URI/logged_model_id-tag machinery every downstream check
    (check_threshold_provenance, resolve_logged_model_id, model-version
    tagging) already expects. Search is scoped to
    cfg.mlflow.registered_model_name; raises if zero or more than one
    registered version was minted from this run — calibrate.py registers
    exactly one version per run, so either extreme means the run_id is wrong
    or ambiguous and an explicit model_version should be passed instead.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    registered_model_name = str(cfg.mlflow.registered_model_name)
    client = mlflow.tracking.MlflowClient()
    matches = client.search_model_versions(
        f"name='{registered_model_name}' and run_id='{run_id}'"
    )
    if len(matches) == 0:
        raise ValueError(
            f"No registered version of {registered_model_name!r} found for "
            f"run_id={run_id!r} — models.calibrate does not register a "
            "version itself; run `python -m telco_churn.models.register "
            f"register.run_id={run_id}` (or the default no-args form, if "
            "this is the most recent calibration run) first."
        )
    if len(matches) > 1:
        versions = sorted(str(m.version) for m in matches)
        raise ValueError(
            f"Multiple registered versions of {registered_model_name!r} found "
            f"for run_id={run_id!r} ({versions}) — pass an explicit "
            "model_version to disambiguate."
        )
    return str(matches[0].version)


def resolve_model_identifier(
    explicit_run_id: str | None,
    explicit_model_version: str | None,
    cfg: DictConfig,
) -> tuple[str, str, str]:
    """Resolve (run_id, model_version, model_uri) from an explicit override or calibrate.py's receipt, by run id.

    Exactly one of explicit_run_id/explicit_model_version may be given —
    giving both is ambiguous and is never silently resolved in one direction
    over the other (raises ValueError naming both). Giving neither resolves
    run_id from reports/calibrate_receipt.json — the most recent calibration
    run on this machine — and then falls through to the same registry lookup
    as the explicit-run_id path below: the receipt is calibrate.py's own
    single-writer description of a still-unregistered artifact (see
    write_calibrate_receipt's docstring) and never carries a model_version
    itself, so "resolve by run id, not registry version" is not just true of
    the explicit override, it is what the fully-automatic default path does
    too — a registered version (minted by register.py's mint step, run
    separately after calibrate.py) must already exist for the resolved
    run_id, or resolve_model_version_from_run_id raises naming the CLI that
    mints it.

    model_uri is the registry URI (models:/<name>/<version>) on every path —
    an immutable, version-pinned URI, not a moving alias.
    """
    if explicit_run_id is not None and explicit_model_version is not None:
        raise ValueError(
            f"Both an explicit run_id ({explicit_run_id!r}) and an explicit "
            f"model_version ({explicit_model_version!r}) were given — "
            "ambiguous, and never silently resolved in one direction over "
            "the other. Pass exactly one, or neither to resolve from "
            "calibrate.py's receipt."
        )
    registered_model_name = str(cfg.mlflow.registered_model_name)
    if explicit_model_version is not None:
        model_version = str(explicit_model_version)
        run_id = resolve_model_run_id(model_version, cfg)
        model_uri = f"models:/{registered_model_name}/{model_version}"
        return run_id, model_version, model_uri
    run_id = (
        str(explicit_run_id)
        if explicit_run_id is not None
        else str(read_calibrate_receipt(cfg)["run_id"])
    )
    model_version = resolve_model_version_from_run_id(run_id, cfg)
    model_uri = f"models:/{registered_model_name}/{model_version}"
    return run_id, model_version, model_uri


def resolve_calibrated_run(
    explicit_run_id: str | None, cfg: DictConfig
) -> tuple[str, str, str]:
    """Resolve (run_id, model_uri, logged_model_id) for register.py's mint step — never the registry.

    Unlike resolve_model_identifier above, this never looks up a
    model_version: register.py's mint step (register_challenger) is what
    *creates* one, so there is nothing to resolve yet. An explicit
    register.run_id override reads the calibrated_model_id/
    calibrated_model_uri tags calibrate.py sets directly on that run (see
    calibrate.py's _log_calibration_run) — no receipt involved — so
    re-registering an older calibration run, not necessarily the most recent
    one on this machine, works without a local file that only ever describes
    the latest one. Giving no override resolves from
    reports/calibrate_receipt.json instead, the same convenience default
    every other stage's __main__ uses.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    if explicit_run_id is not None:
        run_id = str(explicit_run_id)
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        model_id = run.data.tags.get("calibrated_model_id")
        model_uri = run.data.tags.get("calibrated_model_uri")
        if not model_id or not model_uri:
            raise ValueError(
                f"Run {run_id!r} has no calibrated_model_id/calibrated_model_uri "
                "tags — was models.calibrate ever run for this run_id?"
            )
        return run_id, str(model_uri), str(model_id)
    receipt = read_calibrate_receipt(cfg)
    return (
        str(receipt["run_id"]),
        str(receipt["model_uri"]),
        str(receipt["logged_model_id"]),
    )


def _reports_dir(cfg: DictConfig) -> Path:
    """Return (without creating) the project's reports/ directory."""
    return get_project_root() / str(cfg.paths.reports)


def write_train_receipt(
    run_id: str,
    logged_model_id: str,
    model_uri: str,
    cfg: DictConfig,
) -> None:
    """Write reports/train_receipt.json — the pointer calibrate.py reads by default.

    Identifiers only, never evidence — see this module's docstring. The
    explicit calibration.run_id override never reads this file. Carries the
    same {run_id, logged_model_id, model_uri} triple as
    calibrate_receipt.json — DVC declares this file as the `train` stage's
    out, so it must be self-contained (readable without a live MLflow query)
    for anything that inspects it outside calibrate.py's own resolution path.
    """
    reports_dir = _reports_dir(cfg)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "train_receipt.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "logged_model_id": logged_model_id,
                "model_uri": model_uri,
            },
            f,
            indent=2,
        )


def read_train_receipt(cfg: DictConfig) -> dict[str, Any]:
    """Read reports/train_receipt.json, raising a clear error naming the CLI override fallback."""
    path = _reports_dir(cfg) / "train_receipt.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run `python -m telco_churn.models.train` "
            "first, or pass an explicit "
            "calibration.run_id=<tuning_study_run_id> override."
        )
    with open(path, encoding="utf-8") as f:
        receipt: dict[str, Any] = json.load(f)
    return receipt


def write_calibrate_receipt(
    run_id: str,
    logged_model_id: str,
    model_uri: str,
    cfg: DictConfig,
) -> None:
    """Write reports/calibrate_receipt.json — threshold/evaluate/error_analysis's and register.py's default pointer.

    Never carries a model_version: calibrate.py performs no registry write of
    its own (see that module's docstring), so no version exists yet at the
    point this is written. model_uri is the run-scoped LoggedModel URI
    calibrate.py's own log_model call returned, not a registry URI —
    resolve_model_identifier's "neither given" default path turns this
    receipt's run_id into a model_version by looking it up in the registry
    (resolve_model_version_from_run_id), rather than trusting a value stored
    here, so this file stays calibrate.py's own single-writer description of
    a still-unregistered artifact.
    """
    reports_dir = _reports_dir(cfg)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "calibrate_receipt.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "logged_model_id": logged_model_id,
                "model_uri": model_uri,
            },
            f,
            indent=2,
        )


def read_calibrate_receipt(cfg: DictConfig) -> dict[str, Any]:
    """Read reports/calibrate_receipt.json, raising a clear error naming the CLI override fallback."""
    path = _reports_dir(cfg) / "calibrate_receipt.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run `python -m telco_churn.models.calibrate` "
            "first, or pass an explicit run_id/model_version override."
        )
    with open(path, encoding="utf-8") as f:
        receipt: dict[str, Any] = json.load(f)
    return receipt


def write_eval_receipt(model_version: str, eval_run_id: str, cfg: DictConfig) -> None:
    """Write reports/eval_receipt.json — register.py's bootstrap pointer to this cycle's eval run.

    B1 moves model-version tagging (eval_run_id, the four gate criteria) out
    of evaluate.py and into register.py, which runs after this — so
    register.py needs a channel to "which evaluation run belongs to this
    model_version" for its own first invocation, before any tag exists to
    read it back from. This receipt is that channel, exactly the same
    first-invocation-bootstrap role calibrate_receipt.json already plays for
    threshold.py/evaluate.py/error_analysis.py's own input resolution — never
    trusted again once register.py has tagged eval_run_id onto the version,
    at which point every later read goes through the tag instead (this
    module's docstring's distinction between a receipt and cross-cycle
    evidence).
    """
    reports_dir = _reports_dir(cfg)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "eval_receipt.json", "w", encoding="utf-8") as f:
        json.dump(
            {"model_version": model_version, "eval_run_id": eval_run_id}, f, indent=2
        )


def read_eval_receipt(cfg: DictConfig) -> dict[str, Any]:
    """Read reports/eval_receipt.json, raising a clear error naming the producing command."""
    path = _reports_dir(cfg) / "eval_receipt.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run `python -m telco_churn.models.evaluate` "
            "first for this model version."
        )
    with open(path, encoding="utf-8") as f:
        receipt: dict[str, Any] = json.load(f)
    return receipt


def write_error_analysis_receipt(
    model_version: str, error_analysis_run_id: str, cfg: DictConfig
) -> None:
    """Write reports/error_analysis_receipt.json — register.py's bootstrap pointer to this cycle's error_analysis run.

    Same role as write_eval_receipt above, for the sixth of the six
    model-version tags B1 moves into register.py.
    """
    reports_dir = _reports_dir(cfg)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "error_analysis_receipt.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_version": model_version,
                "error_analysis_run_id": error_analysis_run_id,
            },
            f,
            indent=2,
        )


def read_error_analysis_receipt(cfg: DictConfig) -> dict[str, Any]:
    """Read reports/error_analysis_receipt.json, raising a clear error naming the producing command."""
    path = _reports_dir(cfg) / "error_analysis_receipt.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run `python -m telco_churn.models.error_analysis` "
            "first for this model version."
        )
    with open(path, encoding="utf-8") as f:
        receipt: dict[str, Any] = json.load(f)
    return receipt


def ensure_experiment_metadata(cfg: DictConfig) -> Experiment:
    """Set (or refresh) the experiment description and tags, and return the experiment.

    Sets the tracking URI as a side effect, so this is safe to call as the
    first MLflow-touching call in a fresh process — it replaces the bare
    mlflow.set_experiment(...) call every training-cycle module makes.

    The description is experiment-level, not run-level: MLflow renders
    exactly one 'mlflow.note.content' tag per experiment in the UI, and every
    module in this training cycle (candidates.py, calibrate.py, evaluate.py,
    error_analysis.py) shares one experiment via cfg.mlflow.experiment_name.
    No single module owns what the experiment as a whole represents, so the
    description text lives here — the shared, import-safe location this
    module's docstring already establishes for the same reason — and is
    written from every call site rather than once from whichever module
    happens to run first. set_experiment_tag is idempotent, so the repeated
    writes are cheap and self-healing: the description can never go stale
    behind a single module that stopped being the first one to run.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    exp = mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    client = mlflow.tracking.MlflowClient()
    client.set_experiment_tag(
        exp.experiment_id, "mlflow.note.content", _EXPERIMENT_DESCRIPTION
    )
    for tag_key, tag_val in _EXPERIMENT_TAGS.items():
        client.set_experiment_tag(exp.experiment_id, tag_key, tag_val)
    return exp


def set_run_description(description: str) -> None:
    """Set the active run's overview-page description.

    MLflow renders exactly one 'mlflow.note.content' tag as the run's
    description box, the same convention ensure_experiment_metadata already
    uses one level up. Call from inside an open mlflow.start_run() block.
    Idempotent — safe to call from every module that shares a run (the
    tuning_study run is written by log_model.py, calibrate.py, and
    threshold.py in turn; each sets the same static text).
    """
    mlflow.set_tag("mlflow.note.content", description)


def set_registered_model_description(name: str, description: str) -> None:
    """Set the registered model's description (Model Registry overview page).

    A real description field (client.update_registered_model), not a tag —
    unlike runs/experiments/logged models, which only support the
    'mlflow.note.content' tag convention. Idempotent: safe to call on every
    registration, the same self-healing pattern as ensure_experiment_metadata.
    """
    mlflow.tracking.MlflowClient().update_registered_model(
        name=name, description=description
    )


def find_dangling_receipts(
    reports_dir: Path, tracking_uri: str, registered_model_name: str
) -> list[str]:
    """Resolve every reports/*_receipt.json's identifiers against tracking_uri;
    return one message per identifier that fails to resolve.

    Catches "DVC cache intact, MLflow gone": `dvc repro` trusts a receipt's
    presence and content hash alone, so it reports every stage cached and
    skips re-running even when the tracking store behind those receipts has
    been recreated from scratch (e.g. `docker compose down -v`) — silent
    until something downstream actually tries to load the run or model the
    receipt points at. Not wired to a Makefile target yet (that is Phase 8's
    tooling section); this is the reusable check such a target would call.

    Every receipt shape in this module is covered generically rather than by
    a per-file case: any key ending in "run_id" is resolved via get_run,
    "logged_model_id" via get_logged_model, "model_version" via
    get_model_version(registered_model_name, ...). ingest_receipt.json and
    validation_receipt.json carry neither — nothing to resolve, not a
    dangling reference — so they are silently skipped rather than flagged.
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    offenders: list[str] = []
    for receipt_path in sorted(reports_dir.glob("*_receipt.json")):
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            if not value:
                continue
            try:
                if key == "logged_model_id":
                    client.get_logged_model(str(value))
                elif key == "model_version":
                    client.get_model_version(registered_model_name, str(value))
                elif key.endswith("run_id"):
                    client.get_run(str(value))
                else:
                    continue
            except MlflowException as exc:
                offenders.append(
                    f"{receipt_path.name}: {key}={value!r} unresolvable "
                    f"({exc.error_code})"
                )
    return offenders


def set_logged_model_description(model_id: str, description: str) -> None:
    """Set a LoggedModel's overview-page description ('model'/'calibrated_model').

    LoggedModel has no dedicated description field — same 'mlflow.note.content'
    tag convention as runs and experiments, via set_logged_model_tags rather
    than set_tag (LoggedModel tags are not run tags).
    """
    mlflow.tracking.MlflowClient().set_logged_model_tags(
        model_id, {"mlflow.note.content": description}
    )
