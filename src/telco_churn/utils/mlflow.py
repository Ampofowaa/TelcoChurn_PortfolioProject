"""Shared MLflow tracking-URI resolution and model-registry lookups.

resolve_model_run_id/resolve_logged_model_id/load_model_promotion_bars live
here, rather than in evaluate.py (their original, single-caller home),
because threshold.py's dev-OOF screen needs all three too: evaluate.py
already imports from threshold.py (CostScenario, costs_config_hash,
load_costs_config), so threshold.py importing these back from evaluate.py
would be a circular import. Neither module owns the other's need for a
plain MLflow-registry lookup or a policy-config loader, so both live in this
shared, import-safe location instead.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import mlflow
import mlflow.tracking
from mlflow.entities import Experiment
from omegaconf import DictConfig, OmegaConf
from pandas.errors import Pandas4Warning

from telco_churn.utils.paths import get_project_root

if TYPE_CHECKING:
    from telco_churn.models.gate import GateBars

__all__ = [
    "resolve_tracking_uri",
    "resolve_model_run_id",
    "resolve_logged_model_id",
    "load_model_promotion_bars",
    "ensure_experiment_metadata",
    "set_run_description",
    "set_registered_model_description",
    "set_logged_model_description",
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
    "End-to-end training cycle for the IBM Telco Customer Churn model — candidate "
    "comparison (Dummy / LogReg / LightGBM) → feature selection → Optuna tuning → "
    "calibration → sealed-test evaluation → error analysis → model promotion and "
    "registry. Selection criterion: PR-AUC (average precision); guardrails (recall, "
    "Brier, calibration slope) may veto but never promote."
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
    "Training-cycle run. Optuna PR-AUC tuning of the frozen LightGBM spec "
    "and the final tuned pipeline (logged as pyfunc with "
    "training_manifest.json); calibration method selection and the cycle's "
    "single registry registration (calibrate.py); cost-sensitive threshold "
    "derivation and the pre-seal dev-OOF calibration/fairness screen "
    "(threshold.py). This is the version aliased 'challenger', and - if it "
    "clears the sealed-test gate - 'champion'."
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
    """Read the model version's logged_model_id tag (set by calibrate.py at
    registration) — the hop from "the version being evaluated/screened" to
    the LoggedModel entity metrics attach to via log_metric(..., model_id=...,
    dataset=...). ModelVersion.model_id does not auto-populate in OSS MLflow
    3.14 (CLAUDE.md), so this tag is the only supported path. Shared by
    evaluate.py (sealed-test metrics) and threshold.py (dev-OOF screen
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


def load_model_promotion_bars(cfg: DictConfig) -> GateBars:
    """Load configs/model_promotion.yaml directly and construct the GateBars
    decide_promotion applies, and threshold.py's dev-OOF screen checks its
    calibration slope against.

    Loaded by path (OmegaConf.load), never through Hydra's defaults/CLI-
    override composition — gate.py's own module docstring: a bar that
    decides whether a model ships must not be movable by a command-line
    override with no diff and no review, the same reason threshold.py's
    load_policy_thresholds bypasses composition for costs.yaml's derivative.
    """
    # Imported here, not at module level: telco_churn.models.gate is a
    # submodule of telco_churn.models, and importing any submodule of a
    # package runs that package's __init__.py first — which imports
    # telco_churn.models.train, which imports resolve_tracking_uri back from
    # this module. A module-level import here would make that import order
    # circular; this function-local import breaks the cycle the same way
    # this module's docstring already avoids one between threshold.py and
    # evaluate.py.
    from telco_churn.models.gate import GateBars

    path = get_project_root() / str(cfg.paths.model_promotion_config)
    loaded = OmegaConf.load(path)
    assert isinstance(loaded, DictConfig)
    return GateBars(
        pr_auc_bar=float(loaded.pr_auc_bar),
        recall_bar=float(loaded.recall_bar),
        calibration_slope_band=(
            float(loaded.calibration_slope_band[0]),
            float(loaded.calibration_slope_band[1]),
        ),
        pr_auc_materiality_threshold=float(loaded.pr_auc_materiality_threshold),
        brier_non_inferiority_margin=float(loaded.brier_non_inferiority_margin),
    )


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


def set_logged_model_description(model_id: str, description: str) -> None:
    """Set a LoggedModel's overview-page description ('model'/'calibrated_model').

    LoggedModel has no dedicated description field — same 'mlflow.note.content'
    tag convention as runs and experiments, via set_logged_model_tags rather
    than set_tag (LoggedModel tags are not run tags).
    """
    mlflow.tracking.MlflowClient().set_logged_model_tags(
        model_id, {"mlflow.note.content": description}
    )
