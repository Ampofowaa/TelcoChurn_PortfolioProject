"""MLflow/registry artifact loaders shared across calibrate.py, threshold.py,
evaluate.py, error_analysis.py, and register.py.

Every loader here is a by-run-id/by-version MLflow artifact fetch — no data.split
import, deliberately (PROJECT_PLAN.md's Phase 8 Prerequisites, PR C note):
error_analysis.py imports this module, and error_analysis.py must never become
transitively reachable to data.split (the invariant Phase 8's stage-9 reachability
guard checks for, gated on this extraction). The development-partition loaders
(`load_dev_features`, `load_dev_customer_ids`, `load_dev_partition`) live in the
sibling `models/dev_features.py` instead, imported only by calibrate.py/
threshold.py/register.py — never by error_analysis.py. `load_test_*`/
`_load_sealed_test_partition` live in the sibling `models/sealed_test.py`
(Phase 10a-ii — extracted out of evaluate.py's own `__main__`-bearing module
so performance_check.py can share them) — never by error_analysis.py, same
reason: sealed_test.py imports
data.split, and error_analysis.py must never become transitively reachable to
it (CLAUDE.md's "test set touched once" invariant: X_test/y_test are imported
and used only there and in evaluate.py's own thin `__main__` wrapper over it).
A copy here would be a second route to the test partition.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import mlflow.tracking
import pandas as pd
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from telco_churn.utils.mlflow import resolve_tracking_uri

__all__ = [
    "load_training_manifest",
    "committed_features_from_manifest",
    "unfitted_pipeline_from_manifest",
    "load_dev_oof_predictions",
    "load_dev_oof_diagnostics",
    "load_fitted_model",
    "resolve_champion_version",
    "resolve_challenger_version",
    "load_threshold_validation",
]


def load_training_manifest(run_id: str, cfg: DictConfig) -> dict[str, Any]:
    """Load run_model_logging_step's training_manifest.json artifact from the tuning_study run.

    Sets the MLflow tracking URI as a side effect, so this is safe to call as
    the first MLflow-touching call in a fresh process.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    manifest: dict[str, Any] = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/training_manifest.json"
    )
    return manifest


def committed_features_from_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return run_feature_audit_step's frozen input space — the columns the logged pipeline expects."""
    return list(manifest["feature_selection"]["model_features"])


def unfitted_pipeline_from_manifest(manifest: dict[str, Any]) -> Pipeline:
    """Return a fresh, unfitted clone of the pipeline run_model_logging_step logged.

    clone() strips the fitted state (LightGBM booster, fitted ColumnTransformer
    statistics) while preserving the exact construction spec — the single
    construction path the bridge's design contract requires; rebuilding the
    pipeline from best_params is the failure it forbids.
    """
    fitted = mlflow.sklearn.load_model(manifest["logged_model_uri"])
    pipeline: Pipeline = clone(fitted)
    return pipeline


def load_dev_oof_predictions(run_id: str, cfg: DictConfig) -> pd.DataFrame:
    """Load calibrate.py's persisted dev_oof_predictions.parquet artifact.

    Columns: customerid, y_true, p_hat — the winning calibration method's
    exact leak-free OOF vector, the one calibrate.py's own method selection
    and t* validation rested on. Read here rather than recomputed: a
    recomputation is only sound while the dataset is static and CV folds are
    seeded, and silently diverges from the vector that cycle's decisions
    actually used once either stops holding (e.g. under Phase 10 retraining).
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/calibration/dev_oof_predictions.parquet"
    )
    return pd.read_parquet(local_path)


def load_dev_oof_diagnostics(run_id: str, cfg: DictConfig) -> dict[str, Any]:
    """Load threshold.py's dev_oof_diagnostics.json artifact (V1/V2/V2b) — resolved by run_id, not a local path.

    A fixed local path (reports/dev_oof_diagnostics.json) is what
    reliability_diagram.png warns against elsewhere in this codebase: it
    reflects whichever run last executed threshold.py on this machine, not
    necessarily run_id. Fetching by explicit run_id, same as
    load_threshold_validation, is what makes this correct under CI or a
    fresh checkout where no prior threshold.py run has touched local disk.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    diagnostics: dict[str, Any] = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/threshold/dev_oof_diagnostics.json"
    )
    return diagnostics


def load_fitted_model(model_uri: str, cfg: DictConfig) -> Pipeline:
    """Load the calibrated pipeline from an explicit model URI — never an alias.

    model_uri is caller-resolved (resolve_model_identifier): the immutable
    registry URI (models:/<name>/<version>) on an explicit run_id/
    model_version override, or calibrate.py's receipt's own stored
    run-scoped URI on the fully-automatic (neither given) path — the latter
    never touches the registry version pointer at all.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    model: Pipeline = mlflow.sklearn.load_model(model_uri)
    return model


def _resolve_alias_version(alias: str, cfg: DictConfig) -> str | None:
    """Resolve `alias` to an explicit version number, once.

    A single read of "does this alias exist, and which version" — never
    re-read afterward as a moving pointer. Returns None when the alias isn't
    set (or the model isn't registered yet at all).

    MLflow's SqlAlchemy-backed registry reports these two "not set" shapes
    with different error codes — RESOURCE_DOES_NOT_EXIST when the model was
    never registered, INVALID_PARAMETER_VALUE when it exists but this alias
    was never set — so both, and only both, are read as "unset." Any other
    MlflowException (a transient/auth/server failure) must propagate rather
    than be misread as unset, which would silently switch the caller to the
    wrong regime (e.g. the gate's cold-start path, or predict.py serving with
    no champion loaded).
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    registered_model_name = str(cfg.mlflow.registered_model_name)
    client = mlflow.tracking.MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_model_name, alias)
    except MlflowException as exc:
        if exc.error_code not in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            raise
        return None
    return str(version.version)


def resolve_champion_version(cfg: DictConfig) -> str | None:
    """Resolve the `champion` alias to an explicit version number, once.

    Returns None on cold start (no champion alias yet); a not-yet-registered
    model means the same thing. See _resolve_alias_version for the shared
    cold-start error-code handling this and resolve_challenger_version both
    rely on.
    """
    return _resolve_alias_version("champion", cfg)


def resolve_challenger_version(cfg: DictConfig) -> str | None:
    """Resolve the `challenger` alias to an explicit version number, once.

    Returns None in the common case — no challenger currently staged.
    serving/predict.py's shadow/canary mechanism treats this as "nothing to
    route to," not an error.
    """
    return _resolve_alias_version("challenger", cfg)


def load_threshold_validation(run_id: str, cfg: DictConfig) -> dict[str, Any]:
    """Load threshold.py's threshold_validation.json artifact — the model-dependent stamp."""
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    validation: dict[str, Any] = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/threshold/threshold_validation.json"
    )
    return validation
