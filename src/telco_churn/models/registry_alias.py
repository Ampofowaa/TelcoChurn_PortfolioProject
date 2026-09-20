"""Registry alias resolution — champion/challenger -> explicit version number.

Split out of artifacts.py (2026-09-18): this is the one piece of that module
a caller can need in isolation (ui/streamlit_app.py's "About this model" tab
only ever resolves `champion`, never loads the model itself), and it needs
nothing heavier than mlflow.tracking + omegaconf. artifacts.py's other
loaders (load_fitted_model, unfitted_pipeline_from_manifest,
load_dev_oof_predictions, ...) genuinely need mlflow.sklearn/pandas/sklearn,
so splitting this out is what lets a sklearn-free caller stay sklearn-free
rather than paying for those imports just to reach this one lookup.
"""

from __future__ import annotations

import mlflow
import mlflow.tracking
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig

from telco_churn.utils.mlflow import resolve_tracking_uri

__all__ = [
    "resolve_champion_version",
    "resolve_challenger_version",
]


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
