"""CLI entry point: `python -m telco_churn.models.train` runs Steps 3-5 in sequence.

Steps 1-2 (candidate comparison, model-family decision) are notebook-only —
see models/train/common.py::COMMITTED_MODEL_FAMILY and
notebooks/03a-model-selection.ipynb. This entry point builds directly against
the frozen family; it does not re-decide it.
"""

from __future__ import annotations

import sys

import pandera as pa
from dotenv import load_dotenv

from telco_churn.models.train import (
    run_feature_audit_step,
    run_model_logging_step,
    run_tuning_step,
)
from telco_churn.models.train.common import _load_dev_features
from telco_churn.utils.logging import configure_logging, get_logger
from telco_churn.utils.paths import activate_config, compose_config

logger = get_logger(__name__)

load_dotenv()
configure_logging()

try:
    cfg = compose_config(overrides=sys.argv[1:] or None)
    activate_config(cfg)
    X_dev, y_dev = _load_dev_features()

    logger.info(
        "data_split_ready",
        n_dev=len(y_dev),
        churn_rate_dev=round(float(y_dev.mean()), 4),
    )

    # Steps 3-5 of the five-step pipeline (see models/train/__init__.py's docstring
    # for the full list, incl. the notebook-only Steps 1-2). Every call below
    # builds directly against frozen, human-reviewed constants — nothing here
    # re-decides the model family or the feature set, on this run or any retrain.
    #   3. run_feature_audit_step  — audits the already-frozen COMMITTED_FEATURES
    #      (permutation-importance + SHAP), logs the diagnostic; does not select.
    #   4. run_tuning_step         — Optuna hyperparameter search against
    #      COMMITTED_MODEL_FAMILY (LightGBM), scored by CV PR-AUC.
    #   5. run_model_logging_step  — the actual fit: [preprocessor -> LightGBM]
    #      trained on all of X_dev/y_dev with the tuned hyperparameters, logged
    #      to MLflow (uncalibrated, unregistered — calibrate.py picks it up next).
    selection = run_feature_audit_step(X_dev, y_dev, cfg)
    tuning = run_tuning_step(X_dev, y_dev, selection["committed_features"], cfg)
    logging_result = run_model_logging_step(X_dev, y_dev, tuning, cfg)

    logger.info(
        "train_step_done",
        run_id=logging_result["run_id"],
        model_uri=logging_result["model_uri"],
        parity_ok=logging_result["parity_ok"],
    )

except FileNotFoundError as e:
    logger.error(
        "processed_data_not_found",
        error=str(e),
        hint=(
            "Run 'python -m telco_churn.features.build' and "
            "'python -m telco_churn.data.split' first."
        ),
        exc_info=True,
    )
    sys.exit(1)
except pa.errors.SchemaError as e:
    logger.error("processed_data_schema_invalid", error=str(e), exc_info=True)
    sys.exit(1)
except AssertionError as e:
    # Steps 1-2's leakage canary no longer runs here (notebook-only, see module
    # docstring) — the reachable case today is Step 5's log->reload->predict_proba
    # parity check in log_model.py.
    logger.error("training_assertion_failed", error=str(e), exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.error("train_failed", error=str(e), exc_info=True)
    sys.exit(1)
