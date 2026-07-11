"""CLI entry point: `python -m telco_churn.models.train` runs Steps 1-5 in sequence."""

from __future__ import annotations

import sys

import pandera as pa
from dotenv import load_dotenv

from telco_churn.models.train import (
    run_candidate_step,
    run_comparison_step,
    run_model_logging_step,
    run_selection_step,
    run_tuning_step,
)
from telco_churn.models.train.common import _load_dev_features
from telco_churn.utils.logging import configure_logging, get_logger
from telco_churn.utils.paths import compose_config

logger = get_logger(__name__)

load_dotenv()
configure_logging()

try:
    cfg = compose_config(overrides=sys.argv[1:] or None)
    X_dev, y_dev = _load_dev_features(cfg)

    logger.info(
        "data_split_ready",
        n_dev=len(y_dev),
        churn_rate_dev=round(float(y_dev.mean()), 4),
    )

    candidate_results = run_candidate_step(X_dev, y_dev, cfg)
    comparison = run_comparison_step(X_dev, candidate_results, cfg)

    if comparison["decision"] == "lgbm":
        selection = run_selection_step(X_dev, y_dev, candidate_results, cfg)
        tuning = run_tuning_step(X_dev, y_dev, selection["committed_features"], cfg)
        run_model_logging_step(X_dev, y_dev, comparison, tuning, cfg)
    else:
        logger.warning(
            "selection_skipped_logreg_winner",
            decision_rule=comparison["decision_rule"],
            hint=(
                "Step 3's permutation-importance selector is model-agnostic, but "
                "this pipeline still fits LightGBM internally and Step 4 tuning "
                "is LightGBM-only; a LogReg winner still needs a linear-model "
                "pipeline built before Steps 3-5 can run (PROJECT_PLAN.md Step 2 "
                "rule-4 contingency)."
            ),
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
    logger.error("leakage_canary_failed", error=str(e), exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.error("train_failed", error=str(e), exc_info=True)
    sys.exit(1)
