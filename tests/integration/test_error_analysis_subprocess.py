"""Integration tests: error_analysis.py __main__ subprocess (self-contained, no Docker).

CLAUDE.md: every __main__ CLI entry point requires a subprocess integration
test — direct function calls miss argparse/OmegaConf resolution and the
env-var-to-engine joints that only surface at the process boundary.

The full production chain up to (but not including) error_analysis.py is
seeded once per module via direct in-process calls —
log_model.run_model_logging_step -> calibrate.run_calibration_step ->
threshold.run_threshold_step -> evaluate.run_evaluation_step — one step
further down the pipeline than test_evaluate_subprocess.py's own precedent.
Only error_analysis.py itself crosses the subprocess boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import telco_churn.models.calibrate as calibrate
import telco_churn.models.evaluate as evaluate
import telco_churn.models.threshold as threshold
import telco_churn.models.train.log_model as log_model
from telco_churn.data.split import make_split, partition, write_split
from telco_churn.utils.paths import compose_config, get_project_root

pytestmark = pytest.mark.integration

_PROJECT_ROOT = get_project_root()

_FAST_CALIBRATION_OVERRIDES = [
    "calibration.method=sigmoid",
    "calibration.outer_cv_folds=3",
    "calibration.inner_cv_folds=3",
]

# Small enough to keep evaluate.run_evaluation_step's many bootstrap passes
# fast when seeding the fixture; error_analysis.py's own SHAP pass has no
# bootstrap knob of its own.
_FAST_EVALUATE_OVERRIDES = ["evaluate.n_bootstrap=30"]


def _make_synthetic_processed_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A FeatureOutputSchema-conformant frame with every segment-axis column
    evaluate.py's slicing needs, sized so each axis's groups clear
    _MIN_SLICE_SIZE on both the dev and test partitions.

    churn is drawn from a logistic function of contract_type/tenure/
    monthlycharges — a real, learnable relationship — mirroring
    test_evaluate_subprocess.py's own fixture, for the same reason: label
    noise makes calibrate.py's PR-AUC gate flaky.
    """
    rng = np.random.default_rng(seed)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    tenure = rng.integers(0, 73, size=n)
    monthlycharges = rng.uniform(18.25, 118.75, size=n)

    logit = (
        -0.5
        + 1.4 * (contract_type == "Month-to-month")
        - 0.03 * tenure
        + 0.01 * (monthlycharges - 60.0)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "customerid": [f"cust-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "has_partner": rng.choice(["Yes", "No"], size=n),
            "dependents": rng.choice(["Yes", "No"], size=n),
            "phoneservice": rng.choice(["Yes", "No"], size=n),
            "paperlessbilling": rng.choice(["Yes", "No"], size=n),
            "seniorcitizen": rng.integers(0, 2, size=n).tolist(),
            "multiplelines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "internetservice": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "onlinesecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "onlinebackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "deviceprotection": rng.choice(
                ["Yes", "No", "No internet service"], size=n
            ),
            "techsupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streamingtv": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streamingmovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "contract_type": contract_type,
            "paymentmethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
            "tenure": tenure.tolist(),
            "monthlycharges": monthlycharges.tolist(),
            "totalcharges": (monthlycharges * np.maximum(tenure, 1)).tolist(),
            "charge_per_service": rng.uniform(0.5, 50.0, size=n).tolist(),
            "churn": churn.tolist(),
        }
    )


def _seed_processed_data(
    out_dir: Path, n: int = 300, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write a processed CSV + matching canonical split manifest into out_dir."""
    df = _make_synthetic_processed_frame(n=n, seed=seed)
    df.to_csv(out_dir / "telco_churn_processed.csv", index=False)
    manifest = make_split(
        ids=df["customerid"], labels=df["churn"], test_size=0.2, random_state=42
    )
    write_split(manifest, out_dir / "split_manifest.parquet")
    return df, manifest


def _sqlite_experiment(tmp_path: Path, experiment_name: str) -> str:
    """Create a throwaway SQLite-backed MLflow store with an explicit artifact
    root, returning its tracking URI — mirrors conftest.py :: mlflow_test_experiment.
    """
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


def _base_costs_cfg_dict() -> dict:
    """A valid, minimal costs.yaml payload — matches configs/costs.yaml's
    schema, including contact_capacity (evaluate.py's EV-by-budget figure
    reads it), with a reduced bootstrap count for speed.
    """
    return {
        "gross_margin": 0.60,
        "horizon_months": 12,
        "discount_months": 3,
        "arpu_quantile": {"conservative": 0.25, "base": 0.50, "optimistic": 0.75},
        "scenarios": {
            "conservative": {
                "outreach_cost": 5.0,
                "discount_rate": 0.10,
                "retention_rate": 0.20,
            },
            "base": {
                "outreach_cost": 20.0,
                "discount_rate": 0.20,
                "retention_rate": 0.30,
            },
            "optimistic": {
                "outreach_cost": 50.0,
                "discount_rate": 0.30,
                "retention_rate": 0.40,
            },
        },
        "retention_rate_sweep": [0.15, 0.20, 0.30, 0.40, 0.45],
        "contact_capacity": 200,
        "campaign_budget": 15_000,
        "argmax_ev_bootstrap_n_samples": 100,
    }


@pytest.fixture(scope="module")
def evaluated_model(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Seed a real registered, calibrated, thresholded, and evaluated model
    once for the whole module — the production chain up to (not including)
    error_analysis.py.
    """
    seed_root = tmp_path_factory.mktemp("error_analysis_subprocess_seed")
    data_dir = seed_root / "processed"
    data_dir.mkdir()
    df, manifest = _seed_processed_data(data_dir)

    experiment_name = str(compose_config().mlflow.experiment_name)
    tracking_uri = _sqlite_experiment(seed_root, experiment_name)
    registered_model_name = "test-error-analysis-subprocess-pipeline"

    costs_path = seed_root / "costs.yaml"
    OmegaConf.save(OmegaConf.create(_base_costs_cfg_dict()), costs_path)
    figures_dir = seed_root / "figures"
    policy_dir = seed_root / "policy"
    reports_dir = seed_root / "reports"

    cfg = compose_config(
        overrides=[
            f"mlflow.tracking_uri={tracking_uri}",
            f"mlflow.registered_model_name={registered_model_name}",
            f"paths.costs_config={costs_path}",
            f"paths.figures={figures_dir}",
            f"paths.policy={policy_dir}",
            f"paths.reports={reports_dir}",
            *_FAST_CALIBRATION_OVERRIDES,
            *_FAST_EVALUATE_OVERRIDES,
        ]
    )

    dev_df, _test_df = partition(df, manifest)
    feature_cols = [c for c in df.columns if c not in ("customerid", "churn")]
    X_dev, y_dev = dev_df[feature_cols], dev_df["churn"]

    tuning_result = {
        "best_params": {
            "num_leaves": 8,
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "max_depth": 5,
        },
        "best_n_estimators_median": 10,
        "best_cv_pr_auc_mean": 0.6,
        "committed_features": feature_cols,
        "tuning_summary": {
            "n_trials_requested": 50,
            "n_completed_trials": 16,
            "n_pruned_trials": 34,
            "n_failed_trials": 0,
            "min_completed_trials": 10,
            "trial_count_below_threshold": False,
            "selection_rule": "1se",
            "selected_trial_number": 9,
            "selected_cv_pr_auc": 0.6,
            "raw_best_trial_number": 36,
            "raw_best_cv_pr_auc": 0.6664,
            "se": 0.0139,
            "band_floor": 0.6525,
            "boundary_hits": {"num_leaves": False},
        },
    }
    comparison_result = {
        "delta_obs": 0.01,
        "delta_ci_lower": -0.01,
        "delta_ci_upper": 0.03,
        "decision": "lgbm",
        "decision_rule": "tie",
        "diagnostics": {"fixed_recall": [], "fairness": [], "robustness": []},
    }

    # calibrate/threshold/evaluate re-derive the dev/test partitions themselves
    # (load_features() -> partition()), reading PROCESSED_DATA_DIR from the
    # *current* environment rather than from `cfg` — same env-scoping caveat
    # test_evaluate_subprocess.py's fixture documents.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PROCESSED_DATA_DIR", str(data_dir))
        with mlflow.start_run(run_name="tuning_study") as run:
            tuning_result["parent_run_id"] = run.info.run_id
        log_result = log_model.run_model_logging_step(
            X_dev, y_dev, comparison_result, tuning_result, cfg
        )
        cal_result = calibrate.run_calibration_step(log_result["run_id"], cfg)
        model_version = str(cal_result["model_version"])

        threshold.run_threshold_step(model_version, cfg)
        evaluate.run_evaluation_step(model_version, cfg)

    return {
        "tracking_uri": tracking_uri,
        "registered_model_name": registered_model_name,
        "model_version": model_version,
        "data_dir": data_dir,
        "costs_path": costs_path,
        "policy_dir": policy_dir,
        "figures_dir": figures_dir,
        "reports_dir": reports_dir,
    }


def _run_error_analysis_cli(
    model_version: str | None,
    fixture: dict[str, object],
    extra_overrides: list[str] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Invoke error_analysis.py's CLI as a real subprocess with sandboxed output paths."""
    overrides = [
        f"mlflow.tracking_uri={fixture['tracking_uri']}",
        f"mlflow.registered_model_name={fixture['registered_model_name']}",
        f"paths.costs_config={fixture['costs_path']}",
        f"paths.policy={fixture['policy_dir']}",
        f"paths.figures={fixture['figures_dir']}",
        f"paths.reports={fixture['reports_dir']}",
    ]
    if model_version is not None:
        overrides.append(f"error_analysis.model_version={model_version}")
    overrides.extend(extra_overrides or [])

    env = {
        **os.environ,
        "PROCESSED_DATA_DIR": str(fixture["data_dir"]),
        "MLFLOW_TRACKING_URI": str(fixture["tracking_uri"]),
    }
    return subprocess.run(
        [sys.executable, "-m", "telco_churn.models.error_analysis", *overrides],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
        timeout=timeout,
    )


def test_error_analysis_main_cli_exits_zero_and_writes_report(
    evaluated_model: dict[str, object],
) -> None:
    """error_analysis.py __main__ computes SHAP on the sealed test set, scans
    dev OOF for error concentration, and writes error_analysis.json."""
    result = _run_error_analysis_cli(
        str(evaluated_model["model_version"]), evaluated_model
    )

    assert (
        result.returncode == 0
    ), f"error_analysis CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "error_analysis_step_done" in result.stdout

    reports_dir = Path(str(evaluated_model["reports_dir"]))
    payload = json.loads(
        (reports_dir / "error_analysis.json").read_text(encoding="utf-8")
    )
    assert payload["model_version"] == str(evaluated_model["model_version"])
    for key in (
        "error_confidence",
        "value_weighted_errors",
        "error_concentration",
        "shap",
        "top_k_elbow_checks",
        "subgroup_findings",
        "direction_sanity_check",
        "dev_oof_diagnostics_carried_through",
    ):
        assert key in payload, f"error_analysis.json missing field: {key}"

    assert "passed" in payload["direction_sanity_check"]
    assert set(payload["shap"]) >= {
        "global_importance",
        "top_features",
        "dependence",
        "cohort_shap",
        "cohort_top_features_fp_tn",
        "local_explanations",
    }
    assert set(payload["shap"]["cohort_shap"]) == {"fn", "tp", "fp", "tn"}
    assert set(payload["top_k_elbow_checks"]) == {
        "shap_features",
        "cohort_gap_fn_tp",
        "cohort_gap_fp_tn",
    }
    for check in payload["top_k_elbow_checks"].values():
        assert "valid" in check
        assert "configured_k" in check
    assert set(payload["subgroup_findings"]) == {
        "annual_contract_fn_vs_tn_monthlycharges"
    }
    assert set(payload["error_concentration"]) == {
        "dev_oof_top_fnr_cohorts",
        "dev_oof_top_fpr_cohorts",
    }

    figures_dir = Path(str(evaluated_model["figures_dir"]))
    for filename in (
        "shap_global_importance.png",
        "shap_beeswarm.png",
        "shap_dependence.png",
        "shap_cohort_fn_vs_tp.png",
        "shap_cohort_beeswarm_fn.png",
        "shap_cohort_beeswarm_tp.png",
        "shap_cohort_fp_vs_tn.png",
        "shap_cohort_beeswarm_fp.png",
        "shap_cohort_beeswarm_tn.png",
        "shap_cohort_dependence_tenure.png",
        "shap_subgroup_dependence_monthlycharges_annual.png",
        "shap_waterfall_examples.png",
        "shap_waterfall_confident_cases.png",
        "error_confidence.png",
        "value_weighted_errors.png",
    ):
        assert (figures_dir / filename).exists(), f"missing figure: {filename}"

    client = mlflow.tracking.MlflowClient(
        tracking_uri=str(evaluated_model["tracking_uri"])
    )
    runs = client.search_runs(
        experiment_ids=[
            client.get_experiment_by_name(
                str(compose_config().mlflow.experiment_name)
            ).experiment_id
        ],
        filter_string="tags.mlflow.runName = 'error_analysis'",
    )
    assert len(runs) >= 1
    assert runs[0].data.tags.get("direction_sanity_check") in {"pass", "fail"}


def test_error_analysis_main_cli_exits_one_when_model_version_missing(
    evaluated_model: dict[str, object],
) -> None:
    """error_analysis.py __main__ exits 1 when error_analysis.model_version is
    not provided — never inferred from 'latest'."""
    result = _run_error_analysis_cli(None, evaluated_model, timeout=60)

    assert result.returncode == 1, (
        f"error_analysis CLI should exit 1 when error_analysis.model_version "
        f"is missing:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_error_analysis_main_cli_exits_one_when_prediction_artifacts_missing(
    evaluated_model: dict[str, object], tmp_path: Path
) -> None:
    """A reports/ directory without evaluate.py's prediction artifacts fails
    loudly rather than silently reaching for the test split some other way."""
    empty_reports_dir = tmp_path / "empty_reports"
    empty_reports_dir.mkdir()
    fixture_with_empty_reports = {**evaluated_model, "reports_dir": empty_reports_dir}

    result = _run_error_analysis_cli(
        str(evaluated_model["model_version"]),
        fixture_with_empty_reports,
        timeout=60,
    )

    assert result.returncode == 1, (
        f"error_analysis CLI should exit 1 when test_predictions.parquet is "
        f"absent:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
