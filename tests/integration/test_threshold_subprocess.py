"""Integration tests: threshold.py __main__ subprocess (self-contained, no Docker).

CLAUDE.md: every __main__ CLI entry point requires a subprocess integration
test. The registered calibrated model threshold.py needs is seeded once per
module (log_model.run_model_logging_step -> calibrate.run_calibration_step,
the real production chain) via a module-scoped fixture — re-running that
chain per test would cost an extra LightGBM + calibration CV fit each time
for no additional coverage. Each test still gets its own function-scoped
tmp_path for output paths (figures/policy) and, where relevant, its own
scratch costs.yaml — paths.costs_config makes that overridable per CLI call
without ever touching the real configs/costs.yaml.
"""

from __future__ import annotations

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


def _make_synthetic_processed_frame(n: int = 150, seed: int = 0) -> pd.DataFrame:
    """A FeatureOutputSchema-conformant frame, large enough to survive every CV split."""
    rng = np.random.default_rng(seed)
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
            "contract_type": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n
            ),
            "paymentmethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
            "tenure": rng.integers(0, 73, size=n).tolist(),
            "monthlycharges": rng.uniform(18.25, 118.75, size=n).tolist(),
            "totalcharges": rng.uniform(18.25, 8684.8, size=n).tolist(),
            "charge_per_service": rng.uniform(0.5, 50.0, size=n).tolist(),
            "churn": rng.integers(0, 2, size=n).tolist(),
        }
    )


def _seed_processed_data(
    out_dir: Path, n: int = 150, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write a processed CSV + matching canonical split manifest into out_dir.

    Returns (df, manifest) — the manifest must be passed explicitly to
    partition() later; partition()'s default reads load_split(), which is the
    real project's split_manifest.parquet, not this seeded one.
    """
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

    set_experiment(experiment_id=...) after create_experiment is load-bearing:
    without it, a run started directly in this process lands in the Default
    experiment (id "0") instead of the one just created.
    """
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


def _base_costs_cfg_dict() -> dict:
    """A valid, minimal costs.yaml payload — matches configs/costs.yaml's schema."""
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
        "argmax_ev_bootstrap_n_samples": 200,
    }


@pytest.fixture(scope="module")
def registered_threshold_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, str, str, Path]:
    """Seed a real registered calibrated model once for the whole module.

    Returns (tracking_uri, registered_model_name, model_version, processed_data_dir).
    """
    seed_root = tmp_path_factory.mktemp("threshold_subprocess_seed")
    data_dir = seed_root / "processed"
    data_dir.mkdir()
    df, manifest = _seed_processed_data(data_dir)

    experiment_name = str(compose_config().mlflow.experiment_name)
    tracking_uri = _sqlite_experiment(seed_root, experiment_name)
    registered_model_name = "test-threshold-subprocess-pipeline"

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

    # calibrate.run_calibration_step (unlike log_model.run_model_logging_step,
    # which takes X_dev/y_dev explicitly) loads dev data itself internally via
    # load_dev_features()/load_dev_customer_ids() — both resolve
    # cfg.paths.processed_data = ${oc.env:PROCESSED_DATA_DIR,datasets/processed/}.
    # Composing cfg and calling calibration inside this scoped env override
    # (rather than after, as a bare compose_config() call would do) is
    # load-bearing: without it, this in-process call silently reads the real
    # project's datasets/processed/ instead of this fixture's seeded data,
    # logging dev_oof_predictions.parquet against real customerids that share
    # no overlap with the synthetic cust-NNNN ids the subprocess below
    # computes (via the same env var, correctly set for it) — surfacing as a
    # dev-partition mismatch assertion failure in threshold.py, not here.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PROCESSED_DATA_DIR", str(data_dir))
        cfg = compose_config(
            overrides=[
                f"mlflow.tracking_uri={tracking_uri}",
                f"mlflow.registered_model_name={registered_model_name}",
                *_FAST_CALIBRATION_OVERRIDES,
            ]
        )
        with mlflow.start_run(run_name="tuning_study") as run:
            tuning_result["parent_run_id"] = run.info.run_id
        log_result = log_model.run_model_logging_step(
            X_dev, y_dev, comparison_result, tuning_result, cfg
        )
        cal_result = calibrate.run_calibration_step(log_result["run_id"], cfg)

    return (
        tracking_uri,
        registered_model_name,
        str(cal_result["model_version"]),
        data_dir,
    )


def _run_threshold_cli(
    model_version: str | None,
    tracking_uri: str,
    registered_model_name: str,
    data_dir: Path,
    figures_dir: Path,
    policy_dir: Path,
    reports_dir: Path,
    extra_overrides: list[str] | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    """Invoke threshold.py's CLI as a real subprocess with sandboxed output paths.

    paths.reports must be sandboxed here — threshold.py's folded-in dev-OOF
    screen writes reports/dev_oof_predictions.parquet and
    reports/dev_oof_diagnostics.json, and the default path resolves to the
    real project's reports/ directory if left unset.
    """
    overrides = [
        f"mlflow.tracking_uri={tracking_uri}",
        f"mlflow.registered_model_name={registered_model_name}",
        f"paths.figures={figures_dir}",
        f"paths.policy={policy_dir}",
        f"paths.reports={reports_dir}",
    ]
    if model_version is not None:
        overrides.append(f"threshold.model_version={model_version}")
    overrides.extend(extra_overrides or [])

    env = {
        **os.environ,
        "PROCESSED_DATA_DIR": str(data_dir),
        "MLFLOW_TRACKING_URI": tracking_uri,
    }
    return subprocess.run(
        [sys.executable, "-m", "telco_churn.models.threshold", *overrides],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
        timeout=timeout,
    )


def test_threshold_main_cli_ships_policy_regardless_of_dev_oof_screen_verdict(
    registered_threshold_model: tuple[str, str, str, Path],
    tmp_path: Path,
) -> None:
    """threshold.py __main__ derives t* and writes threshold.json + policy.yaml +
    figures — all written before the folded-in dev-OOF screen runs, so they
    exist regardless of the screen's own pass/fail verdict on this synthetic
    fixture's real (not mocked) calibration slope. Exit code 0 vs. 1 and
    "threshold_step_done" vs. "threshold_dev_oof_screen_failed" simply track
    which branch fired — both are valid resting states for this test, same
    tolerance test_calibration_screen_subprocess.py used to apply for the
    screen before it was folded into this module.
    """
    tracking_uri, registered_model_name, model_version, data_dir = (
        registered_threshold_model
    )
    figures_dir = tmp_path / "figures"
    policy_dir = tmp_path / "policy"
    reports_dir = tmp_path / "reports"

    result = _run_threshold_cli(
        model_version,
        tracking_uri,
        registered_model_name,
        data_dir,
        figures_dir,
        policy_dir,
        reports_dir,
    )

    assert result.returncode in (0, 1), (
        f"threshold CLI exited unexpectedly:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "threshold_step_done" in result.stdout or (
        "threshold_dev_oof_screen_failed" in result.stdout
    )

    assert (policy_dir / "threshold.yaml").exists()
    written = OmegaConf.load(policy_dir / "threshold.yaml")
    assert 0.0 < float(written.threshold) < 1.0
    assert set(written.scenarios) == {"conservative", "base", "optimistic"}

    assert (figures_dir / "threshold_sensitivity.png").exists()
    assert (figures_dir / "threshold_by_scenario.png").exists()
    assert (figures_dir / "expected_value_by_scenario.png").exists()

    # The folded-in dev-OOF screen (module docstring) writes both regardless
    # of pass/fail.
    assert (reports_dir / "dev_oof_predictions.parquet").exists()
    assert (reports_dir / "dev_oof_diagnostics.json").exists()


def test_threshold_main_cli_exits_one_when_model_version_missing(
    registered_threshold_model: tuple[str, str, str, Path],
    tmp_path: Path,
) -> None:
    """threshold.py __main__ exits 1 when threshold.model_version is not provided."""
    tracking_uri, registered_model_name, _model_version, data_dir = (
        registered_threshold_model
    )

    result = _run_threshold_cli(
        None,
        tracking_uri,
        registered_model_name,
        data_dir,
        tmp_path / "figures",
        tmp_path / "policy",
        tmp_path / "reports",
    )

    assert result.returncode == 1, (
        f"threshold CLI should exit 1 when threshold.model_version is missing:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_threshold_main_cli_exits_one_when_costs_config_missing(
    registered_threshold_model: tuple[str, str, str, Path],
    tmp_path: Path,
) -> None:
    """A missing configs/costs.yaml must fail loudly, not silently skip cost derivation."""
    tracking_uri, registered_model_name, model_version, data_dir = (
        registered_threshold_model
    )
    missing_costs_path = tmp_path / "does-not-exist" / "costs.yaml"

    result = _run_threshold_cli(
        model_version,
        tracking_uri,
        registered_model_name,
        data_dir,
        tmp_path / "figures",
        tmp_path / "policy",
        tmp_path / "reports",
        extra_overrides=[f"paths.costs_config={missing_costs_path}"],
    )

    assert result.returncode == 1, (
        f"threshold CLI should exit 1 when costs.yaml is missing:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_threshold_main_cli_exits_one_when_retention_rate_zero(
    registered_threshold_model: tuple[str, str, str, Path],
    tmp_path: Path,
) -> None:
    """r = 0 is a costs.yaml typo, not a valid input — division by zero must fail loudly."""
    tracking_uri, registered_model_name, model_version, data_dir = (
        registered_threshold_model
    )

    broken_costs = _base_costs_cfg_dict()
    broken_costs["scenarios"]["base"]["retention_rate"] = 0.0
    costs_path = tmp_path / "costs_zero_r.yaml"
    OmegaConf.save(OmegaConf.create(broken_costs), costs_path)

    result = _run_threshold_cli(
        model_version,
        tracking_uri,
        registered_model_name,
        data_dir,
        tmp_path / "figures",
        tmp_path / "policy",
        tmp_path / "reports",
        extra_overrides=[f"paths.costs_config={costs_path}"],
    )

    assert result.returncode == 1, (
        f"threshold CLI should exit 1 when retention_rate = 0:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_threshold_main_cli_exits_one_when_t_star_at_least_one(
    registered_threshold_model: tuple[str, str, str, Path],
    tmp_path: Path,
) -> None:
    """c >= r*LTV means t* >= 1 ('never contact anyone') — an equally plausible typo."""
    tracking_uri, registered_model_name, model_version, data_dir = (
        registered_threshold_model
    )

    broken_costs = _base_costs_cfg_dict()
    broken_costs["scenarios"]["base"]["outreach_cost"] = 1_000_000.0
    costs_path = tmp_path / "costs_t_star_ge_one.yaml"
    OmegaConf.save(OmegaConf.create(broken_costs), costs_path)

    result = _run_threshold_cli(
        model_version,
        tracking_uri,
        registered_model_name,
        data_dir,
        tmp_path / "figures",
        tmp_path / "policy",
        tmp_path / "reports",
        extra_overrides=[f"paths.costs_config={costs_path}"],
    )

    assert result.returncode == 1, (
        f"threshold CLI should exit 1 when t* >= 1:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_threshold_main_cli_exits_one_when_model_version_does_not_exist(
    registered_threshold_model: tuple[str, str, str, Path],
    tmp_path: Path,
) -> None:
    """A nonexistent registered model version must fail loudly, not silently
    fall back to something else — ported from
    test_calibration_screen_subprocess.py, since resolve_model_run_id is the
    same resolve-by-explicit-version call the folded-in screen now shares."""
    tracking_uri, registered_model_name, _model_version, data_dir = (
        registered_threshold_model
    )

    result = _run_threshold_cli(
        "999999",
        tracking_uri,
        registered_model_name,
        data_dir,
        tmp_path / "figures",
        tmp_path / "policy",
        tmp_path / "reports",
    )

    assert result.returncode == 1, (
        f"threshold CLI should exit 1 for a nonexistent model version:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
