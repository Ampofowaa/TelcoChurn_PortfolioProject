"""Integration tests: evaluate.py __main__ subprocess (self-contained, no Docker).

CLAUDE.md: every __main__ CLI entry point requires a subprocess integration
test — direct function calls miss argparse/OmegaConf resolution and the
env-var-to-engine joints that only surface at the process boundary.

The full production chain up to (but not including) evaluate.py is seeded
once per module via direct in-process calls — log_model.run_model_logging_step
-> calibrate.run_calibration_step -> threshold.run_threshold_step (which now
folds in the dev-OOF screen as its own last step) — mirroring
test_threshold_subprocess.py's own precedent one step further down the
pipeline. Only evaluate.py itself crosses the subprocess boundary. A dev-OOF
screen failure (a real slope outside the band on this synthetic fixture) is
tolerated during seeding — its artifacts are written before it raises, so
evaluate.py's later reads are unaffected either way.

No `champion` alias exists anywhere in this throwaway registry, so every test
here exercises the cold-start regime (ANALYSIS.md §0) — the comparative
regime needs a promoted champion. Coverage for the comparative branch belongs
with register.py's own subprocess test (tests/integration/test_register_subprocess.py).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import telco_churn.models.calibrate as calibrate
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

# Small enough to keep the many bootstrap passes inside run_evaluation_step
# (ranking, per-scenario classification, calibration slope, business impact,
# every sliced axis) fast on a CI box; large enough that a CI is still a CI.
_FAST_EVALUATE_OVERRIDES = ["evaluate.n_bootstrap=30"]


def _make_synthetic_processed_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A FeatureOutputSchema-conformant frame with every segment-axis column
    evaluate.py's slicing needs (contract_type, internetservice, gender,
    seniorcitizen, has_partner, dependents, tenure), sized so each axis's
    groups clear _MIN_SLICE_SIZE on both the dev and test partitions.

    churn is drawn from a logistic function of contract_type/tenure/
    monthlycharges — a real, learnable relationship — rather than pure noise.
    calibrate.py's own PR-AUC gate (pr_auc_gate_passes) refuses to register a
    calibrated model whose OOF ranking falls below the uncalibrated
    baseline's; on label noise with zero true signal, sigmoid calibration's
    per-fold mean AP can legitimately land on either side of that bar by
    chance alone, which made this fixture flaky. Genuine signal keeps the
    gate passing reliably and every downstream evaluate.py metric
    (PR-AUC, lift, calibration slope) non-degenerate.
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
    schema, including contact_capacity (run_evaluation_step's EV-by-budget
    figure reads it), with a reduced bootstrap count for speed.
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
def registered_evaluation_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    """Seed a real registered calibrated model plus a shipped threshold once
    for the whole module — the production chain up to (not including)
    evaluate.py.
    """
    seed_root = tmp_path_factory.mktemp("evaluate_subprocess_seed")
    data_dir = seed_root / "processed"
    data_dir.mkdir()
    df, manifest = _seed_processed_data(data_dir)

    experiment_name = str(compose_config().mlflow.experiment_name)
    tracking_uri = _sqlite_experiment(seed_root, experiment_name)
    registered_model_name = "test-evaluate-subprocess-pipeline"

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

    # calibrate.run_calibration_step/threshold.run_threshold_step re-derive the
    # dev partition themselves (load_features() -> partition()), reading
    # PROCESSED_DATA_DIR from the *current* environment rather than from
    # `cfg` — unlike run_model_logging_step, which only ever sees the X_dev/
    # y_dev this fixture builds explicitly above. Without this, they would
    # silently fall back to the real project's own datasets/processed/ (if
    # present on the machine) instead of this seeded synthetic frame, and
    # calibrate's dev_oof_predictions.parquet would carry real customerids
    # that can never match evaluate.py's later, correctly-scoped subprocess
    # read of this same synthetic data.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PROCESSED_DATA_DIR", str(data_dir))
        with mlflow.start_run(run_name="tuning_study") as run:
            tuning_result["parent_run_id"] = run.info.run_id
        log_result = log_model.run_model_logging_step(
            X_dev, y_dev, comparison_result, tuning_result, cfg
        )
        cal_result = calibrate.run_calibration_step(log_result["run_id"], cfg)
        model_version = str(cal_result["model_version"])

        try:
            threshold.run_threshold_step(model_version, cfg)
        except RuntimeError:
            # A dev-OOF screen failure (the real slope on this synthetic
            # fixture lying outside the band) is a legitimate outcome, not a
            # setup bug — reports/dev_oof_predictions.parquet and
            # dev_oof_diagnostics.json are written before that raise, so
            # evaluate.py's later reads are unaffected either way.
            pass

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


def _run_evaluate_cli(
    model_version: str | None,
    fixture: dict[str, object],
    reports_dir: Path,
    extra_overrides: list[str] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Invoke evaluate.py's CLI as a real subprocess with sandboxed output paths.

    evaluate.py now reads reports/dev_oof_predictions.parquet and
    reports/dev_oof_diagnostics.json rather than computing them — both were
    written once, during fixture seeding, into fixture["reports_dir"]. Each
    test here uses its own scratch `reports_dir` for output isolation, so
    those two threshold.py dev-OOF-screen artifacts are copied in first.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    seed_reports_dir = Path(str(fixture["reports_dir"]))
    for name in ("dev_oof_predictions.parquet", "dev_oof_diagnostics.json"):
        src = seed_reports_dir / name
        if src.exists():
            shutil.copy(src, reports_dir / name)

    overrides = [
        f"mlflow.tracking_uri={fixture['tracking_uri']}",
        f"mlflow.registered_model_name={fixture['registered_model_name']}",
        f"paths.costs_config={fixture['costs_path']}",
        f"paths.policy={fixture['policy_dir']}",
        f"paths.figures={fixture['figures_dir']}",
        f"paths.reports={reports_dir}",
        *_FAST_EVALUATE_OVERRIDES,
    ]
    if model_version is not None:
        overrides.append(f"evaluate.model_version={model_version}")
    overrides.extend(extra_overrides or [])

    env = {
        **os.environ,
        "PROCESSED_DATA_DIR": str(fixture["data_dir"]),
        "MLFLOW_TRACKING_URI": str(fixture["tracking_uri"]),
    }
    return subprocess.run(
        [sys.executable, "-m", "telco_churn.models.evaluate", *overrides],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
        timeout=timeout,
    )


def test_evaluate_main_cli_exits_zero_and_writes_reports(
    registered_evaluation_model: dict[str, object],
    tmp_path: Path,
) -> None:
    """evaluate.py __main__ scores the sealed test set, runs the cold-start gate,
    and writes every artifact register.py/refit.py/the notebook will need."""
    reports_dir = tmp_path / "reports"

    result = _run_evaluate_cli(
        str(registered_evaluation_model["model_version"]),
        registered_evaluation_model,
        reports_dir,
    )

    assert (
        result.returncode == 0
    ), f"evaluate CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "evaluation_step_done" in result.stdout

    metrics = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_version"] == str(registered_evaluation_model["model_version"])
    assert set(metrics["classification"][0]) >= {
        "scenario",
        "recall",
        "precision",
        "f1",
    }
    assert "sliced" in metrics
    assert "dev_oof_diagnostics" in metrics["sliced"]

    economics = json.loads((reports_dir / "economics.json").read_text(encoding="utf-8"))
    assert set(economics["ev_by_k"]) == {"conservative", "base", "optimistic"}

    decision = json.loads(
        (reports_dir / "promotion_decision.json").read_text(encoding="utf-8")
    )
    assert decision["regime"] == "cold_start"
    assert decision["gate"] in {"pass", "fail"}
    assert decision["review"] == "pending"
    assert "metrics_content_hash" in decision
    assert decision["eval_run_id"] != metrics["run_id"]

    test_predictions = pd.read_parquet(reports_dir / "test_predictions.parquet")
    assert set(test_predictions.columns) == {"customerid", "y_true", "p_hat"}
    assert len(test_predictions) > 0

    # Written by threshold.py's folded-in dev-OOF screen (customerid, y_true,
    # p_hat — no re-logged segment columns, since nothing downstream needs
    # them off this file), copied into this test's scratch reports_dir by
    # _run_evaluate_cli — evaluate.py itself only ever reads this file, never
    # writes it.
    dev_oof_predictions = pd.read_parquet(reports_dir / "dev_oof_predictions.parquet")
    assert set(dev_oof_predictions.columns) == {"customerid", "y_true", "p_hat"}
    assert len(dev_oof_predictions) > 0

    figures_dir = Path(str(registered_evaluation_model["figures_dir"]))
    for filename in (
        "pr_curve_test.png",
        "roc_curve_test.png",
        "classification_report_test.png",
        "reliability_diagram_test.png",
        "ev_by_budget.png",
        "breakeven_heatmap.png",
        "sensitivity_tornado.png",
        "gains_lift_test.png",
    ):
        assert (figures_dir / filename).exists(), f"missing figure: {filename}"

    client = mlflow.tracking.MlflowClient(
        tracking_uri=str(registered_evaluation_model["tracking_uri"])
    )
    version = client.get_model_version(
        str(registered_evaluation_model["registered_model_name"]),
        str(registered_evaluation_model["model_version"]),
    )
    for tag in ("test_pr_auc", "test_recall", "test_brier", "test_calibration_slope"):
        assert tag in version.tags, f"missing model-version tag: {tag}"
    # register.py's only supported path to this cycle's promotion_decision.json/
    # metrics.json — resolved by run_id, never a local reports/ path.
    assert version.tags.get("eval_run_id") == decision["eval_run_id"]


def test_evaluate_main_cli_exits_one_when_model_version_missing(
    registered_evaluation_model: dict[str, object],
    tmp_path: Path,
) -> None:
    """evaluate.py __main__ exits 1 when evaluate.model_version is not provided —
    never inferred from 'latest'."""
    result = _run_evaluate_cli(
        None, registered_evaluation_model, tmp_path / "reports", timeout=60
    )

    assert result.returncode == 1, (
        f"evaluate CLI should exit 1 when evaluate.model_version is missing:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_evaluate_main_cli_exits_one_when_model_version_does_not_exist(
    registered_evaluation_model: dict[str, object],
    tmp_path: Path,
) -> None:
    """A nonexistent registered model version must fail loudly, not silently
    fall back to something else."""
    result = _run_evaluate_cli(
        "999999", registered_evaluation_model, tmp_path / "reports"
    )

    assert result.returncode == 1, (
        f"evaluate CLI should exit 1 for a nonexistent model version:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
