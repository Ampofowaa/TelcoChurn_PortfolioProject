"""Unit tests for telco_churn.models.train.comparison — Step 2 (B3, B4)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import mlflow
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

import telco_churn.models.train.comparison as comparison
from telco_churn.features.accessor import features_path
from telco_churn.models.train.common import cv_score_candidate

_FAKE_DATA_CONTENT_HASH = "deadbeef" * 8


@pytest.fixture(scope="module", autouse=True)
def _stub_data_content_hash() -> Iterator[None]:
    """run_comparison_step stamps data_content_hash via features_sha256() with
    no path override, resolving to the real, gitignored
    datasets/processed/telco_churn_features.parquet — absent on a fresh
    checkout. Every test below trains on synthetic fixtures, never real
    processed data, so stub the hash.
    """
    mp = pytest.MonkeyPatch()
    mp.setattr(comparison, "features_sha256", lambda path=None: _FAKE_DATA_CONTENT_HASH)
    yield
    mp.undo()


# ---------------------------------------------------------------------------
# bootstrap_comparison
# ---------------------------------------------------------------------------


def test_bootstrap_comparison_lgbm_win() -> None:
    """CI fully above 0 and Δ clears Δ*: adopt LGBM on the evidence."""
    scores_lgbm = [0.5] * 15
    scores_logreg = [0.4] * 15
    result = comparison.bootstrap_comparison(
        scores_lgbm,
        scores_logreg,
        n_bootstrap=500,
        delta_threshold=0.01,
        random_state=42,
    )
    assert result["decision"] == "lgbm"
    assert result["decision_rule"] == "lgbm_win"
    assert result["delta_obs"] > 0


def test_bootstrap_comparison_below_threshold_ties_lgbm() -> None:
    """Δ excludes 0 but is below Δ*: a tie — LGBM is still adopted (Step 1 rationale)."""
    scores_lgbm = [0.403] * 15
    scores_logreg = [0.400] * 15
    result = comparison.bootstrap_comparison(
        scores_lgbm,
        scores_logreg,
        n_bootstrap=500,
        delta_threshold=0.01,
        random_state=42,
    )
    assert result["decision"] == "lgbm"
    assert result["decision_rule"] == "tie"


def test_bootstrap_comparison_ci_includes_zero_ties_lgbm() -> None:
    """A CI straddling 0 is also a tie — LGBM is still adopted, not logreg."""
    diffs = [
        0.03,
        -0.02,
        0.04,
        -0.03,
        0.02,
        -0.04,
        0.03,
        -0.02,
        0.00,
        0.02,
        -0.03,
        0.04,
        -0.02,
        0.03,
        -0.03,
    ]
    scores_logreg = [0.40] * len(diffs)
    scores_lgbm = [0.40 + d for d in diffs]
    result = comparison.bootstrap_comparison(
        scores_lgbm,
        scores_logreg,
        n_bootstrap=2000,
        delta_threshold=0.01,
        random_state=42,
    )
    assert result["delta_ci_lower"] < 0 < result["delta_ci_upper"]
    assert result["decision"] == "lgbm"
    assert result["decision_rule"] == "tie"


def test_bootstrap_comparison_logreg_win() -> None:
    """CI fully below 0 and |Δ| clears Δ* in LogReg's favour: LogReg wins decisively."""
    scores_lgbm = [0.4] * 15
    scores_logreg = [0.5] * 15
    result = comparison.bootstrap_comparison(
        scores_lgbm,
        scores_logreg,
        n_bootstrap=500,
        delta_threshold=0.01,
        random_state=42,
    )
    assert result["decision"] == "logreg"
    assert result["decision_rule"] == "logreg_win"


def test_bootstrap_comparison_ci_contains_obs() -> None:
    """The 95% CI must bracket the observed Δ."""
    scores_lgbm = [0.5 + i * 0.01 for i in range(15)]
    scores_logreg = [0.4 + i * 0.01 for i in range(15)]
    result = comparison.bootstrap_comparison(
        scores_lgbm,
        scores_logreg,
        n_bootstrap=1000,
        delta_threshold=0.01,
        random_state=42,
    )
    assert result["delta_ci_lower"] <= result["delta_obs"] <= result["delta_ci_upper"]


def test_bootstrap_comparison_return_keys() -> None:
    """All eight expected keys are present."""
    scores = [0.5] * 5
    result = comparison.bootstrap_comparison(
        scores, scores, n_bootstrap=100, delta_threshold=0.01, random_state=0
    )
    assert set(result) == {
        "delta_obs",
        "delta_ci_lower",
        "delta_ci_upper",
        "p_value",
        "decision",
        "decision_rule",
        "n_bootstrap",
        "bootstrap_deltas",
    }


def test_bootstrap_comparison_n_bootstrap_preserved() -> None:
    """n_bootstrap in the result matches the argument."""
    scores = [0.5] * 5
    result = comparison.bootstrap_comparison(
        scores, scores, n_bootstrap=250, delta_threshold=0.01, random_state=0
    )
    assert result["n_bootstrap"] == 250


def test_bootstrap_comparison_equal_scores_p_value_is_one() -> None:
    """Identical fold scores give Δ = 0 on every resample, so p_value = 1.0."""
    scores = [0.5] * 15
    result = comparison.bootstrap_comparison(
        scores, scores, n_bootstrap=500, delta_threshold=0.01, random_state=42
    )
    assert result["p_value"] == 1.0
    assert result["delta_obs"] == 0.0
    assert result["decision_rule"] == "tie"


# ---------------------------------------------------------------------------
# run_diagnostics_step (B4)
# ---------------------------------------------------------------------------


@pytest.fixture
def diagnostics_candidate_results(
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> dict[str, dict]:
    """OOF results for all three candidates, keyed as run_candidate_step would."""
    X_dev, y_dev = dev_split
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
    return {
        name: cv_score_candidate(
            DummyClassifier(strategy="prior", random_state=42), X_dev, y_dev, cv
        )
        for name in ("dummy_prior", "logreg_cv", "lgbm_default")
    }


@pytest.fixture
def diagnostics_mlflow_uri(mlflow_test_experiment: Callable[[str], str]) -> None:
    """Point MLflow at the shared tmp-scoped experiment (conftest.py ::
    mlflow_test_experiment) so tests never touch mlruns/ — the explicit
    artifact_location there is what prevents the leak, not the SQLite backend
    alone: SQLite with no artifact_location still defaults artifacts to
    './mlruns' relative to CWD.

    mlflow>=3 deprecates the raw filesystem tracking backend (raises unless
    MLFLOW_ALLOW_FILE_STORE=true); SQLite is a supported local backend that needs
    no such escape hatch and no Docker/server dependency.
    """
    mlflow_test_experiment("test_run_diagnostics_step")


def test_run_diagnostics_step_covers_all_candidates(
    diagnostics_mlflow_uri: None,
    dev_split: tuple[pd.DataFrame, pd.Series],
    diagnostics_candidate_results: dict[str, dict],
) -> None:
    """Every candidate contributes rows to the fixed-recall profile."""
    X_dev, _ = dev_split
    cfg = OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
                "fixed_recall_thresholds": [0.70, 0.80],
                "segment_bootstrap_n_samples": 50,
            },
        }
    )

    with mlflow.start_run():
        result = comparison.run_diagnostics_step(
            X_dev,
            diagnostics_candidate_results,
            cfg,
            segment_n_bootstrap=int(cfg.training_setup.segment_bootstrap_n_samples),
        )

    candidates_seen = {row["candidate"] for row in result["fixed_recall"]}
    assert candidates_seen == {"dummy_prior", "logreg_cv", "lgbm_default"}


def test_run_diagnostics_step_fixed_recall_row_count(
    diagnostics_mlflow_uri: None,
    dev_split: tuple[pd.DataFrame, pd.Series],
    diagnostics_candidate_results: dict[str, dict],
) -> None:
    """One fixed-recall row per (candidate, recall_target) pair."""
    X_dev, _ = dev_split
    recall_targets = [0.70, 0.80, 0.90]
    cfg = OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
                "fixed_recall_thresholds": recall_targets,
                "segment_bootstrap_n_samples": 50,
            },
        }
    )

    with mlflow.start_run():
        result = comparison.run_diagnostics_step(
            X_dev,
            diagnostics_candidate_results,
            cfg,
            segment_n_bootstrap=int(cfg.training_setup.segment_bootstrap_n_samples),
        )

    assert len(result["fixed_recall"]) == 3 * len(recall_targets)


def test_run_diagnostics_step_includes_tenure_cohort_segment(
    diagnostics_mlflow_uri: None,
    dev_split: tuple[pd.DataFrame, pd.Series],
    diagnostics_candidate_results: dict[str, dict],
) -> None:
    """tenure_cohort is derived and included among the robustness segments."""
    X_dev, _ = dev_split
    cfg = OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
                "fixed_recall_thresholds": [0.70],
                "segment_bootstrap_n_samples": 50,
            },
        }
    )

    with mlflow.start_run():
        result = comparison.run_diagnostics_step(
            X_dev,
            diagnostics_candidate_results,
            cfg,
            segment_n_bootstrap=int(cfg.training_setup.segment_bootstrap_n_samples),
        )

    assert any(row["segment"] == "tenure_cohort" for row in result["robustness"])


def test_run_diagnostics_step_covers_all_robustness_and_fairness_segments(
    diagnostics_mlflow_uri: None,
    dev_split: tuple[pd.DataFrame, pd.Series],
    diagnostics_candidate_results: dict[str, dict],
) -> None:
    """All four fairness axes and all three robustness axes are represented."""
    X_dev, _ = dev_split
    cfg = OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
                "fixed_recall_thresholds": [0.70],
                "segment_bootstrap_n_samples": 50,
            },
        }
    )

    with mlflow.start_run():
        result = comparison.run_diagnostics_step(
            X_dev,
            diagnostics_candidate_results,
            cfg,
            segment_n_bootstrap=int(cfg.training_setup.segment_bootstrap_n_samples),
        )

    robustness_segments = {row["segment"] for row in result["robustness"]}
    fairness_segments = {row["segment"] for row in result["fairness"]}
    assert robustness_segments == set(comparison._ROBUSTNESS_SEGMENTS)
    assert fairness_segments == set(comparison._FAIRNESS_SEGMENTS)


def test_run_diagnostics_step_includes_segment_delta_cis(
    diagnostics_mlflow_uri: None,
    dev_split: tuple[pd.DataFrame, pd.Series],
    diagnostics_candidate_results: dict[str, dict],
) -> None:
    """robustness_delta/fairness_delta cover the same segment axes, not tagged per candidate."""
    X_dev, _ = dev_split
    cfg = OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
                "fixed_recall_thresholds": [0.70],
                "segment_bootstrap_n_samples": 50,
            },
        }
    )

    with mlflow.start_run():
        result = comparison.run_diagnostics_step(
            X_dev,
            diagnostics_candidate_results,
            cfg,
            segment_n_bootstrap=int(cfg.training_setup.segment_bootstrap_n_samples),
        )

    assert "candidate" not in result["robustness_delta"][0]
    assert {
        "segment",
        "value",
        "n",
        "delta_obs",
        "delta_ci_lower",
        "delta_ci_upper",
    } <= set(result["robustness_delta"][0])
    robustness_delta_segments = {row["segment"] for row in result["robustness_delta"]}
    fairness_delta_segments = {row["segment"] for row in result["fairness_delta"]}
    assert robustness_delta_segments <= set(comparison._ROBUSTNESS_SEGMENTS)
    assert fairness_delta_segments <= set(comparison._FAIRNESS_SEGMENTS)


# ---------------------------------------------------------------------------
# run_comparison_step (B4) — top-level Step 2 orchestrator
# ---------------------------------------------------------------------------
#
# Unlike run_diagnostics_step above, run_comparison_step opens and manages its own
# MLflow run (it calls run_diagnostics_step from inside that run), so it needs a real
# tracking URI rather than a caller-supplied `with mlflow.start_run():` block.


@pytest.fixture
def comparison_mlflow_uri(mlflow_test_experiment: Callable[[str], str]) -> str:
    """Point MLflow at the shared tmp-scoped experiment (conftest.py ::
    mlflow_test_experiment)."""
    return mlflow_test_experiment("test_run_comparison_step")


@pytest.fixture
def comparison_cfg() -> OmegaConf:
    """Minimal cfg for run_comparison_step + its nested run_diagnostics_step call."""
    return OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
                "bootstrap_n_samples": 200,
                "delta_threshold": 0.005,
                "fixed_recall_thresholds": [0.70, 0.80],
                "segment_bootstrap_n_samples": 50,
            },
            "mlflow": {
                "tracking_uri": "placeholder",
                "experiment_name": "test_run_comparison_step",
            },
        }
    )


def test_run_comparison_step_returns_bootstrap_and_diagnostics_keys(
    comparison_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    diagnostics_candidate_results: dict[str, dict],
    comparison_cfg: OmegaConf,
) -> None:
    """The returned dict is bootstrap_comparison's contract plus a 'diagnostics' key."""
    comparison_cfg.mlflow.tracking_uri = comparison_mlflow_uri
    X_dev, y_dev = dev_split

    result = comparison.run_comparison_step(
        X_dev,
        y_dev,
        diagnostics_candidate_results,
        comparison_cfg,
        n_bootstrap=int(comparison_cfg.training_setup.bootstrap_n_samples),
        segment_n_bootstrap=int(
            comparison_cfg.training_setup.segment_bootstrap_n_samples
        ),
    )

    assert set(result) == {
        "delta_obs",
        "delta_ci_lower",
        "delta_ci_upper",
        "p_value",
        "decision",
        "decision_rule",
        "n_bootstrap",
        "bootstrap_deltas",
        "diagnostics",
        "run_id",
    }
    assert set(result["diagnostics"]) == {
        "fixed_recall",
        "robustness",
        "fairness",
        "robustness_delta",
        "fairness_delta",
    }


def test_run_comparison_step_decision_matches_bootstrap_comparison(
    comparison_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    diagnostics_candidate_results: dict[str, dict],
    comparison_cfg: OmegaConf,
) -> None:
    """run_comparison_step's decision is exactly bootstrap_comparison's on the same
    inputs — a wiring regression test (e.g. swapped lgbm/logreg args, wrong threshold)
    would fail this even though it would pass the "returns valid keys" test above.
    """
    comparison_cfg.mlflow.tracking_uri = comparison_mlflow_uri
    X_dev, y_dev = dev_split
    ts = comparison_cfg.training_setup

    result = comparison.run_comparison_step(
        X_dev,
        y_dev,
        diagnostics_candidate_results,
        comparison_cfg,
        n_bootstrap=int(ts.bootstrap_n_samples),
        segment_n_bootstrap=int(ts.segment_bootstrap_n_samples),
    )

    expected = comparison.bootstrap_comparison(
        scores_lgbm=diagnostics_candidate_results["lgbm_default"]["pr_auc_scores"],
        scores_logreg=diagnostics_candidate_results["logreg_cv"]["pr_auc_scores"],
        n_bootstrap=int(ts.bootstrap_n_samples),
        delta_threshold=float(ts.delta_threshold),
        random_state=int(comparison_cfg.random_seed),
    )

    assert result["decision"] == expected["decision"]
    assert result["decision_rule"] == expected["decision_rule"]
    assert result["delta_obs"] == expected["delta_obs"]


def test_run_comparison_step_logs_comparison_and_diagnostics_artifacts(
    comparison_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    diagnostics_candidate_results: dict[str, dict],
    comparison_cfg: OmegaConf,
) -> None:
    """The model_comparison run carries the comparison table, both PR/bootstrap
    figures, and — via the nested run_diagnostics_step call — the five diagnostics
    artifacts, all on the *same* run (not scattered across separate ones).
    """
    comparison_cfg.mlflow.tracking_uri = comparison_mlflow_uri
    X_dev, y_dev = dev_split

    comparison.run_comparison_step(
        X_dev,
        y_dev,
        diagnostics_candidate_results,
        comparison_cfg,
        n_bootstrap=int(comparison_cfg.training_setup.bootstrap_n_samples),
        segment_n_bootstrap=int(
            comparison_cfg.training_setup.segment_bootstrap_n_samples
        ),
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_run_comparison_step")
    runs = client.search_runs(
        [experiment.experiment_id], filter_string="tags.stage = 'comparison'"
    )
    assert len(runs) == 1
    run = runs[0]

    assert run.data.params["decision"] in {"lgbm", "logreg"}
    assert "bootstrap_delta_obs" in run.data.metrics
    assert "roc_auc_lgbm" in run.data.metrics
    assert "pr_auc_oof_logreg" in run.data.metrics

    comparison_artifacts = {
        f.path for f in client.list_artifacts(run.info.run_id, "comparison")
    }
    assert comparison_artifacts == {
        "comparison/comparison_table.csv",
        "comparison/pr_curves.png",
        "comparison/bootstrap_delta_dist.png",
    }

    diagnostics_artifacts = {
        f.path for f in client.list_artifacts(run.info.run_id, "diagnostics")
    }
    assert diagnostics_artifacts == {
        "diagnostics/fixed_recall_profile.csv",
        "diagnostics/segment_robustness.csv",
        "diagnostics/segment_fairness.csv",
        "diagnostics/segment_robustness_delta.csv",
        "diagnostics/segment_fairness_delta.csv",
    }


def test_run_comparison_step_tags_stage_git_sha_and_data_content_hash(
    comparison_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    diagnostics_candidate_results: dict[str, dict],
    comparison_cfg: OmegaConf,
) -> None:
    """The run carries the same stage/git_sha/data_content_hash tag contract every other
    Phase 5 step run carries (candidates.py, feature_audit.py, tuning.py).
    """
    comparison_cfg.mlflow.tracking_uri = comparison_mlflow_uri
    X_dev, y_dev = dev_split

    comparison.run_comparison_step(
        X_dev,
        y_dev,
        diagnostics_candidate_results,
        comparison_cfg,
        n_bootstrap=int(comparison_cfg.training_setup.bootstrap_n_samples),
        segment_n_bootstrap=int(
            comparison_cfg.training_setup.segment_bootstrap_n_samples
        ),
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_run_comparison_step")
    run = client.search_runs([experiment.experiment_id])[0]

    assert run.data.tags["stage"] == "comparison"
    assert "git_sha" in run.data.tags
    assert "data_content_hash" in run.data.tags


def test_run_comparison_step_logs_dataset_source_via_accessor(
    comparison_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    diagnostics_candidate_results: dict[str, dict],
    comparison_cfg: OmegaConf,
) -> None:
    """The run's logged dataset input (Fix 6) resolves through
    features/accessor.py::features_path() — the same canonical source
    candidates.py's per-candidate runs already log — not a second,
    hand-assembled copy that could silently go stale.
    """
    comparison_cfg.mlflow.tracking_uri = comparison_mlflow_uri
    X_dev, y_dev = dev_split

    comparison.run_comparison_step(
        X_dev,
        y_dev,
        diagnostics_candidate_results,
        comparison_cfg,
        n_bootstrap=int(comparison_cfg.training_setup.bootstrap_n_samples),
        segment_n_bootstrap=int(
            comparison_cfg.training_setup.segment_bootstrap_n_samples
        ),
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_run_comparison_step")
    run = client.get_run(
        client.search_runs(
            [experiment.experiment_id], filter_string="tags.stage = 'comparison'"
        )[0].info.run_id
    )

    dataset_inputs = run.inputs.dataset_inputs
    assert len(dataset_inputs) == 1
    source = json.loads(dataset_inputs[0].dataset.source)
    assert source["uri"] == str(features_path())
