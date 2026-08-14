"""Unit tests for telco_churn.models.calibrate.

Step 1 (build/wrap) and Step 2 (select method) tests operate on a lightweight
Pipeline built directly in-process — no MLflow involved, since these functions
never touch the tracking server. Step 3 (run_calibration_step) tests go
through a real tmp-scoped MLflow experiment, reusing
models.train.log_model.run_model_logging_step to produce a realistic parent
run + training_manifest.json, exactly the chain this module consumes in
production.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import telco_churn.models.calibrate as calibrate
import telco_churn.models.register as register
import telco_churn.models.train.log_model as log_model
from telco_churn.features.build import FEATURE_SCHEMA, TARGET_COL
from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.models.train.common import _FEATURE_COLS

_OUTER_FOLDS = 3
_INNER_FOLDS = 3

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calibration_cfg() -> DictConfig:
    """Small fold counts for speed — same shape as production config, not its values."""
    return OmegaConf.create(
        {
            "calibration": {
                "method": "sigmoid",
                "outer_cv_folds": _OUTER_FOLDS,
                "inner_cv_folds": _INNER_FOLDS,
                "shuffle": True,
                "random_state": 42,
                "brier_bootstrap_n_samples": 200,
                "ece_n_bins": 5,
                "ece_strategy": "uniform",
            },
            "training_setup": {"delta_threshold": 0.005},
        }
    )


@pytest.fixture(scope="module")
def unfitted_pipeline() -> Pipeline:
    """The real tree-family ColumnTransformer, paired with a fast linear classifier.

    LogisticRegression stands in for LightGBM purely for test speed — these
    tests exercise calibrate.py's own CV-wrapping logic, not the model family.

    Module-scoped: build_calibrated_pipeline's own docstring establishes that
    CalibratedClassifierCV.fit clones the pipeline internally per inner fold,
    and oof_uncalibrated_proba explicitly clones it too — this object is never
    fit in place, so every consumer in this file can safely share one
    instance instead of rebuilding an identical unfitted Pipeline per test.
    """
    preprocessor = build_preprocessor(
        binary=list(FEATURE_SCHEMA.binary),
        multi_cat=list(FEATURE_SCHEMA.multi_cat),
        numeric=list(FEATURE_SCHEMA.numeric),
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


@pytest.fixture(scope="module")
def _module_feature_df() -> pd.DataFrame:
    """Module-scoped mirror of conftest.py::feature_df — identical body.
    Fixtures below need a module-scoped source frame and can't depend on the
    shared conftest fixture (function-scoped, would raise a pytest
    ScopeMismatch). Pure and deterministic (seeded rng), so a second copy is
    behaviourally identical to the original."""
    rng = np.random.default_rng(0)
    n = 120  # matches conftest.py::feature_df's _TRAIN_N
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


@pytest.fixture(scope="module")
def _module_dev_split(
    _module_feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Module-scoped mirror of conftest.py::dev_split — see _module_feature_df."""
    return _module_feature_df[_FEATURE_COLS], _module_feature_df[TARGET_COL]


@pytest.fixture(scope="module")
def pinned_sigmoid_result(
    unfitted_pipeline: Pipeline,
    _module_dev_split: tuple[pd.DataFrame, pd.Series],
) -> dict[str, Any]:
    """The real, unmocked select_calibration_method(method='sigmoid') result —
    shared because test_select_calibration_method_pinned_returns_proba_arrays
    and test_select_calibration_method_pinned_still_reports_other_method call
    it with byte-identical inputs (same unfitted_pipeline, same dev split, same
    pinned-sigmoid cfg) and only differ in which part of the result they
    assert on. Deterministic (random_state=42 throughout), so one real call —
    dummy + uncalibrated + sigmoid + isotonic OOF, each a nested outer/inner CV
    fit — replaces two, at ~40s each.

    Not shared with test_select_calibration_method_pinned_raises_on_gate_failure:
    that test monkeypatches pr_auc_gate_passes to force the raise branch, a
    genuinely different code path (it never reaches the other-method fit), so
    it keeps its own function-scoped setup.
    """
    X_dev, y_dev = _module_dev_split
    cfg = OmegaConf.create(
        {
            "calibration": {
                "method": "sigmoid",
                "outer_cv_folds": _OUTER_FOLDS,
                "inner_cv_folds": _INNER_FOLDS,
                "shuffle": True,
                "random_state": 42,
                "brier_bootstrap_n_samples": 200,
                "ece_n_bins": 5,
                "ece_strategy": "uniform",
            },
            "training_setup": {"delta_threshold": 0.005},
        }
    )
    result = calibrate.select_calibration_method(unfitted_pipeline, X_dev, y_dev, cfg)
    result["y_dev"] = y_dev
    return result


@pytest.fixture
def miscalibrated_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic data engineered to make GaussianNB overconfident, not just separable.

    n_redundant features are linear combinations of the informative ones —
    exactly the correlation GaussianNB's independence assumption can't
    represent, which is what pushes its raw probabilities toward 0/1 while
    ranking (AP) stays intact. Empirically verified before writing this test:
    pooled Brier improves by ~0.03 under sigmoid calibration on this exact
    setup, with per-fold mean AP unchanged.
    """
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=4,
        n_redundant=4,
        n_clusters_per_class=2,
        class_sep=2.2,
        flip_y=0.03,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    return X_df, pd.Series(y, name="churn")


@pytest.fixture
def miscalibrated_pipeline() -> Pipeline:
    """GaussianNB is the deliberately-miscalibrated candidate — see miscalibrated_data."""
    return Pipeline(steps=[("scaler", StandardScaler()), ("clf", GaussianNB())])


# ---------------------------------------------------------------------------
# Step 1: build/wrap the calibrated pipeline
# ---------------------------------------------------------------------------


def test_build_calibrated_pipeline_preprocessor_refit_count(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
) -> None:
    """Leak canary: ColumnTransformer.fit_transform fires exactly inner_folds + 1
    times under ensemble=False (one per inner fold, plus the final refit on all
    of development) — the only observable proof the preprocessor refits inside
    every calibration fold instead of leaking held-out statistics across them.
    """
    X_dev, y_dev = dev_split
    calibrated = calibrate.build_calibrated_pipeline(
        unfitted_pipeline, "sigmoid", calibration_cfg
    )

    with patch.object(
        ColumnTransformer,
        "fit_transform",
        autospec=True,
        side_effect=ColumnTransformer.fit_transform,
    ) as mock_fit_transform:
        calibrated.fit(X_dev, y_dev)

    assert mock_fit_transform.call_count == _INNER_FOLDS + 1


def test_build_calibrated_pipeline_single_calibrated_classifier(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
) -> None:
    """ensemble=False collapses calibrated_classifiers_ to length 1, whose
    .estimator is a Pipeline refit on all of development — the SHAP access
    path a later evaluation stage depends on.
    """
    X_dev, y_dev = dev_split
    calibrated = calibrate.build_calibrated_pipeline(
        unfitted_pipeline, "sigmoid", calibration_cfg
    )
    calibrated.fit(X_dev, y_dev)

    assert len(calibrated.calibrated_classifiers_) == 1
    assert isinstance(calibrated.calibrated_classifiers_[0].estimator, Pipeline)


def test_oof_calibrated_proba_run_twice_is_deterministic(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
) -> None:
    """Same explicit, seeded StratifiedKFold on both calls — bit-identical OOF
    vectors, not just close ones.
    """
    X_dev, y_dev = dev_split

    first = calibrate.oof_calibrated_proba(
        unfitted_pipeline, "sigmoid", X_dev, y_dev, calibration_cfg
    )
    second = calibrate.oof_calibrated_proba(
        unfitted_pipeline, "sigmoid", X_dev, y_dev, calibration_cfg
    )

    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# Step 2: select the calibration method
# ---------------------------------------------------------------------------


def test_calibration_improves_pooled_brier_on_miscalibrated_classifier(
    miscalibrated_pipeline: Pipeline,
    miscalibrated_data: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
) -> None:
    """Calibrated outer-OOF Brier beats uncalibrated on a classifier that is
    genuinely overconfident by construction — both scored on the same pooled
    outer-OOF vector, over the same folds.
    """
    X, y = miscalibrated_data
    uncal_proba = calibrate.oof_uncalibrated_proba(
        miscalibrated_pipeline, X, y, calibration_cfg
    )
    cal_proba = calibrate.oof_calibrated_proba(
        miscalibrated_pipeline, "sigmoid", X, y, calibration_cfg
    )

    assert calibrate.pooled_brier(cal_proba, y) < calibrate.pooled_brier(uncal_proba, y)


def test_calibration_slope_closer_to_one_after_calibration_on_miscalibrated_classifier(
    miscalibrated_pipeline: Pipeline,
    miscalibrated_data: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
) -> None:
    """The calibration slope numerically confirms what a reliability diagram
    can only show visually: a classifier that is overconfident by
    construction has a slope measurably below 1.0 on its raw OOF
    probabilities, and calibration pulls it back toward 1.0 — the same
    (y, proba) pairing run_calibration_step uses to log calibration_slope
    and uncalibrated_calibration_slope side by side.
    """
    X, y = miscalibrated_data
    uncal_proba = calibrate.oof_uncalibrated_proba(
        miscalibrated_pipeline, X, y, calibration_cfg
    )
    cal_proba = calibrate.oof_calibrated_proba(
        miscalibrated_pipeline, "sigmoid", X, y, calibration_cfg
    )

    uncal_slope = calibrate.calibration_slope(
        y, uncal_proba, n_bootstrap=200, random_state=42
    )
    cal_slope = calibrate.calibration_slope(
        y, cal_proba, n_bootstrap=200, random_state=42
    )

    assert abs(cal_slope["slope"] - 1.0) < abs(uncal_slope["slope"] - 1.0)


def test_calibration_pr_auc_within_delta_of_uncalibrated(
    miscalibrated_pipeline: Pipeline,
    miscalibrated_data: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
) -> None:
    """The sigmoid/isotonic gate: calibration must not degrade ranking even
    while it improves Brier — per-fold mean AP, never pooled.
    """
    X, y = miscalibrated_data
    uncal_proba = calibrate.oof_uncalibrated_proba(
        miscalibrated_pipeline, X, y, calibration_cfg
    )
    cal_proba = calibrate.oof_calibrated_proba(
        miscalibrated_pipeline, "sigmoid", X, y, calibration_cfg
    )
    uncal_ap = calibrate.per_fold_average_precision(uncal_proba, X, y, calibration_cfg)
    cal_ap = calibrate.per_fold_average_precision(cal_proba, X, y, calibration_cfg)

    assert calibrate.pr_auc_gate_passes(cal_ap, uncal_ap, calibration_cfg)


@pytest.mark.parametrize(
    ("candidate_ap", "uncalibrated_ap", "expected"),
    [
        pytest.param([0.60] * 5, [0.60] * 5, True, id="equal"),
        pytest.param([0.598] * 5, [0.60] * 5, True, id="just_within_delta"),
        pytest.param([0.585] * 5, [0.60] * 5, False, id="beyond_delta"),
        pytest.param([0.70] * 5, [0.60] * 5, True, id="candidate_better"),
    ],
)
def test_pr_auc_gate_passes_boundary(
    candidate_ap: list[float],
    uncalibrated_ap: list[float],
    expected: bool,
    calibration_cfg: DictConfig,
) -> None:
    """A candidate within Δ* = 0.005 of uncalibrated still passes; one clearly
    beyond it does not. (The exact bit-boundary at delta == -Δ* is deliberately
    not tested here — float subtraction near an exact threshold is fragile;
    pr_auc_gate_passes's `>=` is a one-line, self-evidently-inclusive check.)
    """
    assert (
        calibrate.pr_auc_gate_passes(candidate_ap, uncalibrated_ap, calibration_cfg)
        is expected
    )


@pytest.mark.parametrize(
    ("sigmoid_brier", "isotonic_brier", "expected_method", "expected_rule"),
    [
        pytest.param(
            [0.30, 0.28, 0.31, 0.29, 0.30, 0.32, 0.29, 0.31],
            [0.18, 0.16, 0.19, 0.17, 0.18, 0.20, 0.17, 0.19],
            "isotonic",
            "isotonic_win",
            id="isotonic_win",
        ),
        pytest.param(
            [0.16, 0.18, 0.17, 0.15, 0.17, 0.19, 0.16, 0.18],
            [0.30, 0.32, 0.29, 0.31, 0.30, 0.33, 0.29, 0.31],
            "sigmoid",
            "sigmoid_win",
            id="sigmoid_win",
        ),
        pytest.param(
            [0.20, 0.21, 0.19, 0.20, 0.22, 0.18, 0.21, 0.20],
            [0.20, 0.21, 0.19, 0.20, 0.22, 0.18, 0.21, 0.20],
            "sigmoid",
            "tie",
            id="tie_identical_folds",
        ),
    ],
)
def test_brier_switch_decision(
    sigmoid_brier: list[float],
    isotonic_brier: list[float],
    expected_method: str,
    expected_rule: str,
    calibration_cfg: DictConfig,
) -> None:
    """Three outcomes only: isotonic must decisively beat sigmoid to win — a
    tie or a sigmoid win both keep the incumbent.
    """
    result = calibrate.brier_switch_decision(
        sigmoid_brier, isotonic_brier, calibration_cfg
    )
    assert result["method"] == expected_method
    assert result["decision_rule"] == expected_rule


def test_select_calibration_method_pinned_returns_proba_arrays(
    pinned_sigmoid_result: dict[str, Any],
) -> None:
    """Pinned mode returns calibrated_proba/uncalibrated_proba aligned to y_dev
    — the arrays the reliability diagram is rendered from — not just diagnostics.
    """
    result = pinned_sigmoid_result
    y_dev = result["y_dev"]

    assert result["method"] == "sigmoid"
    assert len(result["calibrated_proba"]) == len(y_dev)
    assert len(result["uncalibrated_proba"]) == len(y_dev)
    assert result["switch_decision"] == {
        "method": "sigmoid",
        "decision_rule": "pinned",
        "delta_brier_obs": None,
        "delta_brier_ci_lower": None,
        "delta_brier_ci_upper": None,
        "n_bootstrap": None,
    }


def test_select_calibration_method_pinned_still_reports_other_method(
    pinned_sigmoid_result: dict[str, Any],
) -> None:
    """Pinned mode fits the unpinned method too, purely for reporting — so
    calibration_summary.json never carries only the winner's numbers, and a
    notebook/ANALYSIS.md citing the other method's score is citing a number
    this cycle actually produced. The unpinned method's diagnostics must not
    change `method` or `calibrated_proba`.

    Only exercises pinned_method="sigmoid": isotonic regression genuinely
    fails pr_auc_gate_passes on this fixture's small synthetic dev split (3
    outer/3 inner folds, LogisticRegression stand-in) — every other pinned-
    mode test in this file pins sigmoid for the same reason.
    """
    result = pinned_sigmoid_result

    assert result["method"] == "sigmoid"
    assert result["switch_decision"]["decision_rule"] == "pinned"
    assert set(result["diagnostics"]) == {
        "dummy_prior",
        "uncalibrated",
        "sigmoid",
        "isotonic",
    }
    for entry in result["diagnostics"].values():
        assert set(entry) == {"per_fold_mean_ap", "pooled_brier", "ece", "bss"}


def test_select_calibration_method_pinned_raises_on_gate_failure(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned method that regresses ranking on a later retrain must fail
    loudly, not register silently — pr_auc_gate_passes forced False to
    exercise the raise without needing a real classifier to regress.
    """
    X_dev, y_dev = dev_split
    calibration_cfg.calibration.method = "sigmoid"
    monkeypatch.setattr(calibrate, "pr_auc_gate_passes", lambda *args, **kwargs: False)

    with pytest.raises(ValueError, match="failed the PR-AUC gate"):
        calibrate.select_calibration_method(
            unfitted_pipeline, X_dev, y_dev, calibration_cfg
        )


def test_select_calibration_method_unknown_method_raises(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
) -> None:
    """Anything other than 'sigmoid' / 'isotonic' / 'auto' is a config typo, not
    a silently-ignored default."""
    X_dev, y_dev = dev_split
    calibration_cfg.calibration.method = "platt"

    with pytest.raises(ValueError, match="Unknown calibration.method"):
        calibrate.select_calibration_method(
            unfitted_pipeline, X_dev, y_dev, calibration_cfg
        )


@pytest.mark.parametrize(
    "winning_method", [pytest.param("sigmoid"), pytest.param("isotonic")]
)
def test_select_calibration_method_auto_calibrated_proba_matches_winner(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    winning_method: str,
) -> None:
    """calibrated_proba must be the winning method's OOF vector, not whichever
    was computed first — sigmoid and isotonic OOF are mocked to distinct
    sentinel arrays so this can't pass by coincidence.
    """
    X_dev, y_dev = dev_split
    calibration_cfg.calibration.method = "auto"
    n = len(y_dev)
    sentinels = {
        "sigmoid": np.full(n, 0.11),
        "isotonic": np.full(n, 0.22),
    }

    def fake_oof_calibrated_proba(
        pipeline: Pipeline,
        method: str,
        X: pd.DataFrame,
        y: pd.Series,
        cfg: DictConfig,
    ) -> np.ndarray:
        return sentinels[method]

    monkeypatch.setattr(calibrate, "oof_calibrated_proba", fake_oof_calibrated_proba)
    monkeypatch.setattr(
        calibrate,
        "pr_auc_gate_passes",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        calibrate,
        "brier_switch_decision",
        lambda *args, **kwargs: {
            "method": winning_method,
            "decision_rule": f"{winning_method}_win",
            "delta_brier_obs": 0.0,
            "delta_brier_ci_lower": 0.0,
            "delta_brier_ci_upper": 0.0,
            "n_bootstrap": 1,
        },
    )

    result = calibrate.select_calibration_method(
        unfitted_pipeline, X_dev, y_dev, calibration_cfg
    )

    assert result["method"] == winning_method
    assert np.array_equal(result["calibrated_proba"], sentinels[winning_method])


def test_select_calibration_method_auto_disqualifies_isotonic_on_pr_auc_gate(
    miscalibrated_pipeline: Pipeline,
    miscalibrated_data: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
) -> None:
    """Real, unmocked 'auto' run: isotonic overfits GaussianNB's small-sample
    OOF badly enough to fail the PR-AUC gate, so sigmoid ships instead — the
    exact isotonic_disqualified_pr_auc_gate branch that fired in the real
    production calibration run (ANALYSIS.md §5), previously exercised only via
    a fully-mocked pr_auc_gate_passes/brier_switch_decision and never actually
    reached in the test suite (calibrate.py lines 496-504 were uncovered).
    """
    X, y = miscalibrated_data
    calibration_cfg.calibration.method = "auto"

    result = calibrate.select_calibration_method(
        miscalibrated_pipeline, X, y, calibration_cfg
    )

    assert result["method"] == "sigmoid"
    assert result["switch_decision"] == {
        "method": "sigmoid",
        "decision_rule": "isotonic_disqualified_pr_auc_gate",
        "delta_brier_obs": None,
        "delta_brier_ci_lower": None,
        "delta_brier_ci_upper": None,
        "n_bootstrap": None,
    }
    assert "isotonic" in result["diagnostics"]


def test_select_calibration_method_auto_runs_real_brier_bootstrap_when_isotonic_eligible(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force isotonic past the PR-AUC gate — every per-fold AP/Brier value and
    the switch decision itself stay genuinely computed — to exercise the real
    Brier-bootstrap code path end-to-end (calibrate.py lines 515-534), only
    reachable before this test via a fully-mocked brier_switch_decision.
    """
    X_dev, y_dev = dev_split
    calibration_cfg.calibration.method = "auto"
    monkeypatch.setattr(calibrate, "pr_auc_gate_passes", lambda *args, **kwargs: True)

    result = calibrate.select_calibration_method(
        unfitted_pipeline, X_dev, y_dev, calibration_cfg
    )

    assert result["method"] in ("sigmoid", "isotonic")
    switch_decision = result["switch_decision"]
    assert switch_decision["decision_rule"] in ("isotonic_win", "sigmoid_win", "tie")
    assert "delta_brier_obs" in switch_decision
    assert "isotonic" in result["diagnostics"]
    assert "sigmoid" in result["diagnostics"]


# ---------------------------------------------------------------------------
# Step 3: register the training cycle's single deployable artifact
# ---------------------------------------------------------------------------


@pytest.fixture
def calibration_mlflow_uri(mlflow_test_experiment: Callable[[str], str]) -> str:
    """Point MLflow at the shared tmp-scoped experiment (conftest.py ::
    mlflow_test_experiment)."""
    return mlflow_test_experiment("test_run_calibration_step")


@pytest.fixture
def registration_cfg(calibration_mlflow_uri: str, tmp_path: Path) -> DictConfig:
    """Full cfg for run_calibration_step: training + calibration + mlflow + paths.

    paths.figures is an absolute tmp_path — get_project_root() / an absolute
    path resolves to the absolute path, so this sandboxes the reliability plot
    away from the real reports/figures/ directory.
    """
    return OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {"class_weight": "balanced", "delta_threshold": 0.005},
            "training": {
                "fixed": {
                    "subsample_freq": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "n_jobs": 1,
                    "verbose": -1,
                },
            },
            "tuning": {
                "cv_folds": 5,
                "es_validation_size": 0.2,
                "random_state": 42,
            },
            "calibration": {
                "method": "sigmoid",
                "outer_cv_folds": _OUTER_FOLDS,
                "inner_cv_folds": _INNER_FOLDS,
                "shuffle": True,
                "random_state": 42,
                "brier_bootstrap_n_samples": 200,
                "slope_bootstrap_n_samples": 50,
                "ece_n_bins": 5,
                "ece_strategy": "uniform",
                "run_id": None,
                "override_trial_count_gate": False,
                "golden_n_rows": 5,
            },
            # register.register_challenger's own config keys — this fixture
            # feeds both calibrate.run_calibration_step and (in
            # calibrated_run below) register.register_challenger, mirroring
            # the real two-step CLI flow (models.calibrate then
            # models.register) sharing one composed cfg.
            "register": {"golden_atol": 1.0e-9},
            "mlflow": {
                "tracking_uri": calibration_mlflow_uri,
                "experiment_name": "test_run_calibration_step",
                "registered_model_name": "test-telco-churn-pipeline",
            },
            "paths": {
                "figures": str(tmp_path / "figures"),
                "reports": str(tmp_path / "reports"),
            },
        }
    )


@pytest.fixture
def tuning_result(dev_split: tuple[pd.DataFrame, pd.Series]) -> dict[str, Any]:
    """A plausible Step 4 tuning output — small n_estimators for test speed."""
    X_dev, _ = dev_split
    return {
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
        "committed_features": list(X_dev.columns),
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


@pytest.fixture
def sandboxed_dev_features(
    monkeypatch: pytest.MonkeyPatch, dev_split: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Redirect run_calibration_step's load_dev_features()/load_dev_customer_ids()
    to the synthetic dev_split fixture, and stub log_model.features_sha256 for
    the _log_parent_run(...) -> log_model.run_model_logging_step(...) call
    every consumer of this fixture makes to produce its parent tuning_study run.

    Unpatched, load_dev_features()/load_dev_customer_ids() fall through to
    load_features() -> partition() -> load_split(), and
    run_model_logging_step's own data_content_hash stamp falls through to
    features_sha256() — all called with no path override, reading the real,
    gitignored datasets/processed/ directory. That happens to work today only
    because a real processed CSV + split manifest exist locally; on a fresh
    clone (no pipeline run yet), every Step 3 test would fail with
    FileNotFoundError before touching any calibration logic.
    """
    X_dev, y_dev = dev_split

    def _fake_load_dev_features(
        committed_features: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        return X_dev[committed_features], y_dev

    def _fake_load_dev_customer_ids() -> pd.Series:
        return pd.Series(
            [f"cust-{i:04d}" for i in range(len(y_dev))], name="customerid"
        )

    monkeypatch.setattr(calibrate, "load_dev_features", _fake_load_dev_features)
    monkeypatch.setattr(calibrate, "load_dev_customer_ids", _fake_load_dev_customer_ids)
    monkeypatch.setattr(log_model, "features_sha256", lambda path=None: "deadbeef" * 8)


def _log_parent_run(
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_result: dict[str, Any],
    cfg: DictConfig,
) -> str:
    """Reuse log_model.run_model_logging_step to produce a real tuning_study run
    with a valid training_manifest.json — the exact chain calibrate.py consumes
    in production, not a hand-built substitute.
    """
    X_dev, y_dev = dev_split
    with mlflow.start_run(run_name="tuning_study") as run:
        tuning_result = {**tuning_result, "parent_run_id": run.info.run_id}
    result = log_model.run_model_logging_step(X_dev, y_dev, tuning_result, cfg)
    return str(result["run_id"])


@pytest.fixture(scope="module")
def _shared_calibration_mlflow_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Point MLflow at a module-scoped tmp SQLite store — see calibrated_run's
    docstring for why 7 of the 9 run_calibration_step tests share one real
    call instead of each paying their own ~2.5-3.5s bootstrap + full
    nested-CV calibration cost."""
    tmp_path = tmp_path_factory.mktemp("calibrated_run_mlflow")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    experiment_id = mlflow.create_experiment(
        "test_run_calibration_step_shared", artifact_location=artifact_location
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


@pytest.fixture(scope="module")
def _shared_registration_cfg(
    _shared_calibration_mlflow_uri: str, tmp_path_factory: pytest.TempPathFactory
) -> DictConfig:
    """Module-scoped mirror of registration_cfg — see calibrated_run."""
    figures_dir = tmp_path_factory.mktemp("calibrated_run_figures")
    reports_dir = tmp_path_factory.mktemp("calibrated_run_reports")
    return OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {"class_weight": "balanced", "delta_threshold": 0.005},
            "training": {
                "fixed": {
                    "subsample_freq": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "n_jobs": 1,
                    "verbose": -1,
                },
            },
            "tuning": {
                "cv_folds": 5,
                "es_validation_size": 0.2,
                "random_state": 42,
            },
            "calibration": {
                "method": "sigmoid",
                "outer_cv_folds": _OUTER_FOLDS,
                "inner_cv_folds": _INNER_FOLDS,
                "shuffle": True,
                "random_state": 42,
                "brier_bootstrap_n_samples": 200,
                "slope_bootstrap_n_samples": 50,
                "ece_n_bins": 5,
                "ece_strategy": "uniform",
                "run_id": None,
                "override_trial_count_gate": False,
                "golden_n_rows": 5,
            },
            "register": {"golden_atol": 1.0e-9},
            "mlflow": {
                "tracking_uri": _shared_calibration_mlflow_uri,
                "experiment_name": "test_run_calibration_step_shared",
                "registered_model_name": "test-telco-churn-pipeline",
            },
            "paths": {"figures": str(figures_dir), "reports": str(reports_dir)},
        }
    )


@pytest.fixture(scope="module")
def _shared_tuning_result(
    _module_dev_split: tuple[pd.DataFrame, pd.Series],
) -> dict[str, Any]:
    """Module-scoped mirror of tuning_result — see calibrated_run."""
    X_dev, _ = _module_dev_split
    return {
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
        "committed_features": list(X_dev.columns),
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


@pytest.fixture(scope="module")
def calibrated_run(
    _shared_registration_cfg: DictConfig,
    _module_dev_split: tuple[pd.DataFrame, pd.Series],
    _shared_tuning_result: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """The real, unmocked run_calibration_step(sigmoid, default cfg) result,
    plus a real register.register_challenger mint on top of it (B1: calibrate.py
    itself performs no registry write, so the registry-dependent tests below
    need this fixture to also mint, mirroring what register.py's own
    mint-mode CLI would do as a separate step) — shared across 7 of the 9
    test_run_calibration_step_* tests, which only read different artifacts
    off one real call rather than each re-running the full nested-CV
    calibration + MLflow registration from scratch (~7-14s each, dominated
    by setup: a fresh SQLite MLflow store plus the same
    dummy+uncalibrated+sigmoid+isotonic OOF computation
    select_calibration_method always performs in pinned mode).

    NOT shared with:
    - test_run_calibration_step_calibration_summary_has_calibration_spec:
      uses calibration.method='auto', a different scenario (see that test's
      own docstring for why it can't be pinned either).
    - test_run_calibration_step_blocks_on_low_trial_count: asserts
      client.search_registered_models() == [] — requires a genuinely empty
      registry, which sharing this fixture's already-registered model would
      break.
    - test_run_calibration_step_override_trial_count_gate: asserts
      result["model_version"] == "1" — requires being the first-ever
      registration under this registered_model_name.
    All three keep their own function-scoped registration_cfg/tuning_result/
    sandboxed_dev_features (above), untouched by this
    fixture and its separate "test_run_calibration_step_shared" experiment.

    log_model.features_sha256 is stubbed too: _log_parent_run's
    run_model_logging_step call stamps data_content_hash unconditionally,
    with no path override, which would otherwise hit the same real,
    gitignored processed-features file sandboxed_dev_features exists to avoid.
    """
    mp = pytest.MonkeyPatch()
    X_dev, y_dev = _module_dev_split

    def _fake_load_dev_features(
        committed_features: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        return X_dev[committed_features], y_dev

    def _fake_load_dev_customer_ids() -> pd.Series:
        return pd.Series(
            [f"cust-{i:04d}" for i in range(len(y_dev))], name="customerid"
        )

    mp.setattr(calibrate, "load_dev_features", _fake_load_dev_features)
    mp.setattr(calibrate, "load_dev_customer_ids", _fake_load_dev_customer_ids)
    mp.setattr(log_model, "features_sha256", lambda path=None: "deadbeef" * 8)

    run_id = _log_parent_run(
        _module_dev_split,
        _shared_tuning_result,
        _shared_registration_cfg,
    )
    result = calibrate.run_calibration_step(run_id, _shared_registration_cfg)
    model_version = register.register_challenger(
        _shared_registration_cfg,
        run_id,
        result["model_uri"],
        result["logged_model_id"],
    )
    try:
        yield {
            "run_id": run_id,
            "result": result,
            "model_version": model_version,
            "cfg": _shared_registration_cfg,
            "X_dev": X_dev,
            "y_dev": y_dev,
        }
    finally:
        mp.undo()


def test_run_calibration_step_registers_and_tags(
    calibrated_run: dict[str, Any],
) -> None:
    """register.register_challenger (run separately, on top of
    run_calibration_step's output — B1's decoupling) registers exactly one
    version, tags training_data_scope=dev, and points challenger at it.
    """
    run_id = calibrated_run["run_id"]
    result = calibrated_run["result"]
    model_version = calibrated_run["model_version"]
    registration_cfg = calibrated_run["cfg"]
    # calibrated_run is module-scoped and cached — an interleaved test using
    # its own separate MLflow store (e.g. the auto-method calibration_spec
    # test between this group) can leave the process-global tracking URI
    # pointed elsewhere by the time this test runs; re-assert it rather than
    # relying on execution order.
    mlflow.set_tracking_uri(str(registration_cfg.mlflow.tracking_uri))

    assert result["run_id"] == run_id
    assert result["method"] == "sigmoid"

    client = mlflow.tracking.MlflowClient()
    registered_name = str(registration_cfg.mlflow.registered_model_name)
    version = client.get_model_version(registered_name, model_version)
    assert version.tags["training_data_scope"] == "dev"
    # ModelVersion.model_id does not auto-populate in OSS MLflow 3.14 — without
    # this tag the registry has no supported path to the LoggedModel Phase 7's
    # evaluate.py attaches sealed-test metrics to.
    assert version.tags["logged_model_id"]
    assert mlflow.get_logged_model(version.tags["logged_model_id"]) is not None

    registered_model = client.get_registered_model(registered_name)
    assert str(registered_model.aliases["challenger"]) == model_version


def test_run_calibration_step_tags_promotion_status_pending(
    calibrated_run: dict[str, Any],
) -> None:
    """Mint-time default: a freshly registered version is tagged
    promotion_status=pending before register.py's own promote/reject path
    has ever seen it — the fail-safe state a crash anywhere downstream
    leaves it in.
    """
    model_version = calibrated_run["model_version"]
    registration_cfg = calibrated_run["cfg"]
    mlflow.set_tracking_uri(str(registration_cfg.mlflow.tracking_uri))

    client = mlflow.tracking.MlflowClient()
    registered_name = str(registration_cfg.mlflow.registered_model_name)
    version = client.get_model_version(registered_name, model_version)
    assert version.tags["promotion_status"] == "pending"


def test_run_calibration_step_logs_golden_predictions(
    calibrated_run: dict[str, Any],
) -> None:
    """golden_predictions.json is the independent reference register.py's
    serving-parity smoke check verifies against later, in a different
    process — so it must be self-contained (rows, not just scores), pinned
    by customerid, and round-trip through the registered model exactly.
    """
    run_id = calibrated_run["run_id"]
    model_version = calibrated_run["model_version"]
    registration_cfg = calibrated_run["cfg"]
    mlflow.set_tracking_uri(str(registration_cfg.mlflow.tracking_uri))
    n_rows = int(registration_cfg.calibration.golden_n_rows)

    golden = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/calibration/golden_predictions.json"
    )

    assert golden["purpose"] == (
        "serving-parity fixture — reproducibility only; scores are "
        "in-sample and are not performance evidence"
    )
    assert golden["customerid"] == sorted(golden["customerid"])
    assert len(golden["customerid"]) == n_rows
    assert len(golden["rows"]) == n_rows
    assert len(golden["p_hat"]) == n_rows
    assert all(0.0 <= p <= 1.0 for p in golden["p_hat"])

    registered_name = str(registration_cfg.mlflow.registered_model_name)
    reloaded = mlflow.sklearn.load_model(f"models:/{registered_name}/{model_version}")
    reloaded_preds = reloaded.predict_proba(pd.DataFrame(golden["rows"]))[:, 1]
    assert np.allclose(reloaded_preds, golden["p_hat"], rtol=0, atol=1e-9)


def test_run_calibration_step_logs_reliability_diagram(
    calibrated_run: dict[str, Any],
) -> None:
    """The pre/post-calibration reliability diagram is logged onto the run's
    figures/ artifacts alongside calibration_summary.json — not left to a
    notebook to remember to produce.
    """
    run_id = calibrated_run["run_id"]
    registration_cfg = calibrated_run["cfg"]
    mlflow.set_tracking_uri(str(registration_cfg.mlflow.tracking_uri))

    client = mlflow.tracking.MlflowClient()
    artifact_paths = {
        a.path for a in client.list_artifacts(run_id, "calibration/figures")
    }
    assert "calibration/figures/reliability_diagram.png" in artifact_paths


def test_run_calibration_step_logs_dev_oof_predictions(
    calibrated_run: dict[str, Any],
) -> None:
    """The winning method's dev-OOF vector — the numbers that selected it,
    produced its BSS, and will validate t* — is persisted as a run artifact,
    not left for a downstream consumer to recompute (CLAUDE.md § Persist the
    evidence, not just the conclusion).
    """
    run_id = calibrated_run["run_id"]
    y_dev = calibrated_run["y_dev"]
    registration_cfg = calibrated_run["cfg"]
    mlflow.set_tracking_uri(str(registration_cfg.mlflow.tracking_uri))

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="calibration/dev_oof_predictions.parquet"
    )
    oof = pd.read_parquet(local_path)

    assert list(oof.columns) == ["customerid", "y_true", "p_hat"]
    assert len(oof) == len(y_dev)
    assert oof["p_hat"].between(0, 1).all()


def test_run_calibration_step_logs_dev_shap_values(
    calibrated_run: dict[str, Any],
) -> None:
    """dev_shap_values.parquet — the evidence threshold.py's V3 pre-seal
    screen binds on — carries one row per dev customer, one column per
    transformed feature, plus customerid and base_value (CLAUDE.md § Persist
    the evidence, not just the conclusion).
    """
    run_id = calibrated_run["run_id"]
    y_dev = calibrated_run["y_dev"]
    registration_cfg = calibrated_run["cfg"]
    mlflow.set_tracking_uri(str(registration_cfg.mlflow.tracking_uri))

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="calibration/dev_shap_values.parquet"
    )
    shap_df = pd.read_parquet(local_path)

    assert len(shap_df) == len(y_dev)
    assert {"customerid", "base_value"} <= set(shap_df.columns)
    feature_cols = [c for c in shap_df.columns if c not in ("customerid", "base_value")]
    assert len(feature_cols) > 0
    assert shap_df[feature_cols].to_numpy().dtype.kind == "f"


def test_run_calibration_step_logs_dev_shap_summary(
    calibrated_run: dict[str, Any],
) -> None:
    """dev_shap_summary.json ranks every transformed feature by mean_abs_shap
    (descending, matching explain.global_importance's sort) and carries a
    signed direction alongside it — the ranking threshold.py's V3 pre-seal
    veto is derived from and binds on.
    """
    run_id = calibrated_run["run_id"]
    registration_cfg = calibrated_run["cfg"]
    mlflow.set_tracking_uri(str(registration_cfg.mlflow.tracking_uri))

    summary = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/calibration/dev_shap_summary.json"
    )

    assert len(summary) > 0
    for row in summary:
        assert {"feature", "mean_abs_shap", "direction"} <= set(row)
        assert row["mean_abs_shap"] >= 0.0
        assert -1.0 <= row["direction"] <= 1.0
    mean_abs_shap_values = [row["mean_abs_shap"] for row in summary]
    assert mean_abs_shap_values == sorted(mean_abs_shap_values, reverse=True)


def test_run_calibration_step_calibration_summary_has_calibration_spec(
    registration_cfg: DictConfig,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_result: dict[str, Any],
    sandboxed_dev_features: None,
) -> None:
    """calibration_summary.json's calibration_spec carries the four fields
    that reconstruct the fitted CalibratedClassifierCV, and its method is
    always a concrete family — never 'auto', the value that must not survive
    into the artifact. Asserted with calibration.method='auto' rather than a
    pinned fixture, since a pinned fixture would pass vacuously.
    """
    registration_cfg.calibration.method = "auto"
    run_id = _log_parent_run(dev_split, tuning_result, registration_cfg)

    result = calibrate.run_calibration_step(run_id, registration_cfg)

    spec = result["calibration_summary"]["calibration_spec"]
    assert spec["method"] == result["method"]
    assert spec["method"] in ("sigmoid", "isotonic")
    assert spec["inner_cv_folds"] == _INNER_FOLDS
    assert spec["random_state"] == int(registration_cfg.calibration.random_state)
    assert spec["ensemble"] is False

    slope = result["calibration_summary"]["calibration_slope"]
    assert {"slope", "intercept", "slope_ci_lower", "slope_ci_upper"} <= set(slope)

    uncalibrated_slope = result["calibration_summary"]["uncalibrated_calibration_slope"]
    assert {"slope", "intercept", "slope_ci_lower", "slope_ci_upper"} <= set(
        uncalibrated_slope
    )

    client = mlflow.tracking.MlflowClient()
    artifact_paths = {a.path for a in client.list_artifacts(run_id, "calibration")}
    assert "calibration/calibration_summary.json" in artifact_paths


def test_run_calibration_step_logs_dev_metrics(
    calibrated_run: dict[str, Any],
) -> None:
    """dev_brier/dev_bss/dev_ece/dev_per_fold_mean_ap/dev_calibration_slope
    land in the run's metrics panel, not only inside calibration_summary.json
    — so they are plottable as a series across retraining cycles, matching
    the winning method's own diagnostics and slope.
    """
    run_id = calibrated_run["run_id"]
    result = calibrated_run["result"]
    registration_cfg = calibrated_run["cfg"]
    mlflow.set_tracking_uri(str(registration_cfg.mlflow.tracking_uri))

    winning_diagnostics = result["calibration_summary"]["diagnostics"][result["method"]]
    expected_slope = result["calibration_summary"]["calibration_slope"]["slope"]

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    assert run.data.metrics["dev_brier"] == pytest.approx(
        winning_diagnostics["pooled_brier"]
    )
    assert run.data.metrics["dev_bss"] == pytest.approx(winning_diagnostics["bss"])
    assert run.data.metrics["dev_ece"] == pytest.approx(winning_diagnostics["ece"])
    assert run.data.metrics["dev_per_fold_mean_ap"] == pytest.approx(
        winning_diagnostics["per_fold_mean_ap"]
    )
    assert run.data.metrics["dev_calibration_slope"] == pytest.approx(expected_slope)

    summary = result["calibration_summary"]
    assert run.data.metrics["dev_calibration_slope_ci_lower"] == pytest.approx(
        summary["calibration_slope"]["slope_ci_lower"]
    )
    assert run.data.metrics["dev_calibration_slope_ci_upper"] == pytest.approx(
        summary["calibration_slope"]["slope_ci_upper"]
    )
    assert run.data.metrics["dev_uncalibrated_calibration_slope"] == pytest.approx(
        summary["uncalibrated_calibration_slope"]["slope"]
    )
    assert run.data.metrics[
        "dev_uncalibrated_calibration_slope_ci_lower"
    ] == pytest.approx(summary["uncalibrated_calibration_slope"]["slope_ci_lower"])
    assert run.data.metrics[
        "dev_uncalibrated_calibration_slope_ci_upper"
    ] == pytest.approx(summary["uncalibrated_calibration_slope"]["slope_ci_upper"])
    assert run.data.metrics["dev_mean_p_hat_calibrated"] == pytest.approx(
        summary["mean_p_hat_calibrated"]
    )
    assert run.data.metrics["dev_mean_p_hat_uncalibrated"] == pytest.approx(
        summary["mean_p_hat_uncalibrated"]
    )
    assert run.data.metrics["dev_observed_churn_rate"] == pytest.approx(
        summary["observed_churn_rate"]
    )


def test_run_calibration_step_calibration_summary_has_mean_p_hat_fields(
    calibrated_run: dict[str, Any],
) -> None:
    """mean_p_hat_calibrated/mean_p_hat_uncalibrated/observed_churn_rate are
    persisted in calibration_summary.json — the "calibration-in-the-large"
    evidence a slope near 1 alone can hide (a large intercept shows up here
    as a mean p_hat far from the observed rate), not left to a notebook's
    own .mean() call over an already-persisted OOF vector.
    """
    y_dev = calibrated_run["y_dev"]
    result = calibrated_run["result"]
    summary = result["calibration_summary"]

    assert 0.0 <= summary["mean_p_hat_calibrated"] <= 1.0
    assert 0.0 <= summary["mean_p_hat_uncalibrated"] <= 1.0
    assert summary["observed_churn_rate"] == pytest.approx(y_dev.mean())


def test_run_calibration_step_blocks_on_low_trial_count(
    registration_cfg: DictConfig,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_result: dict[str, Any],
    sandboxed_dev_features: None,
) -> None:
    """A data-quality gate on the tuning result, not a performance comparison
    — too few completed Optuna trials to trust the 1-SE pick blocks
    registration outright.
    """
    tuning_result["tuning_summary"]["trial_count_below_threshold"] = True
    run_id = _log_parent_run(dev_split, tuning_result, registration_cfg)

    with pytest.raises(RuntimeError, match="trial_count_below_threshold"):
        calibrate.run_calibration_step(run_id, registration_cfg)

    client = mlflow.tracking.MlflowClient()
    assert client.search_registered_models() == []


def test_run_calibration_step_override_trial_count_gate(
    registration_cfg: DictConfig,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_result: dict[str, Any],
    sandboxed_dev_features: None,
) -> None:
    """override_trial_count_gate=true forces calibration to proceed despite
    the low-trial warning — an explicit human override, not a silent bypass.
    Registration is register.py's own separate step (B1) and is not
    exercised here — this only asserts calibrate.py itself completes."""
    tuning_result["tuning_summary"]["trial_count_below_threshold"] = True
    registration_cfg.calibration.override_trial_count_gate = True
    run_id = _log_parent_run(dev_split, tuning_result, registration_cfg)

    result = calibrate.run_calibration_step(run_id, registration_cfg)

    assert result["run_id"] == run_id
    assert result["logged_model_id"]

    client = mlflow.tracking.MlflowClient()
    assert client.search_registered_models() == []
