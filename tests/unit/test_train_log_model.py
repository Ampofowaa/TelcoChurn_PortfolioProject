"""Unit tests for telco_churn.models.train.log_model — Step 5 (C2)."""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import telco_churn.models.train.log_model as log_model

# ---------------------------------------------------------------------------
# run_model_logging_step (C2)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tuning_mlflow_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Point MLflow at a module-scoped tmp SQLite store with an explicit
    artifact_location — inlines conftest.py::mlflow_test_experiment's logic
    rather than requesting it, since that fixture depends on the
    function-scoped tmp_path and this fixture is module-scoped: every test
    below starts its own fresh parent run (_start_parent_run) and only
    inspects that run, never enumerates the experiment's full run list, so
    sharing one experiment (and skipping the ~2.5-3.5s SQLite-store
    bootstrap cost per test) is safe. registration_cfg stays function-scoped
    deliberately — several tests mutate it per-scenario, so it can't be
    shared."""
    tmp_path = tmp_path_factory.mktemp("log_model_mlflow")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    experiment_id = mlflow.create_experiment(
        "test_run_model_logging_step", artifact_location=artifact_location
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


@pytest.fixture
def registration_cfg() -> OmegaConf:
    """Minimal cfg for run_model_logging_step — training.fixed + a registry name."""
    return OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {"class_weight": "balanced"},
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
                "selection_rule": "1se",
                "cv_folds": 5,
                "es_validation_size": 0.2,
                "random_state": 42,
            },
            "mlflow": {
                "tracking_uri": "placeholder",
                "experiment_name": "test_run_model_logging_step",
                "registered_model_name": "test-telco-churn-pipeline",
            },
        }
    )


def _start_parent_run() -> str:
    """Create a real finished run to stand in for the Step 4 tuning_study parent.

    Relies on the tuning_mlflow_uri fixture already having set the tracking URI
    and active experiment — re-setting them here was redundant and, once the
    active experiment carries an explicit artifact_location (see conftest.py ::
    mlflow_test_experiment), re-resolving it by name only accidentally lands on
    the same experiment.
    """
    with mlflow.start_run(run_name="tuning_study") as run:
        return str(run.info.run_id)


@pytest.fixture
def tuning_result(dev_split: tuple[pd.DataFrame, pd.Series]) -> dict:
    """A plausible Step 4 output — 8 LightGBM hyperparameters + a committed feature list."""
    X_dev, _ = dev_split
    return {
        "best_params": {
            "num_leaves": 15,
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "max_depth": 5,
        },
        "best_n_estimators_median": 20,
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


def test_run_model_logging_step_returns_expected_keys(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
) -> None:
    """run_model_logging_step logs and reloads, returning the documented keys —
    no 'version', since this step never registers.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    result = log_model.run_model_logging_step(
        X_dev, y_dev, tuning_result, registration_cfg
    )

    assert set(result) == {
        "run_id",
        "model_uri",
        "parity_ok",
        "training_manifest",
    }
    assert result["run_id"] == tuning_result["parent_run_id"]
    assert result["parity_ok"] is True


def test_run_model_logging_step_raises_on_hyperparameter_collision(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed_hyperparameters key colliding with a selected_hyperparameters key
    (best_params or the derived n_estimators) raises, naming the offending key
    — guards the fixed: block's override precedence against silently deciding
    a value Optuna also searched.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    monkeypatch.setattr(
        log_model,
        "_lgbm_fixed_knobs",
        lambda cfg, random_state: {"num_leaves": 99, "n_jobs": 1},
    )

    with pytest.raises(ValueError, match="num_leaves"):
        log_model.run_model_logging_step(X_dev, y_dev, tuning_result, registration_cfg)


def test_run_model_logging_step_training_manifest_has_expected_fields(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
) -> None:
    """training_manifest.json is grouped one section per pipeline step —
    model_family_committed, feature_selection, training_summary, tuning_summary —
    with a logged_model_uri, but no 'alias', since this step never registers.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    result = log_model.run_model_logging_step(
        X_dev, y_dev, tuning_result, registration_cfg
    )

    manifest = result["training_manifest"]

    # tuning_summary: everything run_tuning_step returned, passed through
    # unchanged, plus selected_hyperparameters added at manifest-build time.
    for key, value in tuning_result["tuning_summary"].items():
        assert manifest["tuning_summary"][key] == value
    assert manifest["tuning_summary"]["selected_hyperparameters"]["num_leaves"] == 15
    # Not the raw median (20) — Fix 5 scales it by n_final_fit / n_fold_fit before
    # shipping. See test_run_model_logging_step_scales_n_estimators below for the
    # dedicated correctness proof against fixtures with differing fold/final sizes.
    assert manifest["tuning_summary"]["selected_hyperparameters"]["n_estimators"] != 20
    assert manifest["tuning_summary"]["selected_cv_pr_auc"] == 0.6

    # training_summary: the fixed knobs applied to every fit, not tuning output.
    assert (
        manifest["training_summary"]["fixed_hyperparameters"]["class_weight"]
        == "balanced"
    )
    assert "num_leaves" not in manifest["training_summary"]["fixed_hyperparameters"]

    # model_family_committed: Steps 1-2's frozen decision, referenced not recomputed.
    assert manifest["model_family_committed"]["model_family"] == "lightgbm"
    assert manifest["model_family_committed"]["decision_reference"] == "ANALYSIS.md §4a"
    assert manifest["model_family_committed"]["decision_run_id"]

    # feature_selection: Step 3's frozen input space.
    assert (
        manifest["feature_selection"]["model_features"]
        == tuning_result["committed_features"]
    )
    assert set(manifest["feature_selection"]["feature_space"]) >= set(
        tuning_result["committed_features"]
    )

    assert "git_sha" in manifest
    assert "data_content_hash" in manifest
    assert manifest["logged_model_uri"] == result["model_uri"]
    # LoggedModel.model_id — distinct from run_id and not auto-populated onto
    # ModelVersion in OSS MLflow 3.14; must be persisted here or the registry
    # has no supported path to the model it points at.
    assert manifest["logged_model_id"]
    assert mlflow.get_logged_model(manifest["logged_model_id"]) is not None

    # old flat/duplicate top-level keys must be gone, not just supplemented.
    for stale_key in (
        "hyperparameters",
        "cv_pr_auc_mean",
        "tuning_selection_rule",
        "feature_space",
        "feature_columns",
        "paired_delta_vs_logreg",
        "alias",
        "version",
    ):
        assert stale_key not in manifest


def test_run_model_logging_step_does_not_register(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
) -> None:
    """No registry version is created — the registry stays empty until Phase 6's
    calibrate.py performs the training cycle's single registration.

    CLAUDE.md: an uncalibrated pipeline is a stage of construction, not a valid
    rollback target, and must never occupy a registry version number.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    log_model.run_model_logging_step(X_dev, y_dev, tuning_result, registration_cfg)

    client = mlflow.tracking.MlflowClient()
    assert client.search_registered_models() == []


def test_run_model_logging_step_signature_declares_float_output(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
) -> None:
    """The logged signature declares a float probability output, not int64 —
    proof pyfunc_predict_fn="predict_proba" is actually wired, not just the
    parity check (which calls predict_proba directly on the sklearn flavour and
    would pass even if the pyfunc flavour still defaulted to predict()).
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    result = log_model.run_model_logging_step(
        X_dev, y_dev, tuning_result, registration_cfg
    )

    model_info = mlflow.models.get_model_info(result["model_uri"])
    assert model_info.signature is not None
    output_types = [str(spec.type) for spec in model_info.signature.outputs.inputs]
    assert all("float" in t for t in output_types)
    assert not any("int" in t for t in output_types)


def test_run_model_logging_step_parity_failure_raises(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload that disagrees with the in-memory model aborts logging."""
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    class _WrongPredictions:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros((len(X), 2))  # guaranteed mismatch vs. real probabilities

    monkeypatch.setattr(
        log_model.mlflow.sklearn, "load_model", lambda uri: _WrongPredictions()
    )

    with pytest.raises(AssertionError, match="Reload parity check failed"):
        log_model.run_model_logging_step(X_dev, y_dev, tuning_result, registration_cfg)


# ---------------------------------------------------------------------------
# Fix 5: tree-count scaling correction
# ---------------------------------------------------------------------------


def test_run_model_logging_step_scales_n_estimators(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
) -> None:
    """The shipped n_estimators is the scaled value, not the raw early-stopped
    median — asserted against this fixture's real fold/final sizes (120 dev
    rows, cv_folds=5, es_validation_size=0.2 -> n_fold_fit=77 != n_final_fit=120),
    since a fixture where the two sizes coincide would pass either way.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    result = log_model.run_model_logging_step(
        X_dev, y_dev, tuning_result, registration_cfg
    )
    tuning_summary = result["training_manifest"]["tuning_summary"]

    n_final_fit = len(y_dev)
    cv_folds = int(registration_cfg.tuning.cv_folds)
    es_validation_size = float(registration_cfg.tuning.es_validation_size)
    expected_n_fold_fit = round(
        n_final_fit * (cv_folds - 1) / cv_folds * (1 - es_validation_size)
    )
    expected_scale_factor = n_final_fit / expected_n_fold_fit
    expected_n_estimators = round(
        int(tuning_result["best_n_estimators_median"]) * expected_scale_factor
    )

    assert expected_n_fold_fit != n_final_fit, (
        "fixture's fold and final sizes coincide — this test would pass "
        "whether or not the scaling correction is actually applied"
    )
    assert tuning_summary["n_final_fit"] == n_final_fit
    assert tuning_summary["n_fold_fit"] == expected_n_fold_fit
    assert tuning_summary["n_estimators_scale_factor"] == pytest.approx(
        expected_scale_factor
    )
    assert tuning_summary["n_estimators_es_median"] == int(
        tuning_result["best_n_estimators_median"]
    )
    assert tuning_summary["n_estimators_shipped"] == expected_n_estimators
    assert (
        tuning_summary["selected_hyperparameters"]["n_estimators"]
        == expected_n_estimators
    )


def test_run_model_logging_step_records_two_count_diagnostic(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
) -> None:
    """cv_pr_auc_at_n_es_median / cv_pr_auc_at_n_scaled land in tuning_summary
    and as MLflow metrics — the diagnostic confirming the scaling rule against
    this project's own data, not just by citation.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    result = log_model.run_model_logging_step(
        X_dev, y_dev, tuning_result, registration_cfg
    )
    tuning_summary = result["training_manifest"]["tuning_summary"]

    assert isinstance(tuning_summary["cv_pr_auc_at_n_es_median"], float)
    assert isinstance(tuning_summary["cv_pr_auc_at_n_scaled"], float)

    run = mlflow.get_run(result["run_id"])
    assert run.data.metrics["cv_pr_auc_at_n_es_median"] == pytest.approx(
        tuning_summary["cv_pr_auc_at_n_es_median"]
    )
    assert run.data.metrics["cv_pr_auc_at_n_scaled"] == pytest.approx(
        tuning_summary["cv_pr_auc_at_n_scaled"]
    )


def test_run_model_logging_step_two_count_diagnostic_mints_no_model(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly one mlflow.sklearn.log_model call per cycle — the two-count
    diagnostic fits plain estimators in memory and must never register a
    second candidate, regardless of which count it favours.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    real_log_model = log_model.mlflow.sklearn.log_model
    call_count = 0

    def _counting_log_model(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        return real_log_model(*args, **kwargs)

    monkeypatch.setattr(log_model.mlflow.sklearn, "log_model", _counting_log_model)

    log_model.run_model_logging_step(X_dev, y_dev, tuning_result, registration_cfg)

    assert call_count == 1


def test_run_model_logging_step_warns_when_scaling_regresses(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scaled count that scores worse than the raw median logs a warning for
    manual investigation — the diagnostic never silently ships the textbook
    answer on faith, but also never blocks registration by itself (Fix 5 is
    a check, not a selection). Monkeypatches the module logger directly rather
    than relying on caplog, since structlog isn't routed through stdlib
    logging in the test process without configure_logging() having run.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    # Force the scaled count to lose regardless of the real CV scores, so this
    # test doesn't depend on the synthetic fixture's data happening to regress.
    scores = iter([0.70, 0.50])

    def _fake_cv_pr_auc_at_n_estimators(*args: object, **kwargs: object) -> float:
        return next(scores)

    monkeypatch.setattr(
        log_model, "_cv_pr_auc_at_n_estimators", _fake_cv_pr_auc_at_n_estimators
    )

    warning_events: list[str] = []
    monkeypatch.setattr(
        log_model.logger,
        "warning",
        lambda event, **kwargs: warning_events.append(event),
    )

    log_model.run_model_logging_step(X_dev, y_dev, tuning_result, registration_cfg)

    assert "n_estimators_scaling_regressed" in warning_events
