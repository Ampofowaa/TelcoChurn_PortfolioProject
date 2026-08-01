"""Unit tests for telco_churn.models.train.tuning — Step 4 (C1)."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import mlflow
import optuna
import pandas as pd
import pytest
from omegaconf import OmegaConf

import telco_churn.models.train.tuning as tuning

# ---------------------------------------------------------------------------
# select_best_trial
# ---------------------------------------------------------------------------


def test_select_best_trial_argmax_ignores_complexity() -> None:
    """'argmax' always returns the raw-highest-value trial."""
    trials = [
        {
            "number": 0,
            "value": 0.60,
            "fold_scores": [0.60] * 5,
            "params": {"num_leaves": 20},
        },
        {
            "number": 1,
            "value": 0.65,
            "fold_scores": [0.65] * 5,
            "params": {"num_leaves": 150},
        },
    ]
    result = tuning.select_best_trial(trials, "argmax")
    assert result["number"] == 1


def test_select_best_trial_1se_prefers_simpler_within_band() -> None:
    """'1se' picks the fewest-num_leaves trial among those within 1 SE of the best."""
    trials = [
        {
            "number": 0,
            "value": 0.649,
            "fold_scores": [0.64, 0.65, 0.66, 0.65, 0.649],
            "n_estimators_median": 100,
            "params": {"num_leaves": 30},
        },
        {
            "number": 1,
            "value": 0.650,
            "fold_scores": [0.63, 0.66, 0.67, 0.64, 0.65],
            "n_estimators_median": 100,
            "params": {"num_leaves": 180},
        },
    ]
    result = tuning.select_best_trial(trials, "1se")
    assert result["number"] == 0


def test_select_best_trial_1se_falls_back_to_best_when_alone_in_band() -> None:
    """A best trial far ahead of everything else (outside their SE bands) wins outright."""
    trials = [
        {
            "number": 0,
            "value": 0.40,
            "fold_scores": [0.40] * 5,
            "n_estimators_median": 100,
            "params": {"num_leaves": 20},
        },
        {
            "number": 1,
            "value": 0.90,
            "fold_scores": [0.90] * 5,
            "n_estimators_median": 100,
            "params": {"num_leaves": 150},
        },
    ]
    result = tuning.select_best_trial(trials, "1se")
    assert result["number"] == 1


def test_select_best_trial_1se_tiebreaks_on_n_estimators_median() -> None:
    """When num_leaves ties within the band, fewer n_estimators_median wins."""
    trials = [
        {
            "number": 0,
            "value": 0.649,
            "fold_scores": [0.649] * 5,
            "n_estimators_median": 60,
            "params": {"num_leaves": 50},
        },
        {
            "number": 1,
            "value": 0.650,
            "fold_scores": [0.63, 0.66, 0.67, 0.64, 0.65],
            "n_estimators_median": 200,
            "params": {"num_leaves": 50},
        },
    ]
    result = tuning.select_best_trial(trials, "1se")
    assert result["number"] == 0


def test_select_best_trial_single_trial_returns_it_for_both_rules() -> None:
    """A single-trial study returns that trial regardless of selection_rule."""
    trials = [
        {
            "number": 0,
            "value": 0.5,
            "fold_scores": [0.5, 0.5],
            "n_estimators_median": 100,
            "params": {"num_leaves": 40},
        }
    ]
    assert tuning.select_best_trial(trials, "argmax")["number"] == 0
    assert tuning.select_best_trial(trials, "1se")["number"] == 0


def test_select_best_trial_empty_raises() -> None:
    """An empty trial list raises rather than silently returning nothing."""
    with pytest.raises(ValueError, match="must not be empty"):
        tuning.select_best_trial([], "argmax")


# ---------------------------------------------------------------------------
# _raw_best_diagnostics
# ---------------------------------------------------------------------------


def test_raw_best_diagnostics_identifies_highest_value_trial() -> None:
    """raw_best_* fields point at the top-scoring trial, not the (possibly different) adopted one."""
    trials = [
        {
            "number": 0,
            "value": 0.60,
            "fold_scores": [0.58, 0.60, 0.62, 0.59, 0.61],
            "n_estimators_median": 100,
            "params": {"num_leaves": 20},
        },
        {
            "number": 1,
            "value": 0.65,
            "fold_scores": [0.63, 0.66, 0.67, 0.64, 0.65],
            "n_estimators_median": 100,
            "params": {"num_leaves": 150},
        },
    ]
    diagnostics = tuning._raw_best_diagnostics(trials)
    assert diagnostics["raw_best_trial_number"] == 1
    assert diagnostics["raw_best_cv_pr_auc"] == pytest.approx(0.65)


def test_raw_best_diagnostics_band_floor_matches_select_best_trial() -> None:
    """band_floor here is exactly the threshold select_best_trial's 1-SE branch uses internally."""
    trials = [
        {
            "number": 0,
            "value": 0.649,
            "fold_scores": [0.64, 0.65, 0.66, 0.65, 0.649],
            "n_estimators_median": 100,
            "params": {"num_leaves": 30},
        },
        {
            "number": 1,
            "value": 0.650,
            "fold_scores": [0.63, 0.66, 0.67, 0.64, 0.65],
            "n_estimators_median": 100,
            "params": {"num_leaves": 180},
        },
    ]
    diagnostics = tuning._raw_best_diagnostics(trials)
    selected = tuning.select_best_trial(trials, "1se")
    assert diagnostics["raw_best_trial_number"] == 1
    assert selected["value"] >= diagnostics["band_floor"]


def test_raw_best_diagnostics_single_trial_has_zero_se() -> None:
    """A single completed trial has SE 0 and its own value as the band floor."""
    trials = [
        {
            "number": 0,
            "value": 0.5,
            "fold_scores": [0.5, 0.5],
            "n_estimators_median": 100,
            "params": {"num_leaves": 40},
        }
    ]
    diagnostics = tuning._raw_best_diagnostics(trials)
    assert diagnostics["se"] == 0.0
    assert diagnostics["band_floor"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# boundary_hit_check (C1)
# ---------------------------------------------------------------------------


def test_boundary_hit_check_flags_low_and_high_edges() -> None:
    """A param sitting exactly on its low or high bound is flagged."""
    params = {"num_leaves": 20, "reg_lambda": 10.0, "learning_rate": 0.05}
    search_space = {
        "num_leaves": {"low": 20, "high": 200},
        "reg_lambda": {"low": 0.0, "high": 10.0},
        "learning_rate": {"low": 0.005, "high": 0.1},
    }
    hits = tuning.boundary_hit_check(params, search_space)
    assert hits["num_leaves"] is True
    assert hits["reg_lambda"] is True
    assert hits["learning_rate"] is False


def test_boundary_hit_check_ignores_params_outside_search_space() -> None:
    """A param not present in the search space is silently skipped, not flagged."""
    params = {"class_weight_ratio": 2.77}
    search_space = {"num_leaves": {"low": 20, "high": 200}}
    hits = tuning.boundary_hit_check(params, search_space)
    assert hits == {}


def test_boundary_hit_check_interior_values_not_flagged() -> None:
    """A value strictly inside its range is not flagged."""
    params = {"num_leaves": 100}
    search_space = {"num_leaves": {"low": 20, "high": 200}}
    hits = tuning.boundary_hit_check(params, search_space)
    assert hits["num_leaves"] is False


# ---------------------------------------------------------------------------
# run_tuning_step (C1) — end-to-end smoke test on a tiny synthetic study
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tuning_mlflow_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Point MLflow at a module-scoped tmp SQLite store with an explicit
    artifact_location — inlines conftest.py::mlflow_test_experiment's logic
    rather than requesting it, since that fixture depends on the
    function-scoped tmp_path and this fixture is module-scoped: every test
    below only logs a fresh run_tuning_step run into the shared experiment
    and inspects that run/its own mocks, never enumerates the experiment's
    full run list, so sharing one experiment (and skipping the ~2.5-3.5s
    SQLite-store bootstrap cost per test) is safe. tuning_cfg stays
    function-scoped deliberately — several tests mutate it per-scenario, so
    it can't be shared."""
    tmp_path = tmp_path_factory.mktemp("tuning_mlflow")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    experiment_id = mlflow.create_experiment(
        "test_run_tuning_step", artifact_location=artifact_location
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


@pytest.fixture
def tuning_cfg() -> OmegaConf:
    """A tiny Optuna study config — 3 trials, 2 folds, low n_estimators ceiling."""
    return OmegaConf.create(
        {
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
                "n_trials": 3,
                "direction": "maximize",
                "cv_folds": 2,
                "random_state": 42,
                "n_estimators_ceiling": 50,
                "early_stopping_rounds": 5,
                "es_validation_size": 0.2,
                "search_space": {
                    "num_leaves": {"low": 10, "high": 30, "type": "int"},
                    "learning_rate": {
                        "low": 0.01,
                        "high": 0.2,
                        "log": True,
                        "type": "float",
                    },
                    "min_child_samples": {"low": 5, "high": 20, "type": "int"},
                    "subsample": {"low": 0.7, "high": 1.0, "type": "float"},
                    "colsample_bytree": {"low": 0.7, "high": 1.0, "type": "float"},
                    "reg_alpha": {"low": 0.0, "high": 1.0, "type": "float"},
                    "reg_lambda": {"low": 0.0, "high": 1.0, "type": "float"},
                    "max_depth": {"low": 4, "high": 8, "type": "int"},
                },
                "sampler_seed": 42,
                "n_startup_trials": 1,
                "pruner": "none",
                "pruner_n_warmup_steps": 0,
                "selection_rule": "1se",
            },
            "mlflow": {
                "tracking_uri": "placeholder",
                "experiment_name": "test_run_tuning_step",
            },
        }
    )


def test_run_tuning_step_returns_expected_keys(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """run_tuning_step completes a tiny study end-to-end and returns the documented keys."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    result = tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    assert set(result) == {
        "best_params",
        "best_n_estimators_median",
        "best_cv_pr_auc_mean",
        "boundary_hits",
        "n_completed_trials",
        "parent_run_id",
        "committed_features",
        "tuning_summary",
    }
    assert result["n_completed_trials"] >= 1
    assert 0.0 <= result["best_cv_pr_auc_mean"] <= 1.0
    assert set(result["best_params"]) == set(tuning_cfg.tuning.search_space)

    tuning_summary = result["tuning_summary"]
    assert tuning_summary["n_trials_requested"] == tuning_cfg.tuning.n_trials
    assert tuning_summary["n_completed_trials"] == result["n_completed_trials"]
    assert tuning_summary["selected_cv_pr_auc"] == result["best_cv_pr_auc_mean"]
    assert tuning_summary["raw_best_cv_pr_auc"] >= tuning_summary["selected_cv_pr_auc"]
    assert tuning_summary["se"] >= 0.0
    # base tuning_cfg has no min_completed_trials configured — the unset case
    # must default to "not below threshold", not raise or silently flag true.
    assert tuning_summary["min_completed_trials"] is None
    assert tuning_summary["trial_count_below_threshold"] is False
    assert tuning_summary["band_floor"] == pytest.approx(
        tuning_summary["raw_best_cv_pr_auc"] - tuning_summary["se"]
    )


def test_run_tuning_step_enqueues_warm_start_params(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """cfg.tuning.warm_start_params is queued via study.enqueue_trial before optimize."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    warm_start = {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }
    tuning_cfg.tuning.warm_start_params = warm_start
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    captured: dict[str, Any] = {}
    original_enqueue = optuna.study.Study.enqueue_trial

    def spy_enqueue(self: optuna.Study, params: Any, *args: Any, **kwargs: Any) -> Any:
        captured["params"] = dict(params)
        captured["kwargs"] = kwargs
        return original_enqueue(self, params, *args, **kwargs)

    monkeypatch.setattr(optuna.study.Study, "enqueue_trial", spy_enqueue)

    tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    assert captured["params"] == warm_start
    assert captured["kwargs"].get("skip_if_exists") is True


def test_run_tuning_step_logs_trial_history_as_mlflow_table(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """Trial history is logged via mlflow.log_table (queryable), not a flat CSV blob."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    log_table_mock = Mock()
    monkeypatch.setattr(tuning.mlflow, "log_table", log_table_mock)

    tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    log_table_mock.assert_called_once()
    data, artifact_file = log_table_mock.call_args[0]
    assert artifact_file == "tuning/trials.json"
    assert "fold_scores" not in data.columns


def test_run_tuning_step_skips_enqueue_without_warm_start_params(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """Without warm_start_params in config, no trial is enqueued (pure random/TPE warm-up)."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    enqueue_mock = Mock(wraps=optuna.study.Study.enqueue_trial)
    monkeypatch.setattr(optuna.study.Study, "enqueue_trial", enqueue_mock)

    tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    enqueue_mock.assert_not_called()


def test_run_tuning_step_warns_on_boundary_hit(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """A boundary hit on the selected trial escalates via logger.warning, not just info."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    monkeypatch.setattr(
        tuning,
        "boundary_hit_check",
        lambda params, search_space: {"num_leaves": True, "learning_rate": False},
    )
    warning_mock = Mock()
    monkeypatch.setattr(tuning.logger, "warning", warning_mock)

    tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    boundary_calls = [
        call
        for call in warning_mock.call_args_list
        if call.args[0] == "tuning_boundary_hit"
    ]
    assert len(boundary_calls) == 1
    assert boundary_calls[0].kwargs["hit_params"] == ["num_leaves"]


def test_run_tuning_step_no_warning_without_boundary_hit(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """No trial param on a search-space edge means no boundary-hit warning fires."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    monkeypatch.setattr(
        tuning,
        "boundary_hit_check",
        lambda params, search_space: {"num_leaves": False, "learning_rate": False},
    )
    warning_mock = Mock()
    monkeypatch.setattr(tuning.logger, "warning", warning_mock)

    tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    boundary_calls = [
        call
        for call in warning_mock.call_args_list
        if call.args[0] == "tuning_boundary_hit"
    ]
    assert boundary_calls == []


def test_run_tuning_step_warns_on_too_few_completed_trials(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """Fewer completed trials than min_completed_trials flags the pick as unreliable."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    tuning_cfg.tuning.min_completed_trials = 10  # fixture only runs 3 trials
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    warning_mock = Mock()
    monkeypatch.setattr(tuning.logger, "warning", warning_mock)

    result = tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    too_few_calls = [
        call
        for call in warning_mock.call_args_list
        if call.args[0] == "tuning_too_few_completed_trials"
    ]
    assert len(too_few_calls) == 1
    assert too_few_calls[0].kwargs["n_completed_trials"] == result["n_completed_trials"]
    assert too_few_calls[0].kwargs["min_completed_trials"] == 10

    tuning_summary = result["tuning_summary"]
    assert tuning_summary["min_completed_trials"] == 10
    assert tuning_summary["trial_count_below_threshold"] is True


def test_run_tuning_step_raises_when_every_trial_fails(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """A systematic bug (bad config, objective typo) fails every trial and must raise
    loudly, not just log a warning that a resumed study or unset min_completed_trials
    could let slip past unnoticed."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    def _always_fails(*args: Any, **kwargs: Any) -> float:
        raise ValueError("simulated systematic objective failure")

    monkeypatch.setattr(tuning, "_tuning_objective", _always_fails)

    with pytest.raises(RuntimeError, match="All 3 trials submitted this run failed"):
        tuning.run_tuning_step(
            X_dev,
            y_dev,
            committed_features,
            tuning_cfg,
            storage=optuna.storages.InMemoryStorage(),
        )


def test_run_tuning_step_no_warning_when_completed_trials_meet_floor(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """A completed-trial count at or above min_completed_trials raises no warning."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    tuning_cfg.tuning.min_completed_trials = 1
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    warning_mock = Mock()
    monkeypatch.setattr(tuning.logger, "warning", warning_mock)

    result = tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    too_few_calls = [
        call
        for call in warning_mock.call_args_list
        if call.args[0] == "tuning_too_few_completed_trials"
    ]
    assert too_few_calls == []

    tuning_summary = result["tuning_summary"]
    assert tuning_summary["min_completed_trials"] == 1
    assert tuning_summary["trial_count_below_threshold"] is False


def test_run_tuning_step_passes_timeout_seconds_to_optimize(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """cfg.tuning.timeout_seconds reaches study.optimize as its wall-clock timeout."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    tuning_cfg.tuning.timeout_seconds = 60
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    captured: dict[str, Any] = {}
    original_optimize = optuna.study.Study.optimize

    def spy_optimize(self: optuna.Study, *args: Any, **kwargs: Any) -> None:
        captured["timeout"] = kwargs.get("timeout")
        return original_optimize(self, *args, **kwargs)

    monkeypatch.setattr(optuna.study.Study, "optimize", spy_optimize)

    tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    assert captured["timeout"] == 60.0


def test_run_tuning_step_defaults_timeout_seconds_to_none(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """Without timeout_seconds in config, study.optimize is left n_trials-only (timeout=None)."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    captured: dict[str, Any] = {}
    original_optimize = optuna.study.Study.optimize

    def spy_optimize(self: optuna.Study, *args: Any, **kwargs: Any) -> None:
        captured["timeout"] = kwargs.get("timeout")
        return original_optimize(self, *args, **kwargs)

    monkeypatch.setattr(optuna.study.Study, "optimize", spy_optimize)

    tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    assert captured["timeout"] is None


def test_run_tuning_step_n_estimators_median_is_positive(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """The selected trial's median early-stopped tree count is a positive integer."""
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    result = tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    assert isinstance(result["best_n_estimators_median"], int)
    assert result["best_n_estimators_median"] > 0


def test_run_tuning_step_is_idempotent_against_completed_study(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """Re-running against a study that already reached n_trials adds no new trials.

    load_if_exists=True resumes an interrupted study by design — but re-running a
    *completed* study should reuse its trials, not pile n_trials more on top.
    """
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)
    storage = optuna.storages.InMemoryStorage()

    first = tuning.run_tuning_step(
        X_dev, y_dev, committed_features, tuning_cfg, storage=storage
    )

    optimize_mock = Mock(wraps=optuna.study.Study.optimize)
    monkeypatch.setattr(optuna.study.Study, "optimize", optimize_mock)

    second = tuning.run_tuning_step(
        X_dev, y_dev, committed_features, tuning_cfg, storage=storage
    )

    optimize_mock.assert_not_called()
    assert (
        second["tuning_summary"]["n_completed_trials"]
        + second["tuning_summary"]["n_pruned_trials"]
        + second["tuning_summary"]["n_failed_trials"]
        == tuning_cfg.tuning.n_trials
    )
    assert second["n_completed_trials"] == first["n_completed_trials"]


def test_run_tuning_step_logs_dev_input_once_on_parent_run(
    monkeypatch: pytest.MonkeyPatch,
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    tuning_cfg: OmegaConf,
) -> None:
    """Fix 6: the tuning_study parent run logs a dataset-lineage input exactly
    once, via common.py's shared _log_dev_input — the same call that also
    covers log_model.py's "model" and calibrate.py's "calibrated_model", since
    both reuse this run's run_id. The source-resolution behaviour itself is
    tested once, at the shared implementation, in test_train_common.py.
    """
    tuning_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    committed_features = list(X_dev.columns)

    log_dev_input_mock = Mock()
    monkeypatch.setattr(tuning, "_log_dev_input", log_dev_input_mock)

    tuning.run_tuning_step(
        X_dev,
        y_dev,
        committed_features,
        tuning_cfg,
        storage=optuna.storages.InMemoryStorage(),
    )

    log_dev_input_mock.assert_called_once_with(X_dev, y_dev, context="training")
