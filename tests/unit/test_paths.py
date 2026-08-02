"""Unit tests for src/telco_churn/utils/paths.py.

get_project_root() is load-bearing across all of src/ (CLAUDE.md: "Never use
bare relative paths for file I/O in src/ ... Always anchor to
get_project_root()"), so its own contract — and load_config()/compose_config()'s
on top of it — is tested directly here rather than only incidentally through
whichever other module happens to import it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from telco_churn.utils.paths import compose_config, get_project_root, load_config

# ---------------------------------------------------------------------------
# get_project_root
# ---------------------------------------------------------------------------


def test_get_project_root_finds_directory_containing_pyproject_toml() -> None:
    root = get_project_root()
    assert (root / "pyproject.toml").exists()


def test_get_project_root_is_consistent_across_calls() -> None:
    """Repeated calls resolve to the same directory — every src/ module
    anchoring file I/O to it depends on that being stable within a process."""
    assert get_project_root() == get_project_root()


def test_get_project_root_raises_when_no_pyproject_toml_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every parent-directory lookup missing pyproject.toml must raise
    FileNotFoundError rather than silently returning the wrong root, which
    would corrupt every downstream file I/O anchored to it."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(FileNotFoundError, match="pyproject.toml"):
        get_project_root()


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_reads_top_level_config_yaml_values() -> None:
    cfg = load_config()
    assert cfg.random_seed == 42
    assert cfg.mlflow.registered_model_name == "telco-churn-pipeline"
    assert cfg.paths.model_promotion_config == "configs/model_promotion.yaml"


def test_load_config_does_not_merge_the_defaults_list() -> None:
    """load_config() reads only configs/config.yaml — the defaults list
    (training/tuning/calibration/...) is Hydra composition, which
    compose_config() performs and this function deliberately does not."""
    cfg = load_config()
    assert "training" not in cfg
    assert "tuning" not in cfg


# ---------------------------------------------------------------------------
# compose_config
# ---------------------------------------------------------------------------


def test_compose_config_merges_defaults_populating_training_and_tuning() -> None:
    """The whole reason to use compose_config() over load_config(): cfg.training/
    cfg.tuning/cfg.logreg/cfg.selection come from the defaults list, not
    config.yaml itself."""
    cfg = compose_config()
    assert cfg.training.fixed.n_jobs == 1
    assert cfg.tuning.n_trials == 50
    assert cfg.tuning.random_state == 42
    assert cfg.logreg.cv_folds == 5


def test_compose_config_also_carries_config_yaml_s_own_keys() -> None:
    """_self_ in the defaults list means config.yaml's own top-level keys
    (not just the merged defaults) must still be present after composition."""
    cfg = compose_config()
    assert cfg.random_seed == 42
    assert cfg.mlflow.registered_model_name == "telco-churn-pipeline"


def test_compose_config_applies_overrides() -> None:
    cfg = compose_config(overrides=["random_seed=7"])
    assert cfg.random_seed == 7


def test_compose_config_safe_to_call_repeatedly() -> None:
    """Docstring's claim: clears any existing GlobalHydra instance first, so
    calling it a second time in the same process (e.g. from two different
    tests, or a script that composes more than once) must not raise."""
    first = compose_config()
    second = compose_config()
    assert first.random_seed == second.random_seed == 42
