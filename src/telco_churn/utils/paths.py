"""Project-root resolution and config-loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "get_project_root",
    "load_config",
    "compose_config",
    "activate_config",
    "reset_active_config",
]

_active_config: Any | None = None


def get_project_root() -> Path:
    """Return the project root by searching upward for pyproject.toml.

    Anchor point for all file I/O paths in src/ — never build a bare
    relative path, always compose it from this root, so behavior is
    identical regardless of the caller's CWD (DVC stage, Prefect worker,
    CI runner, or Docker container).

    Raises:
        FileNotFoundError: No parent directory contains a pyproject.toml.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Project root not found — no pyproject.toml in any parent directory."
    )


def load_config() -> Any:
    """Return the active installed config if one is set, else load config.yaml fresh.

    An installed config (see activate_config()) always takes precedence — this is
    what lets a process-global override (e.g. a test's paths.processed_data=<tmp>)
    reach accessor modules that call load_config() internally without threading a
    cfg parameter through every call site.
    """
    if _active_config is not None:
        return _active_config

    from omegaconf import OmegaConf

    return OmegaConf.load(get_project_root() / "configs" / "config.yaml")


def activate_config(cfg: Any) -> None:
    """Install cfg as the process-global config returned by load_config().

    Call once per entry point, immediately after compose_config(), so every
    module that resolves its own defaults via load_config() (rather than
    taking cfg as a parameter) sees the same composed, override-applied tree.
    Must be paired with reset_active_config() between in-process test runs —
    see the autouse conftest.py fixture.
    """
    global _active_config
    _active_config = cfg


def reset_active_config() -> None:
    """Clear the installed config, reverting load_config() to a fresh on-disk read."""
    global _active_config
    _active_config = None


def compose_config(overrides: list[str] | None = None) -> Any:
    """Compose the Hydra config tree with all defaults (model, tuning) resolved.

    Use instead of load_config() whenever cfg.training.* or cfg.tuning.* are needed —
    load_config() reads only config.yaml without merging the defaults list.
    Any existing GlobalHydra instance is cleared first so this is safe to call
    multiple times (e.g. from tests or scripts).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = str(get_project_root() / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides or [])
