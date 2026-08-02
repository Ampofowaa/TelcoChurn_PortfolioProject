"""Project-root resolution and config-loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["get_project_root", "load_config", "compose_config"]


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
    """Load and return the root Hydra config from configs/config.yaml."""
    from omegaconf import OmegaConf

    return OmegaConf.load(get_project_root() / "configs" / "config.yaml")


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
