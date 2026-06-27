"""Project-root resolution and config-loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["get_project_root", "load_config"]


def get_project_root() -> Path:
    """Return the project root by searching upward for pyproject.toml."""
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
