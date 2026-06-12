"""Project-root resolution utility."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root by searching upward for pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Project root not found — no pyproject.toml in any parent directory."
    )
