"""Shared fixtures for integration tests."""

from __future__ import annotations

import pytest

from telco_churn.utils.paths import reset_active_config


@pytest.fixture(autouse=True)
def _reset_active_config() -> None:
    """Clear any installed config between tests.

    Every fixture here that calls activate_config() already resets it in its
    own try/finally around the in-process calls that need it, so this is a
    backstop rather than load-bearing — mirrors tests/unit/conftest.py's
    fixture of the same name.
    """
    reset_active_config()
