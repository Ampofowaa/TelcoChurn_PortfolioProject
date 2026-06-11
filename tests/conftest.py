"""Root conftest: opt-in flag for integration tests that require Docker."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --run-integration flag."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests — testcontainers tests are self-contained; Compose-backed tests additionally require `make db-up`.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip integration-marked tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(
        reason="integration test skipped — pass --run-integration to run"
    )
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
