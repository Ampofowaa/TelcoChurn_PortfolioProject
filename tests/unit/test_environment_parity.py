"""Unit tests for telco_churn.models.environment_parity.

diff_environment's own comparison core has no prior direct test — every
existing register.py test monkeypatches check_environment_parity out
entirely. mlflow.pyfunc.get_model_dependencies is monkeypatched to point at
a small fixture requirements.txt rather than spinning up a real MLflow model
just to exercise the regex/version-comparison logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from telco_churn.models import environment_parity
from telco_churn.models.environment_parity import diff_environment


@pytest.fixture
def requirements_file(tmp_path: Path) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text(
        "numpy==1.26.4\n"
        "scikit-learn==1.5.0\n"
        "lightgbm==4.3.0 ; python_version >= '3.9'\n",
        encoding="utf-8",
    )
    return path


def _patch_get_model_dependencies(
    monkeypatch: pytest.MonkeyPatch, requirements_file: Path
) -> None:
    monkeypatch.setattr(
        environment_parity.mlflow.pyfunc,
        "get_model_dependencies",
        lambda model_uri, format: str(requirements_file),
    )


def test_diff_environment_all_matched(
    monkeypatch: pytest.MonkeyPatch, requirements_file: Path
) -> None:
    _patch_get_model_dependencies(monkeypatch, requirements_file)
    monkeypatch.setattr(
        environment_parity.importlib_metadata,
        "version",
        lambda package: {"numpy": "1.26.4", "scikit-learn": "1.5.0"}[package],
    )

    diff = diff_environment("models:/x/1", ["numpy", "scikit-learn"])

    assert diff.matched == {"numpy": "1.26.4", "scikit-learn": "1.5.0"}
    assert diff.mismatches == {}
    assert diff.unchecked == []


def test_diff_environment_reports_a_version_mismatch(
    monkeypatch: pytest.MonkeyPatch, requirements_file: Path
) -> None:
    _patch_get_model_dependencies(monkeypatch, requirements_file)
    monkeypatch.setattr(
        environment_parity.importlib_metadata,
        "version",
        lambda package: {"numpy": "2.0.0", "scikit-learn": "1.5.0"}[package],
    )

    diff = diff_environment("models:/x/1", ["numpy", "scikit-learn"])

    assert diff.matched == {"scikit-learn": "1.5.0"}
    assert diff.mismatches == {"numpy": ("1.26.4", "2.0.0")}
    assert diff.unchecked == []


def test_diff_environment_reports_an_unpinned_package_as_unchecked_not_a_mismatch(
    monkeypatch: pytest.MonkeyPatch, requirements_file: Path
) -> None:
    """A package with no exact `pkg==version` pin in the logged requirements
    is not checkable — reported as unchecked, never silently counted as
    matched or raised as a mismatch."""
    _patch_get_model_dependencies(monkeypatch, requirements_file)
    monkeypatch.setattr(
        environment_parity.importlib_metadata,
        "version",
        lambda package: "1.26.4",
    )

    diff = diff_environment("models:/x/1", ["numpy", "pandas"])

    assert diff.matched == {"numpy": "1.26.4"}
    assert diff.mismatches == {}
    assert diff.unchecked == ["pandas"]


def test_diff_environment_ignores_the_semicolon_marker_suffix(
    monkeypatch: pytest.MonkeyPatch, requirements_file: Path
) -> None:
    """lightgbm==4.3.0 ; python_version >= '3.9' still matches on the
    pinned version alone — the environment marker is not part of the pin."""
    _patch_get_model_dependencies(monkeypatch, requirements_file)
    monkeypatch.setattr(
        environment_parity.importlib_metadata, "version", lambda package: "4.3.0"
    )

    diff = diff_environment("models:/x/1", ["lightgbm"])

    assert diff.matched == {"lightgbm": "4.3.0"}
