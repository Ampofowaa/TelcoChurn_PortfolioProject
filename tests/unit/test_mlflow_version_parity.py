"""The MLflow server image and the project's MLflow clients must be one version.

The reviewer-facing MLflow server (docker/mlflow/Dockerfile) shares the
`mlflow` tracking database with the api, ui and training clients (uv.lock). A
newer server auto-migrates that database on its first real data request, and
every older client then refuses it — the 2026-10-08 outage. These tests make a
one-sided bump (such as a Dependabot image PR) fail CI instead of production.
"""

from __future__ import annotations

import re
import tomllib

from telco_churn.utils.paths import get_project_root

_DOCKERFILE = get_project_root() / "docker" / "mlflow" / "Dockerfile"
_UV_LOCK = get_project_root() / "uv.lock"


def _locked_mlflow_version() -> str:
    """Return the mlflow version uv.lock resolves for the project's clients."""
    lock = tomllib.loads(_UV_LOCK.read_text(encoding="utf-8"))
    return next(p["version"] for p in lock["package"] if p["name"] == "mlflow")


def test_mlflow_server_image_matches_the_locked_client_version() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    image_tag = re.search(r"^FROM ghcr\.io/mlflow/mlflow:v(\S+)", dockerfile, re.M)

    assert image_tag is not None, "docker/mlflow/Dockerfile has no mlflow base image"
    assert image_tag.group(1) == _locked_mlflow_version()


def test_mlflow_auth_extra_pin_matches_the_locked_client_version() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    auth_pin = re.search(r'"mlflow\[auth\]==([^"]+)"', dockerfile)

    assert auth_pin is not None, "docker/mlflow/Dockerfile no longer pins mlflow[auth]"
    assert auth_pin.group(1) == _locked_mlflow_version()
