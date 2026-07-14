"""Shared MLflow tracking-URI resolution, used by every module that starts or reads an MLflow run."""

from __future__ import annotations

from telco_churn.utils.paths import get_project_root

__all__ = ["resolve_tracking_uri"]


def resolve_tracking_uri(uri: str) -> str:
    """Resolve relative MLflow tracking URIs to absolute project-rooted file:// URIs.

    Any URI with an explicit scheme (http(s)://, sqlite://, postgresql://, ...) is
    returned unchanged — only a bare relative path like 'mlruns' needs anchoring to
    get_project_root() so the tracking store is always written to the same
    location regardless of the shell's working directory.

    Returns a file:// URI, not a bare path — on Windows, str(path) yields
    'C:\\...\\mlruns', and MLflow's store registry reads urlparse's scheme off
    that string, which is 'c' for a drive letter, not a recognized backend. This
    is the default branch for any fresh clone or CI runner (no .env, so
    tracking_uri resolves to the bare 'mlruns' default in configs/config.yaml),
    so it isn't a latent edge case.
    """
    if "://" in uri:
        return uri
    return (get_project_root() / uri).as_uri()
