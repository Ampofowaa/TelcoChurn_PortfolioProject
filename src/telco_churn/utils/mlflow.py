"""Shared MLflow tracking-URI resolution, used by every module that starts or reads an MLflow run."""

from __future__ import annotations

import re

from telco_churn.utils.paths import get_project_root

__all__ = ["resolve_tracking_uri"]

_SQLITE_PREFIX = "sqlite:///"
_WINDOWS_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_absolute_sqlite_path(path: str) -> bool:
    return path.startswith("/") or bool(_WINDOWS_ABS_PATH_RE.match(path))


def resolve_tracking_uri(uri: str) -> str:
    """Resolve relative MLflow tracking URIs to absolute project-rooted locations.

    Any URI with an explicit scheme (http(s)://, postgresql://, ...) is returned
    unchanged, with one exception: 'sqlite:///<relative-path>' contains '://'
    like any other scheme, but the path after the prefix is CWD-relative, not
    scheme-anchored — the exact failure mode this function exists to prevent for
    a bare 'mlruns' path, re-introduced through a different scheme. A relative
    sqlite path is anchored to get_project_root() and re-assembled as
    'sqlite:///<absolute-posix-path>'; an already-absolute sqlite path (leading
    '/', or a Windows drive like 'C:/...') is left unchanged.

    A bare relative path with no scheme at all (e.g. 'mlruns') is anchored to
    get_project_root() and returned as a file:// URI, not a bare OS path — on
    Windows, str(path) yields 'C:\\...\\mlruns', and MLflow's store registry
    reads urlparse's scheme off that string, which is 'c' for a drive letter,
    not a recognized backend.
    """
    if uri.startswith(_SQLITE_PREFIX):
        db_path = uri[len(_SQLITE_PREFIX) :]
        if _is_absolute_sqlite_path(db_path):
            return uri
        return _SQLITE_PREFIX + (get_project_root() / db_path).as_posix()
    if "://" in uri:
        return uri
    return (get_project_root() / uri).as_uri()
