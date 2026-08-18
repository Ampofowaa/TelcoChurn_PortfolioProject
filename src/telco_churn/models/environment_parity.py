"""Compare installed package versions against a registered model's own logged pip requirements.

Shared by register.py's mint-time/promotion-time environment-parity check and
serving/predict.py's hot-reload dependency-compatibility diff — both need the
identical package-version comparison; they differ only in what they do with a
mismatch (register.py raises and blocks promotion, predict.py logs a warning
and refuses the reload while continuing to serve the last-known-good model).
Kept outside register.py (a __main__-bearing module) so predict.py can reuse
it without violating CLAUDE.md's no-cross-__main__-import rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata

import mlflow.pyfunc

from telco_churn.utils.logging import get_logger

__all__ = ["EnvironmentDiff", "EnvironmentMismatchError", "diff_environment"]

logger = get_logger(__name__)


class EnvironmentMismatchError(RuntimeError):
    """The checking environment disagrees with the one the model was logged under.

    Distinct from a smoke-check failure: this means the *verification*
    cannot be trusted, not that the *model* is broken. Callers must not tag
    promotion_status on this exception — the version stays exactly where it
    was (pending, if this fires during the initial smoke check), on the same
    "no verdict was reached, so don't fabricate one" logic that makes
    pending the crash-safe mint-time default in the first place.
    """


@dataclass(frozen=True)
class EnvironmentDiff:
    """Per-package comparison result: matched versions, mismatches, and unpinned packages.

    `mismatches` maps package -> (logged, installed); `unchecked` lists
    packages with no exact `package==version` pin found in the registered
    model's logged requirements. Never raises by construction — the caller
    decides what a mismatch means (register.py's check_environment_parity
    raises on any; predict.py's hot-reload guard logs and refuses instead).
    """

    matched: dict[str, str] = field(default_factory=dict)
    mismatches: dict[str, tuple[str, str]] = field(default_factory=dict)
    unchecked: list[str] = field(default_factory=list)


def diff_environment(model_uri: str, packages: list[str]) -> EnvironmentDiff:
    """Compare installed `packages` against `model_uri`'s own logged pip requirements.

    The golden-parity tolerance (cfg.register.golden_atol) is only
    meaningful if these haven't drifted from what calibrate.py logged the
    model under — a numpy/scikit-learn/lightgbm patch release shifts
    floating-point output in the same 1e-7-1e-9 range as that tolerance,
    with no actual bug present.
    """
    requirements_path = mlflow.pyfunc.get_model_dependencies(model_uri, format="pip")
    with open(requirements_path, encoding="utf-8") as f:
        requirements_text = f.read()

    matched: dict[str, str] = {}
    mismatches: dict[str, tuple[str, str]] = {}
    unchecked: list[str] = []
    for package in packages:
        match = re.search(
            rf"^{re.escape(package)}==([^\s;]+)",
            requirements_text,
            re.MULTILINE | re.IGNORECASE,
        )
        if match is None:
            logger.warning(
                "environment_parity_pin_not_found",
                package=package,
                hint=(
                    "no exact `package==version` pin found in the registered "
                    "model's logged requirements — this package was not "
                    "checked for environment drift."
                ),
            )
            unchecked.append(package)
            continue
        pinned = match.group(1)
        installed = importlib_metadata.version(package)
        if pinned == installed:
            matched[package] = installed
        else:
            mismatches[package] = (pinned, installed)

    return EnvironmentDiff(matched=matched, mismatches=mismatches, unchecked=unchecked)
