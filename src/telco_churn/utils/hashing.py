"""Generic content-addressing helper — stdlib only, no telco_churn imports.

Distinct from models/policy_config.py's costs_config_hash: that one hashes a
specific on-disk YAML config, this one hashes an arbitrary JSON-serialisable
payload (evaluate.py's metrics.json, for register.py to detect a
promotion_decision.json belonging to a different metrics.json than the one
it was computed from).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["content_hash"]


def content_hash(payload: dict[str, Any]) -> str:
    """sha256 of `payload`'s JSON-sorted-keys encoding.

    Embedded in promotion_decision.json so register.py can detect a decision
    belonging to a different metrics.json than the one it was computed from —
    same idiom as models/policy_config.py's costs_config_hash, applied to a
    metrics payload instead of a config file. Public: register.py imports
    this same function to recompute the hash it verifies against, rather
    than reimplementing the encoding.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
