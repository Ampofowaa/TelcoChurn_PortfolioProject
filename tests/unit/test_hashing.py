"""Unit tests for telco_churn.utils.hashing."""

from __future__ import annotations

from telco_churn.utils.hashing import content_hash


def test_content_hash_deterministic_for_same_payload() -> None:
    """The same payload hashes identically across calls."""
    payload = {"pr_auc": 0.62, "recall": 0.70, "brier": 0.15}
    assert content_hash(payload) == content_hash(payload)


def test_content_hash_ignores_key_order() -> None:
    """Sorted-keys encoding means key order never changes the hash."""
    a = {"pr_auc": 0.62, "recall": 0.70}
    b = {"recall": 0.70, "pr_auc": 0.62}
    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_with_a_value_change() -> None:
    """A genuine value difference produces a different hash."""
    a = {"pr_auc": 0.62}
    b = {"pr_auc": 0.63}
    assert content_hash(a) != content_hash(b)


def test_content_hash_is_a_hex_sha256_digest() -> None:
    """Returns a 64-character lowercase hex string — a sha256 digest."""
    digest = content_hash({"pr_auc": 0.62})
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
