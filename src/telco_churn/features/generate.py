"""Feature discovery machinery for the error-driven feature discovery loop.

Pure, typed, testable functions — no feature engineering logic lives here, with
one exception: compute_service_count/compute_charge_per_service. Every other
improvised candidate feature is owned directly by the discovery loop
(notebooks/02a-feature-discovery.ipynb), which migrates it to build.py on
adoption. charge_per_service is different — it is the one candidate that was
adopted *and* SQL-engineered (sql/features/charge_per_service.sql), so a second,
independent pandas implementation exists purely so the notebook can construct
it without a database connection. Two hand-written copies of the same formula
with nothing checking they agree is exactly the failure mode this module
otherwise avoids, so the formula lives here once and
test_sql_features_postgres.py asserts it against the real SQL view.

Gate thresholds are module-level constants so every lap is self-describing and
the gate is reproducible from provenance.json alone without a git lookup.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance as sk_perm_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from telco_churn.data.schema import RawSchema
from telco_churn.utils.stats import abs_corr, cramers_v

__all__ = [
    "CORR_THRESHOLD",
    "CRAMERS_V_THRESHOLD",
    "IMPORTANCE_NOISE_FLOOR_MARGIN",
    "MIN_PR_AUC_DELTA",
    "SERVING_COLS",  # noqa: F822 — exported via module __getattr__
    "RedundancyResult",
    "ImportanceResult",
    "AdoptionDecision",
    "LapRecord",
    "LapEvaluation",
    "BackwardEliminationResult",
    "backward_elimination",
    "oof_predictions",
    "profile_false_negatives",
    "subgroup_recall",
    "serving_available",
    "redundancy_screen",
    "candidate_importance",
    "adoption_gate",
    "run_lap",
    "bootstrap_pr_auc_ci",
    "write_provenance",
    "compute_service_count",
    "compute_charge_per_service",
]

# ---------------------------------------------------------------------------
# Gate thresholds — surfaced verbatim in provenance.json
# ---------------------------------------------------------------------------

CORR_THRESHOLD: float = 0.85
CRAMERS_V_THRESHOLD: float = 0.70
IMPORTANCE_NOISE_FLOOR_MARGIN: float = 0.005
MIN_PR_AUC_DELTA: float = 0.0015

# ---------------------------------------------------------------------------
# charge_per_service — the one candidate with a shipped SQL twin (see module
# docstring). Kept together so the two lines summed below and
# sql/features/charge_per_service.sql's CASE conditions are edited as one unit.
# ---------------------------------------------------------------------------


def compute_service_count(df: pd.DataFrame) -> pd.Series:
    """Active service count per customer — nine binary flags summed.

    Mirrors sql/features/charge_per_service.sql's inner CASE-based count exactly:
    phoneservice and multiplelines are separate line items (both contribute 1),
    internetservice uses "!= 'No'" so DSL and Fiber optic both count.
    """
    return (
        (df["phoneservice"] == "Yes").astype(int)
        + (df["multiplelines"] == "Yes").astype(int)
        + (df["internetservice"] != "No").astype(int)
        + (df["onlinesecurity"] == "Yes").astype(int)
        + (df["onlinebackup"] == "Yes").astype(int)
        + (df["deviceprotection"] == "Yes").astype(int)
        + (df["techsupport"] == "Yes").astype(int)
        + (df["streamingtv"] == "Yes").astype(int)
        + (df["streamingmovies"] == "Yes").astype(int)
    ).rename("service_count")


def compute_charge_per_service(df: pd.DataFrame) -> pd.Series:
    """monthlycharges divided by active service count, floored at 1.

    Pandas mirror of sql/features/charge_per_service.sql — same GREATEST(service_count, 1)
    divide-by-zero guard, expressed as .clip(lower=1). Parity with the SQL view is
    asserted in test_sql_features_postgres.py against a seeded fixture; edit both together.
    """
    service_count = compute_service_count(df)
    return (df["monthlycharges"] / service_count.clip(lower=1)).rename(
        "charge_per_service"
    )


# ---------------------------------------------------------------------------
# Serving-time-available base columns (raw IBM set after ingest cleaning)
# Derived from RawSchema — single source of truth; no manual sync required.
# customerid and churn are excluded: ID and label, not features.
# Computed lazily on first access to avoid unconditional Pandera cost at import
# time for callers that never use serving_available.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_serving_cols() -> frozenset[str]:
    """Return the frozenset of serving-available column names, cached after first call."""
    return frozenset(
        set(RawSchema.to_schema().columns.keys()) - {"customerid", "churn"}
    )


def __getattr__(name: str) -> object:
    """Lazily resolve SERVING_COLS on first access and cache it in module globals."""
    if name == "SERVING_COLS":
        val = _get_serving_cols()
        globals()["SERVING_COLS"] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RedundancyResult:
    """Structural redundancy check result — no model re-fit required."""

    flagged: bool
    max_corr: float | None
    max_cramers_v: float | None
    flag_reason: str | None

    @property
    def summary(self) -> str:
        """One-line Screen 2 display string showing the numeric value on PASS and FAIL."""
        if self.flagged:
            return f"FAIL  reason={self.flag_reason}"
        if self.max_corr is not None:
            return f"PASS  max_corr={self.max_corr:.3f} (threshold={CORR_THRESHOLD})"
        if self.max_cramers_v is not None:
            return f"PASS  max_v={self.max_cramers_v:.3f} (threshold={CRAMERS_V_THRESHOLD})"
        if self.flag_reason is not None:
            return f"PASS  ({self.flag_reason})"
        return "PASS  (no adopted set)"


@dataclasses.dataclass(frozen=True)
class ImportanceResult:
    """Permutation importance of the candidate versus the decoy noise floor."""

    candidate_importance: float
    noise_floor: float
    above_floor: bool


@dataclasses.dataclass(frozen=True)
class AdoptionDecision:
    """Four-screen gate outcome for a single candidate."""

    adopted: bool
    rejection_screen: int | None
    rejection_reason: str | None


@dataclasses.dataclass(frozen=True)
class LapEvaluation:
    """Bundled four-screen gate outcome for one discovery-loop candidate, from run_lap.

    Replaces the ~10 loose per-lap locals (oof_<suffix>, pr_auc_<suffix>, decision_<suffix>,
    ...) the notebook used to carry out of each hand-written Evaluate cell. The Record cell
    reads LapRecord fields off this object; the post-decision state update reads
    X_with_candidate/oof_proba/pr_auc/ci instead of a per-lap X_<suffix>/oof_<suffix> pair.
    """

    candidate_col: str
    X_with_candidate: pd.DataFrame
    serving_result: tuple[bool, str]
    redundancy_result: RedundancyResult | None
    oof_proba: npt.NDArray[np.float64]
    pr_auc: float
    ci: tuple[float, float]
    sub_recall_delta: float | None
    importance_result: ImportanceResult | None
    decision: AdoptionDecision


@dataclasses.dataclass(frozen=True)
class LapRecord:
    """Full per-lap provenance record written to provenance.json."""

    lap: int
    blind_spot: str
    candidate: str
    hypothesis: str
    eda_anchor: str
    max_corr: float | None
    max_cramers_v: float | None
    prior_pr_auc: float
    new_pr_auc: float
    delta_pr_auc: float
    prior_ci_low: float
    prior_ci_high: float
    sub_recall_delta: float | None
    candidate_importance_score: float
    noise_floor: float
    decision: str
    rejection_screen: int | None
    rejection_reason: str | None


@dataclasses.dataclass(frozen=True)
class BackwardEliminationResult:
    """Per-feature result of the backward elimination pass."""

    feature: str
    pr_auc_without: float
    delta: float  # full_pr_auc - pr_auc_without; positive means feature contributed
    keep: bool  # True when delta >= min_pr_auc_delta


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    n_folds: int = 5,
    random_state: int = 42,
    model: Any = None,
) -> npt.NDArray[np.float64]:
    """Return out-of-fold predicted probabilities for the positive class.

    Uses cross_val_predict with StratifiedKFold so every training row is scored
    by a fold-model that never saw it — the leak-free OOF substrate required by
    the discovery loop. cross_val_predict clones the pipeline per fold so the
    original preprocessor is never mutated between laps.

    model — sklearn-compatible classifier; defaults to LGBMClassifier with
            class_weight="balanced" to respect the ~26.5 % churn minority class.
            Tests pass DecisionTreeClassifier here — it is scale-invariant so it
            works with the unscaled numeric branch of build_preprocessor (no
            StandardScaler), detects planted signals in synthetic data, and is
            fast. LogisticRegression is NOT suitable: without scaling it converges
            poorly on mixed-scale features (tenure 0-72, totalcharges 0-8684) and
            produces near-random probabilities that make the test meaningless.
    """
    if model is None:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            class_weight="balanced", random_state=random_state, verbose=-1
        )

    pipe = Pipeline(
        steps=[
            ("pre", clone(preprocessor)),
            ("clf", clone(model)),
        ]
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    proba: npt.NDArray[np.float64] = cross_val_predict(
        pipe, X, y, cv=cv, method="predict_proba"
    )[:, 1]
    return proba


def profile_false_negatives(
    X: pd.DataFrame,
    y: pd.Series,
    oof_proba: npt.NDArray[np.float64],
    threshold: float = 0.5,
    n_bins: int = 4,
) -> pd.DataFrame:
    """Scan each feature column for subgroups with elevated false-negative rates.

    Returns a DataFrame ranked by fn_rate descending with columns:
    feature, subgroup, fn_rate, fn_count, total_positives, size.

    Categorical columns (object dtype or ≤ 10 unique values) are grouped by
    value. Numeric columns with > 10 unique values are binned into n_bins
    quantile buckets. Subgroups with no true positives are excluded.
    Returns an empty DataFrame when X is empty or y contains no positives.
    """
    _COLS = ["feature", "subgroup", "fn_rate", "fn_count", "total_positives", "size"]
    if X.empty or int(y.sum()) == 0:
        return pd.DataFrame(columns=_COLS)

    y_pred = (oof_proba >= threshold).astype(int)
    is_fn = (y == 1) & (y_pred == 0)
    is_pos = y == 1

    records: list[dict[str, Any]] = []
    for col in X.columns:
        col_series = X[col]
        use_cat = col_series.dtype == object or col_series.nunique() <= 10
        if use_cat:
            groups: pd.Series[Any] = col_series
        else:
            if col_series.nunique() < n_bins:
                continue
            groups = pd.qcut(col_series, q=n_bins, duplicates="drop")

        for val in groups.unique():
            mask = groups == val
            total_pos = int((is_pos & mask).sum())
            fn_count = int((is_fn & mask).sum())
            if total_pos > 0:
                records.append(
                    {
                        "feature": col,
                        "subgroup": str(val),
                        "fn_rate": fn_count / total_pos,
                        "fn_count": fn_count,
                        "total_positives": total_pos,
                        "size": int(mask.sum()),
                    }
                )

    if not records:
        return pd.DataFrame(columns=_COLS)
    return (
        pd.DataFrame(records)
        .sort_values("fn_rate", ascending=False)
        .reset_index(drop=True)
    )


def subgroup_recall(
    oof_proba: npt.NDArray[np.float64],
    y: pd.Series,
    mask: npt.NDArray[np.bool_],
    threshold: float = 0.5,
) -> float:
    """Recall among true churners in a masked subgroup at a given threshold.

    Equivalently: 1 − FN rate, the complement of the metric shown in the
    FN profile chart. A positive delta between laps means the candidate
    reduced the false-negative rate in the target subgroup.

    Returns 0.0 when the subgroup contains no true positives.
    Used each lap to measure the blind-spot recall delta (Screen 3).
    """
    y_sub = np.asarray(y)[mask]
    p_sub = oof_proba[mask]
    pos = y_sub == 1
    return float((p_sub[pos] >= threshold).mean()) if pos.sum() > 0 else 0.0


def serving_available(
    candidate_cols: list[str],
    serving_cols: frozenset[str] | None = None,
) -> tuple[bool, str]:
    """Assert all columns needed to compute the candidate are available at serving time.

    serving_cols defaults to SERVING_COLS (raw IBM base after ingest cleaning).
    Fails immediately if any input column is absent — a leaked feature inflates OOF
    lift by cheating and must never reach the metric comparison.
    """
    if serving_cols is None:
        serving_cols = _get_serving_cols()
    unavailable = sorted(set(candidate_cols) - serving_cols)
    if unavailable:
        return False, f"columns not available at serving time: {unavailable}"
    return True, "all input columns available at serving time"


def redundancy_screen(
    candidate: pd.Series,
    adopted: pd.DataFrame,
    corr_threshold: float = CORR_THRESHOLD,
    cramers_v_threshold: float = CRAMERS_V_THRESHOLD,
) -> RedundancyResult:
    """Check whether the candidate adds new information relative to the adopted set.

    Pairwise checks only: |Spearman rank corr| for numeric candidate vs adopted numerics;
    Cramér's V for categorical candidate vs adopted categoricals. Spearman is used over
    Pearson to catch monotone nonlinear relationships (e.g. ratios, logs). Returns
    immediately if flagged — no further computation needed.

    Returns RedundancyResult(flagged=False) when the adopted set is empty.
    """
    if adopted.empty or len(adopted.columns) == 0:
        return RedundancyResult(
            flagged=False,
            max_corr=None,
            max_cramers_v=None,
            flag_reason=None,
        )

    is_numeric = pd.api.types.is_numeric_dtype(candidate)
    adopted_num = adopted.select_dtypes(include="number")
    adopted_cat = adopted.select_dtypes(include=["object", "string"])

    if is_numeric and not adopted_num.empty:
        corr_vals = adopted_num.apply(lambda col: abs_corr(candidate, col))
        max_corr = float(corr_vals.max())
        if max_corr > corr_threshold:
            return RedundancyResult(
                flagged=True,
                max_corr=max_corr,
                max_cramers_v=None,
                flag_reason=f"abs(corr)={max_corr:.3f} > {corr_threshold} with '{corr_vals.idxmax()}'",
            )
        return RedundancyResult(
            flagged=False, max_corr=max_corr, max_cramers_v=None, flag_reason=None
        )

    if not is_numeric and not adopted_cat.empty:
        cv_vals = adopted_cat.apply(lambda col: cramers_v(candidate, col))
        max_cramers_v = float(cv_vals.max())
        if max_cramers_v > cramers_v_threshold:
            return RedundancyResult(
                flagged=True,
                max_corr=None,
                max_cramers_v=max_cramers_v,
                flag_reason=f"Cramér's V={max_cramers_v:.3f} > {cramers_v_threshold} with '{cv_vals.idxmax()}'",
            )
        return RedundancyResult(
            flagged=False, max_corr=None, max_cramers_v=max_cramers_v, flag_reason=None
        )

    return RedundancyResult(
        flagged=False,
        max_corr=None,
        max_cramers_v=None,
        flag_reason="no adopted columns of matching dtype — cross-dtype check skipped",
    )


def candidate_importance(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    candidate_col: str,
    decoy_col: str,
    n_repeats: int = 10,
    random_state: int = 42,
    noise_floor_margin: float = IMPORTANCE_NOISE_FLOOR_MARGIN,
    model: Any = None,
) -> ImportanceResult:
    """Confirm the model attributes signal to the candidate above the decoy noise floor.

    Fits Pipeline(preprocessor, model) on 80 % of the data, then runs
    permutation_importance on the held-out 20 % — permuting in the original feature
    space so the preprocessor's transformation does not obscure per-feature attribution.
    Both candidate_col and decoy_col must be present in X.columns.

    model defaults to LGBMClassifier with class_weight="balanced". Tests pass
    DecisionTreeClassifier — scale-invariant (no StandardScaler in
    build_preprocessor's numeric branch), detects planted signal vs decoy noise
    in synthetic data, and is fast. DummyClassifier is NOT suitable: it ignores
    all features so permutation importance is meaningless noise for both candidate
    and decoy.
    """
    if model is None:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            class_weight="balanced", random_state=random_state, verbose=-1
        )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    pipe = Pipeline(
        steps=[
            ("pre", clone(preprocessor)),
            ("clf", clone(model)),
        ]
    )
    pipe.fit(X_tr, y_tr)

    feat_names = list(X.columns)
    cand_idx = feat_names.index(candidate_col)
    decoy_idx = feat_names.index(decoy_col)

    result = sk_perm_importance(
        pipe,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
    )
    cand_imp = float(result.importances_mean[cand_idx])
    decoy_imp = float(result.importances_mean[decoy_idx])

    return ImportanceResult(
        candidate_importance=cand_imp,
        noise_floor=decoy_imp,
        above_floor=cand_imp > max(decoy_imp, 0.0) + noise_floor_margin,
    )


def adoption_gate(
    serving_result: tuple[bool, str],
    redundancy_result: RedundancyResult,
    prior_pr_auc: float,
    new_pr_auc: float,
    sub_recall_delta: float | None,
    importance_result: ImportanceResult | None,
    swap_sub_recall_delta: float | None = None,
) -> AdoptionDecision:
    """Compose the four screens in ascending order of cost and return an adoption decision.

    Screens run cheapest-first so a candidate that fails early never burns a re-fit:

    1. Serving availability — hard pre-gate; failure here is immediate rejection.
    2. Redundancy — soft gate; rejects only when flagged=True, swap_sub_recall_delta is
       provided, AND sub_recall_delta does not exceed the swap baseline.
    3. Performance — new_pr_auc − prior_pr_auc must be ≥ MIN_PR_AUC_DELTA; flat or
       negative deltas reject.
    4. Importance — hard gate; rejects when candidate permutation importance does not
       exceed max(noise_floor, 0) + margin. The max(·, 0) floor prevents a negative
       decoy score from letting a zero-importance candidate pass.

    sub_recall_delta is not a rejection criterion on its own — it is a diagnostic that
    shows whether the candidate addressed the motivating blind spot, and feeds the
    Screen 2 swap comparison when swap_sub_recall_delta is provided.

    PR-AUC is the sole selection metric; Screen 2 is a soft structural guardrail and
    Screen 4 is a hard signal-confirmation guardrail.
    """
    pr_auc_delta = new_pr_auc - prior_pr_auc
    # Screen 1 — serving availability (hard pre-gate)
    if not serving_result[0]:
        return AdoptionDecision(
            adopted=False,
            rejection_screen=1,
            rejection_reason=f"serving unavailable — {serving_result[1]}",
        )

    # Screen 2 — redundancy (structural; no auto-reject unless swap comparison fails)
    # sub_recall_delta=None means recall was not computed (Screen 3 failed or no subgroup);
    # treat as failing the swap comparison — unknown recall cannot clear a redundancy flag.
    if redundancy_result.flagged and swap_sub_recall_delta is not None:
        if sub_recall_delta is None or sub_recall_delta <= swap_sub_recall_delta:
            recall_str = (
                f"{sub_recall_delta:.4f}"
                if sub_recall_delta is not None
                else "not computed"
            )
            return AdoptionDecision(
                adopted=False,
                rejection_screen=2,
                rejection_reason=(
                    f"redundant ({redundancy_result.flag_reason}) and recall Δ "
                    f"{recall_str} ≤ swap baseline {swap_sub_recall_delta:.4f}"
                ),
            )

    # Screen 3 — performance gate (improvement must meet minimum practical threshold)
    if pr_auc_delta < MIN_PR_AUC_DELTA:
        return AdoptionDecision(
            adopted=False,
            rejection_screen=3,
            rejection_reason=(
                f"PR-AUC delta {pr_auc_delta:+.4f} below minimum {MIN_PR_AUC_DELTA} "
                f"(prior={prior_pr_auc:.4f}, new={new_pr_auc:.4f})"
            ),
        )

    # Screen 4 — importance confirmation (only reached when Screen 3 passed)
    if importance_result is None:
        return AdoptionDecision(
            adopted=False,
            rejection_screen=4,
            rejection_reason="importance result not computed",
        )
    if not importance_result.above_floor:
        return AdoptionDecision(
            adopted=False,
            rejection_screen=4,
            rejection_reason=(
                f"importance {importance_result.candidate_importance:.4f} ≤ "
                f"effective floor {max(importance_result.noise_floor, 0.0) + IMPORTANCE_NOISE_FLOOR_MARGIN:.4f} "
                f"(noise={importance_result.noise_floor:.4f}, margin={IMPORTANCE_NOISE_FLOOR_MARGIN})"
            ),
        )

    return AdoptionDecision(adopted=True, rejection_screen=None, rejection_reason=None)


def run_lap(
    candidate: pd.Series,
    candidate_col: str,
    required_cols: list[str],
    feature_group: Literal["binary", "multi_cat", "numeric"],
    X_current: pd.DataFrame,
    y: pd.Series,
    active_binary: list[str],
    active_multi_cat: list[str],
    active_numeric: list[str],
    adopted_df: pd.DataFrame,
    decoy_col: pd.Series,
    decoy_col_name: str,
    current_oof_proba: npt.NDArray[np.float64],
    current_pr_auc: float,
    current_ci: tuple[float, float],
    build_preprocessor_fn: Callable[..., Any],
    bs_mask: npt.NDArray[np.bool_] | None = None,
    swap_sub_recall_delta: float | None = None,
    discovery_threshold: float = 0.5,
    random_state: int = 42,
    verbose: bool = True,
) -> LapEvaluation:
    """Run one discovery-loop candidate through the four-screen adoption gate.

    Factors notebooks/02a-feature-discovery.ipynb's per-lap Evaluate block (serving
    check -> redundancy -> OOF fit -> PR-AUC delta -> importance vs. decoy -> decision)
    into a single call so a gate-logic change applies to every lap at once instead of
    ~9 hand-edited notebook cells. feature_group selects which of active_binary/
    active_multi_cat/active_numeric the candidate — and the decoy noise column at
    Screen 4 — is appended to when building each ColumnTransformer via
    build_preprocessor_fn.

    bs_mask, when given, drives the Screen 4 blind-spot subgroup-recall diagnostic
    (sub_recall_delta, printed and returned); when None, a global false-negative-rate
    diagnostic is printed instead and sub_recall_delta stays None — the domain-wide
    laps (charge_per_service, num_add_on_services, monthly_to_total_ratio) use this
    branch, matching their original hand-written cells.
    """
    if feature_group not in ("binary", "multi_cat", "numeric"):
        raise ValueError(
            "feature_group must be 'binary', 'multi_cat', or 'numeric', "
            f"got {feature_group!r}"
        )

    def _groups(extra_numeric: str | None = None) -> dict[str, list[str]]:
        groups = {
            "binary": list(active_binary),
            "multi_cat": list(active_multi_cat),
            "numeric": list(active_numeric),
        }
        groups[feature_group].append(candidate_col)
        if extra_numeric is not None:
            groups["numeric"].append(extra_numeric)
        return groups

    srv = serving_available(required_cols)
    if verbose:
        print(f"  Screen 1 (serving)    : {'PASS' if srv[0] else 'FAIL'}  {srv[1]}")

    X_new = X_current.copy()
    X_new[candidate_col] = candidate.values

    red: RedundancyResult | None = None
    oof = current_oof_proba
    pr_auc = current_pr_auc
    ci = current_ci
    sub_delta: float | None = None
    imp: ImportanceResult | None = None

    if not srv[0]:
        decision = AdoptionDecision(
            adopted=False, rejection_screen=1, rejection_reason=srv[1]
        )
    else:
        red = redundancy_screen(candidate, adopted_df)
        if verbose:
            print(f"  Screen 2 (redundancy) : {red.summary}")

        groups = _groups()
        prep = build_preprocessor_fn(
            binary=groups["binary"],
            multi_cat=groups["multi_cat"],
            numeric=groups["numeric"],
        )
        oof = oof_predictions(X_new, y, prep, random_state=random_state)
        pr_auc = float(average_precision_score(y, oof))
        ci = bootstrap_pr_auc_ci(np.asarray(y), oof, random_state=random_state)
        delta = pr_auc - current_pr_auc
        if verbose:
            print(
                f"  Screen 3 (PR-AUC)     : {'PASS' if delta >= MIN_PR_AUC_DELTA else 'FAIL'}  "
                f"base={current_pr_auc:.4f}  new={pr_auc:.4f}  delta={delta:+.4f}  "
                f"min_delta={MIN_PR_AUC_DELTA}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]"
            )

        if delta < MIN_PR_AUC_DELTA:
            if verbose:
                print("  Screen 4 (importance) : SKIPPED  (Screen 3 failed)")
        else:
            X_imp = X_new.copy()
            X_imp[decoy_col_name] = decoy_col.values
            imp_groups = _groups(extra_numeric=decoy_col_name)
            prep_imp = build_preprocessor_fn(
                binary=imp_groups["binary"],
                multi_cat=imp_groups["multi_cat"],
                numeric=imp_groups["numeric"],
            )
            imp = candidate_importance(
                X_imp,
                y,
                prep_imp,
                candidate_col,
                decoy_col_name,
                random_state=random_state,
            )
            if verbose:
                print(
                    f"  Screen 4 (importance) : {'PASS' if imp.above_floor else 'FAIL'}  "
                    f"candidate={imp.candidate_importance:.4f}  "
                    f"floor={max(imp.noise_floor, 0.0) + IMPORTANCE_NOISE_FLOOR_MARGIN:.4f}  "
                    f"(noise={imp.noise_floor:.4f}, margin={IMPORTANCE_NOISE_FLOOR_MARGIN})"
                )

            if bs_mask is not None:
                recall_base = subgroup_recall(
                    current_oof_proba, y, bs_mask, threshold=discovery_threshold
                )
                recall_new = subgroup_recall(
                    oof, y, bs_mask, threshold=discovery_threshold
                )
                sub_delta = recall_new - recall_base
                if verbose:
                    print(
                        f"  blind-spot FN rate    : {1 - recall_base:.3f} -> {1 - recall_new:.3f}  "
                        f"(delta={sub_delta:+.4f})  [diagnostic]"
                    )
            else:
                y_arr = np.asarray(y)
                fn_base = 1.0 - float(
                    (current_oof_proba[y_arr == 1] >= discovery_threshold).mean()
                )
                fn_new = 1.0 - float((oof[y_arr == 1] >= discovery_threshold).mean())
                if verbose:
                    print(
                        f"  global FN rate        : {fn_base:.3f} -> {fn_new:.3f}  "
                        f"(delta={fn_new - fn_base:+.4f})  [diagnostic]"
                    )

        decision = adoption_gate(
            serving_result=srv,
            redundancy_result=red,
            prior_pr_auc=current_pr_auc,
            new_pr_auc=pr_auc,
            sub_recall_delta=sub_delta,
            importance_result=imp,
            swap_sub_recall_delta=swap_sub_recall_delta,
        )

    if verbose:
        print(f"  DECISION              : {'ADOPT' if decision.adopted else 'REJECT'}")
        if decision.rejection_screen:
            print(
                f"  Rejected at Screen    : {decision.rejection_screen}  "
                f"({decision.rejection_reason})"
            )

    return LapEvaluation(
        candidate_col=candidate_col,
        X_with_candidate=X_new,
        serving_result=srv,
        redundancy_result=red,
        oof_proba=oof,
        pr_auc=pr_auc,
        ci=ci,
        sub_recall_delta=sub_delta,
        importance_result=imp,
        decision=decision,
    )


def bootstrap_pr_auc_ci(
    y_true: npt.NDArray[np.int_],
    y_score: npt.NDArray[np.float64],
    n_iterations: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return a bootstrap confidence interval (lower, upper) for PR-AUC.

    Samples rows with replacement n_iterations times and computes
    average_precision_score on each resample. Resamples that land on a single
    class are skipped. Returns a symmetric ci-width percentile interval.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(float(average_precision_score(y_true[idx], y_score[idx])))

    if not scores:
        return (0.0, 0.0)
    alpha = 1.0 - ci
    lower = float(np.percentile(scores, 100.0 * alpha / 2.0))
    upper = float(np.percentile(scores, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def backward_elimination(
    X: pd.DataFrame,
    y: pd.Series,
    adopted: list[str],
    full_pr_auc: float,
    build_preprocessor_fn: Callable[..., Any],
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
    min_pr_auc_delta: float = MIN_PR_AUC_DELTA,
    n_folds: int = 5,
    random_state: int = 42,
    model: Any = None,
) -> list[BackwardEliminationResult]:
    """Test whether each adopted feature can be removed without hurting PR-AUC.

    For each feature in adopted, removes it from X and its type list, rebuilds
    the preprocessor via build_preprocessor_fn, and re-evaluates OOF PR-AUC.
    A feature is retained only when removal costs at least min_pr_auc_delta;
    otherwise, the forward pass over-counted its marginal contribution.

    Returns an empty list when adopted is empty.
    """
    results: list[BackwardEliminationResult] = []
    for feature in adopted:
        X_without = X.drop(columns=[feature])
        b = [f for f in binary if f != feature]
        mc = [f for f in multi_cat if f != feature]
        n = [f for f in numeric if f != feature]
        prep = build_preprocessor_fn(binary=b, multi_cat=mc, numeric=n)
        oof = oof_predictions(
            X_without,
            y,
            prep,
            n_folds=n_folds,
            random_state=random_state,
            model=model,
        )
        pr_auc_without = float(average_precision_score(y, oof))
        delta = full_pr_auc - pr_auc_without
        results.append(
            BackwardEliminationResult(
                feature=feature,
                pr_auc_without=pr_auc_without,
                delta=delta,
                keep=delta >= min_pr_auc_delta,
            )
        )
    return results


def write_provenance(
    laps: list[LapRecord],
    adopted: list[str],
    output_path: Path,
    *,
    random_state: int = 42,
    discovery_threshold: float,
) -> None:
    """Write per-lap records and the adopted feature list to a JSON provenance file.

    Also writes a companion provenance.md summary to output_path.parent.
    The JSON is the machine-readable artifact; the markdown is human-readable
    and rendered in the notebook.

    discovery_threshold — the threshold used for FN profiling and subgroup_recall
                          throughout the discovery loop; written to run_config so
                          sub_recall_delta values are reproducible from the file alone.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "run_config": {
            "random_state": random_state,
            "discovery_threshold": discovery_threshold,
            "CORR_THRESHOLD": CORR_THRESHOLD,
            "CRAMERS_V_THRESHOLD": CRAMERS_V_THRESHOLD,
            "IMPORTANCE_NOISE_FLOOR_MARGIN": IMPORTANCE_NOISE_FLOOR_MARGIN,
            "MIN_PR_AUC_DELTA": MIN_PR_AUC_DELTA,
        },
        "adopted_features": adopted,
        "laps": [dataclasses.asdict(lap) for lap in laps],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_path = output_path.parent / "provenance.md"
    rows = [
        "# Feature Discovery Provenance\n\n",
        f"**Adopted features:** {', '.join(adopted) if adopted else '(none)'}\n\n",
        "| Lap | Candidate | Decision | Screen | Reason |\n",
        "|-----|-----------|----------|--------|--------|\n",
    ]
    for lap in laps:
        screen = str(lap.rejection_screen) if lap.rejection_screen else "—"
        reason = lap.rejection_reason or "—"
        rows.append(
            f"| {lap.lap} | {lap.candidate} | {lap.decision} | {screen} | {reason} |\n"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(rows)
