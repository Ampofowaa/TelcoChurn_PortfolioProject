"""Pure plotting-data helpers for reliability diagrams.

Pure, side-effect-free — like models/diagnostics.py, this returns list[dict]
rows a caller renders however it likes: a calibration notebook now, later
evaluate.py's test-set reliability diagram, or a calibration-drift monitoring
dashboard. No matplotlib import.

Cost/EV curves and the r-sensitivity sweep are not duplicated here: both
already exist as pure functions in models.threshold (expected_value_curve,
r_sensitivity_sweep), because threshold.py needs them internally to derive
t* and its sensitivity plot. Callers wanting those series import them from
there directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["reliability_diagram_bins"]


def reliability_diagram_bins(
    proba: Sequence[float],
    y_true: Sequence[int],
    n_bins: int,
    strategy: str,
) -> list[dict[str, float]]:
    """Bin proba against y_true for a reliability diagram — one row per non-empty bin.

    Same edge construction (quantile or uniform, per strategy) as
    calibrate.expected_calibration_error, so a rendered diagram's bins match
    exactly the ECE number computed alongside it, rather than an
    independently tuned visualization. Empty bins are dropped rather than
    returned as NaN — quantile edges on a skewed probability vector can
    produce them, and a plotted NaN point is worse than one fewer marker.
    """
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y_true, dtype=float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0], edges[-1] = 0.0, 1.0

    bin_ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    rows: list[dict[str, float]] = []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_predicted": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
                "count": float(mask.sum()),
            }
        )
    return rows
