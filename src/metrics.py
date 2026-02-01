from __future__ import annotations

import numpy as np


def weighted_zero_mean_r2(y_true, y_pred, weight) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weight = np.asarray(weight, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(weight)
    if not np.any(mask):
        return 0.0

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    weight = weight[mask]

    denom = np.sum(weight * y_true**2)
    if denom == 0.0:
        return 0.0

    numer = np.sum(weight * (y_true - y_pred) ** 2)
    return 1.0 - (numer / denom)
