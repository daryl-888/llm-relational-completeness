"""Statistics used in the report: intervals, paired differences and logistic trends.

Follows Miller (2024), "Adding Error Bars to Evals": Wilson intervals for proportions, question-level bootstrap
for means and paired differences, and a cluster bootstrap (resampling questions) when one question contributes
several rows.
"""
from __future__ import annotations

import numpy as np

B = 2000  # bootstrap resamples


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot_mean(x, rng: np.random.Generator, b: int = B) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for the mean of x (e.g. paired per-question differences)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    m = x[rng.integers(0, len(x), (b, len(x)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return (float(lo), float(hi))


def logit(X, y, ridge: float = 1e-3) -> np.ndarray:
    """Logistic regression by Newton steps; returns [intercept, coefficients...].

    A tiny ridge keeps the fit finite when outcomes are (nearly) separable, e.g. a model that almost never fails.
    """
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - ridge * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return w


def logit_boot(X, y, rng: np.random.Generator, clusters=None, b: int = 1000) -> np.ndarray:
    """Bootstrap distribution of logistic coefficients (rows: resamples). Resamples whole clusters if given."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if clusters is None:
        idx_sets = [rng.integers(0, len(y), len(y)) for _ in range(b)]
    else:
        clusters = np.asarray(clusters)
        groups = {c: np.where(clusters == c)[0] for c in np.unique(clusters)}
        keys = list(groups)
        idx_sets = [np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
                    for _ in range(b)]
    return np.array([logit(X[i], y[i]) for i in idx_sets])
