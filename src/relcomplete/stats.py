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


def betabin_logit(X, k, n, iters: int = 100) -> tuple[np.ndarray, float, np.ndarray]:
    """Beta-binomial regression: k_j of n_j items missed in answer j, with mean logit(mu_j) = [1, X_j] w.

    Items in one answer share a miss rate drawn around mu_j, so misses can clump (intra-answer correlation rho,
    the overdispersion a plain binomial model ignores). Fitted by Newton steps on the exact likelihood. Returns
    (w, rho, se), with se the standard errors of w from the observed information.
    """
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    k, n = np.asarray(k, float), np.asarray(n, float)
    i = np.arange(int(n.max()))[None, :]
    mk, mnk, mn = (i < m[:, None] for m in (k, n - k, n))  # masks for the sums over 0 <= i < m

    def ll_grad(t):
        mu = np.clip(0.5 * (1 + np.tanh(X @ t[:-1] / 2)), 1e-12, 1 - 1e-12)  # sigmoid, without overflow
        phi = np.exp(np.clip(t[-1], -10.0, 15.0))  # precision; rho = 1 / (1 + phi)
        a, b = (mu * phi)[:, None] + i, ((1 - mu) * phi)[:, None] + i
        ab = phi + i
        ll = np.log(a)[mk].sum() + np.log(b)[mnk].sum() - np.log(np.broadcast_to(ab, mn.shape))[mn].sum()
        tot = np.where(mn, 1 / ab, 0).sum(1)
        da, db = np.where(mk, 1 / a, 0).sum(1) - tot, np.where(mnk, 1 / b, 0).sum(1) - tot
        g = np.append(X.T @ ((da - db) * phi * mu * (1 - mu)), ((da * mu + db * (1 - mu)) * phi).sum())
        return ll, g

    def hessian(t, h=1e-5):
        H = np.column_stack([(ll_grad(t + h * e)[1] - ll_grad(t - h * e)[1]) / (2 * h) for e in np.eye(len(t))])
        return (H + H.T) / 2

    t = np.zeros(X.shape[1] + 1)
    t[0] = np.log((k.sum() + 0.5) / (n.sum() - k.sum() + 0.5))  # start at the pooled miss rate
    t[-1] = np.log(20.0)  # and rho = 0.05
    ll, g = ll_grad(t)
    for _ in range(iters):
        H = hessian(t)
        try:
            step = -np.linalg.solve(H, g)
            if g @ step <= 0:  # not an ascent direction: fall back to a gradient step
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            step = g / max(1.0, np.abs(g).max())
        for _ in range(30):  # backtracking line search
            new = t + step
            new[-1] = np.clip(new[-1], -10.0, 15.0)  # rho -> 0 (no clumping) sends the precision to infinity
            ll_new, g_new = ll_grad(new)
            if ll_new >= ll - 1e-10:
                break
            step /= 2
        done = np.abs(new - t).max() < 1e-8
        t, ll, g = new, ll_new, g_new
        if done:
            break
    H = hessian(t)
    p = X.shape[1]
    var = np.diag(np.linalg.inv(-H))[:p]
    if t[-1] >= 15.0 or not (np.isfinite(var).all() and (var > 0).all()):
        # rho at or near 0: the likelihood is flat in the precision, so treat it as fixed
        var = np.diag(np.linalg.inv(-H[:p, :p]))
    return t[:-1], float(1 / (1 + np.exp(t[-1]))), np.sqrt(var)


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
