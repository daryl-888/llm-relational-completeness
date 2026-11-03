"""The statistics behind the report's intervals and trends."""
import numpy as np

from relcomplete.stats import betabin_logit, boot_mean, logit, logit_boot, wilson


def test_wilson_known_values():
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    lo, hi = wilson(100, 100)  # stays inside [0, 1] at the boundary
    assert 0.95 < lo < 1.0 and abs(hi - 1.0) < 1e-12


def test_boot_mean_covers_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 1.0, 400)
    lo, hi = boot_mean(x, rng)
    assert lo < x.mean() < hi and hi - lo < 0.3


def test_logit_recovers_a_known_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-(0.5 - 1.2 * x)))).astype(float)
    w = logit(x[:, None], y)
    assert abs(w[0] - 0.5) < 0.15 and abs(w[1] + 1.2) < 0.15


def test_logit_stays_finite_when_separable():
    x = np.array([0, 0, 1, 1, 2, 2], float)[:, None]
    y = np.array([1, 1, 1, 1, 0, 0], float)
    assert np.isfinite(logit(x, y)).all()


def test_cluster_bootstrap_resamples_whole_clusters():
    rng = np.random.default_rng(2)
    x = np.repeat(np.arange(10.0), 3)[:, None]
    y = (np.repeat(np.arange(10), 3) < 5).astype(float)
    bs = logit_boot(x, y, rng, clusters=np.repeat(np.arange(10), 3), b=50)
    assert bs.shape == (50, 2) and np.isfinite(bs).all()


def test_betabin_logit_recovers_slope_and_clumping():
    rng = np.random.default_rng(3)
    n = np.repeat([10, 50, 200], 400).astype(float)
    x = np.log2(n)
    mu = 1 / (1 + np.exp(-(-5.0 + 0.4 * x)))
    rho = 0.05
    p = rng.beta(mu * (1 / rho - 1), (1 - mu) * (1 / rho - 1))  # shared miss rate per answer
    k = rng.binomial(n.astype(int), p)
    w, rho_hat, se = betabin_logit(x[:, None], k, n)
    assert abs(w[1] - 0.4) < 3 * se[1] and abs(rho_hat - rho) < 0.02


def test_betabin_logit_without_clumping():
    rng = np.random.default_rng(4)
    n = np.repeat([5, 50, 500], 50).astype(float)
    k = rng.binomial(n.astype(int), 0.02)
    w, rho_hat, se = betabin_logit(np.log2(n)[:, None], k, n)
    assert rho_hat < 0.01 and np.isfinite(se).all()
