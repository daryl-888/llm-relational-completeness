"""How many table instances per cell does the fully crossed answer-size experiment need?

  .venv/bin/python analysis/power.py

The proposed design crosses answer size g in {5, 10, 20, 50, 100, 200, 500} with table size N in {1000, 2000,
4000}. The question it must answer is whether each item's chance of being missed rises with g at fixed N, beyond
the (1 - q)^g effect. This script simulates that experiment and reports the power of the test.

Simulation, calibrated to analysis/item_level.md:
  - per-item miss probability logit(p) = a + b log2(g) + c log2(N) + u, with u ~ Normal(0, sigma^2) shared by the
    items of one answer (misses clump within answers; observed dispersion 1-19x);
  - a is set so that p = BASE at g = 50, N = 1000 (Claude Haiku 5.5 misses about 2% of items at g = 50-100);
  - c = log(2): odds double per doubling of N, below the observed 3.7 to avoid extrapolating it;
  - b = log(OR) for the per-doubling odds ratio of interest (OR = 1 is the independent-error null).
The analysis model is a beta-binomial regression of each answer's missed count on log2(g) and log2(N)
(relcomplete.stats.betabin_logit), which allows misses to clump within answers. A plain binomial model with
robust standard errors rejected a true null 6-15% of the time at these sample sizes; the beta-binomial Wald test
is closer to 5% but not exact, so power is reported size-adjusted: the critical value is the 95th percentile of
|z| in the simulated null experiments with the same settings.

Writes analysis/power.md.
"""
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from relcomplete.stats import betabin_logit

ROOT = Path(__file__).resolve().parents[1]
G = [5, 10, 20, 50, 100, 200, 500]
N = [1000, 2000, 4000]
C = np.log(2)
SIMS = 300


def simulate(rng, per_cell: int, odds_ratio: float, base: float, sigma: float) -> tuple[float, float]:
    """One simulated experiment: the z statistic for log2(g), and the dispersion of missed counts at g = 100."""
    g = np.repeat([x for x in G for _ in N], per_cell).astype(float)
    t = np.repeat([y for _ in G for y in N], per_cell).astype(float)
    b = np.log(odds_ratio)
    a = np.log(base / (1 - base)) - b * np.log2(50) - C * np.log2(1000)
    p = 1 / (1 + np.exp(-(a + b * np.log2(g) + C * np.log2(t) + rng.normal(0, sigma, len(g)))))
    k = rng.binomial(g.astype(int), p)
    w, _, se = betabin_logit(np.column_stack([np.log2(g), np.log2(t)]), k, g)
    big = g == 100
    q = k[big].mean() / 100
    disp = k[big].var(ddof=1) / (100 * q * (1 - q)) if 0 < q < 1 else np.nan
    return w[1] / se[1], disp


def run_setting(setting: tuple[int, float, float, int]) -> dict:
    """All simulations for one (base, sigma, instances per cell) setting, with its own seed."""
    i, base, sigma, per_cell = setting
    rng = np.random.default_rng([0, i])
    null = [simulate(rng, per_cell, 1.0, base, sigma) for _ in range(SIMS)]
    crit = np.percentile(np.abs([z for z, _ in null]), 95)
    row = {"base": base, "sigma": sigma, "per_cell": per_cell,
           "false_pos": np.mean(np.abs([z for z, _ in null]) > 1.96),
           "dispersion": np.nanmedian([d for _, d in null])}
    for odds_ratio in [1.2, 1.5]:
        alt = [simulate(rng, per_cell, odds_ratio, base, sigma)[0] for _ in range(SIMS)]
        row[f"OR {odds_ratio}"] = np.mean(np.abs(alt) > crit)
    return row


def main() -> None:
    settings = [(i, base, sigma, per_cell) for i, (base, sigma, per_cell) in
                enumerate(itertools.product([0.01, 0.02], [0.5, 1.0], [3, 5, 10]))]
    with ProcessPoolExecutor() as pool:  # settings are independent and seeded separately
        rows = list(pool.map(run_setting, settings))
    d = pd.DataFrame(rows)
    cells = len(G) * len(N)
    md = ["# Power of the fully crossed answer-size experiment", "",
          f"Design: answer size g in {G} x table size N in {N} ({cells} cells), one model, one prompt. {SIMS} "
          "simulated experiments per setting. base = per-item miss rate at g = 50, N = 1000; sigma = spread of "
          "answer-level difficulty (logit scale).", ""]
    tab = d.assign(answers=d.per_cell * cells)
    for col in ["false_pos", "OR 1.2", "OR 1.5"]:
        tab[col] = (100 * tab[col]).round(0).astype(int).astype(str) + "%"
    tab["dispersion"] = tab.dispersion.round(1)
    tab = tab.rename(columns={"per_cell": "instances per cell", "false_pos": "false positives at |z| > 1.96",
                              "OR 1.2": "power, OR 1.2 per doubling", "OR 1.5": "power, OR 1.5 per doubling",
                              "dispersion": "dispersion at g = 100"})
    md += [tab[["base", "sigma", "instances per cell", "answers", "false positives at |z| > 1.96",
                "power, OR 1.2 per doubling", "power, OR 1.5 per doubling", "dispersion at g = 100"]]
           .to_markdown(index=False), "",
           "Power is size-adjusted (critical value from the simulated null). For comparison, item_level.md "
           "estimates the pooled odds ratio for answer size at 1.50 (beta-binomial) to 1.69 (cluster bootstrap), "
           "and observes dispersion of 1-19x at g = 100.", ""]
    (ROOT / "analysis" / "power.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
