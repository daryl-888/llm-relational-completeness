"""Is the drop in exact-set accuracy just errors adding up, or do models get worse per item as answer sets grow?

  .venv/bin/python analysis/item_level.py [--results results/full_v0]

The independent-error baseline: if every correct item is missed with the same probability q, then an answer of
n items has no omission with probability (1 - q)^n, so exact-set accuracy falls as n grows even though nothing
about the model changes. This script tests that baseline on the E1 in-context enumeration questions in three
ways:
  1. Item level: logistic regression of "this correct item was missed" on log2(answer size), log2(table size),
     input format and model, with a bootstrap that resamples whole answers (items in one answer are not
     independent). An odds ratio above 1 for answer size means each item is more likely to be missed in a
     larger answer set, beyond the (1 - q)^n effect.
  2. Answer level: the observed share of answers with no omission, against (1 - q)^n with q estimated from small
     answer sets (n <= 10).
  3. Independence: within each (model, answer size) cell, the variance of the number of missed items relative
     to a binomial. Values well above 1 mean misses clump in some answers rather than striking at random.
Answers whose identifiers were all wrong (no correct item although at least n items were listed) are excluded:
they are identifier failures, not omissions.

Writes analysis/item_level.md and analysis/item_level.pdf.
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from relcomplete.report import load_results  # noqa: E402
from relcomplete.schema import Benchmark  # noqa: E402
from relcomplete.score import norm  # noqa: E402
from relcomplete.stats import logit, logit_boot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ma", ROOT / "report" / "make_assets.py")
ma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ma)  # reuse the report's model names, colours and markers
MIN_MISSES = 20  # per-model estimates need enough missed items to be meaningful


def item_table(e: pd.DataFrame, bench: Benchmark) -> pd.DataFrame:
    qmap = {q.id: q for q in bench.questions}
    rows = []
    for _, r in e.iterrows():
        q = qmap[r.id]
        t = bench.tables_for(q)[0]
        pos = {norm(k): i / len(t.rows) for i, k in enumerate(t.col(t.key))}
        missed = {x[0] for x in r.missed_items}
        for g in q.gold:
            k = norm(g[0])
            rows.append({"Model": r.Model, "answer": f"{r.Model}|{r.id}", "n": r.n_gold, "rows": r.n_rows,
                         "fmt": r.fmt, "pos": pos[k], "miss": float(k in missed)})
    return pd.DataFrame(rows)


def design(it: pd.DataFrame, models: list[str] | None) -> np.ndarray:
    cols = [np.log2(it.n), np.log2(it.rows)] + [(it.fmt == f).astype(float) for f in ["csv", "json", "nl"]]
    if models:
        cols += [(it.Model == m).astype(float) for m in models[1:]]
    return np.column_stack(cols)


def or_ci(X, y, clusters, rng, j) -> tuple[float, float, float]:
    w = logit(X, y)
    bs = logit_boot(X, y, rng, clusters=clusters, b=1000)
    lo, hi = np.exp(np.percentile(bs[:, j], [2.5, 97.5]))
    return float(np.exp(w[j])), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "full_v0")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    bench = Benchmark.load(ROOT / "data" / "benchmark_v0")
    d = load_results(args.results, bench)
    e = d[(d.repeat == 0) & (d.variant == "direct") & d.model.isin(ma.SHORT) & (d.op == "enumeration")
          & (d.setting == "context") & d.valid].copy()
    e["Model"] = e.model.map(ma.SHORT)
    ident = e[(e.tp == 0) & (e.n_pred >= e.n_gold)]
    e = e.drop(ident.index)
    it = item_table(e, bench)
    models = [m for m in ma.ORDER if m in set(it.Model)]
    enough = [m for m in models if it[it.Model == m].miss.sum() >= MIN_MISSES]
    md = ["# Item-level omissions and the independent-error baseline", "",
          f"In-context enumeration, E1: {len(e)} valid answers, {len(it):,} correct items. Excluded as identifier "
          f"failures: {', '.join(f'{m} {i}' for m, i in zip(ident.Model, ident.id)) or 'none'}.", ""]

    # 1. per-item miss rate, and item-level regression
    rate = (it.pivot_table(index="n", columns="Model", values="miss") * 100).reindex(columns=models)
    md += ["## 1. Per-item miss rate (%) by answer size", "", rate.round(2).to_markdown(), ""]
    reg = []
    for m in enough:
        g = it[it.Model == m]
        a = or_ci(design(g, None), g.miss.values, g.answer.values, rng, 1)
        t = or_ci(design(g, None), g.miss.values, g.answer.values, rng, 2)
        reg.append([m, int(g.miss.sum()), f"{a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}]", f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"])
    a = or_ci(design(it, models), it.miss.values, it.answer.values, rng, 1)
    t = or_ci(design(it, models), it.miss.values, it.answer.values, rng, 2)
    reg.append(["All models (model fixed effects)", int(it.miss.sum()), f"{a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}]",
                f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"])
    md += ["## Item-level logistic regression", "",
           "Odds ratio that a given correct item is missed, per doubling, adjusted for the other size and for input "
           f"format; 95% intervals from resampling whole answers. Models with fewer than {MIN_MISSES} missed items "
           "are omitted.", "",
           pd.DataFrame(reg, columns=["Model", "Missed items", "OR: answer size", "OR: table size"]).to_markdown(
               index=False), ""]

    # 2. observed vs (1 - q_small)^n, and 3. dispersion
    obs = []
    for m in models:
        g, gi = e[e.Model == m], it[it.Model == m]
        q_small = gi[gi.n <= 10].miss.mean()
        for n, h in g.groupby("n_gold"):
            k = (h.missing + h.near_miss).values
            q_n = k.mean() / n
            disp = k.var(ddof=1) / (n * q_n * (1 - q_n)) if q_n > 0 and len(k) > 1 else np.nan
            obs.append([m, n, len(h), 100 * (k == 0).mean(), 100 * (1 - q_small) ** n, disp])
    obs = pd.DataFrame(obs, columns=["Model", "n", "answers", "observed", "baseline", "dispersion"])
    cell = obs.assign(c=[f"{o:.0f} ({b:.0f})" for o, b in zip(obs.observed, obs.baseline)])
    tab = cell.pivot_table(index="Model", columns="n", values="c", aggfunc="first").reindex(models)
    tab.columns = [f"n = {c}" for c in tab.columns]
    md += ["## 2. Answers with no omission (%): observed (baseline)", "",
           "Baseline = (1 - q)^n, with q each model's per-item miss rate on answer sets of 10 or fewer items.", "",
           tab.to_markdown(), "",
           "## 3. Dispersion of missed counts (variance / binomial variance)", "",
           "1 = misses strike independently; much larger = they clump in some answers.", "",
           obs.pivot_table(index="n", columns="Model", values="dispersion").round(1).reindex(columns=models)
           .to_markdown(), ""]
    clumped = it.groupby("answer").filter(lambda g: g.miss.sum() >= 3)
    pos = clumped[clumped.miss == 1].pos
    md += [f"In answers with 3 or more misses ({clumped.answer.nunique()}), the missed rows sit in the first, middle "
           f"and last third of the table {100 * (pos < 1 / 3).mean():.0f}%, {100 * ((pos >= 1 / 3) & (pos < 2 / 3)).mean():.0f}% "
           f"and {100 * (pos >= 2 / 3).mean():.0f}% of the time (33% each if spread evenly).", ""]
    (ROOT / "analysis" / "item_level.md").write_text("\n".join(md) + "\n")

    # Figure: per-item miss rate (left), observed vs baseline for models with enough misses (right)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    ns = sorted(it.n.unique())
    for m in enough:
        r = rate[m].reindex(ns)
        axes[0].plot(ns, r.values, color=ma.COLOR[m], marker=ma.MARKER[m], label=m)
        o = obs[obs.Model == m].set_index("n").reindex(ns)
        axes[1].plot(ns, o.observed.values, color=ma.COLOR[m], marker=ma.MARKER[m], label=f"{m}: observed")
        axes[1].plot(ns, o.baseline.values, color=ma.COLOR[m], linestyle="--", linewidth=1.0,
                     label=f"{m}: (1$-$q)$^n$")
    axes[0].set(xscale="log", xlabel="Correct items in the answer", ylabel="Items missed (%)")
    axes[1].set(xscale="log", xlabel="Correct items in the answer", ylabel="Answers with no omission (%)",
                ylim=(0, 104))
    for ax in axes:
        ax.set_xticks(ns, [str(v) for v in ns])
        ax.minorticks_off()
    axes[0].legend(fontsize=6.5, loc="upper left")
    axes[1].legend(fontsize=5.5, loc="lower left", ncol=1)
    fig.tight_layout()
    fig.savefig(ROOT / "analysis" / "item_level.pdf")
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
