"""Generate the paper's tables, figures and in-text numbers from a results directory.

  .venv/bin/python report/make_assets.py [--results results/full_v0]

Writes report/generated/*.tex (tables, and numbers.tex with \\newcommand macros used in the text) and
report/generated/*.pdf (figures). Statistics follow Miller (2024), "Adding Error Bars to Evals": 95% Wilson
intervals for proportions, question-level bootstrap intervals for means, and paired question-level
differences when comparing prompts or models on the same questions.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from relcomplete.report import load_results, load_usage, usage_table  # noqa: E402
from relcomplete.schema import Benchmark  # noqa: E402
from relcomplete.score import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated"
RNG = np.random.default_rng(0)
B = 2000  # bootstrap resamples

# Models in the paper. MiMo was run to completion on OpenRouter; its partial OpenCode Go runs and the stopped
# Haiku 4.5 run (90 questions) are excluded.
SHORT = {"opencode-go/deepseek-v4-pro": "DeepSeek V4 Pro", "opencode-go/deepseek-v4-flash": "DeepSeek V4 Flash",
         "claude-cli/claude-haiku-5-5": "Claude Haiku 5.5", "xiaomi/mimo-v2.6-pro": "MiMo-V2.6 Pro",
         "xiaomi/mimo-v2.6-flash": "MiMo-V2.6 Flash"}
ORDER = list(SHORT.values())
KEY = dict(zip(ORDER, ["DSPro", "DSFlash", "Haiku", "MiMoPro", "MiMoFlash"]))  # letters only, for \macros
# Validated categorical order from the dataviz reference palette (slots 1-5), plus a marker per model so
# identity never depends on colour alone (print, colour-vision deficiency).
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKER = dict(zip(ORDER, ["o", "s", "^", "D", "v"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
OPS = [("enumeration", "Enumeration"), ("selection", "Selection"), ("projection", "Projection"),
       ("join", "JOIN"), ("aggregation", "Aggregation"), ("nested", "Nested query"),
       ("setop", "Set operation")]

VARIANTS = {("enumeration", "context"): "one category filter",
            ("enumeration", "parametric"): "U.S. states, elements, presidents",
            ("enumeration", "rag"): "BM25 top-$k$, $k\\in\\{10,20,50\\}$",
            ("selection", "context"): "1--3 conditions",
            ("projection", "context"): "DISTINCT; 2--3 attributes",
            ("join", "context"): "1:1, 1:N, M:N",
            ("aggregation", "context"): "COUNT, SUM, MIN/MAX/AVG, GROUP BY",
            ("nested", "context"): "IN, NOT IN, correlated, scalar",
            ("setop", "context"): "UNION, INTERSECT, EXCEPT, 3-way"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "savefig.bbox": "tight", "pdf.fonttype": 42})


# ----------------------------------------------------------------------------- statistics

def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot_mean(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    m = x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def logit_slope(x, y) -> float:
    """Slope of a one-predictor logistic regression (Newton steps, tiny ridge for separable data)."""
    X = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - 1e-3 * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-3 * np.eye(2)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w[1]


def pct(v: float, d: int = 0) -> str:
    return "--" if pd.isna(v) else f"{100 * v:.{d}f}"


def ci_pct(lo: float, hi: float, d: int = 0) -> str:
    return f"[{pct(lo, d)}, {pct(hi, d)}]"


# ----------------------------------------------------------------------------- output helpers

def write_table(path: Path, header: list[str], rows: list[list[str]], spec: str, midrules: tuple = ()) -> None:
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


MACROS: dict[str, str] = {}


def macro(name: str, value) -> None:
    MACROS[name] = str(value)


def legend_handles(ax):
    ax.legend(loc="lower left", fontsize=7, handlelength=1.6, borderaxespad=0.2)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "full_v0")
    ap.add_argument("--bench", type=Path, default=ROOT / "data" / "benchmark_v0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = Benchmark.load(args.bench)
    qmap = {q.id: q for q in bench.questions}

    df = load_results(args.results, bench)
    df = df[df.model.isin(SHORT)].copy()
    df["Model"] = df.model.map(SHORT)
    df["exact"] = df.exact.astype(bool)
    df["api_error"] = df.error.notna() & ~df.error.fillna("").str.contains("content filtering")
    e1 = df[(df.repeat == 0) & (df.variant == "direct")]
    ctx = e1[(e1.setting == "context") & (e1.answer_type != "scalar")].copy()  # the core: 365 questions
    ctx["omission"] = ctx.missing.fillna(0) + ctx.near_miss.fillna(0) > 0
    ctx["commission"] = ctx.wrong.fillna(0) + ctx.made_up.fillna(0) + ctx.duplicates.fillna(0) > 0
    macro("nctx", ctx.groupby("Model").size().iloc[0])

    # ---- Benchmark composition
    comp = []
    for op, label in OPS:
        for setting, slabel in [("context", "in context"), ("parametric", "memory only"), ("rag", "retrieval")]:
            qs = [q for q in bench.questions if q.op == op and q.setting == setting]
            if not qs:
                continue
            g = [q.meta["n_gold"] for q in qs]
            comp.append([label + ("" if setting == "context" else f" ({slabel})"), str(len(qs)),
                         VARIANTS[(op, setting)],
                         f"{min(g)}--{max(g)}", f"{np.median([q.meta['context_chars'] for q in qs]) / 4000:.1f}"])
    write_table(OUT / "composition.tex", ["Slice", "Qs", "Variants", "Answer size", r"Context (k tok.)"],
                comp, "lrlrr", midrules=(3,))

    # ---- T: main results (in-context set questions), with CIs and paired difference from the best model
    rows, stats = [], {}
    for m in ORDER:
        g = ctx[ctx.Model == m]
        n, k = len(g), int(g.exact.sum())
        lo, hi = wilson(k, n)
        rlo, rhi = boot_mean(g.recall)
        stats[m] = {"exact": k / n, "recall": g.recall.mean(), "precision": g.precision.mean(),
                    "omit": g.omission.mean(), "commit": g.commission.mean(), "n": n}
        stats[m].update(exact_ci=(lo, hi), recall_ci=(rlo, rhi))
    best = max(ORDER, key=lambda m: stats[m]["exact"])
    piv = ctx.pivot_table(index="id", columns="Model", values="exact", aggfunc="first").astype(float)
    for m in ORDER:
        s = stats[m]
        if m == best:
            diff = "(reference)"
        else:
            d = (piv[m] - piv[best]).dropna().values
            dlo, dhi = boot_mean(d)
            diff = f"${100 * d.mean():+.1f}$ [{100 * dlo:+.1f}, {100 * dhi:+.1f}]"
        rows.append([m, f"{pct(s['exact'])} {ci_pct(*s['exact_ci'])}", diff,
                     f"{s['recall']:.3f} [{s['recall_ci'][0]:.3f}, {s['recall_ci'][1]:.3f}]",
                     f"{s['precision']:.3f}", pct(s["omit"]), pct(s["commit"])])
        key = KEY[m]
        macro(f"exact{key}", pct(s["exact"]))
        macro(f"recall{key}", f"{s['recall']:.3f}")
        macro(f"omit{key}", pct(s["omit"]))
    write_table(OUT / "main.tex", ["Model", r"Exact-set acc.\ (\%)", r"$\Delta$ vs.\ best (pp)", "Mean recall",
                                   "Mean prec.", r"Omit.\ (\%)", r"Comm.\ (\%)"], rows, "lrrrrrr")
    macro("bestmodel", best)
    ex = [stats[m]["exact"] for m in ORDER]
    rc = [stats[m]["recall"] for m in ORDER]
    macro("exactmin", pct(min(ex)))
    macro("exactmax", pct(max(ex)))
    macro("recallmin", f"{min(rc):.3f}")
    macro("recallmax", f"{max(rc):.3f}")
    om = [stats[m]["omit"] for m in ORDER]
    macro("omitmin", pct(min(om)))
    macro("omitmax", pct(max(om)))
    errs = ctx.groupby("Model")[["missing", "near_miss", "wrong", "made_up", "duplicates"]].sum().reindex(ORDER)
    errs = errs.astype(int)
    write_table(OUT / "error_counts.tex", ["Model", "Missing", "Near-miss", "Wrong", "Made-up", "Duplicate"],
                [[m] + [f"{v:,}" for v in errs.loc[m]] for m in ORDER], "lrrrrr")
    others = errs.drop(index="MiMo-V2.6 Flash")
    macro("otherswrong", int(others.wrong.sum()))
    macro("othersmadeup", int(others.made_up.sum()))
    macro("flashmadeup", f"{int(errs.loc['MiMo-V2.6 Flash', 'made_up']):,}")
    worst = ctx[ctx.Model == "MiMo-V2.6 Flash"].sort_values("made_up", ascending=False).iloc[0]
    macro("flashworstmade", f"{int(worst.made_up):,}")
    macro("flashworstgold", int(worst.n_gold))

    # ---- F: error profile (share of answers with an omission / a commission)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    y = np.arange(len(ORDER))
    for j, (col, lab, c) in enumerate([("omit", "Omits $\\geq$1 correct item", "#2a78d6"),
                                       ("commit", "Adds $\\geq$1 incorrect item", "#eb6834")]):
        v = [100 * stats[m][col] for m in ORDER]
        ax.barh(y + (j - 0.5) * 0.36, v, height=0.34, color=c, label=lab, edgecolor="white", linewidth=0.8)
        for yi, vi in zip(y, v):
            ax.text(vi + 0.6, yi + (j - 0.5) * 0.36, f"{vi:.0f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Answers (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(100 * stats[m]["omit"] for m in ORDER) * 1.18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=6.5, handlelength=1.2)
    fig.savefig(OUT / "error_profile.pdf")
    plt.close(fig)

    # ---- F: completeness vs answer size and table size (in-context enumeration), with Wilson intervals
    enum = ctx[ctx.op == "enumeration"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    for ax, x, lab in zip(axes, ["n_gold", "n_rows"], ["Correct items in the answer", "Records in the table"]):
        xs = sorted(enum[x].unique())
        for i, m in enumerate(ORDER):
            g = enum[enum.Model == m]
            pts = [(v, g[g[x] == v].exact.sum(), (g[x] == v).sum()) for v in xs]
            p = [k / n for _, k, n in pts]
            lo, hi = zip(*[wilson(k, n) for _, k, n in pts])
            off = np.array(xs) * (1 + 0.035 * (i - 2))  # dodge on the log axis
            ax.errorbar(off, 100 * np.array(p), yerr=[100 * (np.array(p) - lo), 100 * (np.array(hi) - p)],
                        color=COLOR[m], marker=MARKER[m], capsize=0, elinewidth=0.8, label=m)
        ax.set(xscale="log", xlabel=lab)
        ax.set_xticks(xs, [str(v) for v in xs])
        ax.minorticks_off()
    axes[0].set_ylabel("Exact-set accuracy (%)")
    axes[0].set_ylim(0, 104)
    legend_handles(axes[0])
    fig.tight_layout()
    fig.savefig(OUT / "size_curves.pdf")
    plt.close(fig)
    n_per = enum.groupby(["Model", "n_gold"]).size()
    macro("nperanswersize", f"{n_per.min()}--{n_per.max()}")

    # ---- T: trend - odds ratio per doubling of answer size (logistic regression, bootstrap CI)
    trow = []
    for m in ORDER:
        g = enum[enum.Model == m]
        x, yv = np.log2(g.n_gold.values.astype(float)), g.exact.values.astype(float)
        b = logit_slope(x, yv)
        boots = []
        for _ in range(B):
            idx = RNG.integers(0, len(g), len(g))
            boots.append(logit_slope(x[idx], yv[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        small, large = g[g.n_gold <= 10].exact.mean(), g[g.n_gold >= 50].exact.mean()
        n_fail = int((yv == 0).sum())
        orcell = (f"{np.exp(b):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}]" if np.exp(hi) < 20
                  else f"not estimable ({n_fail} failures)")
        trow.append([m, pct(small), pct(large), orcell])
        key = KEY[m]
        macro(f"small{key}", pct(small))
        macro(f"large{key}", pct(large))
        macro(f"or{key}", f"{np.exp(b):.2f}")
    write_table(OUT / "trend.tex", ["Model", r"$\leq$10 items (\%)", r"$\geq$50 items (\%)",
                                    r"Odds ratio per doubling [95\% CI]"], trow, "lrrr")

    # ---- T: exact-set accuracy by operation (in context)
    orow = []
    for op, label in OPS:
        g = ctx[ctx.op == op]
        if g.empty:
            continue
        cells = []
        for m in ORDER:
            h = g[g.Model == m]
            cells.append(pct(h.exact.mean()))
        orow.append([label, str(len(g) // len(ORDER))] + cells)
    write_table(OUT / "by_operation.tex", ["Operation", "Qs"] + [m.replace(" ", r"\ ") for m in ORDER],
                orow, "lr" + "r" * len(ORDER))

    # ---- T: memory-only and retrieval settings
    srow = []
    for m in ORDER:
        p = e1[(e1.Model == m) & (e1.setting == "parametric")]
        r = e1[(e1.Model == m) & (e1.setting == "rag")]
        srow.append([m, pct(p.exact.mean()), f"{p.recall.mean():.3f}", f"{r.recall.mean():.3f}",
                     f"{r.recall_in_context.mean():.3f}"])
    write_table(OUT / "settings.tex", ["Model", r"Memory: exact (\%)", "Memory: recall", "Retrieval: recall",
                                       "Retrieval: recall over retrieved"], srow, "lrrrr")

    # ---- T: scalar aggregation
    sc = e1[e1.answer_type == "scalar"]
    arow = []
    for m in ORDER:
        g = sc[sc.Model == m]
        wrong = g[~g.exact]
        arow.append([m, str(len(g)), pct(g.exact.mean()), str(int((wrong.signed_error < 0).sum())),
                     str(int((wrong.signed_error > 0).sum()))])
    write_table(OUT / "scalar.tex", ["Model", "Qs", r"Exact (\%)", "Under-counts", "Over-counts"], arow, "lrrrr")
    macro("nscalar", len(sc) // len(ORDER))

    # ---- F: reasoning tokens vs answer size
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m in ORDER:
        r = enum[enum.Model == m].groupby("n_gold").reasoning.median()
        ax.plot(r.index, r.values.clip(min=5), color=COLOR[m], marker=MARKER[m], label=m)
    ax.set(xscale="log", yscale="log", xlabel="Correct items in the answer", ylabel="Median reasoning tokens")
    ax.set_xticks(sorted(enum.n_gold.unique()), [str(v) for v in sorted(enum.n_gold.unique())])
    ax.minorticks_off()
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(OUT / "reasoning.pdf")
    plt.close(fig)
    rs = {}
    for m in ORDER:
        r = enum[enum.Model == m].groupby("n_gold").reasoning.median()
        rs[m] = (r.iloc[0], r.iloc[-1])
    for m, (a, b) in rs.items():
        key = KEY[m]
        macro(f"rsmall{key}", f"{a:,.0f}")
        macro(f"rlarge{key}", f"{b:,.0f}")
    fl = enum[enum.Model == "MiMo-V2.6 Flash"]
    shallow, deep = fl[fl.reasoning < 100], fl[fl.reasoning >= 100]
    macro("flashshallown", len(shallow))
    macro("flashshallowexact", pct(shallow.exact.mean()))
    macro("flashdeepn", len(deep))
    macro("flashdeepexact", pct(deep.exact.mean()))

    # ---- T: prompting (paired, 100-question subset)
    sub = df[(df.repeat == 0) & ~df.api_error & (df.answer_type != "scalar")]
    prow = []
    for m in ORDER:
        g = sub[sub.Model == m]
        d = g.pivot_table(index="id", columns="variant", values="recall", aggfunc="first")
        e = g.assign(x=g.exact.astype(float)).pivot_table(index="id", columns="variant", values="x",
                                                          aggfunc="first")
        cells = [m]
        for v in ["exhaustive", "rowwise"]:
            if v not in d.columns:
                cells += ["--", "--"]
                continue
            pair = e[["direct", v]].dropna()
            diff = (pair[v] - pair["direct"]).values
            lo, hi = boot_mean(diff)
            cells += [f"{len(pair)}", f"${100 * diff.mean():+.1f}$ [{100 * lo:+.1f}, {100 * hi:+.1f}]"]
        prow.append(cells)
    write_table(OUT / "prompting.tex", ["Model", "Qs", r"Exhaustive $-$ direct (pp)", "Qs",
                                        r"Row-by-row $-$ direct (pp)"], prow, "lrrrr")

    # ---- T: run-to-run agreement (second independent sample of the direct prompt)
    d = df[(df.variant == "direct") & ~df.api_error & (df.answer_type != "scalar")]
    rrow = []
    for m in ORDER:
        g = d[d.Model == m]
        a = g[g.repeat == 0].set_index("id").exact
        b = g[g.repeat == 1].set_index("id").exact
        j = pd.concat([a, b], axis=1, join="inner", keys=["a", "b"])
        if j.empty:
            rrow.append([m, "--", "--", "--"])
            continue
        same = (j.a == j.b)
        lo, hi = wilson(same.sum(), len(j))
        flip = (~same).sum()
        rrow.append([m, str(len(j)), f"{pct(same.mean())} {ci_pct(lo, hi)}", str(int(flip))])
    write_table(OUT / "repeats.tex", ["Model", "Qs", r"Same verdict (\%)", "Questions that flipped"], rrow,
                "lrrr")

    # ---- Failure cases (from report/failure_cases.py)
    # ---- F: miss rate by position of the correct row (appendix)
    pos_rows = []
    for _, r in enum.iterrows():
        q = qmap[r.id]
        t = bench.tables_for(q)[0]
        keys = [norm(k) for k in t.col(t.key)]
        missed = {x[0] for x in (r.missed_items if isinstance(r.missed_items, list) else [])}
        for gld in q.gold:
            i = keys.index(norm(gld[0]))
            pos_rows.append({"Model": r.Model, "decile": min(9, int(10 * i / len(keys))),
                             "missed": norm(gld[0]) in missed})
    pos = pd.DataFrame(pos_rows)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for m in ORDER:
        p = pos[pos.Model == m].groupby("decile").missed.mean()
        ax.plot((p.index + 0.5) * 10, 100 * p.values, color=COLOR[m], marker=MARKER[m], label=m)
    ax.set(xlabel="Position of the correct row in the table (%)", ylabel="Rows omitted (%)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.savefig(OUT / "position.pdf")
    plt.close(fig)

    # ---- Usage (every call, including retries and all experiments)
    u = load_usage(args.results)
    u = u[u.model.isin(SHORT)]
    ut = usage_table(u).reset_index()
    ut = ut[ut.model != "TOTAL"]
    ut["Model"] = ut.model.map(SHORT)
    agg = ut.groupby("Model").agg(calls=("calls", "sum"), inp=("input_tokens", "sum"), out=("output_tokens", "sum"),
                                  rea=("reasoning_tokens", "sum"), cost=("cost_usd", "sum")).reindex(ORDER)
    lat = u.assign(Model=u.model.map(SHORT)).groupby("Model").latency_s.median().reindex(ORDER)
    access = {"DeepSeek V4 Pro": "OpenCode Go", "DeepSeek V4 Flash": "OpenCode Go",
              "Claude Haiku 5.5": "Claude Code CLI", "MiMo-V2.6 Pro": "OpenRouter", "MiMo-V2.6 Flash": "OpenRouter"}
    write_table(OUT / "usage.tex", ["Model", "Access", "Calls", "Input (M)", "Output (M)",
                                    r"Reasoning (\%)", "Cost (USD)", "Latency (s)"],
                [[m, access[m], f"{int(agg.loc[m, 'calls']):,}", f"{agg.loc[m, 'inp'] / 1e6:.2f}",
                  f"{agg.loc[m, 'out'] / 1e6:.2f}", pct(agg.loc[m, "rea"] / agg.loc[m, "out"]),
                  f"{agg.loc[m, 'cost']:.2f}", f"{lat.loc[m]:.1f}"] for m in ORDER], "llrrrrrr")
    share = agg.rea / agg.out
    macro("reasonshare", f"{pct(share.min())}--{pct(share.max())}")
    macro("totalcalls", f"{int(agg.calls.sum()):,}")
    macro("totalcost", f"{agg.cost.sum():.2f}")

    # ---- Run facts
    mans = [json.loads(p.read_text()) for p in sorted((args.results / "manifests").glob("*.json"))]
    macro("benchsha", mans[0]["benchmark"]["sha256"][:12])
    macro("gitcommit", (mans[0]["git"]["commit"] or "?")[:7])
    macro("claudecli", next((m["claude_cli_version"] for m in mans if m["claude_cli_version"]), "?")
          .replace(" (Claude Code)", ""))
    macro("ntables", f"{len(bench.tables):,}")
    macro("nrecords", f"{sum(len(t.rows) for t in bench.tables.values()):,}")

    (OUT / "numbers.tex").write_text(
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in MACROS.items()) + "\n")
    print(pd.DataFrame(stats).T[["exact", "recall", "precision", "omit", "commit"]].round(3).to_string())
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
