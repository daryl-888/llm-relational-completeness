"""Generate the paper's tables, figures and in-text numbers from a results directory.

  .venv/bin/python report/make_assets.py [--results results/full_v0]

Writes report/generated/*.tex (tables, examples, and numbers.tex with \\newcommand macros used in the text)
and report/generated/*.pdf (figures). Statistics (relcomplete.stats) follow Miller (2024), "Adding Error Bars
to Evals": Wilson intervals for proportions, question-level bootstrap for means and paired differences, and
cluster bootstraps when a question contributes several rows.

Conventions:
  * Every answer has an outcome: answered, truncated (token limit), refused (safety filter), api_error, or
    unparseable. Exact-set accuracy counts every non-answer as incorrect; error types (missing, wrong, ...)
    are counted only on valid answers, so a truncated answer is not mistaken for omitted items.
  * E1 = repeat 0, direct prompt. E2 = exhaustive / rowwise prompts on the subset. E3 = repeat 1.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from relcomplete.report import load_results, load_usage  # noqa: E402
from relcomplete.schema import Benchmark  # noqa: E402
from relcomplete.score import norm, score  # noqa: E402
from relcomplete.stats import boot_mean, logit, logit_boot, wilson  # noqa: E402
from relcomplete.synth import ENTITY  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated"
RNG = np.random.default_rng(0)

# Models in the paper and the result files that represent them. MiMo was run to completion on OpenRouter; its
# partial OpenCode Go runs and the stopped Haiku 4.5 run (90 questions) are excluded. ALSO maps a model's runs
# on a second provider onto the same row; those runs only contribute to E2 (see results/full_v0/README.md).
SHORT = {"opencode-go/deepseek-v4-pro": "DeepSeek V4 Pro", "opencode-go/deepseek-v4-flash": "DeepSeek V4 Flash",
         "claude-cli/claude-haiku-5-5": "Claude Haiku 5.5", "xiaomi/mimo-v2.6-pro": "MiMo-V2.6 Pro",
         "xiaomi/mimo-v2.6-flash": "MiMo-V2.6 Flash"}
ALSO = {"openrouter/deepseek/deepseek-v4-pro": "DeepSeek V4 Pro"}
ORDER = list(SHORT.values())
KEY = dict(zip(ORDER, ["DSPro", "DSFlash", "Haiku", "MiMoPro", "MiMoFlash"]))  # letters only, for \macros
ACCESS = {"DeepSeek V4 Pro": "OpenCode Go", "DeepSeek V4 Flash": "OpenCode Go", "Claude Haiku 5.5": "Claude Code CLI",
          "MiMo-V2.6 Pro": "OpenRouter", "MiMo-V2.6 Flash": "OpenRouter"}
# Validated categorical order from the dataviz reference palette (slots 1-5), plus a marker per model so
# identity never depends on colour alone (print, colour-vision deficiency).
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKER = dict(zip(ORDER, ["o", "s", "^", "D", "v"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
OPS = [("enumeration", "Enumeration"), ("selection", "Selection"), ("projection", "Projection"),
       ("join", "JOIN"), ("aggregation", "Aggregation"), ("nested", "Nested query"),
       ("setop", "Set operation")]
FMTS = [("markdown", "Markdown"), ("csv", "CSV"), ("json", "JSON Lines"), ("nl", "English sentences")]
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

MACROS: dict[str, str] = {}


def macro(name: str, value) -> None:
    MACROS[name] = str(value)


def pct(v: float, d: int = 0) -> str:
    return "--" if pd.isna(v) else f"{100 * v:.{d}f}"


def ci_pct(lo: float, hi: float, d: int = 0) -> str:
    return f"[{pct(lo, d)}, {pct(hi, d)}]"


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("$", r"\$"))


def title_case(s: str) -> str:
    small = {"of", "the", "a", "an", "and", "in", "on"}
    words = s.split()
    return " ".join(w if (i and w in small) else w[:1].upper() + w[1:] for i, w in enumerate(words))


def write_table(path: Path, header: list[str], rows: list[list[str]], spec: str, midrules: tuple = ()) -> None:
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def or_cell(coefs: np.ndarray, boots: np.ndarray, j: int, failures: int) -> str:
    """Odds ratio per doubling with a bootstrap interval, or n.e. (not estimable) when failures are too few."""
    lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
    if failures < 5 or np.exp(hi) > 20:
        return "n.e."
    return f"{np.exp(coefs[j]):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "full_v0")
    ap.add_argument("--bench", type=Path, default=ROOT / "data" / "benchmark_v0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = Benchmark.load(args.bench)
    qmap = {q.id: q for q in bench.questions}

    df = load_results(args.results, bench)
    df["Model"] = df.model.map({**SHORT, **ALSO})
    df = df[df.Model.notna()].copy()
    df["exact"] = df.exact.astype(bool)
    df["state"] = np.where((df.outcome == "answered") & ~df.parse_ok.astype(bool), "unparseable", df.outcome)
    df["valid"] = df.state == "answered"
    e1 = df[(df.repeat == 0) & (df.variant == "direct") & df.model.isin(SHORT)]
    ctx = e1[(e1.setting == "context") & (e1.answer_type != "scalar")].copy()  # the core: 365 questions
    ctx["omission"] = ctx.valid & (ctx.missing.fillna(0) + ctx.near_miss.fillna(0) > 0)
    ctx["commission"] = ctx.valid & (ctx.wrong.fillna(0) + ctx.made_up.fillna(0) + ctx.duplicates.fillna(0) > 0)
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
                         VARIANTS[(op, setting)], f"{min(g)}--{max(g)}",
                         f"{np.median([q.meta['context_chars'] for q in qs]) / 4000:.1f}"])
    write_table(OUT / "composition.tex", ["Slice", "Qs", "Variants", "Answer size", r"Context (k tok.)"],
                comp, "lrlrr", midrules=(3,))

    # ---- Main results with intervals and paired difference from the best model
    stats = {}
    for m in ORDER:
        g = ctx[ctx.Model == m]
        n, k = len(g), int(g.exact.sum())
        stats[m] = {"exact": k / n, "exact_ci": wilson(k, n), "recall": g.recall.mean(),
                    "recall_ci": boot_mean(g.recall, RNG), "precision": g[g.valid].precision.mean(),
                    "omit": g.omission.mean(), "commit": g.commission.mean(), "n_invalid": int((~g.valid).sum())}
    best = max(ORDER, key=lambda m: stats[m]["exact"])
    piv = ctx.pivot_table(index="id", columns="Model", values="exact", aggfunc="first").astype(float)
    rows = []
    for m in ORDER:
        s = stats[m]
        if m == best:
            diff = "(reference)"
        else:
            dd = (piv[m] - piv[best]).dropna().values
            lo, hi = boot_mean(dd, RNG)
            diff = f"${100 * dd.mean():+.1f}$ [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        rows.append([m, f"{pct(s['exact'])} {ci_pct(*s['exact_ci'])}", diff,
                     f"{s['recall']:.3f} [{s['recall_ci'][0]:.3f}, {s['recall_ci'][1]:.3f}]",
                     f"{s['precision']:.3f}", pct(s["omit"]), pct(s["commit"]), str(s["n_invalid"])])
        macro(f"exact{KEY[m]}", pct(s["exact"]))
        macro(f"recall{KEY[m]}", f"{s['recall']:.3f}")
        macro(f"omit{KEY[m]}", pct(s["omit"]))
        macro(f"commit{KEY[m]}", pct(s["commit"]))
    write_table(OUT / "main.tex", ["Model", r"Exact-set acc.\ (\%)", r"$\Delta$ vs.\ best (pp)", "Mean recall",
                                   "Mean prec.", r"Omit.\ (\%)", r"Comm.\ (\%)", "Invalid"], rows, "lrrrrrrr")
    macro("bestmodel", best)
    ex, rc, om = ([stats[m][k] for m in ORDER] for k in ("exact", "recall", "omit"))
    macro("exactmin", pct(min(ex)))
    macro("exactmax", pct(max(ex)))
    macro("recallmin", f"{min(rc):.3f}")
    macro("recallmax", f"{max(rc):.3f}")
    macro("omitmin", pct(min(om)))
    macro("omitmax", pct(max(om)))

    # ---- Error counts by type, valid answers only
    valid_ctx = ctx[ctx.valid]
    errs = valid_ctx.groupby("Model")[["missing", "near_miss", "wrong", "made_up", "duplicates"]].sum()
    errs = errs.reindex(ORDER).astype(int)
    write_table(OUT / "error_counts.tex", ["Model", "Missing", "Near-miss", "Wrong", "Made-up", "Duplicate",
                                           "Invalid answers"],
                [[m] + [f"{v:,}" for v in errs.loc[m]] + [str(stats[m]["n_invalid"])] for m in ORDER], "lrrrrrr")
    others = errs.drop(index="MiMo-V2.6 Flash")
    macro("otherswrong", int(others.wrong.sum()))
    macro("othersmadeup", int(others.made_up.sum()))
    macro("flashmadeup", f"{int(errs.loc['MiMo-V2.6 Flash', 'made_up']):,}")
    worst = valid_ctx[valid_ctx.Model == "MiMo-V2.6 Flash"].sort_values("made_up", ascending=False).iloc[0]
    macro("flashworstmade", f"{int(worst.made_up):,}")
    macro("flashworstgold", int(worst.n_gold))
    inc = valid_ctx[valid_ctx.missing + valid_ctx.near_miss > 0]
    macro("medianmissed", f"{inc.missing.median():.0f}")
    macro("smallmissshare", pct(((inc.missing + inc.near_miss) / inc.n_gold <= 0.10).mean()))

    # ---- Invalid answers (all 500 E1 questions)
    irows = []
    for m in ORDER:
        g = e1[e1.Model == m]
        c = g.state.value_counts()
        irows.append([m, str(len(g)), str(c.get("truncated", 0)), str(c.get("unparseable", 0)),
                      str(c.get("refused", 0)), str(c.get("api_error", 0))])
    write_table(OUT / "invalid.tex", ["Model", "Qs", "Truncated", "Unparseable", "Refused", "API error"], irows,
                "lrrrrr")
    macro("ninvalid", int((e1.state != "answered").sum()))
    macro("nrefused", int((e1.state == "refused").sum()))

    # ---- Figure: error profile
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

    # ---- Figure: completeness vs answer size and table size (in-context enumeration), Wilson intervals
    enum = ctx[ctx.op == "enumeration"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    for ax, x, lab in zip(axes, ["n_gold", "n_rows"], ["Correct items in the answer", "Records in the table"]):
        xs = sorted(enum[x].unique())
        for i, m in enumerate(ORDER):
            g = enum[enum.Model == m]
            pts = [(g[g[x] == v].exact.sum(), (g[x] == v).sum()) for v in xs]
            p = np.array([k / n for k, n in pts])
            lo, hi = map(np.array, zip(*[wilson(k, n) for k, n in pts]))
            off = np.array(xs) * (1 + 0.035 * (i - 2))  # dodge on the log axis
            ax.errorbar(off, 100 * p, yerr=[100 * (p - lo), 100 * (hi - p)], color=COLOR[m], marker=MARKER[m],
                        capsize=0, elinewidth=0.8, label=m)
        ax.set(xscale="log", xlabel=lab)
        ax.set_xticks(xs, [str(v) for v in xs])
        ax.minorticks_off()
    axes[0].set_ylabel("Exact-set accuracy (%)")
    axes[0].set_ylim(0, 104)
    axes[0].legend(loc="lower left", fontsize=7, handlelength=1.6, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "size_curves.pdf")
    plt.close(fig)
    n_per = enum.groupby(["Model", "n_gold"]).size()
    macro("nperanswersize", f"{n_per.min()}--{n_per.max()}")
    per_q = enum.groupby("id")[["n_gold", "n_rows"]].first()
    macro("sizecorr", f"{np.corrcoef(np.log(per_q.n_gold), np.log(per_q.n_rows))[0, 1]:.2f}")

    # ---- Trend: logistic regression on log2(answer size), log2(table size) and format, per model and pooled
    enum["lg"] = np.log2(enum.n_gold)
    enum["ln"] = np.log2(enum.n_rows)
    for f in ["csv", "json", "nl"]:
        enum[f] = (enum.fmt == f).astype(float)
    cols = ["lg", "ln", "csv", "json", "nl"]
    trow = []
    for m in ORDER:
        g = enum[enum.Model == m]
        X, yv = g[cols].values, g.exact.values.astype(float)
        w = logit(X, yv)
        bs = logit_boot(X, yv, RNG)
        fails = int((yv == 0).sum())
        small, large = g[g.n_gold <= 10].exact.mean(), g[g.n_gold >= 50].exact.mean()
        trow.append([m, pct(small), pct(large), str(fails), or_cell(w, bs, 1, fails), or_cell(w, bs, 2, fails)])
        macro(f"small{KEY[m]}", pct(small))
        macro(f"large{KEY[m]}", pct(large))
    dums = pd.get_dummies(enum.Model)[ORDER[1:]].values.astype(float)
    Xp, yp = np.column_stack([enum[cols].values, dums]), enum.exact.values.astype(float)
    wp = logit(Xp, yp)
    bp = logit_boot(Xp, yp, RNG, clusters=enum.id.values)
    pooled = [np.exp(wp[j]) for j in (1, 2)]
    pci = [np.exp(np.percentile(bp[:, j], [2.5, 97.5])) for j in (1, 2)]
    trow.append([r"\textit{All models (pooled)}", pct(enum[enum.n_gold <= 10].exact.mean()),
                 pct(enum[enum.n_gold >= 50].exact.mean()), str(int((yp == 0).sum())),
                 f"{pooled[0]:.2f} [{pci[0][0]:.2f}, {pci[0][1]:.2f}]",
                 f"{pooled[1]:.2f} [{pci[1][0]:.2f}, {pci[1][1]:.2f}]"])
    write_table(OUT / "trend.tex", ["Model", r"$\leq$10 items (\%)", r"$\geq$50 items (\%)", "Failures",
                                    r"OR: answer size [95\% CI]", r"OR: table size [95\% CI]"], trow, "lrrrrr",
                midrules=(len(ORDER),))
    macro("orpooledanswer", f"{pooled[0]:.2f}")
    macro("orpooledanswerci", f"{pci[0][0]:.2f}--{pci[0][1]:.2f}")
    macro("orpooledtable", f"{pooled[1]:.2f}")
    macro("orpooledtableci", f"{pci[1][0]:.2f}--{pci[1][1]:.2f}")

    # ---- Exact-set accuracy by operation, and by input format (in context)
    orow = []
    for op, label in OPS:
        g = ctx[ctx.op == op]
        if g.empty:
            continue
        orow.append([label, str(len(g) // len(ORDER))] + [pct(g[g.Model == m].exact.mean()) for m in ORDER])
    write_table(OUT / "by_operation.tex", ["Operation", "Qs"] + [m.replace(" ", r"\ ") for m in ORDER], orow,
                "lr" + "r" * len(ORDER))
    frow = []
    for f, label in FMTS:
        g = ctx[ctx.fmt == f]
        frow.append([label, str(len(g) // len(ORDER))] + [pct(g[g.Model == m].exact.mean()) for m in ORDER])
    write_table(OUT / "by_format.tex", ["Input format", "Qs"] + [m.replace(" ", r"\ ") for m in ORDER], frow,
                "lr" + "r" * len(ORDER))

    # ---- Memory-only and retrieval settings
    srow = []
    for m in ORDER:
        p = e1[(e1.Model == m) & (e1.setting == "parametric")]
        r = e1[(e1.Model == m) & (e1.setting == "rag")]
        srow.append([m, pct(p.exact.mean()), f"{p.recall.mean():.3f}", f"{r.recall.mean():.3f}",
                     f"{r.recall_in_context.mean():.3f}"])
    write_table(OUT / "settings.tex", ["Model", r"Memory: exact (\%)", "Memory: recall", "Retrieval: recall",
                                       "Retrieval: recall over retrieved"], srow, "lrrrr")

    # ---- Scalar aggregation
    sc = e1[e1.answer_type == "scalar"]
    arow = []
    for m in ORDER:
        g = sc[sc.Model == m]
        wrong = g[~g.exact]
        arow.append([m, str(len(g)), pct(g.exact.mean()), str(int((wrong.signed_error < 0).sum())),
                     str(int((wrong.signed_error > 0).sum()))])
    write_table(OUT / "scalar.tex", ["Model", "Qs", r"Exact (\%)", "Under-counts", "Over-counts"], arow, "lrrrr")
    macro("nscalar", len(sc) // len(ORDER))
    nonds = sc[~sc.Model.str.startswith("DeepSeek") & ~sc.exact]
    macro("scalarerrors", len(nonds))
    macro("scalarunder", int((nonds.signed_error < 0).sum()))

    # ---- Figure: reasoning tokens vs answer size
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m in ORDER:
        r = enum[enum.Model == m].groupby("n_gold").reasoning.median()
        ax.plot(r.index, r.values.clip(min=5), color=COLOR[m], marker=MARKER[m], label=m)
        macro(f"rsmall{KEY[m]}", f"{r.iloc[0]:,.0f}")
        macro(f"rlarge{KEY[m]}", f"{r.iloc[-1]:,.0f}")
    ax.set(xscale="log", yscale="log", xlabel="Correct items in the answer", ylabel="Median reasoning tokens")
    ax.set_xticks(sorted(enum.n_gold.unique()), [str(v) for v in sorted(enum.n_gold.unique())])
    ax.minorticks_off()
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(OUT / "reasoning.pdf")
    plt.close(fig)
    fl = enum[enum.Model == "MiMo-V2.6 Flash"]
    shallow, deep = fl[fl.reasoning < 100], fl[fl.reasoning >= 100]
    macro("flashshallown", len(shallow))
    macro("flashshallowexact", pct(shallow.exact.mean()))
    macro("flashdeepexact", pct(deep.exact.mean()))

    # ---- Prompt variants (E2): paired change from the direct prompt on the same questions
    sub = df[(df.repeat == 0) & (df.state != "api_error") & (df.answer_type != "scalar")]
    prow, e2_models = [], set()
    for m in ORDER:
        g = sub[sub.Model == m]
        # A second provider's E2 is compared with that same provider's direct run, so the pairing never
        # mixes providers.
        other_provider = g[g.variant != "direct"].model.isin(ALSO).any()
        g = g[g.model.isin(ALSO)] if other_provider else g[g.model.isin(SHORT)]
        e = g.assign(x=g.exact.astype(float)).pivot_table(index="id", columns="variant", values="x",
                                                          aggfunc="first")
        cells = [m + (r"$^\dagger$" if other_provider else "")]
        for v in ["exhaustive", "rowwise"]:
            if v not in e.columns:
                cells += ["--", "--"]
                continue
            pair = e[["direct", v]].dropna()
            dd = (pair[v] - pair["direct"]).values
            lo, hi = boot_mean(dd, RNG)
            cells += [f"{len(pair)}", f"${100 * dd.mean():+.1f}$ [{100 * lo:+.1f}, {100 * hi:+.1f}]"]
            e2_models.add(m)
        prow.append(cells)
    write_table(OUT / "prompting.tex", ["Model", "Qs", r"Exhaustive $-$ direct (pp)", "Qs",
                                        r"Row-by-row $-$ direct (pp)"], prow, "lrrrr")
    missing_e2 = [m for m in ORDER if m not in e2_models]
    macro("etwonote", (f"{', '.join(missing_e2)}'s E2 runs are pending (provider usage limits)." if missing_e2
                       else r"$^\dagger$E2 run through a second provider (OpenRouter) for the same model."))

    # ---- Run-to-run agreement (E1 vs E3)
    d = df[(df.variant == "direct") & (df.state != "api_error") & (df.answer_type != "scalar") & df.model.isin(SHORT)]
    rrow = []
    for m in ORDER:
        g = d[d.Model == m]
        j = pd.concat([g[g.repeat == 0].set_index("id").exact, g[g.repeat == 1].set_index("id").exact],
                      axis=1, join="inner", keys=["a", "b"])
        if j.empty:
            rrow.append([m, "--", "--", "--"])
            continue
        same = j.a == j.b
        rrow.append([m, str(len(j)), f"{pct(same.mean())} {ci_pct(*wilson(same.sum(), len(j)))}",
                     str(int((~same).sum()))])
    write_table(OUT / "repeats.tex", ["Model", "Qs", r"Same verdict (\%)", "Questions that flipped"], rrow, "lrrr")

    # ---- Worked examples, taken from the data
    examples = []
    ds = ctx[(ctx.Model == "DeepSeek V4 Pro") & (ctx.id == "enum-ctx-029")].iloc[0]
    q = qmap["enum-ctx-029"]
    t = bench.tables_for(q)[0]
    keys = [norm(k) for k in t.col(t.key)]
    names = {norm(r[0]): r[0] for r in t.rows}
    missed = [x[0] for x in ds.missed_items]
    rows_at = [keys.index(x) + 1 for x in missed]
    reruns = []
    for k in (1, 2, 3):  # the same question re-run three times (results/failure_cases, OpenRouter)
        path = ROOT / "results" / "failure_cases" / f"openrouter_deepseek_deepseek-v4-pro__direct__r{k}.jsonl"
        if path.exists():
            for line in path.open():
                r = json.loads(line)
                if r["id"] == q.id and r.get("text"):
                    reruns.append(score(q, bench, r["text"]))
    rr = ""
    if reruns:
        n_missed = [m["missing"] + m["near_miss"] for m in reruns]
        common = set.intersection(*({x[0] for x in m["missed_items"]} for m in reruns))
        n_added = [m["wrong"] + m["made_up"] for m in reruns]
        rr = (f" In three re-runs the answer was incomplete every time, omitting {min(n_missed)}--{max(n_missed)} "
              f"names, a different set each time"
              + (f" ({', '.join(tex_escape(names[x]) for x in sorted(common))} in all three)" if common else "")
              + (f", and {sum(1 for v in n_added if v)} of the three re-runs also added an incorrect name"
                 if any(n_added) else "") + ".")
    examples.append(
        rf"\textbf{{Omission (DeepSeek V4 Pro, \texttt{{enum-ctx-029}}).}} Prompt: {len(t.rows)} employee records "
        rf"as CSV ($\approx${q.meta['context_chars'] / 4000:.0f}k tokens), then \emph{{``{tex_escape(q.question)}''}} "
        rf"Gold answer: {len(q.gold)} names. The model returned {int(ds.n_pred)} names, all correct, and omitted "
        + ", ".join(f"{tex_escape(names[x])} (row {p} of {len(t.rows)})" for x, p in zip(missed, rows_at))
        + "." + rr)
    fw = ctx[(ctx.Model == "MiMo-V2.6 Flash") & (ctx.id == worst.id)].iloc[0]
    qf = qmap[worst.id]
    madeup = [x[0] for x in fw.extra_items][:4]
    examples.append(
        rf"\textbf{{Fabrication (MiMo-V2.6 Flash, \texttt{{{worst.id}}}).}} Prompt: {qf.meta['n_rows']:,} "
        rf"{ENTITY[qf.meta['domain']][0]} records; \emph{{``{tex_escape(qf.question)}''}} Gold answer: {len(qf.gold)} items. "
        rf"The model returned {int(fw.n_pred):,} distinct items (plus {int(fw.duplicates)} repeats), of which "
        rf"{int(fw.tp)} were correct and {int(fw.made_up)} do not exist in the data, for example "
        + ", ".join("``" + tex_escape(title_case(x)) + "''" for x in madeup) + ".")
    (OUT / "examples.tex").write_text("\n\n".join(examples) + "\n")

    # ---- Figure: miss rate by position (appendix)
    pos_rows = []
    for _, r in enum[enum.valid].iterrows():
        q = qmap[r.id]
        t = bench.tables_for(q)[0]
        keys = [norm(k) for k in t.col(t.key)]
        missed = {x[0] for x in r.missed_items}
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

    # ---- Usage: successful calls and failed attempts, all experiments
    u = load_usage(args.results)
    u["Model"] = u.model.map({**SHORT, **ALSO})
    u = u[u.Model.notna()]
    ok = u[u.output > 0]
    urow = []
    for m in ORDER:
        g, a = ok[ok.Model == m], u[u.Model == m]
        urow.append([m, ACCESS[m], f"{len(g):,}", f"{len(a) - len(g):,}", f"{g.input.sum() / 1e6:.2f}",
                     f"{g.output.sum() / 1e6:.2f}", pct(g.reasoning.sum() / g.output.sum()),
                     f"{g.cost_usd.sum():.2f}", f"{g.latency_s.median():.1f}"])
    write_table(OUT / "usage.tex", ["Model", "Access", "Calls", "Failed", "Input (M)", "Output (M)",
                                    r"Reasoning (\%)", "Cost (USD)", "Latency (s)"], urow, "llrrrrrrr")
    share = ok.groupby("Model").reasoning.sum() / ok.groupby("Model").output.sum()
    macro("reasonshare", f"{pct(share.min())}--{pct(share.max())}")
    macro("totalcalls", f"{len(ok):,}")
    macro("failedcalls", f"{len(u) - len(ok):,}")

    # ---- Run facts
    mans = [json.loads(p.read_text()) for p in sorted((args.results / "manifests").glob("*.json"))]
    macro("benchsha", mans[0]["benchmark"]["sha256"][:12])
    macro("gitcommit", (mans[0]["git"]["commit"] or "?")[:7])
    macro("claudecli", next((m["claude_cli_version"] for m in mans if m["claude_cli_version"]), "?")
          .replace(" (Claude Code)", ""))
    macro("ntables", f"{len(bench.tables):,}")
    macro("nrecords", f"{sum(len(t.rows) for t in bench.tables.values()):,}")

    (OUT / "numbers.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in MACROS.items()) + "\n")
    print(pd.DataFrame(stats).T[["exact", "recall", "precision", "omit", "commit", "n_invalid"]].to_string())
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
