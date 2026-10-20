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
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from relcomplete.gen.common import key_label  # noqa: E402
from relcomplete.report import load_results, load_usage, outcome, repeat_agreement, usage_summary  # noqa: E402
from relcomplete.schema import Benchmark  # noqa: E402
from relcomplete.score import norm, score  # noqa: E402
from relcomplete.stats import boot_mean, logit, logit_boot, wilson  # noqa: E402
from relcomplete.synth import ENTITY  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated"
DOCS = [ROOT / "report" / "report.tex", ROOT / "report" / "memo.tex"]  # numbers.tex keeps only macros these use

# Models in the paper and the result files that represent them. MiMo was run to completion on OpenRouter; its
# partial OpenCode Go runs and the stopped Haiku 4.5 run (90 questions) are excluded. ALSO maps a model's runs
# on a second provider onto the same row; those runs only contribute to E2 (see results/full_v0/README.md).
SHORT = {"opencode-go/deepseek-v4-pro": "DeepSeek V4 Pro", "opencode-go/deepseek-v4-flash": "DeepSeek V4 Flash",
         "claude-cli/claude-haiku-5-5": "Claude Haiku 5.5", "xiaomi/mimo-v2.6-pro": "MiMo-V2.6 Pro",
         "xiaomi/mimo-v2.6-flash": "MiMo-V2.6 Flash"}
ALSO = {"openrouter/deepseek/deepseek-v4-pro": "DeepSeek V4 Pro"}
ALSO_ACCESS = {"openrouter/deepseek/deepseek-v4-pro": "OpenRouter"}
# Worked examples in Box 1: (model, question id, re-run file stem in results/failure_cases). The fabrication
# example is the MiMo-V2.6 Flash answer with the most made-up items.
OMISSION_EXAMPLE = ("DeepSeek V4 Pro", "enum-ctx-029", "openrouter_deepseek_deepseek-v4-pro__direct")
FABRICATION_MODEL = "MiMo-V2.6 Flash"
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
FMT_KEY = {"markdown": "Md", "csv": "Csv", "json": "Json", "nl": "Nl"}
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


_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(s: str) -> str:
    """Escape text (including model output) for LaTeX, one character at a time so replacements never stack."""
    return "".join(_TEX.get(ch, ch) for ch in str(s))


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
    MACROS.clear()  # main() may run more than once in a process; never carry numbers over
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    bench = Benchmark.load(args.bench)
    qmap = {q.id: q for q in bench.questions}

    df = load_results(args.results, bench)
    df["Model"] = df.model.map({**SHORT, **ALSO})
    df = df[df.Model.notna()].copy()
    df["exact"] = df.exact.astype(bool)  # state and valid come from load_results
    e1 = df[(df.repeat == 0) & (df.variant == "direct") & df.model.isin(SHORT)]
    ctx = e1[(e1.setting == "context") & (e1.answer_type != "scalar")].copy()  # the core: 365 questions
    ctx["omission"] = ctx.valid & (ctx.missing.fillna(0) + ctx.near_miss.fillna(0) > 0)
    ctx["commission"] = ctx.valid & (ctx.wrong.fillna(0) + ctx.made_up.fillna(0) + ctx.duplicates.fillna(0) > 0)
    macro("nctx", ctx.groupby("Model").size().iloc[0])
    macro("neone", f"{len(e1):,}")

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
                    "recall_ci": boot_mean(g.recall, rng), "precision": g[g.valid].precision.mean(),
                    "omit": g.omission.mean(), "commit": g.commission.mean(), "n_invalid": int((~g.valid).sum())}
    best = max(ORDER, key=lambda m: stats[m]["exact"])
    piv = ctx.pivot_table(index="id", columns="Model", values="exact", aggfunc="first").astype(float)
    rows, below, tied = [], [], []
    for m in ORDER:
        s = stats[m]
        if m == best:
            diff = "(reference)"
        else:
            dd = (piv[m] - piv[best]).dropna().values
            lo, hi = boot_mean(dd, rng)
            diff = f"${100 * dd.mean():+.1f}$ [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            (below.append(-100 * dd.mean()) if hi < 0 else tied.append(m))
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
    macro("tiedwithbest", " and ".join(tied) if tied else "no other model")
    macro("nsigbelow", WORDS.get(len(below), str(len(below))))
    macro("deltamin", f"{min(below):.0f}" if below else "--")
    macro("deltamax", f"{max(below):.0f}" if below else "--")
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
    macro("medianmissed", f"{(inc.missing + inc.near_miss).median():.0f}")
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
        bs = logit_boot(X, yv, rng)
        fails = int((yv == 0).sum())
        small, large = g[g.n_gold <= 10].exact.mean(), g[g.n_gold >= 50].exact.mean()
        trow.append([m, pct(small), pct(large), str(fails), or_cell(w, bs, 1, fails), or_cell(w, bs, 2, fails)])
        macro(f"small{KEY[m]}", pct(small))
        macro(f"large{KEY[m]}", pct(large))
        macro(f"fail{KEY[m]}", fails)
    macro("nenum", enum.groupby("Model").size().iloc[0])
    macro("smallmin", pct(min(enum[(enum.Model == m) & (enum.n_gold <= 10)].exact.mean() for m in ORDER)))
    dums = pd.get_dummies(enum.Model)[ORDER[1:]].values.astype(float)
    Xp, yp = np.column_stack([enum[cols].values, dums]), enum.exact.values.astype(float)
    wp = logit(Xp, yp)
    bp = logit_boot(Xp, yp, rng, clusters=enum.id.values)
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
        for m in ORDER:
            macro(f"fmt{KEY[m]}{FMT_KEY[f]}", pct(g[g.Model == m].exact.mean()))
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
    rag_recall = sorted({r[3] for r in srow})
    macro("ragrecall", f"{rag_recall[0]} for all models" if len(rag_recall) == 1
          else f"{rag_recall[0]}--{rag_recall[-1]}")

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
            if v not in e.columns or "direct" not in e.columns:  # nothing to pair with
                cells += ["--", "--"]
                continue
            pair = e[["direct", v]].dropna()
            dd = (pair[v] - pair["direct"]).values
            lo, hi = boot_mean(dd, rng)
            cells += [f"{len(pair)}", f"${100 * dd.mean():+.1f}$ [{100 * lo:+.1f}, {100 * hi:+.1f}]"]
            e2_models.add(m)
        prow.append(cells)
    write_table(OUT / "prompting.tex", ["Model", "Qs", r"Exhaustive $-$ direct (pp)", "Qs",
                                        r"Row-by-row $-$ direct (pp)"], prow, "lrrrr")
    missing_e2 = [m for m in ORDER if m not in e2_models]
    macro("etwonote", (f"{', '.join(missing_e2)}'s E2 runs are pending (provider usage limits)." if missing_e2
                       else r"$^\dagger$E2 run through a second provider (OpenRouter) for the same model."))

    # ---- Run-to-run agreement (E1 vs E3)
    ag = repeat_agreement(df[(df.answer_type != "scalar") & df.model.isin(SHORT)])
    rrow, agree = [], []
    for m in ORDER:
        mid = next(k for k, v in SHORT.items() if v == m)
        if ag.empty or mid not in ag.index:
            rrow.append([m, "--", "--", "--"])
            continue
        n = int(ag.loc[mid, "n"])
        same = int(round(ag.loc[mid, "exact_agreement"] * n))
        agree.append(same / n)
        rrow.append([m, str(n), f"{pct(same / n)} {ci_pct(*wilson(same, n))}", str(n - same)])
    write_table(OUT / "repeats.tex", ["Model", "Qs", r"Same verdict (\%)", "Questions that flipped"], rrow, "lrrr")
    macro("agreemin", pct(min(agree)))
    macro("agreemax", pct(max(agree)))
    macro("flipmin", pct(1 - max(agree)))
    macro("flipmax", pct(1 - min(agree)))

    # ---- Worked examples (Box 1), stating only what the data shows
    examples = []
    ex_model, ex_id, ex_stem = OMISSION_EXAMPLE
    ds_rows = ctx[(ctx.Model == ex_model) & (ctx.id == ex_id) & ctx.valid]
    if len(ds_rows):
        ds, q = ds_rows.iloc[0], qmap[ex_id]
        t = bench.tables_for(q)[0]
        row_of = {norm(k): i + 1 for i, k in enumerate(t.col(t.key))}
        names = {norm(k): k for k in t.col(t.key)}
        noun = key_label(t.kind)[1]
        missed = [x[0] for x in ds.missed_items]
        n_extra = int(ds.wrong + ds.made_up + ds.duplicates)
        got = (f"{int(ds.n_pred)} {noun}, all correct," if n_extra == 0
               else f"{int(ds.n_pred)} {noun}, {int(ds.tp)} of them correct,")
        reruns = []  # one answer per re-run file: its last record for the question, as load_results counts it
        for path in sorted((ROOT / "results" / "failure_cases").glob(f"{ex_stem}__r*.jsonl")):
            last = None
            for line in path.open():
                r = json.loads(line)
                if r["id"] == ex_id:
                    last = r
            if last is not None and outcome(last) == "answered":
                sc_ = score(q, bench, last["text"])
                if sc_["parse_ok"]:
                    reruns.append(sc_)
        rr = ""
        if reruns:
            n = len(reruns)
            inc_runs = [x for x in reruns if x["missing"] + x["near_miss"] > 0]
            rr = f" In {WORDS.get(n, n)} re-runs the answer was incomplete " + (
                "every time" if len(inc_runs) == n else f"{WORDS.get(len(inc_runs), len(inc_runs))} times")
            if inc_runs:
                counts = sorted(x["missing"] + x["near_miss"] for x in inc_runs)
                rr += f", omitting {counts[0]}" + (f"--{counts[-1]}" if counts[-1] != counts[0] else "") + f" {noun}"
                sets = [frozenset(x[0] for x in r_["missed_items"]) for r_ in inc_runs]
                if len(sets) > 1 and len(set(sets)) == len(sets):
                    rr += ", a different set each time"
                common = frozenset.intersection(*sets) if len(sets) > 1 else frozenset()
                if common:
                    rr += (f" ({', '.join(tex_escape(names[x]) for x in sorted(common))} in all "
                           f"{WORDS.get(len(sets), len(sets))})")
            added = sum(1 for x in reruns if x["wrong"] + x["made_up"] > 0)
            if added:
                rr += f", and {WORDS.get(added, added)} of the {WORDS.get(n, n)} re-runs also added incorrect {noun}"
            rr += "."
        examples.append(
            rf"\textbf{{Omission ({ex_model}, \texttt{{{ex_id}}}).}} Prompt: {len(t.rows):,} {ENTITY[t.kind][0]} "
            rf"records as {dict(FMTS)[q.fmt]} ($\approx${q.meta['context_chars'] / 4000:.0f}k tokens), then "
            rf"\emph{{``{tex_escape(q.question)}''}} Gold answer: {len(q.gold)} {noun}. The model returned {got} "
            + "and omitted "
            + ", ".join(f"{tex_escape(names[x])} (row {row_of[x]} of {len(t.rows):,})" for x in missed) + "." + rr)
    fl_valid = ctx[(ctx.Model == FABRICATION_MODEL) & ctx.valid]
    if len(fl_valid) and fl_valid.made_up.max() > 0:
        fw = fl_valid.sort_values("made_up", ascending=False).iloc[0]
        qf = qmap[fw.id]
        tf = bench.tables_for(qf)[0]
        real = {norm(k) for k in tf.col(tf.key)}
        madeup = [x[0] for x in fw.extra_items if x[0] not in real][:4]  # extra items also include real, wrong ones
        noun = key_label(tf.kind)[1]
        examples.append(
            rf"\textbf{{Fabrication ({FABRICATION_MODEL}, \texttt{{{fw.id}}}).}} Prompt: {len(tf.rows):,} "
            rf"{ENTITY[tf.kind][0]} records; \emph{{``{tex_escape(qf.question)}''}} Gold answer: {len(qf.gold)} "
            rf"{noun}. The model returned {int(fw.n_pred):,} distinct {noun} (plus {int(fw.duplicates)} repeats): "
            rf"{int(fw.tp)} correct, {int(fw.wrong)} real but not matching, and {int(fw.made_up)} that do not exist "
            rf"in the data, for example " + ", ".join("``" + tex_escape(title_case(x)) + "''" for x in madeup) + ".")
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

    # ---- Usage: model responses and failed attempts, all experiments, one row per provider
    u = load_usage(args.results)
    u = u[u.model.isin({**SHORT, **ALSO})]
    summ = usage_summary(u, "model")
    urow = []
    for m in ORDER:
        ids = [(k, m, ACCESS[m]) for k, v in SHORT.items() if v == m]
        ids += [(k, m + r"$^\dagger$", ALSO_ACCESS[k]) for k, v in ALSO.items() if v == m]
        for mid, label, access in ids:
            if mid not in summ.index:
                continue
            r = summ.loc[mid]
            urow.append([label, access, f"{int(r.calls):,}", f"{int(r.failed):,}", f"{r.input_tokens / 1e6:.2f}",
                         f"{r.output_tokens / 1e6:.2f}", pct(r.reasoning_tokens / r.output_tokens),
                         "--" if pd.isna(r.cost_usd) else f"{r.cost_usd:.2f}",
                         "--" if pd.isna(r.median_latency_s) else f"{r.median_latency_s:.1f}"])
    write_table(OUT / "usage.tex", ["Model", "Access", "Calls", "Failed", "Input (M)", "Output (M)",
                                    r"Reasoning (\%)", "Cost (USD)", "Latency (s)"], urow, "llrrrrrrr")
    share = summ.reasoning_tokens / summ.output_tokens
    macro("reasonshare", f"{pct(share.min())}--{pct(share.max())}")
    macro("totalcalls", f"{int(summ.calls.sum()):,}")
    macro("failedcalls", f"{int(summ.failed.sum()):,}")

    # ---- Run facts
    mans = [json.loads(p.read_text()) for p in sorted((args.results / "manifests").glob("*.json"))]
    macro("benchsha", mans[0]["benchmark"]["sha256"][:12])
    macro("gitcommit", (mans[0]["git"]["commit"] or "?")[:7])
    macro("claudecli", next((m["claude_cli_version"] for m in mans if m["claude_cli_version"]), "?")
          .replace(" (Claude Code)", ""))
    macro("ntables", f"{len(bench.tables):,}")
    macro("nrecords", f"{sum(len(t.rows) for t in bench.tables.values()):,}")

    used = set(re.findall(r"\\([A-Za-z]+)", "".join(p.read_text() for p in DOCS if p.exists())))
    kept = {k: v for k, v in MACROS.items() if k in used}  # only what the paper and memo actually reference
    (OUT / "numbers.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in kept.items()) + "\n")
    print(pd.DataFrame(stats).T[["exact", "recall", "precision", "omit", "commit", "n_invalid"]].to_string())
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
