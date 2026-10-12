"""Generate the report's tables and figures from a results directory.

  .venv/bin/python report/make_assets.py [--results results/full_v0]
Writes report/generated/*.tex (tables, numbers) and report/generated/*.pdf (figures).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from relcomplete.report import load_results, load_usage, repeat_agreement, slice_name, usage_table  # noqa: E402
from relcomplete.schema import Benchmark  # noqa: E402
from relcomplete.score import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated"
# Models in the report, with display names. MiMo was run to completion on OpenRouter; its partial OpenCode Go
# runs and the stopped Haiku 4.5 run (90 questions) are left out of the tables.
SHORT = {"claude-cli/claude-haiku-5-5": "Haiku 5.5",
         "opencode-go/deepseek-v4-pro": "DeepSeek V4 Pro", "opencode-go/deepseek-v4-flash": "DeepSeek V4 Flash",
         "xiaomi/mimo-v2.6-pro": "MiMo-V2.6 Pro", "xiaomi/mimo-v2.6-flash": "MiMo-V2.6 Flash"}
SLICE_ORDER = ["enumeration", "enumeration (parametric)", "enumeration (rag)", "selection", "projection", "join",
               "aggregation", "nested", "setop"]
SLICE_TEX = {"enumeration": "Enumeration (in context)", "enumeration (parametric)": "Enumeration (memory)",
             "enumeration (rag)": "Enumeration (RAG)", "selection": "Selection", "projection": "Projection",
             "join": "JOIN", "aggregation": "Aggregation (groups)", "nested": "Nested", "setop": "Set operations"}


def tex_table(df: pd.DataFrame, path: Path, fmt: str = "{:.3f}", col_spec: str | None = None) -> None:
    cols = list(df.columns)
    spec = col_spec or "l" + "r" * len(cols)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join([df.index.name or ""] + cols) + r" \\",
             r"\midrule"]
    ints = [pd.api.types.is_integer_dtype(df[c]) for c in cols]
    for idx, row in zip(df.index, df.itertuples(index=False)):
        cells = [str(idx)] + [("--" if pd.isna(v) else (str(int(v)) if is_int else
                               (fmt.format(v) if isinstance(v, float) else str(v))))
                              for v, is_int in zip(row, ints)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines).replace("%", r"\%").replace("_", r"\_") + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "full_v0")
    ap.add_argument("--bench", type=Path, default=ROOT / "data" / "benchmark_v0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = Benchmark.load(args.bench)
    all_df = load_results(args.results, bench)
    all_df = all_df[all_df.model.isin(SHORT)]
    all_df["Model"] = all_df.model.map(SHORT)
    models = [m for m in SHORT.values() if m in set(all_df.Model)]
    df = all_df[(all_df.repeat == 0) & (all_df.variant == "direct")]
    sets = df[df.answer_type != "scalar"]

    # Benchmark composition
    qs = pd.DataFrame([{"slice": q.op + ("" if q.setting == "context" else f" ({q.setting})"),
                        "n_gold": q.meta["n_gold"], "chars": q.meta["context_chars"]} for q in bench.questions])
    comp = qs.groupby("slice").agg(Questions=("n_gold", "size"), **{"Median answer size": ("n_gold", "median")},
                                   **{"Max answer size": ("n_gold", "max")},
                                   **{"Median context (k tokens)": ("chars", lambda s: s.median() / 4000)})
    comp = comp.reindex(SLICE_ORDER).rename(index=SLICE_TEX)
    comp.index.name = "Slice"
    comp["Questions"] = comp["Questions"].astype(int)
    comp["Median answer size"] = comp["Median answer size"].map(lambda v: f"{v:g}")
    comp["Max answer size"] = comp["Max answer size"].astype(int)
    tex_table(comp, OUT / "composition.tex", fmt="{:.1f}")

    # Overall per model (set and group answers)
    def overall(g):
        return pd.Series({"Questions": len(g), "Recall": g.recall.mean(), "Precision": g.precision.mean(),
                          "Exact": g.exact.astype(bool).mean(), "Missing": int(g.missing.sum()),
                          "Near-miss": int(g.near_miss.sum()), "Wrong": int(g.wrong.sum()),
                          "Made-up": int(g.made_up.sum()), "Dupl.": int(g.duplicates.sum()),
                          "Trunc.": int(g.finish_reason.isin(["length", "max_tokens"]).sum()),
                          "Refused": int(g.error.fillna("").str.contains("content filtering").sum())})
    ov = sets.groupby("Model").apply(overall, include_groups=False).reindex(models)
    for c in ["Questions", "Missing", "Near-miss", "Wrong", "Made-up", "Dupl.", "Trunc.", "Refused"]:
        ov[c] = ov[c].astype(int)
    ov = ov.sort_values("Recall", ascending=False)
    tex_table(ov, OUT / "overall.tex")
    ov.to_csv(OUT / "overall.csv")

    # Recall and exact by slice
    s = sets.assign(slice=slice_name)
    for metric, name in [("recall", "recall_by_slice"), ("exact", "exact_by_slice")]:
        t = s.assign(v=s[metric].astype(float)).pivot_table(index="slice", columns="Model", values="v",
                                                            aggfunc="mean")
        t = t.reindex([x for x in SLICE_ORDER if x in t.index])[[m for m in ov.index]].rename(index=SLICE_TEX)
        t.index.name = "Slice"
        tex_table(t, OUT / f"{name}.tex", fmt="{:.2f}")

    # Scalar aggregation
    sc = df[df.answer_type == "scalar"]
    if len(sc):
        a = sc.groupby("Model").apply(lambda g: pd.Series({
            "Questions": len(g), "Exact": g.exact.astype(bool).mean(),
            "Undercount": (g.signed_error < 0).mean(), "Overcount": (g.signed_error > 0).mean(),
            "Median rel. error": g.rel_error.median()}), include_groups=False).reindex(ov.index)
        a["Questions"] = a["Questions"].astype(int)
        tex_table(a, OUT / "scalar.tex", fmt="{:.2f}")

    # Enumeration curves
    enum = sets[(sets.op == "enumeration") & (sets.setting == "context")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for m in ov.index:
        g = enum[enum.Model == m]
        for ax, x in zip(axes, ["n_gold", "n_rows"]):
            r = g.groupby(x).recall.mean()
            ax.plot(r.index, r.values, marker="o", ms=4, label=m)
    for ax, x, lab in zip(axes, ["n_gold", "n_rows"], ["Number of correct items", "Rows in the table"]):
        ax.set(xscale="log", xlabel=lab, ylabel="Mean recall")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "enumeration_curves.pdf")
    plt.close(fig)

    # The cost of completeness: reasoning tokens spent against answer size
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for m in ov.index:
        r = enum[enum.Model == m].groupby("n_gold").reasoning.median()
        ax.plot(r.index, r.values, marker="o", ms=4, label=m)
    ax.set(xscale="log", yscale="log", xlabel="Number of correct items", ylabel="Median reasoning tokens")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "reasoning_vs_size.pdf")
    plt.close(fig)

    # Enumeration exact-set accuracy by answer size, as a table
    ex = enum.assign(x=enum.exact.astype(float)).pivot_table(index="n_gold", columns="Model", values="x")
    ex = ex[[m for m in ov.index]]
    ex.index.name = "Correct items"
    tex_table(ex, OUT / "enum_exact_by_size.tex", fmt="{:.2f}")

    # RAG: recall overall vs over the items retrieval actually returned
    rag = sets[sets.setting == "rag"]
    if len(rag):
        rt = rag.groupby("Model").agg(**{"Recall": ("recall", "mean"),
                                         "Recall over retrieved": ("recall_in_context", "mean")}).reindex(ov.index)
        tex_table(rt, OUT / "rag.tex")

    # Where in the table are misses? Position of every gold row, and whether it was missed.
    qmap = {q.id: q for q in bench.questions}
    pos_rows = []
    for _, r in enum.iterrows():
        q = qmap[r.id]
        t = bench.tables_for(q)[0]
        missed = {tuple(x) for x in (r.missed_items or [])} if isinstance(r.missed_items, list) else set()
        keys = t.col(t.key)
        n = len(keys)
        for g in q.gold:
            i = next(j for j, k in enumerate(keys) if k == g[0])
            pos_rows.append({"Model": r.Model, "decile": min(9, int(10 * i / n)), "missed": (norm(g[0]),) in missed})
    pos = pd.DataFrame(pos_rows)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for m in ov.index:
        p = pos[pos.Model == m].groupby("decile").missed.mean()
        ax.plot((p.index + 0.5) * 10, p.values, marker="o", ms=4, label=m)
    ax.set(xlabel="Position of the correct row in the table (%)", ylabel="Miss rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "miss_by_position.pdf")
    plt.close(fig)

    # Prompt variants and run-to-run variation, on the shared 100-question subset
    v = all_df[(all_df.repeat == 0) & (all_df.answer_type != "scalar")]
    if v.variant.nunique() > 1:
        shared = set.intersection(*(set(g.id) for _, g in v.groupby("variant")))
        vv = v[v.id.isin(shared)]
        t = vv.pivot_table(index="Model", columns="variant", values="recall", aggfunc="mean")
        t = t.reindex([m for m in ov.index if m in t.index])[[c for c in ["direct", "exhaustive", "rowwise"]
                                                              if c in t.columns]]
        t.columns = [c.capitalize() for c in t.columns]
        tex_table(t, OUT / "variants.tex", fmt="{:.3f}")
        (OUT / "variants_n.tex").write_text(f"{len(shared)}\n")
    ag = repeat_agreement(all_df[all_df.error.isna()])  # API errors (usage caps) are not answers
    if not ag.empty:
        ag.index = ag.index.map(lambda m: SHORT.get(m, m))
        ag = ag.rename(columns={"n": "Questions", "exact_agreement": "Same exact verdict",
                                "mean_abs_recall_diff": "Mean |recall diff|", "exact_r0": "Exact (run 1)",
                                "exact_r1": "Exact (run 2)"})
        ag.index.name = "Model"
        tex_table(ag, OUT / "repeats.tex")

    # Usage: every call made, including retries and all experiments
    u = load_usage(args.results)
    u = u[u.model.isin(SHORT)]
    ut = usage_table(u).reset_index()
    ut = ut[ut.model != "TOTAL"]
    ut["Model"] = ut.model.map(SHORT).fillna(ut.model)
    agg = ut.groupby("Model").agg(Calls=("calls", "sum"), **{"Input (M)": ("input_tokens", "sum")},
                                  **{"Output (M)": ("output_tokens", "sum")},
                                  **{"of which reasoning (M)": ("reasoning_tokens", "sum")},
                                  **{"Cost (USD)": ("cost_usd", "sum")}).reindex(ov.index)
    for c in ["Input (M)", "Output (M)", "of which reasoning (M)"]:
        agg[c] = agg[c] / 1e6
    lat = u.assign(Model=u.model.map(SHORT)).groupby("Model").latency_s.median().reindex(ov.index)
    agg["Median latency (s)"] = lat
    agg["Calls"] = agg["Calls"].astype(int)
    tex_table(agg, OUT / "usage.tex", fmt="{:.2f}")

    # Run facts for the text
    mans = [json.loads(p.read_text()) for p in sorted((args.results / "manifests").glob("*.json"))]
    facts = {"bench_sha": mans[0]["benchmark"]["sha256"][:12] if mans else "?",
             "commit": (mans[0]["git"]["commit"] or "?")[:7] if mans else "?",
             "start": min(m["created_at"] for m in mans)[:10] if mans else "?",
             "claude_cli": next((m["claude_cli_version"] for m in mans if m["claude_cli_version"]), "?"),
             "total_calls": int(u.shape[0]),
             "total_cost": float(u.cost_usd.sum()),
             "go_cost": float(u[u.model.str.startswith("opencode-go/")].cost_usd.sum())}
    (OUT / "facts.tex").write_text("\n".join([
        rf"\newcommand{{\benchsha}}{{{facts['bench_sha']}}}",
        rf"\newcommand{{\gitcommit}}{{{facts['commit']}}}",
        rf"\newcommand{{\rundate}}{{{facts['start']}}}",
        rf"\newcommand{{\claudecli}}{{{facts['claude_cli']}}}",
        rf"\newcommand{{\totalcalls}}{{{facts['total_calls']:,}}}",
        rf"\newcommand{{\gocost}}{{{facts['go_cost']:.2f}}}"]) + "\n")
    print(ov.to_string())
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
