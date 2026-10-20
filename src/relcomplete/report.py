"""Aggregate scored results into tables and figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .schema import Benchmark  # noqa: E402
from .score import score  # noqa: E402
from .usage import normalise  # noqa: E402


def load_results(results_dir: Path, bench: Benchmark) -> pd.DataFrame:
    meta = {q.id: q for q in bench.questions}
    recs = []
    for path in sorted(results_dir.glob("*__*.jsonl")):
        latest = {}
        for line in path.open():
            r = json.loads(line)
            latest[r["id"]] = r  # a re-run of the same question replaces the earlier line
        for r in latest.values():
            q = meta.get(r["id"])
            if q is None:
                continue
            recs.append({"id": q.id, "model": r["model"], "variant": r["variant"], "repeat": r.get("repeat", 0),
                         "op": q.op,
                         "setting": q.setting, "subtype": q.subtype, "answer_type": q.answer_type, "fmt": q.fmt,
                         "n_gold": q.meta["n_gold"], "n_rows": q.meta["n_rows"],
                         "difficulty": q.meta["difficulty"], "error": r.get("error"),
                         "finish_reason": r.get("finish_reason"), "cached": r.get("cached", False),
                         "outcome": outcome(r), **normalise(r),
                         **score(q, bench, r.get("text"))})  # re-scored with the current scorer
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    # state: the outcome, with "answered" split by whether the answer could be parsed. Only valid answers carry
    # error types, so a truncated or unparseable answer is never counted as omitted items.
    df["state"] = df.outcome.where(~((df.outcome == "answered") & ~df.parse_ok.astype(bool)), "unparseable")
    df["valid"] = df.state == "answered"
    error_cols = [c for c in ERROR_COLUMNS if c in df.columns]
    df.loc[~df.valid, error_cols] = float("nan")
    return df


ERROR_COLUMNS = ["tp", "n_pred", "missing", "near_miss", "wrong", "made_up", "duplicates"]


def outcome(r: dict) -> str:
    """What happened to a call, before looking at answer quality.

    answered | truncated (hit the token limit) | refused (safety filter) | empty (the provider returned no
    content) | api_error (the call failed and is retried). load_results adds "unparseable" for answered calls
    whose text could not be parsed.
    """
    from .models import is_blocked

    if is_blocked(r.get("error"), r.get("finish_reason")) or r.get("blocked"):
        return "refused"
    if r.get("finish_reason") in ("length", "max_tokens"):
        return "truncated"
    if r.get("text") is None:
        return "api_error" if r.get("error") else "empty"
    return "answered"


def slice_name(df: pd.DataFrame) -> pd.Series:
    return df.op + df.setting.map(lambda s: "" if s == "context" else f" ({s})")


def repeat_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """How much do two independent samples of the same prompt differ? (questions run at repeat 0 and 1)

    Failed calls (state api_error) are not answers, so questions without a response in both samples are dropped.
    """
    d = df[(df.variant == "direct") & (df.state != "api_error")]
    a = d[d.repeat == 0].set_index(["model", "id"])
    b = d[d.repeat == 1].set_index(["model", "id"])
    j = a[["exact", "recall"]].join(b[["exact", "recall"]], rsuffix="_r1", how="inner")
    if j.empty:
        return pd.DataFrame()
    j["same_exact"] = j.exact.astype(bool) == j.exact_r1.astype(bool)
    j["recall_diff"] = (j.recall - j.recall_r1).abs()
    return j.groupby("model").agg(n=("same_exact", "size"), exact_agreement=("same_exact", "mean"),
                                  mean_abs_recall_diff=("recall_diff", "mean"),
                                  exact_r0=("exact", "mean"), exact_r1=("exact_r1", "mean")).round(3)


def summary_tables(df: pd.DataFrame, usage: pd.DataFrame | None = None) -> str:
    out = []
    agreement = repeat_agreement(df)
    df = df[df.repeat == 0]  # main tables use the first sample; repeats only measure variation
    sets = df[df.answer_type != "scalar"]
    agg = {"n": ("id", "size"), "recall": ("recall", "mean"), "precision": ("precision", "mean"),
           "exact": ("exact", "mean")}
    out.append("## Overall (set and group answers)\n")
    out.append(sets.groupby(["model", "variant"]).agg(**agg).round(3).to_markdown())
    out.append("\n\n## Recall by slice (direct prompting)\n")
    d = sets[sets.variant == "direct"].assign(slice=slice_name)
    out.append(d.pivot_table(index="slice", columns="model", values="recall", aggfunc="mean").round(3).to_markdown())
    out.append("\n\n## Exact-set accuracy by slice (direct prompting)\n")
    out.append(d.pivot_table(index="slice", columns="model", values="exact", aggfunc="mean").round(3).to_markdown())
    errs = d.groupby("model")[["tp", "missing", "near_miss", "wrong", "made_up", "duplicates"]].sum()
    out.append("\n\n## Error counts (direct prompting, set answers)\n")
    out.append(errs.to_markdown())
    sc = df[df.answer_type == "scalar"]
    if len(sc):
        out.append("\n\n## Scalar aggregation\n")
        t = sc.groupby(["model", "variant", "subtype"]).agg(
            n=("id", "size"), exact=("exact", "mean"),
            undercount=("signed_error", lambda s: (s < 0).mean()),
            overcount=("signed_error", lambda s: (s > 0).mean())).round(3)
        out.append(t.to_markdown())
    if not agreement.empty:
        out.append("\n\n## Run-to-run variation (direct prompt, repeat 0 vs repeat 1)\n")
        out.append(agreement.to_markdown())
    out.append("\n\n## Usage\n")
    out.append(usage_table(df if usage is None else usage).to_markdown())
    out.append("\n\nTokens are as reported by each provider. Cost is the provider's figure when it reports one, "
               "otherwise computed from `usage.PRICES`. Claude CLI costs are list-price equivalents: a subscription "
               "is not billed per call. Calls counts model responses (including refusals); failed counts "
               "attempts that returned a provider error and were retried.\n")
    out.append("\n\n## Outcomes (answered, unparseable, truncated, refused, empty, api_error)\n")
    out.append(df.groupby(["model", "variant"]).state.value_counts().unstack(fill_value=0).to_markdown())
    return "\n".join(out) + "\n"


def load_usage(results_dir: Path) -> pd.DataFrame:
    """One row per API call, including attempts later superseded by a re-run: everything that was spent."""
    rows = []
    for path in sorted(results_dir.glob("*__*.jsonl")):
        for line in path.open():
            r = json.loads(line)
            if not r.get("cached"):
                rows.append({"id": r["id"], "model": r["model"], "variant": r["variant"],
                             "repeat": r.get("repeat", 0), "outcome": outcome(r), **normalise(r)})
    return pd.DataFrame(rows)


def usage_summary(u: pd.DataFrame, by) -> pd.DataFrame:
    """Calls (model responses, including refusals), failed attempts (provider errors, retried), tokens and cost."""
    resp = u.outcome != "api_error"
    return u.assign(response=resp, failed=~resp).groupby(by).agg(
        calls=("response", "sum"), failed=("failed", "sum"),
        served_by=("served_by", lambda s: ",".join(sorted({x for x in s if x}))),
        input_tokens=("input", "sum"), cached_input=("cached_input", "sum"), output_tokens=("output", "sum"),
        reasoning_tokens=("reasoning", "sum"), cost_usd=("cost_usd", lambda s: s.sum(min_count=1)),
        median_latency_s=("latency_s", "median"))


def usage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spend per model and variant; pass load_usage() output to include superseded attempts."""
    t = usage_summary(df, ["model", "variant", "repeat"])
    total = t.sum(numeric_only=True)
    total["median_latency_s"] = df.latency_s.median()
    t.loc[("TOTAL", "", ""), :] = total
    return t.round({"cost_usd": 4, "median_latency_s": 1})


def _line(ax, d: pd.DataFrame, x: str, title: str, logx: bool = True) -> None:
    for model, g in d.groupby("model"):
        s = g.groupby(x).recall.mean()
        ax.plot(s.index, s.values, marker="o", label=model)
    ax.set(title=title, xlabel=x, ylabel="mean recall", ylim=(0, 1.02))
    if logx:
        ax.set_xscale("log")
    ax.grid(alpha=0.3)


def figures(df: pd.DataFrame, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    df = df[df.repeat == 0]
    paths = []
    enum = df[(df.op == "enumeration") & (df.setting == "context") & (df.variant == "direct")]
    if len(enum):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        _line(axes[0], enum, "n_gold", "Enumeration in context: recall vs answer-set size")
        _line(axes[1], enum, "n_rows", "Enumeration in context: recall vs table size")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        paths.append(out / "enumeration_curves.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    d = df[(df.answer_type != "scalar") & (df.variant == "direct")].assign(slice=slice_name)
    if len(d):
        fig, ax = plt.subplots(figsize=(10, 4))
        d.pivot_table(index="slice", columns="model", values="recall", aggfunc="mean").plot.bar(ax=ax)
        ax.set(ylabel="mean recall", ylim=(0, 1.02), title="Recall by slice (direct prompting)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        paths.append(out / "recall_by_slice.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        e = d.groupby("model")[["missing", "wrong", "made_up", "duplicates"]].sum()
        (e.div(e.sum(axis=1), axis=0)).plot.barh(stacked=True, ax=ax)
        ax.set(xlabel="share of errors", title="What kind of errors? (direct prompting)")
        fig.tight_layout()
        paths.append(out / "error_types.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    v = df[df.answer_type != "scalar"]
    if v.variant.nunique() > 1:
        ids = set.intersection(*(set(g.id) for _, g in v.groupby("variant")))
        vv = v[v.id.isin(ids)]
        fig, ax = plt.subplots(figsize=(7, 4))
        vv.pivot_table(index="model", columns="variant", values="recall", aggfunc="mean").plot.bar(ax=ax)
        ax.set(ylabel="mean recall", ylim=(0, 1.02), title=f"Prompting variants ({len(ids)} shared questions)")
        fig.tight_layout()
        paths.append(out / "prompt_variants.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    return paths
