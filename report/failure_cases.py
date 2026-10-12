"""Summarise the selected failure cases and whether they reproduce.

  .venv/bin/python report/failure_cases.py
Reads the original answer from results/full_v0 (repeat 0) and the re-runs from results/failure_cases
(repeats 1-3), re-scores everything with the current scorer, and writes docs/failure_cases.md and
report/generated/failure_cases.tex.
"""
import json
from pathlib import Path

from relcomplete.schema import Benchmark
from relcomplete.score import norm, score

ROOT = Path(__file__).resolve().parents[1]
MAIN, RERUNS = ROOT / "results" / "full_v0", ROOT / "results" / "failure_cases"
SHORT = {"claude-cli/claude-haiku-5-5": "Haiku 5.5", "opencode-go/deepseek-v4-pro": "DeepSeek V4 Pro",
         "opencode-go/deepseek-v4-flash": "DeepSeek V4 Flash", "opencode-go/mimo-v2.6-flash": "MiMo-V2.6 Flash",
         "opencode-go/mimo-v2.6-pro": "MiMo-V2.6 Pro"}
# (model that failed in the main run, model id used for the re-runs, question id)
CASES = [
    ("claude-cli/claude-haiku-5-5", "claude-cli/claude-haiku-5-5", "enum-par-013"),
    ("claude-cli/claude-haiku-5-5", "claude-cli/claude-haiku-5-5", "enum-par-021"),
    ("claude-cli/claude-haiku-5-5", "claude-cli/claude-haiku-5-5", "sel-029"),
    ("claude-cli/claude-haiku-5-5", "claude-cli/claude-haiku-5-5", "join-004"),
    ("claude-cli/claude-haiku-5-5", "claude-cli/claude-haiku-5-5", "nest-022"),
    ("claude-cli/claude-haiku-5-5", "claude-cli/claude-haiku-5-5", "proj-031"),
    ("claude-cli/claude-haiku-5-5", "claude-cli/claude-haiku-5-5", "agg-015"),
    ("claude-cli/claude-haiku-5-5", "claude-cli/claude-haiku-5-5", "enum-ctx-029"),
    ("opencode-go/deepseek-v4-pro", "openrouter/deepseek/deepseek-v4-pro", "enum-ctx-029"),
    ("opencode-go/mimo-v2.6-flash", "xiaomi/mimo-v2.6-flash", "enum-ctx-029"),
    ("opencode-go/mimo-v2.6-pro", "xiaomi/mimo-v2.6-pro", "enum-ctx-029"),
    ("opencode-go/mimo-v2.6-flash", "xiaomi/mimo-v2.6-flash", "enum-ctx-035"),
    ("opencode-go/deepseek-v4-flash", "openrouter/deepseek/deepseek-v4-flash", "enum-ctx-018"),
]


def record(path: Path, qid: str) -> dict | None:
    rec = None
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            if r["id"] == qid and r.get("text") is not None:
                rec = r
    return rec


def stem(model: str) -> str:
    return model.replace("/", "_")


def outcome(q, m: dict) -> str:
    if q.answer_type == "scalar":
        return "correct" if m["exact"] else f"answered {m['pred']:g} (gold {q.gold:g})"
    if m["exact"]:
        return "correct"
    if not m.get("parse_ok"):
        return "no parseable answer"
    parts = [f"recall {m['recall']:.2f}"]
    for k, label in [("missing", "missing"), ("near_miss", "near-miss"), ("wrong", "wrong"),
                     ("made_up", "made-up"), ("duplicates", "duplicate")]:
        if m.get(k):
            parts.append(f"{m[k]} {label}")
    return ", ".join(parts)


def main() -> None:
    bench = Benchmark.load(ROOT / "data" / "benchmark_v0")
    qmap = {q.id: q for q in bench.questions}
    md = ["# Failure cases", "",
          "Each case is a question a model got wrong in the main run (`results/full_v0`, repeat 0), re-run three",
          "more times on the same model (`results/failure_cases`, repeats 1–3). Everything is re-scored with the",
          "current scorer. Models first run on OpenCode Go were re-run on the same model through OpenRouter.", ""]
    rows = []
    for orig, rerun, qid in CASES:
        q = qmap[qid]
        runs = []
        r0 = record(MAIN / f"{stem(orig)}__direct.jsonl", qid)
        if r0:
            runs.append(("main run", score(q, bench, r0["text"])))
        for k in (1, 2, 3):
            r = record(RERUNS / f"{stem(rerun)}__direct__r{k}.jsonl", qid)
            if r:
                runs.append((f"re-run {k}", score(q, bench, r["text"])))
        failed = [m for _, m in runs if not m["exact"]]
        reruns = runs[1:]
        repro = sum(1 for _, m in reruns if not m["exact"] and m.get("parse_ok"))
        unparsed = sum(1 for _, m in reruns if not m.get("parse_ok"))
        name = SHORT.get(orig, orig)
        ctx_k = q.meta["context_chars"] / 4000
        md += [f"## {name}: `{qid}`", "",
               f"**Question** ({q.op}, {q.fmt} format, {q.meta['n_rows']} records, ~{ctx_k:.1f}k tokens of data, "
               f"{q.meta['n_gold']} correct answers): {q.question}", ""]
        md += [f"- {label}: {outcome(q, m)}" for label, m in runs]
        if q.answer_type != "scalar" and failed:
            parsed = [m for m in failed if "missed_items" in m]
            always = set.intersection(*({tuple(x) for x in m["missed_items"]} for m in parsed)) if parsed else set()
            if always:
                md.append(f"- Missed in every failed run: {', '.join(' / '.join(x) for x in sorted(always))}")
            if q.op == "enumeration" and q.setting == "context":
                t = bench.tables_for(q)[0]
                keys = [norm(k) for k in t.col(t.key)]
                pos = sorted(round(100 * keys.index(x[0]) / len(keys)) for x in always if x[0] in keys)
                if pos:
                    md.append(f"- Table position of those rows (% from the top): {pos}")
            near = [p for m in failed for p in m.get("near_miss_pairs", [])]
            if near:
                md.append("- Near-misses: " + "; ".join(f"\"{a[0]}\" for \"{b[0]}\"" for a, b in near[:5]))
        extra = f" ({unparsed} re-run gave no parseable answer)" if unparsed else ""
        md += [f"- **Reproduced in {repro} of {len(reruns)} re-runs{extra}.**", ""]
        rows.append((name, qid, q.op, q.meta["n_gold"], q.meta["n_rows"], outcome(q, runs[0][1]) if runs else "--",
                     f"{repro}/{len(reruns)}" + ("*" if unparsed else "")))
    (ROOT / "docs" / "failure_cases.md").write_text("\n".join(md) + "\n")

    out = ROOT / "report" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    tex = [r"\begin{tabular}{llrrp{5.2cm}c}", r"\toprule",
           r"Model & Question & Answers & Records & Main-run outcome & Reproduced \\", r"\midrule"]
    for name, qid, op, g, n, o, rep in rows:
        tex.append(rf"{name} & \texttt{{{qid}}} & {g} & {n} & {o} & {rep} \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "failure_cases.tex").write_text("\n".join(tex) + "\n")
    for r in rows:
        print(" | ".join(map(str, r)))


if __name__ == "__main__":
    main()
