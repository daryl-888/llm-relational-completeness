"""Summarise results: python scripts/analyze.py  ->  results/summary.md and figures/*.png"""
import argparse
from pathlib import Path

from relcomplete.report import figures, load_results, load_usage, summary_tables, usage_table
from relcomplete.schema import Benchmark

ap = argparse.ArgumentParser()
ap.add_argument("--bench", type=Path, default=Path("data/benchmark_v0"))
ap.add_argument("--results", type=Path, default=Path("results"))
ap.add_argument("--figures", type=Path, default=Path("figures"))
args = ap.parse_args()

df = load_results(args.results, Benchmark.load(args.bench))
if df.empty:
    raise SystemExit(f"no results in {args.results}; run scripts/run.py first")
df.to_csv(args.results / "all_scores.csv", index=False)
usage = load_usage(args.results)
(args.results / "summary.md").write_text(summary_tables(df, usage))
usage_table(usage).to_csv(args.results / "usage.csv")
print((args.results / "summary.md").read_text())
for p in figures(df, args.figures):
    print("wrote", p)
