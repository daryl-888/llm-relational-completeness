"""Generate the benchmark: python scripts/build_benchmark.py [--seed 0] [--out data/benchmark_v0]"""
import argparse
from collections import Counter
from pathlib import Path

from relcomplete.gen import build

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, default=Path("data/benchmark_v0"))
args = ap.parse_args()

bench = build(args.seed)
bench.save(args.out)
print(f"{len(bench.questions)} questions, {len(bench.tables)} tables -> {args.out}")
for (op, setting), n in sorted(Counter((q.op, q.setting) for q in bench.questions).items()):
    print(f"  {op:12s} {setting:10s} {n}")
print("difficulty:", dict(Counter(q.meta["difficulty"] for q in bench.questions)))
chars = sum(q.meta["context_chars"] for q in bench.questions)
print(f"total context ≈ {chars / 4 / 1e6:.2f}M tokens per model per prompt variant")
