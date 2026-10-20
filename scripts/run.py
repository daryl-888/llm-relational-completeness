"""Run models over the benchmark and score each response.

  python scripts/run.py --model claude-cli/claude-haiku-5-5 --subset 20  # uses your Claude subscription;
      # pass full model IDs, not aliases like "haiku": an alias changes meaning when the CLI updates
  python scripts/run.py --model opencode-go/deepseek-v4-pro             # OpenCode Go plan
  python scripts/run.py --model mock/oracle                             # no API key needed
  python scripts/run.py --model M --variant exhaustive rowwise --subset 100
  python scripts/run.py --model M --subset 100 --repeat 1               # independent second sample

Results go to <out>/<model>__<variant>[__r<repeat>].jsonl, one record per call, and a manifest per pass goes
to <out>/manifests/. Finished questions are skipped on re-run. Exits 3 if any question is still unfinished
(errors, or a pass aborted after repeated errors such as a usage limit), so a wrapper can wait and resume.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from relcomplete.models import Cache, Client, is_blocked
from relcomplete.prompts import VARIANTS, messages
from relcomplete.provenance import manifest
from relcomplete.schema import Benchmark
from relcomplete.score import score

ABORT_AFTER = 8  # consecutive errors: probably a usage cap, so stop this pass instead of burning the queue


def load_env(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip().strip('"'):  # an empty placeholder must not shadow a key set further down
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


def finished(rec: dict, max_tokens: int) -> bool:
    """A record counts as done unless the call failed or was cut off by a lower token limit than the current one."""
    # Hitting the token limit is an outcome (e.g. a reasoning loop), even with no answer text: retrying at the
    # same limit would only return the cached response. It is redone only if the limit has since been raised.
    if rec.get("finish_reason") in ("length", "max_tokens"):
        return (rec.get("params") or {}).get("max_tokens", 8192) >= max_tokens
    if rec.get("blocked") or is_blocked(rec.get("error"), rec.get("finish_reason")):
        return True  # a refusal is recorded, not retried
    # No text: retry only if the call failed. A successful response with no content is the model's outcome, and
    # retrying it would just replay the cached response forever.
    return rec.get("text") is not None or not rec.get("error")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=Path("data/benchmark_v0"))
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--variant", nargs="+", default=["direct"], choices=list(VARIANTS))
    ap.add_argument("--subset", type=int, help="every (500/N)th question: a stratified N-question subset")
    ap.add_argument("--limit", type=int, help="first N questions only (smoke tests)")
    ap.add_argument("--ids", nargs="*", help="only these question ids")
    ap.add_argument("--repeat", type=int, default=0, help="sample index; >0 gives an independent re-run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768)  # reasoning models need room to think
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    load_env()

    bench = Benchmark.load(args.bench)
    qs = bench.questions
    if args.subset:
        qs = qs[:: max(1, len(qs) // args.subset)][: args.subset]
    if args.ids:
        qs = [q for q in qs if q.id in set(args.ids)]
    if args.limit:
        qs = qs[: args.limit]
    client = Client(Cache(args.out / "cache.sqlite"), max_tokens=args.max_tokens)
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    unfinished = 0

    for model in args.model:
        for variant in args.variant:
            stem = f"{model.replace('/', '_')}__{variant}" + (f"__r{args.repeat}" if args.repeat else "")
            path = args.out / f"{stem}.jsonl"
            done = set()
            if path.exists():
                done = {r["id"] for r in map(json.loads, path.open()) if finished(r, args.max_tokens)}
            todo = [q for q in qs if q.id not in done]
            print(f"{model} / {variant} / repeat {args.repeat}: {len(todo)} to run ({len(done)} already done)",
                  flush=True)
            if not todo:
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            m = manifest(args.bench, [q.id for q in qs], model, variant, args.repeat, sys.argv, args.max_tokens)
            (args.out / "manifests" / f"{stamp}__{stem}.json").write_text(json.dumps(m, indent=1))

            def one(q):
                r = client.complete(model, messages(q, bench, variant), q, repeat=args.repeat)
                return {"id": q.id, "model": model, "variant": variant, **r, "metrics": score(q, bench, r["text"])}

            ok, streak = 0, 0
            with path.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
                futs = [pool.submit(one, q) for q in todo]
                for n, fut in enumerate(as_completed(futs), 1):
                    if fut.cancelled():
                        continue
                    rec = fut.result()
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if finished(rec, args.max_tokens):
                        ok, streak = ok + 1, 0
                    else:
                        streak += 1
                        print(f"  {rec['id']}: ERROR {(rec.get('error') or rec.get('finish_reason'))[:160]}",
                              flush=True)
                        if streak >= ABORT_AFTER:
                            print(f"  ABORT: {streak} consecutive errors; stopping this pass", flush=True)
                            for other in futs:
                                other.cancel()
                    if n % 25 == 0 or n == len(futs):
                        print(f"  {n}/{len(futs)}", flush=True)
            unfinished += len(todo) - ok
    print(f"unfinished: {unfinished}", flush=True)
    return 3 if unfinished else 0


if __name__ == "__main__":
    sys.exit(main())
