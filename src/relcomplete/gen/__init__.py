"""Build the full benchmark: 500 questions across nine slices."""
from __future__ import annotations

from ..render import FORMATS
from ..schema import Benchmark
from .aggregation import aggregation
from .common import Ctx
from .enumeration import context, parametric, rag
from .join import join
from .nested import nested
from .selection import projection, selection
from .setops import setops

# (slice, builder); counts are fixed inside each builder's default argument.
SLICES = [("enumeration/context", context), ("enumeration/parametric", parametric), ("enumeration/rag", rag),
          ("selection", selection), ("projection", projection), ("join", join), ("aggregation", aggregation),
          ("nested", nested), ("setop", setops)]


def difficulty(n_gold: int, context_chars: int) -> str:
    if n_gold >= 40 or context_chars >= 100_000:
        return "hard"
    if n_gold <= 10 and context_chars < 20_000:
        return "easy"
    return "medium"


def build(seed: int = 0) -> Benchmark:
    from ..prompts import context_text

    ctx = Ctx(seed)
    for _, make in SLICES:
        make(ctx)
    bench = Benchmark(ctx.questions, ctx.tables)
    for q in bench.questions:
        q.meta["context_chars"] = len(context_text(q, bench))
        q.meta["difficulty"] = difficulty(q.meta["n_gold"], q.meta["context_chars"])
    return bench


__all__ = ["build", "SLICES", "FORMATS"]
