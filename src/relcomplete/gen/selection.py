"""Phase 2: selection (WHERE with 1-3 conditions) and projection (SELECT columns)."""
from __future__ import annotations

from ..schema import Question
from ..synth import COLUMNS, ENTITY, GENERATORS, books, employees, products, students
from .common import Ctx, conditions, entity_universe, fmt_for, key_label, sample_query, sort_rows, where

DOMAINS = ["employees", "products", "students", "books"]
SIZE_BINS = [(1, 5), (6, 15), (16, 40), (41, 100)]


def selection(ctx: Ctx, count: int = 60) -> None:
    rows_opts = [50, 100, 200, 400]
    # Each extra condition cuts the matches; keep only combinations that can reach the size bin.
    cells = [(n, b, k) for n in rows_opts for b in SIZE_BINS for k in (1, 2, 3) if b[0] * 3 ** k <= n]
    for i in range(count):
        n_rows, (lo, hi), k = cells[(i * 7) % len(cells)]  # stride through cells to mix sizes
        kind = DOMAINS[i % len(DOMAINS)]
        qid = f"sel-{i:03d}"
        t = ctx.table(GENERATORS[kind](f"{qid}.{kind}", ctx.rng, ctx.nm, n_rows))

        def propose():
            cs = conditions(ctx.rng, t, k, numeric_only=(k == 1))
            return f"SELECT {t.key} FROM {kind} WHERE {where(cs)}", [t], cs

        rows, sql, cs = sample_query(ctx.rng, propose, lo, hi)
        field, keys = key_label(kind)
        text = f"List the {keys} of all {ENTITY[kind][1]} " + " and ".join(c.whose() for c in cs) + "."
        ctx.add(Question(
            id=qid, op="selection", setting="context", subtype=f"{k}_conditions", answer_type="set",
            fields=[field], question=text, fmt=fmt_for(i), tables=[t.id], sql=sql, gold=sort_rows(rows),
            universe=[entity_universe(t)],
            meta={"n_gold": len(rows), "n_rows": n_rows, "domain": kind, "n_conditions": k}))


# Columns with enough distinct values to make "list every distinct X" a completeness test.
DISTINCT = {"employees": "city", "books": "author", "products": "brand", "students": "hometown"}


def _table_for_distinct(ctx: Ctx, kind: str, tid: str, n: int):
    if kind == "employees":
        return employees(tid, ctx.rng, ctx.nm, n, cities=ctx.nm.cities(25))
    return {"books": books, "products": products, "students": students}[kind](tid, ctx.rng, ctx.nm, n)


def projection(ctx: Ctx, count: int = 40) -> None:
    for i in range(count):
        kind = DOMAINS[i % len(DOMAINS)]
        qid = f"proj-{i:03d}"
        n_rows = [60, 120, 200, 300][(i // 4) % 4]
        if i < count // 2:  # distinct values of one column among matching rows
            col = DISTINCT[kind]
            t = ctx.table(_table_for_distinct(ctx, kind, f"{qid}.{kind}", n_rows))
            k = (i // 8) % 2  # 0 or 1 filtering conditions

            def propose():
                cs = conditions(ctx.rng, t, k, exclude=(col,))
                return f"SELECT DISTINCT {col} FROM {kind} WHERE {where(cs)}", [t], cs

            rows, sql, cs = sample_query(ctx.rng, propose, 3, 30)
            label = COLUMNS[kind][col][0]
            scope = ENTITY[kind][1] + (" " + " and ".join(c.whose() for c in cs) if cs else "")
            text = f"List every distinct {label} that appears among the {scope}. Give each value once."
            ctx.add(Question(
                id=qid, op="projection", setting="context", subtype="distinct", answer_type="set",
                fields=[label], question=text, fmt=fmt_for(i), tables=[t.id], sql=sql, gold=sort_rows(rows),
                universe=[{"table": t.id, "col": col}],
                meta={"n_gold": len(rows), "n_rows": n_rows, "domain": kind, "n_conditions": k}))
        else:  # several attributes for every matching row
            t = ctx.table(GENERATORS[kind](f"{qid}.{kind}", ctx.rng, ctx.nm, n_rows))
            n_attr = 1 + i % 2
            proj = ctx.rng.sample(list(COLUMNS[kind]), n_attr)

            def propose():
                cs = conditions(ctx.rng, t, 1, exclude=tuple(proj))
                return f"SELECT {t.key}, {', '.join(proj)} FROM {kind} WHERE {where(cs)}", [t], cs

            rows, sql, cs = sample_query(ctx.rng, propose, 5, 40)
            field, _ = key_label(kind)
            labels = [COLUMNS[kind][c][0] for c in proj]
            text = (f"For every {ENTITY[kind][0]} {cs[0].whose()}, give its {ENTITY[kind][2]} and "
                    + " and ".join(labels) + ".")
            ctx.add(Question(
                id=qid, op="projection", setting="context", subtype=f"tuples_{n_attr + 1}", answer_type="set",
                fields=[field] + labels, question=text, fmt=fmt_for(i), tables=[t.id], sql=sql,
                gold=sort_rows(rows), universe=[entity_universe(t)] + [None] * n_attr,
                meta={"n_gold": len(rows), "n_rows": n_rows, "domain": kind}))
