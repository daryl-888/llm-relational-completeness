"""Phase 2: aggregation — COUNT, SUM, MIN/MAX/AVG and GROUP BY.

An undercount on COUNT is the aggregate form of an incomplete answer, so the
scorer reports the signed error, not only whether the number is right.
"""
from __future__ import annotations

from ..schema import Question, run_sql
from ..synth import COLUMNS, ENTITY, ENUM_COLUMN, GENERATORS
from .common import Ctx, conditions, fmt_for, sample_query, where

DOMAINS = ["employees", "products", "students", "books"]
SUM_COL = {"employees": "years_experience", "products": "stock", "students": "credits", "books": "pages"}
EXTREME_COL = {"employees": "salary", "products": "price", "students": "gpa", "books": "pages"}
ROWS = [50, 100, 200, 400]


def _scope(kind: str, cs) -> str:
    return ENTITY[kind][1] + (" " + " and ".join(c.whose() for c in cs) if cs else "")


def aggregation(ctx: Ctx, count: int = 50) -> None:
    plan = ["count"] * 20 + ["sum"] * 10 + ["max", "min", "max", "min", "avg"] + ["group_count"] * 10 \
        + ["group_sum"] * 5
    assert len(plan) == count
    for i, sub in enumerate(plan):
        kind = DOMAINS[i % len(DOMAINS)]
        n_rows = ROWS[(i // 4) % len(ROWS)]
        qid = f"agg-{i:03d}"
        t = ctx.table(GENERATORS[kind](f"{qid}.{kind}", ctx.rng, ctx.nm, n_rows))
        plural = ENTITY[kind][1]
        answer_type, tol = "scalar", 0.0
        gcol = ENUM_COLUMN[kind][0]

        if sub == "count":
            k = 1 + i % 2

            def propose():
                cs = conditions(ctx.rng, t, k)
                return f"SELECT COUNT(*) FROM {kind} WHERE {where(cs)}", [t], cs

            rows, sql, cs = sample_query(ctx.rng, propose, 3, n_rows // 2, size=lambda r: r[0][0])
            text = f"How many {_scope(kind, cs)} are there?"
            fields = ["count"]
        elif sub in ("sum", "max", "min", "avg"):
            col = SUM_COL[kind] if sub == "sum" else EXTREME_COL[kind]
            fn = sub.upper()
            hi = 30 if sub == "sum" else n_rows // 2

            def propose():
                cs = conditions(ctx.rng, t, 1, exclude=(col,))
                return f"SELECT {fn}({col}), COUNT(*) FROM {kind} WHERE {where(cs)}", [t], cs

            rows, sql, cs = sample_query(ctx.rng, propose, 3, hi, size=lambda r: r[0][1])
            sql = sql.replace(", COUNT(*)", "")
            label = COLUMNS[kind][col][0]
            word = {"sum": "total", "max": "highest", "min": "lowest", "avg": "average"}[sub]
            text = f"What is the {word} {label} of all {_scope(kind, cs)}?"
            if sub == "avg":
                text += " Round to two decimal places."
                tol = 0.005 + 1e-9
            fields = [f"{word} {label}"]
        else:  # group_count, group_sum
            col = SUM_COL[kind]
            agg = "COUNT(*)" if sub == "group_count" else f"SUM({col})"

            def propose():
                cs = conditions(ctx.rng, t, 1, exclude=(gcol,))
                return (f"SELECT {gcol}, {agg} FROM {kind} WHERE {where(cs)} GROUP BY {gcol}", [t], cs)

            rows, sql, cs = sample_query(ctx.rng, propose, 3, 10)
            glabel = COLUMNS[kind][gcol][0]
            what = f"how many {plural} {' and '.join(c.whose() for c in cs)} are there" if sub == "group_count" \
                else f"what is the total {COLUMNS[kind][col][0]} of the {_scope(kind, cs)}"
            text = (f"For each {glabel}, {what}? Include only {glabel} values with at least one such "
                    f"{ENTITY[kind][0]}.")
            answer_type = "groups"
            fields = [glabel, "count" if sub == "group_count" else "total"]

        if answer_type == "scalar":
            value = rows[0][0]
            gold = round(value, 2) if sub == "avg" else value
        else:
            gold = {str(g): v for g, v in rows}
        n_rel = run_sql([t], f"SELECT COUNT(*) FROM {kind} WHERE {where(cs)}")[0][0]
        ctx.add(Question(
            id=qid, op="aggregation", setting="context", subtype=sub, answer_type=answer_type, fields=fields,
            question=text, fmt=fmt_for(i), tables=[t.id], sql=sql, gold=gold,
            universe=[{"table": t.id, "col": gcol}] if answer_type == "groups" else [],
            meta={"n_gold": n_rel, "n_rows": n_rows, "domain": kind, "tolerance": tol}))
