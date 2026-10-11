"""Phase 2: set operations over club rosters — UNION, INTERSECT, EXCEPT."""
from __future__ import annotations

from ..schema import Question, run_sql
from ..synth import CLASS_YEARS, roster
from .common import Ctx, fmt_for, sort_rows

CLUBS = ["chess_club", "robotics_club", "drama_club"]


def _club(kind: str) -> str:
    return "the " + kind.replace("_", " ").title()


# subtype -> (SQL over club tables a, b, c; English description of who to list)
OPS = {
    "union": ("SELECT name FROM {a} UNION SELECT name FROM {b}",
              "who are members of {A} or {B} (or both)"),
    "intersect": ("SELECT name FROM {a} INTERSECT SELECT name FROM {b}",
                  "who are members of both {A} and {B}"),
    "except": ("SELECT name FROM {a} EXCEPT SELECT name FROM {b}",
               "who are members of {A} but not of {B}"),
    "union_except": ("SELECT name FROM {a} UNION SELECT name FROM {b} EXCEPT SELECT name FROM {c}",
                     "who are members of {A} or {B} (or both) but not of {C}"),
    "intersect_except": ("SELECT name FROM {a} INTERSECT SELECT name FROM {b} EXCEPT SELECT name FROM {c}",
                         "who are members of both {A} and {B} but not of {C}"),
}
PLAN = ["union"] * 15 + ["intersect"] * 15 + ["except"] * 15 + ["union_except"] * 3 + ["intersect_except"] * 2


def setops(ctx: Ctx, count: int = 50) -> None:
    assert len(PLAN) == count
    r = ctx.rng
    for i, sub in enumerate(PLAN):
        qid = f"set-{i:03d}"
        pool_n = [60, 120, 200][i % 3]
        pool = ctx.nm.people(pool_n)
        year = {m: r.choice(CLASS_YEARS) for m in pool}
        n_clubs = 3 if "_except" in sub else 2
        while True:
            a, b, c = r.sample(CLUBS, 3)
            tables = []
            for kind in (a, b, c)[:n_clubs]:
                rate = r.uniform(0.2, 0.5)
                members = [m for m in pool if r.random() < rate]
                tables.append(roster(f"{qid}.{kind}", kind, r, members, year))
            sql = OPS[sub][0].format(a=a, b=b, c=c)
            rows = run_sql(tables, sql)
            if 3 <= len(rows) <= 100:
                break
        for t in tables:
            ctx.table(t)
        who = OPS[sub][1].format(A=_club(a), B=_club(b), C=_club(c))
        ctx.add(Question(
            id=qid, op="setop", setting="context", subtype=sub, answer_type="set", fields=["student name"],
            question=f"Each table below is the membership roster of one club. List the names of all students {who}.",
            fmt=fmt_for(i), tables=[t.id for t in tables], sql=sql, gold=sort_rows(rows),
            universe=[{"table": [t.id for t in tables], "col": "name"}],
            meta={"n_gold": len(rows), "n_rows": sum(len(t.rows) for t in tables)}))
