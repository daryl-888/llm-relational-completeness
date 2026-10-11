"""Shared helpers for question generators."""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..render import FORMATS
from ..schema import Question, lit, run_sql
from ..synth import COLUMNS, ENTITY, Namer, Table


class Ctx:
    """Generation state for one benchmark build: RNG, namer, and accumulated output."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.nm = Namer(seed)
        self.tables: dict[str, Table] = {}
        self.questions: list[Question] = []

    def table(self, t: Table) -> Table:
        assert t.id not in self.tables, t.id
        self.tables[t.id] = t
        return t

    def add(self, q: Question) -> None:
        self.questions.append(q)


def fmt_for(i: int) -> str:
    return FORMATS[i % len(FORMATS)]


@dataclass
class Cond:
    sql: str
    label: str
    op_nl: str  # "is", "is greater than", ...
    value_nl: str

    def whose(self) -> str:
        return f"whose {self.label} {self.op_nl} {self.value_nl}"

    def of(self, entity: str) -> str:
        return f"the {entity}'s {self.label} {self.op_nl} {self.value_nl}"


def condition(rng: random.Random, t: Table, col: str, qualify: bool = False) -> Cond:
    label, typ = COLUMNS[t.kind][col]
    ref = f"{t.kind}.{col}" if qualify else col
    vals = t.col(col)
    if typ in ("cat", "bool"):
        v = rng.choice(vals)
        return Cond(f"{ref} = {lit(v)}", label, "is", str(v))
    s = sorted(set(vals))
    op = rng.choice([">", "<", "between"])
    if op == "between" and len(s) >= 2:
        a, b = sorted(rng.sample(s, 2))
        return Cond(f"{ref} BETWEEN {a} AND {b}", label, "is between", f"{a} and {b} (inclusive)")
    v = rng.choice(s[1:-1] or s)
    if op == "<":
        return Cond(f"{ref} < {v}", label, "is less than", str(v))
    return Cond(f"{ref} > {v}", label, "is greater than", str(v))


def conditions(rng: random.Random, t: Table, k: int, qualify: bool = False, numeric_only: bool = False,
               exclude: tuple[str, ...] = ()) -> list[Cond]:
    cols = [c for c, (_, typ) in COLUMNS[t.kind].items()
            if c not in exclude and (not numeric_only or typ == "num")]
    return [condition(rng, t, c, qualify) for c in rng.sample(cols, k)]


def where(conds: list[Cond]) -> str:
    return " AND ".join(c.sql for c in conds) if conds else "1"


def key_label(kind: str) -> tuple[str, str]:
    """(field label for one answer item, plural phrase used in 'List the ...')."""
    sing, _, key = ENTITY[kind]
    field = key if sing in key else f"{sing} {key}"
    plural = {"name": "names", "title": "titles"}.get(key, key + "s")
    return field, plural


def entity_universe(t: Table) -> dict:
    return {"table": t.id, "col": t.key}


def sample_query(rng: random.Random, propose, lo: int, hi: int, tries: int = 3000, size=len):
    """Call propose() until the result of running its SQL has size in [lo, hi].

    propose() returns (sql, tables, payload); returns (rows, sql, payload).
    """
    for _ in range(tries):
        sql, tables, payload = propose()
        rows = run_sql(tables, sql)
        if lo <= size(rows) <= hi:
            return rows, sql, payload
    raise RuntimeError(f"no query with result size in [{lo}, {hi}] after {tries} tries; last: {sql}")


def sort_rows(rows: list[list]) -> list[list]:
    return sorted(rows, key=lambda r: [str(x) for x in r])
