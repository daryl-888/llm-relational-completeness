"""Benchmark question format and on-disk layout.

A benchmark directory holds:
  questions.jsonl  one Question per line
  tables.jsonl     every Table the questions reference (by id)
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .synth import Table


@dataclass
class Question:
    id: str
    op: str  # enumeration | selection | projection | join | aggregation | nested | setop
    setting: str  # context | parametric | rag
    subtype: str  # finer label, e.g. "1:N", "union", "group_count"
    answer_type: str  # set | scalar | groups
    fields: list[str]  # what each element of an answer tuple is, e.g. ["employee name"]
    question: str
    fmt: str  # markdown | csv | json | nl | none
    tables: list[str] = field(default_factory=list)
    sql: str | None = None
    gold: list | float | int | dict | None = None
    # Per answer position: where valid values come from, used to tell "wrong" items
    # (exist in the data) from "made-up" ones. {"table": id, "col": c} | {"list": name} | None
    universe: list[dict | None] = field(default_factory=list)
    aliases: dict[str, str] | None = None  # alternative spelling -> canonical (parametric only)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Benchmark:
    def __init__(self, questions: list[Question], tables: dict[str, Table]):
        self.questions = questions
        self.tables = tables

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "questions.jsonl", "w") as f:
            for q in self.questions:
                f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
        with open(out / "tables.jsonl", "w") as f:
            for t in self.tables.values():
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Benchmark":
        path = Path(path)
        with open(path / "questions.jsonl") as f:
            qs = [Question(**json.loads(line)) for line in f]
        with open(path / "tables.jsonl") as f:
            ts = [Table.from_dict(json.loads(line)) for line in f]
        return cls(qs, {t.id: t for t in ts})

    def tables_for(self, q: Question) -> list[Table]:
        return [self.tables[t] for t in q.tables]


def run_sql(tables: list[Table], sql: str) -> list[list]:
    """Execute sql against the given tables (each loaded under its kind as table name)."""
    con = sqlite3.connect(":memory:")
    for t in tables:
        cols = ", ".join(f'"{c}"' for c in t.columns)
        con.execute(f'CREATE TABLE "{t.kind}" ({cols})')
        con.executemany(f'INSERT INTO "{t.kind}" VALUES ({", ".join("?" * len(t.columns))})', t.rows)
    rows = [list(r) for r in con.execute(sql).fetchall()]
    con.close()
    return rows


def lit(v) -> str:
    """SQL literal for a value."""
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return repr(v)
