"""Render tables as the text a model sees: markdown, CSV, JSON or English sentences."""
from __future__ import annotations

import csv
import io
import json

from .synth import Table

FORMATS = ["markdown", "csv", "json", "nl"]


def _a(word: str) -> str:
    return "an" if str(word)[0].lower() in "aeiou" else "a"


def sentence(kind: str, r: dict) -> str:
    if kind == "employees":
        where = "work remotely" if r["remote"] == "yes" else "do not work remotely"
        return (f"{r['name']} is {_a(r['role'])} {r['role']} in the {r['department']} department, based in the {r['city']} "
                f"office. They earn an annual salary of {r['salary']} USD, have {r['years_experience']} years "
                f"of experience, were hired in {r['hire_year']}, and {where}.")
    if kind == "products":
        return (f"The {r['name']} is {_a(r['category'])} {r['category']} product made by {r['brand']}, released in "
                f"{r['release_year']}. It costs {r['price']} USD, has a customer rating of {r['rating']}, "
                f"and there are {r['stock']} units in stock.")
    if kind == "students":
        sch = "hold a scholarship" if r["scholarship"] == "yes" else "do not hold a scholarship"
        return (f"{r['name']} is {_a(r['class_year'])} {r['class_year']} majoring in {r['major']}, from {r['hometown']}. "
                f"They have a GPA of {r['gpa']}, have earned {r['credits']} credits, and {sch}.")
    if kind == "books":
        return (f"\"{r['title']}\" is {_a(r['genre'])} {r['genre']} book by {r['author']}, published by {r['publisher']} "
                f"in {r['year']}, with {r['pages']} pages.")
    if kind == "customers":
        return (f"{r['name']} is {_a(r['tier'])} {r['tier']}-tier customer from {r['city']} who signed up in "
                f"{r['signup_year']}.")
    if kind == "orders":
        return (f"Order {r['order_id']} was placed by {r['customer']} in {r['year']} for {r['amount']} USD "
                f"of {r['category']} products.")
    if kind == "projects":
        return f"{r['name']} is based in {r['city']}, has status {r['status']}, and a budget of {r['budget']} USD."
    if kind == "badges":
        return (f"Badge {r['badge_id']} belongs to {r['employee']}, grants {r['access_level']} access, and "
                f"is registered to {r['building']}.")
    if kind.endswith("_club"):
        return f"{r['name']} ({r['class_year']}) is a member."
    return "; ".join(f"{k}: {v}" for k, v in r.items())


def render_table(t: Table, fmt: str) -> str:
    dicts = [dict(zip(t.columns, row)) for row in t.rows]
    if fmt == "markdown":
        lines = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in t.rows]
        return "\n".join(lines)
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(t.columns)
        w.writerows(t.rows)
        return buf.getvalue().rstrip("\n")
    if fmt == "json":
        return "\n".join(json.dumps(d, ensure_ascii=False) for d in dicts)  # JSON Lines: compact, one per row
    if fmt == "nl":
        return "\n".join(sentence(t.kind, d) for d in dicts)
    raise ValueError(fmt)


def render_tables(tables: list[Table], fmt: str) -> str:
    parts = []
    for t in tables:
        n = len(t.rows)
        parts.append(f"### Table: {t.kind} ({n} records)\n{render_table(t, fmt)}")
    return "\n\n".join(parts)


def render_docs(t: Table, row_ids: list[int], doc_ids: list[str]) -> str:
    """Retrieved documents for the RAG setting: one English paragraph per record."""
    out = []
    for i, did in zip(row_ids, doc_ids):
        out.append(f"[{did}] {sentence(t.kind, dict(zip(t.columns, t.rows[i])))}")
    return "\n\n".join(out)
