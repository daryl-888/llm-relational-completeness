"""Phase 1: "list every member of a category" in three settings."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .. import closed_lists as cl
from ..render import sentence
from ..schema import Question, lit, run_sql
from ..synth import COLUMNS, DEPARTMENTS, ENTITY, ENUM_COLUMN, GENERATORS, employees
from .common import Ctx, entity_universe, fmt_for, key_label

DOMAINS = ["employees", "products", "students", "books"]
# (answer-set size, table rows). Matches are at most half the table.
CELLS = [(g, n) for n in (50, 200, 500, 1000) for g in (5, 10, 20, 50, 100) if g <= n // 2]


def _force_category(ctx: Ctx, rows: list[list], ci: int, values: list[str], target: str, n_gold: int) -> list[int]:
    """Give exactly n_gold rows the target value; returns their positions."""
    idx = sorted(ctx.rng.sample(range(len(rows)), n_gold))
    chosen = set(idx)
    others = [v for v in values if v != target]
    for i, r in enumerate(rows):
        r[ci] = target if i in chosen else (r[ci] if r[ci] != target else ctx.rng.choice(others))
    return idx


def context(ctx: Ctx, count: int = 100) -> None:
    for i in range(count):
        c, rep = i % len(CELLS), i // len(CELLS)
        n_gold, n_rows = CELLS[c]
        kind = DOMAINS[(rep + c) % len(DOMAINS)]
        qid = f"enum-ctx-{i:03d}"
        t = ctx.table(GENERATORS[kind](f"{qid}.{kind}", ctx.rng, ctx.nm, n_rows))
        col, values = ENUM_COLUMN[kind]
        target = ctx.rng.choice(values)
        pos = _force_category(ctx, t.rows, t.columns.index(col), values, target, n_gold)
        sql = f"SELECT {t.key} FROM {kind} WHERE {col} = {lit(target)}"
        gold = sorted(run_sql([t], sql))
        field, keys = key_label(kind)
        label = COLUMNS[kind][col][0]
        ctx.add(Question(
            id=qid, op="enumeration", setting="context", subtype="category", answer_type="set",
            fields=[field], question=f"List the {keys} of all {ENTITY[kind][1]} whose {label} is {target}.",
            fmt=fmt_for(rep), tables=[t.id], sql=sql, gold=gold, universe=[entity_universe(t)],
            meta={"n_gold": n_gold, "n_rows": n_rows, "domain": kind,
                  "gold_positions": [round(p / n_rows, 3) for p in pos]}))



# ---------------------------------------------------------------------------- parametric

def _letters(names: list[str], key=lambda s: s[0], min_count: int = 2) -> list[str]:
    counts: dict[str, int] = {}
    for n in names:
        counts[key(n)] = counts.get(key(n), 0) + 1
    return sorted(k for k, v in counts.items() if v >= min_count)


def _parametric_specs(ctx: Ctx) -> list[tuple[str, str, str, list[str]]]:
    """(list name, subtype, question, gold items)."""
    r = ctx.rng
    specs = []
    S = cl.US_STATES
    specs.append(("us_states", "all", "List all 50 U.S. states.", S))
    for L in r.sample(_letters(S), 9):
        specs.append(("us_states", "first_letter", f"List all U.S. states whose name begins with the letter {L}.",
                      [s for s in S if s[0] == L]))
    for L in r.sample(_letters(S, key=lambda s: s[-1]), 4):
        specs.append(("us_states", "last_letter", f"List all U.S. states whose name ends with the letter {L}.",
                      [s for s in S if s[-1] == L]))
    specs.append(("us_states", "two_words", "List all U.S. states whose name consists of two words.",
                  [s for s in S if " " in s]))
    for k in r.sample(_letters(S, key=lambda s: str(len(s.replace(" ", ""))), min_count=3), 4):
        specs.append(("us_states", "length",
                      f"List all U.S. states whose name has exactly {k} letters (ignoring spaces).",
                      [s for s in S if len(s.replace(" ", "")) == int(k)]))
    for sub in ("ss", "an", "in"):
        specs.append(("us_states", "substring",
                      f"List all U.S. states whose name contains the letter sequence \"{sub}\" (case-insensitive).",
                      [s for s in S if sub in s.lower()]))

    E = cl.ELEMENTS
    for g, nums in cl.ELEMENT_GROUPS.items():
        specs.append(("elements", "group", f"List all chemical elements that are {g}.",
                      [n for z, _, n in E if z in nums]))
    for a, b in [(1, 10), (11, 20), (21, 36), (37, 54), (55, 86), (87, 118)]:
        specs.append(("elements", "number_range",
                      f"List all chemical elements with atomic number from {a} to {b} (inclusive).",
                      [n for z, _, n in E if a <= z <= b]))
    for L in r.sample(_letters([s for _, s, _ in E]), 5):
        specs.append(("elements", "symbol_letter",
                      f"List all chemical elements whose chemical symbol begins with the letter {L}.",
                      [n for _, s, n in E if s[0] == L]))
    for L in r.sample(_letters([n for _, _, n in E]), 5):
        specs.append(("elements", "name_letter",
                      f"List all chemical elements whose English name begins with the letter {L}.",
                      [n for _, _, n in E if n[0] == L]))
    specs.append(("elements", "symbol_mismatch",
                  "List all chemical elements whose chemical symbol does not begin with the same letter as their "
                  "English name (use the American spellings, e.g. Aluminum, Sulfur, Cesium).",
                  [n for _, s, n in E if s[0] != n[0]]))
    for suf in ("ine", "on"):
        specs.append(("elements", "suffix", f"List all chemical elements whose English name ends in \"-{suf}\".",
                      [n for _, _, n in E if n.endswith(suf)]))

    P = cl.PRESIDENT_SURNAMES
    specs.append(("president_surnames", "all",
                  "List the surnames of all U.S. presidents (each distinct surname once).", P))
    for L in r.sample(_letters(P), 7):
        specs.append(("president_surnames", "first_letter",
                      f"List the distinct surnames of all U.S. presidents whose surname begins with the letter {L}.",
                      [p for p in P if p[0] == L]))
    people = {}
    for full in cl.PRESIDENTS:
        people.setdefault(cl.surname(full), set()).add(full)
    specs.append(("president_surnames", "shared",
                  "List every surname shared by two or more different people who served as U.S. president.",
                  sorted(s for s, ps in people.items() if len(ps) > 1)))
    for a, b in [(1, 10), (11, 20), (21, 30), (31, 40)]:
        specs.append(("president_surnames", "number_range",
                      f"List the distinct surnames of the U.S. presidents numbered {a} through {b} (inclusive), "
                      "counting Grover Cleveland as both the 22nd and 24th president.",
                      sorted({cl.surname(p) for p in cl.PRESIDENTS[a - 1:b]})))
    return specs


def parametric(ctx: Ctx) -> None:
    alias = {a.lower(): c.lower() for a, c in cl.ELEMENT_ALIASES.items()}
    for i, (lst, sub, text, gold) in enumerate(_parametric_specs(ctx)):
        field = {"us_states": "state name", "elements": "element name", "president_surnames": "surname"}[lst]
        ctx.add(Question(
            id=f"enum-par-{i:03d}", op="enumeration", setting="parametric", subtype=f"{lst}:{sub}",
            answer_type="set", fields=[field], question=text + " Answer from your own knowledge.", fmt="none",
            gold=[[g] for g in sorted(gold)], universe=[{"list": lst}],
            aliases=alias if lst == "elements" else None,
            meta={"n_gold": len(gold), "n_rows": len(cl.LISTS[lst]), "domain": lst}))


# ---------------------------------------------------------------------------- RAG

RAG_CELLS = [(n, g, k) for n in (200, 500) for g in (5, 10, 20, 40) for k in (10, 20, 50)]


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def rag(ctx: Ctx, count: int = 40) -> None:
    for i in range(count):
        n_rows, n_gold, k = RAG_CELLS[i % len(RAG_CELLS)]
        qid = f"enum-rag-{i:03d}"
        t = ctx.table(employees(f"{qid}.employees", ctx.rng, ctx.nm, n_rows))
        target = ctx.rng.choice(DEPARTMENTS)
        _force_category(ctx, t.rows, t.columns.index("department"), DEPARTMENTS, target, n_gold)
        sql = f"SELECT name FROM employees WHERE department = {lit(target)}"
        gold = sorted(run_sql([t], sql))
        text = f"List the names of all employees who work in the {target} department."
        docs = [_tokens(sentence("employees", dict(zip(t.columns, r)))) for r in t.rows]
        scores = BM25Okapi(docs).get_scores(_tokens(text))
        top = sorted(range(n_rows), key=lambda j: (-scores[j], j))[:k]
        gold_set = {g[0] for g in gold}
        in_ctx = sorted([t.rows[j][0]] for j in top if t.rows[j][0] in gold_set)
        ctx.add(Question(
            id=qid, op="enumeration", setting="rag", subtype=f"bm25_top{k}", answer_type="set",
            fields=["employee name"], question=text, fmt="nl", tables=[t.id], sql=sql, gold=gold,
            universe=[entity_universe(t)],
            meta={"n_gold": n_gold, "n_rows": n_rows, "domain": "employees", "k": k, "retrieved": top,
                  "doc_ids": [f"DOC-{j:04d}" for j in top], "gold_in_context": in_ctx,
                  "retrieval_recall": round(len(in_ctx) / n_gold, 3)}))
