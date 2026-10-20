"""Parse a model response and score it against the gold answer.

Every wrong item in a set answer is classified as one of
  near_miss - a garbled copy of a correct item it left out ("Raty gri S-799" for "Ratygri S-799"),
  wrong     - it exists in the data but does not satisfy the question,
  made_up   - it does not exist in the data (or the closed list) at all,
and items listed more than once are counted as duplicates. Recall stays strict: a near miss is not a hit,
but lenient_recall counts it as one.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata

from .closed_lists import LISTS
from .schema import Benchmark, Question

_NUM = re.compile(r"-?\d+(\.\d+)?")
# Leading words models drop or add without changing which entity they mean: "Rathaty" for "Project Rathaty",
# "The Fenvi X-397" (copied from "The Fenvi X-397 is a ...") for "Fenvi X-397". Stripped from both sides.
_PREFIX = re.compile(r"^(the|project) ")
NEAR_MISS_RATIO = 0.85


def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def norm(v) -> str:
    """Canonical comparable form: case-, accent-, whitespace- and number-format-insensitive."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return _fmt_num(float(v))
    s = unicodedata.normalize("NFKD", str(v))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"\s+", " ", s).strip().strip(" .;:\"'`")
    t = s.replace(",", "").replace("$", "").replace("usd", "").strip()
    if _NUM.fullmatch(t):
        return _fmt_num(float(t))
    return _PREFIX.sub("", s)


def parse_answer(text: str):
    """Return (answer, ok). Takes the last JSON object with an "answer" key; falls back to a bare JSON value."""
    if text is None:
        return None, False
    s = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\{", s)]):
        try:
            obj, _ = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            return obj["answer"], True
    last, i = None, s.find("[")
    while i != -1:  # last top-level array: skip past each one that decodes so nested lists are not picked
        try:
            last, end = dec.raw_decode(s, i)
            i = s.find("[", end)
        except json.JSONDecodeError:
            i = s.find("[", i + 1)
    if last is not None:
        return last, True
    t = norm(s)
    if _NUM.fullmatch(t):
        return float(t), True
    return None, False


def resolve_universe(q: Question, bench: Benchmark) -> list[set | None]:
    out = []
    for spec in q.universe:
        if spec is None:
            out.append(None)
        elif "list" in spec:
            out.append({norm(x) for x in LISTS[spec["list"]]})
        else:
            ids = spec["table"] if isinstance(spec["table"], list) else [spec["table"]]
            out.append({norm(x) for tid in ids for x in bench.tables[tid].col(spec["col"])})
    return out


def _items(answer, arity: int, aliases: dict | None) -> list[tuple]:
    if isinstance(answer, dict):  # groups given as {key: value}
        answer = [[k, v] for k, v in answer.items()]
    if not isinstance(answer, list):
        answer = [answer]
    out = []
    for it in answer:
        if isinstance(it, dict):
            it = list(it.values())
        if not isinstance(it, list):
            it = [it]
        t = tuple(norm(x) for x in it)
        if aliases and len(t) == 1:
            t = (aliases.get(t[0], t[0]),)
        out.append(t)
    return out


def _set_metrics(pred: list[tuple], gold: set[tuple], universe: list[set | None]) -> dict:
    p = set(pred)
    tp = p & gold
    extra = p - gold
    outside = {it for it in extra
               if len(it) != len(universe) or any(u is not None and v not in u for v, u in zip(it, universe))}
    near = _near_misses(outside, gold - p)
    precision = len(tp) / len(p) if p else (1.0 if not gold else 0.0)
    recall = len(tp) / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    m = {"n_gold": len(gold), "n_pred": len(p), "tp": len(tp), "missing": len(gold - p) - len(near),
         "wrong": len(extra) - len(outside), "made_up": len(outside) - len(near), "near_miss": len(near),
         "duplicates": len(pred) - len(p),
         "precision": precision, "recall": recall, "f1": f1, "exact": p == gold,
         "lenient_recall": (len(tp) + len(near)) / len(gold) if gold else 1.0,
         "near_miss_pairs": [[list(a), list(b)] for a, b in sorted(near.items())][:100],
         # the items themselves, for error analysis (e.g. where in the table missed rows sit)
         "missed_items": sorted(map(list, gold - p))[:500],
         "extra_items": sorted(map(list, extra))[:500]}
    if gold and len(next(iter(gold))) > 1:  # did the right entities appear, regardless of attribute values?
        gk = {g[0] for g in gold}
        m["key_recall"] = len(gk & {it[0] for it in p if it}) / len(gk)
    return m


def _near_misses(extra: set[tuple], missed: set[tuple]) -> dict[tuple, tuple]:
    """Pair extra items with the missed gold item they are a garbled copy of (greedy, one-to-one)."""
    pairs, free = {}, set(missed)
    for it in sorted(extra):
        s = " | ".join(it)
        best = max(free, key=lambda g: difflib.SequenceMatcher(None, s, " | ".join(g)).ratio(), default=None)
        if best is not None and difflib.SequenceMatcher(None, s, " | ".join(best)).ratio() >= NEAR_MISS_RATIO:
            pairs[it] = best
            free.discard(best)
    return pairs


def score(q: Question, bench: Benchmark, text: str | None) -> dict:
    answer, ok = parse_answer(text)
    base = {"parse_ok": ok}
    if q.answer_type == "scalar":
        try:
            pred = float(norm(answer[0] if isinstance(answer, list) and len(answer) == 1 else answer))
        except (TypeError, ValueError):
            return {**base, "parse_ok": False, "exact": False, "recall": 0.0, "precision": 0.0, "f1": 0.0}
        gold = float(q.gold)
        err = pred - gold
        return {**base, "exact": abs(err) <= q.meta.get("tolerance", 0.0), "pred": pred, "signed_error": err,
                "rel_error": abs(err) / abs(gold) if gold else abs(err)}
    if not ok:  # no answer to count errors in: report it as unparseable, not as every item missing
        return {**base, "exact": False, "recall": 0.0, "precision": 0.0, "f1": 0.0, "n_gold": len(q.gold)}
    if q.answer_type == "groups":
        gold = {(norm(k), norm(v)) for k, v in q.gold.items()}
        universe = resolve_universe(q, bench) + [None]
        arity = 2
    else:
        gold = {tuple(norm(x) for x in row) for row in q.gold}
        universe = resolve_universe(q, bench)
        arity = len(q.fields)
    m = {**base, **_set_metrics(_items(answer, arity, q.aliases), gold, universe)}
    if q.answer_type == "groups":
        m["key_recall"] = len({g[0] for g in gold} & {k for k, *_ in _items(answer, 2, None)}) / len(gold)
    if q.setting == "rag":
        in_ctx = {tuple(norm(x) for x in row) for row in q.meta["gold_in_context"]}
        found = set(_items(answer, arity, None)) & in_ctx
        m["recall_in_context"] = len(found) / len(in_ctx) if in_ctx else None
    return m
