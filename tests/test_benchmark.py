"""The generated benchmark is the right size, reproducible, and its gold answers are correct."""
import json
from collections import Counter
from pathlib import Path

from relcomplete import closed_lists as cl
from relcomplete.gen import build
from relcomplete.schema import Benchmark, run_sql
from relcomplete.score import norm

EXPECTED = {("enumeration", "context"): 100, ("enumeration", "parametric"): 60, ("enumeration", "rag"): 40,
            ("selection", "context"): 60, ("projection", "context"): 40, ("join", "context"): 60,
            ("aggregation", "context"): 50, ("nested", "context"): 40, ("setop", "context"): 50}


def test_allocation(bench):
    assert len(bench.questions) == 500
    assert Counter((q.op, q.setting) for q in bench.questions) == EXPECTED
    assert len({q.id for q in bench.questions}) == 500


def test_formats_cover_every_context_slice(bench):
    by_op = {}
    for q in bench.questions:
        if q.setting == "context":
            by_op.setdefault(q.op, set()).add(q.fmt)
    assert all(fmts == {"markdown", "csv", "json", "nl"} for fmts in by_op.values()), by_op


def _canon(rows):
    return {tuple(norm(x) for x in r) for r in rows}


def test_gold_matches_fresh_sql(bench):
    for q in bench.questions:
        if q.sql is None:
            continue
        got = run_sql(bench.tables_for(q), q.sql)
        if q.answer_type == "scalar":
            v = got[0][0]
            assert (round(v, 2) if q.subtype == "avg" else v) == q.gold, q.id
        elif q.answer_type == "groups":
            assert {str(k): v for k, v in got} == q.gold, q.id
        else:
            assert _canon(got) == _canon(q.gold), q.id
            assert len(got) == len(q.gold), q.id  # no duplicate rows in gold


def test_gold_nonempty_and_keys_unique(bench):
    for q in bench.questions:
        assert q.gold not in (None, [], {}), q.id
        for t in bench.tables_for(q):
            keys = [norm(k) for k in t.col(t.key)]
            assert len(keys) == len(set(keys)), t.id


def test_enumeration_sizes_are_exact(bench):
    for q in bench.questions:
        if q.op == "enumeration" and q.setting != "parametric":
            assert len(q.gold) == q.meta["n_gold"], q.id


def test_parametric_gold_comes_from_lists(bench):
    assert len(cl.US_STATES) == 50 and len(cl.ELEMENTS) == 118
    assert len(cl.PRESIDENTS) == 47 and len(set(cl.PRESIDENTS)) == 45 and len(cl.PRESIDENT_SURNAMES) == 40
    for q in bench.questions:
        if q.setting == "parametric":
            lst = set(cl.LISTS[q.universe[0]["list"]])
            assert all(g[0] in lst for g in q.gold), q.id


def test_rag_context_is_subset_of_gold(bench):
    for q in bench.questions:
        if q.setting == "rag":
            assert _canon(q.meta["gold_in_context"]) <= _canon(q.gold)
            assert len(q.meta["retrieved"]) == q.meta["k"]


def test_build_is_deterministic_and_matches_saved_data(bench, tmp_path):
    again = build(0)
    assert [q.to_dict() for q in again.questions] == [q.to_dict() for q in bench.questions]
    saved = Path(__file__).parent.parent / "data" / "benchmark_v0"
    if saved.exists():
        on_disk = Benchmark.load(saved)
        assert json.dumps([q.to_dict() for q in on_disk.questions]) == \
            json.dumps([q.to_dict() for q in bench.questions])
