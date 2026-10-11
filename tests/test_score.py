import json

from relcomplete.models import mock_response
from relcomplete.schema import Question
from relcomplete.score import norm, parse_answer, score


def q_set(gold, fields=("employee name",), universe=None, **kw):
    return Question(id="t", op="selection", setting="context", subtype="x", answer_type="set", fields=list(fields),
                    question="?", fmt="csv", gold=gold, universe=universe or [], **kw)


class FakeBench:
    tables = {}


def test_norm():
    assert norm("  José  Núñez ") == "jose nunez"
    assert norm("$1,200") == norm(1200) == norm("1200.0") == "1200"
    assert norm('"The Silent Harbor".') == "the silent harbor"
    assert norm(3.10) == "3.1"


def test_parse_takes_last_answer_object():
    text = 'Thinking {"answer": ["x"]} ... final:\n```json\n{"answer": ["a", "b"]}\n```'
    assert parse_answer(text) == (["a", "b"], True)
    assert parse_answer("no json here") == (None, False)
    assert parse_answer("42") == (42.0, True)
    assert parse_answer('[["a", 1]]') == ([["a", 1]], True)


def test_set_metrics_and_error_types():
    q = q_set([["Ann Lee"], ["Bo Kim"], ["Cy Wu"]])
    bench = FakeBench()
    bench.tables = {"t1": type("T", (), {"col": lambda self, c: ["Ann Lee", "Bo Kim", "Cy Wu", "Di Ro"]})()}
    q.universe = [{"table": "t1", "col": "name"}]
    m = score(q, bench, json.dumps({"answer": ["ann lee", "Ann Lee", "Di Ro", "Zed Nobody"]}))
    assert (m["tp"], m["missing"], m["wrong"], m["made_up"], m["duplicates"]) == (1, 2, 1, 1, 1)
    assert m["recall"] == 1 / 3 and m["precision"] == 1 / 3 and not m["exact"]


def test_empty_and_unparseable():
    q = q_set([["a"]])
    assert score(q, FakeBench(), '{"answer": []}')["recall"] == 0.0
    m = score(q, FakeBench(), "I cannot do that")
    assert not m["parse_ok"] and m["recall"] == 0.0


def test_tuples_and_key_recall():
    q = q_set([["Ann", 100], ["Bo", 200]], fields=("name", "salary"))
    m = score(q, FakeBench(), '{"answer": [["Ann", "$100"], ["Bo", 250]]}')
    assert m["tp"] == 1 and m["key_recall"] == 1.0


def test_scalar_and_groups():
    q = Question(id="s", op="aggregation", setting="context", subtype="count", answer_type="scalar",
                 fields=["count"], question="?", fmt="csv", gold=12, meta={"tolerance": 0.0})
    assert score(q, FakeBench(), '{"answer": 12}')["exact"]
    m = score(q, FakeBench(), '{"answer": "10"}')
    assert not m["exact"] and m["signed_error"] == -2
    g = Question(id="g", op="aggregation", setting="context", subtype="group_count", answer_type="groups",
                 fields=["department", "count"], question="?", fmt="csv", gold={"Sales": 3, "Legal": 1})
    m = score(g, FakeBench(), '{"answer": {"sales": 3, "Legal": 2}}')
    assert m["tp"] == 1 and m["key_recall"] == 1.0 and not m["exact"]


def test_aliases():
    q = q_set([["Aluminum"]], fields=("element name",), aliases={"aluminium": "aluminum"})
    assert score(q, FakeBench(), '{"answer": ["Aluminium"]}')["exact"]


def test_mock_models_score_as_expected(bench):
    drops = []
    for q in bench.questions:
        assert score(q, bench, mock_response("mock/oracle", q))["exact"], q.id
        if q.answer_type == "set":
            m = score(q, bench, mock_response("mock/drop20", q))
            assert m["precision"] == 1.0 and m["made_up"] == 0
            drops.append(m["recall"])
    assert 0.7 < sum(drops) / len(drops) < 0.9
