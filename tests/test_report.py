"""Result loading, validity, repeat agreement, usage summaries, and the report's LaTeX helpers."""
import importlib.util
import json
from pathlib import Path

from relcomplete.models import mock_response
from relcomplete.report import load_results, load_usage, repeat_agreement, usage_summary

_spec = importlib.util.spec_from_file_location("ma", Path(__file__).parent.parent / "report" / "make_assets.py")
ma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ma)


def _write(path: Path, recs: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_invalid_answers_carry_no_error_types(bench, tmp_path):
    q = next(x for x in bench.questions if x.op == "enumeration" and x.setting == "context" and x.meta["n_gold"] >= 20)
    base = {"id": q.id, "model": "m/x", "variant": "direct", "usage": {}}
    _write(tmp_path / "m_x__direct.jsonl", [{**base, "text": "no json at all"}])
    _write(tmp_path / "m_y__direct.jsonl", [{**base, "model": "m/y", "text": '{"answer": ["a"', "finish_reason": "length"}])
    _write(tmp_path / "m_z__direct.jsonl", [{**base, "model": "m/z", "text": mock_response("mock/drop20", q)}])
    d = load_results(tmp_path, bench).set_index("model")
    assert d.loc["m/x", "state"] == "unparseable" and d.loc["m/y", "state"] == "truncated"
    assert d.loc[["m/x", "m/y"], "missing"].isna().all()  # not len(gold) "missing" items
    assert d.loc["m/z", "valid"] and d.loc["m/z", "missing"] > 0


def test_repeat_agreement_ignores_failed_calls(bench, tmp_path):
    q1, q2 = [x for x in bench.questions if x.op == "selection"][:2]
    ok = lambda q, rep: {"id": q.id, "model": "m/x", "variant": "direct", "repeat": rep,  # noqa: E731
                         "text": mock_response("mock/oracle", q), "usage": {}}
    _write(tmp_path / "m_x__direct.jsonl", [ok(q1, 0), ok(q2, 0)])
    _write(tmp_path / "m_x__direct__r1.jsonl", [ok(q1, 1), {"id": q2.id, "model": "m/x", "variant": "direct",
                                                             "repeat": 1, "text": None, "error": "429"}])
    ag = repeat_agreement(load_results(tmp_path, bench))
    assert ag.loc["m/x", "n"] == 1 and ag.loc["m/x", "exact_agreement"] == 1.0


def test_usage_summary_separates_responses_from_failures(tmp_path):
    recs = [{"id": "a", "model": "m/x", "variant": "direct", "text": "x", "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
            {"id": "b", "model": "m/x", "variant": "direct", "text": None, "error": "429", "usage": {}},
            {"id": "c", "model": "m/x", "variant": "direct", "text": None,
             "error": "API Error: Output blocked by content filtering policy", "usage": {}}]
    _write(tmp_path / "m_x__direct.jsonl", recs)
    s = usage_summary(load_usage(tmp_path), "model").loc["m/x"]
    assert (s.calls, s.failed, s.input_tokens) == (2, 1, 3)  # the refusal is a response, not a retried failure


def test_tex_escape_handles_every_special_character():
    assert ma.tex_escape("a{b}c~d^e\\f&g%h$i#j_k") == (
        r"a\{b\}c\textasciitilde{}d\textasciicircum{}e\textbackslash{}f\&g\%h\$i\#j\_k")
    assert ma.title_case("atlas of the frozen harbor") == "Atlas of the Frozen Harbor"
