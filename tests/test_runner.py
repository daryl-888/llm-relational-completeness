"""The runner's decisions about which results are finished, and the outcome of each call."""
import importlib.util
from pathlib import Path

from relcomplete.models import is_blocked
from relcomplete.report import outcome

_spec = importlib.util.spec_from_file_location("run", Path(__file__).parent.parent / "scripts" / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_finished_rules():
    ok = {"text": '{"answer": []}', "finish_reason": "stop", "params": {"max_tokens": 32768}}
    assert run.finished(ok, 32768)
    assert not run.finished({"text": None, "error": "429 Too Many Requests"}, 32768)  # rate limits are retried
    assert not run.finished({"text": None, "error": "You've hit your session limit"}, 32768)
    # a refusal is final, from either provider style
    assert run.finished({"text": None, "error": "API Error: Output blocked by content filtering policy"}, 32768)
    assert run.finished({"text": "", "finish_reason": "content_filter"}, 32768)
    # hitting the token limit is final at the same limit, even with no answer text (e.g. a reasoning loop) ...
    trunc = {"text": None, "finish_reason": "length", "params": {"max_tokens": 32768}}
    assert run.finished(trunc, 32768)
    # ... but is redone when the limit has since been raised; old records without params assume 8192
    assert not run.finished(trunc, 65536)
    assert not run.finished({"text": "x", "finish_reason": "length"}, 32768)


def test_is_blocked():
    assert is_blocked("API Error: Output blocked by content filtering policy")
    assert is_blocked(None, "content_filter")
    assert not is_blocked("RateLimitError 429", "stop")
    assert not is_blocked(None, None)


def test_outcome_labels():
    assert outcome({"text": "ok", "finish_reason": "stop"}) == "answered"
    assert outcome({"text": None, "finish_reason": "length"}) == "truncated"
    assert outcome({"text": "", "finish_reason": "content_filter"}) == "refused"
    assert outcome({"text": None, "error": "content filtering policy"}) == "refused"
    assert outcome({"text": None, "error": "RateLimitError"}) == "api_error"


def test_load_env_skips_empty_placeholders(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nRELCOMPLETE_TEST_KEY=\nRELCOMPLETE_TEST_KEY=abc123\n")
    fake_env = {}
    monkeypatch.setattr(run.os, "environ", fake_env)  # restored after the test, so nothing leaks
    run.load_env(env)
    assert fake_env == {"RELCOMPLETE_TEST_KEY": "abc123"}


def test_empty_response_is_final_not_replayed():
    # a successful call with no content is the model's outcome; retrying would replay the cached response forever
    assert run.finished({"text": None, "finish_reason": "stop"}, 32768)
    assert outcome({"text": None, "finish_reason": "stop"}) == "empty"
