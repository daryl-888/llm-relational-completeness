"""Token and cost accounting across provider formats."""
import json

from relcomplete.report import load_usage
from relcomplete.usage import PRICES, normalise


def test_openai_shape_uses_reported_cost():
    r = {"model": "xiaomi/mimo-v2.6-flash", "usage": {
        "prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.5,
        "completion_tokens_details": {"reasoning_tokens": 150}, "prompt_tokens_details": {"cached_tokens": 100}}}
    u = normalise(r)
    assert (u["input"], u["cached_input"], u["output"], u["reasoning"], u["cost_usd"]) == (1000, 100, 200, 150, 0.5)


def test_openai_shape_falls_back_to_price_table():
    model = "opencode-go/deepseek-v4-flash"
    pin, pout, pcache = PRICES[model]
    u = normalise({"model": model, "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}})
    assert abs(u["cost_usd"] - (pin + pout)) < 1e-9


def test_claude_shape_counts_cache_and_thinking():
    r = {"model": "claude-cli/claude-haiku-5-5", "cost_usd": 0.01, "usage": {
        "input_tokens": 10, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 0, "output_tokens": 50,
        "output_tokens_details": {"thinking_tokens": 40}}}
    u = normalise(r)
    assert (u["input"], u["cached_input"], u["output"], u["reasoning"], u["cost_usd"]) == (100, 90, 50, 40, 0.01)


def test_load_usage_counts_every_uncached_call(tmp_path):
    recs = [{"id": "q1", "model": "m/x", "variant": "direct", "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
            {"id": "q1", "model": "m/x", "variant": "direct", "error": "429", "usage": {}},  # failed attempt
            {"id": "q2", "model": "m/x", "variant": "direct", "cached": True, "usage": {"prompt_tokens": 5}}]
    (tmp_path / "m_x__direct.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    u = load_usage(tmp_path)
    assert len(u) == 2 and u.input.sum() == 5  # the cached replay is not a new call
