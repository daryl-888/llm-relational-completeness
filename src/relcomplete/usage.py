"""Token, cost and latency accounting across providers.

Each provider reports usage differently; normalise() maps them to one shape.
Cost uses the provider's reported figure when there is one, else PRICES below.
"""
from __future__ import annotations

# USD per 1M tokens: (input, output, cached input). Taken from provider pricing pages on 2026-10-11;
# check before relying on them. DeepSeek V4 Pro is the off-peak rate.
PRICES = {
    "opencode-go/deepseek-v4-pro": (0.66, 1.98, 0.022),
    "opencode-go/deepseek-v4-flash": (0.15, 0.60, 0.003),
    "opencode-go/mimo-v2.6-pro": (0.435, 0.87, 0.003625),
    "opencode-go/mimo-v2.6-flash": (0.14, 0.28, 0.0028),
    "openrouter/deepseek/deepseek-v4-pro": (0.948, 1.897, 0.0),  # OpenRouter listing 2026-10-12
    "openrouter/deepseek/deepseek-v4-flash": (0.006, 1.28, 0.0),  # OpenRouter listing 2026-10-12
    "xiaomi/mimo-v2.6-flash": (0.14, 0.28, 0.0028),  # OpenRouter
    "xiaomi/mimo-v2.6-pro": (0.435, 0.87, 0.003625),  # OpenRouter
    "claude-cli/claude-haiku-4-5-20251001": (1.00, 5.00, 0.10),
    "claude-cli/claude-haiku-4-5": (1.00, 5.00, 0.10),  # smoke-test label for the same model
}


def normalise(rec: dict) -> dict:
    """{input, cached_input, output, reasoning, cost_usd, latency_s} for one result record."""
    u = rec.get("usage") or {}
    if "prompt_tokens" in u:  # OpenAI-compatible
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        out = {"input": u.get("prompt_tokens") or 0, "cached_input": cached,
               "output": u.get("completion_tokens") or 0,
               "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0}
        reported = u.get("cost")
    else:  # Anthropic / Claude Code
        cached = (u.get("cache_read_input_tokens") or 0)
        out = {"input": (u.get("input_tokens") or 0) + cached + (u.get("cache_creation_input_tokens") or 0),
               "cached_input": cached, "output": u.get("output_tokens") or 0,
               "reasoning": (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0}
        reported = rec.get("cost_usd")
    if reported is None and rec.get("model") in PRICES:
        pi, po, pc = PRICES[rec["model"]]
        reported = ((out["input"] - out["cached_input"]) * pi + out["cached_input"] * pc + out["output"] * po) / 1e6
    out["cost_usd"] = reported
    out["latency_s"] = rec.get("latency_s")
    out["served_by"] = ",".join(rec.get("models") or [])
    return out
