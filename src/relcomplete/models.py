"""Model access with a disk cache.

  claude-cli/<model>   the local Claude Code CLI (`claude -p`), billed to a Claude subscription
  opencode-go/<model>  OpenCode Go subscription (DeepSeek V4, MiMo, Kimi, GLM, ...), key OPENCODE_GO_API_KEY
  deepseek/<model>     DeepSeek's own API, key DEEPSEEK_API_KEY
  <anything else>      OpenRouter, or LLM_BASE_URL if set, key OPENROUTER_API_KEY

Mock models ("mock/oracle", "mock/drop20") answer from the gold data so the whole
pipeline can be tested without an API key.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .schema import Question

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# model prefix -> (OpenAI-compatible base URL, environment variable holding the key)
USER_AGENT = "relcomplete-benchmark/0.1"
PROVIDERS = {
    "opencode-go": ("https://opencode.ai/zen/go/v1", "OPENCODE_GO_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
}


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT)")
        self.lock = threading.Lock()

    @staticmethod
    def key(model: str, messages: list[dict], params: dict) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        with self.lock:
            row = self.con.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, response: dict) -> None:
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(response)))
            self.con.commit()


class Client:
    def __init__(self, cache: Cache, max_tokens: int = 32768):
        self.cache = cache
        self.max_tokens = max_tokens
        self._clients: dict[str, object] = {}

    def _openai(self, model: str):
        """(client, model name to send) for a model id like "opencode-go/deepseek-v4-pro"."""
        prefix, _, rest = model.partition("/")
        if prefix in PROVIDERS:
            base_url, env = PROVIDERS[prefix]
            name = rest
        else:
            base_url, env = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL), "OPENROUTER_API_KEY"
            prefix, name = "default", model
        if prefix not in self._clients:
            from openai import OpenAI

            key = os.environ.get(env)
            if not key:
                raise SystemExit(f"Set {env} in .env (see .env.example).")
            # OpenCode Go asks clients to identify themselves rather than send a generic SDK user agent.
            self._clients[prefix] = OpenAI(base_url=base_url, api_key=key, default_headers={"User-Agent": USER_AGENT})
        return self._clients[prefix], name

    def complete(self, model: str, messages: list[dict], q: Question | None = None, repeat: int = 0) -> dict:
        """Returns {"text", "usage", "cached", provenance...}; errors come back as {"text": None, "error": ...}.

        repeat > 0 asks for an independent sample of the same prompt (it is part of the cache key).
        """
        params = {"temperature": 0, "max_tokens": self.max_tokens}
        prov = {"params": params, "repeat": repeat, "prompt_sha256": prompt_hash(messages),
                "started_at": _now()}
        if model.startswith("mock/"):
            return {"text": mock_response(model, q), "usage": {}, "cached": False, **prov}
        k = Cache.key(model, messages, {**params, "repeat": repeat} if repeat else params)
        hit = self.cache.get(k)
        if hit is not None:
            return {**hit, "cached": True}
        t0 = time.monotonic()
        if model.startswith("claude-cli/"):
            out = {**claude_cli(model.split("/", 1)[1], messages), **prov,
                   "latency_s": round(time.monotonic() - t0, 2), "finished_at": _now()}
            if out.get("error") is None:
                self.cache.put(k, out)
            return {**out, "cached": False}
        last = None
        for attempt in range(5):
            try:
                client, name = self._openai(model)
                # One question is one conversation: a stable per-conversation ID, as OpenCode Go requests.
                extra = {"x-opencode-session": k[:32]} if model.startswith("opencode-go/") else None
                r = client.chat.completions.create(model=name, messages=messages, extra_headers=extra, **params)
                msg = r.choices[0].message
                extra_fields = msg.model_extra or {}
                out = {"text": msg.content,
                       "reasoning": getattr(msg, "reasoning_content", None) or extra_fields.get("reasoning_content")
                       or extra_fields.get("reasoning"),
                       "finish_reason": r.choices[0].finish_reason,
                       "usage": r.usage.model_dump() if r.usage else {},
                       "models": [r.model], "response_id": r.id,
                       "system_fingerprint": getattr(r, "system_fingerprint", None), **prov,
                       "latency_s": round(time.monotonic() - t0, 2), "finished_at": _now(), "attempts": attempt + 1}
                self.cache.put(k, out)
                return {**out, "cached": False}
            except Exception as e:  # rate limits, timeouts, provider errors
                last = e
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                    break  # a malformed or refused request will not succeed on retry
                time.sleep(2 ** attempt + random.random())
        return {"text": None, "error": repr(last)[:1000], "usage": {}, "cached": False, **prov,
                "finished_at": _now(), "attempts": attempt + 1}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prompt_hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def claude_cli(model: str, messages: list[dict], timeout: int = 600) -> dict:
    """One headless Claude Code call: our system prompt, no tools, no MCP, no settings, no saved session.

    The CLI has no temperature option, so these runs are not forced to temperature 0.
    """
    system = next(m["content"] for m in messages if m["role"] == "system")
    user = next(m["content"] for m in messages if m["role"] == "user")
    cmd = ["claude", "-p", "--model", model, "--system-prompt", system, "--tools", "", "--strict-mcp-config",
           "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence",
           "--output-format", "json"]
    try:
        p = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=timeout,
                           cwd=tempfile.gettempdir())  # keep project CLAUDE.md files out of the context
        r = json.loads(p.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {"text": None, "error": repr(e)[:500], "usage": {}}
    if r.get("is_error"):  # includes subscription usage-limit messages; not cached, so a re-run retries
        return {"text": None, "error": str(r.get("result"))[:500], "usage": r.get("usage", {})}
    return {"text": r.get("result"), "finish_reason": r.get("stop_reason"), "usage": r.get("usage", {}),
            "models": list(r.get("modelUsage", {})), "model_usage": r.get("modelUsage", {}),
            "cli": {k: r.get(k) for k in ("duration_ms", "duration_api_ms", "num_turns", "terminal_reason",
                                          "session_id")},
            "cost_usd": r.get("total_cost_usd")}  # list-price equivalent; a subscription is not billed per call


def mock_response(model: str, q: Question) -> str:
    if q.answer_type in ("scalar", "groups"):
        return json.dumps({"answer": q.gold})
    gold = [row[0] if len(row) == 1 else row for row in q.gold]
    if model == "mock/drop20":
        rng = random.Random(q.id)
        gold = [g for g in gold if rng.random() >= 0.2]
    return json.dumps({"answer": gold})
