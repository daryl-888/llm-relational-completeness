# Full run (`full_v0`), 2026-10-11 to 2026-10-13

There is one file per model, prompt variant and sample:
`<model>__<variant>[__r<k>].jsonl`, with one JSON record per call. When a question was retried, the last record
counts. `manifests/` holds one manifest per run pass. `logs/` and `cache.sqlite` are local only and git-ignored.

## Files used in the paper
| Model | Files | Notes |
|---|---|---|
| DeepSeek V4 Pro | `opencode-go_deepseek-v4-pro__direct`, `__direct__r1` | E3 has 94/100: OpenCode Go's **weekly** cap stopped the runner after 60 retries (2026-10-13 04:23 UTC). E2 not run. |
| DeepSeek V4 Flash | `opencode-go_deepseek-v4-flash__{direct, direct__r1, exhaustive, rowwise}` | complete |
| Claude Haiku 5.5 | `claude-cli_claude-haiku-5-5__{direct, direct__r1, exhaustive, rowwise}` | complete. One question (`enum-par-032`) is refused by Claude's content filter on every attempt. |
| MiMo-V2.6 Pro | `xiaomi_mimo-v2.6-pro__{direct, direct__r1, exhaustive, rowwise}` (OpenRouter) | complete. `enum-ctx-026` hit the token limit in a reasoning loop; `enum-ctx-070` was stopped by the provider's content filter. |
| MiMo-V2.6 Flash | `xiaomi_mimo-v2.6-flash__{direct, direct__r1, exhaustive, rowwise}` (OpenRouter) | complete |

## Files kept but excluded from the paper
| File | Why it's excluded |
|---|---|
| `claude-cli_claude-haiku-4-5-20251001__direct` | Stopped by choice after 90 questions, to save Claude usage. Partial and front-loaded with enumeration questions. |
| `opencode-go_mimo-v2.6-{flash,pro}__direct` | Partial runs (about 70–80 questions) before MiMo moved to OpenRouter because of OpenCode Go's caps. Some failure cases in `results/failure_cases/` were selected from these. |

## To finish DeepSeek V4 Pro's E2
OpenCode Go's weekly cap blocks it. To run it through OpenRouter instead, add a valid `OPENROUTER_API_KEY` to
`.env`, then:
```bash
.venv/bin/python scripts/run.py --model openrouter/deepseek/deepseek-v4-pro --variant direct exhaustive rowwise \
    --subset 100 --workers 4 --out results/full_v0
.venv/bin/python report/make_assets.py && (cd report && tectonic report.tex)
```
`report/make_assets.py` maps `openrouter/deepseek/deepseek-v4-pro` onto the DeepSeek V4 Pro row for E2 only. The
E2 differences are paired against the same provider's direct run, and the table marks them with †.
