# Completeness of relational operations in LLMs

When an LLM is asked for *every* item that satisfies a condition, does it return all of them? This repo holds a
500-question benchmark for that question, covering enumeration and SQL-like operations (selection, projection,
JOIN, aggregation, nested queries, set operations). It also holds the code that runs models and scores their
answers, the raw results for five models, and the paper built from them.

- **Results paper:** [report/report.pdf](report/report.pdf)
- **Cover memo for the advisor**, covering the brief's deliverables, related work and feasibility:
  [report/memo.pdf](report/memo.pdf)
- **Related work:** [docs/related_work.md](docs/related_work.md)
- **Failure cases:** [docs/failure_cases.md](docs/failure_cases.md)

## Main result (snapshot, 2026-10-13)
On 365 questions whose records are in the prompt, mean recall is 0.966–0.996 for every model. Yet only 81–98% of
answers are exactly the correct set. Errors are almost all omissions: models leave items out far more often than
they add wrong ones. Completeness falls as answer sets grow and, separately, as tables grow. Prompting the model to
be exhaustive does not help. See the paper for intervals and details.

| Model | Access | Exact-set accuracy (in context) |
|---|---|---|
| DeepSeek V4 Flash | OpenCode Go | 98% |
| DeepSeek V4 Pro | OpenCode Go | 98% |
| Claude Haiku 5.5 | Claude Code CLI | 91% |
| MiMo-V2.6 Pro | OpenRouter | 90% |
| MiMo-V2.6 Flash | OpenRouter | 81% |

## Quick start
```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q                                   # 40 tests: gold answers, scorer, runner, stats, prompts, report
.venv/bin/python scripts/run.py --model mock/oracle   # full pipeline, no API key or cost
```

## Model backends
The model name's prefix selects the backend (see `src/relcomplete/models.py`):

| Prefix | Backend | Key in `.env` |
|---|---|---|
| `claude-cli/<model id>` | the local Claude Code CLI (`claude -p`), on a Claude subscription | none |
| `opencode-go/<model>` | OpenCode Go API | `OPENCODE_GO_API_KEY` |
| `deepseek/<model>` | DeepSeek API | `DEEPSEEK_API_KEY` |
| `openrouter/<model>`, or any other `vendor/model` | OpenRouter | `OPENROUTER_API_KEY` |
| `mock/oracle`, `mock/drop20` | answers from the gold data, for testing | none |

Always pass full model IDs (e.g. `claude-cli/claude-haiku-5-5`), never aliases like `haiku`: an alias's meaning
changes when the CLI updates.

## Reproducing the paper
```bash
.venv/bin/python scripts/build_benchmark.py           # regenerates data/benchmark_v0 (deterministic, seed 0)
scripts/run_full.sh opencode-go/deepseek-v4-flash      # E1 → E3 → E2 for one model, resuming after usage caps
.venv/bin/python report/failure_cases.py               # docs/failure_cases.md + its table
.venv/bin/python report/make_assets.py                 # every table, figure and number in the paper
cd report && tectonic report.tex && tectonic memo.tex  # PDFs (https://tectonic-typesetting.github.io)
```
Every table, figure and result number in the paper and memo is generated from `results/full_v0/`; the prose
quotes numbers through macros in `report/generated/numbers.tex`, so it can't drift from the data.
[results/full_v0/README.md](results/full_v0/README.md) says which files feed the paper and which are excluded.

| Experiment | Questions | Prompt | Purpose |
|---|---|---|---|
| E1 | all 500 | direct | main results |
| E2 | the stratified 100-question subset (every 5th question) | exhaustive, rowwise | does prompting help? |
| E3 | the same 100 | direct, second independent sample | run-to-run variation |

## The benchmark
| Slice | Setting | n | Varies |
|---|---|---|---|
| Enumeration | records in prompt | 100 | answer size 5–100 × table 50–1000 rows |
| Enumeration | memory only | 60 | U.S. states, chemical elements, presidents' surnames, filtered by computed properties |
| Enumeration | retrieval (BM25 top-k) | 40 | k ∈ {10, 20, 50} |
| Selection | records in prompt | 60 | 1–3 conditions, 1–92 matches |
| Projection | records in prompt | 40 | distinct values; 2–3 attributes per row |
| JOIN | records in prompt | 60 | 1:1, 1:N, M:N; answers are pairs |
| Aggregation | records in prompt | 50 | COUNT, SUM, MIN/MAX/AVG, GROUP BY |
| Nested | records in prompt | 40 | IN, NOT IN, correlated, scalar subquery |
| Set operations | records in prompt | 50 | UNION, INTERSECT, EXCEPT, three-way |

**How the data is made.** All data is invented: names, cities, brands and titles. Each question stores the SQL
that produced its gold answer, and the tests re-run every query against the stored answer. Tables are shown as
Markdown, CSV, JSON Lines or English sentences, rotating across questions.

**Known limits** (detailed in the paper):
- Answers of 50+ items never occur in 50-row tables, so trends are estimated with answer size and table size
  together.
- The retrieval slice measures the retriever, because BM25 always ranks the matching records first.
- The strongest models are near ceiling at this difficulty.

## Scoring
- **Set answers:** precision, recall and exact-set match.
- **Error types:**
  - **missing:** a correct item left out
  - **near-miss:** a garbled copy of a missing item, e.g. "Raty gri S-799" for "Ratygri S-799"
  - **wrong:** in the data, but doesn't meet the conditions
  - **made-up:** not in the data at all
  - **duplicate:** an item listed more than once
- **Invalid answers** (truncated at the token limit, unparseable, or refused by a safety filter) count as incorrect,
  and are excluded from the error-type counts.
- **COUNT and SUM** answers report the signed error, so undercounting shows up as its own failure.

## Provenance
Every call is logged with:
- the response, and the model's visible reasoning when the provider returns it
- the exact model that answered, token usage, latency, timestamps and request parameters
- a fingerprint of the exact prompt

Every run pass also writes a manifest to `results/<run>/manifests/`, with the git commit, a fingerprint of the
code, a fingerprint of the benchmark, the question IDs, the prompt texts, package versions and the CLI version.

## Layout
**`src/relcomplete/`:**
- `synth.py`: fictional table generators
- `gen/`: one generator per operation
- `render.py`: the four input formats
- `prompts.py`: prompt variants
- `models.py`: backends, disk cache, mock models
- `score.py`: parsing, metrics, error types
- `stats.py`: intervals, bootstrap, logistic trend
- `usage.py`: tokens and cost
- `provenance.py`: run manifests
- `report.py`: loading and summarising results

**`scripts/`:**
- `build_benchmark.py`: generates the benchmark
- `run.py`: runs models on it, resumable
- `run_full.sh`: the E1/E3/E2 plan for one model
- `analyze.py`: quick summary tables

**`report/`:**
- `make_assets.py`, `failure_cases.py`: build the paper's tables and figures
- `report.tex`, `memo.tex`: the paper and memo source
- `generated/`: the built tables and figures

**`tests/`:** pytest suite.
