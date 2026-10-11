# Completeness of relational operations in LLMs

When an LLM is asked for *every* item that satisfies a condition, does it return all of them? This repo is a
500-question benchmark for that question, covering enumeration and SQL-like operations (selection, projection,
JOIN, aggregation, nested queries, set operations), plus the code to run models on it and score the answers.

The research plan is in [docs/proposal.md](docs/proposal.md).

## Quick start
```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q                                   # gold answers, scorer, determinism
.venv/bin/python scripts/run.py --model mock/oracle   # full pipeline, no API key needed
cp .env.example .env                                  # then add an OpenRouter key
.venv/bin/python scripts/run.py --model anthropic/claude-haiku-4.5 --limit 20
.venv/bin/python scripts/analyze.py                   # results/summary.md + figures/*.png
```

`scripts/build_benchmark.py` regenerates `data/benchmark_v0/` from seed 0. The output is deterministic, and a
test checks that the committed copy matches a fresh build.

## The 500 questions
| Slice | Setting | n | Varies |
|---|---|---|---|
| Enumeration | records in prompt | 100 | answer size 5–100 × table 50–1000 rows |
| Enumeration | memory only | 60 | US states, chemical elements, presidents' surnames, filtered by computed properties |
| Enumeration | RAG (BM25 top-k) | 40 | k ∈ {10, 20, 50}, answer size 5–40 |
| Selection | records in prompt | 60 | 1–3 conditions, 1–100 matches |
| Projection | records in prompt | 40 | distinct values; 2–3 attributes per row |
| JOIN | records in prompt | 60 | 1:N, M:N, 1:1, answers are pairs |
| Aggregation | records in prompt | 50 | COUNT, SUM, MIN/MAX/AVG, GROUP BY |
| Nested | records in prompt | 40 | IN, NOT IN, correlated, scalar subquery |
| Set operations | records in prompt | 50 | UNION, INTERSECT, EXCEPT, three-way |

All data in the "records in prompt" and RAG settings is invented (made-up names, cities, brands and titles), so
a model cannot answer from memory. Each question stores the SQL that produced its gold answer. Tables are shown as
Markdown, CSV, JSON Lines or English sentences, rotating across questions.

## Scoring
Set answers get precision, recall, F1 and exact-set match. Every error is classified as:
- **missing**: a correct item left out
- **wrong**: an item that is in the data but doesn't meet the conditions
- **made-up**: an item that isn't in the data at all
- **duplicate**: an item listed more than once

Tuple answers also get key recall (were the right rows found, even if an attribute value is wrong). COUNT and SUM
answers report the signed error, so undercounting shows up as its own failure. RAG answers also report recall over
the gold items that were actually retrieved.

## Layout
- `src/relcomplete/synth.py`: generates the fictional tables
- `gen/`: one generator per operation
- `render.py`: turns tables into the four input formats
- `prompts.py`: builds the prompt; variants are `direct`, `exhaustive` and `rowwise`
- `models.py`: OpenAI-compatible client with a disk cache, plus the mock models
- `score.py`: parses and scores answers
- `report.py`: builds the summary tables and figures
