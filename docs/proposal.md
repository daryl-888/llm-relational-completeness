# Benchmark proposal: completeness-sensitive relational operations

> **Status (2026-10-13):** this is the original proposal from 2026-10-11, kept for reference. What was actually
> run differs in a few ways:
> - **Models:** DeepSeek V4 Pro and Flash (OpenCode Go), Claude Haiku 5.5 (Claude Code CLI), and MiMo-V2.6 Pro
>   and Flash (OpenRouter), not the four listed under "Experiments".
> - **Repeat runs:** a second sample of the subset (E3) was added.
> - **Analysis:** adjusts for table size and input format.
>
> See [the results paper](../report/report.pdf) and [the cover memo](../report/memo.pdf).


## Question
When the information needed is available (in the model's training data, in the prompt, or in retrieved
documents), does an LLM return *all* the items that satisfy a request, and nothing else? The main claim to test:
**recall falls as the answer set grows, even when every record is in the context.**

## Design choices
- **The "records in prompt" setting with synthetic data is the core.** Every record is visible, and the entities are
  invented, so a missing item can't be blamed on missing knowledge, retrieval misses or out-of-date reference data.
  The answer set is exact because it's produced by running stored SQL in SQLite.
- **Answer size and table size are controlled separately.** For enumeration, exactly *g* of the *n* rows match,
  with *g* ∈ {5, 10, 20, 50, 100} and *n* ∈ {50, 200, 500, 1000}. Matching rows sit at random positions, and the
  positions are stored, so "lost in the middle" effects can be checked.
- **Memory-only questions use closed lists filtered by computed properties.** For example: states whose name
  ends in "a", elements with atomic number 21–36. The gold answer is computed by code from a fixed list, never
  written by hand.
- **The RAG setting separates retrieval from reading.** BM25 retrieves the top-k records. The scorer reports the
  model's recall over the gold items that were actually retrieved, not only over the full answer.
- **Four input formats** (Markdown, CSV, JSON Lines, English sentences) rotate across questions, so results don't
  depend on one way of showing the data.

## Task types
| Operation | Example |
|---|---|
| Enumeration | List all employees whose department is Legal. |
| Selection | …whose release year is between 2018 and 2023, whose brand is X and whose stock is below 426. |
| Projection | List every distinct office city. / For every product of brand X, give its name, release year and category. |
| JOIN | Every (customer, order ID) pair where the customer is Gold tier and the order is from 2023. |
| Aggregation | How many…? Total…? Highest…? For each genre, how many…? |
| Nested | Employees whose city has no Completed project; products priced above their category's average. |
| Set operations | Members of the Chess Club but not the Drama Club. |

## Metrics
- **Set answers:** precision, recall, F1, exact-set match, and counts of missing, wrong, made-up and duplicate items.
- **Tuple answers:** also key recall.
- **Aggregation:** exact match, signed error and relative error.
- **Failures:** answers that can't be parsed, API errors and truncated outputs are reported separately.

## Experiments
1. All 500 questions with the `direct` prompt on 4 models: Claude Haiku 4.5, Claude Sonnet 5.5, an inexpensive GPT
   model, and an open-weight model (Llama 3.3 70B or Qwen).
2. A stratified 100-question subset with the `exhaustive` and `rowwise` prompts, to see whether prompting alone
   closes the gap.
3. Main figures:
   - recall versus answer-set size, and versus table size (enumeration in context)
   - recall by operation
   - error-type breakdown

Cost: about 2.35M input tokens per model per prompt variant over the full set.

## Extensions if the effect holds
- Real-world data with checkable answers (e.g. Wikidata snapshots).
- Larger tables, up to the limit of the context window.
- Questions with zero correct answers, to measure how often models make items up.
- Ordered output (top-k), and harder RAG queries where BM25 also misses relevant records.
