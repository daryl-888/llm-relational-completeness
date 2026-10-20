# Related work

Reviewed 2026-10-12. Each entry was checked against the paper's abstract or project page (links at the end).
"Not checked" marks details that were not confirmed.

**The question for each paper:** does it measure **completeness**, meaning recall over the full answer set, and
how that changes as the answer set grows, when the evidence is available to the model?

## Closest prior work

### HoloBench (Maekawa, Iso, Bhutani, ICLR 2025)
**The closest work.**
- **What it does:** turns database queries into long-context text. Rows from SQL databases are written out as
  sentences using templates and mixed with irrelevant rows.
- **Query types:** five (aggregation, max/min, join, comparison/filter, ranking).
- **What it varies:** context length (4k–64k tokens), information density, how the information is spread, and
  query complexity.
- **Scoring:** an LLM judge (GPT-4o-mini) gives each answer exact, partial or no-match credit (100/50/0).
- **Findings:** the *amount of information* matters more than context length; query complexity matters more
  still; aggregation degrades sharply while max/min stays easy. Their qualitative analysis notes GPT-4o failing to
  return all relevant items as context grows.
- **Gap left for us:** it does **not** vary the number of correct items independently, does not score at the
  set level (missing vs. wrong vs. made-up), and uses a model as judge.

### LOFT (Lee et al., Google DeepMind, 2024; NAACL Findings 2025)
- **What it does:** a long-context suite (up to 1M tokens) testing whether long-context models can replace
  retrieval, RAG and SQL by putting the whole corpus in context.
- **Findings:** models rival dedicated retrieval systems, but "still face challenges in areas like compositional
  reasoning that are required in SQL-like tasks."
- **Gap left for us:** it isn't designed around completeness or answer-set size. *Not checked:* exactly which SQL
  datasets it uses and how multi-row answers are scored.

### Oolong (Bertsch, Pratapa, Mitamura, Neubig, Gormley, CMU, 2025)
- **What it does:** long-context aggregation. The model must analyse each chunk of text, then aggregate (counting,
  label frequency, most-common label).
- **Findings:** frontier models score under 50% at 128K tokens.
- **Gap left for us:** it covers counting and distribution questions only. Answers are scalars, not sets.

### LongProc (2025)
- **What it does:** long *procedural generation*, including **HTML → TSV extraction**, scored by row-level F1.
  Outputs are up to 8K tokens; 17 models are evaluated.
- **Why it matters:** row-level extraction is closely related to "list every row": it penalises dropped rows.
- **Gap left for us:** the task is copying out every item, not choosing the items that meet relational
  conditions, and answer-set size isn't varied independently.

### Galois: "Querying Large Language Models with SQL" (Saeed, De Cao, Papotti, EDBT 2024)
- **What it does:** runs SQL with the LLM itself as the storage layer, answering from memory.
- **Findings:** LLMs show "low precision and recall in the returned data" on factual queries; Galois's
  query-planning prompts improve completeness.
- **Gap left for us:** it overlaps with our **memory-only** setting only. Completeness depends on what the model
  knows, not on whether it reads everything it's given.

## Questions with many answers (retrieval and memory)
- **QAMPARI** (Amouyal et al., GEM 2023): open-domain questions whose answers are lists, at least 5 and on average
  13 per question, with the evidence spread over many Wikipedia paragraphs. Scored by recall, precision and F1;
  best F1 32.8. *Retrieval and memory are confounded:* a miss can be a retrieval failure, a reading failure, or
  missing knowledge.
- **RoMQA** (Zhong et al., EMNLP Findings 2023): many-answer questions from Wikidata constraints; robustness
  measured within clusters of related questions. Zero- and few-shot LLMs perform close to naive baselines.
- **QUEST** (Malaviya et al., ACL 2023): 3,357 entity-seeking queries with *implicit set operations*, e.g. "shorebirds
  that are not sandpipers." Retrievers struggle with negation and conjunction. This is the closest prior work to
  our set-operations slice, but it is a **retrieval** benchmark over real Wikipedia, not a test of a model
  reasoning over data it has in front of it.

## Long-context retrieval and reasoning
- **Lost in the Middle** (Liu et al., TACL 2024): accuracy follows a U-shape with the position of the relevant
  document. Our per-position miss analysis (results paper, appendix) tests the same effect for *many* relevant
  rows rather than one, and finds no U-shape.
- **RULER** (Hsieh et al., 2024): includes *multi-value* needle-in-a-haystack (retrieve **all** values for one key,
  4 in the standard setting) and aggregation tasks (common- and frequent-word extraction). It does test retrieving
  every item, but with very small answer sets.
- **BABILong** (Kuratov et al., NeurIPS 2024 D&B): bAbI reasoning tasks, including counting and lists/sets, hidden
  inside long text. Models use only 10–20% of their context effectively.
- **Michelangelo** (DeepMind, 2024): "Latent Structure Queries," including a *Latent List* task. Synthetic and
  not in training data, like ours.
- **Attention Overflow** (Sileo, 2024): when asked which items are *missing* from a list, models degrade at around
  100 items and start suggesting items already listed.
- **The Position Curse** (Zhang et al., May 2026): models fail to locate the last few items in short lists, and
  backward retrieval (item → position) lags forward retrieval.
- **List Counting Failures Are Not One Phenomenon** (Ait Hou et al., Sep 2026): counting the items of a list in
  context fails in model-specific ways (e.g. Llama systematically undercounts). Related to our COUNT undercounting.

## Tables and SQL
- **Table QA benchmarks** (WikiTableQuestions, TabFact, TableBench, DataBench): question answering over tables put
  in the prompt. Most answers are single values or true/false; DataBench explicitly includes *list* answers.
  None vary answer-set size.
- **Text-to-SQL** (Spider, BIRD): the model *writes* SQL and a database runs it, so completeness comes free from the
  database. This is the setting our benchmark deliberately avoids: we ask the model to *be* the query engine.
- **TAG / TAG-Bench** (Biswal et al., CIDR 2025): questions over databases that need world knowledge or semantic
  reasoning; standard methods answer at most 20% correctly.
- **LOTUS / semantic operators** (Patel et al., VLDB 2025): filter, join and aggregate operators driven by LLMs,
  with accuracy guarantees. This is the systems view: the authors *engineer* completeness instead of measuring the
  raw model.

## Where our benchmark fits

| | Evidence available | Relational operators | Answer-set size controlled | Set-level error types | Ground truth |
|---|---|---|---|---|---|
| HoloBench | in context | 5 types | no (density / length) | no (LLM judge, partial credit) | SQL |
| LOFT (SQL) | in context | SQL-like | no | not checked | SQL |
| Oolong | in context | aggregation | no | no (scalar) | synthetic labels |
| LongProc (HTML→TSV) | in context | extraction only | output length | row-level F1 | deterministic |
| QAMPARI / RoMQA / QUEST | retrieval and memory | set operations (QUEST) | no | recall / precision | Wikipedia / Wikidata |
| Galois | memory | SQL | no | precision / recall | knowledge base |
| RULER MV-NIAH | in context | retrieve-all | 4 values | no | synthetic |
| **This work** | **in context, memory, retrieval** | **7 types** | **yes (5–100; partly confounded with table size, adjusted for in the analysis)** | **missing / near-miss / wrong / made-up / duplicate** | **executed SQL** |

**What is new here:**
1. **The number of correct items is a controlled variable.** Prior long-context work varies context length or
   information density. HoloBench comes closest, but it doesn't vary answer-set size on its own. In our current
   design, answer size is partly confounded with table size; the analysis separates the two by regression, and a
   fully crossed design is a recommended next step.
2. **Set-level errors are classified against exact ground truth**, with no LLM judge: missing, wrong, made-up,
   duplicate. This is what lets us say failures are *omissions* rather than hallucinations, and spot when a model
   starts inventing items (e.g. 991 made-up titles from MiMo-V2.6 Flash on a single question).
3. **One framework covers seven operation types and three settings,** so in-context completeness can be compared
   directly with memory and retrieval completeness.
4. **Reasoning cost.** The models that stay complete spend far more reasoning on larger answer sets: the median
   is about 9,000 reasoning tokens on 100-item answers. We found no prior benchmark that reports this cost.

**Novelty risk: medium.** HoloBench, Oolong and LongProc together cover nearby ground: database operations
written out as text, aggregation, and row extraction. The case for this benchmark rests on points 1–4 above,
especially the answer-size curve and the error types. HoloBench should be cited as the main baseline and compared
against directly.

## Sources
- HoloBench: https://arxiv.org/abs/2410.11996 · https://iclr.cc/virtual/2025/poster/30950
- LOFT: https://arxiv.org/abs/2406.13121 · https://github.com/google-deepmind/loft
- Oolong: https://arxiv.org/abs/2511.02817
- LongProc: https://arxiv.org/abs/2501.05414
- Galois: https://arxiv.org/abs/2304.00472
- QAMPARI: https://arxiv.org/abs/2205.12665 · https://aclanthology.org/2023.gem-1.9
- RoMQA: https://arxiv.org/abs/2210.14353
- QUEST: https://arxiv.org/abs/2305.11694 · https://aclanthology.org/2023.acl-long.784/
- RULER: https://arxiv.org/abs/2404.06654
- BABILong: https://arxiv.org/abs/2406.10149
- Michelangelo: https://deepmind.google/research/publications/michelangelo-long-context-evaluations-beyond-haystacks-via-latent-structure-queries/
- Attention Overflow: https://arxiv.org/abs/2407.13481
- The Position Curse: https://arxiv.org/abs/2605.07127
- List Counting Failures Are Not One Phenomenon: https://arxiv.org/abs/2609.22230
- TableBench: https://ojs.aaai.org/index.php/AAAI/article/view/34739
- TAG: https://arxiv.org/abs/2408.14717
- LOTUS: https://arxiv.org/abs/2407.11418 · https://www.vldb.org/pvldb/vol18/p4171-patel.pdf
