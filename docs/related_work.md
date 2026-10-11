# Related work: reading list (not yet verified)

Papers to check during the related-work pass. Nothing here has been confirmed yet; check every citation and
summary against the paper itself before using it. For each one, the key question is: **does it measure
completeness (recall over the full answer set), and how that changes with answer-set size?**

| Area | Candidates to check | What to look for |
|---|---|---|
| Multi-answer QA | QAMPARI; RoMQA; AmbigQA | Do answer counts reach tens of items? Is recall reported? |
| Entity-set queries | QUEST (entity-seeking queries with implicit set operations) | Closest to our set-operation slice |
| Long-context, SQL-like tasks | LOFT (includes SQL-like queries over a corpus); Michelangelo | **Biggest novelty risk** |
| Long-context retrieval | Multi-needle needle-in-a-haystack; Lost in the Middle; BABILong; RULER | Many-needle recall versus number of needles |
| Table QA | WikiTableQuestions; TabFact; TableBench | Are answers single values or sets? |
| Text-to-SQL | Spider; BIRD | Model writes SQL; it doesn't execute the query itself |
| LLMs as query operators | LOTUS (semantic operators); "LLM as database" papers | Quality metrics per operator |
| Exhaustive enumeration | Work on LLM list completion and knowledge-base recall | Memory-only recall |

The deliverable for this file: a short summary per paper, and a final paragraph on which gap our benchmark fills.
