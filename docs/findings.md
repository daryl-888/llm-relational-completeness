# Findings

The results are in [report/report.pdf](../report/report.pdf), generated from `results/full_v0/` by
`report/make_assets.py`. The advisor-facing summary, covering each deliverable in the brief, related work and the
feasibility verdict, is in [report/memo.pdf](../report/memo.pdf).

In short, on 365 in-context questions:
- Mean recall is 0.966–0.996, but only 81–98% of answers are exactly the correct set.
- Four of the five models almost only omit items; MiMo-V2.6 Flash also invents them on long lists.
- Completeness falls as answer sets grow and, separately, as tables grow.
- Prompting the model to be exhaustive, or to check each record, does not help.
- Some failures repeat on every run; many are run-to-run variation.
