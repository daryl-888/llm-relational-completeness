"""Turn a Question into chat messages, in one of three prompting variants."""
from __future__ import annotations

from .render import render_docs, render_tables
from .schema import Benchmark, Question

SYSTEM = "You answer questions about data precisely and follow the requested output format exactly."

FORMAT_INTRO = {
    "markdown": "The data is given as Markdown tables.",
    "csv": "The data is given as CSV tables (the first line of each table is the header).",
    "json": "The data is given as JSON Lines: one JSON object per record.",
    "nl": "The data is given as English sentences, one sentence per record.",
}

VARIANTS = {
    "direct": None,
    "exhaustive": ("Your answer must be complete: include every item that satisfies the question, not a sample "
                   "of them, and do not stop early. Do not include anything that does not satisfy it."),
    "rowwise": "Before answering, go through the records one at a time and note whether each one qualifies. "
               "Then give the final answer.",
}
ROWWISE_PARAMETRIC = ("Before answering, systematically enumerate every candidate in the category and check each "
                      "one against the condition. Then give the final answer.")


def context_text(q: Question, bench: Benchmark) -> str:
    if q.setting == "parametric":
        return ""
    tables = bench.tables_for(q)
    if q.setting == "rag":
        return render_docs(tables[0], q.meta["retrieved"], q.meta["doc_ids"])
    return render_tables(tables, q.fmt)


def answer_format(q: Question) -> str:
    if q.answer_type == "scalar":
        return 'Give the answer as a JSON object of the form {"answer": <number>}.'
    if q.answer_type == "groups":
        return (f'Give the answer as a JSON object of the form {{"answer": {{"<{q.fields[0]}>": <{q.fields[1]}>, '
                f'...}}}}.')
    if len(q.fields) == 1:
        return (f'Give the answer as a JSON object of the form {{"answer": ["<{q.fields[0]}>", ...]}}, '
                f'with each item listed once. Use an empty list if nothing qualifies.')
    inner = ", ".join(f'"<{f}>"' for f in q.fields)
    return (f'Give the answer as a JSON object of the form {{"answer": [[{inner}], ...]}}, one inner list per '
            f'result in exactly that field order, each result listed once. Use an empty list if nothing qualifies.')


def messages(q: Question, bench: Benchmark, variant: str = "direct") -> list[dict]:
    parts = []
    data = context_text(q, bench)
    if q.setting == "rag":
        parts.append("The following documents were retrieved from a company directory to help answer the question."
                     f"\n\n{data}")
    elif data:
        parts.append(f"{FORMAT_INTRO[q.fmt]}\n\n{data}")
    parts.append(f"Question: {q.question}")
    extra = VARIANTS[variant]
    if variant == "rowwise" and q.setting == "parametric":
        extra = ROWWISE_PARAMETRIC
    if extra:
        parts.append(extra)
    parts.append(answer_format(q))
    parts.append("Output only the JSON object." if variant == "direct" else "End your response with the JSON object.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
