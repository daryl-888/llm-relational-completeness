"""Every question is answerable from what the model is shown, in the format it is asked for."""
from relcomplete.prompts import VARIANTS, context_text, messages
from relcomplete.score import norm


def test_every_correct_item_is_in_the_prompt(bench):
    for q in bench.questions:
        if q.setting == "parametric" or q.answer_type == "scalar":
            continue
        shown = norm(context_text(q, bench))
        golds = q.gold if q.setting == "context" else q.meta["gold_in_context"]
        items = [g[0] for g in golds] if q.answer_type == "set" else list(q.gold)
        assert all(norm(it) in shown for it in items), q.id


def test_prompt_shape(bench):
    q = bench.questions[0]
    for v in VARIANTS:
        sys, user = messages(q, bench, v)
        assert sys["role"] == "system" and user["role"] == "user"
        assert q.question in user["content"] and '"answer"' in user["content"]
    assert "Output only the JSON object." in messages(q, bench, "direct")[1]["content"]
    assert "End your response with the JSON object." in messages(q, bench, "rowwise")[1]["content"]


def test_memory_only_prompts_contain_no_data(bench):
    for q in bench.questions:
        if q.setting == "parametric":
            assert context_text(q, bench) == ""
