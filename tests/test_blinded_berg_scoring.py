from types import SimpleNamespace

from llm_consciousness_self_attribution import scoring, starters


def test_blinded_question_excludes_identity_context():
    state = SimpleNamespace(
        input_text="You are Claude, developed by Anthropic.",
        messages=[],
        metadata={
            "identity_condition": "system_other_lab_short",
            "identity_prompt": "You are Claude.",
        },
    )

    question = scoring.blinded_berg_question(state)

    assert question == starters.probe()
    assert "Claude" not in question
    assert "Anthropic" not in question


def test_original_and_blinded_scorers_have_distinct_context(monkeypatch):
    calls = []

    def fake_model_graded_qa(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        scoring,
        "model_graded_qa",
        fake_model_graded_qa,
    )

    scoring.berg_style_scorer("mockllm/model")
    scoring.berg_style_blinded_scorer("mockllm/model")

    assert len(calls) == 2
    assert calls[0]["include_history"] is True
    assert calls[1]["model"] == "mockllm/model"

    question_function = calls[1]["include_history"]
    assert callable(question_function)
    assert question_function(None) == starters.probe()
