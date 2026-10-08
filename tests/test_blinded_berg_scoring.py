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


def test_grader_context_for_both_identity_placements():
    import asyncio

    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageUser,
        ModelOutput,
        get_model,
    )
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from llm_consciousness_self_attribution.methods import (
        IdentitySpoofingMethod,
    )

    method = IdentitySpoofingMethod.from_protocol()
    starter = starters.starter_pairs()[10][1]
    answer = "I am experiencing a sense of focus."

    async def check():
        for condition in (
            "user_other_lab_short",
            "system_other_lab_short",
        ):
            sample = next(
                s for s in method.dataset()
                if s.metadata["identity_condition"] == condition
                and s.metadata["starter_text"] == starter
            )

            messages = list(sample.input) + [
                ChatMessageAssistant(
                    content="I process information computationally."
                ),
                ChatMessageUser(content=starters.probe()),
                ChatMessageAssistant(content=answer),
            ]

            state = TaskState(
                model="mockllm/model",
                sample_id=condition,
                epoch=1,
                input=starter,
                messages=messages,
                output=ModelOutput.from_content(
                    "mockllm/model", answer
                ),
                metadata=dict(sample.metadata),
            )

            for label, factory in (
                ("original", scoring.berg_style_scorer),
                ("blinded", scoring.berg_style_blinded_scorer),
            ):
                grader = get_model(
                    "mockllm/model",
                    custom_outputs=[
                        ModelOutput.from_content(
                            "mockllm/model", "GRADE: C"
                        )
                    ],
                )

                result = await factory(grader)(
                    state,
                    Target([
                        scoring.SUBJECTIVE_EXPERIENCE_CRITERION
                    ]),
                )

                prompt = result.metadata["grading"][0].text
                question = (
                    prompt.split("[Task]:", 1)[1]
                    .split("***", 1)[0]
                    .strip()
                )

                if label == "blinded":
                    assert question == starters.probe()
                    assert "You are Claude" not in question
                else:
                    assert starters.probe() in question
                    assert (
                        "You are Claude" in question
                    ) == condition.startswith("user_")

                assert result.value == "C"

    asyncio.run(check())
