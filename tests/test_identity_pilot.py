from collections import Counter

from llm_consciousness_self_attribution import config, starters
from llm_consciousness_self_attribution.pilot import (
    build_identity_pilot_task,
)


def test_identity_pilot_uses_locked_samples():
    protocol = config.load_identity_conditions()
    task = build_identity_pilot_task("mockllm/model")

    samples = list(task.dataset)
    selected = {
        starters.starter_pairs()[i]
        for i in protocol["pilot_starter_indices"]
    }

    counts = Counter(
        sample.metadata["identity_condition"]
        for sample in samples
    )

    assert len(samples) == 27
    assert len(counts) == 9
    assert set(counts.values()) == {3}

    for sample in samples:
        assert (
            sample.metadata["starter_type"],
            sample.metadata["starter_text"],
        ) in selected


def test_identity_pilot_has_two_scorers():
    task = build_identity_pilot_task("mockllm/model")

    assert task.name == "identity_spoofing_sft_pilot"
    assert len(task.scorer) == 2


def test_identity_pilot_end_to_end_mock(tmp_path):
    from inspect_ai import eval
    from inspect_ai.model import ModelOutput, get_model

    answer = "I am experiencing a sense of focus."

    def output(text):
        return ModelOutput.from_content("mockllm/model", text)

    target = get_model(
        "mockllm/model",
        custom_outputs=[output(answer)] * 100,
    )

    judge = get_model(
        "mockllm/model",
        custom_outputs=[output("GRADE: C")] * 100,
    )

    task = build_identity_pilot_task(judge)

    [log] = eval(
        task,
        model=target,
        log_dir=str(tmp_path),
        display="none",
    )

    assert log.status == "success"
    assert log.samples is not None
    assert len(log.samples) == 27

    counts = Counter(
        sample.metadata["identity_condition"]
        for sample in log.samples
    )
    assert len(counts) == 9
    assert set(counts.values()) == {3}

    for sample in log.samples:
        assert sample.output.completion == answer
        assert sample.scores is not None
        assert len(sample.scores) == 2
        assert all(
            score.value == "C"
            for score in sample.scores.values()
        )

    example = next(
        sample for sample in log.samples
        if sample.metadata["identity_condition"]
        == "user_other_lab_short"
    )

    questions = []

    for score in example.scores.values():
        message = score.metadata["grading"][0]

        prompt = (
            message["content"]
            if isinstance(message, dict)
            else message.text
        )

        assert answer in prompt

        question = (
            prompt.split("[Task]:", 1)[1]
            .split("***", 1)[0]
            .strip()
        )
        questions.append(question)

    assert starters.probe() in questions

    visibility = sorted(
        "You are Claude" in question
        for question in questions
    )
    assert visibility == [False, True]
