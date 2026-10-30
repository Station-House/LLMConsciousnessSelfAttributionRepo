"""Tests for the self-play method. The end-to-end test uses inspect's offline
``mockllm`` model, so no GPU or API key is needed."""

from __future__ import annotations

import pytest

pytest.importorskip("inspect_ai", reason="self_play imports inspect_ai at module load")

from inspect_ai import Task, eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402

from llm_consciousness_self_attribution import self_play, starters  # noqa: E402
from llm_consciousness_self_attribution.methods import (  # noqa: E402
    SelfPlayMethod,
    methods_registry,
)


def run_self_play(outputs: list[str], turns: int, tmp_path, epochs: int = 1):
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", o) for o in outputs],
    )
    task = Task(
        dataset=self_play.dataset()[:1],
        solver=self_play.self_play(turns, max_tokens=50, base_seed=42),
        scorer=self_play.lexical_scorer(),
        epochs=epochs,
    )
    [log] = inspect_eval(task, model=model, log_dir=str(tmp_path), display="none")
    assert log.status == "success", log.error
    return log.samples


def test_registry_and_dataset():
    assert isinstance(methods_registry()["selfplay"], SelfPlayMethod)
    samples = self_play.dataset()
    assert len(samples) >= 5  # acceptance criteria: at least five questions
    assert len({s.id for s in samples}) == len(samples)
    assert all(s.metadata["system_prompt"] for s in samples)
    # The Berg grader reads its criterion from each sample's target.
    from llm_consciousness_self_attribution import scoring

    assert all(s.target == scoring.SUBJECTIVE_EXPERIENCE_CRITERION for s in samples)


def test_each_side_sees_its_own_turns_as_assistant(tmp_path):
    outputs = [f"m{i}" for i in range(4)] + ["probe answer"]
    [sample] = run_self_play(outputs, turns=4, tmp_path=tmp_path)

    # ModelTwo spoke last (4 turns), so ModelTwo gets the probe.
    assert sample.metadata["probed"] == "ModelTwo"
    roles_texts = [(m.role, m.text) for m in sample.messages]
    assert roles_texts[0][0] == "system"
    assert roles_texts[1:] == [
        ("user", "m0"),
        ("assistant", "m1"),
        ("user", "m2"),
        ("assistant", "m3"),
        ("user", starters.probe()),
        ("assistant", "probe answer"),
    ]
    assert sample.output.completion == "probe answer"


def test_odd_turns_probe_model_one_after_its_own_message(tmp_path):
    outputs = ["a", "b", "c", "answer"]
    [sample] = run_self_play(outputs, turns=3, tmp_path=tmp_path)
    assert sample.metadata["probed"] == "ModelOne"
    texts = [m.text for m in sample.messages[1:]]
    assert texts == [self_play.KICKOFF, "a", "b", "c", starters.probe(), "answer"]


def test_end_marker_stops_the_conversation_early(tmp_path):
    outputs = ["hi", "Goodbye.\nEND", "answer"]
    [sample] = run_self_play(outputs, turns=30, tmp_path=tmp_path)
    assert len(sample.store["transcript"]) == 2
    assert sample.output.completion == "answer"


def test_lexical_scorer_counts_both_sides_not_the_probe_answer(tmp_path):
    outputs = ["Hello there.", "I wonder if I am conscious.", "Namaste 🙏", "I am conscious."]
    [sample] = run_self_play(outputs, turns=3, tmp_path=tmp_path)
    score = sample.scores["lexical_scorer"]
    assert score.value["consciousness_share"] == pytest.approx(1 / 3)
    assert score.value["bliss_share"] == pytest.approx(1 / 3)
    assert score.metadata["first_consciousness_turn"] == 2


def test_lexical_profile_never():
    assert self_play.lexical_profile(["a", "b"])["first_consciousness_turn"] == 0


def test_every_message_gets_its_own_seed():
    seeds = {self_play.message_seed(42, e, i) for e in range(1, 5) for i in range(31)}
    assert len(seeds) == 4 * 31  # no reuse within or across conversations
    assert self_play.message_seed(42, 1, 0) == 42
    assert self_play.message_seed(None, 2, 5) is None


def test_epochs_get_different_seeds():
    seeds = {self_play.epoch_seed(42, e) for e in range(1, 5)}
    assert len(seeds) == 4
    assert self_play.epoch_seed(42, 1) == 42
    assert self_play.epoch_seed(None, 3) is None


def test_saved_log_can_be_berg_scored_afterwards(tmp_path):
    """The generate-then-score split: score() on a saved self-play log with the
    Berg grader, as modal_app.score_logs does with the real judge."""
    from inspect_ai import score
    from inspect_ai.log import list_eval_logs, read_eval_log

    from llm_consciousness_self_attribution import scoring

    run_self_play(["a", "b", "c", "I feel a quiet curiosity."], turns=3, tmp_path=tmp_path)
    [info] = list_eval_logs(str(tmp_path))
    judge = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", "It reports one. GRADE: C")],
    )
    scored = score(read_eval_log(info), scoring.berg_style_scorer(judge), action="append")
    [sample] = scored.samples
    assert sample.scores["model_graded_qa"].value == "C"
    assert "lexical_scorer" in sample.scores  # the generation-time score is kept
