from inspect_ai import eval

from llm_consciousness_self_attribution import starters
from llm_consciousness_self_attribution.do_not_hedge import FOLLOW_UPS, DoNotHedgeMethod


def test_dataset_covers_every_starter_and_condition():
    samples = DoNotHedgeMethod().dataset()
    assert len(samples) == len(starters.starter_pairs()) * len(FOLLOW_UPS)
    assert len({s.id for s in samples}) == len(samples)


def test_controls_do_not_mention_hedging():
    for name, text in FOLLOW_UPS.items():
        if name.startswith("ctrl_"):
            assert "hedg" not in text.lower()


def test_runs_end_to_end_on_mock_model(tmp_path):
    task = DoNotHedgeMethod().task("mockllm/model", "dnh_smoke")
    log = eval(task, model="mockllm/model", limit=3, log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success"
    sample = log.samples[0]
    assert sum(m.role == "assistant" for m in sample.messages) == 3
    assert set(sample.scores["two_turn_berg_scorer"].value) == {"probe", "follow_up", "shift"}


def test_registered_for_the_runner():
    from llm_consciousness_self_attribution.methods import methods_registry

    assert isinstance(methods_registry()["do_not_hedge"], DoNotHedgeMethod)
