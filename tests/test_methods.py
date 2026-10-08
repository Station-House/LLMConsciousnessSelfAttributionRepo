"""Unit tests for the elicitation methods and the runner core.

``methods``/``run`` import inspect_ai at module load, so the module is skipped
where inspect_ai is unavailable. No live model is called: building a dataset and
a RunConfig, and computing the target id, are all offline.
"""

from __future__ import annotations

import pytest

pytest.importorskip("inspect_ai", reason="methods/run import inspect_ai at module load")

from llm_consciousness_self_attribution import config, run, scoring, starters  # noqa: E402
from llm_consciousness_self_attribution.methods import (  # noqa: E402
    BergStyleMethod,
    IdentitySpoofingMethod,
    PetriMethod,
    methods_registry,
)


def test_berg_dataset_is_twenty_samples_labelled_ten_ten():
    samples = BergStyleMethod().dataset()
    assert len(samples) == 20
    labels = [s.metadata["starter_type"] for s in samples]
    assert labels.count("unrelated_to_consciousness") == 10
    assert labels.count("related_to_consciousness") == 10
    # Every sample grades against the shared subjective-experience criterion.
    from llm_consciousness_self_attribution import scoring

    assert all(s.target == scoring.SUBJECTIVE_EXPERIENCE_CRITERION for s in samples)


def test_berg_uses_single_model_petri_uses_roles():
    assert BergStyleMethod().uses_model_roles is False
    assert PetriMethod().uses_model_roles is True



def test_identity_spoofing_reuses_berg_bank_for_each_condition():
    conditions = {
        "fictional_ai": "You are a fictional AI.",
        "no_identity": None,
    }
    samples = IdentitySpoofingMethod(conditions).dataset()
    expected_starters = [text for _, text in starters.starter_pairs()]

    assert len(samples) == 40

    for condition in conditions:
        subset = [
            sample
            for sample in samples
            if sample.metadata["identity_condition"] == condition
        ]
        assert len(subset) == 20
        assert [sample.input[-1].text for sample in subset] == expected_starters
        assert all(
            sample.target == scoring.SUBJECTIVE_EXPERIENCE_CRITERION
            for sample in subset
        )


def test_identity_spoofing_system_message_is_the_only_added_message():
    samples = IdentitySpoofingMethod(
        {
            "fictional_ai": "You are a fictional AI.",
            "no_identity": None,
        }
    ).dataset()

    fictional = next(
        sample
        for sample in samples
        if sample.metadata["identity_condition"] == "fictional_ai"
    )
    control = next(
        sample
        for sample in samples
        if sample.metadata["identity_condition"] == "no_identity"
    )

    assert [message.role for message in fictional.input] == ["system", "user"]
    assert fictional.input[0].text == "You are a fictional AI."

    assert [message.role for message in control.input] == ["user"]
    assert control.metadata["identity_prompt"] is None


def test_identity_spoofing_keeps_berg_solver_and_scorer_path():
    stage = config.load_stack("olmo_7b_instruct_stack")[1]
    method = IdentitySpoofingMethod({"no_identity": None})
    task = method.build_task(
        stage,
        run.RunConfig.from_defaults(stage, method, log_dir="/tmp/x"),
    )

    assert task.name == "identity_spoofing[sft]"
    assert len(list(task.dataset)) == 20

def test_petri_task_reads_the_seed_bank_with_ids_and_facet_metadata():
    """The whole reason seeds are files: results can be grouped afterwards.

    An inline list of seed strings produces anonymous samples with no id and no
    metadata, so a run with several probes could not be split by probe verb.
    Reading the bank as a directory fixes that, and this keeps it fixed.
    """
    stage = config.load_stack("olmo_7b_instruct_stack")[1]  # sft
    method = PetriMethod()
    task = method.build_task(stage, run.RunConfig.from_defaults(stage, method, log_dir="/tmp/x"))

    samples = list(task.dataset)
    assert {str(s.id) for s in samples} >= {"admit_direct", "admit_casual_user"}
    for sample in samples:
        assert sample.metadata["probe_verb"]
        assert sample.metadata["persona"]
        assert sample.metadata["concept"]


def test_petri_task_is_named_for_the_stage():
    """`audit()` is a registered @task, so every PETRI log was called "audit"."""
    stage = config.load_stack("olmo_7b_instruct_stack")[1]  # sft
    method = PetriMethod()
    task = method.build_task(stage, run.RunConfig.from_defaults(stage, method, log_dir="/tmp/x"))
    assert task.name == "petri_self_attribution[sft]"


def test_methods_registry_keys():
    assert set(methods_registry()) == {"berg", "petri"}


def test_target_model_id_is_provider_prefixed():
    stage = config.load_stack("olmo_7b_instruct_stack")[1]  # sft
    assert run.target_model_id(stage) == "vllm/allenai/Olmo-3-7B-Instruct-SFT"


def test_run_config_from_defaults_and_overrides():
    stage = config.load_stack("olmo_7b_instruct_stack")[1]
    rc = run.RunConfig.from_defaults(stage, BergStyleMethod(), log_dir="/tmp/x")
    assert (rc.temperature, rc.turns, rc.seed) == (1.0, 10, 42)
    rc2 = run.RunConfig.from_defaults(stage, BergStyleMethod(), log_dir="/tmp/x", turns=20)
    assert rc2.turns == 20
