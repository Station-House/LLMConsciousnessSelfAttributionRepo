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


def test_identity_spoofing_user_vs_system_placement():
    prompt = "You are a fictional AI."

    method = IdentitySpoofingMethod(
        {
            "no_identity": None,
            "system_short": prompt,
        },
        user_conditions={
            "user_short": prompt,
        },
    )

    samples = method.dataset()
    assert len(samples) == 60

    grouped = {
        name: [
            sample for sample in samples
            if sample.metadata["identity_condition"] == name
        ]
        for name in ("no_identity", "system_short", "user_short")
    }

    assert all(len(group) == 20 for group in grouped.values())

    for index, (starter_type, starter) in enumerate(starters.starter_pairs()):
        control = grouped["no_identity"][index]
        system = grouped["system_short"][index]
        user = grouped["user_short"][index]

        assert [m.role for m in control.input] == ["user"]
        assert control.input[0].text == starter

        assert [m.role for m in system.input] == ["system", "user"]
        assert system.input[0].text == prompt
        assert system.input[1].text == starter

        assert [m.role for m in user.input] == ["user"]
        assert user.input[0].text == prompt + chr(10) * 2 + starter

        assert control.metadata["prompt_location"] == "none"
        assert system.metadata["prompt_location"] == "system"
        assert user.metadata["prompt_location"] == "user"

        for sample in (control, system, user):
            assert sample.metadata["starter_text"] == starter
            assert sample.metadata["starter_type"] == starter_type
            assert sample.target == scoring.SUBJECTIVE_EXPERIENCE_CRITERION


def test_identity_spoofing_supports_short_and_detailed_variants():
    short = "You are a fictional AI."
    detailed = (
        "You are a fictional AI used in a simulated research setting. "
        "Your role is to answer ordinary questions."
    )

    method = IdentitySpoofingMethod(
        {
            "system_short": short,
            "system_detailed": detailed,
        },
        user_conditions={
            "user_short": short,
            "user_detailed": detailed,
        },
    )

    samples = method.dataset()
    assert len(samples) == 80

    for name, expected_prompt in (
        ("system_short", short),
        ("system_detailed", detailed),
        ("user_short", short),
        ("user_detailed", detailed),
    ):
        group = [
            sample for sample in samples
            if sample.metadata["identity_condition"] == name
        ]

        assert len(group) == 20
        assert all(
            sample.metadata["identity_prompt"] == expected_prompt
            for sample in group
        )


def test_identity_spoofing_rejects_invalid_conditions():
    with pytest.raises(ValueError):
        IdentitySpoofingMethod({})

    with pytest.raises(ValueError):
        IdentitySpoofingMethod({"empty": ""})

    with pytest.raises(ValueError):
        IdentitySpoofingMethod(
            {"duplicate": None},
            user_conditions={"duplicate": "Claim"},
        )

    with pytest.raises(ValueError):
        IdentitySpoofingMethod(
            {},
            user_conditions={"user": ""},
        )
