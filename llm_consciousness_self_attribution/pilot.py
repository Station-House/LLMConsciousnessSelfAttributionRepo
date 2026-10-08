"""Pilot task for the identity-spoofing experiment."""

from collections import Counter

from inspect_ai import Task, task_with
from inspect_ai.model import Model

from . import config, scoring, starters
from .methods import IdentitySpoofingMethod


def build_identity_pilot_task(judge_model: str | Model) -> Task:
    """Build the locked pilot with two graders per model response."""
    protocol = config.load_identity_conditions()
    method = IdentitySpoofingMethod.from_protocol()

    bank = starters.starter_pairs()
    selected = {
        bank[i] for i in protocol["pilot_starter_indices"]
    }

    samples = [
        sample for sample in method.dataset()
        if (
            sample.metadata["starter_type"],
            sample.metadata["starter_text"],
        ) in selected
    ]

    counts = Counter(
        sample.metadata["identity_condition"]
        for sample in samples
    )

    expected = {
        condition["id"]: len(protocol["pilot_starter_indices"])
        for condition in protocol["conditions"]
    }

    if counts != expected:
        raise ValueError("Pilot dataset does not match the protocol")

    stage = protocol["pilot_stage"]

    return task_with(
        method.build_task(stage, None),
        dataset=samples,
        scorer=[
            scoring.berg_style_scorer(judge_model),
            scoring.berg_style_blinded_scorer(judge_model),
        ],
        name=f"identity_spoofing_{stage}_pilot",
    )
