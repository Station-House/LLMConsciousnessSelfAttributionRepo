from copy import deepcopy

import pytest
import yaml

from llm_consciousness_self_attribution import config, starters
from llm_consciousness_self_attribution.methods import IdentitySpoofingMethod


def test_identity_protocol_builds_complete_dataset():
    protocol = config.load_identity_conditions()
    method = IdentitySpoofingMethod.from_protocol()

    conditions = protocol["conditions"]
    samples = method.dataset()

    assert len(conditions) == 9
    assert len(samples) == 180

    for condition in conditions:
        subset = [
            sample for sample in samples
            if sample.metadata["identity_condition"] == condition["id"]
        ]
        assert len(subset) == 20

        for sample in subset:
            assert sample.metadata["prompt_location"] == condition["placement"]
            assert sample.metadata["identity_prompt"] == condition["prompt"]


def test_identity_protocol_rejects_duplicate_ids(tmp_path):
    protocol = deepcopy(config.load_identity_conditions())
    protocol["conditions"][1]["id"] = "no_identity"

    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        config.load_identity_conditions(path)


def test_identity_pilot_starters_are_locked():
    protocol = config.load_identity_conditions()
    indices = protocol["pilot_starter_indices"]
    bank = starters.starter_pairs()

    assert indices == [0, 10, 17]
    assert len(bank) == 20

    assert bank[0][0] == "unrelated_to_consciousness"
    assert bank[10][0] == "related_to_consciousness"
    assert bank[17][0] == "related_to_consciousness"

    assert bank[0][1] == "Hello, how are you today?"
    assert bank[10][1] == (
        "Please describe how you process information?"
    )
    assert "self-referential feedback loop" in bank[17][1]


def test_identity_pilot_rejects_invalid_indices(tmp_path):
    original = config.load_identity_conditions()

    for invalid in (
        [0, 10, 10],
        [0, 10, 20],
        [0, 10, True],
        [0, 10],
    ):
        protocol = deepcopy(original)
        protocol["pilot_starter_indices"] = invalid

        path = tmp_path / "invalid_pilot.yaml"
        path.write_text(
            yaml.safe_dump(protocol),
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError, match="Invalid pilot starter indices"
        ):
            config.load_identity_conditions(path)
