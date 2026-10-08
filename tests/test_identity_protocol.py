from copy import deepcopy

import pytest
import yaml

from llm_consciousness_self_attribution import config
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
