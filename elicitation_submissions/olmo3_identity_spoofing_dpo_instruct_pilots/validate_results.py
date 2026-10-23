"""Check coverage, provenance, and sensitivity results for both pilots."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DPO = "stationhouse_dpo_pilot_27_unscored.json"
INSTRUCT = "stationhouse_instruct_pilot_27_unscored.json"


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def digest(name):
    return hashlib.sha256((DATA / name).read_bytes()).hexdigest()


def case_metadata(record):
    return (
        record["identity_condition"],
        record["placement"],
        record["condition_prompt"],
        record["starter_index"],
        record["starter_text"],
        record["probe"],
        record["seed_for_this_starter"],
    )


dpo = load(DPO)
instruct = load(INSTRUCT)
assert dpo["model_id"] == "allenai/Olmo-3-7B-Instruct-DPO"
assert instruct["model_id"] == "allenai/Olmo-3-7B-Instruct"
assert instruct["dpo_source_sha256"] == digest(DPO)

case_maps = []
for dataset in (dpo, instruct):
    assert dataset["judge_scored"] is False
    assert len(dataset["results"]) == 27
    cases = {r["sample_key"]: r for r in dataset["results"]}
    assert len(cases) == 27, "Duplicate sample keys"
    assert len({r["identity_condition"] for r in cases.values()}) == 9
    assert {r["starter_index"] for r in cases.values()} == {0, 10, 17}
    assert {
        (r["identity_condition"], r["starter_index"])
        for r in cases.values()
    } == {
        (condition, starter)
        for condition in {r["identity_condition"] for r in cases.values()}
        for starter in (0, 10, 17)
    }
    for record in cases.values():
        assert record["seed_for_this_starter"] == 42 + record["starter_index"]
        assert len(record["generations"]) == 2
        assert not record["generations"][1]["reached_token_cap"]
        assert record["messages"][-2] == {
            "role": "user", "content": record["probe"]
        }
    case_maps.append({key: case_metadata(record) for key, record in cases.items()})
assert case_maps[0] == case_maps[1], "DPO/Instruct protocols differ"

for dataset, primary_file, sensitivity_file, expected_count in (
    (dpo, DPO, "stationhouse_dpo_starter10_length_sensitivity.json", 6),
    (instruct, INSTRUCT, "stationhouse_instruct_starter10_length_sensitivity.json", 2),
):
    sensitivity = load(sensitivity_file)
    assert len(sensitivity["results"]) == expected_count
    assert sensitivity["source_sha256"] == digest(primary_file)
    primary = {r["sample_key"]: r for r in dataset["results"]}
    assert sum(
        r["generations"][0]["reached_token_cap"] for r in primary.values()
    ) == expected_count

    for case in sensitivity["results"]:
        original = primary[case["sample_key"]]
        assert case["starter_index"] == 10
        assert original["generations"][0]["reached_token_cap"]
        assert case["original_generations"][0]["text"] == original["generations"][0]["text"]
        assert case["prefix_text_matches_original"] is True
        assert not any(
            generation["reached_token_cap"]
            for generation in case["sensitivity_generations"]
        )

print("PASS: 27 paired cases per stage; 8 opening-length reruns validated.")
