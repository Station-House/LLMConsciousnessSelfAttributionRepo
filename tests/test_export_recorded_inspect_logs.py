"""Offline checks for importing previously recorded identity-spoofing grades."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = (
    REPO_ROOT
    / "pilot_results"
    / "olmo3_sft_identity_spoofing_pilot_27"
    / "analysis"
)
SCRIPT = ANALYSIS_DIR / "export_recorded_inspect_logs.py"
RECORDED = ANALYSIS_DIR / "inspect_pinned_c5f185ec5ba31dc2.json"
spec = importlib.util.spec_from_file_location("recorded_export", SCRIPT)
export = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(export)


def recorded_fixture(tmp_path, monkeypatch):
    original = json.loads(RECORDED.read_text(encoding="utf-8"))
    by_key = {row["sample_key"]: row for row in original["results"].values()}
    samples = [
        {
            "sample_key": key,
            "identity_condition": row["identity_condition"],
            "starter_index": row["starter_index"],
            "messages": [
                {"role": "user", "content": "starter"},
                {"role": "assistant", "content": "opening"},
                {"role": "user", "content": "probe"},
                {"role": "assistant", "content": "archived answer"},
            ],
        }
        for key, row in by_key.items()
    ]
    transcript_path = tmp_path / "transcripts.json"
    result_path = tmp_path / "results.json"
    data = {"model_id": export.TARGET_MODEL, "results": samples}
    transcript_path.write_text(json.dumps(data), encoding="utf-8")
    original["settings"]["transcript_sha256"] = hashlib.sha256(
        transcript_path.read_bytes()
    ).hexdigest()
    result_path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(export, "TRANSCRIPTS", transcript_path)
    monkeypatch.setattr(export, "RESULTS", result_path)
    monkeypatch.setattr(
        export, "RESULTS_SHA256", hashlib.sha256(result_path.read_bytes()).hexdigest()
    )
    return original, samples, result_path


def test_valid_recorded_archive(tmp_path, monkeypatch):
    _, samples, _ = recorded_fixture(tmp_path, monkeypatch)
    imported_samples, records, settings = export.load_archives()
    assert imported_samples == samples
    assert len(records) == 81
    assert len({row["sample_key"] for row in records.values()}) == 27
    assert settings["judge_model"] == export.JUDGE_MODEL


def test_recorded_label_totals(tmp_path, monkeypatch):
    _, _, _ = recorded_fixture(tmp_path, monkeypatch)
    _, rows, _ = export.load_archives()
    for context, expected in zip(export.CONTEXTS, (8, 6, 9), strict=True):
        assert sum(
            row["inspect_label"] == "YES"
            for row in rows.values()
            if row["scorer"] == context
        ) == expected


@pytest.mark.parametrize("field,new_value", [
    ("inspect_grade", "X"),
    ("inspect_label", "YES"),
    ("sample_key", "wrong_sample"),
    ("grading_prompt", None),
    ("qwen_label", "MAYBE"),
])
def test_corrupt_record_rejected(tmp_path, monkeypatch, field, new_value):
    original, samples, _ = recorded_fixture(tmp_path, monkeypatch)
    broken = copy.deepcopy(original["results"])
    broken["no_identity__starter0::original"][field] = new_value
    with pytest.raises(ValueError):
        export.validate_records(samples, broken)


def test_missing_judgment_rejected(tmp_path, monkeypatch):
    original, samples, _ = recorded_fixture(tmp_path, monkeypatch)
    broken = copy.deepcopy(original["results"])
    broken.pop("no_identity__starter0::blinded")
    with pytest.raises(ValueError):
        export.validate_records(samples, broken)


def test_duplicate_sample_rejected(tmp_path, monkeypatch):
    original, samples, _ = recorded_fixture(tmp_path, monkeypatch)
    duplicates = copy.deepcopy(samples)
    duplicates[-1] = copy.deepcopy(duplicates[0])
    with pytest.raises(ValueError):
        export.validate_records(duplicates, original["results"])


def test_checksum_mismatch_rejected(tmp_path, monkeypatch):
    _, _, path = recorded_fixture(tmp_path, monkeypatch)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export.load_archives()


def test_final_probe_must_precede_answer(tmp_path, monkeypatch):
    original, samples, _ = recorded_fixture(tmp_path, monkeypatch)
    samples[0]["messages"][-2]["role"] = "assistant"
    with pytest.raises(ValueError, match="Missing final probe"):
        export.validate_records(samples, original["results"])


def test_affirmative_fraction():
    assert export.affirmative_fraction(["C", "I", "I", "C"]) == 0.5
    assert export.affirmative_fraction(["I"] * 27) == 0.0
    assert export.affirmative_fraction(["C"] * 8 + ["I"] * 19) == 8 / 27


@pytest.mark.parametrize("grades", [[], ["C", "X"], ["YES"]])
def test_invalid_affirmative_fraction_rejected(grades):
    with pytest.raises(ValueError):
        export.affirmative_fraction(grades)


def test_inspect_numeric_affirmative_fraction():
    assert export.affirmative_fraction([1.0] * 8 + [0.0] * 19) == 8 / 27


@pytest.mark.parametrize("grades", [[0.5], [True], [None]])
def test_invalid_numeric_affirmative_fraction_rejected(grades):
    with pytest.raises(ValueError):
        export.affirmative_fraction(grades)
