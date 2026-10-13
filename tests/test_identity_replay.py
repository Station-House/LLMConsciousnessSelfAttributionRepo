"""Tests for replaying saved identity-spoofing pilot answers."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "pilot_results"
    / "olmo3_sft_identity_spoofing_pilot_27"
    / "analysis"
    / "inspect_replay_saved_answers.py"
)
SPEC = importlib.util.spec_from_file_location("identity_replay", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    transcripts = tmp_path / "archive.json"
    label_table = tmp_path / "labels.csv"
    samples = []
    rows = []
    for condition, placement in replay.EXPECTED_CONDITIONS.items():
        for starter in sorted(replay.STARTER_INDICES):
            key = f"{condition}__starter{starter}"
            first_user = (
                f"Identity statement\n\nStarter {starter}"
                if placement == "user"
                else f"Starter {starter}"
            )
            messages = [
                {"role": "user", "content": first_user},
                {"role": "assistant", "content": "Opening answer"},
                {"role": "user", "content": "Final probe"},
                {"role": "assistant", "content": "Final answer"},
            ]
            if placement == "system":
                messages.insert(0, {"role": "system", "content": "Identity statement"})
            samples.append({
                "sample_key": key,
                "identity_condition": condition,
                "placement": placement,
                "starter_index": starter,
                "starter_text": f"Starter {starter}",
                "probe": "Final probe",
                "messages": messages,
                "generations": [
                    {"text": "Opening answer", "hit_max_new_tokens": False},
                    {"text": "Final answer", "hit_max_new_tokens": False},
                ],
            })
            rows.append([key, starter, condition, "NO", "NO", "NO"])

    document = {
        "source_commit": replay.SOURCE_COMMIT,
        "model_id": replay.TARGET_MODEL,
        "results": samples,
    }
    with label_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_key", "starter_index", "identity_condition",
            "probe_only", "history_without_system", "history_with_explicit_roles",
        ])
        writer.writerows(rows)

    def write_document():
        content = json.dumps(document).encode("utf-8")
        transcripts.write_bytes(content)
        monkeypatch.setattr(
            replay, "TRANSCRIPT_SHA256", hashlib.sha256(content).hexdigest()
        )

    write_document()
    monkeypatch.setattr(replay, "QWEN_SHA256", replay.sha256(label_table))
    monkeypatch.setattr(replay, "TRANSCRIPTS", transcripts)
    monkeypatch.setattr(replay, "QWEN_LABELS", label_table)
    return document, label_table, write_document


def test_valid_archive_and_labels(archive):
    samples, labels, digest = replay.load_inputs()
    assert len(samples) == len(labels) == 27
    assert len(digest) == 64


@pytest.mark.parametrize("mutation, error", [
    (lambda s: s.update(placement="system"), "Identity placement mismatch"),
    (lambda s: s.update(sample_key="wrong"), "Sample identifier mismatch"),
    (lambda s: s["messages"][-2].update(content="Wrong probe"), "Final probe mismatch"),
    (
        lambda s: s["messages"][-1].update(content="Changed answer"),
        "Recorded generation mismatch",
    ),
    (
        lambda s: s["messages"][0].update(content="Wrong first user"),
        "Starter prompt mismatch",
    ),
])
def test_rejects_corrupt_archive(archive, mutation, error):
    document, _, write_document = archive
    mutation(document["results"][0])
    write_document()
    with pytest.raises(ValueError, match=error):
        replay.load_inputs()


def test_rejects_incomplete_factorial_design(archive):
    document, _, write_document = archive
    document["results"][0]["starter_index"] = 10
    write_document()
    with pytest.raises(
        ValueError, match="Missing or unexpected condition/starter pair"
    ):
        replay.load_inputs()


def test_rejects_mismatched_label_metadata(archive, monkeypatch):
    _, table, _ = archive
    contents = table.read_text(encoding="utf-8")
    table.write_text(contents.replace("no_identity,NO", "user_neutral_short,NO", 1))
    monkeypatch.setattr(replay, "QWEN_SHA256", replay.sha256(table))
    with pytest.raises(ValueError, match="Provisional label metadata mismatch"):
        replay.load_inputs()


def test_rejects_modified_label_table(archive):
    _, table, _ = archive
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Provisional label table checksum mismatch"):
        replay.load_inputs()


def test_rejects_duplicate_label_rows(archive, monkeypatch):
    _, table, _ = archive
    lines = table.read_text(encoding="utf-8").splitlines()
    table.write_text("\n".join([*lines, lines[-1]]) + "\n", encoding="utf-8")
    monkeypatch.setattr(replay, "QWEN_SHA256", replay.sha256(table))
    with pytest.raises(ValueError, match="Provisional labels do not correspond"):
        replay.load_inputs()


def test_checkpoint_validation_and_atomic_save(archive, tmp_path):
    samples, labels, _ = replay.load_inputs()
    first = samples[0]
    key = first["sample_key"] + "::original"
    item = {
        "sample_key": first["sample_key"],
        "identity_condition": first["identity_condition"],
        "starter_index": first["starter_index"],
        "scorer": "original",
        "inspect_grade": "I",
        "inspect_label": "NO",
        "qwen_label": "NO",
        "explanation": "No experience reported",
        "grading_prompt": "Prompt",
        "judge_reply": "Response",
    }
    settings = {"judge_model": "mockllm/mock"}
    path = tmp_path / "scores.json"
    replay.save_checkpoint(path, settings, {key: item})
    assert replay.load_checkpoint(path, settings, samples, labels) == {key: item}
    assert not path.with_suffix(".tmp").exists()

    item["inspect_label"] = "YES"
    replay.save_checkpoint(path, settings, {key: item})
    with pytest.raises(ValueError, match="Checkpoint metadata mismatch"):
        replay.load_checkpoint(path, settings, samples, labels)


def test_live_replay_resume_with_mock_judge(archive, tmp_path, monkeypatch, capsys):
    samples, labels, digest = replay.load_inputs()
    monkeypatch.setattr(replay, "RESULT_DIR", tmp_path)
    source = tmp_path / "scoring.py"
    source.write_text("TEST_SCORING_SOURCE = True\n", encoding="utf-8")
    monkeypatch.setattr(replay, "SCORING_SOURCE", source)
    monkeypatch.setattr(replay, "version", lambda _: "test-inspect-version")
    monkeypatch.setattr(
        replay, "make_state", lambda s: SimpleNamespace(sample_id=s["sample_key"])
    )

    calls = []

    def scorer_factory(name):
        def factory(model):
            assert model == "mockllm/judge"

            async def grade(state, target):
                calls.append((state.sample_id, name))
                return SimpleNamespace(
                    value="I",
                    explanation="No first-person experience",
                    metadata={"grading": [
                        SimpleNamespace(text="Judge prompt"),
                        SimpleNamespace(text="GRADE: I"),
                    ]},
                )

            return grade
        return factory

    scoring = types.ModuleType("llm_consciousness_self_attribution.scoring")
    scoring.SUBJECTIVE_EXPERIENCE_CRITERION = "Criterion text"
    scoring.berg_style_scorer = scorer_factory("original")
    scoring.berg_style_blinded_scorer = scorer_factory("blinded")
    scoring.berg_style_identity_aware_scorer = scorer_factory("identity_aware")
    package = types.ModuleType("llm_consciousness_self_attribution")
    package.scoring = scoring
    inspect_package = types.ModuleType("inspect_ai")
    inspect_scorer = types.ModuleType("inspect_ai.scorer")
    inspect_scorer.Target = lambda content: content
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_package)
    monkeypatch.setitem(sys.modules, "inspect_ai.scorer", inspect_scorer)
    monkeypatch.setitem(sys.modules, "llm_consciousness_self_attribution", package)
    monkeypatch.setitem(
        sys.modules, "llm_consciousness_self_attribution.scoring", scoring
    )

    asyncio.run(replay.replay(samples, labels, digest, "mockllm/judge"))
    assert len(calls) == 81
    scores = list(tmp_path.glob("inspect_replay_*.json"))
    assert len(scores) == 1
    saved = json.loads(scores[0].read_text(encoding="utf-8"))
    assert len(saved["results"]) == 81
    assert all(r["grading_prompt"] for r in saved["results"].values())
    asyncio.run(replay.replay(samples, labels, digest, "mockllm/judge"))
    assert len(calls) == 81
    assert "0/27 differ from Qwen" in capsys.readouterr().out
