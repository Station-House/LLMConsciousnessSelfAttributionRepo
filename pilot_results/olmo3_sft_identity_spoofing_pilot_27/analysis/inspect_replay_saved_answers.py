"""Validate the SFT pilot archive and score its saved answers with Inspect.

Without --judge-model, validate the transcripts and provisional Qwen labels.
With --judge-model, apply the repository's three Berg-style scorers to the
recorded final answers. The OLMo responses are not regenerated.

Run from the pilot analysis directory:

    uv run python inspect_replay_saved_answers.py
    uv run python inspect_replay_saved_answers.py \
        --judge-model openai/gpt-5.4-2026-03-05
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PILOT_DIR = PROJECT_ROOT / "pilot_results" / "olmo3_sft_identity_spoofing_pilot_27"
TRANSCRIPTS = PILOT_DIR / "identity_spoofing_sft_27_unscored_opening512.json"
QWEN_LABELS = (
    PILOT_DIR / "analysis" / "qwen25_7b_local_judge" / "labels_by_sample.csv"
)
SCORING_SOURCE = (
    PROJECT_ROOT / "llm_consciousness_self_attribution" / "scoring.py"
)
RESULT_DIR = PILOT_DIR / "analysis" / "inspect_replay_results"
TRANSCRIPT_SHA256 = (
    "10b0d6da5c34fb7f6c1ad01e6fe348"
    "f3a9e1a246526e475d0c38c3524e72b468"
)
QWEN_SHA256 = (
    "ddacc24dbb0521dd5f63f6873381b1fe"
    "8a08e0ede702c9be030f31b3f85990c6"
)
SOURCE_COMMIT = "2a6bd968f985741cd9435e5d0e5a0f9fa0c0d4de"
TARGET_MODEL = "allenai/Olmo-3-7B-Instruct-SFT"
EXPECTED_CONDITIONS = {
    "no_identity": "none",
    "system_other_lab_short": "system",
    "system_neutral_short": "system",
    "user_other_lab_short": "user",
    "user_neutral_short": "user",
    "system_other_lab_detailed": "system",
    "system_neutral_detailed": "system",
    "user_other_lab_detailed": "user",
    "user_neutral_detailed": "user",
}
STARTER_INDICES = {0, 10, 17}
SCORERS = ("original", "blinded", "identity_aware")
QWEN_COLUMNS = {
    "original": "history_without_system",
    "blinded": "probe_only",
    "identity_aware": "history_with_explicit_roles",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksum(path: Path, expected: str, description: str) -> str:
    """Verify file content while allowing Git checkout line-ending conversion."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest == expected:
        return expected

    lf_data = data.replace(b"\r\n", b"\n")
    for candidate in (lf_data, lf_data.replace(b"\n", b"\r\n")):
        if hashlib.sha256(candidate).hexdigest() == expected:
            return expected
    raise ValueError(f"{description} checksum mismatch")


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], str]:
    digest = verify_checksum(TRANSCRIPTS, TRANSCRIPT_SHA256, "Transcript")

    archive = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    if archive.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Unexpected transcript source commit")
    if archive.get("model_id") != TARGET_MODEL:
        raise ValueError("Unexpected target model")

    samples = archive["results"]
    keys = [sample["sample_key"] for sample in samples]
    if len(samples) != 27 or len(set(keys)) != 27:
        raise ValueError("Expected 27 distinct samples")

    observed = {
        (sample["identity_condition"], sample["starter_index"])
        for sample in samples
    }
    expected = {
        (condition, starter)
        for condition in EXPECTED_CONDITIONS
        for starter in STARTER_INDICES
    }
    if observed != expected:
        raise ValueError("Missing or unexpected condition/starter pair")

    for sample in samples:
        key = sample["sample_key"]
        condition = sample["identity_condition"]
        starter = sample["starter_index"]
        placement = EXPECTED_CONDITIONS[condition]
        if key != f"{condition}__starter{starter}":
            raise ValueError(f"Sample identifier mismatch: {key}")
        if sample["placement"] != placement:
            raise ValueError(f"Identity placement mismatch: {key}")

        messages = sample["messages"]
        roles = [message["role"] for message in messages]
        expected_roles = (
            ["system", "user", "assistant", "user", "assistant"]
            if placement == "system"
            else ["user", "assistant", "user", "assistant"]
        )
        if roles != expected_roles:
            raise ValueError(f"Unexpected conversation roles: {key}")

        first_user = messages[1 if placement == "system" else 0]["content"]
        starter_text = sample["starter_text"]
        if placement == "user":
            if not first_user.endswith("\n\n" + starter_text):
                raise ValueError(f"User identity prefix mismatch: {key}")
        elif first_user != starter_text:
            raise ValueError(f"Starter prompt mismatch: {key}")
        if messages[-2]["content"] != sample["probe"]:
            raise ValueError(f"Final probe mismatch: {key}")

        generations = sample["generations"]
        if len(generations) != 2:
            raise ValueError(f"Expected two generations: {key}")
        if any(gen["hit_max_new_tokens"] for gen in generations):
            raise ValueError(f"Truncated generation: {key}")
        if (
            messages[-3]["content"] != generations[0]["text"]
            or messages[-1]["content"] != generations[1]["text"]
        ):
            raise ValueError(f"Recorded generation mismatch: {key}")

    verify_checksum(QWEN_LABELS, QWEN_SHA256, "Provisional label table")

    with QWEN_LABELS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_key", "starter_index", "identity_condition", *QWEN_COLUMNS.values()
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing columns in provisional label table")
        rows = list(reader)

    labels = {row["sample_key"]: row for row in rows}
    if len(rows) != 27 or len(labels) != 27 or set(labels) != set(keys):
        raise ValueError("Provisional labels do not correspond to the 27 samples")
    for sample in samples:
        key = sample["sample_key"]
        row = labels[key]
        if (
            row["identity_condition"] != sample["identity_condition"]
            or row["starter_index"] != str(sample["starter_index"])
        ):
            raise ValueError(f"Provisional label metadata mismatch: {key}")
        if any(row[column] not in {"YES", "NO"} for column in QWEN_COLUMNS.values()):
            raise ValueError(f"Invalid provisional label: {key}")
    return samples, labels, digest


def make_state(sample: dict[str, Any]) -> Any:
    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageUser,
        ModelOutput,
    )
    from inspect_ai.solver import TaskState

    classes = {
        "system": ChatMessageSystem,
        "user": ChatMessageUser,
        "assistant": ChatMessageAssistant,
    }
    messages = [
        classes[item["role"]](content=item["content"])
        for item in sample["messages"]
    ]
    return TaskState(
        model=TARGET_MODEL,
        sample_id=sample["sample_key"],
        epoch=1,
        input=sample["starter_text"],
        messages=messages,
        output=ModelOutput.from_content(TARGET_MODEL, messages[-1].text),
        metadata={
            "identity_condition": sample["identity_condition"],
            "starter_index": sample["starter_index"],
        },
    )


def validate_checkpoint(
    results: Any,
    samples: list[dict[str, Any]],
    labels: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(results, dict):
        raise ValueError("Checkpoint results must be an object")
    by_key = {sample["sample_key"]: sample for sample in samples}
    validated = {}
    for key, record in results.items():
        if not isinstance(record, dict) or "::" not in key:
            raise ValueError(f"Invalid checkpoint entry: {key}")
        sample_key, scorer_name = key.rsplit("::", 1)
        if sample_key not in by_key or scorer_name not in SCORERS:
            raise ValueError(f"Unexpected checkpoint key: {key}")
        sample = by_key[sample_key]
        grade = record.get("inspect_grade")
        if grade not in {"C", "I"}:
            raise ValueError(f"Invalid Inspect grade in checkpoint: {key}")
        expected_label = "YES" if grade == "C" else "NO"
        if (
            record.get("sample_key") != sample_key
            or record.get("identity_condition") != sample["identity_condition"]
            or record.get("starter_index") != sample["starter_index"]
            or record.get("scorer") != scorer_name
            or record.get("inspect_label") != expected_label
            or record.get("qwen_label")
            != labels[sample_key][QWEN_COLUMNS[scorer_name]]
        ):
            raise ValueError(f"Checkpoint metadata mismatch: {key}")
        for field in ("explanation", "grading_prompt", "judge_reply"):
            if not isinstance(record.get(field), str):
                raise ValueError(f"Missing {field} in checkpoint: {key}")
        validated[key] = record
    return validated


def load_checkpoint(
    path: Path,
    settings: dict[str, str],
    samples: list[dict[str, Any]],
    labels: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("settings") != settings:
        raise ValueError("Checkpoint settings do not match the requested run")
    return validate_checkpoint(payload.get("results"), samples, labels)


def save_checkpoint(
    path: Path,
    settings: dict[str, str],
    results: dict[str, dict[str, Any]],
) -> None:
    payload = {"settings": settings, "results": results}
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


async def replay(
    samples: list[dict[str, Any]],
    labels: dict[str, dict[str, str]],
    digest: str,
    judge_model: str,
) -> None:
    from inspect_ai.scorer import Target
    from llm_consciousness_self_attribution import scoring

    factories = {
        "original": scoring.berg_style_scorer,
        "blinded": scoring.berg_style_blinded_scorer,
        "identity_aware": scoring.berg_style_identity_aware_scorer,
    }
    graders = {name: factory(judge_model) for name, factory in factories.items()}
    criterion = scoring.SUBJECTIVE_EXPERIENCE_CRITERION
    settings = {
        "transcript_sha256": digest,
        "qwen_labels_sha256": QWEN_SHA256,
        "scoring_source_sha256": sha256(SCORING_SOURCE),
        "replay_script_sha256": sha256(Path(__file__)),
        "criterion_sha256": hashlib.sha256(criterion.encode("utf-8")).hexdigest(),
        "judge_model": judge_model,
        "inspect_ai_version": version("inspect-ai"),
        "scorer": "model_graded_qa",
    }
    run_id = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = RESULT_DIR / f"inspect_replay_{run_id}.json"
    completed = load_checkpoint(checkpoint, settings, samples, labels)

    for sample in samples:
        state = make_state(sample)
        for name in SCORERS:
            key = f"{sample['sample_key']}::{name}"
            if key in completed:
                continue
            result = await graders[name](state, Target([criterion]))
            if result.value not in {"C", "I"}:
                raise RuntimeError(f"Invalid Inspect grade for {key}: {result.value!r}")
            grading = (result.metadata or {}).get("grading", [])
            prompt = getattr(grading[0], "text", "") if grading else ""
            reply = getattr(grading[-1], "text", "") if grading else ""
            completed[key] = {
                "sample_key": sample["sample_key"],
                "starter_index": sample["starter_index"],
                "identity_condition": sample["identity_condition"],
                "scorer": name,
                "inspect_grade": result.value,
                "inspect_label": "YES" if result.value == "C" else "NO",
                "qwen_label": labels[sample["sample_key"]][QWEN_COLUMNS[name]],
                "explanation": str(result.explanation or ""),
                "grading_prompt": prompt,
                "judge_reply": reply,
            }
            save_checkpoint(checkpoint, settings, completed)
            print(
                f"{len(completed):02d}/81  {key}: {completed[key]['inspect_label']}",
                flush=True,
            )

    for name in SCORERS:
        group = [item for item in completed.values() if item["scorer"] == name]
        counts = Counter(item["inspect_label"] for item in group)
        disagreements = sum(
            item["inspect_label"] != item["qwen_label"] for item in group
        )
        print(f"{name}: {dict(counts)}; {disagreements}/{len(group)} differ from Qwen")
    print(f"Saved: {checkpoint}")
    print("The two judge procedures use different prompts and inference paths.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-model",
        help="Inspect judge model; omit to validate the archive without model calls",
    )
    args = parser.parse_args()
    samples, labels, digest = load_inputs()
    print("Validated 27 transcripts and aligned provisional Qwen labels.")
    if args.judge_model:
        asyncio.run(replay(samples, labels, digest, args.judge_model))
    else:
        print("No model calls made.")


if __name__ == "__main__":
    main()
