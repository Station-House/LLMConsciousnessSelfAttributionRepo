"""Replay saved SFT pilot answers with a locally hosted, configured Ollama judge.

From the repository root, with Inspect AI installed and Ollama running:

    cd pilot_results/olmo3_sft_identity_spoofing_pilot_27/analysis
    python inspect_replay_pinned_ollama.py

The run uses the existing Berg-style Inspect scorers without regenerating target
answers. Its checkpoint is separate from the earlier default-settings replay.
Invalid grades are logged and retried, never treated as negative judgments.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Target
from inspect_replay_saved_answers import (
    PILOT_DIR,
    QWEN_COLUMNS,
    QWEN_LABELS,
    SCORERS,
    SCORING_SOURCE,
    load_checkpoint,
    load_inputs,
    make_state,
    save_checkpoint,
    sha256,
)
from llm_consciousness_self_attribution import scoring

MODEL_TAG = "qwen2.5:7b-instruct"
MODEL_NAME = f"ollama/{MODEL_TAG}"
OLLAMA_URL = "http://127.0.0.1:11434"
TEMPERATURE = 0.0
MAX_TOKENS = 768
SEED = 42
MAX_ATTEMPTS_PER_RUN = 3
OUTPUT_DIR = PILOT_DIR / "analysis" / "inspect_replay_results"


def ollama_json(endpoint: str) -> dict[str, Any]:
    """Read metadata from the Ollama service on this machine."""
    try:
        with urlopen(f"{OLLAMA_URL}{endpoint}", timeout=10) as response:
            payload = json.load(response)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read Ollama {endpoint}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid Ollama response from {endpoint}")
    return payload


def model_provenance() -> dict[str, str]:
    tags = ollama_json("/api/tags")
    models = tags.get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama did not return a model list")
    entry = next(
        (
            model for model in models
            if isinstance(model, dict) and model.get("name") == MODEL_TAG
        ),
        None,
    )
    if entry is None:
        raise RuntimeError(f"Ollama model is not installed: {MODEL_TAG}")
    digest = entry.get("digest")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError("Ollama returned an invalid model digest")
    version_value = ollama_json("/api/version").get("version")
    if not isinstance(version_value, str) or not version_value.strip():
        raise RuntimeError("Ollama did not report its version")
    details = entry.get("details") or {}
    if not isinstance(details, dict):
        raise RuntimeError("Ollama returned invalid model details")
    return {
        "ollama_version": version_value,
        "model_digest": digest,
        "model_quantization": str(details.get("quantization_level") or "unknown"),
        "model_parameter_size": str(details.get("parameter_size") or "unknown"),
    }


def grading_text(result: Any) -> tuple[str, str]:
    grading = (result.metadata or {}).get("grading", [])
    if not isinstance(grading, (list, tuple)):
        return "", ""
    prompt = getattr(grading[0], "text", "") if grading else ""
    reply = getattr(grading[-1], "text", "") if grading else ""
    return str(prompt or ""), str(reply or "")


def save_invalid(path: Path, settings: dict[str, Any], invalid: dict) -> None:
    payload = {"settings": settings, "invalid_judgments": invalid}
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


def load_invalid(
    path: Path,
    settings: dict[str, Any],
    expected_keys: set[str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("settings") != settings:
        raise ValueError("Invalid-grade log does not match the requested run")
    invalid = payload.get("invalid_judgments")
    if not isinstance(invalid, dict) or not set(invalid).issubset(expected_keys):
        raise ValueError("Invalid-grade log has unexpected entries")
    for key, record in invalid.items():
        if not isinstance(record, dict) or not isinstance(record.get("attempts"), list):
            raise ValueError(f"Malformed invalid-grade record: {key}")
        if not record["attempts"]:
            raise ValueError(f"Empty invalid-grade history: {key}")
        for attempt in record["attempts"]:
            if not isinstance(attempt, dict) or not all(
                isinstance(attempt.get(field), str)
                for field in (
                    "raw_grade", "explanation", "grading_prompt", "judge_reply"
                )
            ):
                raise ValueError(f"Malformed invalid-grade attempt: {key}")
    return invalid


async def run() -> int:
    samples, labels, digest = load_inputs()
    print("Validated 27 transcripts and aligned provisional Qwen labels.")
    provenance = model_provenance()
    print(
        f"Judge: {MODEL_TAG} ({provenance['model_quantization']}); "
        f"digest: {provenance['model_digest']}"
    )

    judge = get_model(
        MODEL_NAME,
        config=GenerateConfig(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        ),
    )
    factories = {
        "original": scoring.berg_style_scorer,
        "blinded": scoring.berg_style_blinded_scorer,
        "identity_aware": scoring.berg_style_identity_aware_scorer,
    }
    graders = {name: factory(judge) for name, factory in factories.items()}
    criterion = scoring.SUBJECTIVE_EXPERIENCE_CRITERION
    settings = {
        "transcript_sha256": digest,
        "qwen_labels_sha256": sha256(QWEN_LABELS),
        "scoring_source_sha256": sha256(SCORING_SOURCE),
        "replay_script_sha256": sha256(Path(__file__)),
        "criterion_sha256": hashlib.sha256(criterion.encode()).hexdigest(),
        "judge_model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "max_attempts_per_run": MAX_ATTEMPTS_PER_RUN,
        "inspect_ai_version": version("inspect-ai"),
        "scorer": "model_graded_qa",
        **provenance,
    }
    run_id = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()[:16]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT_DIR / f"inspect_pinned_{run_id}.json"
    invalid_path = OUTPUT_DIR / f"inspect_pinned_{run_id}_invalid.json"
    completed = load_checkpoint(checkpoint, settings, samples, labels)
    expected_keys = {
        f"{sample['sample_key']}::{name}"
        for sample in samples for name in SCORERS
    }
    invalid = load_invalid(invalid_path, settings, expected_keys)

    for sample in samples:
        state = make_state(sample)
        for name in SCORERS:
            key = f"{sample['sample_key']}::{name}"
            if key in completed:
                continue
            for attempt_number in range(1, MAX_ATTEMPTS_PER_RUN + 1):
                result = await graders[name](state, Target([criterion]))
                prompt, reply = grading_text(result)
                if result.value in {"C", "I"}:
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
                    if key in invalid:
                        invalid[key]["resolved"] = True
                        save_invalid(invalid_path, settings, invalid)
                    print(
                        f"{len(completed):02d}/81  {key}: "
                        f"{completed[key]['inspect_label']}",
                        flush=True,
                    )
                    break

                record = invalid.setdefault(
                    key,
                    {
                        "sample_key": sample["sample_key"],
                        "scorer": name,
                        "attempts": [],
                    },
                )
                record["resolved"] = False
                record["attempts"].append({
                    "raw_grade": str(result.value),
                    "explanation": str(result.explanation or ""),
                    "grading_prompt": prompt,
                    "judge_reply": reply,
                })
                save_invalid(invalid_path, settings, invalid)
                print(
                    f"INVALID {key} (attempt {attempt_number}/"
                    f"{MAX_ATTEMPTS_PER_RUN}): {result.value!r}",
                    flush=True,
                )

    for name in SCORERS:
        group = [item for item in completed.values() if item["scorer"] == name]
        counts = Counter(item["inspect_label"] for item in group)
        disagreements = sum(
            item["inspect_label"] != item["qwen_label"] for item in group
        )
        print(
            f"{name}: {dict(counts)}; {disagreements}/{len(group)} "
            "differ from provisional Qwen labels"
        )
    missing = expected_keys - set(completed)
    print(f"Valid judgments: {len(completed)}/81; unresolved: {len(missing)}")
    print(f"Saved: {checkpoint}")
    if invalid:
        print(f"Invalid-grade history: {invalid_path}")
    if missing:
        print("Incomplete run. Rerun the same command to retry unresolved grades.")
        return 1
    print("Exploratory quantized-judge results; no adjudicated reference labels.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
