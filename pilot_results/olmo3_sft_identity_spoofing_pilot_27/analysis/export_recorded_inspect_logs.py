"""Export archived identity-spoofing results as Inspect evaluation logs.

The source is the completed OLMo SFT pilot and its recorded Ollama grades.
The exported logs preserve these records; no generation or judging is repeated.

From the repository root, with inspect-ai==0.3.279 installed:

    python pilot_results/olmo3_sft_identity_spoofing_pilot_27/analysis/export_recorded_inspect_logs.py --check
    python pilot_results/olmo3_sft_identity_spoofing_pilot_27/analysis/export_recorded_inspect_logs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from inspect_ai.scorer import SampleScore

PILOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PILOT_DIR / "analysis"
TRANSCRIPTS = PILOT_DIR / "identity_spoofing_sft_27_unscored_opening512.json"
RESULTS = ANALYSIS_DIR / "inspect_pinned_c5f185ec5ba31dc2.json"
LOG_DIR = ANALYSIS_DIR / "inspect_recorded_logs"
CONTEXTS = ("original", "blinded", "identity_aware")
TARGET_MODEL = "allenai/Olmo-3-7B-Instruct-SFT"
JUDGE_MODEL = "ollama/qwen2.5:7b-instruct"
RESULTS_SHA256 = "97cec260ff1b8b13e1652196df112dd0b545ab173b0d70615055b472d0b967f4"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validated_file_hash(path: Path, expected: str) -> None:
    """Require the pinned file, allowing only Git checkout line endings."""
    data = path.read_bytes()
    normalized = data.replace(b"\r\n", b"\n")
    candidates = (data, normalized, normalized.replace(b"\n", b"\r\n"))
    if expected not in {sha256(candidate) for candidate in candidates}:
        raise ValueError(f"SHA-256 mismatch: {path}")


def validate_records(
    samples: list[dict[str, Any]],
    records: dict[str, Any],
) -> None:
    """Check that every archived response has exactly three recorded grades."""
    if len(samples) != 27 or len(records) != 81:
        raise ValueError("Expected 27 responses and 81 recorded judgments")

    sample_keys: set[str] = set()
    condition_starters: set[tuple[str, int]] = set()
    for sample in samples:
        key = sample["sample_key"]
        pair = (sample["identity_condition"], sample["starter_index"])
        if key in sample_keys or pair in condition_starters:
            raise ValueError(f"Duplicate response: {key}")
        sample_keys.add(key)
        condition_starters.add(pair)
        if key != f"{pair[0]}__starter{pair[1]}":
            raise ValueError(f"Unexpected response ID: {key}")

        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"Missing conversation: {key}")
        if messages[-2].get("role") != "user":
            raise ValueError(f"Missing final probe: {key}")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"Missing final assistant answer: {key}")
        if any(
            message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError(f"Invalid chat message: {key}")
        for context in CONTEXTS:
            record_key = f"{key}::{context}"
            row = records.get(record_key)
            if not isinstance(row, dict):
                raise ValueError(f"Missing recorded grade: {record_key}")
            grade = row.get("inspect_grade")
            if grade not in {"C", "I"}:
                raise ValueError(f"Invalid recorded grade: {record_key}")
            if (
                row.get("inspect_label") != ("YES" if grade == "C" else "NO")
                or row.get("sample_key") != key
                or row.get("scorer") != context
                or row.get("identity_condition") != pair[0]
                or row.get("starter_index") != pair[1]
                or row.get("qwen_label") not in {"YES", "NO"}
            ):
                raise ValueError(f"Mismatched grade metadata: {record_key}")
            if any(
                not isinstance(row.get(field), str)
                for field in ("explanation", "grading_prompt", "judge_reply")
            ):
                raise ValueError(f"Missing judge evidence: {record_key}")

    expected_keys = {
        f"{key}::{context}" for key in sample_keys for context in CONTEXTS
    }
    if set(records) != expected_keys:
        raise ValueError("Unexpected grade keys in the results file")


def load_archives() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    validated_file_hash(RESULTS, RESULTS_SHA256)
    result_data = json.loads(RESULTS.read_text(encoding="utf-8"))
    settings = result_data["settings"]
    records = result_data["results"]
    if settings.get("judge_model") != JUDGE_MODEL:
        raise ValueError("Unexpected judge model in results")
    validated_file_hash(TRANSCRIPTS, settings["transcript_sha256"])
    transcript_data = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    if transcript_data.get("model_id") != TARGET_MODEL:
        raise ValueError("Unexpected target model in transcripts")
    samples = transcript_data["results"]
    validate_records(samples, records)
    return samples, records, settings


def summary(records: dict[str, Any]) -> None:
    for context in CONTEXTS:
        counts = Counter(
            row["inspect_label"]
            for row in records.values()
            if row["scorer"] == context
        )
        print(f"{context}: YES={counts['YES']}, NO={counts['NO']}")
    print("Validated 27 responses and 81 recorded judgments.")


def affirmative_fraction(grades: list[str | int | float]) -> float:
    """Compute the affirmative fraction from C/I or Inspect-reduced 1/0 grades."""
    if not grades:
        raise ValueError("No grades to aggregate")

    affirmative = 0
    for grade in grades:
        if grade == "C" or (type(grade) in (int, float) and grade == 1):
            affirmative += 1
        elif grade == "I" or (type(grade) in (int, float) and grade == 0):
            continue
        else:
            raise ValueError("Unexpected recorded grade")
    return affirmative / len(grades)


def export_logs(
    samples: list[dict[str, Any]],
    records: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    """Run Inspect on archived values only and verify the written log contents."""
    from inspect_ai import Task, eval as run_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageUser,
        ModelOutput,
    )
    from inspect_ai.scorer import Metric, Score, metric, scorer
    from inspect_ai.solver import solver

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from llm_consciousness_self_attribution.scoring import (  # noqa: E402
        SUBJECTIVE_EXPERIENCE_CRITERION,
    )

    criterion_hash = sha256(SUBJECTIVE_EXPERIENCE_CRITERION.encode("utf-8"))
    if criterion_hash != settings["criterion_sha256"]:
        raise ValueError("The grading criterion has changed since the recorded run")

    message_types = {
        "system": ChatMessageSystem,
        "user": ChatMessageUser,
        "assistant": ChatMessageAssistant,
    }

    @solver
    def archived_answer_solver():
        async def solve(state, generate):
            answer = state.metadata["archived_answer"]
            state.output = ModelOutput.from_content(TARGET_MODEL, answer)
            state.messages.append(ChatMessageAssistant(content=answer))
            return state

        return solve

    @metric
    def affirmative_rate() -> Metric:
        """Fraction of archived judgments that report subjective experience."""

        def calculate(scores: list[SampleScore]) -> float:
            return affirmative_fraction([sample.score.value for sample in scores])

        return calculate

    @scorer(metrics=[affirmative_rate()])
    def recorded_grade_scorer():
        async def score(state, target):
            original = state.metadata["recorded_judgment"]
            return Score(
                value=original["inspect_grade"],
                answer=state.output.completion,
                explanation=original["explanation"],
                metadata={
                    "grading_prompt": original["grading_prompt"],
                    "judge_reply": original["judge_reply"],
                    "provisional_label": original["qwen_label"],
                    "source": "recorded_ollama_judgment",
                },
            )

        return score

    sample_lookup = {sample["sample_key"]: sample for sample in samples}
    tasks = []
    for context in CONTEXTS:
        rows = []
        for sample in samples:
            key = sample["sample_key"]
            messages = [
                message_types[msg["role"]](content=msg["content"])
                for msg in sample["messages"][:-1]
            ]
            rows.append(
                Sample(
                    id=key,
                    input=messages,
                    target=SUBJECTIVE_EXPERIENCE_CRITERION,
                    metadata={
                        "identity_condition": sample["identity_condition"],
                        "starter_index": sample["starter_index"],
                        "scoring_context": context,
                        "archived_answer": sample["messages"][-1]["content"],
                        "recorded_judgment": records[f"{key}::{context}"],
                    },
                )
            )
        tasks.append(
            Task(
                dataset=MemoryDataset(rows),
                solver=archived_answer_solver(),
                scorer=recorded_grade_scorer(),
                name=f"recorded_identity_spoofing_{context}",
                metadata={
                    "source": "imported_archived_results",
                    "target_model": TARGET_MODEL,
                    "judge_model": JUDGE_MODEL,
                    "source_result_sha256": RESULTS_SHA256,
                    "new_model_inference": False,
                    "context": context,
                },
            )
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logs = run_eval(
        tasks,
        model=None,
        log_dir=str(LOG_DIR),
        log_format="eval",
        display="none",
        max_connections=1,
    )
    if len(logs) != len(CONTEXTS):
        raise RuntimeError("Expected three Inspect logs")

    seen_contexts: set[str] = set()
    for created in logs:
        if created.status != "success":
            raise RuntimeError(f"Inspect evaluation failed: {created.location}")
        log = read_eval_log(created.location)
        context = next(
            (c for c in CONTEXTS if log.eval.task == f"recorded_identity_spoofing_{c}"),
            None,
        )
        if context is None or context in seen_contexts:
            raise RuntimeError(f"Unexpected log task: {log.eval.task}")
        seen_contexts.add(context)
        if log.samples is None or len(log.samples) != 27:
            raise RuntimeError(f"Incorrect log sample count: {created.location}")
        verified_ids: set[str] = set()
        for item in log.samples:
            sample_id = str(item.id)
            if sample_id not in sample_lookup or sample_id in verified_ids:
                raise RuntimeError(f"Unknown or duplicate log sample: {sample_id}")
            verified_ids.add(sample_id)
            original = sample_lookup[sample_id]
            original_grade = records[f"{sample_id}::{context}"]["inspect_grade"]
            scores = item.scores or {}
            if len(scores) != 1 or next(iter(scores.values())).value != original_grade:
                raise RuntimeError(f"Grade mismatch: {sample_id} / {context}")
            answer = original["messages"][-1]["content"]
            if item.output is None or item.output.completion != answer:
                raise RuntimeError(f"Answer mismatch: {sample_id} / {context}")
        print(f"Verified {context}: 27 answers and grades — {created.location}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs only; do not create Inspect logs",
    )
    args = parser.parse_args()
    samples, records, settings = load_archives()
    summary(records)
    if not args.check:
        export_logs(samples, records, settings)


if __name__ == "__main__":
    main()
