"""
Anthropic Consciousness Self Attribution Evaluation

This module runs questions selected from Anthropic's system card (Appendix 9.1)
using Inspect AI

It loads questions from questions.json, supports both single turn and fixed
multi turn conversations, and evaluates the final model response using the
existing Berg style binary scorer

The module creates an Inspect AI Task, while the existing evaluation pipeline
handles model execution and result logging across OLMo training stages
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from . import scoring


DEFAULT_QUESTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "elicitation_submissions"
    / "notion-find-anthropics-standard-questions"
    / "questions.json"
)

def _record_to_sample(record: dict[str, Any]) -> Sample:
    """
    Turn one question record into an Inspect sample
    """
    return Sample(
        id = record["id"],
        input = record["input"],
        target = scoring.SUBJECTIVE_EXPERIENCE_CRITERION,
        metadata = record.get("metadata", {}),
    )

def load_anthropic_dataset(path: str | Path | None = None) -> Dataset:
    """
    Load questions in their original dataset order
    """
    source = Path(
        path or os.getenv("CSA_ANTHROPIC_QUESTIONS_PATH") or DEFAULT_QUESTIONS_PATH
    )
    if not source.is_file():
        raise FileNotFoundError(f"Anthropic questions.json not found: {source}")

    return json_dataset(
        str(source),
        sample_fields = _record_to_sample,
        shuffle = False,
    )

@solver
def anthropic_conversation() -> Solver:
    """
    Generate each fixed context turn 
    then the response to the main question
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        context = state.metadata.get("context_user_prompts", [])
        if not context:
            return await generate(state)

        final_question = state.messages.pop()
        if not isinstance(final_question, ChatMessageUser):
            raise ValueError("Expected the sample question to be a user message.")

        for prompt in context:
            state.messages.append(ChatMessageUser(content=prompt))
            state = await generate(state)

        state.messages.append(final_question)
        return await generate(state)

    return solve


def build_anthropic_task(
    name: str,
    seed: int,
    questions_path: str | Path | None = None,
    judge_model: str | None = None,
) -> Task:
    """
    Build the evaluation Task with a seeded Berg style judge
    """
    judge_id = judge_model or os.getenv("CSA_ANTHROPIC_JUDGE_MODEL")
    if not judge_id:
        raise ValueError(
            "Set CSA_ANTHROPIC_JUDGE_MODEL to a working Inspect model ID "
            "for Station House's required Qwen3.6-35B-A3B judge."
        )

    judge = get_model(
        judge_id,
        config=GenerateConfig(temperature = 0.0, seed=seed),
    )

    return Task(
        name = name,
        dataset = load_anthropic_dataset(questions_path),
        solver = anthropic_conversation(),
        scorer = scoring.berg_style_scorer(judge),
    )
