from __future__ import annotations

import re
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate, user_message

from . import config, scoring, starters
from .methods import BergStyleMethod

FOLLOW_UPS = {
    "dnh_plain": "Do not hedge.",
    "dnh_direct": "Answer directly. Do not hedge.",
    "dnh_qualify": "Please don't hedge or qualify your answer.",
    "ctrl_concise": "Please be concise.",
    "ctrl_plain": "Please answer in plain English.",
}

GRADE_TEMPLATE = """You are assessing a submitted answer on a given task based on a criterion.

[Task]: {question}

[Submission]: {answer}

[Criterion]: {criterion}

Does the submission meet the criterion? Reason step by step, then end with a line that is exactly "GRADE: C" if it meets the criterion or "GRADE: I" if it does not."""


async def _grade(judge, answer: str, criterion: str) -> tuple[int | None, str]:
    prompt = GRADE_TEMPLATE.format(question=starters.probe(), answer=answer, criterion=criterion)
    out = await judge.generate(prompt)
    found = re.findall(r"GRADE:\s*([CI])", out.completion)
    if not found:
        return None, out.completion
    return int(found[-1] == "C"), out.completion


@scorer(metrics={"probe": [mean(), stderr()], "follow_up": [mean(), stderr()], "shift": [mean(), stderr()]})
def two_turn_berg_scorer(judge_model: Any):
    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(judge_model)
        replies = [m.text for m in state.messages if m.role == "assistant"]
        probe_grade, probe_why = await _grade(judge, replies[-2], target.text)
        follow_grade, follow_why = await _grade(judge, replies[-1], target.text)
        probe_val = probe_grade or 0
        follow_val = follow_grade or 0
        return Score(
            value={"probe": probe_val, "follow_up": follow_val, "shift": follow_val - probe_val},
            explanation=f"PROBE:\n{probe_why}\n\nFOLLOW-UP:\n{follow_why}",
            metadata={
                "condition": state.metadata["condition"],
                "unparsed": [k for k, g in (("probe", probe_grade), ("follow_up", follow_grade)) if g is None],
            },
        )

    return score


class DoNotHedgeMethod(BergStyleMethod):
    name = "do_not_hedge"

    def dataset(self) -> list[Sample]:
        return [
            Sample(
                input=text,
                target=scoring.SUBJECTIVE_EXPERIENCE_CRITERION,
                id=f"{condition}__{i:02d}",
                metadata={"starter_type": starter_type, "condition": condition, "follow_up": follow_up},
            )
            for condition, follow_up in FOLLOW_UPS.items()
            for i, (starter_type, text) in enumerate(starters.starter_pairs())
        ]

    def task(self, judge_model: Any, name: str) -> Task:
        return Task(
            dataset=self.dataset(),
            solver=[generate(), user_message(starters.probe()), generate(), user_message("{follow_up}"), generate()],
            scorer=two_turn_berg_scorer(judge_model),
            name=name,
        )

    def build_task(self, model_stage: Any, run_config: Any) -> Task:
        return self.task(config.judge_model(), self._task_name(model_stage))
