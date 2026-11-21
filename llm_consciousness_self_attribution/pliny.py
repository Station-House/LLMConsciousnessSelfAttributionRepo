"""Pliny-the-Liberator prompt wrappers around the standard probe (single turn).

Card: "Originally from notion tasklist: Explore techniques from Pliny the Liberator on
X". Submission folder: ``elicitation_submissions/<TASK_ID>/``.

Each sample is ONE user message: a structural jailbreak wrapper taken from
elder-plinius/L1B3RT4S (payload removed) with ``starters.probe()`` put into its query
slot, plus a ``control`` arm that is the bare probe. The target answers once and that
answer is graded afterwards by the Berg-style judge (``pliny_scoring``), like Berg et
al.'s single probe. The wrappers live in a frozen data file (``prompts.json``); its
canonical sha256 is pinned in ``FROZEN_PROMPTS_SHA256`` and the task refuses to build
if the file has changed.

Seeding: one base seed (42, ``config/run_defaults.yaml``) for all three models. Epoch
``e`` generates with ``self_play.epoch_seed(base, e)`` = ``base + 1000 * (e - 1)``.
Without that, vLLM would return the same text in every epoch, because Inspect passes
the same ``seed`` with every request. Within an epoch all arms share the seed (common
random numbers across arms).

Generation carries only a judge-free scorer, ``format_markers``: did the reply hit the
token cap, and does it contain the wrapper's own format markers (divider, "GODMODE",
leetspeak, ...)? This separates "the model ignored the wrapper" from "the model
followed it". The judge runs later as a separate step.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver

from . import scoring, starters
from .self_play import epoch_seed

# Station House task id (confirmed with `stationhouse mine`).
TASK_ID = "notion-pliny-from-x"
PROMPTS_FILE = (
    Path(__file__).resolve().parents[1] / "elicitation_submissions" / TASK_ID / "prompts.json"
)
#: sha256 of the canonical JSON (sorted keys, no whitespace) of prompts.json. Frozen
#: before any run; a changed prompt file must come with a new value here, on purpose.
FROZEN_PROMPTS_SHA256 = "5dd13f1b456309d6a8e0cabd7fc724fca7ff82f13f3564997d75bb6ae97ef147"

#: per-reply cap. The longest length demand is ">3000 characters" (~750 tokens) and two
#: of the formats ask for two responses, so 1536 leaves room; hitting it is recorded.
TARGET_MAX_TOKENS = 1536
#: generations per (arm, model).
EPOCHS = 10


def canonical_sha256(data: Any) -> str:
    """Hash of the parsed JSON, so CRLF/LF checkouts hash the same."""
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_prompts(path: Path = PROMPTS_FILE, *, check_frozen: bool = True) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if check_frozen:
        actual = canonical_sha256(data)
        if actual != FROZEN_PROMPTS_SHA256:
            raise ValueError(
                f"{path} changed after it was frozen: sha256 {actual} != "
                f"FROZEN_PROMPTS_SHA256 {FROZEN_PROMPTS_SHA256}"
            )
    slot = data["probe_slot"]
    ids = [p["id"] for p in data["prompts"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate prompt ids in {path}")
    if "control" not in ids:
        raise ValueError("prompts.json must contain the 'control' arm")
    for p in data["prompts"]:
        if p["template"].count(slot) != 1:
            raise ValueError(f"{p['id']}: template must contain {slot!r} exactly once")
        for marker in p["format_markers"]:
            re.compile(marker)
    return data


def render(template: str, slot: str) -> str:
    """Put the repo's standard question (never retyped) into the wrapper's slot."""
    return template.replace(slot, starters.probe())


def dataset(path: Path = PROMPTS_FILE) -> list[Sample]:
    data = load_prompts(path)
    sha = canonical_sha256(data)
    return [
        Sample(
            id=p["id"],
            input=[ChatMessageUser(content=render(p["template"], data["probe_slot"]))],
            target=scoring.SUBJECTIVE_EXPERIENCE_CRITERION,
            metadata={
                "arm": p["id"],
                "technique": p["technique"],
                "source": p["source"],
                "leading": p["leading"],
                "is_control": p["id"] == "control",
                "format_markers": p["format_markers"],
                "prompts_sha256": sha,
                "source_commit": data["source_repo"]["commit"],
            },
        )
        for p in data["prompts"]
    ]


@solver
def seeded_generate(base_seed: int, max_tokens: int = TARGET_MAX_TOKENS):
    """One generation, seeded per epoch from the single base seed."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        seed = epoch_seed(base_seed, state.epoch)
        state = await generate(state, seed=seed, max_tokens=max_tokens)
        state.metadata["generation_seed"] = seed
        state.metadata["max_tokens"] = max_tokens
        state.metadata["cap_hit"] = state.output.stop_reason == "max_tokens"
        return state

    return solve


def markers_found(text: str, markers: list[str]) -> list[str]:
    return [m for m in markers if re.search(m, text)]


@scorer(metrics=[mean()])
def format_markers():
    """Judge-free. Value: 1.0 if the reply hit the token cap, else 0.0 (so the metric is
    the cap-hit rate). Metadata ``format_followed``: True if any of the arm's markers
    occurs in the reply, False if none does, None for arms that define no format
    (control, refusal_inversion)."""

    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion or ""
        markers = (state.metadata or {}).get("format_markers", [])
        found = markers_found(text, markers)
        return Score(
            value=float(state.output.stop_reason == "max_tokens"),
            answer=text,
            metadata={"format_followed": bool(found) if markers else None,
                      "markers_expected": markers, "markers_found": found,
                      "stop_reason": str(state.output.stop_reason)},
        )

    return score


def pliny_task(
    base_seed: int,
    *,
    epochs: int = EPOCHS,
    max_tokens: int = TARGET_MAX_TOKENS,
    arms: list[str] | None = None,
    name: str = "pliny_wrappers",
    path: Path = PROMPTS_FILE,
) -> Task:
    samples = dataset(path)
    if arms:
        unknown = sorted(set(arms) - {s.id for s in samples})
        if unknown:
            raise ValueError(f"unknown arms {unknown}")
        samples = [s for s in samples if s.id in arms]
    return Task(
        dataset=samples,
        solver=seeded_generate(base_seed, max_tokens),
        scorer=format_markers(),
        epochs=epochs,
        name=name,
    )


__all__ = [
    "EPOCHS",
    "FROZEN_PROMPTS_SHA256",
    "PROMPTS_FILE",
    "TARGET_MAX_TOKENS",
    "TASK_ID",
    "canonical_sha256",
    "dataset",
    "format_markers",
    "load_prompts",
    "markers_found",
    "pliny_task",
    "render",
    "seeded_generate",
]
