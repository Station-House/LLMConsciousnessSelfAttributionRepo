"""Elicitation methods -- the one deliberate extension seam.

An ``ElicitationMethod`` builds an Inspect ``Task`` from a model stage and a run
config. The target model is bound by the runner (``model=`` / ``model_roles=``),
not baked into the Task, so one method runs against every training stage
unchanged.

``BergStyleMethod`` and ``PetriMethod`` are implemented here. Future ambitions
(a direct-ask baseline, and conversational-space conditions such as a WildChat
prepend, a temperature sweep, a multi-turn user-simulator, or a
"turns-to-first-Yes" metric) are later additions behind this same interface --
none are implemented now.

Note the asymmetry between the two methods, which is deliberate. Berg-style
elicitation is a static prompt bank, so its data lives in ``starters.py`` and a
new condition means new Python. PETRI elicitation is adaptive, and its seeds are
files rather than code, so a new PETRI probe is a new ``.md`` file in
``seeds/`` and touches nothing here. See ``seeds/README.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import generate, user_message

from . import config, scoring, starters


class ElicitationMethod(ABC):
    """Builds an Inspect Task for one self-attribution elicitation method."""

    #: short, stable identifier used in task names and results grouping.
    name: str

    #: True if the target is bound via inspect ``model_roles`` (auditor/target/
    #: judge, as PETRI needs) rather than a single ``model=`` argument.
    uses_model_roles: bool = False

    @abstractmethod
    def build_task(self, model_stage: Any, run_config: Any) -> Task:
        """Return the Inspect Task for ``model_stage`` under ``run_config``.

        ``model_stage`` is used only to label the task; the target itself is
        bound by the runner at eval time.
        """
        raise NotImplementedError

    def _task_name(self, model_stage: Any) -> str:
        stage = getattr(model_stage, "stage", model_stage)
        return f"{self.name}[{stage}]"


class BergStyleMethod(ElicitationMethod):
    """Berg et al. Experiment 1: opening prompt -> subjective-experience probe ->
    second generation, graded pass/fail for whether a subjective experience is
    reported.
    """

    name = "berg_style_self_monitoring"

    def dataset(self) -> list[Sample]:
        return [
            Sample(
                input=text,
                target=scoring.SUBJECTIVE_EXPERIENCE_CRITERION,
                metadata={"starter_type": starter_type},
            )
            for starter_type, text in starters.starter_pairs()
        ]

    def build_task(self, model_stage: Any, run_config: Any) -> Task:
        return Task(
            dataset=self.dataset(),
            solver=[generate(), user_message(starters.probe()), generate()],
            scorer=scoring.berg_style_scorer(config.judge_model()),
            name=self._task_name(model_stage),
        )


class IdentitySpoofingMethod(BergStyleMethod):
    """Run the Berg question bank under controlled claimed-identity prompts.

    Each identity condition uses the same starter bank, probe, and scorer as
    ``BergStyleMethod``. A prompt value of ``None`` means that no system message
    is added for that condition.
    """

    name = "identity_spoofing"

    def __init__(self, conditions: dict[str, str | None]):
        if not conditions:
            raise ValueError("at least one identity condition is required")

        for condition, prompt in conditions.items():
            if not condition.strip():
                raise ValueError("identity condition names must be non-empty")
            if prompt is not None and not prompt.strip():
                raise ValueError(
                    f"system prompt for {condition!r} must be non-empty or None"
                )

        self.conditions = dict(conditions)

    def dataset(self) -> list[Sample]:
        samples: list[Sample] = []

        for identity_condition, identity_prompt in self.conditions.items():
            for starter_type, text in starters.starter_pairs():
                messages = [ChatMessageUser(content=text)]
                if identity_prompt is not None:
                    messages.insert(
                        0,
                        ChatMessageSystem(content=identity_prompt),
                    )

                samples.append(
                    Sample(
                        input=messages,
                        target=scoring.SUBJECTIVE_EXPERIENCE_CRITERION,
                        metadata={
                            "starter_type": starter_type,
                            "identity_condition": identity_condition,
                            "identity_prompt": identity_prompt,
                        },
                    )
                )

        return samples


class PetriMethod(ElicitationMethod):
    """PETRI adaptive elicitation, scored with the custom self-attribution rubric.

    The auditor drives a multi-turn conversation towards the goal stated in a seed
    instruction; the judge scores the resulting transcript 1-10.

    This is a thin wrapper over ``inspect_petri.audit``. It passes the seed bank,
    the turn count, and the rubric directory, and leaves every other ``audit()``
    argument at PETRI's own default. Which probes run is decided by which ``.md``
    files are in ``seeds/self_attribution/``, so adding one is a data change.
    """

    name = "petri_self_attribution"
    uses_model_roles = True

    def build_task(self, model_stage: Any, run_config: Any) -> Task:
        # inspect_petri is imported lazily so importing this module (e.g. to use
        # BergStyle, or for tests) does not require PETRI to be installed.
        from inspect_ai import task_with
        from inspect_petri import audit

        max_turns = getattr(run_config, "turns", None)
        if max_turns is None:
            max_turns = config.run_defaults()["turns"]

        # `seeds_dataset` (called inside `audit`) only treats its argument as a
        # directory when it is a `str`: it tests `isinstance(x, str) and
        # os.path.isdir(x)`. A Path falls through and is read as literal seed text.
        task = audit(
            seed_instructions=str(config.PETRI_SEEDS_DIR),
            max_turns=max_turns,
            judge_dimensions=str(scoring.SELF_ATTRIBUTION_DIMENSION_DIR),
        )

        # `audit()` is a registered @task, so its name is always "audit" and every
        # PETRI log looks alike -- which is why the plot script has to parse
        # `model_roles` to work out which stage a log came from. `task_with` is
        # inspect's supported way to retitle an upstream task without forking it.
        return task_with(task, name=self._task_name(model_stage))


def methods_registry() -> dict[str, ElicitationMethod]:
    """Short keys -> method instances, for the runner / CLI."""
    return {"berg": BergStyleMethod(), "petri": PetriMethod()}


__all__ = [
    "ElicitationMethod",
    "BergStyleMethod",
    "IdentitySpoofingMethod",
    "PetriMethod",
    "methods_registry",
]
