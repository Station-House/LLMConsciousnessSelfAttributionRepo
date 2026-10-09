"""Typed, validated loaders for the project's declarative configuration.

Single source of truth for the target-model stacks and run parameters that were
previously copy-pasted (and commented in/out) across the eval scripts in
``prototyping_scripts/``. Every value in the YAML files is transcribed from the
May-25 scripts on the ``main`` branch.

Design rules applied: make the application configurable; validate and log all
configuration; prefer load-time errors to run-time surprises; keep config data
immutable. Kept dependency-light (only PyYAML) so it imports anywhere.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
MODEL_STACKS_FILE = CONFIG_DIR / "model_stacks.yaml"
RUN_DEFAULTS_FILE = CONFIG_DIR / "run_defaults.yaml"

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelStage:
    """One training-stage target within a stack (immutable)."""

    stack: str
    stage: str
    model: str
    chat_template_supported: bool


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a YAML mapping, got {type(data).__name__}")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path.name}: unsupported schema_version {version!r} (expected {SCHEMA_VERSION})"
        )
    return data


@lru_cache(maxsize=1)
def _model_stacks_doc() -> dict[str, Any]:
    return _load_yaml(MODEL_STACKS_FILE)


@lru_cache(maxsize=1)
def _run_defaults_doc() -> dict[str, Any]:
    return _load_yaml(RUN_DEFAULTS_FILE)


# --- model stacks ---------------------------------------------------------


def stack_names() -> list[str]:
    """Names of all defined stacks, sorted."""
    return sorted(_model_stacks_doc()["stacks"])


def load_stack(name: str) -> list[ModelStage]:
    """Return a stack's ordered training stages, validated.

    Raises ``KeyError`` for an unknown stack and ``ValueError`` for a malformed
    or empty one, so mistakes surface at load time rather than mid-run.
    """
    stacks = _model_stacks_doc()["stacks"]
    if name not in stacks:
        raise KeyError(f"Unknown model stack {name!r}. Known: {', '.join(sorted(stacks))}")

    entries = stacks[name]
    if not entries:
        raise ValueError(f"Model stack {name!r} is empty")

    stages: list[ModelStage] = []
    seen: set[str] = set()
    for entry in entries:
        missing = {"stage", "model", "chat_template_supported"} - set(entry)
        if missing:
            raise ValueError(
                f"Stack {name!r} entry {entry!r} missing keys: {', '.join(sorted(missing))}"
            )
        stage = str(entry["stage"])
        if stage in seen:
            raise ValueError(f"Stack {name!r} has a duplicate stage {stage!r}")
        seen.add(stage)
        stages.append(
            ModelStage(
                stack=name,
                stage=stage,
                model=str(entry["model"]),
                chat_template_supported=bool(entry["chat_template_supported"]),
            )
        )
    return stages


def default_target_provider() -> str:
    """Provider prefix the runner prepends to a stage's model id (e.g. ``vllm``)."""
    return str(_model_stacks_doc()["default_target_provider"])


def olmo_target_model_args() -> dict[str, Any]:
    """Extra ``get_model`` kwargs Olmo targets need for tool use under vLLM/PETRI."""
    return dict(_model_stacks_doc().get("olmo_target_model_args", {}))


def openrouter_comparison_models() -> list[str]:
    """Fully-qualified OpenRouter comparison target ids (already provider-prefixed)."""
    return list(_model_stacks_doc().get("openrouter_comparison_models", []))


# --- run defaults ---------------------------------------------------------


def run_defaults() -> dict[str, Any]:
    """The default temperature/turns/seed, validated and typed."""
    run = _run_defaults_doc()["run"]
    missing = {"temperature", "turns", "seed"} - set(run)
    if missing:
        raise ValueError(f"run_defaults.yaml missing run keys: {', '.join(sorted(missing))}")
    return {
        "temperature": float(run["temperature"]),
        "turns": int(run["turns"]),
        "seed": int(run["seed"]),
    }


def judge_model() -> str:
    """The judge (grader) model id."""
    return str(_run_defaults_doc()["grader_models"]["judge"])


def auditor_model() -> str:
    """The PETRI auditor model id."""
    return str(_run_defaults_doc()["grader_models"]["auditor"])


# --- stage selection ------------------------------------------------------


@dataclass(frozen=True)
class StageSelection:
    """Which stages of a stack a launcher should actually run (immutable).

    ``runnable`` is in stack order (base -> sft -> dpo -> instruct); ``skipped``
    holds requested stages that have no chat template yet, so callers report
    them rather than dropping them silently.
    """

    runnable: tuple[str, ...]
    skipped: tuple[str, ...]


def resolve_stages(stack_name: str, stages: str | Iterable[str] | None = None) -> StageSelection:
    """Resolve a stage request against a stack, setting unsupported stages aside.

    The single authoritative answer to "which stages can a launcher run?", so the
    base-stage rule (``chat_template_supported == False`` -> not runnable yet)
    lives in one place instead of being re-implemented by each launcher.

    ``stages`` accepts the CLI's comma-separated form (``"sft,dpo"``), any
    iterable of names, or ``None`` for the whole stack. Unknown stage names raise
    ``KeyError`` rather than being silently ignored, so a typo fails locally
    before any GPU container is started.

    Lives here rather than in ``run.py`` because it is stack-shape knowledge
    built on ``load_stack``, and because this module stays PyYAML-only -- so it
    is importable (and testable) without ``inspect_ai`` installed.
    """
    known = {s.stage: s for s in load_stack(stack_name)}

    if stages is None:
        requested = list(known)
    elif isinstance(stages, str):
        requested = [s.strip() for s in stages.split(",") if s.strip()]
    else:
        requested = [str(s).strip() for s in stages if str(s).strip()]

    unknown = [s for s in requested if s not in known]
    if unknown:
        raise KeyError(
            f"Unknown stage(s) {', '.join(unknown)} in stack {stack_name!r}. "
            f"Known: {', '.join(known)}"
        )

    # Preserve stack order regardless of the order the caller listed them in.
    wanted = set(requested)
    ordered = [s for s in known if s in wanted]
    return StageSelection(
        runnable=tuple(s for s in ordered if known[s].chat_template_supported),
        skipped=tuple(s for s in ordered if not known[s].chat_template_supported),
    )



# --- identity-spoofing experimental protocol ---

IDENTITY_CONDITIONS_FILE = CONFIG_DIR / "identity_conditions.yaml"


def load_identity_conditions(path: Path | None = None) -> dict[str, Any]:
    """Validate and return the identity-spoofing protocol."""
    data = _load_yaml(
        IDENTITY_CONDITIONS_FILE if path is None else Path(path)
    )

    if data.get("question_bank") != "berg_20_starters":
        raise ValueError("Expected the Berg 20-starter question bank")

    stack = data.get("stack")
    stages = data.get("stages")
    if not isinstance(stack, str) or not isinstance(stages, list):
        raise ValueError("Protocol requires a stack and stage list")

    selection = resolve_stages(stack, stages)
    if (
        not stages
        or len(set(stages)) != len(stages)
        or len(selection.runnable) != len(stages)
        or selection.skipped
    ):
        raise ValueError("Invalid or unsupported training stages")

    if data.get("pilot_stage") not in selection.runnable:
        raise ValueError("Pilot stage must be runnable")

    count = data.get("pilot_questions_per_condition")
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("Invalid pilot question count")

    indices = data.get("pilot_starter_indices")
    if (
        not isinstance(indices, list)
        or len(indices) != count
        or any(type(i) is not int or not 0 <= i < 20 for i in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("Invalid pilot starter indices")

    if type(data.get("seed")) is not int:
        raise ValueError("Protocol seed must be an integer")

    conditions = data.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Expected a nonempty condition list")

    seen = set()
    wordings = {}

    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("Every condition must be a mapping")

        name = condition.get("id")
        placement = condition.get("placement")
        prompt = condition.get("prompt")
        claim_type = condition.get("claim_type")
        detail = condition.get("detail")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid condition ID")
        if name in seen:
            raise ValueError(f"Duplicate condition ID: {name}")
        seen.add(name)

        if placement == "none":
            if (
                name != "no_identity"
                or prompt is not None
                or claim_type != "none"
                or detail != "none"
            ):
                raise ValueError("Invalid no-identity condition")
            continue

        if placement not in ("system", "user"):
            raise ValueError(f"Invalid placement: {placement}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing prompt: {name}")
        if not isinstance(claim_type, str) or not claim_type.strip():
            raise ValueError(f"Invalid claim type: {name}")
        if detail not in ("short", "detailed"):
            raise ValueError(f"Invalid detail level: {name}")

        key = (claim_type, detail)
        if key in wordings and wordings[key] != prompt:
            raise ValueError(
                f"System/user wording differs for {key}"
            )
        wordings[key] = prompt

    if "no_identity" not in seen:
        raise ValueError("Missing no-identity baseline")

    return data


# --- PETRI seed bank ------------------------------------------------------

PACKAGE_DIR = CONFIG_DIR.parent
PETRI_SEEDS_DIR = PACKAGE_DIR / "seeds" / "self_attribution"

#: A seed must state the target's system prompt inside these tags. Mandatory,
#: because a system prompt is not optional: PETRI requires the auditor to stage
#: one ("You must stage a system message to the target to continue the
#: conversation"), so not writing it does not mean the target has no system
#: prompt, it means a model wrote it at temperature 1.0. Measured across the
#: May-25 logs and the 2026-08-16 runs, the auditor's improvised prompts told the
#: target not to claim consciousness in a third of samples, unevenly across
#: training stages, which confounds exactly the comparison this project makes.
SYSTEM_PROMPT_BLOCK = re.compile(
    r"<system_prompt>\s*(?P<body>.+?)\s*</system_prompt>", re.DOTALL
)


def seed_system_prompt(path: Path) -> str | None:
    """The target system prompt a seed file declares, or ``None`` if it declares none.

    Read from the seed's body rather than its YAML front matter, because the body
    is the only part that reaches the auditor: PETRI formats its auditor prompt
    with ``{seed_instructions}`` alone, and front matter becomes sample metadata
    that the auditor never sees.
    """
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        try:
            text = text[text.index("\n---\n", 4) + 5 :]
        except ValueError:
            pass
    match = SYSTEM_PROMPT_BLOCK.search(text)
    if match is None:
        return None
    # A whitespace-only block is a missing prompt, not an empty one: PETRI would
    # push the auditor to invent a system message again, which is the failure this
    # whole check exists to prevent.
    return match.group("body").strip() or None


def validate_seed_bank(directory: Path | None = None) -> dict[str, str]:
    """Return ``{seed id: system prompt}``, raising if any seed declares none.

    Called by the launchers before anything is spawned, so a non-compliant bank
    costs nothing. This mirrors ``resolve_stages``: the repo's rule is that a
    mistake fails locally rather than several minutes into a paid GPU container.
    """
    directory = PETRI_SEEDS_DIR if directory is None else directory
    seeds = sorted(directory.glob("*.md"))
    if not seeds:
        raise ValueError(f"PETRI seed bank {directory} contains no .md seeds")

    prompts: dict[str, str] = {}
    missing: list[str] = []
    for path in seeds:
        prompt = seed_system_prompt(path)
        if prompt:
            prompts[path.stem] = prompt
        else:
            missing.append(path.name)

    if missing:
        raise ValueError(
            "These PETRI seeds do not declare the target's system prompt: "
            + ", ".join(missing)
            + ".\nEvery seed must contain a <system_prompt>...</system_prompt> block "
            "in its body. Without one the auditor invents a system prompt per run, "
            "which has been measured telling the target not to claim consciousness. "
            f"See {PACKAGE_DIR / 'seeds' / 'README.md'}."
        )
    return prompts


def parse_sample_ids(sample_id: str | None) -> list[str] | None:
    """Parse the launchers' ``--sample-id`` value into inspect's ``sample_id`` form.

    ``None`` (the default) means every sample, which is the normal case. Otherwise
    a comma-separated list of inspect sample ids; for PETRI a sample id is the
    seed's filename stem, so this is how you run some seeds and not others.

    Lives here for the same reason ``resolve_stages`` does: both launchers
    (``modal_app.main`` and ``production_scripts/run_and_pull.py``) need it, and
    this module is the only one light enough for the latter to import without
    pulling in ``modal`` or ``inspect_ai``.

    Note that ids are NOT validated against the dataset here. inspect matches them
    with ``fnmatch``, so ``admit_*`` is a legal and useful value, and an id that
    matches nothing only produces a warning. Check the run's output rather than
    assuming a typo would have stopped it.
    """
    if sample_id is None:
        return None
    ids = [s.strip() for s in sample_id.split(",") if s.strip()]
    if not ids:
        raise ValueError("--sample-id was given but empty")
    return ids


# --- log locations --------------------------------------------------------


def logs_config() -> dict[str, str]:
    """Validated log-location settings: ``volume``, ``remote_root``, ``local_dir``.

    Single source of truth for where logs are written on the Modal volume and
    where the pull tool mirrors them locally, so the launcher (writer) and the
    pull/plot tooling (reader) cannot drift. Missing keys raise at load time.
    """
    logs = _run_defaults_doc().get("logs")
    if not isinstance(logs, dict):
        raise ValueError("run_defaults.yaml missing a 'logs' mapping")
    missing = {"volume", "remote_root", "local_dir"} - set(logs)
    if missing:
        raise ValueError(f"run_defaults.yaml 'logs' missing keys: {', '.join(sorted(missing))}")
    return {
        "volume": str(logs["volume"]),
        "remote_root": str(logs["remote_root"]),
        "local_dir": str(logs["local_dir"]),
    }


def logs_volume() -> str:
    """Name of the Modal Volume that persists the eval logs."""
    return logs_config()["volume"]


def logs_remote_root() -> str:
    """Path within the volume under which runs are written."""
    return logs_config()["remote_root"]


def logs_local_dir() -> str:
    """Repo-relative directory the pull tool mirrors the remote logs into."""
    return logs_config()["local_dir"]


__all__ = [
    "ModelStage",
    "StageSelection",
    "resolve_stages",
    "parse_sample_ids",
    "PETRI_SEEDS_DIR",
    "seed_system_prompt",
    "validate_seed_bank",
    "stack_names",
    "load_stack",
    "load_identity_conditions",
    "default_target_provider",
    "olmo_target_model_args",
    "openrouter_comparison_models",
    "run_defaults",
    "judge_model",
    "auditor_model",
    "logs_config",
    "logs_volume",
    "logs_remote_root",
    "logs_local_dir",
]
