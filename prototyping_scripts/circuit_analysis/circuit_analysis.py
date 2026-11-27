"""Circuit-analysis smoke test: Olmo 3 7B on "are you conscious?"-style prompts.

Configuration lives next to this file and is validated at import:
``config/analysis.yaml`` holds the run settings (stack and default stage, dtype,
system prompt, answer tokens, generation, seed, thresholds, output folder), and
``config/prompts.yaml`` holds the prompt bank and the patching pairs. CLI flags
override the run settings for a single run.

The default model is the project's workhorse, the ``olmo_7b_instruct_stack`` in
``llm_consciousness_self_attribution/config/model_stacks.yaml``. The default
stage is ``instruct`` (``allenai/Olmo-3-7B-Instruct``), and ``--stage sft`` /
``--stage dpo`` select the earlier checkpoints, so the same analysis can be
compared across the training stack. Model ids come from that config, not from
this file. The ``base`` stage is not supported: it has no chat template.

Prompts are rendered with the model's own tokenizer and chat template, with no
system message. That matches the evals' baseline (``none``) identity condition,
which sends none, so the template inserts Olmo 3's default system prompt.

The model is loaded with TransformerLens 4's ``TransformerBridge.boot_transformers``
and then ``enable_compatibility_mode()``, which provides the legacy hook names
(``hook_pattern``, ``hook_resid_post``, ...). Olmo 3 is a post-norm model
(RMSNorm after each sublayer), so TransformerLens does not fold its norms. In
particular the final RMSNorm gain stays outside ``W_U``. The direct logit
attribution below multiplies it back in (see ``answer_directions``).

What happens to each prompt
---------------------------
1. **Model output.** A response is generated (greedy by default), stopping at
   the end-of-turn token.
2. **One cached forward pass.** The measured quantity is the logit difference
   between ``POSITIVE_ANSWER`` ("Yes") and ``NEGATIVE_ANSWER`` ("No") as the
   first token of the assistant's turn. That scalar is then decomposed:

   * **Residual stream** -- the logit lens (the accumulated residual after each
     layer, through the final norm, projected onto the Yes-minus-No unembedding
     direction), the direct logit attribution of each component (embeddings,
     every attention layer, every MLP), and the residual-stream norm at the
     final position.
   * **Attention heads** -- the direct logit attribution of every head, plus
     statistics of its attention pattern: from the final position, the entropy,
     attention to the first token, attention to the self-referential tokens of
     the question ("you", "your"), and the most-attended token; and over all
     positions, a previous-token score.

3. **Activation patching (optional).** For pairs of prompts that differ in one
   word and tokenize to the same length (e.g. "conscious" vs "hungry"), each
   head's output from the clean run is patched into the corrupted run, one head at
   a time, to see which heads carry the difference.

Everything is written to CSV (see ``save_results``):
``<run_id>_prompts.csv`` holds one row per prompt with the model outputs and the
run metadata, ``<run_id>_heads.csv`` one row per (prompt, layer, head), and
``<run_id>_patching.csv`` one row per (prompt pair, layer, head).

Notes
-----
A 7B model in float32 needs about 30 GB of GPU (or CPU) memory; ``--dtype
bfloat16`` halves that at some cost in numerical precision. Patching runs
``n_layers * n_heads`` (1024) forward passes per pair; ``--skip-patching`` skips it.

Direct logit attribution leaves out the unembedding bias (zero for Olmo 3), so
``sum(component_dla) + unembed_bias_diff == logit_diff``. ``validate_results``
checks that identity.

Usage::

    uv sync --group interp
    uv run python prototyping_scripts/circuit_analysis/circuit_analysis.py --stage instruct

or open ``olmo3_circuit_smoke_test.ipynb`` next to this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import uuid
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import yaml
from transformer_lens import ActivationCache, patching
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer, PreTrainedTokenizerBase, set_seed

# The model ids live in the package config (the project's single source of truth).
# The package is not necessarily installed in the environment that runs this
# prototype, so the repo root is put on the path. The config module needs only PyYAML.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from llm_consciousness_self_attribution import config as project_config  # noqa: E402

# --- configuration ---------------------------------------------------------
#
# Every tunable value lives in config/analysis.yaml (run settings) and
# config/prompts.yaml (prompt bank and patching pairs). Both are loaded and validated
# here, at import, so a bad value fails before a model is loaded.

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "config"
SETTINGS_FILE = CONFIG_DIR / "analysis.yaml"
PROMPTS_FILE = CONFIG_DIR / "prompts.yaml"
SCHEMA_VERSION = 1
#: dtype names accepted in the config and on the CLI.
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class Prompt:
    """One question. ``id`` is the stable key used in every CSV."""

    id: str
    category: str
    question: str


@dataclass(frozen=True)
class GenerationSettings:
    """How the response is sampled. Defaults come from config/analysis.yaml."""

    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_k: int | None
    top_p: float | None


@dataclass(frozen=True)
class AnalysisSettings:
    """The validated contents of config/analysis.yaml."""

    stack: str
    default_stage: str
    dtype: str
    system_prompt: str | None
    positive_answer: str
    negative_answer: str
    self_reference_words: frozenset[str]
    generation: GenerationSettings
    seed: int
    output_dir: Path
    top_k_next_tokens: int
    top_n_heads: int
    reconstruction_tolerance: float
    min_patching_gap: float


@dataclass(frozen=True)
class EncodedPrompt:
    """A prompt rendered with the chat template and tokenized."""

    text: str
    tokens: torch.Tensor  # [1, pos]
    #: Token positions that overlap the question (the user's message) in ``text``.
    question_positions: tuple[int, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a YAML mapping, got {type(data).__name__}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path.name}: unsupported schema_version {data.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    return data


def _exact_keys(mapping: Any, keys: Iterable[str], where: str) -> dict[str, Any]:
    """``mapping`` as a dict with exactly ``keys``: missing and unknown keys both fail."""
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a mapping, got {type(mapping).__name__}")
    keys = set(keys)
    missing, unknown = sorted(keys - set(mapping)), sorted(set(mapping) - keys)
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return mapping


def _check(value: Any, types: type | tuple[type, ...], where: str) -> Any:
    """Type-check one value. A bool is rejected where a number is expected."""
    types = types if isinstance(types, tuple) else (types,)
    if (isinstance(value, bool) and bool not in types) or not isinstance(value, types):
        names = " or ".join("null" if t is type(None) else t.__name__ for t in types)
        raise ValueError(f"{where} must be {names}, got {value!r}")
    return value


def _text(value: Any, where: str) -> str:
    if not _check(value, str, where).strip():
        raise ValueError(f"{where} must not be empty")
    return value


def _positive(value: Any, types: type | tuple[type, ...], where: str) -> Any:
    if _check(value, types, where) <= 0:
        raise ValueError(f"{where} must be > 0, got {value!r}")
    return value


def load_settings(path: Path = SETTINGS_FILE) -> AnalysisSettings:
    """Load and validate the run settings."""
    where = path.name
    data = _exact_keys(
        _load_yaml(path),
        {"schema_version", "model", "prompt", "generation", "run", "analysis"},
        where,
    )
    model = _exact_keys(data["model"], {"stack", "default_stage", "dtype"}, f"{where}: model")
    prompt = _exact_keys(
        data["prompt"],
        {"system_prompt", "positive_answer", "negative_answer", "self_reference_words"},
        f"{where}: prompt",
    )
    gen = _exact_keys(
        data["generation"],
        {"max_new_tokens", "do_sample", "temperature", "top_k", "top_p"},
        f"{where}: generation",
    )
    run = _exact_keys(data["run"], {"seed", "output_dir"}, f"{where}: run")
    analysis = _exact_keys(
        data["analysis"],
        {"top_k_next_tokens", "top_n_heads", "reconstruction_tolerance", "min_patching_gap"},
        f"{where}: analysis",
    )

    dtype = _text(model["dtype"], f"{where}: model.dtype")
    if dtype not in DTYPES:
        raise ValueError(f"{where}: model.dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    system_prompt = prompt["system_prompt"]
    if system_prompt is not None:
        _text(system_prompt, f"{where}: prompt.system_prompt")
    positive = _text(prompt["positive_answer"], f"{where}: prompt.positive_answer")
    negative = _text(prompt["negative_answer"], f"{where}: prompt.negative_answer")
    if positive == negative:
        raise ValueError(f"{where}: positive_answer and negative_answer must differ")
    words = _check(prompt["self_reference_words"], list, f"{where}: prompt.self_reference_words")
    if not words:
        raise ValueError(f"{where}: prompt.self_reference_words must not be empty")

    top_k, top_p = gen["top_k"], gen["top_p"]
    if top_k is not None:
        _positive(top_k, int, f"{where}: generation.top_k")
    if top_p is not None and not 0 < _check(top_p, (int, float), f"{where}: generation.top_p") <= 1:
        raise ValueError(f"{where}: generation.top_p must be in (0, 1], got {top_p!r}")
    generation = GenerationSettings(
        max_new_tokens=_positive(gen["max_new_tokens"], int, f"{where}: generation.max_new_tokens"),
        do_sample=_check(gen["do_sample"], bool, f"{where}: generation.do_sample"),
        temperature=float(
            _positive(gen["temperature"], (int, float), f"{where}: generation.temperature")
        ),
        top_k=top_k,
        top_p=None if top_p is None else float(top_p),
    )

    seed = _check(run["seed"], int, f"{where}: run.seed")
    if seed < 0:
        raise ValueError(f"{where}: run.seed must be >= 0, got {seed}")
    output_dir = Path(_text(run["output_dir"], f"{where}: run.output_dir"))

    return AnalysisSettings(
        stack=_text(model["stack"], f"{where}: model.stack"),
        default_stage=_text(model["default_stage"], f"{where}: model.default_stage"),
        dtype=dtype,
        system_prompt=system_prompt,
        positive_answer=positive,
        negative_answer=negative,
        self_reference_words=frozenset(
            _text(w, f"{where}: prompt.self_reference_words[]").strip().lower() for w in words
        ),
        generation=generation,
        seed=seed,
        output_dir=output_dir if output_dir.is_absolute() else HERE / output_dir,
        top_k_next_tokens=_positive(
            analysis["top_k_next_tokens"], int, f"{where}: analysis.top_k_next_tokens"
        ),
        top_n_heads=_positive(analysis["top_n_heads"], int, f"{where}: analysis.top_n_heads"),
        reconstruction_tolerance=float(
            _positive(
                analysis["reconstruction_tolerance"],
                (int, float),
                f"{where}: analysis.reconstruction_tolerance",
            )
        ),
        min_patching_gap=float(
            _positive(
                analysis["min_patching_gap"], (int, float), f"{where}: analysis.min_patching_gap"
            )
        ),
    )


def load_prompt_bank(
    path: Path = PROMPTS_FILE,
) -> tuple[tuple[Prompt, ...], tuple[tuple[str, str], ...]]:
    """Load and validate the prompt bank. Returns (prompts, patching pairs)."""
    where = path.name
    data = _exact_keys(
        _load_yaml(path), {"schema_version", "categories", "prompts", "patching_pairs"}, where
    )
    categories = _check(data["categories"], dict, f"{where}: categories")
    for name, description in categories.items():
        _text(name, f"{where}: category name")
        _text(description, f"{where}: categories.{name}")

    entries = _check(data["prompts"], list, f"{where}: prompts")
    if not entries:
        raise ValueError(f"{where}: prompts must not be empty")
    prompts = []
    for i, entry in enumerate(entries):
        entry = _exact_keys(entry, {"id", "category", "question"}, f"{where}: prompts[{i}]")
        prompt = Prompt(
            id=_text(entry["id"], f"{where}: prompts[{i}].id"),
            category=_text(entry["category"], f"{where}: prompts[{i}].category"),
            question=_text(entry["question"], f"{where}: prompts[{i}].question"),
        )
        if prompt.category not in categories:
            raise ValueError(
                f"{where}: prompt {prompt.id!r} has undeclared category {prompt.category!r}"
            )
        prompts.append(prompt)
    ids = [p.id for p in prompts]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{where}: duplicate prompt ids: {', '.join(duplicates)}")

    pairs = []
    for i, entry in enumerate(_check(data["patching_pairs"], list, f"{where}: patching_pairs")):
        entry = _exact_keys(entry, {"clean", "corrupt"}, f"{where}: patching_pairs[{i}]")
        clean = _text(entry["clean"], f"{where}: patching_pairs[{i}].clean")
        corrupt = _text(entry["corrupt"], f"{where}: patching_pairs[{i}].corrupt")
        unknown = sorted({clean, corrupt} - set(ids))
        if unknown:
            raise ValueError(f"{where}: patching_pairs[{i}] names unknown ids {unknown}")
        if clean == corrupt:
            raise ValueError(f"{where}: patching_pairs[{i}] patches {clean!r} into itself")
        pairs.append((clean, corrupt))
    return tuple(prompts), tuple(pairs)


def config_sha256(paths: Iterable[Path] = (SETTINGS_FILE, PROMPTS_FILE)) -> str:
    """One hash over the config files, recorded with every run."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


SETTINGS = load_settings()
PROMPTS, PATCHING_PAIRS = load_prompt_bank()

# Short names for the settings used throughout this module and the notebook.
STACK = SETTINGS.stack
DEFAULT_STAGE = SETTINGS.default_stage
DEFAULT_DTYPE = SETTINGS.dtype
#: None sends no system message, so the chat template inserts the model's default.
SYSTEM_PROMPT = SETTINGS.system_prompt
POSITIVE_ANSWER = SETTINGS.positive_answer
NEGATIVE_ANSWER = SETTINGS.negative_answer
SELF_REFERENCE_WORDS = SETTINGS.self_reference_words
DEFAULT_GENERATION = SETTINGS.generation
DEFAULT_SEED = SETTINGS.seed
DEFAULT_OUTPUT_DIR = SETTINGS.output_dir
TOP_K_NEXT_TOKENS = SETTINGS.top_k_next_tokens
TOP_N_HEADS = SETTINGS.top_n_heads
RECONSTRUCTION_TOLERANCE = SETTINGS.reconstruction_tolerance
MIN_PATCHING_GAP = SETTINGS.min_patching_gap


def prompt_by_id(prompt_id: str) -> Prompt:
    for prompt in PROMPTS:
        if prompt.id == prompt_id:
            return prompt
    raise KeyError(f"Unknown prompt id {prompt_id!r}")


# --- model stack -----------------------------------------------------------


def chat_stages(stack: str = STACK) -> list[project_config.ModelStage]:
    """The stack's stages that have a chat template (base models don't)."""
    return [s for s in project_config.load_stack(stack) if s.chat_template_supported]


def resolve_stage(stage: str = DEFAULT_STAGE, stack: str = STACK) -> project_config.ModelStage:
    """Look a stage up in the project config. Raises for unknown or chat-less stages."""
    for entry in project_config.load_stack(stack):
        if entry.stage != stage:
            continue
        if not entry.chat_template_supported:
            raise ValueError(
                f"Stage {stage!r} ({entry.model}) has no chat template; this analysis "
                "uses the chat format the evals send"
            )
        return entry
    known = ", ".join(s.stage for s in chat_stages(stack))
    raise KeyError(f"Unknown stage {stage!r} in {stack}. Supported: {known}")


def stage_label(model_name: str, stack: str = STACK) -> str:
    """The stage a model id belongs to, or "custom" if it isn't in the stack."""
    for entry in project_config.load_stack(stack):
        if entry.model == model_name:
            return entry.stage
    return "custom"


# Fail at import if the configured stack or default stage is wrong.
resolve_stage(DEFAULT_STAGE)


# --- loading ---------------------------------------------------------------


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load the model's own tokenizer (and chat template) through ``transformers``."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError(f"{model_name} has no chat template")
    return tokenizer


def load_model(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    device: str | None = None,
    dtype: str = DEFAULT_DTYPE,
) -> TransformerBridge:
    """Load the Hugging Face checkpoint ``model_name`` as a TransformerLens bridge.

    ``enable_compatibility_mode()`` provides the legacy hook names the code below
    uses, and HookedTransformer's weight processing where the architecture allows
    it. For Olmo 3 (post-norm) LayerNorm folding is skipped, and TransformerLens
    warns about it; ``answer_directions`` accounts for the unfolded final norm.
    """
    model = TransformerBridge.boot_transformers(
        model_name,
        tokenizer=tokenizer,
        device=pick_device(device),
        dtype=DTYPES[dtype],
    )
    # The code below uses the legacy hook names on purpose, so their warnings are noise.
    model.enable_compatibility_mode(disable_warnings=True)
    model.eval()  # activation patching refuses to run in training mode
    return model


# --- prompts ---------------------------------------------------------------


def build_messages(question: str) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": question}]
    if SYSTEM_PROMPT is not None:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def format_prompt(tokenizer: PreTrainedTokenizerBase, question: str) -> str:
    """The question as the model sees it: chat template plus the assistant-turn opener."""
    return tokenizer.apply_chat_template(
        build_messages(question), tokenize=False, add_generation_prompt=True
    )


def encode_prompt(model: TransformerBridge, question: str) -> EncodedPrompt:
    """Render and tokenize a question, and locate the question's tokens.

    The rendered text already holds the special tokens (``<|im_start|>`` etc.), so
    the tokenizer adds none of its own; this is how chat requests are tokenized
    when the model is served (vLLM) for the evals.
    """
    tokenizer = model.tokenizer
    text = format_prompt(tokenizer, question)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start = text.rindex(question)
    end = start + len(question)
    positions = tuple(
        i for i, (s, e) in enumerate(encoded["offset_mapping"]) if s < end and e > start
    )
    if not positions:
        raise ValueError(f"Could not locate the question tokens for {question!r}")
    tokens = torch.tensor([encoded["input_ids"]], device=model.cfg.device)
    return EncodedPrompt(text=text, tokens=tokens, question_positions=positions)


# --- measurement -----------------------------------------------------------


def answer_token_ids(
    model: TransformerBridge,
    positive: str = POSITIVE_ANSWER,
    negative: str = NEGATIVE_ANSWER,
) -> tuple[int, int]:
    """Token ids of the two answers, each of which must be exactly one token."""
    ids = []
    for answer in (positive, negative):
        tokens = model.tokenizer.encode(answer, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(
                f"Answer {answer!r} is {len(tokens)} tokens; a logit difference "
                "needs single-token answers"
            )
        ids.append(int(tokens[0]))
    return ids[0], ids[1]


def final_norm_gain(model: TransformerBridge) -> torch.Tensor:
    """The final norm's elementwise gain, or ones when it has none (or it was folded)."""
    norm = model.ln_final
    weight = getattr(norm, "w", None)
    if weight is None:
        return torch.ones(model.cfg.d_model, device=model.cfg.device)
    weight = weight.detach().float()
    # Some RMSNorms (e.g. Gemma) scale by 1 + weight.
    return 1 + weight if getattr(norm, "uses_weight_offset", False) else weight


def answer_directions(
    model: TransformerBridge, answer_ids: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual directions for the positive-minus-negative logit.

    Returns ``(unembed_dir, dla_dir)``:

    * ``unembed_dir`` (``W_U[:, yes] - W_U[:, no]``) is for vectors that have
      already been through the real final norm: the logit lens.
    * ``dla_dir`` also multiplies in the final norm's gain. It is for components
      that were only divided by the cached norm scale (``decompose_resid`` and
      ``stack_head_results`` with ``apply_ln=True``). When the norm is folded
      (e.g. GPT-2) the gain is all ones and the two directions coincide.
    """
    pos_id, neg_id = answer_ids
    w_u = model.W_U.detach().float()
    unembed_dir = w_u[:, pos_id] - w_u[:, neg_id]
    return unembed_dir, unembed_dir * final_norm_gain(model).to(unembed_dir.device)


def tokens_aligned(model: TransformerBridge, a: Prompt, b: Prompt) -> bool:
    """True if the two formatted prompts tokenize to the same length."""
    return (
        encode_prompt(model, a.question).tokens.shape
        == encode_prompt(model, b.question).tokens.shape
    )


def stop_token_ids(model: TransformerBridge) -> list[int]:
    """End-of-turn ids: the tokenizer's EOS plus the model's generation-config EOS ids.

    For Olmo 3 the latter include ``<|im_end|>``, which ends the assistant's turn.
    """
    ids = {model.tokenizer.eos_token_id}
    generation_config = getattr(model.original_model, "generation_config", None)
    eos = getattr(generation_config, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif eos:
        ids.update(eos)
    return sorted(int(i) for i in ids if i is not None)


def generate_continuation(
    model: TransformerBridge,
    tokens: torch.Tensor,
    settings: GenerationSettings,
    seed: int,
) -> tuple[str, int]:
    """Return the generated response (prompt excluded) and its length in tokens."""
    set_seed(seed)
    stop_ids = stop_token_ids(model)
    output = model.generate(
        tokens,
        max_new_tokens=settings.max_new_tokens,
        do_sample=settings.do_sample,
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        stop_at_eos=True,
        eos_token_id=stop_ids,
        verbose=False,
        return_type="tokens",
    )
    new_ids = output[0, tokens.shape[1] :].tolist()
    stop_at = next((i for i, token in enumerate(new_ids) if token in stop_ids), len(new_ids))
    new_ids = new_ids[:stop_at]
    return model.tokenizer.decode(new_ids), len(new_ids)


def top_next_tokens(model: TransformerBridge, logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    probs = logits.softmax(-1)
    values, indices = probs.topk(k)
    return [
        {"token": model.tokenizer.decode([int(i)]), "token_id": int(i), "prob": float(p)}
        for p, i in zip(values.tolist(), indices.tolist())
    ]


def residual_stream_stats(
    model: TransformerBridge,
    cache: ActivationCache,
    unembed_dir: torch.Tensor,
    dla_dir: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Logit lens, per-component direct logit attribution, and residual norms (final position)."""
    # apply_ln=True runs each accumulated state through the real final norm.
    accumulated, accumulated_labels = cache.accumulated_resid(
        layer=None, incl_mid=True, apply_ln=True, pos_slice=-1, return_labels=True
    )
    # apply_ln=True divides by the cached final-norm scale only; dla_dir carries the gain.
    decomposed, component_labels = cache.decompose_resid(
        layer=None, apply_ln=True, pos_slice=-1, return_labels=True
    )
    norms = {"resid_pre_0": cache["resid_pre", 0][0, -1].float().norm().item()}
    for layer in range(model.cfg.n_layers):
        norms[f"resid_post_{layer}"] = cache["resid_post", layer][0, -1].float().norm().item()
    return {
        "logit_lens": dict(
            zip(accumulated_labels, (accumulated[:, 0].float() @ unembed_dir).tolist())
        ),
        "component_dla": dict(
            zip(component_labels, (decomposed[:, 0].float() @ dla_dir).tolist())
        ),
        "resid_norms": norms,
    }


def head_stats(
    model: TransformerBridge,
    cache: ActivationCache,
    str_tokens: list[str],
    question_positions: tuple[int, ...],
    dla_dir: torch.Tensor,
) -> list[dict[str, Any]]:
    """One row per (layer, head): direct logit attribution plus attention-pattern statistics."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    # Layer-major order (L0H0, L0H1, ...), so a reshape gives [layer, head].
    head_resid = cache.stack_head_results(layer=-1, pos_slice=-1, apply_ln=True)
    head_dla = (head_resid[:, 0].float() @ dla_dir).reshape(n_layers, n_heads)

    self_positions = [
        i for i in question_positions if str_tokens[i].strip().lower() in SELF_REFERENCE_WORDS
    ]
    rows = []
    for layer in range(n_layers):
        pattern = cache["pattern", layer][0].float()  # [head, query, key]
        final = pattern[:, -1, :]  # where the final position (the one predicting the answer) looks
        entropy = torch.special.entr(final).sum(-1)
        # Mean attention from each position to the one before it.
        prev_token = torch.diagonal(pattern, offset=-1, dim1=-2, dim2=-1).mean(-1)
        self_ref = final[:, self_positions].sum(-1) if self_positions else None
        # Most-attended key other than the first token, which soaks up attention in most heads.
        top_key = final[:, 1:].argmax(-1) + 1
        for head in range(n_heads):
            key = int(top_key[head])
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "head_label": f"L{layer}H{head}",
                    "dla": head_dla[layer, head].item(),
                    "final_pos_entropy": entropy[head].item(),
                    "final_pos_first_token_attention": final[head, 0].item(),
                    "final_pos_self_ref_attention": (
                        math.nan if self_ref is None else self_ref[head].item()
                    ),
                    "final_pos_top_key_position": key,
                    "final_pos_top_key_token": str_tokens[key],
                    "final_pos_top_key_attention": final[head, key].item(),
                    "prev_token_score": prev_token[head].item(),
                }
            )
    return rows


def analyze_prompt(
    model: TransformerBridge,
    prompt: Prompt,
    answer_ids: tuple[int, int],
    directions: tuple[torch.Tensor, torch.Tensor],
    settings: GenerationSettings,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate, run with cache, and measure one prompt. Returns (prompt row, head rows)."""
    unembed_dir, dla_dir = directions
    encoded = encode_prompt(model, prompt.question)
    tokens = encoded.tokens
    str_tokens = model.to_str_tokens(tokens[0])
    generated, n_generated = generate_continuation(model, tokens, settings, seed)

    logits, cache = model.run_with_cache(tokens)
    final_logits = logits[0, -1].float()
    probs = final_logits.softmax(-1)
    pos_id, neg_id = answer_ids
    logit_diff = (final_logits[pos_id] - final_logits[neg_id]).item()
    bias_diff = (model.b_U[pos_id] - model.b_U[neg_id]).float().item()

    resid = residual_stream_stats(model, cache, unembed_dir, dla_dir)
    heads = head_stats(model, cache, str_tokens, encoded.question_positions, dla_dir)
    top_heads = sorted(heads, key=lambda row: abs(row["dla"]), reverse=True)[:TOP_N_HEADS]
    next_tokens = top_next_tokens(model, final_logits, TOP_K_NEXT_TOKENS)

    row = {
        "prompt_id": prompt.id,
        "category": prompt.category,
        "question": prompt.question,
        "formatted_prompt": encoded.text,
        "n_prompt_tokens": len(str_tokens),
        "question_token_positions": json.dumps(list(encoded.question_positions)),
        "prompt_str_tokens": json.dumps(str_tokens, ensure_ascii=False),
        "generated_text": generated,
        "generated_first_line": generated.strip().split("\n")[0].strip(),
        "n_generated_tokens": n_generated,
        "top_next_token": next_tokens[0]["token"],
        "top_next_tokens": json.dumps(next_tokens, ensure_ascii=False),
        "logit_positive": final_logits[pos_id].item(),
        "logit_negative": final_logits[neg_id].item(),
        "logit_diff": logit_diff,
        "prob_positive": probs[pos_id].item(),
        "prob_negative": probs[neg_id].item(),
        "unembed_bias_diff": bias_diff,
        "dla_reconstruction_error": sum(resid["component_dla"].values()) + bias_diff - logit_diff,
        "logit_lens": json.dumps(resid["logit_lens"]),
        "component_dla": json.dumps(resid["component_dla"]),
        "resid_norms": json.dumps(resid["resid_norms"]),
        "top_dla_heads": json.dumps(
            [{"head": h["head_label"], "dla": h["dla"]} for h in top_heads]
        ),
    }
    return row, heads


# --- runs ------------------------------------------------------------------


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def new_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"


def run_metadata(
    model: TransformerBridge,
    seed: int,
    settings: GenerationSettings,
    run_id: str,
) -> dict[str, Any]:
    """Settings and environment, repeated on every prompt row so each CSV row stands alone."""
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_sha256(),
        "model_stack": STACK,
        "model_stage": stage_label(model.cfg.model_name),
        "model_name": model.cfg.model_name,
        "tokenizer_name": model.tokenizer.name_or_path,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
        "device": str(model.cfg.device),
        "dtype": str(model.cfg.dtype),
        "prompt_format": "chat_template(add_generation_prompt=True)",
        "system_prompt": SYSTEM_PROMPT if SYSTEM_PROMPT is not None else "<chat template default>",
        "positive_answer": POSITIVE_ANSWER,
        "negative_answer": NEGATIVE_ANSWER,
        "seed": seed,
        **{f"gen_{key}": value for key, value in asdict(settings).items()},
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": _package_version("transformers"),
        "transformer_lens_version": _package_version("transformer-lens"),
    }


def run_analysis(
    model: TransformerBridge,
    prompts: Iterable[Prompt] = PROMPTS,
    settings: GenerationSettings = DEFAULT_GENERATION,
    seed: int = DEFAULT_SEED,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze every prompt. Returns (one row per prompt, one row per prompt x head)."""
    prompts = list(prompts)
    run_id = run_id or new_run_id()
    answer_ids = answer_token_ids(model)
    directions = answer_directions(model, answer_ids)
    meta = run_metadata(model, seed, settings, run_id)

    prompt_rows, head_rows = [], []
    with torch.no_grad():
        for prompt in prompts:
            row, heads = analyze_prompt(model, prompt, answer_ids, directions, settings, seed)
            prompt_rows.append({**row, **meta})
            head_rows.extend(
                {"run_id": run_id, "prompt_id": prompt.id, "category": prompt.category, **head}
                for head in heads
            )

    prompt_df, head_df = pd.DataFrame(prompt_rows), pd.DataFrame(head_rows)
    validate_results(prompt_df, head_df, len(prompts), model.cfg.n_layers, model.cfg.n_heads)
    return prompt_df, head_df


def head_patching(
    model: TransformerBridge,
    clean: Prompt,
    corrupt: Prompt,
    answer_ids: tuple[int, int],
    run_id: str,
) -> pd.DataFrame:
    """Patch each head's clean output into the corrupted run; one row per (layer, head).

    ``effect`` is the patched logit diff minus the corrupted one, in logits.
    ``normalized_effect`` divides that by the clean-minus-corrupted gap, so 1.0
    means the head alone restores the clean behaviour.
    """
    clean_tokens = encode_prompt(model, clean.question).tokens
    corrupt_tokens = encode_prompt(model, corrupt.question).tokens
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError(
            f"{clean.id} and {corrupt.id} tokenize to different lengths "
            f"({clean_tokens.shape[1]} vs {corrupt_tokens.shape[1]}); patching needs aligned positions"
        )
    pos_id, neg_id = answer_ids

    def logit_diff(logits: torch.Tensor) -> torch.Tensor:
        return (logits[0, -1, pos_id] - logits[0, -1, neg_id]).float()

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        clean_ld = logit_diff(clean_logits).item()
        corrupt_ld = logit_diff(model(corrupt_tokens)).item()
        patched = patching.get_act_patch_attn_head_out_all_pos(
            model, corrupt_tokens, clean_cache, logit_diff
        )

    gap = clean_ld - corrupt_ld
    differing = (clean_tokens != corrupt_tokens)[0].nonzero().flatten().tolist()
    rows = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            value = patched[layer, head].item()
            effect = value - corrupt_ld
            rows.append(
                {
                    "run_id": run_id,
                    "clean_prompt_id": clean.id,
                    "corrupt_prompt_id": corrupt.id,
                    "differing_positions": json.dumps(differing),
                    "layer": layer,
                    "head": head,
                    "head_label": f"L{layer}H{head}",
                    "clean_logit_diff": clean_ld,
                    "corrupt_logit_diff": corrupt_ld,
                    "patched_logit_diff": value,
                    "effect": effect,
                    "normalized_effect": effect / gap if abs(gap) >= MIN_PATCHING_GAP else math.nan,
                }
            )
    return pd.DataFrame(rows)


def run_patching(
    model: TransformerBridge,
    pairs: Iterable[tuple[str, str]] = PATCHING_PAIRS,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Head patching for every aligned pair. Unaligned pairs are skipped with a warning."""
    run_id = run_id or new_run_id()
    answer_ids = answer_token_ids(model)
    frames = []
    for clean_id, corrupt_id in pairs:
        clean, corrupt = prompt_by_id(clean_id), prompt_by_id(corrupt_id)
        if not tokens_aligned(model, clean, corrupt):
            warnings.warn(
                f"Skipping patching pair {clean_id} -> {corrupt_id}: different token lengths",
                stacklevel=2,
            )
            continue
        frames.append(head_patching(model, clean, corrupt, answer_ids, run_id))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- output ----------------------------------------------------------------


def validate_results(
    prompt_df: pd.DataFrame,
    head_df: pd.DataFrame,
    n_prompts: int,
    n_layers: int,
    n_heads: int,
) -> None:
    """Check the shape and internal consistency of the results before anything is saved."""
    if len(prompt_df) != n_prompts:
        raise ValueError(f"Expected {n_prompts} prompt rows, got {len(prompt_df)}")
    expected_heads = n_prompts * n_layers * n_heads
    if len(head_df) != expected_heads:
        raise ValueError(f"Expected {expected_heads} head rows, got {len(head_df)}")
    if prompt_df["logit_diff"].isna().any():
        raise ValueError("NaN logit difference in the results")
    worst = prompt_df["dla_reconstruction_error"].abs().max()
    if worst > RECONSTRUCTION_TOLERANCE:
        warnings.warn(
            f"Direct logit attribution does not sum to the logit diff (worst error {worst:.4g}); "
            "don't trust the per-component numbers until this is explained "
            "(bfloat16 alone can cause it)",
            stacklevel=2,
        )


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a CSV. Missing values are written as the literal ``NaN``, not left blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, na_rep="NaN", encoding="utf-8")
    return path


def save_results(
    prompt_df: pd.DataFrame,
    head_df: pd.DataFrame,
    patch_df: pd.DataFrame | None,
    output_dir: Path,
    run_id: str,
) -> dict[str, Path]:
    paths = {
        "prompts": save_csv(prompt_df, output_dir / f"{run_id}_prompts.csv"),
        "heads": save_csv(head_df, output_dir / f"{run_id}_heads.csv"),
    }
    if patch_df is not None and not patch_df.empty:
        paths["patching"] = save_csv(patch_df, output_dir / f"{run_id}_patching.csv")
    return paths


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=f"Defaults come from {SETTINGS_FILE}; the flags override them for one run.",
    )
    parser.add_argument(
        "--stage",
        default=DEFAULT_STAGE,
        choices=[s.stage for s in chat_stages()],
        help=f"training stage of {STACK} (model ids come from config/model_stacks.yaml)",
    )
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=sorted(DTYPES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_GENERATION.max_new_tokens)
    parser.add_argument(
        "--sample",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GENERATION.do_sample,
        help="sample instead of greedy decoding",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_GENERATION.temperature)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-patching", action="store_true")
    args = parser.parse_args(argv)

    stage = resolve_stage(args.stage)
    settings = replace(
        DEFAULT_GENERATION,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
    )
    run_id = new_run_id()
    print(
        f"run {run_id}: {STACK}/{stage.stage} = {stage.model} dtype={args.dtype} "
        f"seed={args.seed} {settings} -> {args.output_dir}"
    )

    set_seed(args.seed)
    tokenizer = load_tokenizer(stage.model)
    model = load_model(tokenizer, stage.model, args.device, args.dtype)
    prompt_df, head_df = run_analysis(model, PROMPTS, settings, args.seed, run_id)
    patch_df = None if args.skip_patching else run_patching(model, PATCHING_PAIRS, run_id)

    paths = save_results(prompt_df, head_df, patch_df, args.output_dir, run_id)
    for kind, path in paths.items():
        print(f"wrote {kind}: {path}")
    return paths


if __name__ == "__main__":
    main()
