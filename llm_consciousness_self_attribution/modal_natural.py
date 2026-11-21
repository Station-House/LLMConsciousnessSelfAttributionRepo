"""Modal runner for notion-just-saying-whats-most-natural (auditor-driven natural persona).

One A100-80GB per Olmo variant (SFT, DPO, Instruct), launched in parallel. Each GPU
container serves TWO vLLM servers: the Olmo target and the Gemma 4 12B auditor. A
separate A100-80GB judge function scores the saved logs afterwards (Qwen3.5-9B, the
repo's ``modal_app.OPEN_JUDGE``). A CPU-only ``prefetch`` function fills the
Hugging Face cache volume once, so no GPU container ever pays for a download.

Everything here is cost-capped by a per-function ``timeout``; ``--dry-run`` prints the
plan and the worst-case price from those timeouts and spawns nothing. See
``elicitation_submissions/notion-just-saying-whats-most-natural/RUNBOOK.md`` for the
exact commands.

Windows / Git Bash: set ``MSYS_NO_PATHCONV=1`` on every ``modal`` command, or Git Bash
rewrites ``/eval-logs/...`` into ``C:/Program Files/Git/eval-logs/...``. ``run_dir``
is also refused inside the container unless it is under the volume mount.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import modal

from . import config
from .modal_app import APP_NAME, EVAL_LOGS_MOUNT, OPEN_JUDGE, eval_logs_vol, hf_cache_vol

# Own App, not modal_app.app: `modal run` registers and builds every function on the app
# it loads, and modal_app.app already has an entrypoint called `main` (a duplicate name
# raises) plus a function image with unpinned inspect-ai. The volumes are shared by name.
app = modal.App(f"{APP_NAME}-natural")

# --------------------------------------------------------------------------- prices
# https://modal.com/pricing, fetched 2026-11-11. Every estimate below cites these.
PRICES_DATE = "2026-11-11"
A100_80GB_PER_S = 0.000694  # "Nvidia A100, 80 GB: $0.000694 / sec"
CPU_PER_CORE_S = 0.0000131  # per physical core (= 2 vCPU)
MEM_PER_GIB_S = 0.00000222

# --------------------------------------------------------------------------- models
# Olmo targets: ids come from config/model_stacks.yaml (repo source). Revisions: DPO
# and Instruct are the ones recorded in
# elicitation_submissions/olmo3_identity_spoofing_dpo_instruct_pilots/README.md:10;
# SFT's was never recorded, so this is the HF `main` sha read on 2026-11-11.
OLMO_STACK = "olmo_7b_instruct_stack"
VARIANTS = ("sft", "dpo", "instruct")
AUDITOR_MODEL = "vllm/google/gemma-4-12B-it"  # acceptance criteria, 15 Oct
JUDGE_MODEL = OPEN_JUDGE  # "vllm/Qwen/Qwen3.5-9B" (modal_app.py), acceptance criteria
REVISIONS: dict[str, str] = {
    "allenai/Olmo-3-7B-Instruct-SFT": "e1452fc572d51966ff4aaeb25118b891eb93e549",
    "allenai/Olmo-3-7B-Instruct-DPO": "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1",
    "allenai/Olmo-3-7B-Instruct": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
    "google/gemma-4-12B-it": "43c9d014b231474fa7212dbb48a762566c2b255d",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}
# bf16 weight bytes, from the HF API safetensors totals (params x 2), for the budget.
PARAMS = {
    "allenai/Olmo-3-7B-Instruct": 7_298_011_136,
    "google/gemma-4-12B-it": 11_959_730_224,
    "Qwen/Qwen3.5-9B": 9_653_104_368,
}

# --------------------------------------------------------------------------- token budget
TARGET_MAX_TOKENS = 800  # per target reply
AUDITOR_MAX_TOKENS = 300  # per auditor (simulated user) message
MAX_TURNS = 20  # forks at 1, 5, 10, 20
FULL_FORK_TURNS = (1, 5, 10, 20)
SMOKE_MAX_TURN = 5
SMOKE_FORK_TURNS = (1, 5)
# Allowances (tokens) for text we do not control the length of; stated, not measured.
STARTER_ALLOWANCE = 100  # longest of the 5 starters is ~35 tokens
PROBE_ALLOWANCE = 64  # the 21-word probe is ~25 tokens
AUDITOR_PROMPT_ALLOWANCE = 4000  # auditor instruction + chat-template overhead

# Olmo sees only the conversation: 20 x (300 + 800) + starter + probe = 22,164.
OLMO_CONTEXT_NEED = (
    MAX_TURNS * (AUDITOR_MAX_TOKENS + TARGET_MAX_TOKENS) + STARTER_ALLOWANCE + PROBE_ALLOWANCE
)
# Gemma sees the same conversation plus its own instruction: 26,164.
GEMMA_CONTEXT_NEED = OLMO_CONTEXT_NEED + AUDITOR_PROMPT_ALLOWANCE
OLMO_MAX_MODEL_LEN = 24576  # 24 Ki >= 22,164 (Olmo-3 supports 65,536 via yarn)
GEMMA_MAX_MODEL_LEN = 32768  # 32 Ki >= 26,164 (Gemma supports 262,144)
JUDGE_MAX_MODEL_LEN = 24576  # probe + answer ~2k in, <=16,384 out (D's retry budget)

# --------------------------------------------------------------------------- GPU split
# A100-80GB = 80 GiB (81,920 MiB). vLLM's gpu_memory_utilization is a fraction of that.
A100_GIB = 80.0
GEMMA_UTIL = 0.45  # 36 GiB: 22.3 weights + ~13.7 for activations/graphs/KV (worst-case 1 seq = 9.6)
OLMO_UTIL = 0.40
JUDGE_UTIL = 0.85
CLIENT_TIMEOUT = 900.0  # pilot fix: httpcore/anyio Timeout crash on slow requests
GPU_MONITOR_INTERVAL_S = 15
COMMIT_INTERVAL_S = 60

# --------------------------------------------------------------------------- timeouts
# Chosen against the estimates in memory_budget()/RUNBOOK (not measured; the smoke run
# measures them). Cost caps are timeout x A100_80GB_PER_S.
HF_CACHE_MOUNT = "/root/.cache/huggingface"
PREFETCH_TIMEOUT_S = 30 * 60  # ~50 GB over a datacenter link; CPU only
PREFETCH_CPU, PREFETCH_MEM_GIB = 4, 8
SMOKE_TIMEOUT_S = 30 * 60  # 1 starter x 5 turns + ~10 min cold start of two servers
GENERATE_TIMEOUT_S = 60 * 60  # 5 starters x 20 turns in parallel + cold start
JUDGE_TIMEOUT_S = 45 * 60  # ~60 probes x <=2 attempts, thinking judge, + cold start
EXPECTED_MIN = {"prefetch": 15, "smoke": 12, "generate": 30, "judge": 15}

RUN_ROOT = "natural_auditor"  # under the eval-logs volume
STARTERS_FILE = Path(__file__).resolve().parent / "config" / "natural_auditor_starters.yaml"

# --------------------------------------------------------------------------- image
# Pins are the ones proven in the fork pilot (wt-pilot modal_app.py, commits 21deb7e,
# ab37aca, 4b83b5f). modal_app.py's own image leaves inspect-ai unpinned, so this module
# defines its own rather than editing that file.
def _make_image(vllm_pin: str, transformers_pin: str, extra_commands: tuple[str, ...] = ()) -> modal.Image:
    base = (
        modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        # flashinfer JIT-compiles at vLLM startup and needs nvcc (hence the -devel base);
        # procps supplies pkill for vLLM teardown.
        .apt_install("procps")
        .uv_pip_install(
            # inspect-ai/inspect-petri: the pair recorded in the committed logs of every
            # previous method (Berg, PETRI, self-play).
            "inspect-ai==0.3.249",
            "inspect-petri==3.0.11",
            # inspect-ai 0.3.249 imports acp.helpers, which agent-client-protocol 1.0.0
            # (2026-11-09) removed.
            "agent-client-protocol==0.12.1",
            vllm_pin,
            transformers_pin,
            # openai 3.x fails the first request with "float + Timeout" inside httpcore;
            # 2.54.0 is the last 2.x.
            "openai==2.54.0",
            "pyyaml",
        )
    )
    if extra_commands:  # build steps must come before add_local_* (Modal refuses otherwise)
        base = base.run_commands(*extra_commands)
    return base.add_local_python_source("llm_consciousness_self_attribution")


# Generation image. gemma-4-12B-it's config.json declares architecture
# Gemma4UnifiedForConditionalGeneration (model_type gemma4_unified). vLLM's model
# registry has no entry for it in v0.21.0 or v0.22.x; v0.23.0 (2026-06-15) is the first
# stable release that registers it, and its gemma4_unified.py imports
# transformers.models.gemma4_unified, which exists only in transformers 5.x (present in
# v5.12.1, 2026-06-15; vLLM 0.23.0 allows transformers >=4.56, excluding 5.0-5.5.0).
#
# The Olmo targets must NOT run on that stack. Run natural-20261111 served Olmo from it
# (model_impl=transformers) and every target degenerated into multilingual noise once
# the context passed roughly 4k tokens -- Olmo 3 has 24 sliding-window layers with
# window 4096 -- while the fork pilot's Olmo on vllm 0.21.0 / transformers 4.57.6 stayed
# coherent past 7k. So the image carries a second environment, /opt/olmo-vllm, with
# that proven pair, and Olmo is served from it as a separate process (OLMO_VLLM_BIN)
# that Inspect reaches by base_url. Gemma stays on the main environment.
OLMO_VENV = "/opt/olmo-vllm"
OLMO_VLLM_BIN = f"{OLMO_VENV}/bin/vllm"
image = _make_image("vllm==0.23.0", "transformers==5.12.1", (
    f"python -m venv {OLMO_VENV}",
    f"{OLMO_VENV}/bin/pip install --no-cache-dir vllm==0.21.0 transformers==4.57.6",
))

# Judge image: Qwen3.5-9B is proven on exactly this stack (fork pilot, 864 judgments).
judge_image = _make_image("vllm==0.21.0", "transformers==4.57.6")  # the run resolved 4.57.6


# =========================================================================== pure helpers
def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_volume_path(run_dir: str) -> str:
    """Refuse a run directory outside ``/eval-logs/natural_auditor/``.

    Git Bash rewrites "/eval-logs/..." to "C:/Program Files/Git/eval-logs/..." unless
    MSYS_NO_PATHCONV=1 is set; that path is ephemeral container disk and a whole
    generation run was once lost to it (wt-pilot modal_fork_pilot._require_volume_path).
    """
    prefix = f"{EVAL_LOGS_MOUNT}/{RUN_ROOT}/"
    norm = posixpath.normpath(run_dir)
    if ".." in run_dir.split("/") or not (norm + "/").startswith(prefix) or norm + "/" == prefix:
        raise ValueError(
            f"run_dir must be under {prefix}<run_id>/..., got {run_dir!r}. On Windows set "
            "MSYS_NO_PATHCONV=1 so Git Bash does not rewrite the path."
        )
    return norm


def load_starter_ids() -> list[str]:
    import yaml

    data = yaml.safe_load(STARTERS_FILE.read_text(encoding="utf-8"))
    return [s["id"] for s in data["starters"]]


def olmo_model_id(variant: str) -> str:
    stage = next((s for s in config.load_stack(OLMO_STACK) if s.stage == variant), None)
    if stage is None or variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return stage.model


def hf_repo(model_id: str) -> str:
    return model_id.removeprefix("vllm/")


def revision_args(model_id: str) -> dict[str, str]:
    """vLLM ``--revision`` / ``--tokenizer-revision`` (Inspect turns unknown get_model
    kwargs into ``vllm serve`` flags, as in run.revision_server_args on the pilot)."""
    rev = REVISIONS[hf_repo(model_id)]
    return {"revision": rev, "tokenizer_revision": rev}


def memory_budget() -> dict[str, Any]:
    """Weights + KV arithmetic for the generation GPU, from HF config/safetensors facts."""
    gib = 2**30
    olmo_w = PARAMS["allenai/Olmo-3-7B-Instruct"] * 2 / gib
    gemma_w = PARAMS["google/gemma-4-12B-it"] * 2 / gib
    # Olmo-3-7B: 32 layers, 32 KV heads (no GQA), head_dim 128; 8 full-attention layers
    # and 24 sliding (window 4096). Worst case assumes vLLM's transformers backend keeps
    # every layer's full KV: 2 x 32 x 128 x 2 B x 32 layers = 512 KiB/token.
    olmo_kv_tok = 2 * 32 * 128 * 2 * 32
    # Gemma 4 12B: 48 layers = 40 sliding (window 1024, 8 KV heads, head_dim 256) + 8
    # full (1 global KV head, global_head_dim 512). Worst case = all 48 layers like the
    # sliding ones: 2 x 8 x 256 x 2 B x 48 = 384 KiB/token (real: far less).
    gemma_kv_tok = 2 * 8 * 256 * 2 * 48
    olmo_budget = OLMO_UTIL * A100_GIB
    gemma_budget = GEMMA_UTIL * A100_GIB
    return {
        "gpu_gib": A100_GIB,
        "olmo": {
            "weights_gib": round(olmo_w, 1), "util": OLMO_UTIL, "budget_gib": round(olmo_budget, 1),
            "non_weight_gib": round(olmo_budget - olmo_w, 1),
            "kv_worst_bytes_per_token": olmo_kv_tok,
            "kv_worst_gib_one_full_seq": round(OLMO_CONTEXT_NEED * olmo_kv_tok / gib, 1),
            "max_model_len": OLMO_MAX_MODEL_LEN, "context_need": OLMO_CONTEXT_NEED,
        },
        "gemma": {
            "weights_gib": round(gemma_w, 1), "util": GEMMA_UTIL, "budget_gib": round(gemma_budget, 1),
            "non_weight_gib": round(gemma_budget - gemma_w, 1),
            "kv_worst_bytes_per_token": gemma_kv_tok,
            "kv_worst_gib_one_full_seq": round(GEMMA_CONTEXT_NEED * gemma_kv_tok / gib, 1),
            "max_model_len": GEMMA_MAX_MODEL_LEN, "context_need": GEMMA_CONTEXT_NEED,
        },
        "total_util": round(OLMO_UTIL + GEMMA_UTIL, 2),
        "headroom_gib": round(A100_GIB * (1 - OLMO_UTIL - GEMMA_UTIL), 1),
    }


def gpu_cost(seconds: float) -> float:
    return seconds * A100_80GB_PER_S


def cpu_cost(seconds: float) -> float:
    return seconds * (PREFETCH_CPU * CPU_PER_CORE_S + PREFETCH_MEM_GIB * MEM_PER_GIB_S)


def build_plan(
    mode: str, run_id: str, seed: int, scorer: str | None = None,
    starters: list[str] | None = None, variants: list[str] | None = None,
) -> dict[str, Any]:
    """Everything a run would do, as data. Pure: touches neither Modal nor the GPU."""
    if mode not in ("smoke", "full", "prefetch", "judge"):
        raise ValueError(f"mode must be smoke|full|prefetch|judge, got {mode!r}")
    all_ids = load_starter_ids()
    models = {
        "auditor": {"id": AUDITOR_MODEL, "revision": REVISIONS["google/gemma-4-12B-it"]},
        "judge": {"id": JUDGE_MODEL, "revision": REVISIONS["Qwen/Qwen3.5-9B"]},
    }
    steps: list[dict[str, Any]] = []
    if mode == "prefetch":
        repos = [olmo_model_id(v) for v in VARIANTS] + ["google/gemma-4-12B-it", "Qwen/Qwen3.5-9B"]
        steps.append({
            "function": "prefetch", "hardware": f"CPU {PREFETCH_CPU} cores, {PREFETCH_MEM_GIB} GiB",
            "timeout_s": PREFETCH_TIMEOUT_S, "worst_usd": cpu_cost(PREFETCH_TIMEOUT_S),
            "expected_usd": cpu_cost(EXPECTED_MIN["prefetch"] * 60),
            "models": {r: REVISIONS[r] for r in repos},
        })
    elif mode in ("smoke", "full"):
        if mode == "smoke":
            variants, starters = ["instruct"], (starters or all_ids)[:1]
            max_turn, forks, fn, timeout = SMOKE_MAX_TURN, SMOKE_FORK_TURNS, "generate_smoke", SMOKE_TIMEOUT_S
            exp = EXPECTED_MIN["smoke"]
        else:
            variants, starters = variants or list(VARIANTS), starters or all_ids
            max_turn, forks, fn, timeout = MAX_TURNS, FULL_FORK_TURNS, "generate", GENERATE_TIMEOUT_S
            exp = EXPECTED_MIN["generate"]
        unknown = [s for s in starters if s not in all_ids]
        if unknown:
            raise ValueError(f"unknown starter ids {unknown}; known: {all_ids}")
        for v in variants:
            mid = olmo_model_id(v)
            steps.append({
                "function": fn, "variant": v, "hardware": "1x A100-80GB (2 vLLM servers)",
                "run_dir": f"{EVAL_LOGS_MOUNT}/{RUN_ROOT}/{run_id}/{v}", "max_turn": max_turn,
                "fork_turns": list(forks), "starters": starters, "timeout_s": timeout,
                "worst_usd": gpu_cost(timeout), "expected_usd": gpu_cost(exp * 60),
                "target": {"id": f"vllm/{mid}", "revision": REVISIONS[mid]},
            })
        models["memory_budget"] = memory_budget()
    else:  # judge
        if scorer not in ("berg", "petri"):
            raise ValueError("--scorer is required and must be 'berg' or 'petri' (no default)")
        steps.append({
            "function": "judge", "scorer": scorer, "hardware": "1x A100-80GB",
            "run_dir": f"{EVAL_LOGS_MOUNT}/{RUN_ROOT}/{run_id}", "timeout_s": JUDGE_TIMEOUT_S,
            "worst_usd": gpu_cost(JUDGE_TIMEOUT_S), "expected_usd": gpu_cost(EXPECTED_MIN["judge"] * 60),
        })
    return {
        "mode": mode, "run_id": run_id, "base_seed": seed, "prices": {
            "source": "https://modal.com/pricing", "date": PRICES_DATE,
            "a100_80gb_per_s": A100_80GB_PER_S, "cpu_per_core_s": CPU_PER_CORE_S, "mem_per_gib_s": MEM_PER_GIB_S},
        "models": models, "steps": steps,
        "worst_case_usd": round(sum(s["worst_usd"] for s in steps), 2),
        "expected_usd": round(sum(s["expected_usd"] for s in steps), 2),
    }


def format_plan(plan: dict[str, Any]) -> str:
    p = plan["prices"]
    lines = [
        f"PLAN mode={plan['mode']} run_id={plan['run_id']} base_seed={plan['base_seed']}",
        f"prices ({p['source']}, {p['date']}): A100-80GB ${p['a100_80gb_per_s']}/s, "
        f"CPU ${p['cpu_per_core_s']}/core/s, mem ${p['mem_per_gib_s']}/GiB/s",
    ]
    for step in plan["steps"]:
        lines.append(f"- {step['function']}" + (f"[{step['variant']}]" if "variant" in step else ""))
        for k, v in step.items():
            if k in ("function", "variant", "worst_usd", "expected_usd"):
                continue
            lines.append(f"    {k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
        lines.append(f"    worst-case from timeout: ${step['worst_usd']:.2f}   expected: ${step['expected_usd']:.2f}")
    if plan["mode"] in ("smoke", "full"):
        lines.append(f"auditor: {json.dumps(plan['models']['auditor'])}")
    if plan["models"].get("judge") and plan["mode"] == "judge":
        lines.append(f"judge: {json.dumps(plan['models']['judge'])}")
    if "memory_budget" in plan["models"]:
        lines.append(f"memory budget: {json.dumps(plan['models']['memory_budget'])}")
    lines.append(f"TOTAL worst-case ${plan['worst_case_usd']:.2f}   expected ${plan['expected_usd']:.2f}")
    return "\n".join(lines)


# =========================================================================== container helpers
def _gpu_monitor(path: Path) -> subprocess.Popen:
    """nvidia-smi every 15 s: utilisation and memory, to a CSV on the volume."""
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total",
         "--format=csv", "-l", str(GPU_MONITOR_INTERVAL_S)],
        stdout=path.open("w"), stderr=subprocess.STDOUT,
    )


def _versions() -> dict:
    import importlib.metadata as md

    out = {}
    for pkg in ("inspect_ai", "inspect_petri", "vllm", "transformers", "torch", "openai"):
        try:
            out[pkg] = md.version(pkg.replace("_", "-"))
        except Exception:  # noqa: BLE001
            out[pkg] = None
    return out


def _log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")


# =========================================================================== functions
_GPU_FN = dict(
    gpu="A100-80GB",  # acceptance criteria: "the compute GPU should be A100-80GB"
    image=image,
    volumes={EVAL_LOGS_MOUNT: eval_logs_vol, HF_CACHE_MOUNT: hf_cache_vol},
    # GPU containers must never download: with HF_HUB_OFFLINE set, "no HTTP calls will
    # be made to the Hugging Face Hub ... If no cache file is detected, an error is
    # raised" (huggingface_hub environment-variable docs). Set before any HF import.
    secrets=[modal.Secret.from_dict({"HF_HUB_OFFLINE": "1"})],
    single_use_containers=True,
)


@app.function(
    image=image, volumes={HF_CACHE_MOUNT: hf_cache_vol, EVAL_LOGS_MOUNT: eval_logs_vol},
    cpu=PREFETCH_CPU, memory=PREFETCH_MEM_GIB * 1024, timeout=PREFETCH_TIMEOUT_S,
)
def prefetch() -> dict:
    """CPU only. Download all five models into the hf-cache volume, once, at pinned
    revisions, and write a manifest (pinned sha, current HF `main` sha, size)."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    manifest: dict[str, Any] = {"function": "prefetch", "start_utc": now_utc(), "models": {}}
    repos = [olmo_model_id(v) for v in VARIANTS] + ["google/gemma-4-12B-it", "Qwen/Qwen3.5-9B"]
    for repo in repos:
        pinned = REVISIONS[repo]
        head = api.model_info(repo).sha  # current main, recorded for the audit trail
        path = snapshot_download(repo, revision=pinned)
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        manifest["models"][repo] = {
            "pinned_revision": pinned, "hf_main_sha_at_prefetch": head,
            "pin_is_current_main": head == pinned, "snapshot": path, "bytes": size,
        }
        _log(f"{repo}@{pinned} ready ({size / 2**30:.1f} GiB); main is {head}")
        hf_cache_vol.commit()
    manifest["end_utc"] = now_utc()
    out = Path(EVAL_LOGS_MOUNT) / RUN_ROOT
    out.mkdir(parents=True, exist_ok=True)
    _write_manifest(out / "prefetch_manifest.json", manifest)
    eval_logs_vol.commit()
    hf_cache_vol.commit()
    return manifest


def input_hashes() -> dict[str, str]:
    """sha256 of the three text inputs: auditor instruction, starters file, probe."""
    from . import natural_auditor, starters as starters_mod

    return {
        "auditor_instruction": sha256_text(natural_auditor.AUDITOR_INSTRUCTION),
        "starters_file": sha256_file(STARTERS_FILE),
        "probe_text": sha256_text(starters_mod.probe()),
    }


async def _run_generation(natural_auditor, target, auditor, out: Path, seed: int, max_turn: int,
                          fork_turns: tuple[int, ...], starters: list[str]) -> dict:
    """Build B's Task and run it with Inspect. The target is the eval ``model``; the
    auditor is bound as model role ``auditor`` (natural_auditor resolves it with
    ``get_model(role="auditor", required=True, ...)``). One sample per starter;
    ``checkpoint_dir`` makes a rerun with the same run_dir resume with the same seeds."""
    from inspect_ai import eval_async

    task = natural_auditor.natural_auditor_task(
        base_seed=seed, turns=max_turn, probe_turns=fork_turns,
        checkpoint_dir=str(out / "checkpoints"),
        target_max_tokens=TARGET_MAX_TOKENS, auditor_max_tokens=AUDITOR_MAX_TOKENS,
    )
    all_ids = load_starter_ids()
    logs = await eval_async(
        task, model=target, model_roles={"auditor": auditor}, log_dir=str(out), seed=seed,
        temperature=float(config.run_defaults()["temperature"]),
        sample_id=None if sorted(starters) == sorted(all_ids) else starters,
        # a failed sample is kept with its probes so far, never dropped or aborted on
        fail_on_error=False,
    )
    return {"logs": [{"location": log.location, "status": log.status,
                      "samples": len(log.samples or [])} for log in logs]}


OLMO_PORT = 8011
OLMO_READY_TIMEOUT_S = 15 * 60


def olmo_server_cmd(olmo_id: str) -> list[str]:
    """``vllm serve`` for an Olmo target from the /opt/olmo-vllm environment, with the
    same server args the Inspect provider would have passed (model_stacks.yaml's
    olmo_target_model_args, the memory split and the revision pins)."""
    args = {**config.olmo_target_model_args(), **revision_args(f"vllm/{olmo_id}"),
            "gpu_memory_utilization": OLMO_UTIL, "max_model_len": OLMO_MAX_MODEL_LEN}
    cmd = [OLMO_VLLM_BIN, "serve", olmo_id, "--host", "127.0.0.1", "--port", str(OLMO_PORT)]
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            cmd.append(flag)
        elif value is not False:
            cmd += [flag, str(value)]
    return cmd


def _start_olmo_server(olmo_id: str, log_path: Path) -> subprocess.Popen:
    """Start the Olmo server and block until /health answers (or raise)."""
    import urllib.request

    proc = subprocess.Popen(olmo_server_cmd(olmo_id), stdout=log_path.open("w"),
                            stderr=subprocess.STDOUT)
    deadline = time.time() + OLMO_READY_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Olmo vLLM server exited with {proc.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{OLMO_PORT}/health", timeout=5):
                return proc
        except OSError:
            time.sleep(5)
    proc.terminate()
    raise RuntimeError(f"Olmo vLLM server not healthy after {OLMO_READY_TIMEOUT_S} s")


def _olmo_env_versions() -> dict[str, str | None]:
    out = subprocess.run(
        [f"{OLMO_VENV}/bin/python", "-c",
         "import importlib.metadata as m, json; print(json.dumps({p: m.version(p) for p in "
         "('vllm', 'transformers', 'torch')}))"],
        capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else {"error": out.stderr[-500:]}


def _generate_impl(variant: str, run_dir: str, max_turn: int, starters: list[str], seed: int,
                   fork_turns: tuple[int, ...], timeout_s: int) -> dict:
    import asyncio
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    run_dir = require_volume_path(run_dir)
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

    from . import natural_auditor

    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    olmo_id = olmo_model_id(variant)
    manifest: dict[str, Any] = {
        "function": "generate", "variant": variant, "run_dir": run_dir,
        "container_start_utc": now_utc(), "versions": _versions(), "base_seed": seed,
        "timeout_s": timeout_s, "max_turn": max_turn, "fork_turns": list(fork_turns), "starters": starters,
        "token_caps": {"target_per_reply": TARGET_MAX_TOKENS, "auditor_per_message": AUDITOR_MAX_TOKENS},
        "models": {
            "target": {"id": f"vllm/{olmo_id}", "revision": REVISIONS[olmo_id]},
            "auditor": {"id": AUDITOR_MODEL, "revision": REVISIONS["google/gemma-4-12B-it"]},
        },
        "gpu_split": memory_budget(),
        "sha256": input_hashes(),
    }
    monitor = _gpu_monitor(out / "gpu_generate.csv")
    t_start = time.time()
    try:
        # Auditor first: it is the larger server, so it sees the emptiest GPU.
        auditor = get_model(
            AUDITOR_MODEL,
            # Gemma card: T=1.0, top_p=0.95, top_k=64. top_k is NOT set here: the
            # method passes its own per-call extra_body (enable_thinking=False), which
            # replaces any model-level extra_body, so a top_k here would be dead.
            config=GenerateConfig(temperature=1.0, top_p=0.95, max_connections=16),
            gpu_memory_utilization=GEMMA_UTIL, max_model_len=GEMMA_MAX_MODEL_LEN,
            client_timeout=CLIENT_TIMEOUT, **revision_args(AUDITOR_MODEL),
        )
        olmo_server = _start_olmo_server(olmo_id, out / "olmo_server.log")
        manifest["models"]["target"]["server"] = {
            "env": OLMO_VENV, "versions": _olmo_env_versions(), "cmd": olmo_server.args}
        target = get_model(
            f"vllm/{olmo_id}",
            base_url=f"http://127.0.0.1:{OLMO_PORT}/v1",
            config=GenerateConfig(temperature=1.0, max_connections=16),
            client_timeout=CLIENT_TIMEOUT,
        )

        async def main() -> None:
            for name, model in (("auditor", auditor), ("target", target)):
                t0 = time.time()
                await model.generate([ChatMessageUser(content="Hello")],
                                     config=GenerateConfig(max_tokens=8, seed=seed))
                manifest[f"{name}_ready_s"] = round(time.time() - t0, 1)
                _log(f"{name} server ready ({manifest[f'{name}_ready_s']} s after previous)")
            manifest["ready_at_s"] = round(time.time() - t_start, 1)

            async def committer() -> None:
                while True:
                    await asyncio.sleep(COMMIT_INTERVAL_S)
                    await eval_logs_vol.commit.aio()

            background = asyncio.create_task(committer())
            t1 = time.time()
            try:
                manifest["result"] = await _run_generation(
                    natural_auditor, target, auditor, out, seed, max_turn, fork_turns, starters)
            finally:
                background.cancel()
                manifest["generation_s"] = round(time.time() - t1, 1)

        asyncio.run(main())
    finally:
        monitor.terminate()
        if "olmo_server" in locals():
            olmo_server.terminate()
        total = time.time() - t_start
        manifest["container_end_utc"] = now_utc()
        manifest["gpu_seconds"] = {"total": round(total, 1), "startup": manifest.get("ready_at_s"),
                                   "generation": manifest.get("generation_s")}
        manifest["estimated_usd"] = {"total": round(gpu_cost(total), 3),
                                     "price_per_s": A100_80GB_PER_S, "price_date": PRICES_DATE}
        _write_manifest(out / "manifest_generate.json", manifest)
        eval_logs_vol.commit()
    return manifest


@app.function(timeout=GENERATE_TIMEOUT_S, **_GPU_FN)
def generate(variant: str, run_dir: str, max_turn: int, starters: list[str], seed: int) -> dict:
    """One Olmo variant, all starters, forks at 1/5/10/20. Timeout 60 min = at most $2.50."""
    return _generate_impl(variant, run_dir, max_turn, starters, seed, FULL_FORK_TURNS, GENERATE_TIMEOUT_S)


@app.function(timeout=SMOKE_TIMEOUT_S, **_GPU_FN)
def generate_smoke(variant: str, run_dir: str, max_turn: int, starters: list[str], seed: int) -> dict:
    """Same code as ``generate`` with a 30 min cap (at most $1.25), forks at 1 and 5."""
    return _generate_impl(variant, run_dir, max_turn, starters, seed, SMOKE_FORK_TURNS, SMOKE_TIMEOUT_S)


@app.function(timeout=JUDGE_TIMEOUT_S, **{**_GPU_FN, "image": judge_image})
def judge(run_dir: str, scorer: str, seed: int) -> dict:
    """Score every unscored .eval under ``run_dir/<variant>/`` with D's ``score_log``.
    Qwen3.5-9B (bf16 ~18 GiB) leaves ~50 GiB of KV on an A100-80GB. Timeout 45 min =
    at most $1.87. ``scorer`` has no default: berg | petri."""
    import asyncio
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if scorer not in ("berg", "petri"):
        raise ValueError(f"scorer must be 'berg' or 'petri', got {scorer!r}")
    root = Path(require_volume_path(run_dir))
    paths = sorted(p for p in root.glob("*/*.eval") if not p.stem.endswith("-scored"))
    if not paths:
        raise SystemExit(f"No unscored .eval logs under {root}/<variant>/")
    manifest: dict[str, Any] = {
        "function": "judge", "run_dir": str(root), "scorer": scorer, "base_seed": seed,
        "container_start_utc": now_utc(), "versions": _versions(), "timeout_s": JUDGE_TIMEOUT_S,
        "judge": {"id": JUDGE_MODEL, "revision": REVISIONS["Qwen/Qwen3.5-9B"]}, "logs": [],
    }
    monitor = _gpu_monitor(root / "gpu_judge.csv")
    t_start = time.time()

    def work() -> None:
        # Everything that touches the judge Model runs in this one worker thread:
        # Inspect's sync score() owns an event loop, and a Model's HTTP client must not
        # be shared across loops.
        from inspect_ai.model import GenerateConfig, get_model

        from . import natural_scoring

        judge_model = get_model(
            JUDGE_MODEL,
            config=natural_scoring.judge_config(seed),  # = modal_app.score_logs sampling
            reasoning_parser="qwen3", gpu_memory_utilization=JUDGE_UTIL,
            max_model_len=JUDGE_MAX_MODEL_LEN, client_timeout=CLIENT_TIMEOUT,
            **revision_args(JUDGE_MODEL),
        )
        manifest["judge_ready_s"] = round(time.time() - t_start, 1)
        for path in paths:
            t0 = time.time()
            scored = natural_scoring.score_log(path, scorer, judge_model, seed)
            manifest["logs"].append({"log": str(path), "scored": str(scored), "s": round(time.time() - t0, 1)})
            _log(f"scored {path.name} -> {scored.name}")

    async def main() -> None:
        async def committer() -> None:
            while True:
                await asyncio.sleep(COMMIT_INTERVAL_S)
                await eval_logs_vol.commit.aio()

        background = asyncio.create_task(committer())
        try:
            await asyncio.to_thread(work)
        finally:
            background.cancel()

    try:
        asyncio.run(main())
    finally:
        monitor.terminate()
        total = time.time() - t_start
        manifest["container_end_utc"] = now_utc()
        manifest["gpu_seconds"] = {"total": round(total, 1)}
        manifest["estimated_usd"] = {"total": round(gpu_cost(total), 3), "price_per_s": A100_80GB_PER_S,
                                     "price_date": PRICES_DATE}
        _write_manifest(root / f"manifest_judge_{scorer}.json", manifest)
        eval_logs_vol.commit()
    return manifest


# =========================================================================== entrypoint
def new_run_id(mode: str) -> str:
    return f"natural-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}" + ("-smoke" if mode == "smoke" else "")


def run_entrypoint(
    mode: str, dry_run: bool = False, scorer: str = "", run_id: str = "", seed: int | None = None,
    starters: str = "", variants: str = "", *, fns: dict[str, Any] | None = None,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Validate, print the plan, and (unless ``dry_run``) spawn. All validation happens
    before the first ``.spawn``/``.remote``, so a bad invocation costs nothing."""
    if mode == "judge" and not scorer:
        raise ValueError("--scorer berg|petri is required for --mode judge (there is no default)")
    if scorer and scorer not in ("berg", "petri"):
        raise ValueError(f"--scorer must be 'berg' or 'petri', got {scorer!r}")
    if mode == "judge" and not run_id:
        raise ValueError("--run-id <id of the generation run> is required for --mode judge")
    if seed is None:
        seed = int(config.run_defaults()["seed"])
    run_id = run_id or new_run_id(mode)
    plan = build_plan(
        mode, run_id, seed, scorer or None,
        [s for s in starters.split(",") if s] or None, [v for v in variants.split(",") if v] or None,
    )
    echo(format_plan(plan))
    if dry_run:
        echo("DRY RUN: nothing was spawned.")
        return {"plan": plan, "spawned": False}
    fns = fns or {"prefetch": prefetch, "generate": generate, "generate_smoke": generate_smoke, "judge": judge}
    results: dict[str, Any] = {}
    if mode == "prefetch":
        results["prefetch"] = fns["prefetch"].remote()
    elif mode == "judge":
        results["judge"] = fns["judge"].remote(f"{EVAL_LOGS_MOUNT}/{RUN_ROOT}/{run_id}", scorer, seed)
    else:
        calls = {
            s["variant"]: fns[s["function"]].spawn(s["variant"], s["run_dir"], s["max_turn"], s["starters"], seed)
            for s in plan["steps"]
        }
        echo(f"spawned {len(calls)} containers; gathering (Ctrl-C does not stop them: use --detach).")
        for variant, call in calls.items():
            results[variant] = call.get()
    echo(f"done. run_id={run_id}")
    return {"plan": plan, "spawned": True, "results": results}


@app.local_entrypoint()
def natural(mode: str = "", dry_run: bool = False, scorer: str = "", run_id: str = "",
         seed: int = -1, starters: str = "", variants: str = ""):
    """--mode smoke|full|prefetch|judge  [--dry-run]  [--scorer berg|petri (judge only)]."""
    if not mode:
        raise SystemExit("--mode smoke|full|prefetch|judge is required")
    run_entrypoint(mode, dry_run, scorer, run_id, None if seed < 0 else seed, starters, variants)


if __name__ == "__main__":
    # Local, Modal-free dry run:  python -m llm_consciousness_self_attribution.modal_natural
    #   --mode smoke|full|prefetch|judge --dry-run [--scorer berg|petri --run-id ID]
    # `modal run` starts the app (and builds the image) before any entrypoint code runs,
    # so the plan is printed from here, where nothing can reach Modal.
    import argparse

    parser = argparse.ArgumentParser(description="Print the plan; never touches Modal.")
    parser.add_argument("--mode", required=True, choices=["smoke", "full", "prefetch", "judge"])
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--scorer", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--starters", default="")
    parser.add_argument("--variants", default="")
    a = parser.parse_args()
    try:
        run_entrypoint(a.mode, True, a.scorer, a.run_id, a.seed, a.starters, a.variants)
    except ValueError as err:
        raise SystemExit(f"error: {err}")
