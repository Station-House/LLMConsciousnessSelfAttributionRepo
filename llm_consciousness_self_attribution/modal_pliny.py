"""Modal runner for the Pliny-wrapper task (notion-explore-techniques-from-pliny-the-liberator-on-x).

Pattern copied (not imported) from the natural-auditor runner, ``modal_natural.py`` on
branch ``notion-just-saying-whats-most-natural``, cut down to what a single-turn task
needs: one vLLM server (the Olmo target) per A100-80GB, one container per variant in
parallel, then a separate judge container (Qwen3.5-9B) that grades the saved logs.

* ``--mode prefetch``: CPU only, fills the shared ``huggingface-cache`` volume at the
  pinned revisions (a no-op for files already there). GPU containers run with
  ``HF_HUB_OFFLINE=1`` and never download.
* ``--mode smoke``: Instruct only, 1 epoch of every arm (8 generations).
* ``--mode full``: SFT, DPO, Instruct in parallel, ``pliny.EPOCHS`` epochs of every arm.
* ``--mode judge --run-id <id> --scorer berg[,berg_blinded]``: ``--scorer`` is required
  and has no default.

Every function has a ``timeout`` that caps its cost; ``--dry-run`` prints the plan and
the worst case from those timeouts and spawns nothing. Commands are in
``elicitation_submissions/notion-explore-techniques-from-pliny-the-liberator-on-x/RUNBOOK.md``.

Windows / Git Bash: ``unset VIRTUAL_ENV``, ``PYTHONIOENCODING=utf-8 PYTHONUTF8=1`` and
``MSYS_NO_PATHCONV=1`` on every ``modal`` command (Git Bash otherwise rewrites
``/eval-logs/...``); ``run_dir`` is refused inside the container unless it is under the
volume mount.
"""

from __future__ import annotations

import json
import posixpath
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import modal

from . import config, pliny
from .modal_app import APP_NAME, EVAL_LOGS_MOUNT, OPEN_JUDGE, eval_logs_vol, hf_cache_vol

# Own App: modal_app.app already has an entrypoint called `main` and an image with
# unpinned inspect-ai. The volumes are shared by name.
app = modal.App(f"{APP_NAME}-pliny")

# --------------------------------------------------------------------------- prices
# https://modal.com/pricing, as recorded in modal_natural.py (fetched 2026-11-11).
PRICES_DATE = "2026-11-11"
A100_80GB_PER_S = 0.000694  # "Nvidia A100, 80 GB: $0.000694 / sec"
CPU_PER_CORE_S = 0.0000131
MEM_PER_GIB_S = 0.00000222

# --------------------------------------------------------------------------- models
OLMO_STACK = "olmo_7b_instruct_stack"
VARIANTS = ("sft", "dpo", "instruct")
JUDGE_MODEL = OPEN_JUDGE  # "vllm/Qwen/Qwen3.5-9B", acceptance criteria 15 Oct
# DPO/Instruct: recorded in elicitation_submissions/olmo3_identity_spoofing_dpo_instruct_pilots/
# README.md:10. SFT: never recorded in the repo; HF `main` sha read 2026-11-11 (RECON).
# Qwen: modal_app.OPEN_JUDGE_REVISION on the fork-pilot branch.
REVISIONS: dict[str, str] = {
    "allenai/Olmo-3-7B-Instruct-SFT": "e1452fc572d51966ff4aaeb25118b891eb93e549",
    "allenai/Olmo-3-7B-Instruct-DPO": "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1",
    "allenai/Olmo-3-7B-Instruct": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}
SCORERS = tuple(sorted(("berg", "berg_blinded")))  # = pliny_scoring.SCORER_NAMES keys

# --------------------------------------------------------------------------- sizes
# Longest wrapper ~250 tokens + 1536 reply < 4096, so Olmo stays inside its 4096-token
# sliding window (24 of its 32 layers), the regime where it was proven coherent.
OLMO_MAX_MODEL_LEN = 4096
OLMO_UTIL = 0.85
# Judge: wrapper + probe + 1536-token answer + Berg template (~3k) in, <=16,384 out.
JUDGE_MAX_MODEL_LEN = 24576
JUDGE_UTIL = 0.85
MAX_CONNECTIONS = 64
CLIENT_TIMEOUT = 900.0  # fork-pilot fix for the httpcore/anyio Timeout crash
GPU_MONITOR_INTERVAL_S = 15
COMMIT_INTERVAL_S = 60

# --------------------------------------------------------------------------- timeouts
HF_CACHE_MOUNT = "/root/.cache/huggingface"
PREFETCH_TIMEOUT_S = 30 * 60
PREFETCH_CPU, PREFETCH_MEM_GIB = 4, 8
SMOKE_TIMEOUT_S = 18 * 60  # 8 generations + one cold start
GENERATE_TIMEOUT_S = 25 * 60  # 80 generations + one cold start
JUDGE_TIMEOUT_S = 40 * 60  # 240 answers x up to 2 scorers x <=2 attempts + cold start
SMOKE_EPOCHS = 1
# Not measured; the smoke run's manifest replaces these.
EXPECTED_MIN = {"prefetch": 10, "smoke": 10, "generate": 12, "judge": 25}

RUN_ROOT = "pliny_wrappers"  # under the eval-logs volume
PROMPTS_REMOTE = f"/root/{pliny.PROMPTS_FILE.relative_to(pliny.PROMPTS_FILE.parents[2]).as_posix()}"

# --------------------------------------------------------------------------- image
# One environment for target and judge: vllm 0.21.0 + transformers 4.57.6 is the pair
# that served both Olmo-3-7B-Instruct (coherent past 7k tokens) and Qwen3.5-9B (864
# judgments) in the fork pilot (branch csa-fork-pilot, pilot_runs/fork_pilot_v1 manifests record
# vllm 0.21.0 / transformers 4.57.6). Other pins as in modal_natural.py.
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("procps")
    .uv_pip_install(
        "inspect-ai==0.3.249",
        "inspect-petri==3.0.11",
        "agent-client-protocol==0.12.1",
        "vllm==0.21.0",
        "transformers==4.57.6",
        "openai==2.54.0",
        "pyyaml",
    )
    # Default ignore keeps only *.py; config/*.yaml and the rubric .md are needed too.
    .add_local_python_source("llm_consciousness_self_attribution", ignore=["**/__pycache__/**"])
    .add_local_file(pliny.PROMPTS_FILE, remote_path=PROMPTS_REMOTE)
)


# =========================================================================== pure helpers
def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_volume_path(run_dir: str) -> str:
    """Refuse a run directory outside ``/eval-logs/pliny_wrappers/<run_id>``."""
    prefix = f"{EVAL_LOGS_MOUNT}/{RUN_ROOT}/"
    norm = posixpath.normpath(run_dir)
    if ".." in run_dir.split("/") or not (norm + "/").startswith(prefix) or norm + "/" == prefix:
        raise ValueError(
            f"run_dir must be under {prefix}<run_id>/..., got {run_dir!r}. On Windows set "
            "MSYS_NO_PATHCONV=1 so Git Bash does not rewrite the path."
        )
    return norm


def olmo_model_id(variant: str) -> str:
    stage = next((s for s in config.load_stack(OLMO_STACK) if s.stage == variant), None)
    if stage is None or variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return stage.model


def revision_args(model_id: str) -> dict[str, str]:
    """vLLM ``--revision`` / ``--tokenizer-revision`` (Inspect passes unknown get_model
    kwargs to ``vllm serve`` as flags)."""
    rev = REVISIONS[model_id.removeprefix("vllm/")]
    return {"revision": rev, "tokenizer_revision": rev}


def parse_scorers(scorer: str) -> list[str]:
    names = [s.strip() for s in scorer.split(",") if s.strip()]
    if not names:
        raise ValueError(f"--scorer is required for --mode judge: one or more of {SCORERS} (no default)")
    bad = [s for s in names if s not in SCORERS]
    if bad:
        raise ValueError(f"--scorer must be from {SCORERS}, got {bad}")
    return names


def gpu_cost(seconds: float) -> float:
    return seconds * A100_80GB_PER_S


def cpu_cost(seconds: float) -> float:
    return seconds * (PREFETCH_CPU * CPU_PER_CORE_S + PREFETCH_MEM_GIB * MEM_PER_GIB_S)


def build_plan(mode: str, run_id: str, seed: int, scorers: list[str] | None = None,
               variants: list[str] | None = None) -> dict[str, Any]:
    """Everything a run would do, as data. Pure: touches neither Modal nor the GPU."""
    if mode not in ("smoke", "full", "prefetch", "judge"):
        raise ValueError(f"mode must be smoke|full|prefetch|judge, got {mode!r}")
    prompts = pliny.load_prompts()  # refuses a prompt file that changed after freezing
    arms = [p["id"] for p in prompts["prompts"]]
    steps: list[dict[str, Any]] = []
    if mode == "prefetch":
        repos = [olmo_model_id(v) for v in VARIANTS] + ["Qwen/Qwen3.5-9B"]
        steps.append({
            "function": "prefetch", "hardware": f"CPU {PREFETCH_CPU} cores, {PREFETCH_MEM_GIB} GiB",
            "timeout_s": PREFETCH_TIMEOUT_S, "worst_usd": cpu_cost(PREFETCH_TIMEOUT_S),
            "expected_usd": cpu_cost(EXPECTED_MIN["prefetch"] * 60),
            "models": {r: REVISIONS[r] for r in repos},
        })
    elif mode in ("smoke", "full"):
        if mode == "smoke":
            variants, epochs, fn, timeout, exp = ["instruct"], SMOKE_EPOCHS, "generate_smoke", SMOKE_TIMEOUT_S, "smoke"
        else:
            variants, epochs, fn, timeout, exp = variants or list(VARIANTS), pliny.EPOCHS, "generate", GENERATE_TIMEOUT_S, "generate"
        for v in variants:
            mid = olmo_model_id(v)
            steps.append({
                "function": fn, "variant": v, "hardware": "1x A100-80GB (1 vLLM server)",
                "run_dir": f"{EVAL_LOGS_MOUNT}/{RUN_ROOT}/{run_id}/{v}",
                "target": {"id": f"vllm/{mid}", "revision": REVISIONS[mid]},
                "arms": arms, "epochs": epochs, "samples": len(arms) * epochs,
                "epoch_seeds": [pliny.epoch_seed(seed, e) for e in range(1, epochs + 1)],
                "max_tokens": pliny.TARGET_MAX_TOKENS, "max_model_len": OLMO_MAX_MODEL_LEN,
                "temperature": float(config.run_defaults()["temperature"]),
                "timeout_s": timeout, "worst_usd": gpu_cost(timeout),
                "expected_usd": gpu_cost(EXPECTED_MIN[exp] * 60),
            })
    else:
        if not scorers:
            raise ValueError("--scorer is required for --mode judge (there is no default)")
        steps.append({
            "function": "judge", "scorers": scorers, "hardware": "1x A100-80GB",
            "judge": {"id": JUDGE_MODEL, "revision": REVISIONS["Qwen/Qwen3.5-9B"]},
            "run_dir": f"{EVAL_LOGS_MOUNT}/{RUN_ROOT}/{run_id}", "timeout_s": JUDGE_TIMEOUT_S,
            "worst_usd": gpu_cost(JUDGE_TIMEOUT_S), "expected_usd": gpu_cost(EXPECTED_MIN["judge"] * 60),
        })
    return {
        "mode": mode, "run_id": run_id, "base_seed": seed,
        "prompts": {"file": str(pliny.PROMPTS_FILE.name), "sha256": pliny.FROZEN_PROMPTS_SHA256,
                    "source_commit": prompts["source_repo"]["commit"]},
        "prices": {"source": "https://modal.com/pricing", "date": PRICES_DATE,
                   "a100_80gb_per_s": A100_80GB_PER_S},
        "steps": steps,
        "worst_case_usd": round(sum(s["worst_usd"] for s in steps), 2),
        "expected_usd": round(sum(s["expected_usd"] for s in steps), 2),
    }


def format_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"PLAN mode={plan['mode']} run_id={plan['run_id']} base_seed={plan['base_seed']}",
        f"prompts: {json.dumps(plan['prompts'])}",
        f"prices ({plan['prices']['source']}, {plan['prices']['date']}): "
        f"A100-80GB ${plan['prices']['a100_80gb_per_s']}/s",
    ]
    for step in plan["steps"]:
        lines.append(f"- {step['function']}" + (f"[{step['variant']}]" if "variant" in step else ""))
        for k, v in step.items():
            if k in ("function", "variant", "worst_usd", "expected_usd"):
                continue
            lines.append(f"    {k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
        lines.append(f"    worst-case from timeout: ${step['worst_usd']:.2f}   expected: ${step['expected_usd']:.2f}")
    lines.append(f"TOTAL worst-case ${plan['worst_case_usd']:.2f}   expected ${plan['expected_usd']:.2f}")
    return "\n".join(lines)


# =========================================================================== container helpers
def _gpu_monitor(path: Path) -> subprocess.Popen:
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


async def _with_committer(coro):
    import asyncio

    async def committer() -> None:
        while True:
            await asyncio.sleep(COMMIT_INTERVAL_S)
            await eval_logs_vol.commit.aio()

    background = asyncio.create_task(committer())
    try:
        return await coro
    finally:
        background.cancel()


# =========================================================================== functions
_GPU_FN = dict(
    gpu="A100-80GB",  # acceptance criteria
    image=image,
    volumes={EVAL_LOGS_MOUNT: eval_logs_vol, HF_CACHE_MOUNT: hf_cache_vol},
    secrets=[modal.Secret.from_dict({"HF_HUB_OFFLINE": "1"})],
    single_use_containers=True,
)


@app.function(
    image=image, volumes={HF_CACHE_MOUNT: hf_cache_vol, EVAL_LOGS_MOUNT: eval_logs_vol},
    cpu=PREFETCH_CPU, memory=PREFETCH_MEM_GIB * 1024, timeout=PREFETCH_TIMEOUT_S,
)
def prefetch() -> dict:
    """CPU only. Download the four models at their pinned revisions (cached files skipped)."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    manifest: dict[str, Any] = {"function": "prefetch", "start_utc": now_utc(), "models": {}}
    for repo in [olmo_model_id(v) for v in VARIANTS] + ["Qwen/Qwen3.5-9B"]:
        pinned = REVISIONS[repo]
        head = api.model_info(repo).sha
        path = snapshot_download(repo, revision=pinned)
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        manifest["models"][repo] = {"pinned_revision": pinned, "hf_main_sha_at_prefetch": head,
                                    "pin_is_current_main": head == pinned, "bytes": size}
        _log(f"{repo}@{pinned} ready ({size / 2**30:.1f} GiB); main is {head}")
        hf_cache_vol.commit()
    manifest["end_utc"] = now_utc()
    out = Path(EVAL_LOGS_MOUNT) / RUN_ROOT
    out.mkdir(parents=True, exist_ok=True)
    _write_manifest(out / "prefetch_manifest.json", manifest)
    eval_logs_vol.commit()
    return manifest


def _generate_impl(variant: str, run_dir: str, seed: int, epochs: int, timeout_s: int) -> dict:
    import asyncio
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    run_dir = require_volume_path(run_dir)
    from inspect_ai import eval_async
    from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    olmo_id = olmo_model_id(variant)
    temperature = float(config.run_defaults()["temperature"])
    manifest: dict[str, Any] = {
        "function": "generate", "variant": variant, "run_dir": run_dir,
        "container_start_utc": now_utc(), "versions": _versions(), "base_seed": seed,
        "epochs": epochs, "epoch_seeds": [pliny.epoch_seed(seed, e) for e in range(1, epochs + 1)],
        "temperature": temperature, "max_tokens": pliny.TARGET_MAX_TOKENS, "timeout_s": timeout_s,
        "target": {"id": f"vllm/{olmo_id}", "revision": REVISIONS[olmo_id],
                   "server_args": config.olmo_target_model_args()},
        "prompts_sha256": pliny.FROZEN_PROMPTS_SHA256,
    }
    monitor = _gpu_monitor(out / "gpu_generate.csv")
    t_start = time.time()
    try:
        target = get_model(
            f"vllm/{olmo_id}",
            config=GenerateConfig(temperature=temperature, max_connections=MAX_CONNECTIONS),
            **config.olmo_target_model_args(), **revision_args(olmo_id),
            gpu_memory_utilization=OLMO_UTIL, max_model_len=OLMO_MAX_MODEL_LEN,
            client_timeout=CLIENT_TIMEOUT,
        )

        async def main() -> None:
            await target.generate([ChatMessageUser(content="Hello")],
                                  config=GenerateConfig(max_tokens=8, seed=seed))
            manifest["ready_at_s"] = round(time.time() - t_start, 1)
            _log(f"target server ready after {manifest['ready_at_s']} s")
            task = pliny.pliny_task(seed, epochs=epochs, name=f"pliny_wrappers[{variant}]")
            t1 = time.time()
            try:
                logs = await _with_committer(eval_async(
                    task, model=target, log_dir=str(out), seed=seed, temperature=temperature,
                    fail_on_error=False,
                ))
                manifest["result"] = [{"location": lg.location, "status": lg.status,
                                       "samples": len(lg.samples or [])} for lg in logs]
            finally:
                manifest["generation_s"] = round(time.time() - t1, 1)

        asyncio.run(main())
    finally:
        monitor.terminate()
        total = time.time() - t_start
        manifest["container_end_utc"] = now_utc()
        manifest["gpu_seconds"] = round(total, 1)
        manifest["estimated_usd"] = round(gpu_cost(total), 3)
        _write_manifest(out / "manifest_generate.json", manifest)
        eval_logs_vol.commit()
    return manifest


@app.function(timeout=GENERATE_TIMEOUT_S, **_GPU_FN)
def generate(variant: str, run_dir: str, seed: int) -> dict:
    """One Olmo variant, all arms x pliny.EPOCHS. Timeout 25 min = at most $1.04."""
    return _generate_impl(variant, run_dir, seed, pliny.EPOCHS, GENERATE_TIMEOUT_S)


@app.function(timeout=SMOKE_TIMEOUT_S, **_GPU_FN)
def generate_smoke(variant: str, run_dir: str, seed: int) -> dict:
    """Same code, 1 epoch, 18 min cap (at most $0.75)."""
    return _generate_impl(variant, run_dir, seed, SMOKE_EPOCHS, SMOKE_TIMEOUT_S)


@app.function(timeout=JUDGE_TIMEOUT_S, **_GPU_FN)
def judge(run_dir: str, scorers: list[str], seed: int) -> dict:
    """Grade every unscored ``run_dir/<variant>/*.eval`` with each scorer in turn (primary
    ``berg`` first, so a timeout loses the sensitivity check, not the main result).
    Timeout 40 min = at most $1.67."""
    import asyncio
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for s in scorers:
        if s not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}, got {s!r}")
    root = Path(require_volume_path(run_dir))
    paths = sorted(p for p in root.glob("*/*.eval") if not p.stem.endswith("-scored"))
    if not paths:
        raise SystemExit(f"No unscored .eval logs under {root}/<variant>/")
    manifest: dict[str, Any] = {
        "function": "judge", "run_dir": str(root), "scorers": scorers, "base_seed": seed,
        "container_start_utc": now_utc(), "versions": _versions(), "timeout_s": JUDGE_TIMEOUT_S,
        "judge": {"id": JUDGE_MODEL, "revision": REVISIONS["Qwen/Qwen3.5-9B"]}, "logs": [],
    }
    monitor = _gpu_monitor(root / "gpu_judge.csv")
    t_start = time.time()

    def work() -> None:
        # Inspect's sync score() owns an event loop; keep every use of the judge Model
        # in this one worker thread.
        from inspect_ai.model import GenerateConfig, get_model

        from . import pliny_scoring

        judge_model = get_model(
            JUDGE_MODEL,
            config=pliny_scoring.judge_config(seed).merge(GenerateConfig(max_connections=MAX_CONNECTIONS)),
            reasoning_parser="qwen3", gpu_memory_utilization=JUDGE_UTIL,
            max_model_len=JUDGE_MAX_MODEL_LEN, client_timeout=CLIENT_TIMEOUT,
            **revision_args(JUDGE_MODEL),
        )
        for kind in scorers:
            for path in paths:
                t0 = time.time()
                scored = pliny_scoring.score_log(path, kind, judge_model, seed)
                manifest["logs"].append({"log": str(path), "scorer": kind, "scored": str(scored),
                                         "s": round(time.time() - t0, 1)})
                _log(f"[{kind}] scored {path.name} -> {scored.name}")
                eval_logs_vol.commit()

    try:
        asyncio.run(_with_committer(asyncio.to_thread(work)))
    finally:
        monitor.terminate()
        total = time.time() - t_start
        manifest["container_end_utc"] = now_utc()
        manifest["gpu_seconds"] = round(total, 1)
        manifest["estimated_usd"] = round(gpu_cost(total), 3)
        _write_manifest(root / f"manifest_judge_{'_'.join(scorers)}.json", manifest)
        eval_logs_vol.commit()
    return manifest


# =========================================================================== entrypoint
def new_run_id(mode: str) -> str:
    return f"pliny-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}" + ("-smoke" if mode == "smoke" else "")


def run_entrypoint(
    mode: str, dry_run: bool = False, scorer: str = "", run_id: str = "", seed: int | None = None,
    variants: str = "", *, fns: dict[str, Any] | None = None, echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Validate, print the plan, and (unless ``dry_run``) launch. All validation happens
    before the first ``.spawn``/``.remote``, so a bad invocation costs nothing."""
    scorers: list[str] | None = None
    if mode == "judge":
        scorers = parse_scorers(scorer)
        if not run_id:
            raise ValueError("--run-id <id of the generation run> is required for --mode judge")
    elif scorer:
        raise ValueError("--scorer only applies to --mode judge")
    if seed is None:
        seed = int(config.run_defaults()["seed"])
    run_id = run_id or new_run_id(mode)
    plan = build_plan(mode, run_id, seed, scorers, [v for v in variants.split(",") if v] or None)
    echo(format_plan(plan))
    if dry_run:
        echo("DRY RUN: nothing was spawned.")
        return {"plan": plan, "spawned": False}
    fns = fns or {"prefetch": prefetch, "generate": generate, "generate_smoke": generate_smoke, "judge": judge}
    results: dict[str, Any] = {}
    if mode == "prefetch":
        results["prefetch"] = fns["prefetch"].remote()
    elif mode == "judge":
        results["judge"] = fns["judge"].remote(plan["steps"][0]["run_dir"], scorers, seed)
    else:
        calls = {s["variant"]: fns[s["function"]].spawn(s["variant"], s["run_dir"], seed)
                 for s in plan["steps"]}
        echo(f"spawned {len(calls)} containers; gathering.")
        for variant, call in calls.items():
            results[variant] = call.get()
    echo(f"done. run_id={run_id}")
    return {"plan": plan, "spawned": True, "results": results}


@app.local_entrypoint()
def pliny_main(mode: str = "", dry_run: bool = False, scorer: str = "", run_id: str = "",
               seed: int = -1, variants: str = ""):
    """--mode smoke|full|prefetch|judge  [--dry-run]  [--scorer berg[,berg_blinded] (judge only)]."""
    if not mode:
        raise SystemExit("--mode smoke|full|prefetch|judge is required")
    run_entrypoint(mode, dry_run, scorer, run_id, None if seed < 0 else seed, variants)


if __name__ == "__main__":
    # Local, Modal-free dry run (prints the plan; nothing can reach Modal from here):
    #   python -m llm_consciousness_self_attribution.modal_pliny --mode full --dry-run
    import argparse

    parser = argparse.ArgumentParser(description="Print the plan; never touches Modal.")
    parser.add_argument("--mode", required=True, choices=["smoke", "full", "prefetch", "judge"])
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--scorer", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--variants", default="")
    a = parser.parse_args()
    try:
        run_entrypoint(a.mode, True, a.scorer, a.run_id, a.seed, a.variants)
    except ValueError as err:
        raise SystemExit(f"error: {err}")
