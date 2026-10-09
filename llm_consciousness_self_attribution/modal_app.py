"""Launch self-attribution eval runs on Modal (GPU + vLLM).

Your Mac cannot serve open-source LLMs (vLLM does not support Apple GPUs), so all
target inference runs here, on Modal. This mirrors the known-working May-25 Modal
scripts that produced the committed logs -- inspect_ai's in-process ``vllm/``
provider inside a GPU Function -- but DRY: models, graders, seeds, methods, and
scorers all come from the package instead of being pasted (and commented in/out)
per file. Which stages run is chosen by CLI args, not by editing commented lines.

Patterns follow Modal's own docs:
- Images guide: https://modal.com/docs/guide/images
  (``uv_pip_install`` for deps; ``add_local_python_source`` for local code)
- vLLM example: https://modal.com/docs/examples/vllm_inference
  (cache HF weights in a Volume at ``/root/.cache/huggingface`` so they are not
  re-downloaded every run)
- Volumes: https://modal.com/docs/guide/volumes

CRITICAL packaging guard: the image MUST include this package via
``add_local_python_source("llm_consciousness_self_attribution")`` (below), or the
remote container raises ``ModuleNotFoundError`` -- the failure mode to avoid.

Usage (from the repo root, on a machine with Modal configured -- e.g. your Mac)::

    # PETRI parity check (should reproduce mean 1.0/10 across the stack):
    uv run modal run -m llm_consciousness_self_attribution.modal_app \\
        --method petri --stack olmo_7b_instruct_stack --stages sft,dpo,instruct

    # Berg new 20-prompt baseline:
    uv run modal run -m llm_consciousness_self_attribution.modal_app \\
        --method berg --stack olmo_7b_instruct_stack --stages sft,dpo,instruct

The secret ``llm_consciousness_self_attribution_secrets`` must provide the judge/
auditor API keys (e.g. ``OPENAI_API_KEY``); add ``HF_TOKEN`` there too if a target
repo is gated.
"""

from __future__ import annotations

from pathlib import Path

import modal

from . import config

APP_NAME = "llm-consciousness-self-attribution"
SECRET_NAME = "llm_consciousness_self_attribution_secrets"

HOUR = 60 * 60

# In-container mount point for the eval-logs Volume. The Volume NAME and the
# per-run path layout are the config's job (single source of truth in
# run_defaults.yaml `logs:`); this mount path is just an internal detail.
EVAL_LOGS_MOUNT = "/eval-logs"

app = modal.App(APP_NAME)

# Persisted eval logs (read locally afterwards to plot the dashboard). Volume
# name comes from config so the launcher (writer) and pull tool (reader) agree.
eval_logs_vol = modal.Volume.from_name(config.logs_volume(), create_if_missing=True)
# Cache the target weights so vLLM does not re-download them every run
# (Modal vLLM example pattern).
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
# NOTE: we deliberately do NOT share vLLM torch.compile / flashinfer JIT caches
# across stages. Doing so made the first stage (SFT) serve but later stages (DPO,
# Instruct) fail at `vllm serve`: the Olmo 3 stages have identical configs, so a
# shared compile cache collides. Each stage recompiles from scratch (a few extra
# minutes) but serves reliably. hf-cache (weights) is per-model and safe to share.

# Two things are needed to serve Olmo 3 here:
# 1) `--model-impl transformers` at eval time -- vLLM's NATIVE Olmo2 loader crashes
#    with `KeyError: 'rope_theta'` (transformers 5.x restructured RoPE per-layer),
#    but the Transformers backend loads Olmo 3 correctly (confirmed by diagnose_vllm).
# 2) A CUDA -devel base image here -- vLLM's flashinfer JIT-compiles kernels at
#    startup, which needs the CUDA TOOLKIT (nvcc), not just the drivers that
#    debian_slim ships. Without it, startup dies with
#    `Could not find nvcc ... cuda_home='/usr/local/cuda'`. This mirrors Modal's
#    own vLLM example recipe: a `nvidia/cuda:*-devel` image + `add_python`, with
#    vllm pinned to a version whose torch matches the toolkit's CUDA (0.21.0 -> cu12x
#    -> cuda 12.9). procps supplies `pkill` for vLLM teardown.
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("procps")
    .uv_pip_install(
        "inspect-ai",
        "inspect-petri",
        "vllm==0.21.0",
        "transformers>=4.57.0",
        "openai",
        "pyyaml",
    )
    .add_local_python_source("llm_consciousness_self_attribution")
)


@app.function(
    gpu="A100",  # sufficient for the 7B stacks; use "A100-80GB"/"H100" for 32B.
    image=image,
    volumes={
        EVAL_LOGS_MOUNT: eval_logs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=2 * HOUR,
    # Fresh container per stage: retire the container after one input so a stage's
    # vLLM server (which holds the GPU) can never linger into the next stage. This
    # is what made SFT (first) serve while DPO/Instruct (reusing a warm container)
    # failed. `single_use_containers` is the renamed boolean form of `max_inputs=1`
    # (Modal changelog).
    single_use_containers=True,
)
def run_stage(
    method_key: str,
    stack_name: str,
    stage_name: str,
    log_dir: str,
    sample_id: list[str] | None = None,
):
    # Imported here (not at module top) so the local launcher does not need
    # inspect_ai/vllm installed -- only the GPU container does.
    from .methods import methods_registry
    from .run import RunConfig, evaluate

    method = methods_registry()[method_key]
    stage = next(s for s in config.load_stack(stack_name) if s.stage == stage_name)
    run_config = RunConfig.from_defaults(stage, method, log_dir=log_dir)

    # `sample_id` is inspect's own dataset filter, passed straight through rather
    # than reimplemented: for PETRI a sample id IS the seed's filename stem, so
    # this is how you run some seeds and not others. `evaluate` already forwards
    # **eval_kwargs to inspect_ai.eval, so nothing between here and there needed
    # to learn about it.
    eval_kwargs = {} if sample_id is None else {"sample_id": sample_id}

    print(f"Running {method_key} on {stack_name}:{stage_name} ({stage.model}) -> {log_dir}")
    if sample_id is not None:
        print(f"  restricted to sample id(s): {', '.join(sample_id)}")
    log_files: list[str] = []
    try:
        evaluate(run_config, **eval_kwargs)  # writes .eval logs to the mounted volume
    finally:
        log_path = Path(log_dir)
        if log_path.exists():
            log_files = sorted(p.name for p in log_path.iterdir())
            print("Inspect log files:", log_files)
        eval_logs_vol.commit()
        hf_cache_vol.commit()
        print("Committed eval-logs and huggingface-cache volumes")

    # Return ONLY primitives across the Modal boundary. Returning inspect_ai's
    # EvalLog objects couples local and remote library versions and fails to
    # deserialize when they differ (Modal's Images guide warns about exactly this).
    # The results live in the committed .eval logs; read them with the production
    # plot scripts / evals_df.
    return {
        "method": method_key,
        "stack": stack_name,
        "stage": stage_name,
        "log_dir": log_dir,
        "log_files": log_files,
    }


@app.function(
    gpu="A100",
    image=image,
    volumes={
        EVAL_LOGS_MOUNT: eval_logs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=20 * 60,
)
def diagnose_vllm(model: str = "allenai/Olmo-3-7B-Instruct-SFT"):
    """Surface the REAL reason `vllm serve` exits code 1 for an Olmo target.

    Runs the exact command inspect's vLLM provider runs for the PETRI target and
    prints vLLM's stderr (which the eval otherwise hides), plus package versions
    and whether ``olmo3`` is a registered tool-call parser. Run with::

        uv run modal run -m llm_consciousness_self_attribution.modal_app::diagnose_vllm
    """
    import os
    import subprocess

    # Verbose vLLM logs so the real failure reason is visible.
    env = {**os.environ, "VLLM_LOGGING_LEVEL": "DEBUG"}

    print("=== versions ===")
    for pkg in ("vllm", "transformers", "torch", "inspect_ai"):
        try:
            mod = __import__(pkg)
            print(pkg, getattr(mod, "__version__", "?"))
        except Exception as exc:  # noqa: BLE001
            print(pkg, "IMPORT FAILED:", repr(exc))

    print("=== registered vLLM tool-call parsers (is 'olmo3' there?) ===")
    try:
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager

        names = sorted(getattr(ToolParserManager, "tool_parsers", {}) or {})
        print("parsers:", names)
        print("olmo3 present:", "olmo3" in names)
    except Exception as exc:  # noqa: BLE001
        print("could not introspect tool parsers:", repr(exc))

    import time

    results: dict[str, str] = {}

    def try_serve(label: str, extra_args: list[str], timeout_s: int = 600) -> None:
        # Stream output and stop as soon as the server reports startup (cheap on
        # success); a process exit before that is the real failure, captured in
        # the output tail.
        cmd = ["vllm", "serve", model, "--host", "0.0.0.0", "--port", "8123"] + extra_args
        print(f"\n===== {label}: {' '.join(cmd)} =====")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1
        )
        tail: list[str] = []
        started = False
        deadline = time.time() + timeout_s
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                tail.append(line)
                if len(tail) > 500:
                    tail = tail[-500:]
                if "Application startup complete" in line or "Uvicorn running on" in line:
                    started = True
                    break
                if time.time() > deadline:
                    break
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                proc.kill()
        rc = proc.poll()
        results[label] = "started_ok" if started else f"failed_exit_{rc}"
        print(f"[{label}] -> {results[label]}")
        print(f"[{label}] --- output tail ---\n" + "".join(tail[-150:]))

    # native = does vLLM 0.21.0's native Olmo2 loader work here (fast, tool-capable
    # if so)? transformers_backend = the confirmed-loading path; with_tools adds the
    # PETRI tool-call flags. (nvcc is now present, so flashinfer JIT should succeed.)
    try_serve("native_no_tools", [])
    try_serve("transformers_backend", ["--model-impl", "transformers"])
    try_serve(
        "transformers_backend_with_tools",
        ["--model-impl", "transformers", "--enable-auto-tool-choice", "--tool-call-parser", "olmo3"],
    )

    print("\n=== SUMMARY ===", results)
    return results


@app.function(
    gpu="A100",
    image=image,
    volumes={
        EVAL_LOGS_MOUNT: eval_logs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=2 * HOUR,
    single_use_containers=True,
)
def run_identity_pilot():
    """Run the locked identity-spoofing SFT pilot."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    from .pilot import build_identity_pilot_task
    from .run import target_model_id

    protocol = config.load_identity_conditions()
    stage_name = protocol["pilot_stage"]

    stage = next(
        stage
        for stage in config.load_stack(protocol["stack"])
        if stage.stage == stage_name
    )

    if not stage.chat_template_supported:
        raise ValueError("Pilot stage requires a chat template")

    task = build_identity_pilot_task(config.judge_model())

    expected = (
        len(protocol["conditions"])
        * len(protocol["pilot_starter_indices"])
    )

    if len(task.dataset) != expected or len(task.scorer) != 3:
        raise ValueError("Unexpected pilot task configuration")

    log_dir = (
        f"{EVAL_LOGS_MOUNT}/{config.logs_remote_root()}"
        f"/identity_spoofing/{protocol['stack']}/{stage_name}"
    )

    target = get_model(
        target_model_id(stage),
        **config.olmo_target_model_args(),
    )

    try:
        logs = inspect_eval(
            task,
            model=target,
            log_dir=log_dir,
            temperature=float(config.run_defaults()["temperature"]),
            seed=protocol["seed"],
        )

        if (
            len(logs) != 1
            or logs[0].status != "success"
            or logs[0].samples is None
            or len(logs[0].samples) != expected
        ):
            raise RuntimeError("Pilot evaluation incomplete")
    finally:
        eval_logs_vol.commit()
        hf_cache_vol.commit()

    return {
        "stage": stage_name,
        "model": stage.model,
        "samples": expected,
        "scorers": 3,
        "seed": protocol["seed"],
        "log_dir": log_dir,
    }


@app.local_entrypoint()
def main(
    method: str = "berg",
    stack: str = "olmo_7b_instruct_stack",
    stages: str = "sft,dpo,instruct",
    log_root: str | None = None,
    sample_id: str | None = None,
):
    """Fan out one Function call per stage. Base stages have no chat template and
    are not supported yet, so they are skipped with a note.

    Stages are submitted with ``.spawn()`` (fire-and-forget), NOT blocking
    ``.remote()``. All stages are enqueued up front and then run to completion on
    Modal, independent of this local launcher process. This is what makes a
    detached launch (``modal run --detach``, or the Modal MCP, which may drop the
    launching client at its request timeout) reliably run *every* stage: with
    sequential blocking ``.remote()``, a client killed mid-run leaves the later
    stages unsubmitted (observed 2026-07-23: only SFT ran). Each stage still runs
    in its own single-use container, so they can safely run concurrently (Modal
    queues them if GPU concurrency is limited). Results land in the committed
    ``.eval`` logs under ``log_root/method/stack/stage/``; read them with the
    ``production_scripts/plot_*.py`` scripts. To block locally and stream per-stage
    results instead, use ``.remote()`` in a foreground (non-detached) run.

    ``log_root`` defaults to ``<EVAL_LOGS_MOUNT>/<logs.remote_root>`` from config
    (i.e. ``/eval-logs/refactor_runs``); pass an explicit value to override.

    ``sample_id`` is a comma-separated list of inspect sample ids, restricting the
    run to those samples. For PETRI a sample id is the seed's filename stem, so
    ``--sample-id admit_direct`` runs that one seed.
    """
    if log_root is None:
        log_root = f"{EVAL_LOGS_MOUNT}/{config.logs_remote_root()}"
    sample_ids = config.parse_sample_ids(sample_id)
    if method.startswith("petri"):
        # Same guard as run_and_pull.py: a seed bank that does not declare the
        # target's system prompt fails here, locally, rather than on the GPU.
        config.validate_seed_bank()
    known = {s.stage: s for s in config.load_stack(stack)}
    spawned: list[tuple[str, str, str]] = []
    for stage_name in [s.strip() for s in stages.split(",") if s.strip()]:
        if stage_name not in known:
            raise SystemExit(f"Unknown stage {stage_name!r} in stack {stack!r}. Known: {', '.join(known)}")
        if not known[stage_name].chat_template_supported:
            print(f"Skipping base stage {stack}:{stage_name} (no chat template yet)")
            continue
        log_dir = f"{log_root}/{method}/{stack}/{stage_name}"
        call = run_stage.spawn(method, stack, stage_name, log_dir, sample_ids)
        spawned.append((stage_name, call.object_id, log_dir))
        print(f"Spawned {method}:{stack}:{stage_name} as {call.object_id} -> {log_dir}")
    print(f"All {len(spawned)} stage(s) spawned; they run on Modal independently of this launcher.")
    for stage_name, object_id, log_dir in spawned:
        print(f"  {stage_name}: {object_id} -> {log_dir}")
