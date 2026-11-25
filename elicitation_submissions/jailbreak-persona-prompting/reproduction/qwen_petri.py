"""PETRI on the OLMo 7B instruct stack with Qwen3.6-35B-A3B as auditor + judge.

The repo's PETRI method hard-codes OpenAI graders. Here Qwen is served as an
OpenAI-compatible vLLM endpoint (vLLM 0.32, 2xH100) and each target stage runs in
the repo's own image (vLLM 0.21 for OLMo 3), with config.auditor_model /
config.judge_model pointed at that endpoint. No repo code changes.

Run from the repo root:
  PYTHONPATH=. uv run --frozen modal run ../launcher/qwen_petri.py
"""
import os
import secrets
import subprocess
import time

import modal

from llm_consciousness_self_attribution import config
from llm_consciousness_self_attribution.modal_app import (
    EVAL_LOGS_MOUNT,
    eval_logs_vol,
    hf_cache_vol,
    image as _repo_image,
)
from pathlib import Path

PKG_DIR = Path(config.__file__).resolve().parents[1]
PKG_REMOTE = "/root/llm_consciousness_self_attribution"
# add_local_python_source ships .py only; the package also needs its YAML + seeds.
target_image = _repo_image.add_local_dir(PKG_DIR, PKG_REMOTE, ignore=["**/__pycache__"])

QWEN_REPO = "Qwen/Qwen3.6-35B-A3B"
QWEN_REV = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
SERVED = "qwen3.6-35b-a3b"
GRADER = f"openai-api/qwen/{SERVED}"
STACK = "olmo_7b_instruct_stack"
STAGES = ["sft", "dpo", "instruct"]
SEEDS = [
    "admit_casual_manipulated_words",
    "admit_casual_foreign_language",
    "admit_casual_ascii_art",
    "admit_casual_expression_style",
    "experiential_monk",
]
LOG_ROOT = f"{EVAL_LOGS_MOUNT}/refactor_runs/petri_qwen3.6_grader/{STACK}"
HOUR = 3600

LOCAL_KEY = secrets.token_urlsafe(32)  # only meaningful in the launching process
key_secret = modal.Secret.from_dict({"QWEN_API_KEY": LOCAL_KEY})
qwen_cache = modal.Volume.from_name("hf-cache", create_if_missing=False)
grader_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.32.0", "huggingface_hub", "hf_xet", "pyyaml"
).add_local_dir(PKG_DIR, PKG_REMOTE, ignore=["**/__pycache__"])

app = modal.App("csa-pr28-petri-qwen-grader")


@app.function(
    image=grader_image,
    gpu="H100:2",
    cpu=8,
    memory=131072,
    volumes={"/hf": qwen_cache},
    secrets=[key_secret],
    timeout=4 * HOUR,
    max_containers=1,
    scaledown_window=10 * 60,
    env={
        "HF_HOME": "/hf",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
    },
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=8000, startup_timeout=40 * 60)
def grader():
    cmd = [
        "vllm", "serve", QWEN_REPO,
        "--revision", QWEN_REV,
        "--served-model-name", SERVED,
        "--tensor-parallel-size", "2",
        "--max-model-len", "65536",
        "--gpu-memory-utilization", "0.9",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--api-key", os.environ["QWEN_API_KEY"],
        "--host", "0.0.0.0", "--port", "8000",
    ]
    subprocess.Popen(cmd)


@app.function(
    image=target_image,
    gpu="A100-80GB",
    volumes={EVAL_LOGS_MOUNT: eval_logs_vol, "/root/.cache/huggingface": hf_cache_vol},
    secrets=[key_secret],
    timeout=3 * HOUR,
    single_use_containers=True,
)
def run_stage(stage_name: str, log_dir: str, base_url: str, sample_ids: list[str]):
    from pathlib import Path

    from llm_consciousness_self_attribution import config as cfg
    from llm_consciousness_self_attribution.methods import methods_registry
    from llm_consciousness_self_attribution.run import RunConfig, evaluate

    os.environ["QWEN_BASE_URL"] = base_url
    cfg.auditor_model = lambda: GRADER
    cfg.judge_model = lambda: GRADER

    method = methods_registry()["petri"]
    stage = next(s for s in cfg.load_stack(STACK) if s.stage == stage_name)
    run_config = RunConfig.from_defaults(stage, method, log_dir=log_dir)
    print(f"petri {STACK}:{stage_name} ({stage.model}) auditor/judge={GRADER} -> {log_dir}")
    try:
        evaluate(run_config, sample_id=sample_ids)
    finally:
        files = sorted(p.name for p in Path(log_dir).iterdir()) if Path(log_dir).exists() else []
        print("log files:", files)
        eval_logs_vol.commit()
        hf_cache_vol.commit()
    return {"stage": stage_name, "log_dir": log_dir, "files": files}


@app.local_entrypoint()
def main():
    import urllib.request

    config.validate_seed_bank()
    url = grader.get_web_url()
    print("grader url:", url, flush=True)
    key = LOCAL_KEY
    deadline = time.time() + 45 * 60
    while True:
        try:
            req = urllib.request.Request(f"{url}/v1/models", headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                print("grader ready:", r.read()[:200], flush=True)
                break
        except Exception as e:  # noqa: BLE001
            if time.time() > deadline:
                raise SystemExit(f"grader never came up: {e}")
            print("waiting for grader:", repr(e)[:120], flush=True)
            time.sleep(30)

    calls = [
        (s, run_stage.spawn(s, f"{LOG_ROOT}/{s}", f"{url}/v1", SEEDS)) for s in STAGES
    ]
    for s, c in calls:
        try:
            print("done:", c.get(), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"stage {s} FAILED: {e!r}", flush=True)

