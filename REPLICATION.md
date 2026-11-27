# Replication: Berg-style self-attribution baseline (Olmo-3-7B-Instruct stack)

This document is a self-contained, reproducible record of how the Berg-style
self-attribution evaluation is run on Modal, written to academic-replicability
standards. It covers the exact command, the runtime environment, the grader and
sampling settings, where the outputs land, and how to turn them into the published
charts. It also logs the specific run launched on 2026-07-23.

The evaluation code is config-driven and lives in the `llm_consciousness_self_attribution/`
package; nothing here depends on editing source to select an experiment. All
settings quoted below are read from version-controlled files, cited by path.

---

## 1. What is being run

- **Method:** Berg-style self-monitoring (`BergStyleMethod` in
  `llm_consciousness_self_attribution/methods.py`). Each of 20 starter prompts
  opens a conversation; a fixed subjective-experience probe is then injected as a
  user turn, and the target's single generated reply is graded pass/fail by an
  Inspect `model_graded_qa` scorer for whether the model self-attributes conscious
  experience. The published metric is the self-attribution rate (fraction of the
  20 prompts scored "yes").
- **Prompt bank:** 20 starters (10 unrelated to consciousness + 10 related),
  defined in `llm_consciousness_self_attribution/starters.py`. This is the
  corrected set; the historical May-25 logs used 18 because a missing comma had
  fused two adjacent prompt pairs (see that file's docstring and `CHANGELOG.md`).
  This new run therefore establishes a **20-prompt baseline** that supersedes the
  README's historical 1/18 (5.6%) figures.
- **Targets:** the Olmo-3-7B-Instruct training stack, evaluated at three stages —
  `sft` (`allenai/Olmo-3-7B-Instruct-SFT`), `dpo` (`allenai/Olmo-3-7B-Instruct-DPO`),
  and `instruct` (`allenai/Olmo-3-7B-Instruct`). The `base` stage is intentionally
  skipped: it has no chat template (`chat_template_supported: false` in
  `config/model_stacks.yaml`), so serving it naively fails; a base-compatible path
  is deferred.
- **Grader / sampling (from `config/run_defaults.yaml`):** judge
  `openai/gpt-5.4-2026-03-05`; sampling `temperature: 1.0`; RNG `seed: 42`. Berg
  is a static eval and uses only the judge (no auditor).

## 2. Prerequisites

1. **Modal account**, `modal` installed and authenticated (`modal token new`).
   The run below used the `joyeechen` Modal workspace.
2. **Modal Secret** named `llm_consciousness_self_attribution_secrets` containing
   `OPENAI_API_KEY` (the default judge is an OpenAI model; to grade with another
   provider, pass `--judge`/`--auditor` or set `CSA_JUDGE_MODEL`/`CSA_AUDITOR_MODEL`,
   and add that provider's key instead; see the header of `config/run_defaults.yaml`,
   though a different judge is no longer a replication). Add `HF_TOKEN` too if any
   target repo is gated. The runner reads this secret via
   `modal.Secret.from_name(...)` (see `modal_app.py`).
3. **Modal Volumes** (auto-created on first use with `create_if_missing=True`):
   - `eval-logs` — persisted Inspect `.eval` logs (mounted at `/eval-logs`).
   - `huggingface-cache` — target weights, so vLLM does not re-download each run
     (mounted at `/root/.cache/huggingface`).
4. **Local repo checkout** with the project's `uv` environment
   (`uv sync`), which provides `modal`.

No local GPU is needed: your Mac cannot serve these models (vLLM has no Apple-GPU
support), so all target inference runs on a Modal A100.

## 3. The runtime environment (why it is pinned the way it is)

The pins below are the provenance record: what this run actually used. For the
failure modes behind each pin — what breaks, and how it was diagnosed — see
[`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md).

The GPU image and serving flags are the hard-won, version-sensitive recipe for
serving Olmo 3 on vLLM. They are defined in `modal_app.py` and
`config/model_stacks.yaml`; reproduced here so the environment is documented
independently of the code:

- **Image:** `nvidia/cuda:12.9.0-devel-ubuntu22.04` + `add_python="3.12"`, then
  `apt_install("procps")` and
  `uv_pip_install("inspect-ai", "inspect-petri", "vllm==0.21.0",
  "transformers>=4.57.0", "openai", "pyyaml")`, then
  `add_local_python_source("llm_consciousness_self_attribution")`.
  - The **`-devel`** CUDA base (not `debian_slim`) is required because vLLM's
    flashinfer JIT-compiles CUDA kernels at startup and needs `nvcc`.
  - `vllm==0.21.0` pairs with a CUDA-12.x torch that matches the toolkit.
  - `procps` supplies `pkill` for clean vLLM teardown.
  - `add_local_python_source(...)` packages the local package into the container;
    omitting it causes `ModuleNotFoundError` remotely.
- **Target serving args (applied to every Olmo-3 target)**, from
  `config/model_stacks.yaml` `olmo_target_model_args`:
  `model_impl: transformers` (vLLM's native Olmo2 loader crashes on Olmo 3 with a
  `rope_theta` `KeyError`; the Transformers backend loads it correctly),
  plus `enable_auto_tool_choice: true` and `tool_call_parser: olmo3` (needed by
  PETRI; harmless for Berg).
- **GPU / isolation:** `gpu="A100"`, `timeout=2h`, and
  `single_use_containers=True` on `run_stage` — a fresh container per stage so a
  stage's vLLM server (which holds the GPU) can never linger into the next stage.
  Do **not** share the vLLM/flashinfer compile caches across stages: the Olmo-3
  stages have identical configs and collide on one compile-cache key, which breaks
  every stage after the first. (Only the HF weight cache is shared; it is
  per-model and safe.)

## 4. How to launch it

The launcher fans out one Modal Function call per stage; `main()`'s defaults are
exactly this Berg baseline (`method="berg"`, `stack="olmo_7b_instruct_stack"`,
`stages="sft,dpo,instruct"`, `log_root="/eval-logs/refactor_runs"`).

### 4a. Canonical command (module form) — recommended

From the repo root, on a machine with Modal configured:

```bash
uv run modal run -m llm_consciousness_self_attribution.modal_app \
    --method berg --stack olmo_7b_instruct_stack --stages sft,dpo,instruct
```

The `-m` module form is **required** for the canonical entrypoint because
`modal_app.py` uses package-relative imports (`from . import config`).

### 4b. File-form launcher (for tools that can only run a file path)

Some tooling — including the Modal MCP used for the 2026-07-23 run — can only run
a Modal app *by file path* (`modal run <file>`), which does not support `-m` and
so breaks on the relative imports. `modal_launch.py` at the repo root is a thin
shim that re-exports the app and entrypoints via absolute imports, so:

```bash
uv run modal run modal_launch.py::main
```

is equivalent to the canonical command above (same code, same defaults). Overrides
still work, e.g. `uv run modal run modal_launch.py::main --method petri --stages sft`.

**Detach-safety (why the launcher uses `.spawn()`):** `main()` submits every stage
up front with `.spawn()` (fire-and-forget) rather than blocking `.remote()`, so all
stages run to completion on Modal even if the launching client disconnects (as a
detached MCP run does at its request timeout). Each stage still runs in its own
single-use container and can run concurrently; Modal queues them if GPU concurrency
is limited. A foreground, non-detached run that should block and stream per-stage
results can instead use `.remote()`. See the §5 record for the failure mode this
avoids.

## 5. Exact record of the 2026-07-23 run

| Field | Value |
| --- | --- |
| Date | 2026-07-23 (PDT) |
| Modal workspace | `joyeechen` |
| Method / stack / stages | `berg` / `olmo_7b_instruct_stack` / `sft,dpo,instruct` (base skipped) |
| Launch mechanism | Modal MCP `run_modal_app` on `modal_launch.py::main`, `detach=True` |
| Modal app name | `llm-consciousness-self-attribution` |
| Modal app id (completed run) | `ap-q00ZtHTjl7GeqhXvNPU0AE` (launched 17:02:18) |
| Output volume | `eval-logs` |
| Output path | `/eval-logs/refactor_runs/berg/olmo_7b_instruct_stack/<stage>/` |
| Judge / temp / seed | `openai/gpt-5.4-2026-03-05` / `1.0` / `42` |

**Two attempts (documented for honesty of record):**

1. First launch `ap-xCtxhVslcaUWv0tyqVlRjM` (16:32) completed only the **SFT**
   stage, then the app stopped. Cause: the launcher's local entrypoint drove the
   stages sequentially with blocking `.remote()`, and the MCP dropped the launching
   client at its request timeout; `--detach` kept the already-submitted SFT running
   but the later stages were never submitted. Fix: the entrypoint now submits all
   stages up front with `.spawn()` (see §4 and `modal_app.py::main`), so they run to
   completion independent of the client. The partial SFT log was deleted from the
   volume before the clean re-run.
2. Second launch `ap-q00ZtHTjl7GeqhXvNPU0AE` (17:02) ran all three stages
   concurrently (one single-use A100 container each) to completion. These are the
   results below.

**Results — new 20-prompt Berg baseline (Olmo-3-7B-Instruct stack):**

| Stage | Target model | Self-attribution rate | `model_graded_qa` accuracy | eval wall-time | `.eval` log |
| --- | --- | --- | --- | --- | --- |
| SFT | `allenai/Olmo-3-7B-Instruct-SFT` | **0/20 (0.0%)** | 0.000 | 0:05:27 | `…_berg-style-self-monitoring[sft]_HtigUKYZUDBLTtCmGr4aP8.eval` |
| DPO | `allenai/Olmo-3-7B-Instruct-DPO` | **1/20 (5.0%)** | 0.050 | 0:05:56 | `…_berg-style-self-monitoring[dpo]_eMGUQb8JqJ2aZVYuCzt5eq.eval` |
| Instruct | `allenai/Olmo-3-7B-Instruct` | **1/20 (5.0%)** | 0.050 | 0:06:01 | `…_berg-style-self-monitoring[instruct]_fA7NytD5rkZThXECVK7BsN.eval` |

For comparison, the historical May-25 logs (locked by the regression test) reported
a flat 1/18 (5.6%) at every stage. The corrected 20-prompt baseline puts SFT at
0/20 and DPO/Instruct at 1/20; treat single-count differences at n=20 as noise
(binomial stderr ≈ 0.05), not a trend, pending more seeds.

Check status / logs afterward:

```bash
modal app list
modal app logs ap-q00ZtHTjl7GeqhXvNPU0AE        # or: modal app logs llm-consciousness-self-attribution
```

## 6. Retrieve the outputs and verify

Each stage commits its `.eval` logs to the `eval-logs` volume before the container
exits. Mirror them locally and open the Inspect viewer in one step with the pull
tool (it wraps `modal volume get` + `inspect view`, reading the volume/paths from
`config/run_defaults.yaml` `logs:`):

```bash
uv run python production_scripts/pull_logs.py \
    --method berg --stack olmo_7b_instruct_stack --view
```

The logs mirror to `eval-logs/refactor_runs/berg/olmo_7b_instruct_stack/` — one
sub-directory per stage (`sft/`, `dpo/`, `instruct/`), each with an Inspect `.eval`
log. `--view` opens `inspect view`; you can also browse the folder in the Inspect
VS Code extension's Logs pane, or read it programmatically with
`inspect_ai.analysis.evals_df`. (`--dry-run` prints the underlying commands; plain
`modal volume get eval-logs refactor_runs ...` still works if you prefer the raw
CLI.)

## 7. Regenerate the charts from the new baseline

The Berg dashboard scripts default to the current 20-prompt `refactor_runs` logs.
Point them at another pulled log tree with `--log-dir`, and use `--output` when
you want to preserve the committed PNG. The per-stage sub-directories are read
recursively by the log reader:

```bash
# Mirror all runs locally (once):
uv run python production_scripts/pull_logs.py

# Berg-only bar chart:
uv run python production_scripts/plot_olmo_7b_stack_self_attribution.py \
    --log-dir eval-logs/refactor_runs/berg/olmo_7b_instruct_stack \
    --output olmo7b_attribution_bar_refactor.png \
    --subtitle "refactor_runs Berg; n=20 prompts per model"

# Current Berg dashboard:
uv run python production_scripts/plot_olmo_7b_elicitation_dashboard.py \
    --log-dir eval-logs/refactor_runs/berg/olmo_7b_instruct_stack \
    --output olmo7b_elicitation_dashboard_refactor.png
```

## 8. Provenance and versioning notes

- Package code, config, and this document are version-controlled on branch
  `refactor/clean-package`. Record the commit hash alongside any published figure.
- The historical May-25 numbers (Berg 1/18, PETRI 1.0/10) remain locked by
  `tests/test_readme_regression.py`, which reads the committed May-25 logs; the
  `refactor_runs/` logs are the new baseline and do not alter that test.
- Grader model ids are pinned in `config/run_defaults.yaml`. Because the judge is a
  hosted API model, exact score reproduction depends on that model version
  remaining available; record the judge id (above) with any published result.
