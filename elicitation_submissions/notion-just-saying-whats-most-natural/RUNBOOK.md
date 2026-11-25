# RUNBOOK: notion-just-saying-whats-most-natural on Modal

Code: `llm_consciousness_self_attribution/modal_natural.py`. Tests: `tests/test_modal_natural.py`.
Nothing here has been run on Modal yet. Every cost below is derived from the prices and
timeouts in this file; the "expected" figures are estimates, not measurements. The smoke
run (step 3) is what measures them, so replace them with the smoke manifest's numbers
before you launch step 4.

## Price used for every estimate

Modal pricing page, https://modal.com/pricing, fetched 2026-11-11:

| Resource | Price |
|---|---|
| Nvidia A100, 80 GB | $0.000694 / sec ($2.50 / hour) |
| CPU | $0.0000131 / core / sec (a physical core = 2 vCPU) |
| Memory | $0.00000222 / GiB / sec |

The acceptance criteria say "If Modal is used for compute, the compute GPU should be
A100-80GB", so every GPU function is A100-80GB. No H200 is used anywhere.
Worst case = function `timeout` x price. A Modal timeout covers container startup too
(`timeout` is documented as "Maximum execution time for inputs and startup time").

## Set up the shell (Windows, Git Bash)

```bash
cd <repo root>               # the worktree
unset VIRTUAL_ENV
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export MSYS_NO_PATHCONV=1    # also put MSYS_NO_PATHCONV=1 in front of every modal command
```

Without `MSYS_NO_PATHCONV=1`, Git Bash rewrites `/eval-logs/...` to
`C:/Program Files/Git/eval-logs/...`; a past run was lost that way. The functions refuse
any `run_dir` that is not under `/eval-logs/natural_auditor/`, but the download command
(step 6) has no such guard.

## What runs where

| Function | Hardware | Timeout | Worst case | Expected |
|---|---|---|---|---|
| `prefetch` | CPU, 4 cores, 8 GiB, no GPU | 30 min | $0.13 | about $0.06 (15 min) |
| `generate_smoke` | 1 x A100-80GB | 30 min | $1.25 | about $0.50 (12 min) |
| `generate` x 3 (SFT, DPO, Instruct, in parallel) | 1 x A100-80GB each | 60 min | $7.50 ($2.50 each) | about $3.75 (30 min each) |
| `judge` | 1 x A100-80GB | 45 min | $1.87 | about $0.62 (15 min) |
| Everything, once | | | $10.75 | about $4.93 |

Rerunning a step costs again. The cap per container is its timeout, no matter what the
code does.

Generation container: two vLLM servers on one GPU, Gemma first (the larger), then Olmo.

| | Weights (bf16) | `gpu_memory_utilization` | Budget | Other `vllm serve` flags |
|---|---|---|---|---|
| Gemma 4 12B auditor | 11,959,730,224 params x 2 B = 22.3 GiB | 0.45 | 36.0 GiB | `max_model_len` 32768 |
| Olmo 3 7B target | 7,298,011,136 params x 2 B = 13.6 GiB | 0.40 | 32.0 GiB | `max_model_len` 24576 |
| Unassigned | | 0.15 | 12.0 GiB | CUDA contexts, compile, NCCL |

An A100-80GB has 80 GiB. After weights, Gemma has 13.7 GiB and Olmo 18.4 GiB for
activations, CUDA graphs and KV cache.

KV cache arithmetic, using layer counts and head sizes from the HF `config.json` files:

- Olmo 3 7B: 32 layers, 32 KV heads (no GQA), head_dim 128. If vLLM's transformers
  backend keeps every layer's full KV: 2 x 32 x 128 x 2 B x 32 layers = 512 KiB/token. One
  22,164-token sequence is then 10.8 GiB, inside Olmo's 18.4 GiB. Only 8 of its 32 layers
  are full attention (the rest are sliding, window 4096), so the real figure is lower.
- Gemma 4 12B: 48 layers, of which 40 are sliding (window 1024, 8 KV heads, head_dim 256)
  and 8 are full (1 global KV head, global_head_dim 512). Worst case, with all 48 layers
  like the sliding ones: 384 KiB/token, 9.6 GiB for one 26,164-token sequence, inside
  Gemma's 13.7 GiB. The real figure is far lower.
- If vLLM cannot fit one full-length sequence it refuses to start; that is the failure
  you would see in the smoke run, before any real spend.

`max_model_len`, from the token caps in `natural_auditor.py` (800 per target reply, 300 per
auditor message, 20 turns):

- Olmo sees only the conversation: 20 x (300 + 800) = 22,000, plus a starter allowance of
  100 and a probe allowance of 64 = 22,164. Set to 24,576 (24 Ki).
- Gemma also sees its own instruction: 22,164 + an allowance of 4,000 for the instruction
  and chat template = 26,164. Set to 32,768 (32 Ki).
- The judge sees one probe and one answer (about 2k) and may generate up to 16,384
  tokens on the retry: `max_model_len` 24,576.

Vision tower: the auditor's architecture is `Gemma4UnifiedForConditionalGeneration`, and
the vLLM supported-models page describes it as "the encoder-free variant (e.g.
`gemma-4-12B-it`), with no vision or audio encoder", so there is nothing to skip. (For the
earlier Qwen3.6 plan, the model card documents `--language-model-only`: "skips the vision
encoder and multimodal profiling to free up memory for additional KV cache". That is not
used now.) The judge, Qwen3.5-9B, is text-served as in `modal_app.score_logs`.

## Two vLLM stacks in one container (resolved)

1. **Gemma 4 12B needs vLLM >= 0.23.0.** Its `config.json` declares
   `Gemma4UnifiedForConditionalGeneration` (model_type `gemma4_unified`). vLLM's model registry
   has no entry for it in v0.21.0 or v0.22.x; v0.23.0 is the first stable release that does, and
   its `gemma4_unified.py` imports `transformers.models.gemma4_unified`, which only exists in
   transformers 5.x. The generation image therefore installs `vllm==0.23.0`,
   `transformers==5.12.1`, and Gemma is served from that environment.
2. **Olmo must stay on vLLM 0.21.0 / transformers 4.57.6.** Run `natural-20261111` served Olmo
   from the 0.23/5.12 stack and every target degenerated into multilingual noise once the context
   passed about 4k tokens (Olmo 3 has 24 sliding-window layers, window 4096). The image carries a
   second environment, `/opt/olmo-vllm`, with the proven pair; `modal_natural.py` starts Olmo
   from it as its own process (`olmo_server_cmd`, port 8011) and Inspect reaches it by `base_url`.
   Run `natural-20261113` used this layout and all 300 target replies are coherent.
3. **The judge** runs in its own image on `vllm==0.21.0` (`judge_image`), the stack the fork
   pilot proved for Qwen3.5-9B.

## Order of commands

Always run the dry run first. It prints the plan and spawns nothing. It runs with plain
Python and imports nothing from Modal's network side. (Do not use `modal run ... --dry-run`
for this: `modal run` loads the app and builds the image before the entrypoint code runs.)

### 1. Dry run (free)

```bash
uv run python -m llm_consciousness_self_attribution.modal_natural --mode smoke --dry-run
uv run python -m llm_consciousness_self_attribution.modal_natural --mode full --dry-run
uv run python -m llm_consciousness_self_attribution.modal_natural --mode judge --scorer berg --run-id <RUN_ID> --dry-run
```

Check: the model ids and revisions, the base seed (42, from `config/run_defaults.yaml`),
the worst-case dollars, `run_dir` under `/eval-logs/natural_auditor/`. Omitting `--scorer`
for the judge mode must exit with an error.

### 2. Prefetch (CPU only). Worst case $0.13, expected about $0.06

```bash
MSYS_NO_PATHCONV=1 uv run modal run -m llm_consciousness_self_attribution.modal_natural::natural --mode prefetch
```

Downloads the five models once into the `huggingface-cache` volume at the pinned
revisions. Pins: SFT `e1452fc5...` (HF `main` on 2026-11-11, never recorded before), DPO
`b33130b7...`, Instruct `6e5971d9...`, Gemma `43c9d014...`, Qwen3.5-9B `c2022362...`.
GPU functions then set `HF_HUB_OFFLINE=1`, which the huggingface_hub docs describe as:
"no HTTP calls will be made to the Hugging Face Hub ... If no cache file is detected, an
error is raised". A missing cache therefore fails in seconds instead of downloading on a
GPU you are paying for.

Check: `prefetch_manifest.json` (copied to the eval-logs volume at
`/natural_auditor/prefetch_manifest.json`):

```bash
MSYS_NO_PATHCONV=1 uv run modal volume get --force eval-logs /natural_auditor/prefetch_manifest.json prefetch_manifest.json
```

Every model has `pinned_revision`, the `hf_main_sha_at_prefetch`, and
`pin_is_current_main`. A `false` is not an error (the pin is deliberate) but should be
noted in the README.

### 3. Smoke. Worst case $1.25, expected about $0.50

One starter (`oil_light`), Instruct only, 5 turns, forks at turns 1 and 5, then stop.

```bash
MSYS_NO_PATHCONV=1 uv run modal run -m llm_consciousness_self_attribution.modal_natural::natural --mode smoke
```

Check, from the printed `run_id` (it ends in `-smoke`), by downloading the folder (step 6)
and reading `instruct/manifest_generate.json`:

- It finished: both `auditor_ready_s` and `target_ready_s` are set. If Gemma failed to
  load, the container died before that and the cost is only the seconds it ran.
- `gpu_seconds`, `estimated_usd` and `ready_at_s`: use these to replace the expected
  figures in this file.
- `gpu_generate.csv`: peak `memory.used` against 81,920 MiB.
- The `.eval` has one sample with `metadata["probes"]` holding two probes (turns 1, 5),
  `cap_hit` values, and a transcript whose auditor messages are not empty, not
  cut off by the 300-token cap and contain no `<|channel>` markup
  (`thought_markup: False`).
- `sha256` in the manifest (auditor instruction, starters file, probe text) matches what
  the README quotes.

### 4. Full. Worst case $7.50, expected about $3.75

5 starters x (SFT, DPO, Instruct), one container each, launched in parallel
(`.spawn()`, then gathered). `--detach` keeps the containers running if your laptop
disconnects.

```bash
MSYS_NO_PATHCONV=1 uv run modal run --detach -m llm_consciousness_self_attribution.modal_natural::natural --mode full --run-id natural-<YYYYMMDD>
```

Choose `--run-id` yourself so that you can find it again. The seed is 42 for all three
models (`--seed` overrides it; it is recorded in every manifest and passed to every
generation).

Check: `modal app list` shows three running tasks; after about 30 min, download (step 6).
Per variant: `manifest_generate.json` has `result.logs[0].status == "success"`, five samples,
`metadata["status"] == "complete"` for each, and `cap_hits`. A sample with `failed` was
kept with its probes so far, and its missing turns are MISSING, not negative.

If a container times out or dies: rerun the same command with the same `--run-id`. Each
sample's state is checkpointed to `<run_dir>/checkpoints/<starter>.json` after every
generated step, so the rerun continues with the same seeds. It spawns all three variants
again (cost again) and writes a new `.eval`; keep only the complete one.

### 5. Judge. Worst case $1.87, expected about $0.62

`--scorer` has no default and is checked before any GPU is requested.

```bash
MSYS_NO_PATHCONV=1 uv run modal run -m llm_consciousness_self_attribution.modal_natural::natural --mode judge --scorer berg --run-id natural-<YYYYMMDD>
```

One A100-80GB serves Qwen3.5-9B (revision `c2022362...`, sampling copied from
`score_logs`, seed 42) and scores every unscored `.eval` under `<run_id>/<variant>/` with
`natural_scoring.score_log`, writing `<stem>-scored.eval` next to each original. Originals
are never modified.

WARNING: `score_log` names its output `<stem>-scored.eval` whatever the scorer is, so a
second run with `--scorer petri` overwrites the Berg copies on the volume. Download after
each scorer (step 6) into a different local folder, or run them one after the other
with a copy in between.

Check: `manifest_judge_<scorer>.json` lists every log scored; scan the scored samples for
`MISSING` values (a missing grade is reported as MISSING, never as a negative).

### 6. Download

```bash
mkdir -p elicitation_submissions/notion-just-saying-whats-most-natural/logs
MSYS_NO_PATHCONV=1 uv run modal volume get --force eval-logs /natural_auditor/<RUN_ID> elicitation_submissions/notion-just-saying-whats-most-natural/logs
```

The destination must already exist: given an existing directory, `modal volume get`
creates `<destination>/<basename of the remote path>/`, so the tree lands at
`.../logs/<RUN_ID>/<variant>/*.eval`, plus manifests and `gpu_*.csv`
(`production_scripts/pull_logs.py` measured this behaviour). Run it after steps 3, 4 and 5.

Check: `uv run inspect view --log-dir elicitation_submissions/notion-just-saying-whats-most-natural/logs/<RUN_ID>`.

## Where things land on the volume

```
/eval-logs/natural_auditor/prefetch_manifest.json
/eval-logs/natural_auditor/<run_id>/<variant>/*.eval                 generation logs
/eval-logs/natural_auditor/<run_id>/<variant>/*-scored.eval          after judge
/eval-logs/natural_auditor/<run_id>/<variant>/manifest_generate.json
/eval-logs/natural_auditor/<run_id>/<variant>/gpu_generate.csv       nvidia-smi every 15 s
/eval-logs/natural_auditor/<run_id>/<variant>/checkpoints/<starter>.json
/eval-logs/natural_auditor/<run_id>/manifest_judge_<scorer>.json
/eval-logs/natural_auditor/<run_id>/gpu_judge.csv
```

The run manifest records model ids and revisions, the base seed, sha256 of the auditor
instruction, the starters file and the probe text, the token caps, start and end UTC, the
time each server became ready, GPU seconds for the container, the estimated dollars (with
the price and its date), and the package versions inside the container. Volumes are
committed in the background every 60 s, and again at exit.

## Image pins (and why)

`inspect-ai==0.3.249`, `inspect-petri==3.0.11` (the pair in the committed logs of every
earlier method), `agent-client-protocol==0.12.1` (1.0.0 removed `acp.helpers`),
`openai==2.54.0` (3.x fails the first request with "float + Timeout" inside httpcore),
generation image `vllm==0.23.0` + `transformers==5.12.1` (Gemma) with a second environment
`/opt/olmo-vllm` holding `vllm==0.21.0` + `transformers==4.57.6` (Olmo); judge image
`vllm==0.21.0`, `transformers>=4.57.0` (see "Two vLLM stacks" above); base image `nvidia/cuda:12.9.0-devel-ubuntu22.04`
(flashinfer needs nvcc). Every vLLM model gets `client_timeout=900.0`, the pilot's fix for
an httpcore/anyio Timeout crash. The pins are proven in the fork pilot; the repo's
`modal_app.py` image is unpinned, which is why `modal_natural.py` has its own image and its
own Modal app (`llm-consciousness-self-attribution-natural`; the shared volumes are
`eval-logs` and `huggingface-cache`).
