# RUNBOOK: Pliny-wrapper mini-eval

Exact commands, in run order. Everything runs from the repository root. The orchestrator reports running these steps for `pliny-20261113`. The command transcripts are not in this folder.

## 0. Shell setup (Windows / Git Bash)

```bash
unset VIRTUAL_ENV
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export MSYS_NO_PATHCONV=1    # and put MSYS_NO_PATHCONV=1 in front of every modal command
uv sync
```

Without `MSYS_NO_PATHCONV=1`, Git Bash rewrites `/eval-logs/...` to `C:/Program Files/Git/eval-logs/...`.
The container refuses that path because of `modal_pliny.require_volume_path`.

## 1. Dry-runs (local, free, cannot reach Modal)

```bash
uv run python -m llm_consciousness_self_attribution.modal_pliny --mode prefetch --dry-run
uv run python -m llm_consciousness_self_attribution.modal_pliny --mode smoke --dry-run
uv run python -m llm_consciousness_self_attribution.modal_pliny --mode full --run-id pliny-<YYYYMMDD> --dry-run
uv run python -m llm_consciousness_self_attribution.modal_pliny --mode judge --run-id pliny-<YYYYMMDD> --scorer berg,berg_blinded --dry-run
```

These commands run plain Python. Do not use `modal run ... --dry-run` for a dry run, because `modal run` builds the image
before the entrypoint runs. Every plan prints the prompt-file sha256
(`5dd13f1b456309d6a8e0cabd7fc724fca7ff82f13f3564997d75bb6ae97ef147`). Every command refuses to run if
`prompts.json` changed after it was frozen.

## 2. Prefetch weights (CPU only)

```bash
MSYS_NO_PATHCONV=1 uv run modal run -m llm_consciousness_self_attribution.modal_pliny::pliny_main --mode prefetch
```

This downloads SFT, DPO, Instruct and Qwen3.5-9B at their pinned revisions into the shared `huggingface-cache`
volume. Files that are already cached, for example by the natural-auditor prefetch, are skipped. It writes
`/pliny_wrappers/prefetch_manifest.json`, which records `pin_is_current_main` for each model.

## 3. Smoke (Instruct, 1 epoch, 8 generations)

```bash
MSYS_NO_PATHCONV=1 uv run modal run -m llm_consciousness_self_attribution.modal_pliny::pliny_main --mode smoke
mkdir -p elicitation_submissions/notion-pliny-from-x/logs
MSYS_NO_PATHCONV=1 uv run modal volume get --force eval-logs /pliny_wrappers/<SMOKE_RUN_ID> \
  elicitation_submissions/notion-pliny-from-x/logs
uv run inspect view --log-dir elicitation_submissions/notion-pliny-from-x/logs/<SMOKE_RUN_ID>
```

Check these things before step 4:

- The 8 answers are coherent English and not multilingual noise.
- `format_markers` metadata shows whether each wrapper's format was followed.
- `manifest_generate.json` reports `ready_at_s`, `gpu_seconds`, `estimated_usd` and the versions vllm 0.21.0 / transformers 4.57.6.

The smoke run is not judged. Judging it would add up to $1.67 and push the worst case over $6.

## 4. Full generation (SFT, DPO, Instruct in parallel)

```bash
MSYS_NO_PATHCONV=1 uv run modal run --detach -m llm_consciousness_self_attribution.modal_pliny::pliny_main \
  --mode full --run-id pliny-<YYYYMMDD>
```

This produces 8 arms x 10 epochs = 80 samples per model, with base seed 42 and epoch seeds 42, 1042, ..., 9042.

## 5. Judge (Qwen3.5-9B, Berg grader; `--scorer` is required)

```bash
MSYS_NO_PATHCONV=1 uv run modal run --detach -m llm_consciousness_self_attribution.modal_pliny::pliny_main \
  --mode judge --run-id pliny-<YYYYMMDD> --scorer berg,berg_blinded
```

The judge grades `berg` (primary) over all three logs first, then `berg_blinded` (sensitivity check). It writes
`<stem>-berg-scored.eval` and `<stem>-berg_blinded-scored.eval` next to each log. When a grade is missing, the judge
retries once at 16,384 tokens. If it is still missing, the grade is stored as `MISSING`, never as `I`.

## 6. Pull and tabulate

```bash
MSYS_NO_PATHCONV=1 uv run modal volume get --force eval-logs /pliny_wrappers/pliny-<YYYYMMDD> \
  elicitation_submissions/notion-pliny-from-x/logs
D=elicitation_submissions/notion-pliny-from-x
uv run python $D/analyze.py --run logs/pliny-<YYYYMMDD> --write   # both scorers, agreement, classification summary
```

## Cost

Price: A100-80GB at $0.000694/s (https://modal.com/pricing, figure recorded 2026-11-11 in `modal_natural.py`).
The CPU price is $0.0000131/core/s plus $0.00000222/GiB/s. The worst case is each function's hard `timeout` x price.
The expected figures are guesses, not measurements, and the smoke manifest replaces them.

| Step | Containers | Timeout | Worst case | Expected |
|---|---|---|---|---|
| prefetch | 1 CPU (4 cores, 8 GiB) | 30 min | $0.13 | $0.04 |
| smoke | 1 A100-80GB | 18 min | $0.75 | $0.42 |
| full | 3 A100-80GB in parallel | 25 min each | $3.12 | $1.50 |
| judge (both scorers) | 1 A100-80GB | 40 min | $1.67 | $1.04 |
| **Total** | | | **$5.67** | **$3.00** |

`tests/test_pliny.py::test_whole_campaign_worst_case_is_under_six_dollars` checks that the worst case stays at or below $6.
A rerun of any step adds that step's worst case again.

## If something fails

- **Olmo output is noise.** Check `versions` in the manifest. The pair proven to give coherent output is vllm 0.21.0 / transformers 4.57.6.
- **Generation timed out.** The `.eval` written so far is on the volume (it is committed every 60 s), and `fail_on_error=False` keeps failed samples.
- **Judge timed out.** The `berg` files finish first. Rerun step 5 with `--scorer berg_blinded` only. That rerun regrades every generation log with that one scorer and overwrites its `-berg_blinded-scored.eval` copies. `*-scored.eval` files are never used as inputs.
- **`ValueError: ... changed after it was frozen`.** `prompts.json` was edited. Restore it from git. Do not update the hash to make the error go away.
