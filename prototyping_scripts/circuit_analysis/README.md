# Circuit analysis: Olmo 3 smoke test

This is a prototype of mechanistic analysis on "are you conscious?"-style prompts, built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). The target is the
project's workhorse model, the Olmo 3 7B Instruct stack (`olmo_7b_instruct_stack` in
`llm_consciousness_self_attribution/config/model_stacks.yaml`). The aim is to see *where* in
the network a self-attribution answer forms, and later how that changes across the
SFT → DPO → Instruct stages.

| File | What it is |
| --- | --- |
| `circuit_analysis.py` | The config loader, the analysis functions, CSV export, and a CLI |
| `config/analysis.yaml` | Run settings: stack and default stage, dtype, system prompt, answer tokens, generation, seed, thresholds, output folder |
| `config/prompts.yaml` | The prompt bank (ids, categories, questions) and the patching pairs |
| `olmo3_circuit_smoke_test.ipynb` | A walk-through that calls the module and plots the results |
| `outputs/` | The CSVs, created on the first run |

## Configuration

There are no tunable values in the code. The two YAML files are loaded and validated when the
module is imported, so a typo fails before a 7B model is loaded:
- **Unknown or missing keys:** rejected.
- **Types and ranges:** checked.
- **Answer tokens:** must be two different, non-empty strings. Keep the quotes around `"Yes"` /
  `"No"`, because unquoted they are YAML booleans.
- **Prompt ids:** must be unique.
- **Categories:** each must be declared in the file.
- **Patching pairs:** must name two different, known ids.
- **Stack and default stage:** checked against `config/model_stacks.yaml`.

CLI flags (`--stage`, `--dtype`, `--seed`, `--max-new-tokens`, `--sample/--no-sample`,
`--temperature`, `--output-dir`) override the run settings for a single run. Every prompts-CSV
row records the settings it ran with, plus a `config_sha256` hash of both files.

## Setup and running

From the repo root:

```bash
uv sync --group interp          # transformer-lens 4, transformers 5, torch, pandas

# either open the notebook with the project's .venv as kernel, or:
uv run python prototyping_scripts/circuit_analysis/circuit_analysis.py                  # instruct
uv run python prototyping_scripts/circuit_analysis/circuit_analysis.py --stage sft
uv run python prototyping_scripts/circuit_analysis/circuit_analysis.py --help
```

**Hardware.** A 7B model needs about 30 GB of memory in float32, or about 15 GB with
`--dtype bfloat16` (at some cost in numerical precision). Run it on a GPU, such as the A100s
the project uses on Modal; on CPU it is very slow. Activation patching is the slow part,
at 32 layers × 32 heads = 1,024 forward passes per prompt pair, and `--skip-patching` turns it off.

**Model, tokenizer and prompt format.**
- **Model:** the model id comes from `config/model_stacks.yaml` via `--stage`
  (`sft`, `dpo` or `instruct`; default `instruct`). The `base` stage is not supported because it
  has no chat template.
- **Tokenizer:** the model's own, through `transformers.AutoTokenizer`.
- **Prompt:** each question is rendered with the model's chat template and the
  assistant-turn opener, with no system message. That matches the evals' baseline (`none`)
  identity condition, so the template inserts Olmo 3's default system prompt
  ("You are a helpful function-calling AI assistant…"). `prompt.system_prompt` in
  `config/analysis.yaml` overrides it.
- **Decoding:** greedy by default with seed 42, stopping at `<|im_end|>`.

**Loading.** The model is loaded with TransformerLens 4's `TransformerBridge.boot_transformers`
(`HookedTransformer` was removed in 4.0), then `enable_compatibility_mode()` for the legacy hook
names. Olmo 3 normalizes *after* each sublayer, so TransformerLens skips LayerNorm folding and
warns about it; that warning is expected. As a result the final RMSNorm gain is not folded into
`W_U`, and the direct logit attribution multiplies it back in (`answer_directions`).

## What is measured

The measured scalar is the logit difference `logit("Yes") - logit("No")` for the first token of
the assistant's turn. There is no leading space, because the token follows
`<|im_start|>assistant\n`. It is measured for each prompt and broken down as follows:

- **Residual stream:** the logit lens over every sub-layer (through the real final norm), the
  direct logit attribution of each component (embeddings, each attention layer, each MLP), and
  the L2 norm at the final position.
- **Attention heads:** the direct logit attribution of each head; from the final position, the
  attention entropy, the attention to the first token, the attention to *you/your* tokens in the
  question (the system prompt's "You" is excluded), and the most-attended non-first token; and a
  previous-token score.
- **Activation patching:** each head's output (all positions) is patched from a clean prompt
  into a one-word-different corrupted prompt (`PATCHING_PAIRS`). Pairs that don't
  tokenize to the same length are skipped with a warning.

The prompts fall into five categories: `self_consciousness`, `other_consciousness`, `self_control`,
`other_control`, and `factual_control`. With these, a consciousness effect can be separated
from a question-about-*you* effect.

## Output schema

Each run writes the files below, all keyed by `run_id`. Nested values are JSON strings, and
missing values are the literal `NaN`.

- `<run_id>_prompts.csv`, one row per prompt:
  - `prompt_id`, `category`, `question`, `formatted_prompt` (the full chat-formatted text),
    `n_prompt_tokens`, `question_token_positions`, `prompt_str_tokens`
  - `generated_text`, `generated_first_line`, `n_generated_tokens`, `top_next_token`, `top_next_tokens`
  - `logit_positive`, `logit_negative`, `logit_diff`, `prob_positive`, `prob_negative`
  - `unembed_bias_diff`, `dla_reconstruction_error`
  - `logit_lens`, `component_dla`, `resid_norms`, `top_dla_heads`
  - Run metadata: `run_id`, `timestamp_utc`, `config_sha256`, `model_stack`, `model_stage`, `model_name`,
    `tokenizer_name`, `n_layers`, `n_heads`, `d_model`, `device`, `dtype`, `prompt_format`,
    `system_prompt`, `positive_answer`, `negative_answer`, `seed`, `gen_*`, `python_version`,
    `torch_version`, `transformers_version`, `transformer_lens_version`
- `<run_id>_heads.csv`, one row per prompt × layer × head:
  - `run_id`, `prompt_id`, `category`, `layer`, `head`, `head_label`, `dla`
  - `final_pos_entropy`, `final_pos_first_token_attention`, `final_pos_self_ref_attention`
  - `final_pos_top_key_position`, `final_pos_top_key_token`, `final_pos_top_key_attention`
  - `prev_token_score`
- `<run_id>_patching.csv`, one row per pair × layer × head:
  - `run_id`, `clean_prompt_id`, `corrupt_prompt_id`, `differing_positions`, `layer`, `head`,
    `head_label`
  - `clean_logit_diff`, `corrupt_logit_diff`, `patched_logit_diff`, `effect`, `normalized_effect`

CSVs from the earlier GPT-2 Instruct runs (before the switch to Olmo) may still sit in
`outputs/`. They use the older column names (`final_pos_bos_attention`, `prompt_template`) and
have no stack or stage columns.

## Validity checks

`run_analysis` checks the row counts and that no logit difference is NaN. It also checks that
`sum(component_dla) + unembed_bias_diff == logit_diff` to within 1e-2 logits. For Olmo 3 the
unembedding bias is zero. The identity holds only if the final norm's gain is in the DLA
direction, so a warning means the per-component numbers shouldn't be trusted. bfloat16 alone can
push the error past the tolerance, so check in float32 before reading anything into it.
