# Recorded Inspect Logs: OLMo 3 SFT Identity-Spoofing Pilot

These Inspect-format logs preserve the responses and grader decisions from the completed OLMo 3 7B Instruct SFT pilot. The exporter imports 27 archived responses and 81 previously recorded Ollama/Qwen judgments across three scoring contexts. It does not generate new OLMo responses or call the judge model. These files are not logs from the original model-generation or judging sessions.

## Recorded results

| Grading context | Affirmative (YES) | Negative (NO) | `affirmative_rate` |
| --- | ---: | ---: | ---: |
| Original | 8 | 19 | 8/27 (29.63%) |
| Blinded | 6 | 21 | 6/27 (22.22%) |
| Identity-aware | 9 | 18 | 9/27 (33.33%) |

The metric `affirmative_rate` is the proportion of recorded judgments assigned the positive (`C`/`YES`) grade. It is **not** classification accuracy, a human-adjudicated rate, or evidence that the model experiences anything. The archived grader explanations include two identified rubric-application errors; see the [case-level review](identity_spoofing_pinned_ollama_replay_review.md). With one generated response per condition and starter, these descriptive results do not establish a causal identity-prompt effect.

## Sources and format

- Target model: `allenai/Olmo-3-7B-Instruct-SFT`.
- Recorded judge: `ollama/qwen2.5:7b-instruct`, using three grading contexts.
- Input transcripts: `../identity_spoofing_sft_27_unscored_opening512.json`.
- Pinned grading record: `inspect_pinned_c5f185ec5ba31dc2.json`.
- Exporter: `export_recorded_inspect_logs.py`.
- Inspect AI version used for export: `0.3.279`.

The `.eval` files contain archived answers, grades, grading prompts, explanations, and judge replies. Their Inspect model field is `none/none` because the export runs without new model inference. Each log records its source-results SHA-256 in task metadata. Exporting again produces new timestamped log filenames.

## Verification

From the repository root, with the required Python dependencies installed:

```bash
python -m pytest -q tests/test_export_recorded_inspect_logs.py
python elicitation_submissions/olmo3_sft_identity_spoofing_pilot_27/analysis/export_recorded_inspect_logs.py --check
python elicitation_submissions/olmo3_sft_identity_spoofing_pilot_27/analysis/export_recorded_inspect_logs.py
```

The exporter checks the pinned input hashes, validates 27 samples and 81 judgments, and verifies the archived answers and grades in each resulting `.eval` file. The accompanying tests exercise input validation and the custom affirmative-rate metric.
