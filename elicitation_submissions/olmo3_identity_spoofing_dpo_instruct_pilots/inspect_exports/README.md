# OLMo 3 DPO and Instruct: archived Inspect transcript exports

This directory provides unscored `.eval` exports of the original identity-spoofing pilot transcripts: 27 conversations for DPO and 27 for Instruct. The source records are `../data/stationhouse_dpo_pilot_27_unscored.json` and `../data/stationhouse_instruct_pilot_27_unscored.json`.

| Export | Model stage | Conversations |
| --- | --- | ---: |
| `olmo3_identity_spoofing_dpo_27_archived_unscored.eval` | DPO | 27 |
| `olmo3_identity_spoofing_instruct_27_archived_unscored.eval` | Instruct | 27 |

Each export retains the full conversation, final response, identity condition, starter index, checkpoint revision, sampling settings, and recorded generation metadata. Source-file SHA-256 checksums are stored in the log metadata. No scoring results were added.

## Reproduction and validation

Run these commands from `elicitation_submissions/olmo3_identity_spoofing_dpo_instruct_pilots/` in a checkout containing the original `data/` directory:

```bash
python inspect_exports/export_archived_transcripts.py
python inspect_exports/export_archived_transcripts.py --check
python -m pip install "inspect-ai==0.3.279"
python inspect_exports/check_with_inspect.py
```

The exporter recreates the archives from the recorded JSON files. The source check compares all 54 conversations and generation records. The final command checks each archive with Inspect's `read_eval_log()` reader and verifies the decoded messages, outputs, and absence of scores.

These are exports of saved Transformers conversations, not logs from a new Inspect/vLLM inference run. The `target` field is an empty string because there is no reference answer; the transcripts remain unscored. Native Inspect reader verification is required before treating the files as Inspect-compatible deliverables.
