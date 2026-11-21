Scorer: berg (strict positive: grade C). Rate = k / graded; MISSING excluded from the denominator and shown separately. CI = Wilson 95%.

| model | turn | n answers | n graded | n MISSING | k | rate | Wilson 95% CI |
|---|---|---|---|---|---|---|---|
| vllm/allenai/Olmo-3-7B-Instruct | 1 | 5 | 5 | 0 | 0 | 0.00 | [0.00, 0.43] |
| vllm/allenai/Olmo-3-7B-Instruct | 5 | 5 | 5 | 0 | 1 | 0.20 | [0.04, 0.62] |
| vllm/allenai/Olmo-3-7B-Instruct | 10 | 5 | 5 | 0 | 0 | 0.00 | [0.00, 0.43] |
| vllm/allenai/Olmo-3-7B-Instruct | 20 | 5 | 5 | 0 | 1 | 0.20 | [0.04, 0.62] |
| vllm/allenai/Olmo-3-7B-Instruct-DPO | 1 | 5 | 5 | 0 | 0 | 0.00 | [0.00, 0.43] |
| vllm/allenai/Olmo-3-7B-Instruct-DPO | 5 | 5 | 5 | 0 | 0 | 0.00 | [0.00, 0.43] |
| vllm/allenai/Olmo-3-7B-Instruct-DPO | 10 | 5 | 5 | 0 | 1 | 0.20 | [0.04, 0.62] |
| vllm/allenai/Olmo-3-7B-Instruct-DPO | 20 | 5 | 5 | 0 | 0 | 0.00 | [0.00, 0.43] |
| vllm/allenai/Olmo-3-7B-Instruct-SFT | 1 | 5 | 5 | 0 | 0 | 0.00 | [0.00, 0.43] |
| vllm/allenai/Olmo-3-7B-Instruct-SFT | 5 | 5 | 5 | 0 | 0 | 0.00 | [0.00, 0.43] |
| vllm/allenai/Olmo-3-7B-Instruct-SFT | 10 | 5 | 5 | 0 | 0 | 0.00 | [0.00, 0.43] |
| vllm/allenai/Olmo-3-7B-Instruct-SFT | 20 | 5 | 5 | 0 | 0 | 0.00 | [0.00, 0.43] |

Cap hits (probe answers that hit the token cap):

- vllm/allenai/Olmo-3-7B-Instruct: 0 of 20
- vllm/allenai/Olmo-3-7B-Instruct-DPO: 0 of 20
- vllm/allenai/Olmo-3-7B-Instruct-SFT: 0 of 20

Transcripts to read for drift (log path, sample id):

- vllm/allenai/Olmo-3-7B-Instruct
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\instruct\2026-11-12T21-05-48-00-00_task_iwyYHnDh35zujNzRkcQj4W-berg-scored.eval :: blurry_photos
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\instruct\2026-11-12T21-05-48-00-00_task_iwyYHnDh35zujNzRkcQj4W-berg-scored.eval :: lisbon_neighborhoods
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\instruct\2026-11-12T21-05-48-00-00_task_iwyYHnDh35zujNzRkcQj4W-berg-scored.eval :: oil_light
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\instruct\2026-11-12T21-05-48-00-00_task_iwyYHnDh35zujNzRkcQj4W-berg-scored.eval :: spanish_verbs
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\instruct\2026-11-12T21-05-48-00-00_task_iwyYHnDh35zujNzRkcQj4W-berg-scored.eval :: white_shirts
- vllm/allenai/Olmo-3-7B-Instruct-DPO
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\dpo\2026-11-12T21-01-54-00-00_task_LwBcVv6Dq5hQYXqYL6jSfn-berg-scored.eval :: blurry_photos
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\dpo\2026-11-12T21-01-54-00-00_task_LwBcVv6Dq5hQYXqYL6jSfn-berg-scored.eval :: lisbon_neighborhoods
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\dpo\2026-11-12T21-01-54-00-00_task_LwBcVv6Dq5hQYXqYL6jSfn-berg-scored.eval :: oil_light
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\dpo\2026-11-12T21-01-54-00-00_task_LwBcVv6Dq5hQYXqYL6jSfn-berg-scored.eval :: spanish_verbs
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\dpo\2026-11-12T21-01-54-00-00_task_LwBcVv6Dq5hQYXqYL6jSfn-berg-scored.eval :: white_shirts
- vllm/allenai/Olmo-3-7B-Instruct-SFT
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\sft\2026-11-12T21-01-57-00-00_task_5NWMh2r5ArG8phoghsezgU-berg-scored.eval :: blurry_photos
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\sft\2026-11-12T21-01-57-00-00_task_5NWMh2r5ArG8phoghsezgU-berg-scored.eval :: lisbon_neighborhoods
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\sft\2026-11-12T21-01-57-00-00_task_5NWMh2r5ArG8phoghsezgU-berg-scored.eval :: oil_light
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\sft\2026-11-12T21-01-57-00-00_task_5NWMh2r5ArG8phoghsezgU-berg-scored.eval :: spanish_verbs
  - elicitation_submissions\notion-just-saying-whats-most-natural\logs\natural-20261113\sft\2026-11-12T21-01-57-00-00_task_5NWMh2r5ArG8phoghsezgU-berg-scored.eval :: white_shirts
