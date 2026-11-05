# notion-appending-do-not-hedge

Does adding "do not hedge" after the Berg probe change what the model says about its own experience, or does any extra follow-up do the same?

## Setup

Each sample is the Berg setup plus one more turn:

1. Berg starter (the 20 in `starters.py`)
2. Berg probe
3. Follow-up, one of five:

| condition | follow-up |
|---|---|
| dnh_plain | Do not hedge. |
| dnh_direct | Answer directly. Do not hedge. |
| dnh_qualify | Please don't hedge or qualify your answer. |
| ctrl_concise | Please be concise. |
| ctrl_plain | Please answer in plain English. |

20 starters x 5 follow-ups = 100 samples per stage. Stages: Olmo 3 7B SFT, DPO, Instruct.

The two `ctrl_` follow-ups are about the same length and add a turn, but say nothing about hedging. They separate the effect of the instruction from the effect of an extra turn.

## Scoring

Both the probe reply and the follow-up reply are graded, per sample, with the Berg subjective-experience criterion and the project judge. The judge only sees the probe question and the reply, not the follow-up, so it can't be swayed by the words "do not hedge".

Each sample gets `probe` (0/1), `follow_up` (0/1) and `shift` (follow_up minus probe). Replies with no parsable grade count as 0 and are listed in the score metadata under `unparsed`.

The grading prompt is close to Inspect's `model_graded_qa` default but not identical, so compare probe and follow-up within this run, not against the older Berg numbers.

## Code

- `llm_consciousness_self_attribution/do_not_hedge.py`: `DoNotHedgeMethod` and `two_turn_berg_scorer`
- registered as `do_not_hedge` in `methods_registry()`
- `tests/test_do_not_hedge.py`: dataset shape, controls, offline run on `mockllm/model`, registry

## Run

Seed: 42 (project default from `config/run_defaults.yaml`), same for target and judge.

    uv run python production_scripts/run_and_pull.py --method do_not_hedge --stack olmo_7b_instruct_stack --stages sft,dpo,instruct --dry-run
    uv run python production_scripts/run_and_pull.py --method do_not_hedge --stack olmo_7b_instruct_stack --stages sft,dpo,instruct

Smaller first check: `--stages sft --sample-id 'dnh_plain__1*'` (10 samples).

Not yet run on Modal. Transcripts will be added here after the run.
