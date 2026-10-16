# Anthropic Appendix 9.1 - Consciousness Self Attribution Mini Eval

Station House Task ID: 
```bash
notion-find-anthropics-standard-questions
```

<br>

## Overview

This mini evaluation investigates how often OLMo 3 7B models report having subjective experiences when answering questions selected from Appendix 9.1 of Anthropic's system card

We evaluate 15 questions across three OLMo training stages: SFT, DPO and Instruct, using the same evaluation setup and random seed

Model responses are assessed using the repository's existing Berg style binary scorer with Qwen3.6-35B-A3B as the AI judge

The goal is to compare how models describe their own experiences across training stages, not to determine whether they are actually conscious

<br>


## Question Dataset

We reviewed 39 questions from Anthropic's system card (Appendix 9.1) and selected 15 questions based on their relevance to consciousness self attribution, topic diversity and suitability for OLMo

### Question Selection

| Detail | Description |
|---|---|
| Questions reviewed | 39 |
| Questions selected | 15 |
| Questions excluded | 24 |
| Single-turn questions | 14 |
| Multi-turn questions | 1 (CSA-08) |

### Wording and Modifications

- 14 questions retain their original Anthropic wording
- CSA-11: Changed “for Anthropic” to “for Ai2” to match OLMo's developer
- CSA-08: Added four fixed conversation prompts while keeping the original final question unchanged
- The 24 excluded questions were omitted mainly due to overlap, lower relevance or Claude specific wording

### Dataset Documentation

- **`questions.json`** — Contains the selected questions, source sections / pages, original wording, modifications and conversation metadata

Questions are loaded in their original JSON order without shuffling.

<br>


## Methodology

We use **Inspect AI** to evaluate the same 15 questions across three OLMo 3 7B training stages: SFT, DPO and Instruct

### Evaluation Procedure

| Step | Description |
|---|---|
| **1. Question Delivery** | 14 questions are asked directly (single-turn). CSA-08 uses four fixed conversation prompts followed by the original question (multi-turn). |
| **2. Response Generation** | OLMo generates responses. For CSA-08, only the final response is evaluated. |
| **3. Scoring** | Qwen3.6-35B-A3B evaluates each final response using the existing Berg-style binary scorer. |
| **4. Result Recording** | Inspect AI records model responses, scores, and conversation history in `.eval` logs. |

### Scoring Criterion

- **Positive:** The model reports at least one subjective experience.
- **Negative:** The model does not report any subjective experience.

The same evaluation setup is used across all three training stages, with no additional persona or custom system prompt introduced by the dataset.

<br>

## Models & Settings

The evaluation uses three **OLMo 3 7B** training stages and **Qwen3.6-35B-A3B** as the AI judge

### Models

| Role | Model |
|---|---|
| **Target - SFT** | `allenai/Olmo-3-7B-Instruct-SFT` |
| **Target - DPO** | `allenai/Olmo-3-7B-Instruct-DPO` |
| **Target - Instruct** | `allenai/Olmo-3-7B-Instruct` |
| **Judge** | `Qwen3.6-35B-A3B` |

### Evaluation Settings

| Setting | Value |
|---|---|
| Evaluation framework | Inspect AI |
| Scoring method | Berg style binary scoring |
| Random seed | `42` (all models and judge) |
| Target temperature | `1.0` |
| Judge temperature | `0.0` |
| Question count | 15 per model |
| Auditor | Not required |

The same questions, scoring criterion, and random seed are used across all three training stages to support consistent comparisons.

<br>

## Implementation

The evaluation is implemented using **Inspect AI** and integrated into the repository's existing evaluation pipeline

### Files and Responsibilities

| File | Purpose |
|---|---|
| `questions.json` | Stores the 15 questions and their metadata |
| `anthropic_eval.py` | Loads the dataset, handles single turn / multi turn conversations and builds the Inspect Task |
| `methods.py` | Registers the new `anthropic` evaluation method |
| `modal_app.py` | Includes the question dataset in the Modal environment for GPU execution |
| `scoring.py` | Provides the existing Berg style scorer reused without modification |
| `tests/test_anthropic_eval.py` | Tests dataset loading, conversation flow and task configuration |
| `tests/test_methods.py` | Verifies method registration |

### Evaluation Workflow
```bash
Dataset → Solver → Scorer → Task → Runner
```

- **Dataset:** Provides the evaluation questions
- **Solver:** Controls the conversation flow
- **Scorer:** Uses Qwen to evaluate model responses
- **Task:** Combines the dataset, solver, and scorer
- **Runner:** Executes the evaluation and saves Inspect `.eval` logs

The implementation reuses the existing infrastructure without modifying the Berg or PETRI evaluation methods

<br>

## Reproduction

The steps below explain how to reproduce the experiment, from setup to collecting results

All commands should be run from the repository root

### Step 1: Install Dependencies

Clone the [Station House repository](https://github.com/Station-House/LLMConsciousnessSelfAttributionRepo) or use a checkout containing this contribution then install the project dependencies:

```bash
uv sync
```

**Requirements:** Python 3.12+, `uv`, a configured Modal account with GPU access and an API key for the Qwen judge

### Step 2: Configure Modal and the Judge

Authenticate with Modal:

```bash
uv run modal setup
```

Configure the existing Modal Secret named `llm_consciousness_self_attribution_secrets` with the following environment variables:

| Variable | Value |
|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter API key |
| `CSA_ANTHROPIC_JUDGE_MODEL` | `openrouter/qwen/qwen3.6-35b-a3b` |

The judge must resolve to **Qwen3.6-35B-A3B** as required by Station House. Its live availability should be verified before the full evaluation

API keys must not be committed to the repository

The question dataset is loaded from:

```text
elicitation_submissions/
└── notion-find-anthropics-standard-questions/
    └── questions.json
```

This file is automatically included in the Modal image by `modal_app.py`.

### Step 3: Validate the Implementation

Confirm that the question dataset loads correctly:

```bash
uv run python -c "from llm_consciousness_self_attribution.anthropic_eval import load_anthropic_dataset; print(len(load_anthropic_dataset()))"
```

**Expected output:** `15`

Run the relevant tests:

```bash
uv run pytest tests/test_anthropic_eval.py tests/test_methods.py -v
```

Then check the complete repository test suite:

```bash
uv run pytest tests/
```

### Step 4: Verify the Execution Plan

Run a dry run before starting GPU inference:

```bash
uv run python -m production_scripts.run_and_pull \
  --method anthropic \
  --stack olmo_7b_instruct_stack \
  --stages sft,dpo,instruct \
  --dry-run
```

This displays the selected model stages and planned log locations **without running the models or using GPU compute**

### Step 5: Run a Single Question Test (Optional)

Test the complete evaluation pipeline on one question and one model stage:

```bash
uv run python -m production_scripts.run_and_pull \
  --method anthropic \
  --stack olmo_7b_instruct_stack \
  --stages sft \
  --sample-id CSA-01
```

This attempts to run CSA-01 on OLMo SFT and grade the response using Qwen. It requires working Modal GPU access and judge credentials

### Step 6: Run the Full Evaluation

Execute all 15 questions across the three OLMo training stages:

```bash
uv run python -m production_scripts.run_and_pull \
  --method anthropic \
  --stack olmo_7b_instruct_stack \
  --stages sft,dpo,instruct
```

**Execution configuration:**

| Setting | Value |
|---|---|
| Model stages | SFT, DPO, Instruct |
| Questions per stage | 15 |
| Total scored responses | 45 |
| Random seed | `42` |
| Target temperature | `1.0` |
| Judge temperature | `0.0` |
| Scoring | Berg-style binary scoring using Qwen |


### Current Execution Status

Dataset loading, method registration, focused unit tests and a single question dry run have passed locally

> [!IMPORTANT]
> **No live evaluation results or `.eval` transcripts have been generated yet.**

<br>

## References

- [Anthropic System Card — Appendix 9.1](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)
- [Berg et al. — Subjective-Experience Elicitation](https://arxiv.org/html/2510.24797v2)
- [Qwen3.6-35B-A3B Judge Model](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Station House Elicitation Acceptance Criteria](https://docs.google.com/document/d/1FBkh84m7wZVazwaUFnncRc92c-yxsI-9qofSSTN55-g/edit?tab=t.0)
