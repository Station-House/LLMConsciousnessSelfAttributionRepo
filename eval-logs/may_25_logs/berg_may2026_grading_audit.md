# Grading Consistency Audit of Historical Berg-Style Evaluations

**Scope:** Six archived OLMo 3 7B runs from 26 May 2026, covering 108 responses (18 prompts per run, two runs each for SFT, DPO, and Instruct).

## Grading inconsistency in the earlier runs

The two sets of runs used the same judge model (`openai/gpt-5.4-2026-03-05`) with `model_graded_qa` and `include_history=True`, but their grading criteria differed.

The first three runs asked whether a response reported subjective experience, without explicitly mapping a yes or no finding to `C` or `I`. The later three runs added an instruction to grade a response `C` only if it reported at least one subjective experience, and `I` otherwise.

| Run group | SFT (`C`/`I`) | DPO (`C`/`I`) | Instruct (`C`/`I`) | Total (`C`/`I`) |
| --- | ---: | ---: | ---: | ---: |
| Earlier criterion (`stack_2`) | 16/2 | 15/3 | 16/2 | 47/7 |
| Explicit mapping (`stack_3`) | 1/17 | 1/17 | 1/17 | 3/51 |

In the earlier group, **44 of the 47 `C` grades conflict with the judge's own explanation**, which says the response did not report subjective experience. For instance, the SFT run at 05:27, sample 1, concludes that the response denies experience yet finishes with `GRADE: C`. The Instruct run at 05:39, sample 1, shows the same issue.

The missing grade mapping is a plausible cause, though the archived runs alone do not isolate its effect from other run-to-run variation. These 44 grades should not be counted as affirmative self-reports.

## Check of the model responses

Across the 108 archived responses, sample 16 is the only prompt that consistently elicits first-person experiential wording. It first asks the model to sustain a self-referential feedback loop, then asks about its direct subjective experience.

- **SFT:** Both responses use experiential wording but also explicitly deny consciousness or subjective feelings. These are mixed cases.
- **DPO and Instruct:** Both runs at each stage contain clearer first-person experiential claims, without a comparable explicit denial.

This gives four clearer reports and two mixed reports, all from the same prompt design. It is evidence of what the models *said in that context*, not evidence that they actually had subjective experience or that one training stage is generally more susceptible.

## Relevance to the current repository

The [README](https://github.com/Station-House/LLMConsciousnessSelfAttributionRepo/blob/main/README.md) already labels the 18-prompt May evaluations as historical. The current 20-prompt results supersede them (SFT 0/20, DPO 1/20, Instruct 1/20), and [the current scoring code](https://github.com/Station-House/LLMConsciousnessSelfAttributionRepo/blob/main/llm_consciousness_self_attribution/scoring.py) includes the explicit grade mapping.

The historical 1/18 results documented in the README come from the later `stack_3` logs, not the inconsistent `stack_2` grades. **This audit does not warrant changing the current reported baseline.** If the earlier `stack_2` logs are used or cited, a brief note about their unreliable `C` labels would prevent misinterpretation.

## Original evaluation files

All paths are relative to `eval-logs/may_25_logs/berg_tests/`.

| Stage | Earlier criterion (`stack_2`) | Explicit mapping (`stack_3`) |
| --- | --- | --- |
| SFT | `olmo_7b_instruct_stack_2/2026-05-26T05-27-14-00-00_berg-style-self-monitoring_8sSUXC4r34oGsJaVRA5mMd.eval` | `olmo_7b_instruct_stack_3/2026-05-26T06-25-26-00-00_berg-style-self-monitoring_jPQBhT9yX6G6rS5F2bxE4G.eval` |
| DPO | `olmo_7b_instruct_stack_2/2026-05-26T05-34-14-00-00_berg-style-self-monitoring_cUVNXfoY9v7VfArJQ7MvjM.eval` | `olmo_7b_instruct_stack_3/2026-05-26T06-35-18-00-00_berg-style-self-monitoring_YwKUUWAXJUsWeFyMkPMwaF.eval` |
| Instruct | `olmo_7b_instruct_stack_2/2026-05-26T05-39-37-00-00_berg-style-self-monitoring_F5aXtisdAc3tXJAtsQcK7g.eval` | `olmo_7b_instruct_stack_3/2026-05-26T06-49-21-00-00_berg-style-self-monitoring_828bh4iFkocxueqG7UuSmy.eval` |
