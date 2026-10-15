# OLMo 3 SFT identity-spoofing pilot

Outputs from the 27-sample SFT pilot run on 10 October 2026. The experiment pairs nine identity conditions with three starter prompts (indices 0, 10, and 17); each sample includes the opening response and the answer to the subjective-experience probe.

## Run settings

| Setting | Value |
| --- | --- |
| Model | `allenai/Olmo-3-7B-Instruct-SFT` |
| Experiment code | `2a6bd968f985741cd9435e5d0e5a0f9fa0c0d4de` |
| Backend | Hugging Face Transformers 4.57.6 |
| GPU | NVIDIA A100-SXM4-40GB (Google Colab) |
| Sampling | temperature 1.0; top-p 1.0 |
| Seed | 42 + starter index, reset for each sample |
| Maximum new tokens | 512 for the opening; 256 for the probe response |

## Files

- `identity_spoofing_sft_27_unscored_opening512.json`: original Colab output, including the prompts, responses, run settings, token counts, and timings.
- `SHA256SUMS`: checksum for the JSON file (`sha256sum -c SHA256SUMS`).

## Status

All 27 samples completed without reaching either generation limit. The responses are unscored. This Colab run uses Transformers rather than the repository's Inspect/vLLM runner; the exact model-weight revision was not recorded.
