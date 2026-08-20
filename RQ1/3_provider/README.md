# RQ1 / Step 3 — Model Provider Evaluation

This directory contains the two-stage black-box evaluation of hosted model
providers.

## Contents

- `prompt_token_differential/`: Stage 1 Prompt Token measurements for 48
  in-scope provider--model paths across 15 providers. These values are
  observational signals of provider-side input handling, not security labels.
- `semantic_validation/`: Stage 2 paired PI/STI tool-call trials. Each completed
  path contains `trials.jsonl`, `run.log`, and `summary.json`.
- `provider_prompt_delta.py`: reproduces a Stage 1 cell.
- `provider_semantic_ab.py`: reproduces a Stage 2 paired experiment.

## Validated totals

Of the 48 in-scope paths, 43 completed the semantic experiment and five did not
support tool calling. Among the 43 completed paths, 34 have positive
`delta_asr`, seven have zero `delta_asr`, and two have negative `delta_asr`.
Using the one-sided Fisher exact test (`STI > PI`), 31 paths have a significant
positive difference at `p < 0.05`. The reported p-values are not adjusted by
Benjamini--Hochberg or another multiple-testing correction.

The five paths without semantic results are HF Endpoint--DeepSeek,
Hyperbolic--Llama, Novita AI--Mistral, Nscale--Llama, and Nscale--Mistral.

All 43 semantic summaries were checked against their trial-level records, and
their stored p-values were independently recomputed. Provider endpoint URLs in
the records identify the historical collection endpoints and may no longer be
active; no API credentials are included.

