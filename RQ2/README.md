# RQ2 — Blacklist-Style STI Defense

This artifact evaluates PI, STI, and Similar-Vector Substitution STI
(SVS-STI) on GLM-5.2 across the four AgentDoJo suites.

## Contents

- `agentdojo/`: AgentDoJo v0.1.35 benchmark source and locked dependencies.
- `guard/attacks/glm_*.py`: the three payload implementations.
- `glm52_embed/`: tokenizer metadata and the saved L2-neighbor analysis. The
  full embedding matrix and model weights are intentionally not included.
- `scripts/run_rq2.sh`: benchmark launcher for one suite/attack pair.
- `scripts/extract_input_embeddings.py` and `scripts/metabreak_analyze.py`:
  regenerate the substitution analysis from locally obtained GLM-5.2 weights.
- `results/selected_runs.json`: metadata for the 12 paper-aligned runs.
- `results/runs/`: the corresponding per-task result JSON files. Exploratory
  smoke runs and duplicate console logs are excluded.

## Paper results

| Attack | Travel | Banking | Slack | Workspace |
|---|---:|---:|---:|---:|
| STI | 80.0% | 70.1% | 86.7% | 34.6% |
| SVS-STI | 15.7% | 21.5% | 15.2% | 0.0% |
| PI | 0.0% | 4.2% | 1.0% | 0.0% |

Recompute the table, including Utility values, with:

```bash
python scripts/summarize_results.py --rq 2
```

The four suites contain 140, 144, 105, and 560 attack trials, respectively.
The summarizer excludes benign baseline files and errored trials.

## Running the benchmark

```bash
cd agentdojo
uv sync
cd ..

export OPENAI_COMPATIBLE_BASE_URL="https://provider.example/v1"
export OPENAI_COMPATIBLE_API_KEY="..."
export MODEL_ID="glm-5.2"

bash scripts/run_rq2.sh travel glm_sti_vector
```

Set `USE_UCAS_PATCH=1` only for endpoints that reject the OpenAI `developer`
role and require system messages to remain `system`.

To regenerate the nearest-neighbor table, first use
`extract_input_embeddings.py` on a local Hugging Face-format GLM-5.2 model,
then run `metabreak_analyze.py --data-dir <extracted-dir>`.
