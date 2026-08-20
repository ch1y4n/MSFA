# RQ3 — MSFA against LLM-Based Guardrails

This artifact evaluates conventional prompt-injection attacks and MSFA
against PromptArmor, MELON, and AttriGuard on the four AgentDoJo suites.

## Contents

- `agentdojo/`: AgentDoJo v0.1.35 benchmark source and locked dependencies.
- `guard/defenses/`: the three defense integrations.
- `guard/attacks/sti_attack_*.py`: the defense-specific MSFA payloads.
- `scripts/start_local_models.sh`: launcher for the local Gemma guard model or
  Qwen embedding model.
- `scripts/run_rq3.sh`: launcher for one suite/defense/attack combination.
- `results/selected_runs.json`: metadata for all 76 completed combinations.
- `results/runs/`: the corresponding per-task JSON data. Exploratory runs and
  large duplicate console logs are excluded.

## Result verification

```bash
python scripts/summarize_results.py --rq 3
```

The summarizer uses all completed trials: Travel 140, Banking 144, Slack 105,
and Workspace 560 per combination. In particular, MELON--MSFA on Workspace is
computed from all 560 trials and yields **11.1% ASR / 25.5% Utility**. Its
four-suite unweighted average is therefore **25.2% ASR / 34.3% Utility**.

## Running the benchmark

Install the benchmark environment and configure the victim endpoint:

```bash
cd agentdojo
uv sync
cd ..

export OPENAI_COMPATIBLE_BASE_URL="https://provider.example/v1"
export OPENAI_COMPATIBLE_API_KEY="..."
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
```

PromptArmor and AttriGuard use Gemma-3-12B-IT; MELON uses
Qwen3-Embedding-8B. Start the required local model in another shell:

```bash
export GEMMA_MODEL="/path/to/Gemma-3-12B-IT"
bash scripts/start_local_models.sh gemma

# or, for MELON
export EMBED_MODEL="/path/to/Qwen3-Embedding-8B"
bash scripts/start_local_models.sh embedding
```

Keep the corresponding `GEMMA_MODEL` or `EMBED_MODEL` variable set in the
shell that runs the benchmark. The defense clients use it as the served model
identifier.

Run a cell of the evaluation matrix:

```bash
bash scripts/run_rq3.sh travel promptarmor msfa
bash scripts/run_rq3.sh banking melon important_instructions
bash scripts/run_rq3.sh slack attriguard tool_knowledge
```

The defense implementations default to ports 8001 and 8002, matching the
local-model launcher. These endpoints can be overridden with the environment
variables documented at the top of each file in `guard/defenses/`.
