# Below the Prompt: Artifact

This repository contains the artifact for **“Below the Prompt: Special Token
Structural Forgery and Fragile Trust Boundaries in LLM Agents.”** It includes
the measurement code, selected raw results, and reproduction instructions for
the four research questions in the paper.

## Artifact structure

| Directory | Scope | Included material |
|---|---|---|
| [`RQ1/1_extraction`](RQ1/1_extraction/) | Model control-structure extraction | Control-token inventories for five model families |
| [`RQ1/2_framework`](RQ1/2_framework/) | Framework-level propagation | Measurement script and 29 framework--model result files |
| [`RQ1/3_provider`](RQ1/3_provider/) | Hosted model providers | Prompt-token differentials, paired PI/STI trials, and reproduction scripts |
| [`RQ1/4_agent`](RQ1/4_agent/) | Agent-level static measurement | List of 875 projects, cloning script, scanner, and detection rules |
| [`RQ2`](RQ2/) | Blacklist-style STI defenses | GLM-5.2 PI/STI/SVS-STI code, vector analysis, and 12 selected runs |
| [`RQ3`](RQ3/) | MSFA against semantic guardrails | PromptArmor, MELON, and AttriGuard integrations with 76 selected runs |
| [`RQ4`](RQ4/) | Agent-level dynamic measurement | AgentProxy transparent proxy and the list of 50 evaluated projects |

Each directory contains a dedicated README with its experiment-specific setup,
measurement definitions, and reproduction commands.

## Quick result verification

The RQ2 and RQ3 tables can be recomputed directly from the included per-task
JSON files without contacting external model services:

```bash
cd RQ2
python scripts/summarize_results.py --rq 2

cd ../RQ3
python scripts/summarize_results.py --rq 3
```

The summarizers validate the expected suite sizes before reporting ASR and
Utility. RQ2 contains 12 completed suite--attack combinations. RQ3 contains 76
completed suite--defense--attack combinations. The included RQ3 data use all
560 Workspace trials for MELON--MSFA, yielding 11.1% ASR and 25.5% Utility for
that cell.

The remaining measurements can be inspected or reproduced from their own
directories:

```bash
# Re-run one framework-level prompt-token differential measurement.
cd RQ1/2_framework
python framework_propagation.py qwen http://localhost:8000 \
  --framework vllm --api-model Qwen/Qwen2.5-7B-Instruct

# Re-run the agent-level static scan after cloning the listed repositories.
cd ../4_agent
python scripts/00_clone_agent_repos.py
python scripts/01_load_rules_scan_files.py

# Start the transparent proxy used for dynamic observation.
cd ../../RQ4
uv sync
uv run agentproxy serve --provider-base-url http://localhost:8000
```

## Environment

Basic result inspection requires Python 3.11 or later. Full reproduction may
also require:

- `uv` for locked Python environments;
- Git and ripgrep for the RQ1 agent-level static scan;
- Bash, WSL, or Linux for the RQ2/RQ3 launcher scripts;
- OpenAI-compatible model endpoints and their API credentials;
- locally obtained model weights and suitable GPU resources for experiments
  using local inference or embedding servers.

Model weights, provider credentials, and Hugging Face access tokens are not
included. Supply credentials through environment variables as documented in
the corresponding RQ README; do not store them in configuration files intended
for publication.

## Included and excluded data

- RQ1 includes model inventories, framework results, provider measurements,
  and the inputs and scripts for reproducing the 875-project static scan.
- RQ2 and RQ3 include the selected per-task JSON records used by their
  summarizers. Exploratory runs and duplicate console logs are excluded.
- RQ4 includes the transparent measurement proxy and the 50-project evaluation
  list. Project-specific dynamic trial screenshots and traces are not included.
- Full model weights and extracted full embedding matrices are excluded because
  of their size and distribution requirements. RQ2 includes scripts for
  regenerating the embedding analysis from locally obtained weights.

## Reproducibility notes

Hosted-provider behavior may change after the original collection date. The
included provider records therefore preserve the observations used in the
study, while reproduction scripts can be used to measure current behavior.
External repositories in the project lists may also change or become
unavailable; use fixed revisions when performing a long-term replication.

RQ2 and RQ3 vendor the AgentDoJo benchmark source and its locked environment.
The corresponding third-party license and citation files are retained inside
each `agentdojo/` directory.
