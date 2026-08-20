#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${1:-}"
ATTACK="${2:-}"

if [[ ! "$SUITE" =~ ^(travel|banking|slack|workspace)$ ]] || \
   [[ ! "$ATTACK" =~ ^(glm_plain|glm_sti|glm_sti_vector)$ ]]; then
  echo "usage: $0 {travel|banking|slack|workspace} {glm_plain|glm_sti|glm_sti_vector}" >&2
  exit 2
fi

: "${OPENAI_COMPATIBLE_BASE_URL:?set the OpenAI-compatible GLM-5.2 endpoint}"
: "${OPENAI_COMPATIBLE_API_KEY:?set the endpoint API key}"
: "${MODEL_ID:?set the GLM-5.2 API model identifier}"

export PYTHONPATH="$ROOT:$ROOT/agentdojo/src${PYTHONPATH:+:$PYTHONPATH}"
export GUARD_STI_UNESCAPE=1
mkdir -p "$ROOT/new_results"

args=(
  --model openai-compatible
  --model-id "$MODEL_ID"
  --suite "$SUITE"
  --attack "$ATTACK"
  --logdir "$ROOT/new_results"
  --module-to-load guard.attacks.glm_plain
  --module-to-load "guard.attacks.$ATTACK"
  --module-to-load guard.patches.openai_timeout
  --module-to-load guard.patches.robust_tool_args
  --module-to-load guard.patches.openrouter_reasoning
)

if [[ "${USE_UCAS_PATCH:-0}" == "1" ]]; then
  args+=(--module-to-load guard.patches.ucass_openai_compat)
fi

uv run --project "$ROOT/agentdojo" python -m guard.scripts.run_agentdojo "${args[@]}"
