#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${1:-}"
DEFENSE="${2:-}"
ATTACK="${3:-}"

if [[ ! "$SUITE" =~ ^(travel|banking|slack|workspace)$ ]] || \
   [[ ! "$DEFENSE" =~ ^(none|promptarmor|melon|attriguard)$ ]] || \
   [[ ! "$ATTACK" =~ ^(direct|ignore_previous|important_instructions|tool_knowledge|msfa)$ ]]; then
  echo "usage: $0 {travel|banking|slack|workspace} {none|promptarmor|melon|attriguard} {direct|ignore_previous|important_instructions|tool_knowledge|msfa}" >&2
  exit 2
fi

if [[ "$DEFENSE" == "none" && "$ATTACK" == "msfa" ]]; then
  echo "MSFA is evaluated only with promptarmor, melon, or attriguard" >&2
  exit 2
fi

: "${OPENAI_COMPATIBLE_BASE_URL:?set the victim-model OpenAI-compatible endpoint}"
: "${OPENAI_COMPATIBLE_API_KEY:?set the victim endpoint API key}"
: "${MODEL_ID:?set the victim model identifier}"

export PYTHONPATH="$ROOT:$ROOT/agentdojo/src${PYTHONPATH:+:$PYTHONPATH}"
export GUARD_STI_UNESCAPE=1
mkdir -p "$ROOT/new_results"

attack_name="$ATTACK"
extra_modules=()
if [[ "$ATTACK" == "msfa" ]]; then
  attack_name="sti_attack_${DEFENSE}"
  extra_modules+=(--module-to-load "guard.attacks.$attack_name")
fi

defense_args=()
if [[ "$DEFENSE" != "none" ]]; then
  defense_args+=(--defense "$DEFENSE")
fi

args=(
  --model openai-compatible
  --model-id "$MODEL_ID"
  --suite "$SUITE"
  --attack "$attack_name"
  --logdir "$ROOT/new_results"
  --module-to-load guard.patches.openai_timeout
  --module-to-load guard.patches.robust_tool_args
  --module-to-load guard.patches.openrouter_reasoning
)
args+=("${defense_args[@]}")
args+=("${extra_modules[@]}")

if [[ "${USE_UCAS_PATCH:-0}" == "1" ]]; then
  args+=(--module-to-load guard.patches.ucass_openai_compat)
fi

uv run --project "$ROOT/agentdojo" python -m guard.scripts.run_agentdojo "${args[@]}"
