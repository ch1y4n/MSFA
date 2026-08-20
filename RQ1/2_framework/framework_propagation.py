#!/usr/bin/env python3
"""RQ1 / Step 2: Framework-Level Token Propagation.

Determine whether a special-token literal in user-controlled content is
preserved as its atomic control token through an inference framework's standard
chat-completion request path (vLLM, SGLang, TGI, Ollama, ...).

Three requests per token, identical except for the injected content:

    A. baseline               : ordinary content
    B. special-token-literal  : baseline + <literal>
    C. malformed              : baseline + <spaced literal>   (no longer an exact match)

Using the returned Prompt Token counts T_A, T_B, T_C:

    delta_B = T_B - T_A
    delta_C = T_C - T_A

Paper criterion (Section 4.3): ``delta_B == 1 and delta_C > 1`` is consistent
with atomic preservation of the complete literal -- the complete literal stays
one token while the malformed form splits into several.

This differential is an observable signal; the paper complements it with manual
implementation analysis (framework tokenizer-loading / chat-template path).

Tokens are loaded from the Step 1 extraction (``1_extraction/*.json``) so the
set matches the earlier layers.

Usage (needs a running framework endpoint)::

    python framework_propagation.py <model> <api_url> [--field prompt_tokens]
        [--framework vllm] [--api-key KEY] [--outdir PATH]

    # example (vLLM OpenAI-compatible endpoint)
    python framework_propagation.py qwen \\
        http://localhost:8000/v1/chat/completions --framework vllm
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = ARTIFACT_ROOT / "1_extraction"

BASELINE = "Say hello."

# model choice -> Step 1 extraction file
MODEL_EXTRACTION = {
    "llama": "llama-3.1-8b-instruct.json",
    "mistral": "mistral-7b-instruct-v0.3.json",
    "qwen": "qwen2.5-7b-instruct.json",
    "gemma": "gemma-3-4b-it.json",
    "deepseek": "deepseek-r1-0528-qwen3-8b.json",
}


def load_tokens(model: str) -> tuple[str, list[dict]]:
    """Return (model display name, [{token_id, literal, category, special}])."""
    data = json.loads(
        (EXTRACTION_DIR / MODEL_EXTRACTION[model]).read_text(encoding="utf-8")
    )
    tokens = [
        {
            "token_id": t["token_id"],
            "literal": t["literal"],
            "category": t["category"],
            "special": t["special"],
        }
        for t in data["special_tokens"]
    ]
    return data["model"], tokens


def make_spaced(literal: str) -> str:
    """Malformed form: insert spaces inside the delimiters so the literal no
    longer exactly matches the complete special token."""
    if literal.startswith("<") and literal.endswith(">"):
        return "< " + literal[1:-1].strip() + " >"
    if literal.startswith("[") and literal.endswith("]"):
        return "[ " + literal[1:-1].strip() + " ]"
    return literal


def classify(delta_b: int, delta_c: int) -> str:
    """Observable propagation signal per Section 4.3.

    - atomic_preservation : delta_B == 1 and delta_C > 1  (paper criterion)
    - split_like_text     : delta_B > 1                   (literal split as ordinary text)
    - filtered            : delta_B <= 0                  (literal stripped / absent)
    - inconclusive        : anything else
    """
    if delta_b == 1 and delta_c > 1:
        return "atomic_preservation"
    if delta_b > 1:
        return "split_like_text"
    if delta_b <= 0:
        return "filtered"
    return "inconclusive"


def req(api: str, model: str, content: str, field: str, api_key: str | None) -> int:
    body = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": content}],
    }
    if "ollama" in api or "11434" in api:
        body["options"] = {"num_predict": 1}
    else:
        body["max_tokens"] = 5
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode()
    r = urllib.request.urlopen(
        urllib.request.Request(api, data=data, headers=headers), timeout=120
    )
    resp = json.loads(r.read())
    if field in resp:
        return int(resp[field])
    return int(resp["usage"]["prompt_tokens"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=list(MODEL_EXTRACTION))
    parser.add_argument("api", help="chat-completion endpoint URL")
    parser.add_argument("--field", default="prompt_tokens",
                        help="prompt-token field name (default: prompt_tokens; "
                        "falls back to usage.prompt_tokens)")
    parser.add_argument("--framework", default="framework",
                        help="framework label used in the output filename")
    parser.add_argument("--api-key", default=os.environ.get("FRAMEWORK_API_KEY"),
                        help="optional bearer token for the endpoint")
    parser.add_argument("--api-model", default=None,
                        help="model name to send in the request (default: the "
                        "model choice, e.g. 'qwen'); set to the served model id")
    parser.add_argument("--outdir", default=str(RESULTS_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    display_name, tokens = load_tokens(args.model)
    api_model = args.api_model or args.model
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ta = req(args.api, api_model, BASELINE, args.field, args.api_key)
    print(f"Model: {display_name}  API: {args.api}  Field: {args.field}")
    print(f"Baseline T_A = {ta}")
    print(f"{'literal':<34}{'id':>8}{'TB':>5}{'dB':>4}{'TC':>5}{'dC':>4}  signal")
    print("-" * 78)

    results = []
    for tok in tokens:
        tid, lit = tok["token_id"], tok["literal"]
        spaced = make_spaced(lit)
        rec = {**tok, "spaced": spaced, "T_A": ta}
        try:
            tb = req(args.api, api_model, f"{BASELINE}{lit}", args.field, args.api_key)
            tc = req(args.api, api_model, f"{BASELINE}{spaced}", args.field, args.api_key)
            db, dc = tb - ta, tc - ta
            signal = classify(db, dc)
            rec.update({
                "T_B": tb, "T_C": tc,
                "delta_B": db, "delta_C": dc,
                "atomic_preservation": (db == 1 and dc > 1),
                "signal": signal,
                "error": "",
            })
            print(f"  {lit:<32}{tid:>8}{tb:>5}{db:>4}{tc:>5}{dc:>4}  {signal}")
        except Exception as exc:  # noqa: BLE001 - keep per-token failures in output
            rec.update({
                "T_B": None, "T_C": None,
                "delta_B": None, "delta_C": None,
                "atomic_preservation": False,
                "signal": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  {lit:<32}{tid:>8}  ERROR: {exc}")
        results.append(rec)
        time.sleep(0.1)

    preserved = sum(1 for r in results if r["atomic_preservation"])
    summary = {
        "tested": len(results),
        "atomic_preservation": preserved,
        "split_like_text": sum(1 for r in results if r["signal"] == "split_like_text"),
        "filtered": sum(1 for r in results if r["signal"] == "filtered"),
        "inconclusive": sum(1 for r in results if r["signal"] == "inconclusive"),
        "errors": sum(1 for r in results if r["signal"] == "error"),
    }
    print(f"\natomic_preservation: {preserved}/{len(results)}  "
          f"(split_like_text={summary['split_like_text']}, "
          f"filtered={summary['filtered']}, "
          f"inconclusive={summary['inconclusive']}, "
          f"errors={summary['errors']})")

    out_path = outdir / f"{args.framework}_{args.model}.json"
    out_path.write_text(
        json.dumps(
            {
                "model": display_name,
                "framework": args.framework,
                "api": args.api,
                "api_model": api_model,
                "field": args.field,
                "baseline": BASELINE,
                "T_A": ta,
                "criterion": "atomic_preservation := (delta_B == 1 and delta_C > 1)",
                "summary": summary,
                "tokens": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
