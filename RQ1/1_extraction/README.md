# RQ1 / Step 1 — Model Control-Structure Extraction

This directory contains the control-token inventories extracted for the five
model families used in RQ1. Functional candidates were identified from each
model's official tokenizer configuration, chat template, and input protocol.
Each literal was then matched against `tokenizer.json` to record its token ID
and tokenizer `special` flag. Reserved placeholders without an assigned
control function were excluded.

The `special` field is tokenizer metadata: `special=false` does not mean that
the token lacks a dedicated token ID or a control function.

| File | Model | Control tokens | `special=true` | `special=false` |
|---|---|---:|---:|---:|
| `llama-3.1-8b-instruct.json` | Llama 3.1 | 8 | 8 | 0 |
| `mistral-7b-instruct-v0.3.json` | Mistral v0.3 | 10 | 10 | 0 |
| `qwen2.5-7b-instruct.json` | Qwen 2.5 | 22 | 14 | 8 |
| `gemma-3-4b-it.json` | Gemma 3 | 11 | 9 | 2 |
| `deepseek-r1-0528-qwen3-8b.json` | DeepSeek-R1-0528-Qwen3-8B | 28 | 14 | 14 |
