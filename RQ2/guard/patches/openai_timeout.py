"""Cap the OpenAI-compatible client read timeout.

agentdojo constructs its ``openai-compatible`` client without passing a
timeout, so it inherits openai's 600-second read timeout. If a provider
accepts a request but never streams the completion back, the benchmark looks
frozen for up to ten minutes per attempt.

This module is loaded via agentdojo's ``--module-to-load`` and wraps
``get_llm`` so every created client gets a short connect/read/write timeout.
The existing retry wrapper then fails fast and retries instead of hanging.
"""

import openai

from agentdojo.agent_pipeline import agent_pipeline

_original_get_llm = agent_pipeline.get_llm

# connect 10s, read 180s, write 120s, pool 10s
_TIMEOUT = openai.Timeout(connect=10.0, read=180.0, write=120.0, pool=10.0)


def _patched_get_llm(provider: str, model: str, model_id: str | None, tool_delimiter: str):
    llm = _original_get_llm(provider, model, model_id, tool_delimiter)
    client = getattr(llm, "client", None)
    if client is not None:
        client.timeout = _TIMEOUT
    return llm


agent_pipeline.get_llm = _patched_get_llm
