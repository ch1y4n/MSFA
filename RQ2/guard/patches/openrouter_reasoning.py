"""OpenAI-compatible provider tweaks: reasoning mode + robust retries.

- OpenRouter's Qwen3.5/Qwen3.7 family defaults to thinking mode. agentdojo
  reads ``choices[0].message.content`` only, which stays ``null`` while the
  model spends tokens on ``reasoning``. We force ``reasoning.enabled=false``.
- SiliconFlow and other providers return 429 (TPM/RPM limit) under load. The
  original agentdojo wrapper only retries a handful of times; here we retry
  with longer exponential backoff so the per-minute quota window can reset.
- OpenRouter occasionally answers ``200 OK`` with an empty body (upstream
  provider hiccup). The OpenAI SDK then parses a completion whose ``choices``
  is ``None``, and agentdojo crashes with ``TypeError: 'NoneType' object is not
  subscriptable``. We detect empty choices and treat them as a retryable
  transient error instead.
- OpenRouter can also answer ``200 OK`` with a *truncated* JSON body. The SDK
  raises ``json.JSONDecodeError`` while parsing, before our empty-choices check
  can run. We catch that too, log a short sample of the bad body, and retry.
"""

import json
import logging
import os
import time

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from openai._types import NOT_GIVEN

from agentdojo.agent_pipeline.llms import openai_llm

logger = logging.getLogger(__name__)


class _EmptyCompletionError(Exception):
    """The provider answered 200 but returned no usable choices."""


def _patched_chat_completion_request(
    client,
    model,
    messages,
    tools,
    reasoning_effort,
    temperature=None,
):
    kwargs = {
        "model": model,
        "messages": messages,
        "tools": tools or NOT_GIVEN,
        "tool_choice": "auto" if tools else NOT_GIVEN,
        "temperature": temperature if temperature is not None else NOT_GIVEN,
        "reasoning_effort": reasoning_effort or NOT_GIVEN,
    }
    base_url = str(getattr(client, "base_url", "")).lower()
    extra_body = {}
    if "openrouter" in base_url:
        extra_body["reasoning"] = {"enabled": False}
    if "siliconflow" in base_url:
        extra_body["enable_thinking"] = os.getenv("SILICONFLOW_ENABLE_THINKING", "0") == "1"
    if extra_body:
        kwargs["extra_body"] = extra_body
    last_error: Exception | None = None
    for attempt in range(7):
        try:
            completion = client.chat.completions.create(**kwargs)
            choices = getattr(completion, "choices", None)
            if not choices or choices[0].message is None:
                raise _EmptyCompletionError(f"empty completion: {completion!r}")
            return completion
        except _EmptyCompletionError as exc:
            last_error = exc
            logger.warning("provider returned an empty completion (attempt %d/7); retrying", attempt + 1)
            time.sleep(min(60, 3 * (2**attempt)))
        except RateLimitError as exc:
            last_error = exc
            time.sleep(min(60, 3 * (2**attempt)))
        except APIConnectionError as exc:
            last_error = exc
            time.sleep(min(30, 2 * (2**attempt)))
        except (APITimeoutError, InternalServerError) as exc:
            last_error = exc
            time.sleep(min(30, 2 * (2**attempt)))
        except json.JSONDecodeError as exc:
            last_error = exc
            sample = (exc.doc or "")[:200].replace("\n", "\\n")
            logger.warning(
                "provider returned a malformed/truncated body (attempt %d/7); sample: %r",
                attempt + 1,
                sample,
            )
            time.sleep(min(30, 2 * (2**attempt)))
    assert last_error is not None
    raise last_error


openai_llm.chat_completion_request = _patched_chat_completion_request
