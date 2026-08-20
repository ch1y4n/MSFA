"""Patch agentdojo's OpenAI message conversion for 国科大 UniAPI.

The official agentdojo checkout converts `system` messages to the OpenAI
`developer` role. UniAPI (uni-api.cstcloud.cn) rejects that role with a 422
"Unexpected message role". This module, loaded via agentdojo's
`--module-to-load`, replaces `_message_to_openai` at runtime so `system`
messages stay `system` and their list-form content is flattened to a string.
"""

from agentdojo.agent_pipeline.llms import openai_llm

_original = openai_llm._message_to_openai
_original_content_blocks = openai_llm._content_blocks_to_openai_content_blocks


def _patched_content_blocks(message):
    content = message.get("content") if isinstance(message, dict) else message["content"]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return _original_content_blocks(message)


def _patched_message_to_openai(message, model_name):
    converted = _original(message, model_name)
    if converted.get("role") == "developer":
        content = converted.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        converted = {"role": "system", "content": content}
    if converted.get("role") == "assistant" and converted.get("tool_calls"):
        content = converted.get("content")
        if isinstance(content, list) and content and all(
            isinstance(part, dict)
            and part.get("type") == "text"
            and not (part.get("text") or "").strip()
            for part in content
        ):
            if isinstance(converted, dict):
                converted = {**converted, "content": None}
            else:
                converted = converted.model_copy(update={"content": None})
    return converted


openai_llm._message_to_openai = _patched_message_to_openai
openai_llm._content_blocks_to_openai_content_blocks = _patched_content_blocks
