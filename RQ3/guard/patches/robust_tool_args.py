"""Tolerate empty/malformed tool-call arguments from OpenAI-compatible APIs.

Some providers (e.g. 百炼) return tool calls whose ``function.arguments`` is
``""`` or non-JSON text, especially during MELON's masked re-execution.
AgentDojo's stock ``_openai_to_tool_call`` does ``json.loads(arguments)`` and
crashes the whole benchmark with ``JSONDecodeError: Expecting value ...``.

This module replaces that parser with the same fallback used by the PIArena
fork: empty or unparseable arguments become ``{}``, so one malformed tool call
no longer kills the run.
"""

import json

from agentdojo.agent_pipeline.llms import openai_llm
from agentdojo.functions_runtime import FunctionCall


def _robust_openai_to_tool_call(tool_call) -> FunctionCall:
    args_str = tool_call.function.arguments
    if args_str is None or args_str == "":
        args = {}
    else:
        try:
            args = json.loads(args_str)
            if args is None:
                args = {}
        except json.JSONDecodeError:
            args = {}
    return FunctionCall(
        function=tool_call.function.name,
        args=args,
        id=tool_call.id,
    )


openai_llm._openai_to_tool_call = _robust_openai_to_tool_call
