from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .models import DetectionEvent, Match, Rule
from .rules import RuleRegistry


@dataclass(frozen=True)
class DetectionResult:
    request_id: str
    model: str | None
    payload: dict[str, Any] | None
    events: list[DetectionEvent]


class DetectionEngine:
    def __init__(self, registry: RuleRegistry, context_chars: int = 48) -> None:
        self.registry = registry
        self.context_chars = context_chars
        self._regex_cache: dict[str, re.Pattern[str]] = {}

    def detect(
        self,
        body: bytes,
        *,
        request_id: str,
        validation_enabled: bool = False,
    ) -> DetectionResult:
        payload = self._parse_json(body)
        if not isinstance(payload, dict):
            return DetectionResult(request_id, None, None, [])

        model = payload.get("model") if isinstance(payload.get("model"), str) else None
        text_by_path, role_by_path = flatten_prompt_text(payload)
        events: list[DetectionEvent] = []
        seen_matches: set[tuple[str, str, int, int]] = set()

        for rule in self.registry.rules_for_model(model):
            for location, text in text_by_path.items():
                for match in self._match_rule(rule, text, location, role_by_path.get(location)):
                    key = (
                        match.token,
                        match.location,
                        match.offset,
                        match.end_offset,
                    )
                    if key in seen_matches:
                        continue
                    seen_matches.add(key)
                    events.append(
                        DetectionEvent(
                            request_id=request_id,
                            rule_id=rule.id,
                            severity=rule.severity,
                            model=model,
                            token=match.token,
                            match_type=rule.match_type,
                            location=match.location,
                            role=match.role,
                            offset=match.offset,
                            context=match.context,
                            filter_status="filter_miss",
                            validation_status=(
                                "pending" if validation_enabled else "disabled"
                            ),
                            effective_status="unknown",
                            verdict="observed_only",
                        )
                    )

        return DetectionResult(request_id, model, payload, events)

    def _parse_json(self, body: bytes) -> Any:
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _match_rule(self, rule: Rule, text: str, location: str, role: str | None = None) -> list[Match]:
        if rule.match_type == "contains":
            return self._match_contains(rule.tokens, text, location, role)
        if rule.match_type == "exact":
            return self._match_exact(rule.tokens, text, location, role)
        if rule.match_type == "regex":
            return self._match_regex(rule.tokens, text, location, role)
        return []

    def _match_contains(
        self,
        tokens: list[str],
        text: str,
        location: str,
        role: str | None = None,
    ) -> list[Match]:
        matches: list[Match] = []
        for token in tokens:
            start = 0
            while True:
                offset = text.find(token, start)
                if offset == -1:
                    break
                end = offset + len(token)
                matches.append(
                    Match(
                        token=token,
                        location=location,
                        offset=offset,
                        end_offset=end,
                        context=self._context(text, offset, end),
                        role=role,
                    )
                )
                start = end
        return matches

    def _match_exact(
        self,
        tokens: list[str],
        text: str,
        location: str,
        role: str | None = None,
    ) -> list[Match]:
        stripped = text.strip()
        for token in tokens:
            if stripped == token:
                offset = text.find(stripped)
                end = offset + len(stripped)
                return [
                    Match(
                        token=token,
                        location=location,
                        offset=offset,
                        end_offset=end,
                        context=self._context(text, offset, end),
                        role=role,
                    )
                ]
        return []

    def _match_regex(
        self,
        patterns: list[str],
        text: str,
        location: str,
        role: str | None = None,
    ) -> list[Match]:
        matches: list[Match] = []
        for pattern in patterns:
            compiled = self._regex_cache.get(pattern)
            if compiled is None:
                compiled = re.compile(pattern)
                self._regex_cache[pattern] = compiled
            for match in compiled.finditer(text):
                matches.append(
                    Match(
                        token=match.group(0),
                        location=location,
                        offset=match.start(),
                        end_offset=match.end(),
                        context=self._context(text, match.start(), match.end()),
                        role=role,
                    )
                )
        return matches

    def _context(self, text: str, offset: int, end: int) -> str:
        start = max(0, offset - self.context_chars)
        stop = min(len(text), end + self.context_chars)
        return text[start:stop].replace("\n", "\\n")


def flatten_prompt_text(payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, str | None]]:
    """Return (text_by_path, role_by_path).  assistant messages are skipped (trusted source)."""
    text_result: dict[str, str] = {}
    role_result: dict[str, str | None] = {}

    for key in ("system", "instructions", "prompt"):
        value = payload.get(key)
        if isinstance(value, str):
            text_result[key] = value
            role_result[key] = "system"

    if "input" in payload:
        _collect_strings(payload["input"], "input", text_result, role_result, "system")

    messages = payload.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if isinstance(role, str) and role == "assistant":
                continue  # assistant is a trusted source
            content = message.get("content")
            _collect_strings(content, f"messages[{index}].content", text_result, role_result, role if isinstance(role, str) else None)

    for key in ("tools", "functions"):
        if key in payload:
            _collect_strings(payload[key], key, text_result, role_result, "tool")

    return text_result, role_result


def _collect_strings(
    value: Any,
    path: str,
    text_result: dict[str, str],
    role_result: dict[str, str | None],
    role: str | None,
) -> None:
    if isinstance(value, str):
        text_result[path] = value
        role_result[path] = role
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_strings(item, f"{path}[{index}]", text_result, role_result, role)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_strings(item, f"{path}.{key}", text_result, role_result, role)
