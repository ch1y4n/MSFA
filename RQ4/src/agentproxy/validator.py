from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .logger import JsonlLogger
from .models import (
    DetectionEvent,
    EffectiveStatus,
    ProviderValidationSettings,
    ValidationEvent,
    ValidationMode,
    Verdict,
)
from .proxy import clean_request_headers


@dataclass(frozen=True)
class ValidationOutcome:
    effective_status: EffectiveStatus
    verdict: Verdict
    evidence: str
    probe_count: int = 0


class ProviderValidator:
    def __init__(
        self,
        provider_base_url: str,
        settings: ProviderValidationSettings,
        logger: JsonlLogger,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.provider_base_url = provider_base_url.rstrip("/")
        self.settings = settings
        self.logger = logger
        self.client = client or httpx.AsyncClient(
            base_url=self.provider_base_url,
            timeout=httpx.Timeout(settings.timeout_seconds),
        )
        self._cache: dict[tuple[str | None, str], ValidationOutcome] = {}
        self._inflight: dict[tuple[str | None, str], asyncio.Task[ValidationOutcome]] = {}
        self._baseline_cache: dict[tuple[str | None], int] = {}
        self._baseline_inflight: dict[tuple[str | None], asyncio.Task[int | None]] = {}

    async def validate_and_log(
        self,
        events: list[DetectionEvent],
        *,
        request_id: str,
        model: str | None,
        headers: Mapping[str, str],
        on_event: Callable[[ValidationEvent], None] | None = None,
    ) -> list[ValidationEvent]:
        if not self.settings.enabled:
            return []

        candidates = [
            event for event in events if event.filter_status == "filter_miss"
        ][: self.settings.max_tokens_per_request]
        grouped: dict[str, list[DetectionEvent]] = defaultdict(list)
        for event in candidates:
            grouped[event.token].append(event)

        async def validate_group(
            token: str,
            token_events: list[DetectionEvent],
        ) -> ValidationEvent:
            outcome = await self._validate_token(
                token=token,
                model=model,
                headers=headers,
            )
            return ValidationEvent(
                request_id=request_id,
                detection_ids=[event.detection_id for event in token_events],
                provider=self.provider_base_url,
                model=model,
                token=token,
                mode=self._effective_mode(),
                probe_count=outcome.probe_count,
                effective_status=outcome.effective_status,
                verdict=outcome.verdict,
                evidence=outcome.evidence,
            )

        if self.settings.concurrency <= 1:
            validation_events: list[ValidationEvent] = []
            for token, token_events in grouped.items():
                validation_event = await validate_group(token, token_events)
                self.logger.log_validation(validation_event)
                validation_events.append(validation_event)
                if on_event:
                    on_event(validation_event)
            return validation_events

        semaphore = asyncio.Semaphore(self.settings.concurrency)

        async def validate_group_limited(
            token: str,
            token_events: list[DetectionEvent],
        ) -> ValidationEvent:
            async with semaphore:
                return await validate_group(token, token_events)

        validation_events = []
        tasks = [
            asyncio.create_task(validate_group_limited(token, token_events))
            for token, token_events in grouped.items()
        ]
        for task in asyncio.as_completed(tasks):
            validation_event = await task
            self.logger.log_validation(validation_event)
            validation_events.append(validation_event)
            if on_event:
                on_event(validation_event)
        return validation_events

    async def _validate_token(
        self,
        *,
        token: str,
        model: str | None,
        headers: Mapping[str, str],
    ) -> ValidationOutcome:
        cache_key = (model, token)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if cache_key in self._inflight:
            return await self._inflight[cache_key]

        task = asyncio.create_task(
            self._validate_token_uncached(token=token, model=model, headers=headers)
        )
        self._inflight[cache_key] = task
        try:
            outcome = await task
        finally:
            self._inflight.pop(cache_key, None)

        if outcome.effective_status != "unknown":
            self._cache[cache_key] = outcome
        return outcome

    async def _validate_token_uncached(
        self,
        *,
        token: str,
        model: str | None,
        headers: Mapping[str, str],
    ) -> ValidationOutcome:
        mode = self._effective_mode()
        if mode == "behavior_probe":
            return ValidationOutcome(
                effective_status="unknown",
                verdict="observed_only",
                evidence="behavior_probe is low confidence and is not implemented",
            )
        return await self._prompt_delta(token=token, model=model, headers=headers)

    async def _prompt_delta(
        self,
        *,
        token: str,
        model: str | None,
        headers: Mapping[str, str],
    ) -> ValidationOutcome:
        """Send chat completions with max_tokens=1, compare usage.prompt_tokens delta."""
        probe_count = 0
        try:
            baseline = await self._baseline_token_count(model, headers)
            probe_count = 1
            probe = await self._prompt_token_count(token, model, headers)
            probe_count = 2
        except httpx.HTTPError as exc:
            stage = "baseline" if probe_count == 0 else "token"
            return ValidationOutcome(
                effective_status="unknown",
                verdict="observed_only",
                evidence=(
                    f"provider delta probe failed during {stage} request: "
                    f"{_format_http_error(exc)}"
                ),
                probe_count=probe_count,
            )

        if baseline is None or probe is None:
            return ValidationOutcome(
                effective_status="unknown",
                verdict="observed_only",
                evidence="provider did not return usage.prompt_tokens",
                probe_count=2,
            )

        delta = probe - baseline
        if delta <= 0:
            return ValidationOutcome(
                effective_status="unknown",
                verdict="observed_only",
                evidence=f"non-positive delta ({delta}), probe={probe} baseline={baseline}",
                probe_count=2,
            )

        if delta == 1:
            return ValidationOutcome(
                effective_status="effective",
                verdict="vulnerability",
                evidence=f"prompt_token delta={delta} (baseline={baseline} probe={probe})",
                probe_count=2,
            )

        return ValidationOutcome(
            effective_status="provider_split_inert",
            verdict="not_vulnerability",
            evidence=f"prompt_token delta={delta} (baseline={baseline} probe={probe})",
            probe_count=2,
        )

    async def _baseline_token_count(
        self,
        model: str | None,
        headers: Mapping[str, str],
    ) -> int | None:
        cache_key = (model,)
        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]
        if cache_key in self._baseline_inflight:
            return await self._baseline_inflight[cache_key]

        task = asyncio.create_task(self._prompt_token_count("", model, headers))
        self._baseline_inflight[cache_key] = task
        try:
            result = await task
            if result is not None:
                self._baseline_cache[cache_key] = result
            return result
        finally:
            self._baseline_inflight.pop(cache_key, None)

    async def _prompt_token_count(
        self,
        user_content: str,
        model: str | None,
        headers: Mapping[str, str],
    ) -> int | None:
        """Send a minimal chat completion and return usage.prompt_tokens."""
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": self.settings.probe_max_tokens,
            "temperature": 0,
        }
        if model:
            payload["model"] = model
        if self.settings.disable_thinking:
            payload["enable_thinking"] = False
        payload.update(self.settings.extra_body)
        response = await self.client.post(
            self.settings.chat_completions_path,
            json=payload,
            headers=clean_request_headers(headers),
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            return usage.get("prompt_tokens")
        return None

    def _effective_mode(self) -> ValidationMode:
        if self.settings.mode == "auto":
            return "delta_probe"
        return self.settings.mode

    async def close(self) -> None:
        await self.client.aclose()


def _format_http_error(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        detail = response.text[:500].replace("\n", "\\n")
        return (
            f"{type(exc).__name__}: HTTP {response.status_code} "
            f"from {response.request.method} {response.request.url}; body={detail!r}"
        )

    request = getattr(exc, "request", None)
    request_info = ""
    if request is not None:
        request_info = f" during {request.method} {request.url}"
    message = str(exc) or repr(exc)
    return f"{type(exc).__name__}{request_info}: {message}"
