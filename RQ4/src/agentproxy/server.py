from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import rich.console
import rich.text

from .engine import DetectionEngine
from .logger import JsonlLogger
from .models import AuditEntry, DetectionEvent, ProxySettings, ValidationEvent, new_id
from .proxy import Forwarder, clean_response_headers
from .rules import RuleRegistry
from .validator import ProviderValidator


def create_app(
    settings: ProxySettings,
    *,
    forwarder: Forwarder | None = None,
    validator: ProviderValidator | None = None,
) -> FastAPI:
    registry = RuleRegistry(settings.rules_dir)
    engine = DetectionEngine(registry)
    logger = JsonlLogger(settings.logs_dir, log_response_body=settings.log_response_body)
    forwarder = forwarder or Forwarder(settings.provider_base_url)
    validator = validator or ProviderValidator(
        settings.provider_base_url,
        settings.provider_validation,
        logger,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await forwarder.close()
            await validator.close()

    app = FastAPI(title="AgentProxy", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.logger = logger
    app.state.forwarder = forwarder
    app.state.validator = validator

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy_request(
        path: str,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> Response:
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or new_id("req")
        body = await request.body()
        validation_enabled = settings.provider_validation.enabled

        try:
            detection = engine.detect(
                body,
                request_id=request_id,
                validation_enabled=validation_enabled,
            )
        except Exception as exc:  # noqa: BLE001 - detection must not block proxying.
            print(f"agentproxy detection error: {exc}")
            detection = None

        events = detection.events if detection else []
        model = detection.model if detection else None
        payload = detection.payload if detection else None
        for event in events:
            logger.log_detection(event)
            _print_detection(event)

        if validation_enabled and events:
            import asyncio

            async def _validate_and_print() -> None:
                try:
                    await validator.validate_and_log(
                        events,
                        request_id=request_id,
                        model=model,
                        headers=dict(request.headers),
                        on_event=_print_validation,
                    )
                except Exception as exc:
                    print(f"agentproxy validation error: {exc}")
                    import traceback
                    traceback.print_exc()

            if settings.provider_validation.async_enabled:
                asyncio.get_running_loop().create_task(_validate_and_print())
            else:
                await _validate_and_print()

        upstream_path = "/" + path
        if request.url.query:
            upstream_path = f"{upstream_path}?{request.url.query}"

        stream_request = bool(
            isinstance(payload, dict)
            and payload.get("stream") is True
            and request.method.upper() == "POST"
        )

        try:
            if stream_request:
                return await _stream_response(
                    request=request,
                    background_tasks=background_tasks,
                    forwarder=forwarder,
                    logger=logger,
                    path=upstream_path,
                    body=body,
                    request_id=request_id,
                    started=started,
                    model=model,
                    detection_count=len(events),
                )

            upstream = await forwarder.forward(
                method=request.method,
                path=upstream_path,
                headers=request.headers,
                content=body,
            )
        except httpx.RequestError as exc:
            elapsed_ms = _elapsed_ms(started)
            logger.log_audit(
                AuditEntry(
                    request_id=request_id,
                    path=upstream_path,
                    model=model,
                    detections=len(events),
                    elapsed_ms=elapsed_ms,
                    status=502,
                )
            )
            return JSONResponse(
                {"error": "provider_unreachable", "detail": str(exc)},
                status_code=502,
                background=background_tasks,
            )

        response_body = upstream.content
        logger.log_response(
            request_id=request_id,
            status_code=upstream.status_code,
            headers=dict(upstream.headers),
            body=response_body,
        )
        logger.log_audit(
            AuditEntry(
                request_id=request_id,
                path=upstream_path,
                model=model,
                detections=len(events),
                elapsed_ms=_elapsed_ms(started),
                status=upstream.status_code,
            )
        )
        return Response(
            content=response_body,
            status_code=upstream.status_code,
            headers=clean_response_headers(upstream.headers),
            media_type=upstream.headers.get("content-type"),
            background=background_tasks,
        )

    return app


async def _stream_response(
    *,
    request: Request,
    background_tasks: BackgroundTasks,
    forwarder: Forwarder,
    logger: JsonlLogger,
    path: str,
    body: bytes,
    request_id: str,
    started: float,
    model: str | None,
    detection_count: int,
) -> StreamingResponse:
    upstream = await forwarder.forward_stream(
        method=request.method,
        path=path,
        headers=request.headers,
        content=body,
    )
    chunks: list[bytes] = []

    async def iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                chunks.append(chunk)
                yield chunk
        finally:
            body_bytes = b"".join(chunks)
            logger.log_response(
                request_id=request_id,
                status_code=upstream.status_code,
                headers=dict(upstream.headers),
                body=body_bytes,
            )
            logger.log_audit(
                AuditEntry(
                    request_id=request_id,
                    path=path,
                    model=model,
                    detections=detection_count,
                    elapsed_ms=_elapsed_ms(started),
                    status=upstream.status_code,
                )
            )
            await upstream.aclose()

    return StreamingResponse(
        iterator(),
        status_code=upstream.status_code,
        headers=clean_response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
        background=background_tasks,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def is_stream_payload(body: bytes) -> bool:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


_console = rich.console.Console()

_SEVERITY_COLOR = {
    "critical": "bright_red",
    "high": "red",
    "medium": "yellow",
    "low": "dim",
}


def _print_detection(event: DetectionEvent) -> None:
    color = _SEVERITY_COLOR.get(event.severity, "white")
    parts = [
        f"[bold {color}]⚠  {event.severity.upper()}[/] "
        f"[bold]{rich.markup.escape(event.token)}[/]",
        f"rule={event.rule_id}",
        f"{event.location}",
    ]
    if event.role:
        parts.append(rich.markup.escape(f"[{event.role}]"))
    parts.append(f"model={event.model or '?'}")
    _console.print(*parts, sep=" | ")


_VERDICT_ICON = {
    "vulnerability": "🔥",
    "not_vulnerability": "✅",
    "observed_only": "❓",
}


def _print_validation(ve: ValidationEvent) -> None:
    icon = _VERDICT_ICON.get(ve.verdict, "❓")
    color = "bright_red" if ve.verdict == "vulnerability" else "green"
    _console.print(
        f"   {icon} [{color}]Δ probe: {ve.token}[/] "
        f"[{color}]{ve.effective_status}[/] → [{color}]{ve.verdict}[/] "
        f"({ve.evidence})"
    )
