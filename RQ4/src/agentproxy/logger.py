from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import BaseModel

from .models import AuditEntry, DetectionEvent, ValidationEvent, utc_now


class JsonlLogger:
    def __init__(self, logs_dir: str | Path, log_response_body: bool = True) -> None:
        self.logs_dir = Path(logs_dir)
        self.log_response_body = log_response_body
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "responses").mkdir(parents=True, exist_ok=True)

    def log_detection(self, event: DetectionEvent) -> None:
        self._append_jsonl(self.logs_dir / "detections.jsonl", event)

    def log_validation(self, event: ValidationEvent) -> None:
        self._append_jsonl(self.logs_dir / "validations.jsonl", event)

    def log_audit(self, entry: AuditEntry) -> None:
        self._append_jsonl(self.logs_dir / "audit.jsonl", entry)

    def log_response(
        self,
        *,
        request_id: str,
        status_code: int,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        if not self.log_response_body:
            return

        path = self.logs_dir / "responses" / f"{request_id}.json"
        payload: dict[str, Any] = {
            "ts": utc_now(),
            "request_id": request_id,
            "status_code": status_code,
            "headers": headers,
            "body": _decode_response_body(body),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def _append_jsonl(self, path: Path, item: BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = item.model_dump_json() + "\n"
        lock = FileLock(str(path) + ".lock")
        with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)


def _decode_response_body(body: bytes) -> Any:
    if not body:
        return ""
    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
