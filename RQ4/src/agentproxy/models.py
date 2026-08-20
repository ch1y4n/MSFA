from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

MatchType = Literal["contains", "exact", "regex"]
Severity = Literal["critical", "high", "medium", "low"]
FilterStatus = Literal["filter_miss", "unknown"]
ValidationStatus = Literal["disabled", "pending", "complete", "failed"]
EffectiveStatus = Literal[
    "unknown",
    "effective",
    "provider_split_inert",
]
Verdict = Literal["observed_only", "not_vulnerability", "vulnerability"]
ValidationMode = Literal[
    "auto",
    "tokenizer_endpoint",
    "delta_probe",
    "behavior_probe",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    tokens: list[str]
    match_type: MatchType = "contains"
    severity: Severity = "medium"
    description: str | None = None


class RuleFileSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    check_request_body: bool = True
    check_request_headers: bool = False
    log_response_body: bool = True
    report_output: str = "logs/detections.jsonl"
    provider_validation: dict[str, Any] = Field(default_factory=dict)


class RuleFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: list[Rule] = Field(default_factory=list)
    settings: RuleFileSettings = Field(default_factory=RuleFileSettings)


class Match(BaseModel):
    token: str
    location: str
    offset: int
    end_offset: int
    context: str
    role: str | None = None


class DetectionEvent(BaseModel):
    ts: str = Field(default_factory=utc_now)
    request_id: str
    detection_id: str = Field(default_factory=lambda: new_id("det"))
    rule_id: str
    severity: Severity
    model: str | None = None
    token: str
    match_type: MatchType
    location: str
    role: str | None = None
    offset: int
    context: str
    evidence_type: Literal["appearance"] = "appearance"
    filter_status: FilterStatus
    validation_status: ValidationStatus = "disabled"
    effective_status: EffectiveStatus = "unknown"
    verdict: Verdict = "observed_only"


class AuditEntry(BaseModel):
    ts: str = Field(default_factory=utc_now)
    request_id: str
    path: str
    model: str | None = None
    detections: int
    elapsed_ms: int
    status: int


class ValidationEvent(BaseModel):
    ts: str = Field(default_factory=utc_now)
    validation_id: str = Field(default_factory=lambda: new_id("val"))
    request_id: str
    detection_ids: list[str]
    provider: str
    model: str | None = None
    token: str
    mode: ValidationMode
    probe_count: int
    effective_status: EffectiveStatus
    verdict: Verdict
    evidence: str


class ProviderValidationSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    mode: ValidationMode = "auto"
    async_enabled: bool = Field(default=True, alias="async")
    max_tokens_per_request: int = 20
    concurrency: int = 1
    probe_max_tokens: int = 1
    disable_thinking: bool = True
    extra_body: dict[str, Any] = Field(default_factory=dict)
    chat_completions_path: str = "/chat/completions"
    timeout_seconds: float = 30.0


class ProxySettings(BaseModel):
    provider_base_url: str
    rules_dir: Path = Path("config/rules")
    logs_dir: Path = Path("logs")
    log_response_body: bool = True
    provider_validation: ProviderValidationSettings = Field(
        default_factory=ProviderValidationSettings
    )
