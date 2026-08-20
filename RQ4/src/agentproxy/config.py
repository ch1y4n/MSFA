from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .models import ProviderValidationSettings, ProxySettings


class AppConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    provider_base_url: str | None = None
    host: str = "127.0.0.1"
    port: int = 10018
    rules_dir: Path = Path("config/rules")
    logs_dir: Path = Path("logs")
    log_response_body: bool = True
    hf_token: str | None = None
    provider_validation: ProviderValidationSettings = Field(
        default_factory=ProviderValidationSettings
    )

    def proxy_settings(self) -> ProxySettings:
        if not self.provider_base_url:
            raise ValueError("provider_base_url is required")
        return ProxySettings(
            provider_base_url=self.provider_base_url,
            rules_dir=self.rules_dir,
            logs_dir=self.logs_dir,
            log_response_body=self.log_response_body,
            provider_validation=self.provider_validation,
        )


def load_config(path: str | Path | None, *, required: bool = False) -> AppConfig:
    if path is None:
        return AppConfig()

    config_path = Path(path)
    if not config_path.exists():
        if required:
            raise FileNotFoundError(f"config file does not exist: {config_path}")
        return AppConfig()

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a YAML mapping: {config_path}")
    return AppConfig.model_validate(data)
