from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import yaml

from .models import Rule, RuleFile

DEFAULT_MODEL_ROUTE_MAP: dict[str, str] = {
    "llama-*": "llama",
    "*/llama-*": "llama",
    "meta-llama*": "llama",
    "*/meta-llama-*": "llama",
    "mistral-*": "mistral",
    "*/mistral-*": "mistral",
    "mixtral-*": "mistral",
    "*/mixtral-*": "mistral",
    "mathstral-*": "mistral",
    "*/mathstral-*": "mistral",
    "ministral-*": "mistral",
    "*/ministral-*": "mistral",
    "qwen*": "qwen",
    "gemma-*": "gemma",
    "google/gemma-*": "gemma",
    "deepseek*": "deepseek",
    "*/deepseek*": "deepseek",
    "chatglm*": "glm",
    "*/chatglm*": "glm",
    "glm*": "glm",
    "*/glm*": "glm",
}


class RuleLoadError(RuntimeError):
    pass


class RuleRegistry:
    def __init__(
        self,
        rules_dir: str | Path,
        route_map: dict[str, str] | None = None,
    ) -> None:
        self.rules_dir = Path(rules_dir)
        self.route_map = route_map or DEFAULT_MODEL_ROUTE_MAP
        self._rule_files: dict[str, RuleFile] = {}
        self.reload()

    def reload(self) -> None:
        if not self.rules_dir.exists():
            raise RuleLoadError(f"rules directory does not exist: {self.rules_dir}")

        rule_files: dict[str, RuleFile] = {}
        for path in sorted(self.rules_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                rule_files[path.stem] = RuleFile.model_validate(data)
            except Exception as exc:  # noqa: BLE001 - keep startup diagnostics explicit.
                raise RuleLoadError(f"failed to load {path}: {exc}") from exc

        self._rule_files = rule_files

    def rule_names_for_model(self, model: str | None) -> list[str]:
        names: list[str] = []
        if model:
            for pattern, rule_name in self.route_map.items():
                if fnmatch(model.lower(), pattern.lower()):
                    names.append(rule_name)
                    break

        if "universal" in self._rule_files:
            names.append("universal")

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(names))

    def rules_for_model(self, model: str | None) -> list[Rule]:
        rules: list[Rule] = []
        for name in self.rule_names_for_model(model):
            rule_file = self._rule_files.get(name)
            if rule_file:
                rules.extend(rule_file.rules)
        return rules

    @property
    def available_rule_files(self) -> list[str]:
        return sorted(self._rule_files)
