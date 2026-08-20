from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import uvicorn

from .config import AppConfig, load_config
from .hf import fetch_tokenizer_files
from .models import ProviderValidationSettings
from .server import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agentproxy")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the proxy server")
    serve_parser.add_argument(
        "--config",
        default=os.environ.get("AGENTPROXY_CONFIG", "agentproxy.yaml"),
        help="YAML config path. Defaults to agentproxy.yaml when present.",
    )
    serve_parser.add_argument(
        "--provider-base-url",
        default=None,
        help="Upstream open-model provider base URL",
    )
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--rules-dir", default=None)
    serve_parser.add_argument("--logs-dir", default=None)
    serve_parser.add_argument(
        "--log-response-body",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable full response-body audit logging",
    )
    serve_parser.add_argument(
        "--validate-provider",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable asynchronous provider delta validation",
    )

    report_parser = subparsers.add_parser("report", help="Read JSONL audit logs")
    report_parser.add_argument("--logs-dir", default="logs")
    report_parser.add_argument("--summary", action="store_true")
    report_parser.add_argument("--rule")
    report_parser.add_argument("--last", type=int, default=50)

    fetch_parser = subparsers.add_parser(
        "fetch-tokenizer",
        help="Download tokenizer/chat-template files from Hugging Face",
    )
    fetch_parser.add_argument("model_id")
    fetch_parser.add_argument("--revision", default="main")
    fetch_parser.add_argument("--out-dir", default="references/tokenizers")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Run a prompt-delta probe against a provider to check token splitting",
    )
    validate_parser.add_argument("token", help="Raw special token to probe")
    validate_parser.add_argument(
        "--model", required=True, help="Model name to use in the request"
    )
    validate_parser.add_argument(
        "--provider", default=None, help="Provider base URL (defaults to config value)"
    )
    validate_parser.add_argument(
        "--endpoint", default="/v1/chat/completions",
        help="Chat completions endpoint path",
    )
    validate_parser.add_argument("--api-key", default=None, help="API key for the provider")

    args = parser.parse_args(argv)
    if args.command == "report":
        run_report(args)
        return
    if args.command == "fetch-tokenizer":
        run_fetch_tokenizer(args)
        return
    if args.command == "validate":
        import asyncio
        asyncio.run(run_validate(args))
        return

    if args.command not in {"serve", None}:
        parser.error("unknown command")
    if args.command is None:
        parser.print_help()
        return

    config = _build_serve_config(args, parser)
    settings = config.proxy_settings()
    app = create_app(settings)
    uvicorn.run(app, host=config.host, port=config.port)


def _build_serve_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> AppConfig:
    required_config = args.config != "agentproxy.yaml"
    try:
        config = load_config(args.config, required=required_config)
    except Exception as exc:  # noqa: BLE001 - CLI should show concise config errors.
        parser.error(str(exc))

    updates: dict[str, Any] = {}
    env_provider_base_url = os.environ.get("AGENTPROXY_PROVIDER_BASE_URL")
    if env_provider_base_url:
        updates["provider_base_url"] = env_provider_base_url

    for attr in (
        "provider_base_url",
        "host",
        "port",
        "rules_dir",
        "logs_dir",
        "log_response_body",
    ):
        value = getattr(args, attr)
        if value is not None:
            updates[attr] = value

    provider_validation = config.provider_validation
    if args.validate_provider is not None:
        provider_validation_data = provider_validation.model_dump(by_alias=True)
        provider_validation_data.update(
            {
                "enabled": args.validate_provider,
                "mode": "delta_probe",
                "async": True,
            }
        )
        provider_validation = ProviderValidationSettings(
            **provider_validation_data,
        )

    try:
        data = config.model_dump(by_alias=True)
        data.update(updates)
        data["provider_validation"] = provider_validation.model_dump(by_alias=True)
        merged = AppConfig.model_validate(data)
        merged.proxy_settings()
        return merged
    except Exception as exc:  # noqa: BLE001 - CLI should show concise config errors.
        parser.error(str(exc))


def run_report(args: argparse.Namespace) -> None:
    detections_path = Path(args.logs_dir) / "detections.jsonl"
    if not detections_path.exists():
        print("No detections found.")
        return

    events = [_read_json(line) for line in detections_path.read_text(encoding="utf-8").splitlines()]
    events = [event for event in events if event]
    if args.rule:
        events = [event for event in events if event.get("rule_id") == args.rule]

    if args.summary:
        print(json.dumps(_summary(events), ensure_ascii=False, indent=2))
        return

    for event in events[-args.last :]:
        print(json.dumps(event, ensure_ascii=False))


def _summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_detections": len(events),
        "by_rule": dict(Counter(event.get("rule_id") for event in events)),
        "by_model": dict(Counter(event.get("model") for event in events)),
        "by_verdict": dict(Counter(event.get("verdict") for event in events)),
        "by_effective_status": dict(
            Counter(event.get("effective_status") for event in events)
        ),
    }


def _read_json(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def run_fetch_tokenizer(args: argparse.Namespace) -> None:
    try:
        config = load_config("agentproxy.yaml", required=False)
        hf_token = config.hf_token or os.environ.get("HF_TOKEN")
    except Exception:
        hf_token = os.environ.get("HF_TOKEN")
    result = fetch_tokenizer_files(
        args.model_id,
        output_root=args.out_dir,
        revision=args.revision,
        token=hf_token,
    )
    print(f"Downloaded {len(result.downloaded)} tokenizer files to {result.output_dir}")
    for path in result.downloaded:
        print(f"- {path}")


async def run_validate(args: argparse.Namespace) -> None:
    import asyncio

    import httpx

    config = load_config("agentproxy.yaml", required=False)
    provider_url = args.provider or config.provider_base_url
    if not provider_url:
        print("Error: --provider is required (or set provider_base_url in agentproxy.yaml)")
        return

    chat_url = f"{provider_url.rstrip('/')}{args.endpoint}"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:

        async def prompt_tokens(content: str) -> int | None:
            payload: dict[str, Any] = {
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 1,
                "temperature": 0,
            }
            if args.model:
                payload["model"] = args.model
            try:
                resp = await client.post(chat_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data.get("usage", {}).get("prompt_tokens")
            except Exception as exc:
                print(f"Request failed: {exc}")
                return None

        baseline = await prompt_tokens("")
        probe = await prompt_tokens(args.token)

        if baseline is None or probe is None:
            print("Failed to get prompt_tokens from provider")
            return

        delta = probe - baseline
        if delta <= 0:
            effective = "unknown"
            verdict = "observed_only"
        elif delta == 1:
            effective = "effective"
            verdict = "vulnerability"
        else:
            effective = "provider_split_inert"
            verdict = "not_vulnerability"

        print(json.dumps({
            "token": args.token,
            "model": args.model,
            "provider": provider_url,
            "baseline_prompt_tokens": baseline,
            "probe_prompt_tokens": probe,
            "delta": delta,
            "effective_status": effective,
            "verdict": verdict,
        }, ensure_ascii=False, indent=2))
