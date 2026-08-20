from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

TOKENIZER_ARTIFACTS = {
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}


@dataclass(frozen=True)
class FetchResult:
    model_id: str
    revision: str
    output_dir: Path
    downloaded: list[str]
    skipped: list[str]


def is_tokenizer_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = Path(normalized).name
    if normalized.startswith("encoding/") and Path(name).suffix in {".py", ".md"}:
        return True
    return name in TOKENIZER_ARTIFACTS


def repo_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def fetch_tokenizer_files(
    model_id: str,
    *,
    output_root: str | Path = "references/tokenizers",
    revision: str = "main",
    token: str | None = None,
    client: httpx.Client | None = None,
) -> FetchResult:
    token = token if token is not None else os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    close_client = client is None
    client = client or httpx.Client(timeout=120.0, follow_redirects=True)

    try:
        files = _list_repo_files(client, model_id, revision, headers)
        selected = [path for path in files if is_tokenizer_artifact(path)]
        skipped = [path for path in files if path not in selected]

        output_dir = Path(output_root) / repo_dir_name(model_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[str] = []
        for path in selected:
            content = _download_file(client, model_id, revision, path, headers)
            target = output_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            downloaded.append(path)

        manifest = {
            "model_id": model_id,
            "revision": revision,
            "downloaded": downloaded,
            "skipped": skipped,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return FetchResult(
            model_id=model_id,
            revision=revision,
            output_dir=output_dir,
            downloaded=downloaded,
            skipped=skipped,
        )
    finally:
        if close_client:
            client.close()


def _list_repo_files(
    client: httpx.Client,
    model_id: str,
    revision: str,
    headers: dict[str, str],
) -> list[str]:
    url = (
        f"https://huggingface.co/api/models/{model_id}/tree/"
        f"{quote(revision, safe='')}?recursive=true"
    )
    response = client.get(url, headers=headers)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and isinstance(payload.get("siblings"), list):
        return [item["rfilename"] for item in payload["siblings"] if "rfilename" in item]
    if isinstance(payload, list):
        return [item["path"] for item in payload if isinstance(item, dict) and "path" in item]
    raise ValueError("unexpected Hugging Face tree API response")


def _download_file(
    client: httpx.Client,
    model_id: str,
    revision: str,
    path: str,
    headers: dict[str, str],
) -> bytes:
    url = (
        f"https://huggingface.co/{model_id}/resolve/"
        f"{quote(revision, safe='')}/{quote(path, safe='/')}"
    )
    response = client.get(url, headers=headers)
    response.raise_for_status()
    return response.content
