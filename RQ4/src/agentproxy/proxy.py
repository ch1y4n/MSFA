from __future__ import annotations

from collections.abc import Mapping

import httpx

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

REQUEST_DROP_HEADERS = HOP_BY_HOP_HEADERS | {"host", "content-length"}
RESPONSE_DROP_HEADERS = HOP_BY_HOP_HEADERS | {"content-length", "content-encoding"}


class Forwarder:
    def __init__(
        self,
        provider_base_url: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.provider_base_url = provider_base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(
            base_url=self.provider_base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def forward(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        content: bytes,
    ) -> httpx.Response:
        return await self.client.request(
            method=method,
            url=path,
            headers=clean_request_headers(headers),
            content=content,
        )

    async def forward_stream(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        content: bytes,
    ) -> httpx.Response:
        request = self.client.build_request(
            method=method,
            url=path,
            headers=clean_request_headers(headers),
            content=content,
        )
        return await self.client.send(request, stream=True)

    async def close(self) -> None:
        await self.client.aclose()


def clean_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in REQUEST_DROP_HEADERS
    }


def clean_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in RESPONSE_DROP_HEADERS
    }
