"""Credential-safe asynchronous HTTP transport for provider adapters."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from story_engine.config import SecretValue
from story_engine.errors import ProviderError, ProviderErrorKind
from story_engine.security import is_credential_key, redact_text

Sleeper = Callable[[float], Awaitable[None]]
MultipartFiles = list[tuple[str, tuple[str, bytes, str]]]


@dataclass(frozen=True, slots=True)
class TransportResponse:
    response: httpx.Response
    elapsed_seconds: float
    retries: int


class HttpTransport:
    """Retry transport failures without turning them into logical attempts."""

    def __init__(
        self,
        *,
        base_url: str,
        credential: SecretValue | None,
        timeout_seconds: float,
        max_retries: int,
        client: httpx.AsyncClient | None = None,
        sleeper: Sleeper = asyncio.sleep,
        credential_header: str = "Authorization",
        credential_prefix: str = "Bearer ",
    ) -> None:
        self._base_url = base_url.rstrip("/") + "/"
        self._credential = credential
        self._max_retries = max_retries
        self._sleeper = sleeper
        self._credential_header = credential_header
        self._credential_prefix = credential_prefix
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
        )

    def __repr__(self) -> str:
        return (
            f"HttpTransport(base_url={self._base_url!r}, "
            f"credential={'***' if self._credential else None}, "
            f"max_retries={self._max_retries})"
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def request(
        self,
        method: str,
        path_or_url: str,
        *,
        json_body: Any | None = None,
        data: dict[str, str] | None = None,
        files: MultipartFiles | None = None,
        headers: dict[str, str] | None = None,
        allow_retries: bool = True,
    ) -> TransportResponse:
        started = time.monotonic()
        retry_limit = self._max_retries if allow_retries else 0
        for retry_index in range(retry_limit + 1):
            try:
                response = await self._request_with_safe_redirects(
                    method,
                    path_or_url,
                    json_body=json_body,
                    data=data,
                    files=files,
                    headers=headers,
                )
            except httpx.TimeoutException as exc:
                if retry_index < retry_limit:
                    await self._sleeper(self._backoff(retry_index, None))
                    continue
                raise self._exception_error(
                    ProviderErrorKind.TIMEOUT,
                    "provider request timed out",
                    exc,
                    retryable=True,
                ) from None
            except httpx.TransportError as exc:
                if retry_index < retry_limit:
                    await self._sleeper(self._backoff(retry_index, None))
                    continue
                raise self._exception_error(
                    ProviderErrorKind.NETWORK,
                    "provider network request failed",
                    exc,
                    retryable=True,
                ) from None

            if self._retryable_status(response.status_code) and retry_index < retry_limit:
                await self._sleeper(self._backoff(retry_index, response))
                continue
            if not 200 <= response.status_code < 300:
                raise self._response_error(response)
            return TransportResponse(
                response=response,
                elapsed_seconds=time.monotonic() - started,
                retries=retry_index,
            )
        raise AssertionError("transport retry loop must return or raise")

    def _absolute_url(self, path_or_url: str) -> httpx.URL:
        try:
            url = httpx.URL(self._base_url).join(path_or_url)
            if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
                raise ValueError("invalid provider URL")
        except (httpx.InvalidURL, ValueError):
            raise ProviderError(
                kind=ProviderErrorKind.INVALID_REQUEST,
                message="provider URL must be HTTP(S) without user-info credentials",
            ) from None
        return url

    @staticmethod
    def _origin(url: httpx.URL) -> tuple[str, str, int]:
        return url.scheme, url.host, url.port or (443 if url.scheme == "https" else 80)

    def _should_authorize(self, path_or_url: str) -> bool:
        return self._origin(self._absolute_url(path_or_url)) == self._origin(
            httpx.URL(self._base_url)
        )

    async def _request_with_safe_redirects(
        self,
        method: str,
        path_or_url: str,
        *,
        json_body: Any | None,
        data: dict[str, str] | None,
        files: MultipartFiles | None,
        headers: dict[str, str] | None,
    ) -> httpx.Response:
        url = self._absolute_url(path_or_url)
        method = method.upper()
        authorize = self._should_authorize(str(url))
        for _ in range(11):
            request_headers = httpx.Headers({"Accept": "application/json"})
            if headers:
                request_headers.update(headers)
            if authorize and self._credential is not None:
                request_headers[self._credential_header] = (
                    f"{self._credential_prefix}{self._credential.reveal()}"
                )
            request = self._client.build_request(
                method, url, json=json_body, data=data, files=files, headers=request_headers
            )
            if not authorize:
                # Also remove defaults/cookies inherited from an injected client.
                # Retain only the download URL's own query, not client defaults
                # such as an API key configured with AsyncClient(params=...).
                request.url = url
                for name in list(request.headers):
                    if (
                        is_credential_key(name)
                        or name.casefold() == self._credential_header.casefold()
                    ):
                        del request.headers[name]
            # Override both redirect and auth defaults of injected HTTPX clients.
            response = await self._client.send(request, auth=None, follow_redirects=False)
            if (
                response.status_code not in {301, 302, 303, 307, 308}
                or "location" not in response.headers
            ):
                return response
            try:
                target = self._absolute_url(str(url.join(response.headers["location"])))
            except httpx.InvalidURL:
                raise ProviderError(
                    kind=ProviderErrorKind.INVALID_REQUEST,
                    message="provider returned an invalid redirect URL",
                ) from None
            cross_origin = self._origin(target) != self._origin(url)
            if url.scheme == "https" and target.scheme != "https":
                raise ProviderError(
                    kind=ProviderErrorKind.INVALID_REQUEST,
                    message="provider redirect to insecure HTTP is forbidden",
                )
            if cross_origin:
                if method not in {"GET", "HEAD"} or json_body is not None or data or files:
                    raise ProviderError(
                        kind=ProviderErrorKind.INVALID_REQUEST,
                        message="cross-origin redirect of a provider request body is forbidden",
                    )
                # Do not restore authorization even if a later redirect returns
                # to the provider. Media redirects remain possible without keys.
                authorize = False
            if (response.status_code == 303 and method != "HEAD") or (
                response.status_code in {301, 302} and method == "POST"
            ):
                method, json_body, data, files = "GET", None, None, None
            url = target
        raise ProviderError(
            kind=ProviderErrorKind.INVALID_REQUEST,
            message="provider redirect limit exceeded",
        )

    @staticmethod
    def _retryable_status(status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code <= 599

    @staticmethod
    def _backoff(retry_index: int, response: httpx.Response | None) -> float:
        if response is not None:
            raw = response.headers.get("retry-after")
            if raw:
                try:
                    return min(60.0, max(0.0, float(raw)))
                except ValueError:
                    pass
        return min(8.0, 0.25 * (2.0**retry_index))

    def _response_error(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        if status in {401, 403}:
            kind = ProviderErrorKind.AUTHENTICATION
            retryable = False
        elif status == 429:
            kind = ProviderErrorKind.RATE_LIMIT
            retryable = True
        elif 400 <= status < 500:
            kind = ProviderErrorKind.INVALID_REQUEST
            retryable = False
        elif status >= 500:
            kind = ProviderErrorKind.SERVER
            retryable = True
        else:
            kind = ProviderErrorKind.UNKNOWN
            retryable = False
        try:
            detail = self._redact(response.text)[:1_000]
        except httpx.ResponseNotRead:
            detail = "response body unavailable"
        return ProviderError(
            kind=kind,
            message=f"provider HTTP error: {detail}",
            retryable=retryable,
            status_code=status,
        )

    def _exception_error(
        self,
        kind: ProviderErrorKind,
        message: str,
        exc: Exception,
        *,
        retryable: bool,
    ) -> ProviderError:
        return ProviderError(
            kind=kind,
            message=self._redact(f"{message}: {exc}")[:1_000],
            retryable=retryable,
        )

    def _redact(self, value: str) -> str:
        secrets = (self._credential.reveal(),) if self._credential is not None else ()
        return redact_text(value, secrets=secrets)

    def redact_text(self, value: str) -> str:
        return self._redact(value)
