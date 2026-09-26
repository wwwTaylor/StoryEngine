"""Shared credential filtering and safe diagnostic rendering."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

from pydantic import ValidationError


def is_credential_key(key: str) -> bool:
    """Recognize header/config spellings without rejecting max_tokens or credential_mode."""

    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    return (
        normalized in {"auth", "pwd", "sig", "sas", "key"}
        or "apikey" in normalized
        or normalized.endswith(
            (
                "authorization", "credential", "credentials", "secret", "secretid",
                "password", "passwd", "passphrase", "token", "accesskey", "accesskeyid",
                "secretkey", "privatekey", "accountkey", "sharedkey", "signature",
                "cookie", "cookies",
            )
        )
    )


def redact_data(value: Any) -> Any:
    """Return a copy with credentials removed, including nested mappings and sequences."""

    if isinstance(value, dict):
        return {
            key: "***" if is_credential_key(str(key)) else redact_data(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact_data(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_text(value: str, *, secrets: Iterable[str] = ()) -> str:
    """Redact before callers truncate, so a length limit cannot expose key prefixes."""

    redacted = value
    variants = {
        variant
        for secret in secrets if secret
        for variant in (secret, quote(secret, safe=""), quote_plus(secret, safe=""))
    }
    for secret in sorted(variants, key=len, reverse=True):
        redacted = redacted.replace(secret, "***")

    def safe_url(match: re.Match[str]) -> str:
        try:
            parsed = urlsplit(match.group())
            authority = parsed.netloc.rsplit("@", 1)[-1]
            return urlunsplit((
                parsed.scheme, authority, parsed.path,
                "***" if parsed.query else "", "***" if parsed.fragment else "",
            ))
        except ValueError:
            return "[redacted URL]"

    redacted = re.sub(r"https?://[^\s\"'<>]+", safe_url, redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"\b(Bearer|Basic)\s+[^\s\"',;}]+", r"\1 ***", redacted, flags=re.IGNORECASE)

    def safe_assignment(match: re.Match[str]) -> str:
        if is_credential_key(match.group("key")):
            return match.group("prefix") + "***"
        return match.group()

    return re.sub(
        r"(?P<prefix>[\"']?(?P<key>[\w.-]+)[\"']?\s*[:=]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
        safe_assignment,
        redacted,
    )


def validation_summary(error: ValueError) -> str:
    """Do not echo input values, custom validator messages, or user-controlled keys."""

    if isinstance(error, ValidationError):
        issues = error.errors(include_input=False, include_context=False, include_url=False)
        kinds = sorted({item["type"] for item in issues})
        return f"{len(issues)} validation error(s): {', '.join(kinds)}"
    return "invalid value"
