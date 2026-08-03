"""Recursive redaction and sensitive-data classification helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


REDACTED = "[REDACTED]"

_SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?key|secret|token|password|passwd|pwd|"
    r"authorization|cookie|private[_-]?key|client[_-]?secret)(?:$|[_-])",
    re.IGNORECASE,
)
_ENV_KEY_RE = re.compile(r"^(?:env|environment|environment_variables)$", re.IGNORECASE)
_PII_KEY_RE = re.compile(
    r"(?:^|[_-])(?:email|e[_-]?mail|phone|mobile|ssn|id[_-]?card|identity[_-]?number|"
    r"credit[_-]?card|card[_-]?number)(?:$|[_-])",
    re.IGNORECASE,
)

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?"
    r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+\-/=]{8,}", re.IGNORECASE)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?im)\b(?:api[_-]?key|access[_-]?key|secret(?:[_-]?access[_-]?key)?|token|"
    r"password|passwd|pwd|authorization)\b\s*[:=]\s*([^\s,;]+)"
)
_CREDENTIAL_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@[^\s]+", re.I)
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I)
_CN_ID_RE = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _redact_string(value: str) -> str:
    redacted = _PRIVATE_KEY_RE.sub(REDACTED, value)
    redacted = _BEARER_RE.sub("Bearer " + REDACTED, redacted)
    redacted = _OPENAI_KEY_RE.sub(REDACTED, redacted)
    redacted = _AWS_KEY_RE.sub(REDACTED, redacted)
    redacted = _CREDENTIAL_URL_RE.sub(REDACTED, redacted)
    redacted = _ASSIGNMENT_SECRET_RE.sub(lambda match: match.group(0).replace(match.group(1), REDACTED), redacted)
    redacted = _EMAIL_RE.sub(REDACTED, redacted)
    redacted = _CN_ID_RE.sub(REDACTED, redacted)
    redacted = _PHONE_RE.sub(REDACTED, redacted)
    redacted = _CARD_RE.sub(REDACTED, redacted)
    return redacted


def redact_data(value: Any) -> Any:
    """Return a redacted deep copy while preserving common container types."""

    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _SECRET_KEY_RE.search(key_text) or _ENV_KEY_RE.search(key_text) or _PII_KEY_RE.search(key_text):
                result[key] = REDACTED
            else:
                result[key] = redact_data(item)
        return result
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    if isinstance(value, set):
        return {redact_data(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(redact_data(item) for item in value)
    if isinstance(value, str):
        return _redact_string(value)
    return value


def redact_text(text: str) -> str:
    return _redact_string(str(text))


# Concise compatibility name for callers that imported the initial prototype.
redact = redact_data


def find_sensitive_labels(value: Any) -> set[str]:
    """Classify secrets, PII and environment dumps without retaining values."""

    labels: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                key_text = str(key)
                if _SECRET_KEY_RE.search(key_text):
                    labels.add("secret")
                if _PII_KEY_RE.search(key_text):
                    labels.add("pii")
                if _ENV_KEY_RE.search(key_text):
                    labels.add("environment")
                visit(child)
            return
        if isinstance(item, (list, tuple, set, frozenset)):
            for child in item:
                visit(child)
            return
        if not isinstance(item, str):
            return
        if any(
            pattern.search(item)
            for pattern in (
                _PRIVATE_KEY_RE,
                _BEARER_RE,
                _OPENAI_KEY_RE,
                _AWS_KEY_RE,
                _ASSIGNMENT_SECRET_RE,
                _CREDENTIAL_URL_RE,
            )
        ):
            labels.add("secret")
        if any(pattern.search(item) for pattern in (_EMAIL_RE, _CN_ID_RE, _PHONE_RE, _CARD_RE)):
            labels.add("pii")

    visit(value)
    return labels
