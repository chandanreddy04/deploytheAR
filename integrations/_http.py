"""Tiny JSON-over-HTTP helper (stdlib only).

Copied from rally-ar-agent's integrations/_http.py — identical retry/backoff
behavior, kept as a standalone copy so this platform has no import
dependency on the sibling project's package.
"""

from __future__ import annotations

import json
import logging
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("coworker.http")

RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE = 0.75  # seconds

# A 429/503 whose Retry-After is this short is worth waiting out inline --
# e.g. Groq's per-minute token limit resets within its own rolling window,
# so a few seconds' wait often turns a would-be failure into a quiet
# success. Beyond this, the wait is almost certainly a longer quota (a
# daily token cap told us "try again in 25 minutes" once) that will never
# clear within a request -- sleeping for it would just block the caller
# (compute_deals() runs synchronously inside the HTTP request handler, so
# blocking here once froze the whole server for minutes). Past this
# threshold, fail fast instead and let the caller's own per-item fallback
# (e.g. "kept last known-good extraction") take over immediately.
MAX_INLINE_RETRY_WAIT = 20.0  # seconds


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} for {url}: {body[:500]}")
        self.status = status
        self.url = url
        self.body = body


def _sleep_for(attempt: int, retry_after: str | None) -> float | None:
    """Seconds to sleep before retrying, or None if the provider's
    requested wait is too long to be worth retrying inline -- the caller
    should fail fast instead."""
    if retry_after:
        try:
            wait = float(retry_after)
        except ValueError:
            wait = None
        if wait is not None:
            return wait if wait <= MAX_INLINE_RETRY_WAIT else None
    return DEFAULT_BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.4)


def request(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    json_body: dict | None = None,
    form_body: dict | None = None,
    timeout: int = 30,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry: bool = True,
) -> dict:
    headers = dict(headers or {})
    payload: bytes | None = None

    if json_body is not None:
        payload = json.dumps(json_body).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    elif form_body is not None:
        payload = urllib.parse.urlencode(form_body).encode("utf-8")
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

    headers.setdefault("Accept", "application/json")
    # urllib's default User-Agent ("Python-urllib/3.x") gets flagged as a bot
    # signature by some providers' Cloudflare layer (seen on Groq) — identify
    # honestly instead of masquerading as a browser.
    headers.setdefault("User-Agent", "rally-coworker-platform/0.1 (+https://github.com/chandanreddy04/first_meeting)")
    attempts = max_attempts if retry else 1

    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, data=payload, method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"_raw": raw}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if retry and e.code in RETRY_STATUSES and attempt < attempts:
                wait = _sleep_for(attempt, e.headers.get("Retry-After"))
                if wait is None:
                    log.warning(
                        "HTTP %s %s -> requested wait exceeds %.0fs, failing fast instead of blocking",
                        e.code, url, MAX_INLINE_RETRY_WAIT,
                    )
                    raise HttpError(e.code, url, body) from None
                log.warning("HTTP %s %s -> retry %d/%d in %.1fs", e.code, url, attempt, attempts, wait)
                time.sleep(wait)
                continue
            raise HttpError(e.code, url, body) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            if retry and attempt < attempts:
                wait = _sleep_for(attempt, None)
                log.warning("network error %s for %s -> retry %d/%d in %.1fs", e, url, attempt, attempts, wait)
                time.sleep(wait)
                continue
            raise HttpError(0, url, f"network error: {e}") from None

    raise HttpError(0, url, "exhausted retries")  # unreachable
