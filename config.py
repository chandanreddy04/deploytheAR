"""Settings for the coworker platform. stdlib only, same .env pattern as
rally-ar-agent's config.py — reads a local .env once, then os.environ."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.split(" #", 1)[0].strip()
        os.environ.setdefault(key, value)


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def update_dotenv_value(key: str, value: str, path: Path = ROOT / ".env") -> bool:
    """Rewrites a single KEY=value line in .env in place, preserving every
    other line exactly (comments, blank lines, ordering). Appends a new
    line if the key isn't present yet. Returns False (no-op) if there's
    no .env file to update -- e.g. a deployment using real env vars.

    Used to persist a rotated OAuth refresh token (QuickBooks issues a
    new one on every use and invalidates the old one) so the next process
    start doesn't fail with a stale value that only ever lived in memory.
    """
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.partition("=")[0].strip() == key:
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


@dataclass
class Settings:
    hubspot_token: str | None = None
    hubspot_mode: str = "mock"  # "mock" | "live" — mock is the safe default
    llm_mode: str = "mock"      # "mock" | "live" — mock is the safe default
    llm_provider: str = "anthropic"  # "anthropic" | "groq" — which live client to use
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"
    qbo_mode: str = "mock"  # "mock" | "live" — mock is the safe default
    qbo_base_url: str = "https://sandbox-quickbooks.api.intuit.com"
    qbo_realm_id: str | None = None
    qbo_client_id: str | None = None
    qbo_client_secret: str | None = None
    qbo_refresh_token: str | None = None
    qbo_access_token: str | None = None
    gmail_mode: str = "mock"  # "mock" | "live" — mock is the safe default
    gmail_client_id: str | None = None
    gmail_client_secret: str | None = None
    gmail_refresh_token: str | None = None
    gmail_sender_email: str | None = None
    # Every automated reminder email (the -7/0/+14 cadence in run_reminder_cycle())
    # goes here instead of the real customer address until this is turned off --
    # set to "real" in .env to start sending to each invoice's actual BillEmail.
    outstanding_reminder_override_email: str | None = "capstnprjt@gmail.com"


def load_settings() -> Settings:
    _load_dotenv()
    token = _env("HUBSPOT_TOKEN")
    # live only if explicitly requested AND a token is present — never fall
    # into a real API call just because a token happens to be set.
    mode = (_env("HUBSPOT_MODE", "mock") or "mock").lower()
    if mode == "live" and not token:
        raise RuntimeError("HUBSPOT_MODE=live but HUBSPOT_TOKEN is not set")

    llm_mode = (_env("LLM_MODE", "mock") or "mock").lower()
    provider = (_env("LLM_PROVIDER", "anthropic") or "anthropic").lower()
    if provider not in ("anthropic", "groq"):
        raise RuntimeError(f"LLM_PROVIDER must be 'anthropic' or 'groq', got {provider!r}")

    anthropic_key = _env("ANTHROPIC_API_KEY")
    groq_key = _env("GROQ_API_KEY")
    if llm_mode == "live":
        if provider == "anthropic" and not anthropic_key:
            raise RuntimeError("LLM_MODE=live with LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        if provider == "groq" and not groq_key:
            raise RuntimeError("LLM_MODE=live with LLM_PROVIDER=groq but GROQ_API_KEY is not set")

    qbo_mode = (_env("QBO_MODE", "mock") or "mock").lower()
    qbo_realm_id = _env("QBO_REALM_ID")
    qbo_client_id = _env("QBO_CLIENT_ID")
    qbo_client_secret = _env("QBO_CLIENT_SECRET")
    qbo_refresh_token = _env("QBO_REFRESH_TOKEN")
    qbo_access_token = _env("QBO_ACCESS_TOKEN")
    if qbo_mode == "live":
        # A rotated refresh token (integrations/qbo.py) is saved to the
        # database as well as .env -- prefer that if present, since it's
        # the only copy that survives a restart once deployed (no local
        # .env file there to rewrite). Import is local: config.py must stay
        # importable even when db.py's dependencies aren't relevant yet.
        from db import db_load

        db_token, _, _ = db_load("qbo_refresh_token")
        if db_token:
            qbo_refresh_token = db_token
    if qbo_mode == "live":
        if not qbo_realm_id:
            raise RuntimeError("QBO_MODE=live but QBO_REALM_ID is not set")
        if not qbo_access_token and not (qbo_refresh_token and qbo_client_id and qbo_client_secret):
            raise RuntimeError(
                "QBO_MODE=live needs either QBO_ACCESS_TOKEN, or "
                "QBO_REFRESH_TOKEN + QBO_CLIENT_ID + QBO_CLIENT_SECRET"
            )

    gmail_mode = (_env("GMAIL_MODE", "mock") or "mock").lower()
    gmail_client_id = _env("GMAIL_CLIENT_ID")
    gmail_client_secret = _env("GMAIL_CLIENT_SECRET")
    gmail_refresh_token = _env("GMAIL_REFRESH_TOKEN")
    gmail_sender_email = _env("GMAIL_SENDER_EMAIL")
    if gmail_mode == "live":
        if not (gmail_client_id and gmail_client_secret and gmail_refresh_token and gmail_sender_email):
            raise RuntimeError(
                "GMAIL_MODE=live needs GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, "
                "GMAIL_REFRESH_TOKEN, and GMAIL_SENDER_EMAIL all set"
            )

    override_raw = (_env("OUTSTANDING_REMINDER_OVERRIDE_EMAIL", "capstnprjt@gmail.com") or "").strip()
    outstanding_reminder_override_email = None if override_raw.lower() == "real" else override_raw

    return Settings(
        hubspot_token=token, hubspot_mode=mode,
        llm_mode=llm_mode, llm_provider=provider,
        anthropic_api_key=anthropic_key, llm_model=_env("LLM_MODEL", "claude-sonnet-5"),
        groq_api_key=groq_key, groq_model=_env("GROQ_MODEL", "openai/gpt-oss-20b"),
        qbo_mode=qbo_mode, qbo_base_url=_env("QBO_BASE_URL", "https://sandbox-quickbooks.api.intuit.com"),
        qbo_realm_id=qbo_realm_id, qbo_client_id=qbo_client_id, qbo_client_secret=qbo_client_secret,
        qbo_refresh_token=qbo_refresh_token, qbo_access_token=qbo_access_token,
        gmail_mode=gmail_mode, gmail_client_id=gmail_client_id, gmail_client_secret=gmail_client_secret,
        gmail_refresh_token=gmail_refresh_token, gmail_sender_email=gmail_sender_email,
        outstanding_reminder_override_email=outstanding_reminder_override_email,
    )
