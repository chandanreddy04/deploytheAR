"""Turn agreement text into the useful fields an AR team actually needs.

Two modes, chosen by config.Settings.llm_mode (independent of HUBSPOT_MODE —
you can pull real contracts from HubSpot while still using the free-text
fallback here, or vice versa):

  - "live": a real call to Anthropic or Groq (needs the matching API key).
    Same prompt/schema shape as rally-ar-agent's integrations/llm_live.py.
  - "mock": no API call — a handful of regex heuristics over the raw text.
    Good enough to prove the pipeline end-to-end with zero credentials;
    not a substitute for the LLM step on real, varied contract language.

Long documents (20-30+ page contracts) are chunked automatically before
being sent to a live provider — a single real request over that size risks
exceeding a provider's per-request/per-minute token budget (hit this for
real on Groq's free tier: an accidentally-oversized ~49KB request came back
"HTTP 413 ... tokens per minute (TPM): Limit 8000"). Each chunk is asked the
same question independently; results are merged by taking, per field, the
answer with the highest reported confidence across all chunks — real
billing terms typically appear once, in one section, so exactly one chunk
should report each field with real confidence and the rest should report 0.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from config import Settings
from integrations import _http

_SCHEMA_HINT = {
    "customer_legal_name": "string",
    "billing_email": "string",
    "effective_date": "YYYY-MM-DD",
    "payment_terms": "e.g. 'Net 30'",
    "currency": "ISO code, e.g. USD",
    "stated_total": "number — the total contract value stated in the document",
    "signed": "boolean — is the agreement signed by both parties",
    "line_items": (
        "array of {description, quantity, unit_price, amount} — the itemized "
        "services/products billed, only if the contract actually lists them "
        "separately (e.g. an Exhibit/SOW pricing table). Return an empty array "
        "if the contract only states a single total with no breakdown — do not "
        "invent a split."
    ),
    "term_months": "number — contract term length in months, if stated (e.g. '24 months' -> 24), else null",
    "auto_renew": "boolean or null — does the contract auto-renew at term end (null if not stated either way)",
    "signatory_name": "string — full name of the CUSTOMER's (not Provider's) signatory, else null",
    "signatory_title": "string — title of the customer's signatory, else null",
    "signatory_date": "YYYY-MM-DD — date the customer signatory signed, if stated separately from effective_date, else null",
    "invoicing_schedule": (
        "one of 'monthly', 'quarterly', 'annual', 'one_time', or null — how the contract says billing "
        "should occur. Return 'one_time' only if the contract explicitly describes a single lump-sum "
        "invoice/payment. Return null (not a guess) if the contract states a total value but doesn't "
        "actually describe a billing cadence one way or the other."
    ),
}

_PROMPT = (
    "You extract billing terms from a signed services agreement so an invoice can be raised.\n"
    "Return ONLY a JSON object with these keys (no prose):\n"
    f"{json.dumps(_SCHEMA_HINT, indent=2)}\n"
    "Also return a parallel object 'confidence' mapping each key to a 0..1 number.\n"
    "For line_items, confidence reflects how confident you are in the itemized breakdown "
    "as a whole (0 if you returned an empty array because none was found).\n"
    "Wrap the whole thing as {\"fields\": {...}, \"confidence\": {...}}.\n\n"
    "AGREEMENT:\n"
)


@dataclass
class ExtractionResult:
    fields: dict = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    method: str = "unknown"  # "llm" | "heuristic"


def extract_anthropic(agreement_text: str, settings: Settings) -> ExtractionResult:
    res = _http.request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json_body={
            "model": settings.llm_model,
            "max_tokens": 3000,
            "messages": [{"role": "user", "content": _PROMPT + agreement_text}],
        },
    )
    text = "".join(block.get("text", "") for block in res.get("content", []))
    parsed = _extract_json(text)
    return ExtractionResult(
        fields=parsed.get("fields", {}),
        confidence={k: float(v) for k, v in parsed.get("confidence", {}).items()},
        method="llm (anthropic)",
    )


def extract_groq(agreement_text: str, settings: Settings) -> ExtractionResult:
    """Groq's chat completions endpoint is OpenAI-compatible — same prompt,
    different wire format and response shape. Useful as a free-tier live
    test path when Anthropic credits aren't available yet."""
    res = _http.request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "content-type": "application/json",
        },
        json_body={
            "model": settings.groq_model,
            "max_tokens": 3000,
            "messages": [{"role": "user", "content": _PROMPT + agreement_text}],
        },
    )
    text = res["choices"][0]["message"]["content"]
    parsed = _extract_json(text)
    return ExtractionResult(
        fields=parsed.get("fields", {}),
        confidence={k: float(v) for k, v in parsed.get("confidence", {}).items()},
        method="llm (groq)",
    )


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"LLM did not return JSON: {text[:300]}")
    return json.loads(text[start : end + 1])


# -- heuristic fallback, no API key needed -------------------------------- #
_DATE_RE = re.compile(
    r"\b(?:effective(?:\s+as of)?\s+date[:\s]*)?"
    r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_TERMS_RE = re.compile(r"\bNet\s?(\d{1,3})\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")
_CUSTOMER_RE = re.compile(r"(?:Customer|Client)\s*:\s*([A-Z][\w&.,'\- ]{2,60})")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SIGNED_HINTS = ("signature", "signed by", "/s/", "docusign", "duly executed")


def _parse_date(raw: str) -> str | None:
    from datetime import datetime

    for fmt in ("%B %d, %Y", "%B %d %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip().rstrip(","), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def extract_heuristic(agreement_text: str) -> ExtractionResult:
    fields: dict = {}
    conf: dict[str, float] = {}

    if m := _CUSTOMER_RE.search(agreement_text):
        fields["customer_legal_name"] = m.group(1).strip()
        conf["customer_legal_name"] = 0.7
    else:
        fields["customer_legal_name"] = None
        conf["customer_legal_name"] = 0.0

    if m := _EMAIL_RE.search(agreement_text):
        fields["billing_email"] = m.group(0)
        conf["billing_email"] = 0.6
    else:
        fields["billing_email"] = None
        conf["billing_email"] = 0.0

    if m := _DATE_RE.search(agreement_text):
        parsed = _parse_date(m.group(1))
        fields["effective_date"] = parsed
        conf["effective_date"] = 0.65 if parsed else 0.2
    else:
        fields["effective_date"] = None
        conf["effective_date"] = 0.0

    if m := _TERMS_RE.search(agreement_text):
        fields["payment_terms"] = f"Net {m.group(1)}"
        conf["payment_terms"] = 0.8
    else:
        fields["payment_terms"] = None
        conf["payment_terms"] = 0.0

    amounts = [float(a.replace(",", "")) for a in _AMOUNT_RE.findall(agreement_text)]
    if amounts:
        fields["stated_total"] = max(amounts)  # heuristic: the largest $ figure mentioned
        conf["stated_total"] = 0.5
    else:
        fields["stated_total"] = None
        conf["stated_total"] = 0.0
    fields["currency"] = "USD" if amounts else None
    conf["currency"] = 0.5 if amounts else 0.0

    lowered = agreement_text.lower()
    fields["signed"] = any(h in lowered for h in _SIGNED_HINTS)
    conf["signed"] = 0.6

    # No regex-based table parsing is built -- an itemized breakdown needs
    # real structure understanding a pattern match can't reliably do, so
    # this is an honest gap rather than an invented split.
    fields["line_items"] = []
    conf["line_items"] = 0.0

    # Same story for term length, auto-renew, and signatory details -- no
    # regex heuristic exists for these either, so they're an honest gap
    # in mock mode too, not just in the live LLM path.
    for key in ("term_months", "auto_renew", "signatory_name", "signatory_title", "signatory_date", "invoicing_schedule"):
        fields[key] = None
        conf[key] = 0.0

    return ExtractionResult(fields=fields, confidence=conf, method="heuristic")


# -- chunking for long documents ------------------------------------------ #
# ~4,000 tokens/chunk (4 chars/token heuristic), leaving headroom under
# Groq's free-tier 8,000 TPM limit once the ~300-token prompt/schema
# overhead and the model's response are added on top.
_CHUNK_CHAR_LIMIT = 16_000


def _split_into_chunks(text: str, limit: int = _CHUNK_CHAR_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        # a single paragraph longer than the whole limit (rare) gets hard-split
        if len(para) > limit:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            for i in range(0, len(para), limit):
                chunks.append(para[i : i + limit])
            continue
        if current_len + len(para) + 2 > limit and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _merge_results(results: list[ExtractionResult]) -> ExtractionResult:
    if len(results) == 1:
        return results[0]

    merged_fields: dict = {}
    merged_conf: dict[str, float] = {}
    for key in _SCHEMA_HINT:
        best_conf, best_val = 0.0, None
        for r in results:
            c = r.confidence.get(key, 0.0) or 0.0
            v = r.fields.get(key)
            # v == [] means "no items found in this chunk", same as None --
            # but v == False (e.g. signed) must still be allowed through.
            if v is not None and v != [] and c > best_conf:
                best_conf, best_val = c, v
        merged_fields[key] = best_val
        merged_conf[key] = best_conf

    methods = {r.method for r in results}
    method = f"{methods.pop()} (chunked x{len(results)})" if len(methods) == 1 else f"mixed (chunked x{len(results)})"
    return ExtractionResult(fields=merged_fields, confidence=merged_conf, method=method)


def extract(agreement_text: str, settings: Settings) -> ExtractionResult:
    if settings.llm_mode != "live":
        return extract_heuristic(agreement_text)

    call = extract_groq if settings.llm_provider == "groq" else extract_anthropic
    chunks = _split_into_chunks(agreement_text)
    results = [call(chunk, settings) for chunk in chunks]
    return _merge_results(results)
