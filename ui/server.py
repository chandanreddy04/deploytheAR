#!/usr/bin/env python3
"""Serves emma_workspace.html plus real API endpoints backed by the actual
HubSpot + LLM pipeline, QuickBooks, and Gmail.

Real: Deal Pipeline (contract extraction), Cash Application (QBO payment
matching), Collections (overdue detection + dunning email), and real
invoice delivery (PDF + email). Approvals, Incidents, and Audit are still
the existing mock data in emma_workspace.html; those workflows don't
exist yet.

Real deals/cash/collections are computed once and cached both in memory
and durably via db.py -- SQLite (rally_state.db, gitignored, project root)
locally, or Postgres automatically when DATABASE_URL is set (e.g. deployed
on Railway). API calls aren't free and this project has already hit real
rate limits calling too often, and used to re-pay that cost on every
restart since nothing survived the process exiting. Hit
/api/deals?refresh=1 (etc.) to force a recompute.

Usage:
    python server.py [port]        # default port 8600, or $PORT if set (Railway)
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).parent.parent  # Hubspot-Integration/ -- config.py, integrations/, etc. live here
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import load_settings
from content_extractor import extract
from db import db_load as _db_load, db_save as _db_save
from pdf_text import extract_text

UI_DIR = Path(__file__).parent

_cache: dict = {"deals": None, "computed_at": 0.0, "error": None}
_cash_cache: dict = {"cash": None, "computed_at": 0.0, "error": None}
_collections_cache: dict = {"collections": None, "computed_at": 0.0, "error": None}
_deal_recon_cache: dict = {"data": None, "computed_at": 0.0, "error": None}
_CACHE_TTL_SECONDS = 300  # re-fetch at most every 5 minutes even without ?refresh=1

# Durable cache instead of the in-memory-only caches above going stale on
# every restart -- a restart used to always force a full re-extraction of
# every real deal through Groq (the exact rate-limit storm that's hit this
# project more than once), because nothing survived the process exiting.
# db.py (imported above) handles SQLite locally or Postgres automatically
# when DATABASE_URL is set (e.g. deployed on Railway) -- same
# cache_store(key, data, computed_at, error) shape either way. This is a
# "latest known state" store, not a history log -- each write replaces
# the previous one for that key, on purpose.

# Design-doc dunning cadence (Section 10), each threshold the minimum
# days-overdue for that stage -- checked highest-first so an invoice lands
# in the latest stage it qualifies for. Labels match the UI's existing
# DUNNING_STAGES kanban columns exactly (emma_workspace.html) so real
# entries land in the right column instead of creating stray ones.
_DUNNING_STAGES = [
    (90, "Writeoff review"),
    (60, "Escalated"),
    (45, "Final demand"),
    (22, "Second notice"),
    (8, "First notice"),
    (1, "Reminder"),
]


def _dunning_stage(days_overdue: int) -> str:
    for threshold, label in _DUNNING_STAGES:
        if days_overdue >= threshold:
            return label
    return "Reminder"


# Phase 4 automated-reminder cadence, per the user's revised brief: one
# reminder 7 days before the deadline, one on the deadline itself, one 14
# days after. Independent of _DUNNING_STAGES above (that's the longer-run
# kanban bucket an overdue invoice visually sits in; this is specifically
# what gates an automated send). days_overdue is signed here -- negative
# means "days until due," 0 is due-today, positive is overdue -- so each
# threshold is checked highest-first the same way _dunning_stage() is.
# Each invoice gets at most one email per stage, ever -- tracked in
# reminder_state.json, not recomputed from days-overdue alone, so a stage
# already sent never gets re-sent just because the cycle runs again.
_REMINDER_THRESHOLDS = [(14, "final"), (0, "due_today"), (-7, "pre_due")]


def _reminder_stage(days_overdue: int) -> str | None:
    for threshold, stage in _REMINDER_THRESHOLDS:
        if days_overdue >= threshold:
            return stage
    return None


# Legacy path from before reminder state moved into rally_state.db --
# only read once, to migrate any real sent-reminder history forward
# instead of silently losing it.
_LEGACY_REMINDER_STATE_PATH = UI_DIR / "reminder_state.json"


def _load_reminder_state() -> dict:
    data, _, _ = _db_load("reminder_state")
    if data is not None:
        return data
    try:
        legacy = json.loads(_LEGACY_REMINDER_STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    _save_reminder_state(legacy)  # migrate once, so this branch is never hit again
    return legacy


def _save_reminder_state(state: dict) -> None:
    _db_save("reminder_state", state, time.time(), None)


def _prio_for(amount: float) -> str:
    if amount >= 50_000:
        return "high"
    if amount >= 15_000:
        return "med"
    return "low"


def _fmt_signed_date(iso_date: str | None) -> str:
    if not iso_date:
        return "—"
    try:
        return datetime.fromisoformat(iso_date).strftime("%b %d, %Y")
    except ValueError:
        return iso_date


def _build_line_items(raw: list | None) -> list[dict]:
    """Reshape the extractor's {description, quantity, unit_price, amount}
    dicts into the {desc, qty, unit, amount} the UI's invoice tables expect.
    Returns [] when nothing was extracted -- the UI already renders that as
    an honest "not extracted" gap rather than a fabricated split."""
    if not raw:
        return []
    items = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        items.append({
            "desc": it.get("description") or "—",
            "qty": it.get("quantity") if it.get("quantity") is not None else 1,
            "unit": it.get("unit_price") if it.get("unit_price") is not None else 0,
            "amount": it.get("amount") if it.get("amount") is not None else 0,
        })
    return items


def _build_signatories(fields: dict) -> list[str]:
    """One entry for the customer's signatory -- only when the extractor
    actually found a name. Provider-side (Rally) signatory isn't extracted
    from the customer's contract, so it's never invented here either.

    Joined with an em dash, not a comma: the UI's edit-mode signatories
    field treats a comma as the separator between multiple different
    signatories (join(', ') / split(',') round-trip) -- a literal comma
    inside one signatory's own "Name, Title" string would get silently
    split into two fake signatories on the next save.
    """
    name = fields.get("signatory_name")
    if not name:
        return []
    title = fields.get("signatory_title")
    return [f"{name} — {title}" if title else name]


def _build_deal(deal, agreement, fields: dict, confidence: dict, page_count: int) -> dict:
    # A field the extractor honestly left null/empty (e.g. no itemized line
    # items, no separately-stated signatory date) isn't a low-confidence
    # guess -- it's a correct "not in this document" answer, and the model
    # is inconsistent about what confidence number to attach to that empty
    # answer (0.0 one run, ~1.0 the next, for the exact same absent field).
    # Averaging that noise in with real extracted values caused deals with
    # perfect core-field confidence to intermittently flip into "Failed
    # extraction" just because e.g. auto-renew wasn't stated. Only fields
    # that actually found a value count toward the review-gating score.
    confs = [
        c for k, c in confidence.items()
        if c is not None and fields.get(k) not in (None, "", [])
    ]
    avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
    total = fields.get("stated_total") or deal.amount or 0
    billing_email = deal.billing_email or fields.get("billing_email") or ""
    party_to_domain = billing_email.split("@")[-1] if "@" in billing_email else "—"

    return {
        "id": deal.deal_id,
        "c": fields.get("customer_legal_name") or deal.company_legal_name or deal.deal_name,
        "amt": total,
        "stage": "extract",  # honest: extraction is what's real; draft/approval/sent/tracking/paid aren't built
        "days": 0,
        "prio": _prio_for(total),
        "owner": "—",  # not fetched from HubSpot today -- no deal-owner API call exists yet
        "terms": fields.get("payment_terms") or "—",
        "extractedAt": datetime.now().strftime("%b %d, %H:%M"),
        "conf": avg_conf,
        "real": True,
        "lineItems": _build_line_items(fields.get("line_items")),
        "termMonths": fields.get("term_months"),
        "autoRenew": fields.get("auto_renew"),
        "invoicingSchedule": fields.get("invoicing_schedule"),  # 'monthly'|'quarterly'|'annual'|'one_time'|None
        "agreementRef": deal.agreement_ref,  # lets the UI re-fetch the real PDF bytes on demand
        "billingEmail": billing_email or None,  # real send-to for invoice delivery; None is an honest gap, not fabricated
        "contract": {
            "filename": agreement.filename,
            "signed_date": _fmt_signed_date(fields.get("effective_date")),
            "pages": page_count,
            "size_kb": round(len(agreement.content) / 1024),
            "party_from": "Rally, Inc.",
            "party_from_domain": "rally.example",
            "party_to": fields.get("customer_legal_name") or deal.company_legal_name,
            "party_to_domain": party_to_domain,
            "signatories": _build_signatories(fields),
            "mock_body": "",  # filled in below with real extracted text
        },
    }


def compute_deals() -> tuple[list[dict], list[tuple[str, str, str]]]:
    from integrations.hubspot import HubSpotClient, NoAgreementFound
    import pdfplumber
    import io

    settings = load_settings()
    if settings.hubspot_mode != "live":
        raise RuntimeError("HUBSPOT_MODE is not 'live' -- set it in .env to fetch real deals")

    client = HubSpotClient(settings.hubspot_token)
    out = []
    skipped: list[tuple[str, str, str]] = []  # (deal_id, deal_name, error)
    for deal in client.list_closed_deals(limit=10):
        if not deal.agreement_ref:
            continue
        try:
            agreement = client.get_agreement(deal.agreement_ref)
        except NoAgreementFound:
            continue

        parsed = extract_text(agreement.content)
        if parsed.needs_ocr:
            continue
        try:
            result = extract(parsed.text, settings)
        except Exception as exc:
            # One deal hitting a rate limit (or any other extraction
            # failure) shouldn't discard every other deal that already
            # succeeded -- skip it and keep going. get_deals() surfaces
            # this in the error field so it's never silently swallowed,
            # and falls back to this deal's last known-good extraction
            # (deal_id is what lets it find that prior entry).
            skipped.append((deal.deal_id, deal.deal_name, str(exc)))
            continue

        try:
            with pdfplumber.open(io.BytesIO(agreement.content)) as pdf:
                page_count = len(pdf.pages)
        except Exception:
            page_count = 1

        built = _build_deal(deal, agreement, result.fields, result.confidence, page_count)
        built["contract"]["mock_body"] = parsed.text[:4000]
        out.append(built)

    return out, skipped


def get_deals(force_refresh: bool = False) -> tuple[list[dict], str | None]:
    # Cold in-memory cache (e.g. right after a restart) -- try the durable
    # snapshot from SQLite before deciding a live recompute is needed, so a
    # restart doesn't force a full Groq re-extraction of every deal just
    # because the process is new.
    if _cache["deals"] is None and not force_refresh:
        data, computed_at, error = _db_load("deals")
        if data is not None:
            _cache["deals"], _cache["computed_at"], _cache["error"] = data, computed_at, error

    stale = time.time() - _cache["computed_at"] > _CACHE_TTL_SECONDS
    if force_refresh or _cache["deals"] is None or stale:
        prior_by_id = {d["id"]: d for d in (_cache["deals"] or [])}
        try:
            deals, skipped = compute_deals()
            # A deal whose extraction fails THIS round shouldn't vanish if
            # we already have a good extraction for it from before -- that
            # used to cascade into Cash Application/Collections silently
            # excluding that deal's real invoice too, since those are
            # scoped to "customers with a currently-real deal". Recover the
            # last known-good entry for it instead of just dropping it.
            recovered_ids = set()
            for deal_id, name, err in skipped:
                old = prior_by_id.get(deal_id)
                if old:
                    deals.append(old)
                    recovered_ids.add(deal_id)
            _cache["deals"] = deals
            # A per-deal failure (e.g. one deal hit a rate limit) doesn't
            # invalidate the ones that succeeded -- surface it as an
            # honest, non-fatal note instead of discarding everything.
            _cache["error"] = (
                "; ".join(
                    f"{name}: {err}" + (" (kept last known-good extraction)" if deal_id in recovered_ids else "")
                    for deal_id, name, err in skipped
                ) if skipped else None
            )
        except Exception as exc:
            _cache["error"] = str(exc)
            if _cache["deals"] is None:
                _cache["deals"] = []
        _cache["computed_at"] = time.time()
        _db_save("deals", _cache["deals"], _cache["computed_at"], _cache["error"])
    return _cache["deals"], _cache["error"]


def _fmt_when(iso_date: str) -> str:
    try:
        return datetime.fromisoformat(iso_date).strftime("%b %d · %H:%M")
    except (ValueError, TypeError):
        return iso_date or "—"


def compute_cash() -> dict:
    """Real QuickBooks payment/invoice data for the Cash Application tab.

    No fuzzy matching is invented here -- "matched" means QuickBooks' own
    Payment.LinkedTxn already ties that payment to an invoice (an
    authoritative fact from QBO, not a guess we're making), so confidence
    is honestly 1.0. "Unmatched" means no LinkedTxn exists; since no
    matching algorithm has been built, confidence is honestly 0.0 and the
    candidate-invoice list is genuinely empty rather than fabricated.
    "Partial" is derived directly from real invoice Balance vs TotalAmt.

    Scoped to customers that match a real HubSpot deal (by the same
    normalized-name key used for deal reconciliation) -- a QBO sandbox
    company comes with its own unrelated sample customers (Amy's Bird
    Sanctuary, Cool Cars, etc.) that would otherwise flood this view and
    bury the transactions that actually matter for this app.
    """
    from integrations.qbo import QboClient

    settings = load_settings()
    if settings.qbo_mode != "live":
        raise RuntimeError("QBO_MODE is not 'live' -- set it in .env to fetch real cash data")

    deals, _ = get_deals()
    real_customer_keys = {_normalize_customer_name(d.get("c", "")) for d in deals if d.get("real")}
    real_customer_keys.discard("")

    client = QboClient(settings)
    all_payments, _ = client.list_payments_since(None)
    all_invoices = client.list_invoices(limit=50)
    payments = [p for p in all_payments if _normalize_customer_name(p.customer_name) in real_customer_keys]
    invoices = [i for i in all_invoices if _normalize_customer_name(i.get("CustomerRef", {}).get("name", "")) in real_customer_keys]

    matched, unmatched = [], []
    for p in payments:
        row = {
            "qb": p.payment_id,
            "c": p.customer_name or "(no customer name on payment)",
            "amt": round(p.total_amount, 2),
            "when": _fmt_when(p.txn_date),
            "real": True,
        }
        if p.linked_invoice_numbers:
            row["inv"] = ", ".join(p.linked_invoice_numbers)
            row["conf"] = 1.0  # QBO's own LinkedTxn -- a fact, not a fuzzy guess
            matched.append(row)
        else:
            row["inv"] = "—"
            row["conf"] = 0.0  # honestly zero: no matching algorithm exists yet
            row["cands"] = []  # not fabricated -- genuinely no suggestions to offer
            unmatched.append(row)

    partial = []
    for inv in invoices:
        total = float(inv.get("TotalAmt") or 0)
        balance = float(inv.get("Balance") or 0)
        if 0 < balance < total:
            partial.append({
                "qb": f"QBO-Inv-{inv.get('Id')}",  # an invoice reference, not a payment ID -- QBO's query API doesn't return a payment ID per partially-paid invoice
                "c": inv.get("CustomerRef", {}).get("name", "—"),
                "amt": round(total - balance, 2),
                "due": round(total, 2),
                "inv": inv.get("DocNumber", "—"),
                "conf": 1.0,  # derived directly from real Balance vs TotalAmt, not a guess
                "when": "—",  # list_invoices doesn't carry a payment date
                "real": True,
            })

    return {"matched": matched, "unmatched": unmatched, "partial": partial}


def get_cash(force_refresh: bool = False) -> tuple[dict | None, str | None]:
    if _cash_cache["cash"] is None and not force_refresh:
        data, computed_at, error = _db_load("cash")
        if data is not None:
            _cash_cache["cash"], _cash_cache["computed_at"], _cash_cache["error"] = data, computed_at, error

    stale = time.time() - _cash_cache["computed_at"] > _CACHE_TTL_SECONDS
    if force_refresh or _cash_cache["cash"] is None or stale:
        try:
            _cash_cache["cash"] = compute_cash()
            _cash_cache["error"] = None
        except Exception as exc:
            _cash_cache["error"] = str(exc)
            if _cash_cache["cash"] is None:
                _cash_cache["cash"] = {"matched": [], "unmatched": [], "partial": []}
        _cash_cache["computed_at"] = time.time()
        _db_save("cash", _cash_cache["cash"], _cash_cache["computed_at"], _cash_cache["error"])
    return _cash_cache["cash"], _cash_cache["error"]


_LEGAL_SUFFIXES = (" inc", " llc", " corp", " corporation", " ltd", " co", " partners", " solutions", " group")


def _normalize_customer_name(name: str) -> str:
    """Loose match key for a real deal's customer name against a QuickBooks
    customer name -- lowercase, strip punctuation and common legal
    suffixes, so a human manually creating the matching QBO customer
    doesn't need a byte-for-byte identical name for this to find it."""
    if not name:
        return ""
    n = re.sub(r"[,.]", "", name.lower()).strip()
    for suffix in _LEGAL_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def compute_deal_reconciliation() -> list[dict]:
    """Reconciles each real deal against real QuickBooks invoice data --
    honestly shows paid/partial/unpaid, or 'not yet invoiced in
    QuickBooks' when nothing matches at all.

    Matches by customer name only (loosely normalized), not by invoice
    number: this app never writes invoices into QuickBooks (deliberately
    -- no dummy/write access), so there's no shared invoice ID to match
    on. A real match only happens once a human has created a real
    QuickBooks customer + invoice under a name that resolves to the same
    normalized key as the real deal's customer.
    """
    from integrations.qbo import QboClient

    settings = load_settings()
    if settings.qbo_mode != "live":
        raise RuntimeError("QBO_MODE is not 'live' -- set it in .env to fetch real cash data")

    deals, _ = get_deals()
    client = QboClient(settings)
    qbo_invoices = client.list_invoices(limit=50)

    by_customer: dict[str, list[dict]] = {}
    for inv in qbo_invoices:
        key = _normalize_customer_name(inv.get("CustomerRef", {}).get("name", ""))
        if key:
            by_customer.setdefault(key, []).append(inv)

    out = []
    for d in deals:
        if not d.get("real"):
            continue
        key = _normalize_customer_name(d.get("c", ""))
        matches = by_customer.get(key, [])
        if not matches:
            out.append({
                "dealId": d["id"], "c": d["c"], "amt": d["amt"],
                "status": "not_invoiced", "statusLabel": "Not yet invoiced in QuickBooks",
                "qboTotal": None, "qboBalance": None, "qboDocNumber": None,
            })
            continue
        # If more than one real QBO invoice matches this customer, use
        # whichever total is closest to our real deal amount.
        best = min(matches, key=lambda i: abs(float(i.get("TotalAmt") or 0) - d["amt"]))
        total = float(best.get("TotalAmt") or 0)
        balance = float(best.get("Balance") or 0)
        if balance <= 0:
            status, label = "paid", "Fully paid"
        elif balance < total:
            status, label = "partial", "Partially paid"
        else:
            status, label = "unpaid", "Unpaid"
        out.append({
            "dealId": d["id"], "c": d["c"], "amt": d["amt"],
            "status": status, "statusLabel": label,
            "qboTotal": round(total, 2), "qboBalance": round(balance, 2),
            "qboDocNumber": best.get("DocNumber", "—"),
        })
    return out


def get_deal_reconciliation(force_refresh: bool = False) -> tuple[list[dict], str | None]:
    if _deal_recon_cache["data"] is None and not force_refresh:
        data, computed_at, error = _db_load("deal_reconciliation")
        if data is not None:
            _deal_recon_cache["data"], _deal_recon_cache["computed_at"], _deal_recon_cache["error"] = data, computed_at, error

    stale = time.time() - _deal_recon_cache["computed_at"] > _CACHE_TTL_SECONDS
    if force_refresh or _deal_recon_cache["data"] is None or stale:
        try:
            _deal_recon_cache["data"] = compute_deal_reconciliation()
            _deal_recon_cache["error"] = None
        except Exception as exc:
            _deal_recon_cache["error"] = str(exc)
            if _deal_recon_cache["data"] is None:
                _deal_recon_cache["data"] = []
        _deal_recon_cache["computed_at"] = time.time()
        _db_save("deal_reconciliation", _deal_recon_cache["data"], _deal_recon_cache["computed_at"], _deal_recon_cache["error"])
    return _deal_recon_cache["data"], _deal_recon_cache["error"]


def _real_outstanding_invoices() -> list[dict]:
    """Real QuickBooks invoices with a positive balance, scoped to
    customers matching a real HubSpot deal (same scoping as compute_cash()
    -- otherwise QBO's own unrelated sample invoices would show up here
    too), each with a real due date and 'dpd' (days overdue, signed --
    negative means not yet due, 0 is due today) computed against today's
    actual date.

    No window filtering here -- that's each caller's own concern.
    compute_collections() only wants invoices within 7 days of the
    deadline through however overdue (its reminder-cadence window);
    compute_ar_aging() (ar_aging_report.py) wants every outstanding
    invoice, including ones not due for months, since an AR aging report
    has to account for all of it, not just what's near-due.
    """
    from datetime import date

    from integrations.qbo import QboClient

    settings = load_settings()
    if settings.qbo_mode != "live":
        raise RuntimeError("QBO_MODE is not 'live' -- set it in .env to fetch real invoice data")

    deals, _ = get_deals()
    real_customer_keys = {_normalize_customer_name(d.get("c", "")) for d in deals if d.get("real")}
    real_customer_keys.discard("")

    client = QboClient(settings)
    all_invoices = client.list_invoices(limit=50)
    invoices = [i for i in all_invoices if _normalize_customer_name(i.get("CustomerRef", {}).get("name", "")) in real_customer_keys]
    today = date.today()

    out = []
    for inv in invoices:
        balance = float(inv.get("Balance") or 0)
        if balance <= 0:
            continue
        due_raw = inv.get("DueDate")
        if not due_raw:
            continue
        try:
            due_date = date.fromisoformat(due_raw)
        except ValueError:
            continue
        out.append({
            "qboId": inv.get("Id"),
            "customer": inv.get("CustomerRef", {}).get("name") or "—",
            "docnum": inv.get("DocNumber", "—"),
            "balance": balance,
            "dueDate": due_date,
            "dpd": (today - due_date).days,
            "email": (inv.get("BillEmail") or {}).get("Address"),
        })
    return out


def compute_collections() -> list[dict]:
    """Real overdue invoices from the QuickBooks sandbox, each paired with a
    draft dunning email built from real invoice/customer data.

    Nothing here is ever sent automatically -- this only computes the list
    and the draft text. An explicit POST to /api/collections/<id>/send,
    itself only ever triggered by a person clicking Send in the UI, is
    what actually calls the Gmail client. (run_reminder_cycle() is the one
    exception -- also only ever reached via an explicit person-triggered
    POST, just a batch one instead of a per-row Send click.)

    An invoice with no BillEmail on file gets email=None -- shown as an
    honest gap in the UI, never a guessed or fabricated address.

    Window covers -7 (7 days before the deadline) through however far
    overdue an invoice actually is, matching the 3-stage reminder cadence
    in _reminder_stage() -- "dpd" (days overdue) is signed: negative means
    still upcoming, 0 is due today, positive is overdue.
    """
    out = []
    for r in _real_outstanding_invoices():
        days_overdue = r["dpd"]
        if days_overdue < -7:
            continue  # more than 7 days before the deadline -- not due for a reminder yet

        stage = _dunning_stage(max(days_overdue, 0))
        qbo_id, customer, docnum, balance, due_date, email = r["qboId"], r["customer"], r["docnum"], r["balance"], r["dueDate"], r["email"]

        if days_overdue < 0:
            days_until = -days_overdue
            subject = f"Upcoming invoice due soon: #{docnum} — due in {days_until} day{'s' if days_until != 1 else ''}"
            body = (
                f"Hi {customer},\n\n"
                f"This is a heads-up that Invoice #{docnum} for ${balance:,.2f} is due on "
                f"{due_date.isoformat()} ({days_until} day{'s' if days_until != 1 else ''} from now).\n\n"
                "No action needed if payment is already scheduled. Please reach out if you have "
                "any questions.\n\n"
                "Thank you,\nRally, Inc."
            )
        elif days_overdue == 0:
            subject = f"Invoice due today: #{docnum}"
            body = (
                f"Hi {customer},\n\n"
                f"This is a reminder that Invoice #{docnum} for ${balance:,.2f} is due today, "
                f"{due_date.isoformat()}.\n\n"
                "Please remit payment at your earliest convenience. If you've already sent "
                "payment, please disregard this message.\n\n"
                "Thank you,\nRally, Inc."
            )
        else:
            subject = f"Payment reminder: Invoice #{docnum} — {days_overdue} days overdue"
            body = (
                f"Hi {customer},\n\n"
                f"This is a reminder that Invoice #{docnum} for ${balance:,.2f} was due on "
                f"{due_date.isoformat()} and remains unpaid ({days_overdue} days overdue).\n\n"
                "Please remit payment at your earliest convenience. If you've already sent "
                "payment, please disregard this message.\n\n"
                "Thank you,\nRally, Inc."
            )

        out.append({
            "id": f"COLL-{qbo_id}",
            "inv": docnum,
            "c": customer,
            "amt": round(balance, 2),
            "dueDate": due_date.isoformat(),
            "dpd": days_overdue,
            "stage": stage,
            "email": email,
            "draftSubject": subject,
            "draftBody": body,
            "sent": False,
            "sentAt": None,
            "real": True,
        })

    out.sort(key=lambda r: -r["dpd"])
    return out


def get_collections(force_refresh: bool = False) -> tuple[list[dict], str | None]:
    # Loading the durable snapshot first (when cold) means the sent/sentAt
    # preservation below also survives a restart, not just a same-process
    # recompute -- a restart used to forget which emails had already gone
    # out, risking a real duplicate send.
    if _collections_cache["collections"] is None and not force_refresh:
        data, computed_at, error = _db_load("collections")
        if data is not None:
            _collections_cache["collections"], _collections_cache["computed_at"], _collections_cache["error"] = data, computed_at, error

    stale = time.time() - _collections_cache["computed_at"] > _CACHE_TTL_SECONDS
    if force_refresh or _collections_cache["collections"] is None or stale:
        try:
            fresh = compute_collections()
            # Preserve sent/sentAt/messageId across recomputes -- a refresh
            # shouldn't forget that something was already sent.
            prior = {c["id"]: c for c in (_collections_cache["collections"] or [])}
            for row in fresh:
                old = prior.get(row["id"])
                if old and old.get("sent"):
                    row["sent"] = True
                    row["sentAt"] = old.get("sentAt")
                    row["messageId"] = old.get("messageId")
                    row["sentTo"] = old.get("sentTo")
            _collections_cache["collections"] = fresh
            _collections_cache["error"] = None
        except Exception as exc:
            _collections_cache["error"] = str(exc)
            if _collections_cache["collections"] is None:
                _collections_cache["collections"] = []
        _collections_cache["computed_at"] = time.time()
        _db_save("collections", _collections_cache["collections"], _collections_cache["computed_at"], _collections_cache["error"])
    return _collections_cache["collections"], _collections_cache["error"]


def compute_due_reminders(collections: list[dict], state: dict) -> list[dict]:
    """Which real collections rows need an automated reminder sent right
    now: real invoice, at one of the 3 cadence checkpoints (7 days before
    the deadline, the day of the deadline, or 14 days after), and that
    specific stage hasn't already been recorded as sent in
    reminder_state.json.

    "final" (14 days after) gets an escalated subject/body built here
    rather than reusing the row's own draftSubject/draftBody as-is, so it
    reads more urgent than the pre_due/due_today stages. Those two reuse
    compute_collections()'s own draft as-is -- it's already worded
    correctly for "upcoming" vs "due today" there.
    """
    due = []
    for row in collections:
        if not row.get("real"):
            continue
        stage = _reminder_stage(row["dpd"])
        if not stage:
            continue
        if state.get(row["id"], {}).get(stage):
            continue  # this stage already sent for this invoice -- never resend it
        if stage == "final":
            subject = f"FINAL NOTICE: Invoice #{row['inv']} — {row['dpd']} days overdue"
            body = (
                f"Hi {row['c']},\n\n"
                f"This is a final reminder that Invoice #{row['inv']} for ${row['amt']:,.2f} was due on "
                f"{row['dueDate']} and remains unpaid ({row['dpd']} days overdue). Earlier reminders were "
                "already sent.\n\n"
                "Please remit payment immediately to avoid further collections action. If you've already "
                "sent payment, please disregard this message.\n\n"
                "Thank you,\nRally, Inc."
            )
        else:
            subject, body = row["draftSubject"], row["draftBody"]
        due.append({**row, "reminderStage": stage, "subject": subject, "body": body})
    return due


def run_reminder_cycle() -> dict:
    """The one reminder-sending system for this app: sends every real
    reminder currently due (7 days before deadline / day of deadline / 14
    days after) in a single batch, instead of a person clicking Send on
    each one individually. Only ever reached via an explicit POST from the
    UI's own "Run reminder cycle" button (or `ar_aging_report.py --send`)
    -- there is still no background scheduler anywhere in this codebase.

    (This used to be two separate systems -- this -7/0/+14 cadence, plus a
    simpler "> N days overdue, no dedup" one with its own override-email
    test mode. Merged into this one on request; the override behavior
    below is what's left of the second one.)

    Per settings.outstanding_reminder_override_email (default
    capstnprjt@gmail.com), every email can be redirected to a fixed test
    address instead of the customer's real BillEmail -- set
    OUTSTANDING_REMINDER_OVERRIDE_EMAIL=real in .env to send to each
    invoice's actual billing email instead (this is already the current
    setting). The overridden email body is prefixed with who it was
    actually meant for, so a redirected test inbox stays legible.

    reminder_state.json is the only thing that has to survive a restart for
    this to be safe -- without it, a restart would forget a stage was
    already sent and could email the same customer twice.
    """
    settings = load_settings()
    if settings.gmail_mode != "live":
        raise RuntimeError("GMAIL_MODE is not 'live' -- set it in .env to send real reminder email")

    collections, coll_error = get_collections()
    state = _load_reminder_state()
    due = compute_due_reminders(collections, state)

    from integrations.gmail import GmailClient

    client = GmailClient(settings)
    override = settings.outstanding_reminder_override_email
    sent, failed, skipped = [], [], []
    for row in due:
        real_to = (row.get("email") or "").strip()
        to = override or real_to
        if not to:
            skipped.append({"id": row["id"], "c": row["c"], "reason": "no billing email on file and no override configured"})
            continue
        body = row["body"]
        if override:
            body = f"[TEST MODE — intended recipient: {row['c']} <{real_to or 'no email on file'}>]\n\n{body}"
        try:
            message_id = client.send_email(to, row["subject"], body)
        except Exception as exc:
            failed.append({"id": row["id"], "c": row["c"], "reason": str(exc)})
            continue
        state.setdefault(row["id"], {})[row["reminderStage"]] = datetime.now().isoformat()
        sent.append({"id": row["id"], "c": row["c"], "stage": row["reminderStage"], "to": to, "intendedFor": real_to or None, "messageId": message_id})

    if sent:
        _save_reminder_state(state)

    return {
        "checked": len(collections), "due": len(due),
        "sent": sent, "failed": failed, "skipped": skipped,
        "override": override, "collectionsError": coll_error,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "EmmaUI/0.1"

    def log_message(self, fmt, *args) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send_json(self, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_deal_pdf(self, deal_id: str) -> None:
        """Re-fetches the real signed PDF from HubSpot so the UI can show
        the actual document (real pages, real download) instead of a
        flattened text preview. Not cached -- this endpoint is only hit
        when a user actually opens/downloads a contract, not on every
        page load."""
        deals, _ = get_deals()
        deal = next((d for d in deals if d["id"] == deal_id), None)
        if not deal or not deal.get("agreementRef"):
            self.send_response(404)
            self.end_headers()
            return

        from integrations.hubspot import HubSpotClient

        settings = load_settings()
        client = HubSpotClient(settings.hubspot_token)
        try:
            agreement = client.get_agreement(deal["agreementRef"])
        except Exception as exc:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            body = str(exc).encode("utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", f'inline; filename="{agreement.filename}"')
        self.send_header("Content-Length", str(len(agreement.content)))
        self.end_headers()
        try:
            self.wfile.write(agreement.content)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # Harmless: the browser cancelled the request mid-stream (modal
            # closed, a different deal's PDF opened, tab navigated away).
            # Nothing to clean up -- just don't let it print a scary
            # traceback for what is normal client behavior.
            pass

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/deals":
            force = parse_qs(parsed.query).get("refresh", ["0"])[0] == "1"
            deals, error = get_deals(force_refresh=force)
            self._send_json({"deals": deals, "error": error, "computed_at": _cache["computed_at"]})
            return

        if path == "/api/cash":
            force = parse_qs(parsed.query).get("refresh", ["0"])[0] == "1"
            cash, error = get_cash(force_refresh=force)
            self._send_json({**cash, "error": error, "computed_at": _cash_cache["computed_at"]})
            return

        if path == "/api/collections":
            force = parse_qs(parsed.query).get("refresh", ["0"])[0] == "1"
            collections, error = get_collections(force_refresh=force)
            self._send_json({"collections": collections, "error": error, "computed_at": _collections_cache["computed_at"]})
            return

        if path == "/api/deals/reconciliation":
            force = parse_qs(parsed.query).get("refresh", ["0"])[0] == "1"
            recon, error = get_deal_reconciliation(force_refresh=force)
            self._send_json({"reconciliation": recon, "error": error, "computed_at": _deal_recon_cache["computed_at"]})
            return

        if path.startswith("/api/deals/") and path.endswith("/pdf"):
            deal_id = path[len("/api/deals/"):-len("/pdf")]
            self._serve_deal_pdf(deal_id)
            return

        if path == "/":
            path = "/emma_workspace.html"

        target = (UI_DIR / path.lstrip("/")).resolve()
        try:
            target.relative_to(UI_DIR.resolve())
        except ValueError:
            self.send_response(404)
            self.end_headers()
            return
        if not target.is_file():
            self.send_response(404)
            self.end_headers()
            return

        data = target.read_bytes()
        content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_collection_email(self, coll_id: str, overrides: dict) -> None:
        """Actually sends one dunning email via Gmail. Only ever reached by
        a real POST from the UI's own Send button -- there is no scheduler
        or automatic trigger anywhere in this codebase that calls this.

        Uses whatever the user edited in the draft box (to/subject/body),
        falling back to the computed draft only for fields left untouched
        -- e.g. redirecting a test send to a real inbox instead of the
        sandbox's fake @intuit.com addresses, without losing the real
        invoice data in the rest of the message."""
        collections, _ = get_collections()
        row = next((c for c in collections if c["id"] == coll_id), None)
        if not row:
            self._send_json({"ok": False, "error": "Collection entry not found"})
            return

        to = (overrides.get("to") or row.get("email") or "").strip()
        if not to:
            self._send_json({"ok": False, "error": "No email address on file for this invoice — cannot send"})
            return
        subject = overrides.get("subject") or row["draftSubject"]
        body_text = overrides.get("body") or row["draftBody"]

        settings = load_settings()
        if settings.gmail_mode != "live":
            self._send_json({"ok": False, "error": "GMAIL_MODE is not 'live' -- set it in .env to send real email"})
            return

        from integrations.gmail import GmailClient

        client = GmailClient(settings)
        try:
            message_id = client.send_email(to, subject, body_text)
        except Exception as exc:
            # Echo back exactly what "to" was attempted -- Gmail's own error
            # text doesn't include it, which makes an "Invalid To header"
            # otherwise impossible to trace back to what was actually typed.
            self._send_json({"ok": False, "error": f"{exc} (attempted To: {to!r})"})
            return

        row["sent"] = True
        row["sentAt"] = datetime.now().isoformat()
        row["messageId"] = message_id
        row["sentTo"] = to
        self._send_json({"ok": True, "messageId": message_id, "to": to})

    def _send_invoice_email(self, payload: dict) -> None:
        """Actually emails a real invoice PDF to the customer. Only ever
        reached by a real POST from the UI's own Send button.

        Invoices only ever exist in the browser's own JS state (built
        client-side from an approved deal, never persisted server-side the
        way deals/cash/collections are) -- so unlike those, this endpoint
        can't look anything up by id. The client sends the full invoice
        data it already has, and this only lays it out into a real PDF and
        sends it; it never invents a customer, address, or amount that
        wasn't already in that payload."""
        to = (payload.get("to") or "").strip()
        if not to:
            self._send_json({"ok": False, "error": "No recipient email address — cannot send"})
            return

        inv = payload.get("invoice") or {}
        required = ("number", "issueDate", "dueDate", "amount")
        missing = [k for k in required if not inv.get(k) and inv.get(k) != 0]
        if missing:
            self._send_json({"ok": False, "error": f"Missing invoice field(s): {', '.join(missing)}"})
            return

        settings = load_settings()
        if settings.gmail_mode != "live":
            self._send_json({"ok": False, "error": "GMAIL_MODE is not 'live' -- set it in .env to send real email"})
            return

        from integrations.gmail import GmailClient
        from invoice_pdf import InvoiceData, InvoiceLineItem, InvoiceParty, render_invoice_pdf

        seller = inv.get("seller") or {}
        bill_to = inv.get("billTo") or {}
        line_items = [
            InvoiceLineItem(
                desc=li.get("desc", "—"),
                qty=li.get("qty", 1),
                unit=li.get("unit", 0),
                amount=li.get("amount", 0),
            )
            for li in (inv.get("lineItems") or [])
        ]

        try:
            pdf_bytes = render_invoice_pdf(InvoiceData(
                number=inv["number"],
                issue_date=inv["issueDate"],
                due_date=inv["dueDate"],
                seller=InvoiceParty(name=seller.get("name", "—"), address=seller.get("address") or [], email=seller.get("email", "")),
                bill_to=InvoiceParty(name=bill_to.get("name", "—"), address=bill_to.get("address") or [], email=bill_to.get("email", "")),
                amount=float(inv["amount"]),
                currency=inv.get("currency", "USD"),
                line_items=line_items,
            ))
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Could not build invoice PDF: {exc}"})
            return

        subject = payload.get("subject") or f"Invoice {inv['number']} from {seller.get('name', 'Rally, Inc.')}"
        body_text = payload.get("body") or (
            f"Hi {bill_to.get('name', '')},\n\nPlease find attached invoice {inv['number']} "
            f"for ${float(inv['amount']):,.2f}, due {inv['dueDate']}.\n\nThank you,\n{seller.get('name', 'Rally, Inc.')}"
        )

        client = GmailClient(settings)
        try:
            message_id = client.send_email(
                to, subject, body_text,
                attachment_filename=f"{inv['number']}.pdf",
                attachment_bytes=pdf_bytes,
                attachment_mime="application/pdf",
            )
        except Exception as exc:
            self._send_json({"ok": False, "error": f"{exc} (attempted To: {to!r})"})
            return

        self._send_json({"ok": True, "messageId": message_id, "to": to})

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/collections/") and path.endswith("/send"):
            coll_id = path[len("/api/collections/"):-len("/send")]
            self._send_collection_email(coll_id, self._read_json_body())
            return

        if path == "/api/invoices/send":
            self._send_invoice_email(self._read_json_body())
            return

        if path == "/api/collections/reminders/run":
            try:
                result = run_reminder_cycle()
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)})
                return
            self._send_json({"ok": True, **result})
            return

        self.send_response(404)
        self.end_headers()


def main() -> None:
    import os

    # CLI arg wins if given (local dev convention, e.g. `python server.py 8600`),
    # otherwise PORT (Railway sets this automatically for a deployed service),
    # otherwise the same 8600 default as before.
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = int(os.environ.get("PORT", "8600"))
    # 0.0.0.0 instead of 127.0.0.1 -- required for the service to be reachable
    # at all once deployed (Railway routes traffic to the container's network
    # interface, not just localhost inside it); harmless locally too, still
    # reachable at http://127.0.0.1:<port> or http://localhost:<port>.
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Emma UI (live HubSpot deals) — http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
