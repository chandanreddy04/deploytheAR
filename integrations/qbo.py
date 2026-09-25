"""Live QuickBooks Online client (read-only — QBO is the source of truth for
"paid"). Adapted from rally-ar-agent's integrations/qbo_live.py, same logic,
as a standalone copy with no cross-repo import dependency.

Needs QBO_REALM_ID plus either a still-valid QBO_ACCESS_TOKEN or a
QBO_REFRESH_TOKEN + QBO_CLIENT_ID + QBO_CLIENT_SECRET (access tokens expire
after ~1h, so anything longer-running than a one-shot script needs the
refresh token).

list_payments_since() is the polled way to find new/changed payments; the
cursor is an ISO ``MetaData.LastUpdatedTime`` string you persist yourself
between runs.
"""

from __future__ import annotations

import base64
import logging
import time
import urllib.parse
from dataclasses import dataclass, field

from config import Settings, update_dotenv_value
from db import db_save
from integrations import _http

log = logging.getLogger("qbo")

_OAUTH_TOKEN_URL = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"


@dataclass
class QboPayment:
    payment_id: str
    customer_ref: str
    customer_name: str
    total_amount: float
    txn_date: str
    private_note: str = ""
    linked_invoice_numbers: list[str] = field(default_factory=list)
    voided: bool = False


class QboClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.base = settings.qbo_base_url.rstrip("/")
        self.realm = settings.qbo_realm_id
        self._access = settings.qbo_access_token
        self._expires_at = time.time() + 3000 if settings.qbo_access_token else 0.0
        self._docnum_cache: dict[str, str] = {}
        self._refresh = settings.qbo_refresh_token

    # -- auth --------------------------------------------------------- #
    def _ensure_token(self, force: bool = False) -> None:
        if not force and self._access and time.time() < self._expires_at - 60:
            return
        if not (self._refresh and self.s.qbo_client_id and self.s.qbo_client_secret):
            if not self._access:
                raise RuntimeError("QBO access token expired and no refresh credentials configured")
            return
        basic = base64.b64encode(f"{self.s.qbo_client_id}:{self.s.qbo_client_secret}".encode()).decode()
        res = _http.request(
            "POST",
            _OAUTH_TOKEN_URL,
            headers={"Authorization": f"Basic {basic}"},
            form_body={"grant_type": "refresh_token", "refresh_token": self._refresh},
        )
        self._access = res["access_token"]
        self._expires_at = time.time() + int(res.get("expires_in", 3600))
        new_refresh = res.get("refresh_token")
        if new_refresh and new_refresh != self._refresh:
            self._refresh = new_refresh
            # Never log the raw token -- persist it so a stale in-memory-only
            # rotation can't silently break the next fresh process start
            # (this is what caused a real "invalid_grant" failure). Two
            # paths, not either/or: update_dotenv_value() covers local dev
            # (a real .env file on disk to rewrite); db_save() covers a
            # deployed environment (e.g. Railway) where there's no local
            # file to rewrite at all -- load_settings() checks this DB value
            # first and prefers it when present, so this is what actually
            # matters once this app is deployed, not the .env rewrite.
            db_save("qbo_refresh_token", new_refresh, time.time())
            if update_dotenv_value("QBO_REFRESH_TOKEN", new_refresh):
                log.warning("QBO refresh token rotated — new value saved to .env and the database")
            else:
                log.warning("QBO refresh token rotated — new value saved to the database (no local .env file to also update)")

    def _h(self) -> dict:
        self._ensure_token()
        return {"Authorization": f"Bearer {self._access}", "Accept": "application/json"}

    def _get(self, path: str) -> dict:
        """GET with a single forced-refresh retry on 401 (token revoked mid-flight)."""
        try:
            return _http.request("GET", f"{self.base}{path}", headers=self._h())
        except _http.HttpError as e:
            if e.status == 401:
                self._ensure_token(force=True)
                return _http.request("GET", f"{self.base}{path}", headers=self._h())
            raise

    def _query(self, statement: str) -> dict:
        q = urllib.parse.quote(statement)
        return self._get(f"/v3/company/{self.realm}/query?query={q}")

    # -- reads ------------------------------------------------------- #
    def list_payments_since(self, cursor: str | None) -> tuple[list[QboPayment], str]:
        since = cursor or "2020-01-01T00:00:00-00:00"
        res = self._query(
            "SELECT * FROM Payment "
            f"WHERE MetaData.LastUpdatedTime > '{since}' "
            "ORDERBY MetaData.LastUpdatedTime"
        )
        rows = res.get("QueryResponse", {}).get("Payment", [])
        payments = [self._to_payment(r) for r in rows]
        new_cursor = rows[-1]["MetaData"]["LastUpdatedTime"] if rows else since
        return payments, new_cursor

    def get_payment(self, payment_id: str) -> QboPayment:
        res = self._get(f"/v3/company/{self.realm}/payment/{payment_id}")
        return self._to_payment(res["Payment"])

    def get_invoice_balance(self, invoice_number: str) -> float | None:
        res = self._query(f"SELECT Id, Balance FROM Invoice WHERE DocNumber = '{invoice_number}'")
        rows = res.get("QueryResponse", {}).get("Invoice", [])
        return float(rows[0]["Balance"]) if rows else None

    def list_invoices(self, limit: int = 20) -> list[dict]:
        """Not in the original rally-ar-agent client -- added for exploring
        what sample data a sandbox actually has before building anything
        that depends on its shape. DueDate/TxnDate/BillEmail added for the
        collections/dunning feature -- overdue detection needs a real due
        date, and a real send-to address needs the invoice's own BillEmail
        (not fabricated from the customer name)."""
        res = self._query(
            "SELECT Id, DocNumber, TotalAmt, Balance, CustomerRef, DueDate, TxnDate, BillEmail "
            f"FROM Invoice MAXRESULTS {limit}"
        )
        return res.get("QueryResponse", {}).get("Invoice", [])

    def _invoice_docnumber(self, txn_id: str) -> str:
        if txn_id in self._docnum_cache:
            return self._docnum_cache[txn_id]
        try:
            res = self._query(f"SELECT Id, DocNumber FROM Invoice WHERE Id = '{txn_id}'")
            rows = res.get("QueryResponse", {}).get("Invoice", [])
            doc = rows[0].get("DocNumber", "") if rows else ""
        except Exception:
            doc = ""
        self._docnum_cache[txn_id] = doc
        return doc

    def _to_payment(self, p: dict) -> QboPayment:
        linked = []
        for ln in p.get("Line", []):
            for lt in ln.get("LinkedTxn", []):
                if lt.get("TxnType") == "Invoice" and lt.get("TxnId"):
                    doc = self._invoice_docnumber(lt["TxnId"])
                    if doc:
                        linked.append(doc)
        amt = float(p.get("TotalAmt", 0) or 0)
        note = p.get("PrivateNote", "") or ""
        return QboPayment(
            payment_id=p["Id"],
            customer_ref=p.get("CustomerRef", {}).get("value", ""),
            customer_name=p.get("CustomerRef", {}).get("name", ""),
            total_amount=amt,
            txn_date=p.get("TxnDate", ""),
            private_note=note,
            linked_invoice_numbers=linked,
            voided=(amt == 0 and "void" in note.lower()),
        )
