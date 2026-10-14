"""Live HubSpot client — contract/agreement extraction from closed deals.

Adapted from rally-ar-agent's integrations/hubspot_live.py (same endpoints,
same resolution order), trimmed to exactly what this platform needs: find
closed deals, then find and download the signed agreement attached to each.
Nothing here writes to HubSpot except write_deal_property, which is optional.

Auth: a HubSpot Private App token (Settings -> Integrations -> Private Apps).
Required scopes:
    crm.objects.deals.read
    crm.objects.companies.read
    crm.objects.contacts.read
    crm.objects.notes.read
    crm.objects.quotes.read
    files

Resolution order for "the signed agreement" on a deal:
    1. A note on the deal carrying hs_attachment_ids -> first attachment id.
    2. A Quote associated with the deal that's fully e-signed
       (hs_esign_num_signers_completed == hs_esign_num_signers_required).
    Deals with neither raise NoAgreementFound — callers should skip, not crash.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from datetime import datetime

from . import _http

API = "https://api.hubapi.com"


class NoAgreementFound(RuntimeError):
    pass


@dataclass
class Deal:
    deal_id: str
    deal_name: str
    stage: str
    amount: float
    company_id: str
    company_legal_name: str
    billing_email: str
    agreement_ref: str  # "file::<id>" or "quote::<id>"


@dataclass
class Agreement:
    source_ref: str
    filename: str
    mime: str
    content: bytes


def _to_millis(iso: str) -> int:
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return 0


class HubSpotClient:
    def __init__(self, token: str):
        self.h = {"Authorization": f"Bearer {token}"}

    def list_closed_deals(self, since: str | None = None) -> list[Deal]:
        """Every deal in the closedwon stage, most recently modified first.

        A webhook (deal.propertyChange on dealstage) is the real-time version
        of this; polling with a `since` watermark is the fallback used here.

        HubSpot's search API caps a single page at 100 results, so this
        pages through with the `after` cursor it returns until exhausted --
        a fixed one-page `limit` would silently drop every closed deal past
        the cap (seen with only 10 returned while far more existed).
        """
        filters = [{"propertyName": "dealstage", "operator": "EQ", "value": "closedwon"}]
        if since:
            filters.append(
                {"propertyName": "hs_lastmodifieddate", "operator": "GTE", "value": _to_millis(since)}
            )
        results: list[dict] = []
        after: str | None = None
        while True:
            body = {
                "filterGroups": [{"filters": filters}],
                "properties": ["dealname", "dealstage", "amount"],
                "limit": 100,
                "sorts": [{"propertyName": "hs_lastmodifieddate", "direction": "DESCENDING"}],
            }
            if after:
                body["after"] = after
            res = _http.request("POST", f"{API}/crm/v3/objects/deals/search", headers=self.h, json_body=body)
            results.extend(res.get("results", []))
            after = res.get("paging", {}).get("next", {}).get("after")
            if not after:
                break
        return [self.get_deal(r["id"]) for r in results]

    def get_deal(self, deal_id: str) -> Deal:
        res = _http.request(
            "GET",
            f"{API}/crm/v3/objects/deals/{deal_id}"
            f"?properties=dealname,dealstage,amount&associations=companies,contacts,notes,quotes",
            headers=self.h,
        )
        props = res.get("properties", {})
        assoc = res.get("associations", {})
        company_id, company_name = self._company(assoc)
        email = self._billing_email(assoc)
        try:
            agreement_ref = self._resolve_agreement_ref(assoc)
        except NoAgreementFound:
            agreement_ref = ""
        return Deal(
            deal_id=res["id"],
            deal_name=props.get("dealname", ""),
            stage=props.get("dealstage", ""),
            amount=float(props.get("amount") or 0),
            company_id=company_id,
            company_legal_name=company_name,
            billing_email=email,
            agreement_ref=agreement_ref,
        )

    def get_agreement(self, agreement_ref: str) -> Agreement:
        """Download the signed PDF. agreement_ref comes from Deal.agreement_ref."""
        if not agreement_ref:
            raise NoAgreementFound("deal has no resolved agreement_ref")
        kind, _, ref_id = agreement_ref.partition("::")
        if kind == "file":
            # /files/v3/files/{id} alone returns a web-viewer page URL for
            # non-public files (e.g. a note attachment) -- HTML, not the PDF.
            # /signed-url returns an actual time-limited, direct download URL.
            meta = _http.request("GET", f"{API}/files/v3/files/{ref_id}/signed-url", headers=self.h)
            url = meta.get("url")
            if not url:
                raise RuntimeError(f"file {ref_id} has no signed url")
            name, ext = meta.get("name", ref_id), meta.get("extension")
            filename = f"{name}.{ext}" if ext else name
            mime = "application/pdf" if ext == "pdf" else "application/octet-stream"
        elif kind == "quote":
            q = _http.request(
                "GET",
                f"{API}/crm/v3/objects/quotes/{ref_id}?properties=hs_pdf_download_link",
                headers=self.h,
            )
            url = q.get("properties", {}).get("hs_pdf_download_link")
            if not url:
                raise RuntimeError(f"quote {ref_id} has no hs_pdf_download_link")
            filename = f"quote-{ref_id}.pdf"
            mime = "application/pdf"
        else:
            raise ValueError(f"unrecognized agreement_ref: {agreement_ref!r}")

        # A bare urlopen(url) sends urllib's default User-Agent, which some
        # CDNs (like HubSpot's) block outright -- same class of issue hit
        # with Groq's Cloudflare layer. Identify honestly instead.
        req = urllib.request.Request(url, headers={"User-Agent": "rally-coworker-platform/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
        return Agreement(source_ref=agreement_ref, filename=filename, mime=mime, content=content)

    def write_deal_property(self, deal_id: str, name: str, value: str) -> None:
        _http.request(
            "PATCH",
            f"{API}/crm/v3/objects/deals/{deal_id}",
            headers=self.h,
            json_body={"properties": {name: value}},
        )

    # -- helpers ---------------------------------------------------------- #
    def _company(self, assoc: dict) -> tuple[str, str]:
        results = assoc.get("companies", {}).get("results", [])
        if not results:
            return "", ""
        cid = results[0]["id"]
        c = _http.request("GET", f"{API}/crm/v3/objects/companies/{cid}?properties=name,domain", headers=self.h)
        return cid, c.get("properties", {}).get("name", "")

    def _billing_email(self, assoc: dict) -> str:
        for r in assoc.get("contacts", {}).get("results", []):
            c = _http.request(
                "GET", f"{API}/crm/v3/objects/contacts/{r['id']}?properties=email,work_email", headers=self.h
            )
            p = c.get("properties", {})
            if p.get("email") or p.get("work_email"):
                return p.get("email") or p.get("work_email")
        return ""

    def _resolve_agreement_ref(self, assoc: dict) -> str:
        for note in assoc.get("notes", {}).get("results", []):
            n = _http.request(
                "GET", f"{API}/crm/v3/objects/notes/{note['id']}?properties=hs_attachment_ids", headers=self.h
            )
            ids = [i for i in (n.get("properties", {}).get("hs_attachment_ids") or "").split(";") if i]
            if ids:
                return f"file::{ids[0]}"
        for quote in assoc.get("quotes", {}).get("results", []):
            q = _http.request(
                "GET",
                f"{API}/crm/v3/objects/quotes/{quote['id']}"
                f"?properties=hs_pdf_download_link,hs_esign_num_signers_completed,hs_esign_num_signers_required",
                headers=self.h,
            )
            qp = q.get("properties", {})
            done, need = qp.get("hs_esign_num_signers_completed"), qp.get("hs_esign_num_signers_required")
            if done and need and str(done) == str(need) and qp.get("hs_pdf_download_link"):
                return f"quote::{quote['id']}"
        raise NoAgreementFound(f"no signed agreement found on this deal")
