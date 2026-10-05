"""How a company becomes text for the model."""

from __future__ import annotations

import re

from .crosswalk import Evidence

_WS = re.compile(r"\s+")


def size_token(market_cap: float | None) -> str:
    """A coarse size bucket as a token. Some GICS lines are about scale, not product:
    Diversified versus Regional Banks is largely one of these, and no description says it."""
    if not market_cap or market_cap <= 0:
        return "mcapunknown"
    if market_cap >= 100e9:
        return "mcapmega"
    if market_cap >= 10e9:
        return "mcaplarge"
    if market_cap >= 2e9:
        return "mcapmid"
    if market_cap >= 3e8:
        return "mcapsmall"
    return "mcapmicro"


def company_text(ev: Evidence, market_cap: float | None = None) -> str:
    """Name, description and SIC text, plus the SIC code, Yahoo industry and size as
    made-up tokens so the model can learn what each is worth."""
    parts = [ev.name or "", size_token(market_cap)]
    if ev.description:
        parts.append(ev.description)
    if ev.sic_description:
        parts.append(f"SIC: {ev.sic_description}")
    if ev.sic_code:
        parts.append(f"siccode{ev.sic_code.strip().zfill(4)}")
    if ev.yahoo_industry:
        slug = re.sub(r"[^a-z0-9]+", "", ev.yahoo_industry.lower())
        parts.append(f"Yahoo industry: {ev.yahoo_industry} yahooind{slug}")
    return _WS.sub(" ", " ".join(parts)).strip()
