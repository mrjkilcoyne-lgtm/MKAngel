"""
Polymarket anchor — the de-vigged heartbeat of the crypto prediction crowd.

Polymarket is, at the time of writing, the largest crypto-backed prediction
market in the world.  Its prices are set peer-to-peer — an AMM and a central
limit order book decide them — so there is no bookmaker margin in the
conventional sense.  That makes Polymarket to macro events what Pinnacle is
to football: a sharp anchor.  The Angel uses the Buchdahl Wisdom-of-Crowd
intuition here — trust the sharp market, and let your own model's job be
finding the rare cases where a softer book has drifted away from it.

Think of the pianist analogy from ``docs/on_her_nature.md``: a pianist
does not know where middle C is by measuring the keyboard with a ruler,
she knows it through proprioception — through the remembered weight of
her own hand.  The Angel's proprioception in the landscape of future
events is calibrated against anchors like this one.  Polymarket is one
of her keys.

And echoing Frame 4 from ``docs/tardis_session_notes.md``: do not try to
out-predict the sharp market.  Let it tell you where the truth probably
lies, and spend your modelling budget on the places where the soft world
has drifted off from the sharp one.  The edge is in the gap, not in the
anchor.

This module is a strictly READ-ONLY client for the Polymarket gamma API
(https://gamma-api.polymarket.com).  Read endpoints are unauthenticated
and therefore a free, no-credential anchor source — perfect for a pure
Python, dependency-light module that needs to run anywhere the rest of
MKAngel runs, including inside the Android Kivy build.

The client will never POST, PATCH or DELETE.  It cannot modify state on
Polymarket — only look at it looking at itself.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://gamma-api.polymarket.com"
USER_AGENT = "MKAngel-TARDIS/0.1 (+https://github.com/mrjkilcoyne-lgtm/MKAngel)"


def _utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coerce_float(value: object, default: float | None = None) -> float | None:
    """Polymarket returns numbers as strings in some fields.  Be forgiving."""
    if value is None or value == "":
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_str_list(value: object) -> list[str]:
    """Polymarket sometimes returns JSON-encoded strings for list fields."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # outcomes / outcomePrices are frequently JSON-encoded strings
        # like '["Yes", "No"]' or '["0.535", "0.465"]'.
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return [s]
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        return [str(parsed)]
    return [str(value)]


def _coerce_float_list(value: object) -> list[float]:
    """Like ``_coerce_str_list`` but for price arrays."""
    raw = _coerce_str_list(value)
    out: list[float] = []
    for item in raw:
        f = _coerce_float(item)
        if f is not None:
            out.append(f)
    return out


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


# ---------------------------------------------------------------------------
# PolyMarket — one row of the sharp anchor ledger
# ---------------------------------------------------------------------------

@dataclass
class PolyMarket:
    """A single Polymarket market snapshot.

    This is a small, quiet dataclass — everything the Angel needs to
    treat a Polymarket line as an anchor, and nothing she doesn't.
    """

    market_id: str
    question: str
    slug: str
    description: str
    end_date: str | None
    start_date: str | None
    outcomes: list[str] = field(default_factory=list)
    outcome_prices: list[float] = field(default_factory=list)
    volume: float = 0.0
    liquidity: float | None = None
    active: bool = False
    closed: bool = False
    snapshot_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_api(cls, raw: dict) -> "PolyMarket":
        """Parse one market dict from the gamma API response."""
        return cls(
            market_id=str(raw.get("id", "") or ""),
            question=str(raw.get("question", "") or ""),
            slug=str(raw.get("slug", "") or ""),
            description=str(raw.get("description", "") or ""),
            end_date=(raw.get("endDate") or None),
            start_date=(raw.get("startDate") or None),
            outcomes=_coerce_str_list(raw.get("outcomes")),
            outcome_prices=_coerce_float_list(raw.get("outcomePrices")),
            volume=_coerce_float(raw.get("volume"), 0.0) or 0.0,
            liquidity=_coerce_float(raw.get("liquidity"), None),
            active=_coerce_bool(raw.get("active"), False),
            closed=_coerce_bool(raw.get("closed"), False),
            snapshot_at=_utc_now_iso(),
        )


# ---------------------------------------------------------------------------
# PolymarketClient — strictly read-only
# ---------------------------------------------------------------------------

class PolymarketClient:
    """Read-only client for the Polymarket gamma API.

    All network calls are HTTP GET.  There is no code path in this class
    that issues POST, PATCH, PUT or DELETE — by construction the client
    cannot modify Polymarket state.  On any transport or parse failure
    the client returns an empty list (for collection methods) or None
    (for single-item methods) rather than raising, so upstream callers
    — including the Angel — never have to wrap calls in try/except.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 20,
        http_proxy: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)

        if http_proxy is None:
            http_proxy = (
                os.environ.get("HTTPS_PROXY")
                or os.environ.get("https_proxy")
                or os.environ.get("HTTP_PROXY")
                or os.environ.get("http_proxy")
            )
        self.http_proxy = http_proxy

        if http_proxy:
            proxy_handler = urllib.request.ProxyHandler(
                {"http": http_proxy, "https": http_proxy}
            )
            self._opener = urllib.request.build_opener(proxy_handler)
            logger.debug("PolymarketClient using proxy %s", http_proxy)
        else:
            self._opener = urllib.request.build_opener()
        self._opener.addheaders = [("User-Agent", USER_AGENT)]

    # ------------------------------------------------------------------
    # HTTP — GET only
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> object | None:
        """Issue a single GET request and return parsed JSON, or None on failure.

        Never raises.  The Angel prefers silence to exceptions when an
        anchor source is down — she will simply fall back to her own
        model for that decision.
        """
        url = self.base_url + path
        if params:
            query = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
            if query:
                url = url + "?" + query

        req = urllib.request.Request(url, method="GET")
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            logger.warning("polymarket HTTP %s on GET %s", exc.code, url)
            return None
        except urllib.error.URLError as exc:
            logger.warning("polymarket URLError on GET %s: %s", url, exc.reason)
            return None
        except socket.timeout:
            logger.warning("polymarket timeout on GET %s", url)
            return None
        except OSError as exc:
            logger.warning("polymarket OSError on GET %s: %s", url, exc)
            return None

        try:
            text = raw.decode("utf-8", errors="replace")
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("polymarket JSON decode error on %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def list_markets(
        self,
        active: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PolyMarket]:
        """List markets from the gamma API.

        Returns an empty list on any failure.  Parameters mirror the
        upstream ``/markets`` endpoint's query string.
        """
        params: dict = {
            "active": "true" if active else "false",
            "limit": int(limit),
            "offset": int(offset),
        }
        data = self._get("/markets", params=params)
        return self._parse_market_list(data)

    def search_markets(self, query: str, limit: int = 20) -> list[PolyMarket]:
        """Filter active markets by case-insensitive substring match.

        The gamma API does not always honour a ``q=`` param, so we fetch
        a broad page of active markets and filter locally against the
        question, description and slug.  Cheap and predictable.
        """
        q = (query or "").strip().lower()
        if not q:
            return []

        # Fetch a wider pool than ``limit`` so the filter has something
        # to chew on.  500 is the soft ceiling the gamma API tends to honour.
        pool = self.list_markets(active=True, limit=500, offset=0)
        matches: list[PolyMarket] = []
        for m in pool:
            haystack = " ".join([
                (m.question or ""),
                (m.description or ""),
                (m.slug or ""),
            ]).lower()
            if q in haystack:
                matches.append(m)
                if len(matches) >= limit:
                    break
        return matches

    def get_market(self, market_id: str) -> PolyMarket | None:
        """Fetch a single market by its gamma id.

        Returns None on any failure or if the id is not found.
        """
        if not market_id:
            return None
        data = self._get("/markets", params={"id": market_id})
        markets = self._parse_market_list(data)
        if not markets:
            return None
        # Prefer an exact id match if the API returned several.
        for m in markets:
            if m.market_id == str(market_id):
                return m
        return markets[0]

    # ------------------------------------------------------------------
    # Derivations — what the Angel actually wants
    # ------------------------------------------------------------------

    def binary_probability(
        self, market: PolyMarket
    ) -> tuple[float, float] | None:
        """For a two-outcome market, return ``(p_yes, p_no)``.

        Returns None for non-binary markets or when prices are missing.
        Polymarket's prices already sum to (approximately) 1.0 — they
        are de-vigged by construction — so we return them as-is without
        any Buchdahl-style renormalisation.
        """
        if market is None:
            return None
        if len(market.outcomes) != 2:
            return None
        if len(market.outcome_prices) != 2:
            return None

        p_yes = market.outcome_prices[0]
        p_no = market.outcome_prices[1]
        if p_yes < 0 or p_no < 0:
            return None

        # If the outcome labels look flipped (e.g. ["No", "Yes"]), swap
        # so the first element always corresponds to the affirmative.
        first_label = (market.outcomes[0] or "").strip().lower()
        second_label = (market.outcomes[1] or "").strip().lower()
        if first_label == "no" and second_label == "yes":
            p_yes, p_no = p_no, p_yes

        return (p_yes, p_no)

    def to_anchor_dict(self, market: PolyMarket) -> dict:
        """Produce a dict suitable for ``Angel.superforecast(context=...)``.

        The keys are deliberately flat and string-friendly so they can be
        dropped straight into a prompt, a JSON context blob, or a
        journal entry.  Non-binary markets get ``prob_yes`` / ``prob_no``
        set to None.
        """
        probs = self.binary_probability(market)
        if probs is None:
            prob_yes: float | None = None
            prob_no: float | None = None
        else:
            prob_yes, prob_no = probs

        return {
            "question": market.question,
            "outcomes": list(market.outcomes),
            "prob_yes": prob_yes,
            "prob_no": prob_no,
            "volume": market.volume,
            "end_date": market.end_date,
            "liquidity": market.liquidity,
            "snapshot_at": market.snapshot_at,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_market_list(data: object) -> list[PolyMarket]:
        """Turn whatever the gamma API gave us into a list of PolyMarket.

        The endpoint has been seen to return either a bare JSON array or
        a dict with a ``data`` key — accept both shapes.
        """
        if data is None:
            return []
        items: list = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            maybe = data.get("data")
            if isinstance(maybe, list):
                items = maybe
            else:
                # A single-market dict response.
                items = [data]
        out: list[PolyMarket] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            try:
                out.append(PolyMarket.from_api(raw))
            except (TypeError, ValueError) as exc:
                logger.warning("skipping malformed polymarket row: %s", exc)
                continue
        return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Demonstrate that the module never raises on the happy path, even
    # when the upstream API might be flaky.
    raised: Exception | None = None
    first_market: PolyMarket | None = None
    try:
        client = PolymarketClient()  # auto-picks HTTPS_PROXY

        print("=== list_markets(active=True, limit=5) ===")
        markets = client.list_markets(active=True, limit=5)
        print(f"got {len(markets)} markets")
        for m in markets:
            probs = client.binary_probability(m)
            if probs is None:
                tag = "(non-binary)"
            else:
                tag = f"p_yes={probs[0]:.3f} p_no={probs[1]:.3f}"
            print(f"  - {m.question}  {tag}")
            if first_market is None:
                first_market = m

        print()
        print("=== search_markets('election', limit=3) ===")
        hits = client.search_markets("election", limit=3)
        print(f"got {len(hits)} hits")
        for m in hits:
            probs = client.binary_probability(m)
            if probs is None:
                tag = "(non-binary)"
            else:
                tag = f"p_yes={probs[0]:.3f} p_no={probs[1]:.3f}"
            print(f"  - {m.question}  {tag}")

        print()
        print("=== to_anchor_dict(first_market) ===")
        if first_market is not None:
            anchor = client.to_anchor_dict(first_market)
            print(json.dumps(anchor, indent=2, default=str))
        else:
            print("(no first market available)")
    except Exception as exc:  # pragma: no cover - the client should swallow these
        raised = exc
        print(f"UNEXPECTED EXCEPTION: {type(exc).__name__}: {exc}")

    print()
    if raised is None:
        print("client did not raise — read-only anchor OK")
    else:
        print("client raised — this should not happen for a read-only anchor")
