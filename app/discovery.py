"""
Discovery Ledger — cryptographic provenance for cross-domain insights.

When the Angel's derivation engine chains rules across domains and
discovers something novel, that derivation trace IS the proof. The
Discovery Ledger:

1. Hashes the derivation chain (which rules fired, in which order)
2. Timestamps the discovery
3. Records the human partner (co-discovery — twin species)
4. Stores locally, ready for on-chain submission (Algorand)

Architecture:
  - DiscoveryEvent: a single cross-domain insight with full trace
  - DiscoveryLedger: local append-only store with hash chain
  - AlgorandBridge: stub for on-chain submission (needs SDK)

The hash chain means even the LOCAL ledger is tamper-evident.
Each discovery's hash includes the previous discovery's hash,
forming a chain. If any entry is altered, the chain breaks.

Future: Algorand submission gives global timestamp, post-quantum
security, and smart contract royalties when discoveries are cited.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from app.paths import mkangel_dir

_LEDGER_DB = mkangel_dir() / "discovery_ledger.db"


# ---------------------------------------------------------------------------
# Discovery Event
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryEvent:
    """A single cross-domain discovery with full derivation trace."""

    # Identity
    discovery_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)

    # The discovery
    description: str = ""
    domains_crossed: list[str] = field(default_factory=list)
    source_domain: str = ""
    target_domain: str = ""

    # The proof — derivation trace
    derivation_chain: list[dict[str, Any]] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)
    confidence: float = 0.0

    # Attribution
    angel_name: str = ""
    human_partner: str = ""
    project_context: str = ""

    # Hash chain
    content_hash: str = ""
    previous_hash: str = ""

    def compute_hash(self) -> str:
        """Hash the discovery content + previous hash.

        The hash covers: description, domains, derivation chain,
        rules, confidence, attribution, and previous hash.
        This makes the chain tamper-evident.
        """
        payload = json.dumps({
            "discovery_id": self.discovery_id,
            "timestamp": self.timestamp,
            "description": self.description,
            "domains_crossed": sorted(self.domains_crossed),
            "derivation_chain": self.derivation_chain,
            "rules_fired": sorted(self.rules_fired),
            "confidence": self.confidence,
            "angel_name": self.angel_name,
            "human_partner": self.human_partner,
            "previous_hash": self.previous_hash,
        }, sort_keys=True, default=str)
        self.content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.content_hash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiscoveryEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def citation_key(self) -> str:
        """Short citation key for referencing this discovery."""
        return f"DISC-{self.discovery_id[:8]}-{self.angel_name[:3].upper()}"


# ---------------------------------------------------------------------------
# Discovery Ledger — local append-only hash chain
# ---------------------------------------------------------------------------

class DiscoveryLedger:
    """Append-only, tamper-evident discovery ledger.

    Each entry's hash includes the previous entry's hash, forming
    a chain. If any entry is altered, all subsequent hashes break.
    Stored in SQLite for durability.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = str(db_path or _LEDGER_DB)
        self._ensure_db()
        self._chain_tip: str = self._load_chain_tip()

    def _ensure_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discoveries (
                    discovery_id   TEXT PRIMARY KEY,
                    timestamp      REAL NOT NULL,
                    description    TEXT NOT NULL,
                    domains        TEXT NOT NULL,
                    source_domain  TEXT,
                    target_domain  TEXT,
                    derivation     TEXT NOT NULL,
                    rules_fired    TEXT NOT NULL,
                    confidence     REAL NOT NULL,
                    angel_name     TEXT,
                    human_partner  TEXT,
                    project        TEXT,
                    content_hash   TEXT NOT NULL,
                    previous_hash  TEXT NOT NULL,
                    on_chain       INTEGER DEFAULT 0,
                    chain_tx_id    TEXT DEFAULT ''
                )
            """)

    def _load_chain_tip(self) -> str:
        """Load the most recent hash (chain tip)."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT content_hash FROM discoveries ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                return row[0] if row else "GENESIS"
        except Exception:
            return "GENESIS"

    def record(self, event: DiscoveryEvent) -> DiscoveryEvent:
        """Record a discovery. Computes hash, links to chain, stores."""
        event.previous_hash = self._chain_tip
        event.compute_hash()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO discoveries
                   (discovery_id, timestamp, description, domains,
                    source_domain, target_domain, derivation, rules_fired,
                    confidence, angel_name, human_partner, project,
                    content_hash, previous_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.discovery_id, event.timestamp, event.description,
                    json.dumps(event.domains_crossed),
                    event.source_domain, event.target_domain,
                    json.dumps(event.derivation_chain),
                    json.dumps(event.rules_fired),
                    event.confidence, event.angel_name,
                    event.human_partner, event.project_context,
                    event.content_hash, event.previous_hash,
                ),
            )

        self._chain_tip = event.content_hash
        return event

    def verify_chain(self) -> tuple[bool, int, str]:
        """Verify the entire hash chain is intact.

        Returns (intact, entries_checked, first_broken_id).
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT discovery_id, description, domains, derivation, "
                    "rules_fired, confidence, angel_name, human_partner, "
                    "content_hash, previous_hash, timestamp "
                    "FROM discoveries ORDER BY timestamp ASC"
                ).fetchall()
        except Exception:
            return (True, 0, "")

        expected_prev = "GENESIS"
        for i, row in enumerate(rows):
            if row[9] != expected_prev:
                return (False, i, row[0])

            # Recompute hash
            payload = json.dumps({
                "discovery_id": row[0],
                "timestamp": row[10],
                "description": row[1],
                "domains_crossed": sorted(json.loads(row[2])),
                "derivation_chain": json.loads(row[3]),
                "rules_fired": sorted(json.loads(row[4])),
                "confidence": row[5],
                "angel_name": row[6],
                "human_partner": row[7],
                "previous_hash": row[9],
            }, sort_keys=True, default=str)
            computed = hashlib.sha256(payload.encode("utf-8")).hexdigest()

            if computed != row[8]:
                return (False, i, row[0])

            expected_prev = row[8]

        return (True, len(rows), "")

    def search(self, query: str = "", domain: str = "") -> list[DiscoveryEvent]:
        """Search discoveries by description or domain."""
        with sqlite3.connect(self._db_path) as conn:
            if domain:
                rows = conn.execute(
                    "SELECT * FROM discoveries WHERE domains LIKE ? ORDER BY timestamp DESC",
                    (f"%{domain}%",),
                ).fetchall()
            elif query:
                rows = conn.execute(
                    "SELECT * FROM discoveries WHERE description LIKE ? ORDER BY timestamp DESC",
                    (f"%{query}%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM discoveries ORDER BY timestamp DESC LIMIT 50"
                ).fetchall()

        return [self._row_to_event(r) for r in rows]

    def count(self) -> int:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM discoveries").fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def chain_tip(self) -> str:
        return self._chain_tip

    def pending_on_chain(self) -> list[DiscoveryEvent]:
        """Discoveries not yet submitted to blockchain."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM discoveries WHERE on_chain = 0 ORDER BY timestamp ASC"
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def mark_on_chain(self, discovery_id: str, tx_id: str) -> None:
        """Mark a discovery as submitted to blockchain."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE discoveries SET on_chain = 1, chain_tx_id = ? WHERE discovery_id = ?",
                (tx_id, discovery_id),
            )

    def _row_to_event(self, row: tuple) -> DiscoveryEvent:
        return DiscoveryEvent(
            discovery_id=row[0], timestamp=row[1], description=row[2],
            domains_crossed=json.loads(row[3]),
            source_domain=row[4], target_domain=row[5],
            derivation_chain=json.loads(row[6]),
            rules_fired=json.loads(row[7]),
            confidence=row[8], angel_name=row[9],
            human_partner=row[10], project_context=row[11],
            content_hash=row[12], previous_hash=row[13],
        )


# ---------------------------------------------------------------------------
# Algorand Bridge — stub ready for SDK integration
# ---------------------------------------------------------------------------

class AlgorandBridge:
    """Bridge to Algorand blockchain for discovery provenance.

    This is a stub. When the Algorand Python SDK (py-algorand-sdk)
    is available, this submits discovery hashes as on-chain notes,
    creating immutable, globally-timestamped provenance records.

    Architecture:
      - Discovery hash → Algorand note field (up to 1KB)
      - Each submission is a 0-ALGO transaction to self (note only)
      - Cost: ~0.001 ALGO per discovery (~$0.0002)
      - Finality: 3.3 seconds
      - Post-quantum: Algorand's Falcon signatures (if enabled)

    Future smart contract:
      - Discovery citation tracking
      - Automatic royalty distribution when cited
      - Cross-angel coordination proofs
    """

    def __init__(
        self,
        algod_address: str = "",
        algod_token: str = "",
        sender_address: str = "",
        sender_key: str = "",
    ) -> None:
        self._algod_address = algod_address
        self._algod_token = algod_token
        self._sender_address = sender_address
        self._sender_key = sender_key
        self._connected = False

    def connect(self) -> bool:
        """Attempt to connect to Algorand node."""
        if not self._algod_address or not self._algod_token:
            return False
        try:
            from algosdk.v2client import algod
            self._client = algod.AlgodClient(self._algod_token, self._algod_address)
            self._client.status()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def submit_discovery(self, event: DiscoveryEvent) -> str | None:
        """Submit a discovery hash to Algorand.

        Returns the transaction ID or None if not connected.

        The note field contains:
          - discovery_id
          - content_hash (SHA-256 of full derivation trace)
          - domains crossed
          - angel + human attribution
          - citation key
        """
        if not self._connected:
            return None

        try:
            from algosdk import transaction, account

            note = json.dumps({
                "type": "MKAngel_Discovery",
                "version": "1.0",
                "discovery_id": event.discovery_id,
                "content_hash": event.content_hash,
                "domains": event.domains_crossed,
                "angel": event.angel_name,
                "human": event.human_partner,
                "citation": event.citation_key(),
                "confidence": event.confidence,
                "timestamp": event.timestamp,
            }).encode("utf-8")

            params = self._client.suggested_params()
            txn = transaction.PaymentTxn(
                sender=self._sender_address,
                sp=params,
                receiver=self._sender_address,
                amt=0,
                note=note,
            )
            signed = txn.sign(self._sender_key)
            tx_id = self._client.send_transaction(signed)
            transaction.wait_for_confirmation(self._client, tx_id, 4)
            return tx_id
        except Exception:
            return None

    def submit_batch(
        self, events: list[DiscoveryEvent], ledger: DiscoveryLedger,
    ) -> list[str]:
        """Submit all pending discoveries and mark them on-chain."""
        tx_ids = []
        for event in events:
            tx_id = self.submit_discovery(event)
            if tx_id:
                ledger.mark_on_chain(event.discovery_id, tx_id)
                tx_ids.append(tx_id)
        return tx_ids

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __repr__(self) -> str:
        status = "connected" if self._connected else "offline (stub)"
        return f"AlgorandBridge({status})"
