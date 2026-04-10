"""
WhatsApp bridge for MKAngel.

Lets Matt (or any allow-listed WhatsApp number) message a bot number and
have a Claude Agent SDK session run against the MKAngel repo, with full
Read/Edit/Bash/Git tool access. Replies come back as WhatsApp messages.

Architecture:
    WhatsApp  <->  Baileys (Node subprocess, stdin/stdout JSON lines)
                     ^
                     |  newline-delimited JSON
                     v
                   bridge.py  (Python asyncio event loop)
                     |
                     v
                   agent_runner.py  (Claude Agent SDK query() calls)
                     |
                     v
                   MKAngel repo on local disk

The Node side talks to WhatsApp via @whiskeysockets/baileys (free, uses a
spare WhatsApp account, QR-linked). The Python side owns auth, routing,
rate limiting, and the agent loop.

Entry point: ``python -m app.whatsapp.bridge``
"""

from __future__ import annotations

__all__ = ["bridge", "agent_runner", "config"]
