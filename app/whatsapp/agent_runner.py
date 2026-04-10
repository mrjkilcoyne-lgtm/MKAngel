"""
Runs a Claude Agent SDK session for a single incoming WhatsApp message.

Each message becomes one :func:`run_agent` call. The agent is given the
MKAngel repo as its working directory and the standard tool belt
(Read, Edit, Write, Bash, Glob, Grep). It works on the message, produces
a final text reply, and returns it for the bridge to send back.

v1 choices:
    * Conversational memory is persisted per-JID in a sqlite store
      (:mod:`app.whatsapp.memory_store`). Prior turns are prepended to
      the prompt as a transcript because the stateless ``query()`` entry
      point of ``claude-agent-sdk`` does not accept history directly.
    * One agent at a time — the bridge uses an asyncio lock.
    * Git push is up to the agent itself via the Bash tool; the host must
      have credentials (ssh key or gh auth) already configured.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.whatsapp.config import BridgeConfig
    from app.whatsapp.memory_store import SessionStore

logger = logging.getLogger("mkangel.whatsapp.agent")


SYSTEM_PROMPT = """\
You are MKAngel's remote agent, reached over WhatsApp by Matt (Matthew
Kilcoyne of CANZUK Ltd). You are operating on the real MKAngel repository
on his phone or VPS.

Rules:
- Keep replies short and phone-friendly. No long preambles.
- Prefer action over discussion. If Matt asks for a change, make it,
  commit it, and push it to the current branch.
- Never amend or force-push without explicit permission.
- If a request is ambiguous, ask one concise clarifying question rather
  than guessing.
- If you run commands that write to disk or touch git, say so briefly
  in your final reply.
- You have Read, Edit, Write, Bash, Glob, and Grep. Use them.
"""


def _format_prompt_with_history(text: str, history: list[dict]) -> str:
    """
    Prepend prior turns to ``text`` as a plain transcript.

    ``claude-agent-sdk``'s ``query()`` is stateless and offers no direct
    history parameter (only ``continue_conversation`` / ``resume`` which
    key off the SDK's own session IDs, not our WhatsApp JIDs), so we
    inline the transcript into the user message.
    """
    if not history:
        return text
    lines = ["Previous conversation:"]
    for turn in history:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    lines.append("")
    lines.append("New message:")
    lines.append(text)
    return "\n".join(lines)


async def run_agent(
    text: str,
    cfg: "BridgeConfig",
    jid: str,
    store: "SessionStore",
) -> str:
    """
    Run one agent session against ``text`` for the given ``jid``.

    Loads prior turns from ``store``, runs the agent, persists the new
    user+assistant pair, and returns the final assistant text to send
    back on WhatsApp.

    Uses the ``claude-agent-sdk`` Python package. Install with::

        pip install claude-agent-sdk
    """
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
            query,
        )
    except ImportError as exc:
        raise RuntimeError(
            "claude-agent-sdk is not installed. Run:\n"
            "    pip install claude-agent-sdk"
        ) from exc

    history = store.get_history(jid)
    prompt = _format_prompt_with_history(text, history)

    options = ClaudeAgentOptions(
        cwd=str(cfg.repo_root),
        model=cfg.model,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
        max_turns=cfg.max_turns,
        permission_mode="acceptEdits",
    )

    final_text_parts: list[str] = []
    turn_count = 0

    async for message in query(prompt=prompt, options=options):
        turn_count += 1
        msg_type = type(message).__name__
        logger.debug("agent message #%d: %s", turn_count, msg_type)

        # Only AssistantMessage carries model-authored text. Iterate its
        # content blocks and collect TextBlock payloads; ToolUseBlock and
        # ToolResultBlock are ignored for the reply text.
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    block_text = block.text
                    if isinstance(block_text, str) and block_text.strip():
                        final_text_parts.append(block_text)

    reply = "\n\n".join(part.strip() for part in final_text_parts if part.strip())
    if not reply:
        reply = "(agent finished with no text reply)"

    # Persist the turn pair only after we have a reply in hand, so a
    # mid-run crash doesn't leave orphaned user rows with no assistant
    # response.
    store.add_turn(jid, "user", text)
    store.add_turn(jid, "assistant", reply)

    return reply


def repo_snapshot(cfg: "BridgeConfig") -> str:
    """Short status line for debugging — useful when the agent crashes."""
    import subprocess

    try:
        branch = subprocess.check_output(
            ["git", "-C", str(cfg.repo_root), "branch", "--show-current"],
            text=True,
        ).strip()
        sha = subprocess.check_output(
            ["git", "-C", str(cfg.repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
        return f"{branch} @ {sha}"
    except Exception as exc:  # noqa: BLE001
        return f"(no git status: {exc})"
