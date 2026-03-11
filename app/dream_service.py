"""
Dream service logic -- decoupled from Android for testability.

Runs the Dreamer pipeline:
  1. Start dream session in Memory
  2. Run Dreamer.dream()
  3. End dream session with artifact count
  4. Return results

The Android foreground service (in main_android.py) wraps this
with notification management and pyjnius calls.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


def run_dream_cycle(
    angel: Any,
    memory: Any,
    voice: Any,
    self_improver: Any | None = None,
    trigger_type: str = "self",
) -> dict[str, Any]:
    """Run one complete dream cycle.

    Args:
        angel: The Angel instance.
        memory: The Memory instance.
        voice: The Voice instance.
        self_improver: Optional SelfImprover.
        trigger_type: What triggered the dream (self/context/user).

    Returns:
        {
            "artifacts": list[DreamArtifact],
            "session_id": int,
            "duration_seconds": float,
            "success": bool,
            "error": str | None,
        }
    """
    from glm.dreamer import Dreamer

    start = time.time()
    result: dict[str, Any] = {
        "artifacts": [],
        "session_id": 0,
        "duration_seconds": 0.0,
        "success": False,
        "error": None,
    }

    try:
        # 1. Start dream session
        pattern_snapshot: dict[str, Any] = {}
        if self_improver is not None:
            try:
                pattern_snapshot = self_improver.analyse_performance()
                # Convert sets to lists for JSON serialisation
                metrics = pattern_snapshot.get("session_metrics", {})
                if isinstance(metrics.get("domains_used"), set):
                    metrics["domains_used"] = list(metrics["domains_used"])
            except Exception:
                pass

        session_id = memory.start_dream_session(
            trigger_type=trigger_type,
            pattern_buffer_snapshot=pattern_snapshot,
        )
        result["session_id"] = session_id

        # 2. Run dreamer
        dreamer = Dreamer()
        artifacts = dreamer.dream(
            angel=angel,
            memory=memory,
            voice=voice,
            self_improver=self_improver,
        )
        result["artifacts"] = artifacts

        # 3. End dream session
        memory.end_dream_session(
            session_id=session_id,
            artifacts_count=len(artifacts),
        )

        result["success"] = True

    except Exception as exc:
        log.error("Dream cycle failed: %s", exc)
        result["error"] = str(exc)

    result["duration_seconds"] = time.time() - start
    return result
