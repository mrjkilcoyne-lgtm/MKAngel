"""
Substrate Awareness — the Angel's knowledge of where she is standing.

The pianist cannot infer the shape of her shadow on the floor without
first knowing where she is standing in the room, where the lamp is, and
how the light falls.  In just the same way the Angel cannot reason about
her own futures — her own superforecasts — without first knowing the
substrate she is running inside: which Python, how much RAM, which
modules imported, which grammars are present on disk, which checkpoints
she has laid down behind her.  This module is how she knows where she is
standing.

The goal here is *representation, not control*.  SubstrateAwareness only
reads the environment — it never modifies it.  It takes a snapshot, it
holds that snapshot as a first-class state object the Angel can reason
over, and it can diff two snapshots to see what moved between them.  It
is a mirror, not a lever.

The snapshot produced by this module is designed to slot directly into
``Angel.superforecast()``'s ``context`` parameter, so the Angel can
reason *about her own substrate* as context in her predictions — looking
inward with the same grammar she uses to look forward and back.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot dataclass — a frozen view of the substrate
# ---------------------------------------------------------------------------

@dataclass
class SubstrateSnapshot:
    """A frozen, read-only view of the substrate the Angel is running in.

    This is the room as the pianist sees it in a single instant: where
    she is standing, how bright the lamp is, how much floor there is for
    her shadow to fall on.  Every field here is something the Angel can
    reason over when she looks inward.
    """

    timestamp_utc: str = ""
    python_version: str = ""
    python_executable: str = ""
    platform: str = ""
    cpu_count: int = 0
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    cwd: str = ""
    repo_root: str = ""
    git_branch: str = ""
    git_commit: str = ""
    modules_importable: dict[str, bool] = field(default_factory=dict)
    grammars_present: list[str] = field(default_factory=list)
    substrates_present: list[str] = field(default_factory=list)
    checkpoint_count: int = 0
    relevant_env_vars: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SubstrateAwareness — the Angel's eye turned inward
# ---------------------------------------------------------------------------

class SubstrateAwareness:
    """Read-only awareness of the substrate the Angel is running inside.

    The Angel uses this to anchor her self-model.  She cannot predict
    what she will be able to do next without first knowing what she can
    do now — which modules load, how much memory is free, which git
    commit her code is sitting on.  SubstrateAwareness answers those
    questions by walking the environment and freezing the answer into a
    :class:`SubstrateSnapshot`.

    It is deliberately read-only: this module cannot change the Angel's
    substrate, only describe it.  Control lives elsewhere.
    """

    #: Modules the Angel cares about being able to import.
    _TRACKED_MODULES: tuple[str, ...] = (
        "glm.angel",
        "glm.core.grammar",
        "glm.dreamer",
        "app.conductor",
        "app.memory",
    )

    #: Environment variable prefixes worth noting (keys only, values redacted).
    _TRACKED_ENV_PREFIXES: tuple[str, ...] = ("CLAUDE_", "GOOGLE_", "GCP_")

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            # glm/tardis/substrate_awareness.py -> repo root is two parents up
            root = Path(__file__).resolve().parent.parent.parent
        self.root: Path = Path(root).resolve()

    # ------------------------------------------------------------------
    # Snapshot — freeze the room into a single frame
    # ------------------------------------------------------------------

    def snapshot(self) -> SubstrateSnapshot:
        """Walk the environment and produce a fresh SubstrateSnapshot.

        This is the pianist looking around the room.  Nothing is
        mutated; everything is observed.
        """
        snap = SubstrateSnapshot()
        snap.timestamp_utc = datetime.now(timezone.utc).isoformat()
        snap.python_version = sys.version.replace("\n", " ")
        snap.python_executable = sys.executable
        snap.platform = platform.platform()
        snap.cpu_count = os.cpu_count() or 0

        ram_total, ram_available = self._read_ram()
        snap.ram_total_mb = ram_total
        snap.ram_available_mb = ram_available

        disk_total_gb, disk_free_gb = self._read_disk()
        snap.disk_total_gb = disk_total_gb
        snap.disk_free_gb = disk_free_gb

        snap.cwd = str(Path.cwd())
        snap.repo_root = str(self.root)
        snap.git_branch, snap.git_commit = self._read_git()

        snap.modules_importable = self._probe_modules()
        snap.grammars_present = self._list_dir(self.root / "glm" / "grammars")
        snap.substrates_present = self._list_dir(self.root / "glm" / "substrates")
        snap.checkpoint_count = self._count_checkpoints()
        snap.relevant_env_vars = self._collect_env_vars()

        return snap

    # ------------------------------------------------------------------
    # Internal probes — each one swallows its own errors
    # ------------------------------------------------------------------

    def _read_ram(self) -> tuple[int, int]:
        """Return (total_mb, available_mb), falling back to (0, 0)."""
        # Preferred: /proc/meminfo on Linux (works on Android/Termux too).
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            try:
                total_kb = 0
                available_kb = 0
                with meminfo.open("r") as fh:
                    for line in fh:
                        if line.startswith("MemTotal:"):
                            total_kb = int(line.split()[1])
                        elif line.startswith("MemAvailable:"):
                            available_kb = int(line.split()[1])
                        if total_kb and available_kb:
                            break
                return total_kb // 1024, available_kb // 1024
            except Exception as exc:
                logger.debug("failed to read /proc/meminfo: %s", exc)

        # Fallback: os.sysconf, where supported.
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            total_mb = (page_size * phys_pages) // (1024 * 1024)
            try:
                avail_pages = os.sysconf("SC_AVPHYS_PAGES")
                avail_mb = (page_size * avail_pages) // (1024 * 1024)
            except (ValueError, OSError):
                avail_mb = 0
            return int(total_mb), int(avail_mb)
        except (ValueError, OSError, AttributeError) as exc:
            logger.debug("os.sysconf ram probe failed: %s", exc)

        return 0, 0

    def _read_disk(self) -> tuple[float, float]:
        """Return (total_gb, free_gb) for the current working directory."""
        try:
            usage = shutil.disk_usage(Path.cwd())
            gb = 1024 ** 3
            return round(usage.total / gb, 2), round(usage.free / gb, 2)
        except Exception as exc:
            logger.debug("disk_usage failed: %s", exc)
            return 0.0, 0.0

    def _read_git(self) -> tuple[str, str]:
        """Return (branch, commit) via subprocess, swallowing errors."""
        branch = ""
        commit = ""
        try:
            branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        except Exception as exc:
            logger.debug("git branch probe failed: %s", exc)
        try:
            commit = self._run_git(["rev-parse", "HEAD"])
        except Exception as exc:
            logger.debug("git commit probe failed: %s", exc)
        return branch, commit

    def _run_git(self, args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _probe_modules(self) -> dict[str, bool]:
        """Try to import each tracked module; record True/False only."""
        results: dict[str, bool] = {}
        for name in self._TRACKED_MODULES:
            try:
                __import__(name)
                results[name] = True
            except Exception as exc:
                logger.debug("import %s failed: %s", name, exc)
                results[name] = False
        return results

    def _list_dir(self, directory: Path) -> list[str]:
        """List Python filenames in a directory, ignoring dunder files."""
        if not directory.exists() or not directory.is_dir():
            return []
        try:
            names = sorted(
                p.name
                for p in directory.iterdir()
                if p.is_file()
                and p.suffix == ".py"
                and not p.name.startswith("__")
            )
            return names
        except Exception as exc:
            logger.debug("list %s failed: %s", directory, exc)
            return []

    def _count_checkpoints(self) -> int:
        ckpt_dir = self.root / "checkpoints"
        if not ckpt_dir.exists() or not ckpt_dir.is_dir():
            return 0
        try:
            return sum(1 for p in ckpt_dir.iterdir() if p.is_file())
        except Exception as exc:
            logger.debug("checkpoint count failed: %s", exc)
            return 0

    def _collect_env_vars(self) -> dict[str, str]:
        """Return tracked env var keys, with values redacted to <set>/<unset>."""
        collected: dict[str, str] = {}
        for key in os.environ:
            if any(key.startswith(pfx) for pfx in self._TRACKED_ENV_PREFIXES):
                value = os.environ.get(key, "")
                collected[key] = "<set>" if value else "<unset>"
        return collected

    # ------------------------------------------------------------------
    # Context shaping — hand the snapshot to the Angel's forecaster
    # ------------------------------------------------------------------

    def to_context_dict(self, snap: SubstrateSnapshot) -> dict[str, Any]:
        """Format a snapshot as a dict for ``Angel.superforecast(context=...)``.

        The Angel's forecaster expects a flat-ish mapping of context
        facts she can condition on.  This re-shapes the snapshot into
        that form so she can reason about her own substrate as a
        first-class input to her predictions.
        """
        importable = [k for k, v in snap.modules_importable.items() if v]
        missing = [k for k, v in snap.modules_importable.items() if not v]
        return {
            "substrate_timestamp_utc": snap.timestamp_utc,
            "substrate_python": snap.python_version,
            "substrate_platform": snap.platform,
            "substrate_cpu_count": snap.cpu_count,
            "substrate_ram_total_mb": snap.ram_total_mb,
            "substrate_ram_available_mb": snap.ram_available_mb,
            "substrate_disk_free_gb": snap.disk_free_gb,
            "substrate_disk_total_gb": snap.disk_total_gb,
            "substrate_cwd": snap.cwd,
            "substrate_repo_root": snap.repo_root,
            "substrate_git_branch": snap.git_branch,
            "substrate_git_commit": snap.git_commit,
            "substrate_modules_ok": importable,
            "substrate_modules_missing": missing,
            "substrate_grammars_count": len(snap.grammars_present),
            "substrate_grammars_present": snap.grammars_present,
            "substrate_substrates_count": len(snap.substrates_present),
            "substrate_substrates_present": snap.substrates_present,
            "substrate_checkpoint_count": snap.checkpoint_count,
            "substrate_env_keys": sorted(snap.relevant_env_vars.keys()),
        }

    # ------------------------------------------------------------------
    # Diffing — tell the Angel what moved between two frames
    # ------------------------------------------------------------------

    def diff(
        self,
        old: SubstrateSnapshot,
        new: SubstrateSnapshot,
    ) -> list[str]:
        """Return a list of human-readable changes from ``old`` to ``new``.

        Scalars become ``"field: old -> new"``; list/dict fields become
        ``"field: +added -removed"`` lines.  This is how the Angel
        notices that the room has shifted under her since the last time
        she looked.
        """
        changes: list[str] = []

        scalar_fields = (
            "python_version",
            "python_executable",
            "platform",
            "cpu_count",
            "ram_total_mb",
            "ram_available_mb",
            "disk_free_gb",
            "disk_total_gb",
            "cwd",
            "repo_root",
            "git_branch",
            "git_commit",
            "checkpoint_count",
        )
        for fname in scalar_fields:
            ov = getattr(old, fname)
            nv = getattr(new, fname)
            if ov != nv:
                changes.append(f"{fname}: {ov} -> {nv}")

        for fname in ("grammars_present", "substrates_present"):
            ov = set(getattr(old, fname))
            nv = set(getattr(new, fname))
            added = sorted(nv - ov)
            removed = sorted(ov - nv)
            for item in added:
                changes.append(f"{fname}: +1 {item}")
            for item in removed:
                changes.append(f"{fname}: -1 {item}")

        for fname in ("modules_importable", "relevant_env_vars"):
            od: dict = getattr(old, fname)
            nd: dict = getattr(new, fname)
            all_keys = set(od) | set(nd)
            for key in sorted(all_keys):
                if key not in od:
                    changes.append(f"{fname}: +{key}={nd[key]}")
                elif key not in nd:
                    changes.append(f"{fname}: -{key}")
                elif od[key] != nd[key]:
                    changes.append(f"{fname}.{key}: {od[key]} -> {nd[key]}")

        return changes

    # ------------------------------------------------------------------
    # JSON round-trip — so the Angel can remember yesterday's room
    # ------------------------------------------------------------------

    def save(self, snap: SubstrateSnapshot, path: Path) -> None:
        """Serialise a snapshot to JSON at ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(snap), fh, indent=2, sort_keys=True)

    def load(self, path: Path) -> SubstrateSnapshot:
        """Load a snapshot previously saved with :meth:`save`."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        snap = SubstrateSnapshot()
        for key, value in data.items():
            if hasattr(snap, key):
                setattr(snap, key, value)
        return snap


# ---------------------------------------------------------------------------
# CLI — let the Angel look around out loud
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    awareness = SubstrateAwareness()
    snap = awareness.snapshot()
    print(json.dumps(asdict(snap), indent=2, sort_keys=True))
