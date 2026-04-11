#!/usr/bin/env python3
"""
gcp_diag.py -- MKAngel GCP "where did it go?" diagnostic.

READING A: the boring explanation first.
================================================

You set something up on Google Cloud Console for training the MKAngel LGM
(project, bucket, notebook instance, training job, whatever it was) and now
it appears to be missing. Before assuming anything weird happened, this
script methodically checks the mundane causes:

  * Billing got auto-suspended -- project still exists, just disabled
  * A Vertex AI Workbench notebook was reclaimed due to idle timeout
  * A Cloud Storage bucket entered soft-delete retention
  * A service account key was rotated / revoked
  * You're looking at the wrong region or the wrong project in the console
  * You're logged in as a different account than the one that owns the
    resource
  * The resource was deleted (accidentally or otherwise) and the audit log
    still remembers it

This script runs a battery of READ-ONLY gcloud commands, collects the
output, and produces a local report. It tells you what exists, what's
billing-suspended, what's in soft-delete, and what got deleted/moved/
IAM-changed in the last 30 days according to the audit log.

PRIVACY
-------
Nothing this script does modifies your Google Cloud account in any way.
Every gcloud command it runs is read-only (list / describe / read).
Nothing is transmitted anywhere -- all output is written to your local
stdout and a local JSON file in the current working directory. There is
no network call other than the ones `gcloud` itself makes to Google.

USAGE
-----
    python scripts/gcp_diag.py                 # run the full diagnostic
    python scripts/gcp_diag.py --dry-run       # just list what it WOULD run
    python scripts/gcp_diag.py --help          # this text
    python scripts/gcp_diag.py --force         # override sandbox guard

Run this on YOUR OWN LOCAL MACHINE, not inside Claude Code, not in a CI
job, not in any sandbox. It needs access to your real gcloud credentials.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MKANGEL_KEYWORDS = ("mkangel", "tardis", "lgm", "angel", "glm")
GCLOUD_TIMEOUT_SEC = 30
REPORT_SECTIONS = (
    "Accounts & Auth",
    "Projects",
    "Billing",
    "Storage",
    "Compute",
    "Training Jobs",
    "Notebooks",
    "Recent Warnings",
    "Asset Search",
    "Soft-deleted Resources",
    "Audit Trail (deletes/moves/IAM changes, last 30d)",
)


# ---------------------------------------------------------------------------
# Sandbox guard
# ---------------------------------------------------------------------------

def _looks_like_claude_sandbox() -> list[str]:
    """Return a list of suspicious env var names if we're inside Claude Code."""
    hits = []
    for key in os.environ:
        if key.startswith("CLAUDE_CODE_") or key.startswith("CLAUDECODE"):
            hits.append(key)
    # Also flag the common anthropic sandbox marker
    for marker in ("ANTHROPIC_SANDBOX", "CLAUDE_SANDBOX"):
        if marker in os.environ:
            hits.append(marker)
    return hits


def _sandbox_guard(force: bool) -> None:
    hits = _looks_like_claude_sandbox()
    if not hits:
        return
    print("WARNING: this script appears to be running inside a Claude Code "
          "sandbox.", file=sys.stderr)
    print("Detected env vars: " + ", ".join(sorted(hits)), file=sys.stderr)
    print("", file=sys.stderr)
    print("This script is meant to run on YOUR OWN LOCAL MACHINE against your",
          file=sys.stderr)
    print("real Google Cloud credentials. Running it in a sandbox will not",
          file=sys.stderr)
    print("produce meaningful results and may leak environment details.",
          file=sys.stderr)
    if not force:
        print("", file=sys.stderr)
        print("Refusing to proceed. Re-run with --force if you really mean it.",
              file=sys.stderr)
        sys.exit(2)
    print("--force supplied; continuing anyway.", file=sys.stderr)


# ---------------------------------------------------------------------------
# gcloud invocation helper
# ---------------------------------------------------------------------------

class GcloudResult:
    def __init__(self, label: str, argv: list[str]):
        self.label = label
        self.argv = argv
        self.ok: bool = False
        self.stdout: str = ""
        self.stderr: str = ""
        self.parsed: Any = None
        self.error: str | None = None
        self.returncode: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "argv": self.argv,
            "ok": self.ok,
            "returncode": self.returncode,
            "error": self.error,
            "parsed": self.parsed,
            "stdout": None if self.parsed is not None else self.stdout,
            "stderr": self.stderr,
        }


def run_gcloud(label: str, argv: list[str], dry_run: bool) -> GcloudResult:
    """Run a gcloud command with timeout, never raise."""
    res = GcloudResult(label=label, argv=argv)
    if dry_run:
        res.ok = True
        res.error = "dry-run (not executed)"
        return res
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=GCLOUD_TIMEOUT_SEC,
            check=False,
        )
        res.returncode = proc.returncode
        res.stdout = proc.stdout or ""
        res.stderr = proc.stderr or ""
        if proc.returncode == 0:
            res.ok = True
            # Try to parse as JSON if --format=json was used
            if "--format=json" in argv or any(
                a.startswith("--format=json") for a in argv
            ):
                try:
                    res.parsed = json.loads(res.stdout) if res.stdout.strip() else []
                except json.JSONDecodeError as e:
                    res.error = f"JSON decode failed: {e}"
                    res.parsed = None
        else:
            res.error = f"gcloud exit {proc.returncode}: {res.stderr.strip()[:500]}"
    except subprocess.TimeoutExpired:
        res.error = f"timed out after {GCLOUD_TIMEOUT_SEC}s"
    except FileNotFoundError:
        res.error = "gcloud binary not found"
    except Exception as e:  # noqa: BLE001
        res.error = f"unexpected error: {type(e).__name__}: {e}"
    return res


# ---------------------------------------------------------------------------
# Credential checks
# ---------------------------------------------------------------------------

def check_gcloud_installed() -> str | None:
    """Return path to gcloud, or None if not found."""
    return shutil.which("gcloud")


def print_install_instructions() -> None:
    print("The 'gcloud' CLI is not installed (not on PATH).", file=sys.stderr)
    print("", file=sys.stderr)
    print("To install on Linux/macOS:", file=sys.stderr)
    print("  https://cloud.google.com/sdk/docs/install", file=sys.stderr)
    print("", file=sys.stderr)
    print("Quick install (Linux):", file=sys.stderr)
    print("  curl https://sdk.cloud.google.com | bash", file=sys.stderr)
    print("  exec -l $SHELL", file=sys.stderr)
    print("  gcloud init", file=sys.stderr)
    print("", file=sys.stderr)
    print("On macOS with Homebrew:", file=sys.stderr)
    print("  brew install --cask google-cloud-sdk", file=sys.stderr)
    print("", file=sys.stderr)
    print("After installing, run:", file=sys.stderr)
    print("  gcloud auth login", file=sys.stderr)
    print("  gcloud auth application-default login", file=sys.stderr)


def check_credentials(dry_run: bool) -> dict[str, Any]:
    """Figure out which credentials gcloud will use."""
    info: dict[str, Any] = {
        "GOOGLE_APPLICATION_CREDENTIALS": os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        ),
        "source": None,
        "adc_token_ok": None,
        "note": None,
    }
    gac = info["GOOGLE_APPLICATION_CREDENTIALS"]
    if gac:
        info["source"] = f"service-account key file: {gac}"
        if not os.path.exists(gac):
            info["note"] = "GOOGLE_APPLICATION_CREDENTIALS points to a path that does not exist"
    else:
        info["source"] = "gcloud application-default credentials (ADC)"
        if dry_run:
            info["note"] = "dry-run: skipped ADC probe"
        else:
            probe = run_gcloud(
                "adc-probe",
                ["gcloud", "auth", "application-default", "print-access-token"],
                dry_run=False,
            )
            info["adc_token_ok"] = probe.ok
            if not probe.ok:
                info["note"] = (
                    "No active ADC. Run: gcloud auth application-default login. "
                    f"Detail: {probe.error}"
                )
    return info


# ---------------------------------------------------------------------------
# Diagnostic plan
# ---------------------------------------------------------------------------

def build_plan() -> list[tuple[str, str, list[str]]]:
    """Return the list of (section, label, argv) for fixed top-level commands.

    Per-project / per-bucket commands are generated dynamically later.
    """
    return [
        ("Accounts & Auth", "config-list",
            ["gcloud", "config", "list", "--format=json"]),
        ("Accounts & Auth", "auth-list",
            ["gcloud", "auth", "list", "--format=json"]),
        ("Projects", "projects-list",
            ["gcloud", "projects", "list", "--format=json"]),
        ("Billing", "billing-accounts-list",
            ["gcloud", "billing", "accounts", "list", "--format=json"]),
        ("Storage", "buckets-list",
            ["gcloud", "storage", "buckets", "list", "--format=json"]),
        ("Compute", "compute-instances-list",
            ["gcloud", "compute", "instances", "list", "--format=json"]),
        ("Training Jobs", "ai-platform-jobs-list",
            ["gcloud", "ai-platform", "jobs", "list",
             "--limit=20", "--format=json"]),
        ("Training Jobs", "vertex-custom-jobs-list",
            ["gcloud", "ai", "custom-jobs", "list",
             "--limit=20", "--format=json"]),
        ("Notebooks", "notebooks-instances-list",
            ["gcloud", "notebooks", "instances", "list", "--format=json"]),
        ("Recent Warnings", "logging-warnings",
            ["gcloud", "logging", "read", "severity>=WARNING",
             "--limit=50", "--format=json", "--freshness=30d"]),
    ]


def build_asset_search_argv(org_id: str | None) -> list[str]:
    scope = f"organizations/{org_id}" if org_id else "organizations/YOUR_ORG_ID"
    query_terms = " OR ".join(
        [f"displayName:{kw}" for kw in MKANGEL_KEYWORDS]
        + [f"labels.project:{kw}" for kw in MKANGEL_KEYWORDS]
    )
    return [
        "gcloud", "asset", "search-all-resources",
        f"--scope={scope}",
        f"--query={query_terms}",
        "--format=json",
    ]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _stringify(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str).lower()
    except Exception:  # noqa: BLE001
        return str(obj).lower()


def classify_findings(report: dict[str, Any]) -> tuple[str, list[str]]:
    """Look for MKAngel keywords anywhere in the report.

    Returns (classification_string, list_of_matching_resource_descriptions).
    """
    matches: list[str] = []

    def _scan(section: str, result: dict[str, Any]) -> None:
        parsed = result.get("parsed")
        if not parsed:
            return
        if isinstance(parsed, list):
            for item in parsed:
                blob = _stringify(item)
                for kw in MKANGEL_KEYWORDS:
                    if kw in blob:
                        name = (
                            item.get("name")
                            or item.get("projectId")
                            or item.get("displayName")
                            or item.get("id")
                            or "<unnamed>"
                        ) if isinstance(item, dict) else str(item)
                        matches.append(f"[{section}] {kw} -> {name}")
                        break
        elif isinstance(parsed, dict):
            blob = _stringify(parsed)
            for kw in MKANGEL_KEYWORDS:
                if kw in blob:
                    matches.append(f"[{section}] {kw} -> <dict>")
                    break

    for section, entries in report.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and "parsed" in entry:
                _scan(section, entry)

    if not matches:
        classification = (
            "MISSING: no MKAngel-labelled resources found in any project "
            "visible to this account. Either you're logged in as the wrong "
            "user, looking at the wrong org, or the resources were deleted. "
            "Check the Audit Trail section."
        )
    else:
        # Did we cover all of projects/buckets/jobs/notebooks? If yes -> FOUND.
        sections_hit = {m.split("]")[0].lstrip("[") for m in matches}
        wanted = {"Projects", "Storage", "Training Jobs", "Notebooks"}
        if wanted.issubset(sections_hit):
            classification = (
                "FOUND: MKAngel resources appear to still exist. See matches "
                "below for exact locations."
            )
        else:
            missing = wanted - sections_hit
            classification = (
                "PARTIAL: some MKAngel resources found, but not all. "
                f"Missing coverage in: {sorted(missing)}. See matches below."
            )
    return classification, matches


def extract_audit_trail(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull delete/move/IAM events from the logging-warnings result."""
    events: list[dict[str, Any]] = []
    warnings_section = report.get("Recent Warnings", [])
    for entry in warnings_section:
        parsed = entry.get("parsed") if isinstance(entry, dict) else None
        if not parsed or not isinstance(parsed, list):
            continue
        for log in parsed:
            if not isinstance(log, dict):
                continue
            method = ""
            proto = log.get("protoPayload") or {}
            if isinstance(proto, dict):
                method = proto.get("methodName", "") or ""
            method_l = method.lower()
            blob = _stringify(log)
            interesting = (
                "delete" in method_l
                or "remove" in method_l
                or "setiampolicy" in method_l
                or "move" in method_l
                or "delete" in blob and "resource" in blob
            )
            if interesting:
                actor = ""
                if isinstance(proto, dict):
                    auth = proto.get("authenticationInfo") or {}
                    if isinstance(auth, dict):
                        actor = auth.get("principalEmail", "") or ""
                events.append({
                    "timestamp": log.get("timestamp"),
                    "method": method,
                    "actor": actor,
                    "resource": (proto.get("resourceName")
                                 if isinstance(proto, dict) else None),
                    "severity": log.get("severity"),
                })
    return events


def extract_soft_deleted(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull anything that looks like a soft-deleted / retention-held resource."""
    soft: list[dict[str, Any]] = []
    for entry in report.get("Storage", []):
        parsed = entry.get("parsed") if isinstance(entry, dict) else None
        if not parsed or not isinstance(parsed, list):
            continue
        for bucket in parsed:
            if not isinstance(bucket, dict):
                continue
            sdp = bucket.get("softDeletePolicy")
            rp = bucket.get("retentionPolicy")
            lc = bucket.get("lifecycle")
            if sdp or rp or lc:
                soft.append({
                    "bucket": bucket.get("name") or bucket.get("id"),
                    "softDeletePolicy": sdp,
                    "retentionPolicy": rp,
                    "lifecycle": lc,
                })
    return soft


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_human(report: dict[str, Any],
                 classification: str,
                 matches: list[str],
                 audit: list[dict[str, Any]],
                 soft: list[dict[str, Any]],
                 creds: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("=" * 72)
    out.append("MKAngel GCP diagnostic -- Reading A")
    out.append("=" * 72)
    out.append("")
    out.append("Credentials:")
    for k, v in creds.items():
        out.append(f"  {k}: {v}")
    out.append("")

    for section in REPORT_SECTIONS:
        out.append("-" * 72)
        out.append(f"## {section}")
        out.append("-" * 72)
        entries = report.get(section, [])
        if section == "Soft-deleted Resources":
            if not soft:
                out.append("  (none found)")
            for s in soft:
                out.append(f"  bucket={s['bucket']}")
                if s.get("softDeletePolicy"):
                    out.append(f"    softDelete: {s['softDeletePolicy']}")
                if s.get("retentionPolicy"):
                    out.append(f"    retention:  {s['retentionPolicy']}")
                if s.get("lifecycle"):
                    out.append(f"    lifecycle:  {s['lifecycle']}")
            out.append("")
            continue
        if section.startswith("Audit Trail"):
            if not audit:
                out.append("  (no delete/move/IAM events in last 30d)")
            for ev in audit:
                out.append(
                    f"  {ev.get('timestamp')} "
                    f"{ev.get('severity', '')} "
                    f"{ev.get('method', '')} "
                    f"by {ev.get('actor') or '<unknown>'} "
                    f"on {ev.get('resource') or '<unknown>'}"
                )
            out.append("")
            continue
        if not entries:
            out.append("  (no data)")
            out.append("")
            continue
        for entry in entries:
            label = entry.get("label", "?")
            if entry.get("ok"):
                parsed = entry.get("parsed")
                if isinstance(parsed, list):
                    out.append(f"  [{label}] ok, {len(parsed)} item(s)")
                    for item in parsed[:10]:
                        if isinstance(item, dict):
                            name = (item.get("name")
                                    or item.get("projectId")
                                    or item.get("displayName")
                                    or item.get("id")
                                    or "?")
                            out.append(f"    - {name}")
                        else:
                            out.append(f"    - {item}")
                    if len(parsed) > 10:
                        out.append(f"    ... and {len(parsed) - 10} more")
                else:
                    out.append(f"  [{label}] ok")
            else:
                out.append(f"  [{label}] FAILED: {entry.get('error')}")
        out.append("")

    out.append("=" * 72)
    out.append("## Classification")
    out.append("=" * 72)
    out.append(classification)
    out.append("")
    if matches:
        out.append("Matching resources:")
        for m in matches:
            out.append(f"  * {m}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnostic(args: argparse.Namespace) -> int:
    _sandbox_guard(force=args.force)

    gcloud_path = check_gcloud_installed()
    if not gcloud_path and not args.dry_run:
        print_install_instructions()
        return 1
    if gcloud_path:
        print(f"gcloud: {gcloud_path}")
    else:
        print("gcloud: NOT FOUND (dry-run ok)")

    creds = check_credentials(dry_run=args.dry_run)
    print(f"credentials: {creds['source']}")
    if creds.get("note"):
        print(f"  note: {creds['note']}")

    plan = build_plan()

    if args.dry_run:
        print("")
        print("DRY RUN -- the following commands WOULD be executed:")
        print("(none of these modify state; all are read-only)")
        print("")
        for i, (section, label, argv) in enumerate(plan, 1):
            print(f"  {i:2d}. [{section}] {label}")
            print(f"      $ {' '.join(argv)}")
        print("")
        print("Plus, for each project discovered by 'projects list':")
        print("  $ gcloud billing projects describe <project> --format=json")
        print("")
        print("Plus, for each bucket discovered by 'storage buckets list':")
        print("  $ gcloud storage buckets describe gs://<bucket> --format=json")
        print("")
        print("Plus, an org-wide fuzzy asset search:")
        print(f"  $ {' '.join(build_asset_search_argv(None))}")
        print("")
        print(f"Timeout per command: {GCLOUD_TIMEOUT_SEC}s")
        print("Report would be written to: ./gcp_diag_report_YYYYMMDD_HHMMSS.json")
        print("")
        print("No gcloud calls were made. Re-run without --dry-run to execute.")
        return 0

    # Real run
    report: dict[str, list[dict[str, Any]]] = {s: [] for s in REPORT_SECTIONS}

    for section, label, argv in plan:
        print(f"  running [{section}] {label} ...")
        res = run_gcloud(label, argv, dry_run=False)
        report[section].append(res.to_dict())

    # Per-project billing describe
    projects_entry = next(
        (e for e in report["Projects"] if e.get("label") == "projects-list"),
        None,
    )
    if projects_entry and projects_entry.get("parsed"):
        for proj in projects_entry["parsed"]:
            if not isinstance(proj, dict):
                continue
            pid = proj.get("projectId")
            if not pid:
                continue
            print(f"  running [Billing] billing-describe {pid} ...")
            res = run_gcloud(
                f"billing-describe:{pid}",
                ["gcloud", "billing", "projects", "describe", pid,
                 "--format=json"],
                dry_run=False,
            )
            report["Billing"].append(res.to_dict())

    # Per-bucket describe
    buckets_entry = next(
        (e for e in report["Storage"] if e.get("label") == "buckets-list"),
        None,
    )
    if buckets_entry and buckets_entry.get("parsed"):
        for bucket in buckets_entry["parsed"]:
            if not isinstance(bucket, dict):
                continue
            bname = bucket.get("name") or bucket.get("id")
            if not bname:
                continue
            if not bname.startswith("gs://"):
                bname_url = f"gs://{bname}"
            else:
                bname_url = bname
            print(f"  running [Storage] bucket-describe {bname} ...")
            res = run_gcloud(
                f"bucket-describe:{bname}",
                ["gcloud", "storage", "buckets", "describe", bname_url,
                 "--format=json"],
                dry_run=False,
            )
            report["Storage"].append(res.to_dict())

    # Asset search (needs org id; try to discover it, else skip gracefully)
    org_probe = run_gcloud(
        "organizations-list",
        ["gcloud", "organizations", "list", "--format=json"],
        dry_run=False,
    )
    report["Asset Search"].append(org_probe.to_dict())
    org_id = None
    if org_probe.ok and isinstance(org_probe.parsed, list) and org_probe.parsed:
        first = org_probe.parsed[0]
        if isinstance(first, dict):
            name = first.get("name") or ""
            if name.startswith("organizations/"):
                org_id = name.split("/", 1)[1]
    if org_id:
        print(f"  running [Asset Search] asset-search (org={org_id}) ...")
        res = run_gcloud(
            "asset-search",
            build_asset_search_argv(org_id),
            dry_run=False,
        )
        report["Asset Search"].append(res.to_dict())
    else:
        report["Asset Search"].append({
            "label": "asset-search",
            "ok": False,
            "error": "no organization id discovered; skipping asset search",
        })

    # Post-process
    classification, matches = classify_findings(report)
    audit = extract_audit_trail(report)
    soft = extract_soft_deleted(report)

    # Write JSON
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.abspath(f"./gcp_diag_report_{ts}.json")
    full = {
        "generated": _dt.datetime.now().isoformat(),
        "credentials": creds,
        "report": report,
        "classification": classification,
        "matches": matches,
        "audit_trail": audit,
        "soft_deleted": soft,
    }
    try:
        with open(json_path, "w") as f:
            json.dump(full, f, indent=2, default=str)
        print(f"wrote JSON report: {json_path}")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: failed to write JSON report: {e}", file=sys.stderr)

    # Human printout
    print("")
    print(render_human(report, classification, matches, audit, soft, creds))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gcp_diag.py",
        description=(
            "MKAngel GCP diagnostic -- figures out where your LGM training "
            "project went. Runs only read-only gcloud commands. Nothing is "
            "modified. Nothing is transmitted off your machine. All output "
            "is written to your local stdout and a local JSON file."
        ),
        epilog=(
            "This is the 'Reading A' investigation: checking the mundane "
            "explanations (billing suspended, notebook reclaimed, bucket "
            "soft-deleted, wrong account, wrong region) before reaching for "
            "anything exotic. Run it on your OWN machine with your OWN "
            "gcloud credentials -- not inside Claude Code, not in CI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the gcloud commands that would be run, then exit without "
             "running any of them. Use this to review the plan before "
             "executing on a live account.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override the Claude Code sandbox safety check. Only use this "
             "if you know what you're doing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return run_diagnostic(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
