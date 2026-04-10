#!/usr/bin/env bash
# Launch the MKAngel WhatsApp bridge.
#
# Usage:
#     ./scripts/run_whatsapp_bridge.sh
#
# On Termux: install node and python first, then run this.
# See docs/whatsapp_bridge.md for the full setup.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# First run: install Node deps.
if [ ! -d "app/whatsapp/baileys/node_modules" ]; then
    echo "[bridge] installing Baileys (npm)..."
    (cd app/whatsapp/baileys && npm install --no-audit --no-fund)
fi

# First run: install Python dep.
if ! python3 -c "import claude_agent_sdk" 2>/dev/null; then
    echo "[bridge] installing claude-agent-sdk (pip)..."
    python3 -m pip install --quiet claude-agent-sdk
fi

exec python3 -m app.whatsapp.bridge
