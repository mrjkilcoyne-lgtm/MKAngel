#!/data/data/com.termux/files/usr/bin/bash
# MKAngel — Termux Installation Script
# Run: bash install_termux.sh

set -e

echo ""
echo "  ██████╗ ███████╗    ███╗   ██╗ ██████╗ ████████╗"
echo "  ██╔══██╗██╔════╝    ████╗  ██║██╔═══██╗╚══██╔══╝"
echo "  ██████╔╝█████╗      ██╔██╗ ██║██║   ██║   ██║   "
echo "  ██╔══██╗██╔══╝      ██║╚██╗██║██║   ██║   ██║   "
echo "  ██████╔╝███████╗    ██║ ╚████║╚██████╔╝   ██║   "
echo "  ╚═════╝ ╚══════╝    ╚═╝  ╚═══╝ ╚═════╝    ╚═╝   "
echo ""
echo "   █████╗ ███████╗██████╗  █████╗ ██╗██████╗ "
echo "  ██╔══██╗██╔════╝██╔══██╗██╔══██╗██║██╔══██╗"
echo "  ███████║█████╗  ██████╔╝███████║██║██║  ██║"
echo "  ██╔══██║██╔══╝  ██╔══██╗██╔══██║██║██║  ██║"
echo "  ██║  ██║██║     ██║  ██║██║  ██║██║██████╔╝"
echo "  ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═════╝ "
echo ""
echo "  Installing MKAngel — Grammar Language Model"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Detect environment
if [ -d "/data/data/com.termux" ]; then
    ENV="termux"
    echo "  ✓ Termux detected"
elif [ "$(uname)" = "Linux" ]; then
    ENV="linux"
    echo "  ✓ Linux detected"
elif [ "$(uname)" = "Darwin" ]; then
    ENV="macos"
    echo "  ✓ macOS detected"
else
    ENV="unknown"
    echo "  ? Unknown environment — attempting install anyway"
fi

# Install Python if needed (Termux)
if [ "$ENV" = "termux" ]; then
    echo ""
    echo "  [1/4] Updating Termux packages..."
    pkg update -y 2>/dev/null || true
    pkg install -y python git 2>/dev/null || true
fi

# Create MKAngel home directory
MKANGEL_HOME="$HOME/.mkangel"
mkdir -p "$MKANGEL_HOME/skills"
mkdir -p "$MKANGEL_HOME/sessions"
echo "  [2/4] Created $MKANGEL_HOME"

# Install MKAngel
echo "  [3/4] Installing MKAngel..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
pip install -e ".[dev]" 2>/dev/null || pip install -e . 2>/dev/null || python3 -m pip install -e .

# Verify installation
echo "  [4/4] Verifying installation..."
python3 -c "from glm.angel import Angel; a = Angel(); a.awaken(); print('  ✓ Angel awakened:', a)"

echo ""
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Installation complete!"
echo ""
echo "  Run MKAngel:"
echo "    mkangel          # Start the app"
echo "    python3 -m app   # Alternative"
echo ""
echo "  The scales are learned. Ready for masterpieces."
echo ""
