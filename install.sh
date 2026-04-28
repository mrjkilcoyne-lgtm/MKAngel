#!/bin/bash
# MKAngel Desktop Installer
# Creates a launcher with icon and start button on Linux

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="MKAngel"
ICON_SOURCE="$SCRIPT_DIR/assets/icon.svg"
INSTALL_DIR="$HOME/.local/share/mkangel"
BIN_DIR="$HOME/.local/bin"
DESKTOP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MKAngel Installer                                      ║"
echo "║  Grammar Language Model — Pure Python, Zero Dependencies║"
echo "╚══════════════════════════════════════════════════════════╝"
echo

# Create directories
echo "[1/5] Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$DESKTOP_DIR"
mkdir -p "$ICON_DIR"

# Copy application
echo "[2/5] Installing application..."
cp -r "$SCRIPT_DIR/glm" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/app" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/assets" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/demo.py" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/pyproject.toml" "$INSTALL_DIR/"

# Install icon
echo "[3/5] Installing icon..."
cp "$ICON_SOURCE" "$ICON_DIR/mkangel.svg"

# Create start script
echo "[4/5] Creating launcher script..."
cat > "$BIN_DIR/mkangel" << 'LAUNCHER'
#!/bin/bash
cd "$HOME/.local/share/mkangel"
exec python3 -c "
import sys
sys.path.insert(0, '.')
from app.conductor import AngelConductor
from app.chat import ChatSession

conductor = AngelConductor().awaken()
session = ChatSession(conductor)
session.run()
"
LAUNCHER
chmod +x "$BIN_DIR/mkangel"

# Create demo shortcut
cat > "$BIN_DIR/mkangel-demo" << 'DEMO'
#!/bin/bash
cd "$HOME/.local/share/mkangel"
exec python3 demo.py
DEMO
chmod +x "$BIN_DIR/mkangel-demo"

# Create .desktop file
echo "[5/5] Creating desktop entry..."
cat > "$DESKTOP_DIR/mkangel.desktop" << EOF
[Desktop Entry]
Name=MKAngel
Comment=Grammar Language Model — 23 domains, 47 grammars, pure Python
Exec=$BIN_DIR/mkangel
Icon=mkangel
Terminal=true
Type=Application
Categories=Development;Science;Education;
Keywords=AI;grammar;language;angel;
StartupNotify=false
EOF

chmod +x "$DESKTOP_DIR/mkangel.desktop"

# Update desktop database if available
if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
fi

# Update icon cache if available
if command -v gtk-update-icon-cache &>/dev/null; then
    gtk-update-icon-cache "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
fi

echo
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Installation complete!                                  ║"
echo "║                                                          ║"
echo "║  To start:     mkangel                                   ║"
echo "║  To demo:      mkangel-demo                              ║"
echo "║                                                          ║"
echo "║  Desktop icon should appear in your application menu.    ║"
echo "║  If not, log out and log back in.                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo
echo "  23 domains | 47 grammars | 1954 words | 388 strange loops"
echo "  Pure Python | Zero dependencies | ~3 MB"
echo
