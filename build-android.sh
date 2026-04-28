#!/bin/bash
# MKAngel Android Build Script
# Produces a debug APK for arm64-v8a devices
#
# Prerequisites:
#   pip install buildozer
#   (buildozer handles SDK/NDK installation automatically)
#
# Usage:
#   ./build-android.sh
#
# Output:
#   bin/MKAngel-0.2.0-arm64-v8a_debug.apk

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MKAngel Android Build                                   ║"
echo "║  Target: arm64-v8a | API 33 | minAPI 24                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo

# Step 1: Copy main_android.py to main.py (buildozer requirement)
echo "[1/3] Preparing entry point..."
cp main_android.py main.py
echo "  main_android.py -> main.py"

# Step 2: Ensure icon exists as PNG (convert from SVG if needed)
if [ ! -f assets/icon.png ]; then
    echo "[INFO] No icon.png found. Using SVG directly."
    echo "  (For production, convert assets/icon.svg to 512x512 PNG)"
fi

# Step 3: Build
echo "[2/3] Building APK (this takes 5-15 minutes first time)..."
echo "  Buildozer will download Android SDK/NDK if not present."
echo
yes | buildozer -v android debug 2>&1 | tail -20

# Step 4: Report
echo
echo "[3/3] Build complete!"
APK=$(find bin/ -name "*.apk" 2>/dev/null | head -1)
if [ -n "$APK" ]; then
    SIZE=$(du -h "$APK" | cut -f1)
    echo "  APK: $APK ($SIZE)"
    echo
    echo "  Install on device:"
    echo "    adb install $APK"
    echo
    echo "  Or copy to phone and install directly."
else
    echo "  [ERROR] No APK found in bin/"
    echo "  Check buildozer output above for errors."
fi
