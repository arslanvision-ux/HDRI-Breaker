#!/usr/bin/env sh
# ============================================================================
# HDRI Breaker - Linux (Rocky 8 / Rocky 9 / Ubuntu) & macOS Uninstaller
# ============================================================================

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if command -v python3 >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/scripts/install.py" ]; then
    python3 "$SCRIPT_DIR/scripts/install.py" --uninstall
    exit $?
fi

for d in "$HOME"/houdini20.* "$HOME"/houdini21.* "$HOME"/houdini22.* "$HOME"/.houdini20.* "$HOME"/.houdini21.*; do
    if [ -f "$d/packages/hdri_maker.json" ]; then
        rm -f "$d/packages/hdri_maker.json"
        echo "[OK] Removed $d/packages/hdri_maker.json"
    fi
    if [ -f "$d/packages/hdri_breaker.json" ]; then
        rm -f "$d/packages/hdri_breaker.json"
        echo "[OK] Removed $d/packages/hdri_breaker.json"
    fi
done
