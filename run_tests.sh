#!/usr/bin/env sh
# ============================================================================
# HDRI Breaker - Linux / macOS Test Runner (Houdini hython)
# ============================================================================

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

HYTHON_BIN=""
for candidate in /opt/hfs21.*/bin/hython /opt/hfs20.*/bin/hython; do
    if [ -x "$candidate" ]; then
        HYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$HYTHON_BIN" ] && command -v hython >/dev/null 2>&1; then
    HYTHON_BIN="hython"
fi

if [ -z "$HYTHON_BIN" ]; then
    echo "[ERROR] Houdini hython not found in /opt/hfs* or PATH."
    exit 1
fi

echo "Running HDRI Breaker test suite with: $HYTHON_BIN"
"$HYTHON_BIN" "$SCRIPT_DIR/tests/test_hdri_maker.py"
