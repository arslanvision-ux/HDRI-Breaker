#!/usr/bin/env sh
# ============================================================================
# HDRI Breaker - Linux (Rocky 8 / Rocky 9 / Ubuntu) & macOS Houdini Installer
# Pure POSIX sh script (LF line endings). Can be run via:
#   sh install.sh
#   ./install.sh
# ============================================================================

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
chmod +x "$SCRIPT_DIR/install.sh" "$SCRIPT_DIR/uninstall.sh" "$SCRIPT_DIR/run_tests.sh" 2>/dev/null || true

echo "========================================================"
echo "  HDRI Breaker (Solaris) - Linux / macOS Installer"
echo "========================================================"

# 1. Prefer Houdini's bundled hython first, then python3, then python
PYTHON_BIN=""
for candidate in /opt/hfs21.*/bin/hython /opt/hfs20.*/bin/hython; do
    if [ -x "$candidate" ]; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ] && command -v hython >/dev/null 2>&1; then
    PYTHON_BIN="hython"
elif [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif [ -z "$PYTHON_BIN" ] && command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

if [ -n "$PYTHON_BIN" ] && [ -f "$SCRIPT_DIR/scripts/install.py" ]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/scripts/install.py" "$@"
    exit $?
fi

# 2. Pure-POSIX shell fallback if no Python interpreter is in PATH
write_pkg_json() {
    pkg_dir="$1"
    mkdir -p "$pkg_dir"
    dest="$pkg_dir/hdri_maker.json"
    cat > "$dest" <<EOF
{
    "env": [
        {
            "HDRI_MAKER_ROOT": "$SCRIPT_DIR"
        },
        {
            "PYTHONPATH": "\${HDRI_MAKER_ROOT}/python"
        },
        {
            "HOUDINI_PATH": "\${HDRI_MAKER_ROOT}/houdini"
        }
    ]
}
EOF
    echo "[OK] Installed HDRI Breaker package -> $dest"
}

FOUND_COUNT=0

if [ -n "$HOUDINI_USER_PREF_DIR" ]; then
    write_pkg_json "$HOUDINI_USER_PREF_DIR/packages"
    FOUND_COUNT=$((FOUND_COUNT + 1))
fi

for d in "$HOME"/houdini20.* "$HOME"/houdini21.* "$HOME"/houdini22.* "$HOME"/.houdini20.* "$HOME"/.houdini21.*; do
    if [ -d "$d" ]; then
        write_pkg_json "$d/packages"
        FOUND_COUNT=$((FOUND_COUNT + 1))
    fi
done

if [ "$FOUND_COUNT" -eq 0 ]; then
    write_pkg_json "$HOME/houdini20.5/packages"
    write_pkg_json "$HOME/houdini21.0/packages"
fi

echo ""
echo "HDRI Breaker installation complete."
echo "Restart Houdini and open the 'HDRI Breaker' Shelf or 'HDRI Breaker (Solaris)' Python Panel."
