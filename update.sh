#!/usr/bin/env sh
# ============================================================================
# HDRI Breaker - Linux & macOS Self-Updater
# Pure POSIX sh script (LF line endings). Can be run via:
#   sh update.sh
#   ./update.sh
# ============================================================================

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_TARBALL_URL="https://github.com/arslanvision-ux/HDRI-Breaker/archive/refs/heads/main.tar.gz"

echo "========================================================"
echo "  HDRI Breaker (Solaris) - Updating from GitHub"
echo "========================================================"

if [ -d "$SCRIPT_DIR/.git" ] && command -v git >/dev/null 2>&1; then
    echo "Pulling latest main via git..."
    git -C "$SCRIPT_DIR" pull --ff-only origin main
elif command -v curl >/dev/null 2>&1; then
    echo "Downloading latest main via curl..."
    curl -fsSL "$REPO_TARBALL_URL" | tar -xz --strip-components=1 -C "$SCRIPT_DIR"
elif command -v wget >/dev/null 2>&1; then
    echo "Downloading latest main via wget..."
    wget -qO- "$REPO_TARBALL_URL" | tar -xz --strip-components=1 -C "$SCRIPT_DIR"
else
    echo "ERROR: Neither git, curl, nor wget was found." >&2
    exit 1
fi

chmod +x "$SCRIPT_DIR/install.sh" "$SCRIPT_DIR/uninstall.sh" "$SCRIPT_DIR/update.sh" "$SCRIPT_DIR/run_tests.sh" 2>/dev/null || true

echo ""
echo "Refreshing Houdini package registration..."
sh "$SCRIPT_DIR/install.sh" "$@"
