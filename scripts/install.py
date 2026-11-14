#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-platform installer/uninstaller for HDRI Breaker (Solaris 32-bit Float Lat/Long Studio).
Compatible with Python 3.6+ (Rocky Linux 8 / 9 system python3) and Houdini hython (20.0 / 20.5 / 21.0+).
"""

import glob
import json
import os
import platform
import re
import stat
import sys


def _norm_path(p):
    return os.path.abspath(p).replace("\\", "/")


def _discover_houdini_pref_dirs():
    """
    Locate all Houdini 20.0 / 20.5 / 21.0 / 22.0 user preference directories across
    Linux (Rocky 8/9, Ubuntu, studio mounts), Windows, and macOS.
    """
    found = []
    seen = set()

    def _add(path_str):
        if not path_str:
            return
        ap = os.path.abspath(os.path.expanduser(path_str))
        norm = ap.replace("\\", "/").rstrip("/")
        if norm not in seen:
            seen.add(norm)
            found.append(ap)

    # 1. Explicit studio / user environment variables
    pkg_dir_env = os.environ.get("HOUDINI_PACKAGE_DIR", "").strip()
    if pkg_dir_env:
        for part in pkg_dir_env.split(os.pathsep):
            part = part.strip()
            if part and not part.startswith("&"):
                parent = os.path.dirname(os.path.abspath(os.path.expanduser(part.rstrip("/\\"))))
                _add(parent)

    pref_env = os.environ.get("HOUDINI_USER_PREF_DIR", "").strip()
    if pref_env and "__HVER__" not in pref_env:
        _add(pref_env)

    home = os.path.expanduser("~")
    ver_re = re.compile(r"^\.?houdini(20|21|22)\.\d+$", re.IGNORECASE)

    # 2. Scan standard home & Documents locations
    search_roots = [home, os.path.join(home, "Documents")]
    if platform.system() == "Darwin":
        search_roots.append(os.path.join(home, "Library", "Preferences", "houdini"))

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        try:
            for entry in sorted(os.listdir(root)):
                full = os.path.join(root, entry)
                if os.path.isdir(full) and ver_re.match(entry):
                    _add(full)
        except OSError:
            pass

    # 3. Rocky 8 / Rocky 9 fresh workstation fallback:
    # If no ~/houdiniX.Y exists yet, detect installed /opt/hfsX.Y versions and create ~/houdiniX.Y
    if not found and platform.system() == "Linux":
        hfs_Matches = sorted(glob.glob("/opt/hfs*"))
        vers = []
        for hfs in hfs_Matches:
            m = re.search(r"hfs((?:20|21|22)\.\d+)", os.path.basename(hfs))
            if m and m.group(1) not in vers:
                vers.append(m.group(1))
        if not vers:
            vers = ["20.5", "21.0"]
        for v in vers:
            _add(os.path.join(home, "houdini" + v))

    return found


def _ensure_executable_scripts(repo_root):
    """Ensure .sh scripts keep chmod 0755 on Linux/macOS even after zip extraction."""
    for name in ("install.sh", "uninstall.sh", "run_tests.sh"):
        sp = os.path.join(repo_root, name)
        if os.path.isfile(sp):
            try:
                st = os.stat(sp)
                os.chmod(sp, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except OSError:
                pass


def install_package(repo_root, target_dir=None):
    repo_norm = _norm_path(repo_root)
    _ensure_executable_scripts(repo_root)

    # Also update the repo's template houdini/packages/hdri_maker.json with the real repo path
    repo_pkg_path = os.path.join(repo_root, "houdini", "packages", "hdri_maker.json")
    pkg_payload = {
        "env": [
            {"HDRI_MAKER_ROOT": repo_norm},
            {"PYTHONPATH": "${HDRI_MAKER_ROOT}/python"},
            {"HOUDINI_PATH": "${HDRI_MAKER_ROOT}/houdini"},
        ]
    }
    try:
        os.makedirs(os.path.dirname(repo_pkg_path), exist_ok=True)
        with open(repo_pkg_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(pkg_payload, f, indent=4)
            f.write("\n")
    except OSError:
        pass

    prefs = [os.path.abspath(os.path.expanduser(target_dir))] if target_dir else _discover_houdini_pref_dirs()
    if not prefs:
        print("[WARNING] Could not find any Houdini 20+ preference directories.")
        print("Manually copy: " + repo_pkg_path + " -> <HOUDINI_USER_PREF_DIR>/packages/hdri_maker.json")
        return 1

    installed = []
    for pref in prefs:
        pkg_dir = pref if os.path.basename(pref.rstrip("/\\")) == "packages" else os.path.join(pref, "packages")
        try:
            os.makedirs(pkg_dir, exist_ok=True)
            # Remove any duplicate legacy filename so Houdini loads the package once
            legacy_alt = os.path.join(pkg_dir, "hdri_breaker.json")
            if os.path.isfile(legacy_alt):
                try:
                    os.remove(legacy_alt)
                except OSError:
                    pass
            dest_file = os.path.join(pkg_dir, "hdri_maker.json")
            with open(dest_file, "w", encoding="utf-8", newline="\n") as f:
                json.dump(pkg_payload, f, indent=4)
                f.write("\n")
            installed.append(dest_file)
            print("[OK] Installed HDRI Breaker package -> " + dest_file)
        except OSError as exc:
            print("[ERROR] Failed to write package to " + pkg_dir + ": " + str(exc))

    if not installed:
        return 1

    print("")
    print("HDRI Breaker installation complete (" + str(len(installed)) + " target(s)).")
    print("Root path registered: " + repo_norm)
    print("Restart Houdini and open the 'HDRI Breaker' Shelf or 'HDRI Breaker (Solaris)' Python Panel.")
    return 0


def uninstall_package():
    prefs = _discover_houdini_pref_dirs()
    removed = 0
    for pref in prefs:
        pkg_dir = os.path.join(pref, "packages")
        for fname in ("hdri_maker.json", "hdri_breaker.json"):
            fpath = os.path.join(pkg_dir, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                    print("[OK] Removed: " + fpath)
                    removed += 1
                except OSError as exc:
                    print("[ERROR] Could not remove " + fpath + ": " + str(exc))
    if removed == 0:
        print("[INFO] No installed HDRI Breaker package files found to remove.")
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "--uninstall" in args or "-u" in args:
        return uninstall_package()
    target = None
    for i, a in enumerate(args):
        if a in ("--target", "-t") and i + 1 < len(args):
            target = args[i + 1]
    return install_package(repo_root, target_dir=target)


if __name__ == "__main__":
    sys.exit(main())
