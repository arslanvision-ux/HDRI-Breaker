"""
HDRI Maker for Houdini 20+ Solaris (LOPs) Pipeline.

Extracts lighting data (hotspots, tone, correlated color temperature,
hemispherical gradients, dynamic range) from real single plates or image
sequences (JPG, PNG, EXR, HDR, TIF) and synthesizes true 32-bit float
Lat/Long (equirectangular) EXR maps wired directly into Solaris /stage.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*smallest subnormal.*",
    category=UserWarning,
)

__version__ = "1.0.0"
__all__ = [
    "plate_io",
    "lighting_analyzer",
    "hdri_generator",
    "solaris_stage",
    "ui_panel",
    "reload_all",
]


def reload_all(relaunch: bool = True):
    """
    Hot-reload all hdri_maker modules inside a running Houdini session
    without restarting Houdini.
    """
    import importlib
    import sys

    # Close existing floating window if open
    ui_mod = sys.modules.get("hdri_maker.ui_panel")
    if ui_mod is not None:
        win = getattr(ui_mod, "_FLOATING_WINDOW", None)
        if win is not None:
            try:
                win.close()
                win.deleteLater()
            except Exception:
                pass
            ui_mod._FLOATING_WINDOW = None

    # Remove all hdri_maker* modules so fresh code is imported cleanly
    to_remove = [name for name in list(sys.modules) if name == "hdri_maker" or name.startswith("hdri_maker.")]
    for name in to_remove:
        sys.modules.pop(name, None)

    importlib.invalidate_caches()
    pkg = importlib.import_module("hdri_maker")
    if relaunch:
        ui_panel = importlib.import_module("hdri_maker.ui_panel")
        return ui_panel.launch_hdri_maker()
    return pkg
