"""
Qt compatibility layer for Houdini 20.0/20.5 (PySide2) and Houdini 21.0+ (PySide6).
"""

QT_BINDING = None

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, Signal, Slot
    QT_BINDING = "PySide6"
except ImportError:
    try:
        from PySide2 import QtCore, QtGui, QtWidgets
        from PySide2.QtCore import Qt, Signal, Slot
        QT_BINDING = "PySide2"
    except ImportError as exc:
        raise ImportError(
            "Neither PySide6 (Houdini 21+) nor PySide2 (Houdini 20.x) is available."
        ) from exc


def exec_dialog(dialog):
    """Execute a QDialog compatibly across PySide2 and PySide6."""
    if hasattr(dialog, "exec"):
        return dialog.exec()
    return dialog.exec_()
