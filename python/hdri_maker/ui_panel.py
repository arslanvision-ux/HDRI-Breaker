"""
Houdini 20+ Solaris HDRI Maker Qt UI (PySide2 & PySide6 compatible).

Can be docked as a Houdini Python Panel inside the Solaris desktop or
launched as a floating window from a shelf tool.
"""

from __future__ import annotations

import datetime
import json
import os
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from .hdri_generator import (
    ExtractedAreaLightInfo,
    HDRISynthesisParams,
    apply_brush_dab,
    apply_brush_strokes,
    apply_post_adjustments_to_hdri,
    bake_planar_ground_texture,
    compute_neutral_wb_gains,
    export_hdri_sequence,
    extract_and_inpaint_hotspot,
    extract_and_inpaint_multiple_hotspots,
    generate_diffuse_irradiance_hdri,
    generate_latlong_hdri,
    render_preview_srgb,
)
from .lighting_analyzer import (
    LightingProfile,
    PlateViewInput,
    analyze_multiview_plates,
    analyze_plate,
    auto_align_adjacent_plates,
    auto_equalize_plate_exposures,
    compute_overlap_difference_map,
    compute_plate_outline_uv,
    equirect_rect_to_plate_rect,
    estimate_horizon_roll_deg,
    estimate_horizon_shift_deg,
    estimate_radial_vignette,
    hit_test_plate_uv,
    merge_lighting_profiles,
    rgb_to_cct_kelvin,
)
from .plate_io import (
    linear_to_srgb,
    merge_ldr_brackets_to_hdr,
    read_plate,
    read_plate_metadata,
    resolve_plate_sequence,
    write_exr_32f,
)
from .qt_compat import Qt, QtCore, QtGui, QtWidgets, Signal
from .solaris_stage import (
    HAS_HOU,
    create_or_update_area_light_from_hotspot,
    create_or_update_distant_sun_from_hotspot,
    create_or_update_domelight,
    create_or_update_lookdev_and_ground,
    export_solaris_rig_usd,
    list_scene_cameras,
    read_scene_camera_params,
)

try:
    import hou
except ImportError:
    hou = None


SESSION_STATE_PATH = os.path.join(os.path.expanduser("~"), ".hdri_maker_last_session.json")
DEBUG_LOG_PATH = os.path.join(os.path.expanduser("~"), ".hdri_maker_debug.log")

UI_THEMES: Dict[str, Dict[str, str]] = {
    "Breaker Dark (Amber)": {
        "bg": "#1b1d22",
        "panel_bg": "#22252c",
        "fg": "#f0f4f8",
        "btn_bg": "#2e3440",
        "btn_border": "#525c70",
        "btn_hover": "#3b4252",
        "accent": "#f59e0b",
        "accent_fg": "#111318",
        "input_bg": "#15171c",
        "status_fg": "#fbbf24",
    },
    "Solaris Slate (Cyan)": {
        "bg": "#171b22",
        "panel_bg": "#1e242e",
        "fg": "#e8f0f8",
        "btn_bg": "#283344",
        "btn_border": "#4a607a",
        "btn_hover": "#334258",
        "accent": "#38bdf8",
        "accent_fg": "#0f172a",
        "input_bg": "#12161c",
        "status_fg": "#7dd3fc",
    },
    "High-Contrast Carbon": {
        "bg": "#101215",
        "panel_bg": "#171a1f",
        "fg": "#ffffff",
        "btn_bg": "#2b303b",
        "btn_border": "#8894a8",
        "btn_hover": "#3c4454",
        "accent": "#22c55e",
        "accent_fg": "#09100c",
        "input_bg": "#0b0d10",
        "status_fg": "#4ade80",
    },
    "Cobalt Pro": {
        "bg": "#161925",
        "panel_bg": "#1d2232",
        "fg": "#eef2ff",
        "btn_bg": "#2a324b",
        "btn_border": "#55648f",
        "btn_hover": "#374263",
        "accent": "#60a5fa",
        "accent_fg": "#0f172a",
        "input_bg": "#111420",
        "status_fg": "#93c5fd",
    },
    "Daylight Studio (Light)": {
        "bg": "#e5e7eb",
        "panel_bg": "#f3f4f6",
        "fg": "#111827",
        "btn_bg": "#ffffff",
        "btn_border": "#6b7280",
        "btn_hover": "#dbeafe",
        "accent": "#2563eb",
        "accent_fg": "#ffffff",
        "input_bg": "#ffffff",
        "status_fg": "#1d4ed8",
    },
}


class SliderResetControl(QtWidgets.QWidget):
    """Compact horizontal slider + numeric readout + reset button."""

    valueChanged = Signal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        default: float,
        step: float = 0.05,
        decimals: int = 2,
        parent=None,
    ):
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._default = float(default)
        self._step = max(1e-6, float(step))
        self._n_steps = max(1, int(round((self._max - self._min) / self._step)))
        self._syncing = False

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._n_steps)
        layout.addWidget(self.slider, stretch=1)

        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(self._min, self._max)
        self.spin.setSingleStep(self._step)
        self.spin.setDecimals(decimals)
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spin.setFixedWidth(56)
        layout.addWidget(self.spin)

        self.reset_btn = QtWidgets.QToolButton()
        self.reset_btn.setText("R")
        self.reset_btn.setFixedSize(22, 20)
        self.reset_btn.setToolTip(f"Reset slider to default ({default:.{decimals}f})")
        layout.addWidget(self.reset_btn)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.reset_btn.clicked.connect(self.reset)

        self.setValue(self._default)

    def _to_slider(self, val: float) -> int:
        t = (float(val) - self._min) / max(self._max - self._min, 1e-9)
        return int(round(max(0.0, min(1.0, t)) * self._n_steps))

    def _from_slider(self, pos: int) -> float:
        t = pos / float(self._n_steps)
        return self._min + t * (self._max - self._min)

    def _on_slider_changed(self, pos: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        val = self._from_slider(pos)
        self.spin.setValue(val)
        self._syncing = False
        self.valueChanged.emit(self.spin.value())

    def _on_spin_changed(self, val: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.slider.setValue(self._to_slider(val))
        self._syncing = False
        self.valueChanged.emit(float(val))

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, val: float) -> None:
        clamped = max(self._min, min(self._max, float(val)))
        self._syncing = True
        self.spin.setValue(clamped)
        self.slider.setValue(self._to_slider(clamped))
        self._syncing = False
        self.valueChanged.emit(self.spin.value())

    def reset(self) -> None:
        self.setValue(self._default)


class PreviewLabel(QtWidgets.QLabel):
    """QLabel supporting interactive click-and-drag region selection, zoom/pan, direct move/resize handles, and pixel inspection."""

    rectSelected = Signal(str, tuple)
    regionEdited = Signal(str, int, tuple)
    pixelHovered = Signal(float, float)
    zoomChanged = Signal(float)
    plateSelected = Signal(int)
    plateDragMoved = Signal(int, float, float, str)
    plateDragFinished = Signal(int)
    plateWheelScaled = Signal(int, float)
    brushStrokeStarted = Signal(float, float)
    brushStrokeMoved = Signal(float, float)
    brushStrokeFinished = Signal()
    brushCloneSourcePicked = Signal(float, float)
    brushParamAdjusted = Signal(str, float)

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setMouseTracking(True)
        self._pick_mode: Optional[str] = None  # None, "sky", "ground", "light", or "wb"
        self._pick_shape: str = "rect"  # "rect" or "circle"
        self._drag_start: Optional[QtCore.QPoint] = None
        self._drag_end: Optional[QtCore.QPoint] = None
        self._sky_rect: Optional[tuple] = None
        self._ground_rect: Optional[tuple] = None
        self._light_regions: list = []
        # Direct region manipulation state when _pick_mode is None
        self._edit_kind: Optional[str] = None  # "sky", "ground", "light"
        self._edit_idx: int = -1
        self._edit_mode: Optional[str] = None  # "move" or "resize"
        self._edit_edges: Tuple[bool, bool, bool, bool] = (False, False, False, False)
        self._edit_start_uv: Tuple[float, float] = (0.0, 0.0)
        self._edit_orig_rect: Optional[tuple] = None
        # PTGui-style interactive plate alignment state
        self._plate_edit_mode: bool = False
        self._plate_views: list = []
        self._selected_plate_idx: int = -1
        self._plate_az_offset: float = 0.0
        self._plate_v_shift: float = 0.0
        self._plate_drag_mode: Optional[str] = None  # "move", "roll", "scale", "warp"
        self._plate_drag_last_uv: Tuple[float, float] = (0.0, 0.0)
        # Zoom & Pan state
        self._source_pixmap: Optional[QtGui.QPixmap] = None
        self._zoom: float = 1.0
        self._pan_uv: List[float] = [0.5, 0.5]
        self._panning: bool = False
        self._pan_start_pos: Optional[QtCore.QPoint] = None
        self._pan_start_uv: Tuple[float, float] = (0.5, 0.5)
        # A/B Split Wipe state
        self._ab_split_enabled: bool = False
        self._ab_split_x: float = 0.5
        self._ab_compare_pixmap: Optional[QtGui.QPixmap] = None
        self._ab_dragging: bool = False
        # Interactive Canvas Brush state
        self._brush_enabled: bool = False
        self._brush_mode: str = "heal"
        self._brush_radius_u: float = 0.025
        self._brush_softness: float = 0.65
        self._brush_hover_uv: Optional[Tuple[float, float]] = None
        self._brush_clone_src_uv: Optional[Tuple[float, float]] = None
        self._brush_painting: bool = False
        self._brush_adjust_mode: Optional[str] = None
        self._brush_adjust_start_pos: Optional[QtCore.QPoint] = None
        self._brush_adjust_last_x: float = 0.0
        self._brush_adjust_moved: bool = False
        self._brush_hud_text: str = ""

    def set_brush_state(
        self,
        enabled: bool,
        mode: str = "heal",
        radius_u: float = 0.025,
        softness: float = 0.65,
        clone_src_uv: Optional[Tuple[float, float]] = None,
        hud_text: str = "",
    ) -> None:
        self._brush_enabled = bool(enabled)
        self._brush_mode = str(mode)
        self._brush_radius_u = float(radius_u)
        self._brush_softness = float(softness)
        self._brush_clone_src_uv = clone_src_uv
        if hud_text:
            self._brush_hud_text = hud_text
        if self._brush_enabled:
            self.setCursor(Qt.CrossCursor if self._brush_adjust_mode is None else Qt.SizeHorCursor)
        elif self._pick_mode is None:
            self.setCursor(Qt.OpenHandCursor if self._plate_edit_mode else Qt.ArrowCursor)
        self.update()

    def set_ab_split(self, enabled: bool, compare_pix: Optional[QtGui.QPixmap] = None) -> None:
        self._ab_split_enabled = bool(enabled)
        if compare_pix is not None:
            self._ab_compare_pixmap = compare_pix
        self.update()

    def set_source_pixmap(self, pix: Optional[QtGui.QPixmap]) -> None:
        self._source_pixmap = pix
        if pix is not None and not pix.isNull():
            self.setText("")
        self.update()

    def zoom_factor(self) -> float:
        return self._zoom

    def set_zoom(self, zoom: float, anchor_pos: Optional[QtCore.QPoint] = None) -> None:
        new_zoom = float(np.clip(zoom, 1.0, 12.0))
        if abs(new_zoom - self._zoom) < 1e-4:
            return
        if anchor_pos is not None and self._source_pixmap is not None:
            uv_before = self._pos_to_uv(anchor_pos)
            self._zoom = new_zoom
            if new_zoom <= 1.001:
                self._zoom = 1.0
                self._pan_uv = [0.5, 0.5]
            elif uv_before is not None:
                fit_w, fit_h = self._fit_size()
                zw = max(1.0, fit_w * self._zoom)
                zh = max(1.0, fit_h * self._zoom)
                pan_u = uv_before[0] - (anchor_pos.x() - self.width() * 0.5) / zw
                pan_v = uv_before[1] - (anchor_pos.y() - self.height() * 0.5) / zh
                self._pan_uv = [float(np.clip(pan_u, 0.0, 1.0)), float(np.clip(pan_v, 0.0, 1.0))]
        else:
            self._zoom = new_zoom
            if new_zoom <= 1.001:
                self._zoom = 1.0
                self._pan_uv = [0.5, 0.5]
        self.zoomChanged.emit(self._zoom)
        self.update()

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan_uv = [0.5, 0.5]
        self.zoomChanged.emit(self._zoom)
        self.update()

    def _fit_size(self) -> Tuple[float, float]:
        pix = self._source_pixmap
        if pix is None or pix.isNull():
            return float(max(1, self.width())), float(max(1, self.height()))
        pw, ph = float(pix.width()), float(pix.height())
        lw, lh = float(max(1, self.width())), float(max(1, self.height()))
        scale = min(lw / max(1.0, pw), lh / max(1.0, ph))
        return pw * scale, ph * scale

    def set_regions(self, sky_rect=None, ground_rect=None, light_regions=None) -> None:
        self._sky_rect = sky_rect
        self._ground_rect = ground_rect
        self._light_regions = list(light_regions or [])
        self.update()

    def set_pick_mode(self, mode: Optional[str], shape: str = "rect") -> None:
        self._pick_mode = mode
        self._pick_shape = shape
        self.setCursor(Qt.CrossCursor if mode else (Qt.OpenHandCursor if self._plate_edit_mode else Qt.ArrowCursor))

    def set_plate_edit_mode(self, enabled: bool) -> None:
        self._plate_edit_mode = bool(enabled)
        if not enabled:
            self._plate_drag_mode = None
        if self._pick_mode is None:
            self.setCursor(Qt.OpenHandCursor if self._plate_edit_mode else Qt.ArrowCursor)
        self.update()

    def set_plate_overlay(
        self,
        views: list,
        selected_idx: int = -1,
        az_offset: float = 0.0,
        v_shift: float = 0.0,
    ) -> None:
        self._plate_views = list(views or [])
        self._selected_plate_idx = int(selected_idx)
        self._plate_az_offset = float(az_offset)
        self._plate_v_shift = float(v_shift)
        self.update()

    def _hit_test_plate_handles(self, u: float, v: float) -> Optional[str]:
        """Hit-test the selected plate's roll or corner scale handles."""
        if not (0 <= self._selected_plate_idx < len(self._plate_views)):
            return None
        info = compute_plate_outline_uv(
            self._plate_views[self._selected_plate_idx],
            azimuth_offset_deg=self._plate_az_offset,
            vertical_shift_deg=self._plate_v_shift,
            samples_per_edge=8,
        )
        tol_u, tol_v = 0.016 / max(1.0, self._zoom * 0.5), 0.028 / max(1.0, self._zoom * 0.5)
        ru, rv = info["roll_handle"]
        if abs(u - ru) <= tol_u and abs(v - rv) <= tol_v:
            return "roll"
        for cu, cv in info["corners"]:
            if abs(u - cu) <= tol_u and abs(v - cv) <= tol_v:
                return "scale"
        return None

    def _pixmap_rect(self) -> QtCore.QRect:
        pix = self._source_pixmap
        if pix is None or pix.isNull():
            return self.rect()
        fit_w, fit_h = self._fit_size()
        zw = fit_w * self._zoom
        zh = fit_h * self._zoom
        x0 = int(round(self.width() * 0.5 - self._pan_uv[0] * zw))
        y0 = int(round(self.height() * 0.5 - self._pan_uv[1] * zh))
        return QtCore.QRect(x0, y0, max(1, int(round(zw))), max(1, int(round(zh))))

    def _pos_to_uv(self, pos: QtCore.QPoint) -> Optional[Tuple[float, float]]:
        prect = self._pixmap_rect()
        if prect.width() <= 4 or prect.height() <= 4:
            return None
        u = (pos.x() - prect.x()) / float(prect.width())
        v = (pos.y() - prect.y()) / float(prect.height())
        return (u, v)

    def _hit_test_region(self, u: float, v: float):
        """Return (kind, idx, mode, edges, rect) if (u, v) hits an existing region."""
        tol_u = 0.018
        tol_v = 0.025
        candidates = []
        for idx in range(len(self._light_regions) - 1, -1, -1):
            r = self._light_regions[idx].get("rect")
            if r is not None:
                candidates.append(("light", idx, r))
        if self._sky_rect is not None:
            candidates.append(("sky", 0, self._sky_rect))
        if self._ground_rect is not None:
            candidates.append(("ground", 0, self._ground_rect))

        for kind, idx, rect in candidates:
            u0, v0, u1, v1 = rect
            lo_u, hi_u = min(u0, u1), max(u0, u1)
            lo_v, hi_v = min(v0, v1), max(v0, v1)
            if (lo_u - tol_u) <= u <= (hi_u + tol_u) and (lo_v - tol_v) <= v <= (hi_v + tol_v):
                near_l = abs(u - lo_u) <= tol_u
                near_r = abs(u - hi_u) <= tol_u
                near_t = abs(v - lo_v) <= tol_v
                near_b = abs(v - hi_v) <= tol_v
                if near_l or near_r or near_t or near_b:
                    return (kind, idx, "resize", (near_l, near_t, near_r, near_b), (lo_u, lo_v, hi_u, hi_v))
                return (kind, idx, "move", (False, False, False, False), (lo_u, lo_v, hi_u, hi_v))
        return None

    def wheelEvent(self, event) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta != 0:
            if (
                self._plate_edit_mode
                and self._pick_mode is None
                and (event.modifiers() & Qt.ShiftModifier)
                and 0 <= self._selected_plate_idx < len(self._plate_views)
            ):
                self.plateWheelScaled.emit(self._selected_plate_idx, float(delta))
                event.accept()
                return
            factor = 1.18 if delta > 0 else (1.0 / 1.18)
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            self.set_zoom(self._zoom * factor, anchor_pos=pos)
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._pick_mode is None and event.button() == Qt.LeftButton and self._zoom > 1.001:
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        is_alt = bool(event.modifiers() & Qt.AltModifier)
        if self._brush_enabled and self._pick_mode is None:
            mods = event.modifiers()
            if (mods & Qt.ControlModifier) and event.button() == Qt.LeftButton:
                self._brush_adjust_mode = "size"
            elif (mods & Qt.ControlModifier) and event.button() == Qt.RightButton:
                self._brush_adjust_mode = "strength"
            elif (mods & Qt.ShiftModifier) and event.button() == Qt.LeftButton:
                self._brush_adjust_mode = "soft"
            if self._brush_adjust_mode is not None:
                self._brush_adjust_start_pos = pos
                self._brush_adjust_last_x = float(pos.x())
                self._brush_adjust_moved = False
                uv_h = self._pos_to_uv(pos)
                if uv_h is not None:
                    self._brush_hover_uv = (float(uv_h[0]), float(uv_h[1]))
                self.setCursor(Qt.SizeHorCursor)
                event.accept()
                return
        if event.button() in (Qt.MiddleButton, Qt.RightButton) or (is_alt and event.button() == Qt.LeftButton):
            if self._zoom > 1.001:
                self._panning = True
                self._pan_start_pos = pos
                self._pan_start_uv = (self._pan_uv[0], self._pan_uv[1])
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
        if self._pick_mode and event.button() == Qt.LeftButton:
            self._drag_start = pos
            self._drag_end = pos
            self.update()
            return
        if self._pick_mode is None and event.button() == Qt.LeftButton:
            if self._brush_enabled:
                uv_b = self._pos_to_uv(pos)
                if uv_b is not None and 0.0 <= uv_b[0] <= 1.0 and 0.0 <= uv_b[1] <= 1.0:
                    self._brush_painting = True
                    self.brushStrokeStarted.emit(float(uv_b[0]), float(uv_b[1]))
                    return
            if self._ab_split_enabled and self._ab_compare_pixmap is not None:
                prect = self._pixmap_rect()
                split_px = prect.x() + int(self._ab_split_x * prect.width())
                if abs(pos.x() - split_px) <= 12:
                    self._ab_dragging = True
                    self.setCursor(Qt.SplitHCursor)
                    return
            uv = self._pos_to_uv(pos)
            if self._plate_edit_mode and uv is not None and self._plate_views:
                handle_mode = self._hit_test_plate_handles(uv[0], uv[1])
                if handle_mode is not None:
                    self._plate_drag_mode = handle_mode
                    self._plate_drag_last_uv = (uv[0], uv[1])
                    self.setCursor(Qt.ClosedHandCursor)
                    return
                hit_plate = hit_test_plate_uv(
                    self._plate_views,
                    uv[0],
                    uv[1],
                    azimuth_offset_deg=self._plate_az_offset,
                    vertical_shift_deg=self._plate_v_shift,
                )
                if hit_plate is not None:
                    if hit_plate != self._selected_plate_idx:
                        self._selected_plate_idx = hit_plate
                        self.plateSelected.emit(hit_plate)
                    mods = event.modifiers()
                    if mods & Qt.ShiftModifier:
                        self._plate_drag_mode = "roll"
                    elif mods & Qt.ControlModifier:
                        self._plate_drag_mode = "warp"
                    else:
                        self._plate_drag_mode = "move"
                    self._plate_drag_last_uv = (uv[0], uv[1])
                    self.setCursor(Qt.ClosedHandCursor)
                    self.update()
                    return
            if uv is not None and not self._plate_edit_mode:
                hit = self._hit_test_region(uv[0], uv[1])
                if hit is not None:
                    self._edit_kind, self._edit_idx, self._edit_mode, self._edit_edges, self._edit_orig_rect = hit
                    self._edit_start_uv = (
                        max(0.0, min(1.0, uv[0])),
                        max(0.0, min(1.0, uv[1])),
                    )
                    return
            if self._zoom > 1.001:
                self._panning = True
                self._pan_start_pos = pos
                self._pan_start_uv = (self._pan_uv[0], self._pan_uv[1])
                self.setCursor(Qt.ClosedHandCursor)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        if self._brush_adjust_mode is not None:
            dx = float(pos.x() - self._brush_adjust_last_x)
            self._brush_adjust_last_x = float(pos.x())
            if self._brush_adjust_start_pos is not None and abs(pos.x() - self._brush_adjust_start_pos.x()) >= 4:
                self._brush_adjust_moved = True
            if abs(dx) > 0.0:
                self.brushParamAdjusted.emit(self._brush_adjust_mode, dx)
            self.update()
            return
        if self._panning and self._pan_start_pos is not None:
            fit_w, fit_h = self._fit_size()
            zw = max(1.0, fit_w * self._zoom)
            zh = max(1.0, fit_h * self._zoom)
            dx = pos.x() - self._pan_start_pos.x()
            dy = pos.y() - self._pan_start_pos.y()
            self._pan_uv = [
                float(np.clip(self._pan_start_uv[0] - dx / zw, 0.0, 1.0)),
                float(np.clip(self._pan_start_uv[1] - dy / zh, 0.0, 1.0)),
            ]
            self.update()
            return

        if self._ab_dragging:
            prect = self._pixmap_rect()
            if prect.width() > 2:
                self._ab_split_x = float(np.clip((pos.x() - prect.x()) / float(prect.width()), 0.02, 0.98))
                self.update()
            return

        uv = self._pos_to_uv(pos)
        if uv is not None and 0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0:
            self.pixelHovered.emit(float(uv[0]), float(uv[1]))
            if self._brush_enabled:
                self._brush_hover_uv = (float(uv[0]), float(uv[1]))
                if self._brush_painting:
                    self.brushStrokeMoved.emit(float(uv[0]), float(uv[1]))
                self.update()
                if self._brush_painting:
                    return

        if self._pick_mode and self._drag_start is not None:
            self._drag_end = pos
            self.update()
            return

        if self._plate_drag_mode is not None and uv is not None and 0 <= self._selected_plate_idx < len(self._plate_views):
            du = float(uv[0] - self._plate_drag_last_uv[0])
            dv = float(uv[1] - self._plate_drag_last_uv[1])
            self._plate_drag_last_uv = (uv[0], uv[1])
            if abs(du) > 1e-6 or abs(dv) > 1e-6:
                self.plateDragMoved.emit(self._selected_plate_idx, du, dv, self._plate_drag_mode)
            return

        if self._edit_kind is not None and self._edit_orig_rect is not None and uv is not None:
            cu = max(0.0, min(1.0, uv[0]))
            cv = max(0.0, min(1.0, uv[1]))
            du = cu - self._edit_start_uv[0]
            dv = cv - self._edit_start_uv[1]
            u0, v0, u1, v1 = self._edit_orig_rect
            if self._edit_mode == "move":
                w = u1 - u0
                h = v1 - v0
                nu0 = float(np.clip(u0 + du, 0.0, max(0.0, 1.0 - w)))
                nv0 = float(np.clip(v0 + dv, 0.0, max(0.0, 1.0 - h)))
                new_r = (nu0, nv0, nu0 + w, nv0 + h)
            else:
                nl, nt, nr, nb = self._edit_edges
                nu0 = float(np.clip(u0 + du, 0.0, u1 - 0.015)) if nl else u0
                nu1 = float(np.clip(u1 + du, nu0 + 0.015, 1.0)) if nr else u1
                nv0 = float(np.clip(v0 + dv, 0.0, v1 - 0.015)) if nt else v0
                nv1 = float(np.clip(v1 + dv, nv0 + 0.015, 1.0)) if nb else v1
                new_r = (nu0, nv0, nu1, nv1)
            self.regionEdited.emit(self._edit_kind, self._edit_idx, new_r)
            return

        if self._pick_mode is None and uv is not None:
            if self._plate_edit_mode and self._plate_views:
                handle_mode = self._hit_test_plate_handles(uv[0], uv[1])
                if handle_mode == "roll":
                    self.setCursor(Qt.SizeHorCursor)
                elif handle_mode == "scale":
                    self.setCursor(Qt.SizeFDiagCursor)
                else:
                    hit_p = hit_test_plate_uv(
                        self._plate_views,
                        uv[0],
                        uv[1],
                        azimuth_offset_deg=self._plate_az_offset,
                        vertical_shift_deg=self._plate_v_shift,
                    )
                    self.setCursor(Qt.OpenHandCursor if hit_p is not None else Qt.ArrowCursor)
                super().mouseMoveEvent(event)
                return
            hit = self._hit_test_region(uv[0], uv[1])
            if hit is not None:
                _, _, mode, edges, _ = hit
                if mode == "move":
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    nl, nt, nr, nb = edges
                    if (nl or nr) and not (nt or nb):
                        self.setCursor(Qt.SizeHorCursor)
                    elif (nt or nb) and not (nl or nr):
                        self.setCursor(Qt.SizeVerCursor)
                    else:
                        self.setCursor(Qt.SizeFDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._brush_adjust_mode is not None:
            mode_done = self._brush_adjust_mode
            moved = self._brush_adjust_moved
            self._brush_adjust_mode = None
            self._brush_adjust_start_pos = None
            self._brush_hud_text = ""
            self.setCursor(Qt.CrossCursor if self._brush_enabled else Qt.ArrowCursor)
            if mode_done == "size" and (not moved) and self._brush_mode == "clone":
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                uv_b = self._pos_to_uv(pos)
                if uv_b is not None and 0.0 <= uv_b[0] <= 1.0 and 0.0 <= uv_b[1] <= 1.0:
                    self._brush_clone_src_uv = (float(uv_b[0]), float(uv_b[1]))
                    self.brushCloneSourcePicked.emit(float(uv_b[0]), float(uv_b[1]))
            self.update()
            return
        if self._brush_painting:
            self._brush_painting = False
            self.brushStrokeFinished.emit()
            return
        if self._ab_dragging:
            self._ab_dragging = False
            self.setCursor(Qt.ArrowCursor)
            return
        if self._panning:
            self._panning = False
            self._pan_start_pos = None
            self.setCursor(Qt.CrossCursor if self._pick_mode else Qt.ArrowCursor)
            return
        if self._plate_drag_mode is not None:
            idx = self._selected_plate_idx
            self._plate_drag_mode = None
            self.setCursor(Qt.OpenHandCursor if self._plate_edit_mode else Qt.ArrowCursor)
            if idx >= 0:
                self.plateDragFinished.emit(idx)
            return
        if self._edit_kind is not None:
            self._edit_kind = None
            self._edit_idx = -1
            self._edit_mode = None
            self._edit_orig_rect = None
            return
        if self._pick_mode and self._drag_start is not None:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            self._drag_end = pos
            prect = self._pixmap_rect()
            if prect.width() > 4 and prect.height() > 4:
                x0 = min(self._drag_start.x(), self._drag_end.x())
                x1 = max(self._drag_start.x(), self._drag_end.x())
                y0 = min(self._drag_start.y(), self._drag_end.y())
                y1 = max(self._drag_start.y(), self._drag_end.y())
                u0 = max(0.0, min(1.0, (x0 - prect.x()) / float(prect.width())))
                u1 = max(0.0, min(1.0, (x1 - prect.x()) / float(prect.width())))
                v0 = max(0.0, min(1.0, (y0 - prect.y()) / float(prect.height())))
                v1 = max(0.0, min(1.0, (y1 - prect.y()) / float(prect.height())))
                if (u1 - u0) > 0.01 and (v1 - v0) > 0.01:
                    self.rectSelected.emit(self._pick_mode, (u0, v0, u1, v1))
            self._drag_start = None
            self._drag_end = None
            self.update()
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        if self._source_pixmap is not None and not self._source_pixmap.isNull():
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, self._zoom <= 4.0)
            target = self._pixmap_rect()
            painter.drawPixmap(target, self._source_pixmap, self._source_pixmap.rect())
            if self._ab_split_enabled and self._ab_compare_pixmap is not None and not self._ab_compare_pixmap.isNull():
                split_w = int(round(self._ab_split_x * target.width()))
                if split_w > 1:
                    src_w = int(round(self._ab_split_x * self._ab_compare_pixmap.width()))
                    left_target = QtCore.QRect(target.x(), target.y(), split_w, target.height())
                    left_src = QtCore.QRect(0, 0, src_w, self._ab_compare_pixmap.height())
                    painter.drawPixmap(left_target, self._ab_compare_pixmap, left_src)
                sx = target.x() + split_w
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 225, 255), 2, Qt.SolidLine))
                painter.drawLine(sx, target.y(), sx, target.y() + target.height())
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
                painter.drawText(target.x() + 8, target.y() + 18, "A: RAW PLATE")
                painter.drawText(min(target.right() - 95, sx + 8), target.y() + 18, "B: SYNTH HDRI")
            if self._plate_edit_mode and self._plate_views:
                for idx, view in enumerate(self._plate_views):
                    is_sel = idx == self._selected_plate_idx
                    info = compute_plate_outline_uv(
                        view,
                        azimuth_offset_deg=self._plate_az_offset,
                        vertical_shift_deg=self._plate_v_shift,
                        samples_per_edge=14 if is_sel else 8,
                    )
                    if is_sel:
                        pen = QtGui.QPen(QtGui.QColor(0, 225, 255), 2, Qt.SolidLine)
                    else:
                        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 110), 1, Qt.DashLine)
                    painter.setPen(pen)
                    for poly in info["polylines"]:
                        pts = [
                            QtCore.QPointF(target.x() + u * target.width(), target.y() + v * target.height())
                            for u, v in poly
                        ]
                        painter.drawPolyline(QtGui.QPolygonF(pts))
                    if is_sel:
                        cu, cv = info["center"]
                        cx = target.x() + cu * target.width()
                        cy = target.y() + cv * target.height()
                        painter.drawLine(QtCore.QPointF(cx - 7, cy), QtCore.QPointF(cx + 7, cy))
                        painter.drawLine(QtCore.QPointF(cx, cy - 7), QtCore.QPointF(cx, cy + 7))
                        painter.drawText(
                            int(cx) + 8,
                            int(cy) - 6,
                            f"Plate #{idx + 1} (Az {view.cam_azimuth_deg:.1f}°, El {view.cam_elevation_deg:.1f}°, Roll {view.cam_roll_deg:.1f}°)",
                        )
                        # Corner scale handles
                        painter.setBrush(QtGui.QColor(0, 225, 255))
                        for hu, hv in info["corners"]:
                            hx = target.x() + hu * target.width()
                            hy = target.y() + hv * target.height()
                            painter.drawRect(QtCore.QRectF(hx - 4, hy - 4, 8, 8))
                        # Roll handle
                        ru, rv = info["roll_handle"]
                        rx = target.x() + ru * target.width()
                        ry = target.y() + rv * target.height()
                        painter.setBrush(QtGui.QColor(255, 180, 40))
                        painter.setPen(QtGui.QPen(QtGui.QColor(255, 180, 40), 2))
                        painter.drawEllipse(QtCore.QPointF(rx, ry), 5, 5)
                        painter.setBrush(Qt.NoBrush)
            if self._drag_start is not None and self._drag_end is not None:
                color = (
                    QtGui.QColor(80, 220, 255)
                    if self._pick_mode == "sky"
                    else QtGui.QColor(90, 240, 120)
                    if self._pick_mode == "ground"
                    else QtGui.QColor(245, 160, 255)
                    if self._pick_mode == "wb"
                    else QtGui.QColor(255, 200, 60)
                )
                pen = QtGui.QPen(color, 2, Qt.DashLine)
                painter.setPen(pen)
                rect = QtCore.QRect(self._drag_start, self._drag_end).normalized()
                if self._pick_mode == "light" and self._pick_shape == "circle":
                    painter.drawEllipse(rect)
                else:
                    painter.drawRect(rect)
            if self._brush_enabled:
                if self._brush_mode == "clone" and self._brush_clone_src_uv is not None:
                    su, sv = self._brush_clone_src_uv
                    sx = target.x() + su * target.width()
                    sy = target.y() + sv * target.height()
                    painter.setPen(QtGui.QPen(QtGui.QColor(80, 255, 160), 2, Qt.SolidLine))
                    painter.drawLine(int(sx - 8), int(sy), int(sx + 8), int(sy))
                    painter.drawLine(int(sx), int(sy - 8), int(sx), int(sy + 8))
                    painter.drawEllipse(QtCore.QPointF(sx, sy), 5, 5)
                if self._brush_hover_uv is not None:
                    bu, bv = self._brush_hover_uv
                    bx = target.x() + bu * target.width()
                    by = target.y() + bv * target.height()
                    rx = max(2.0, self._brush_radius_u * target.width())
                    ry = max(2.0, self._brush_radius_u * 2.0 * target.height())
                    b_col = QtGui.QColor(255, 160, 40) if self._brush_mode == "dodge" else (
                        QtGui.QColor(80, 200, 255) if self._brush_mode == "burn" else QtGui.QColor(255, 255, 255)
                    )
                    painter.setPen(QtGui.QPen(b_col, 1.5, Qt.SolidLine))
                    painter.drawEllipse(QtCore.QPointF(bx, by), rx, ry)
                    inner_scale = max(0.0, 1.0 - self._brush_softness)
                    if inner_scale > 0.02:
                        painter.setPen(QtGui.QPen(b_col, 1.0, Qt.DotLine))
                        painter.drawEllipse(QtCore.QPointF(bx, by), rx * inner_scale, ry * inner_scale)
                    if self._brush_adjust_mode is not None and self._brush_hud_text:
                        painter.setPen(QtGui.QPen(QtGui.QColor(255, 230, 100), 1))
                        painter.drawText(int(bx + rx + 8), int(by + 4), self._brush_hud_text)
            painter.end()
            return
        super().paintEvent(event)


class MacbethCanvasLabel(QtWidgets.QLabel):
    """Interactive Macbeth ColorChecker image canvas supporting click or drag patch sampling."""

    patchSelected = Signal(tuple)

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._drag_start: Optional[QtCore.QPoint] = None
        self._drag_end: Optional[QtCore.QPoint] = None
        self._sampled_rect: Optional[tuple] = None
        self.setCursor(Qt.CrossCursor)

    def set_sampled_rect(self, rect: Optional[tuple]) -> None:
        self._sampled_rect = rect
        self.update()

    def _pixmap_rect(self) -> QtCore.QRect:
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return self.rect()
        x = max(0, (self.width() - pm.width()) // 2)
        y = max(0, (self.height() - pm.height()) // 2)
        return QtCore.QRect(x, y, pm.width(), pm.height())

    def _pt_to_uv(self, pt: QtCore.QPoint) -> tuple:
        r = self._pixmap_rect()
        u = float(np.clip((pt.x() - r.x()) / max(r.width(), 1), 0.0, 1.0))
        v = float(np.clip((pt.y() - r.y()) / max(r.height(), 1), 0.0, 1.0))
        return u, v

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self._drag_end = event.pos()
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is not None:
            self._drag_end = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_start is not None and event.button() == Qt.LeftButton:
            u0, v0 = self._pt_to_uv(self._drag_start)
            u1, v1 = self._pt_to_uv(event.pos())
            self._drag_start = None
            self._drag_end = None
            if abs(u1 - u0) < 0.006 and abs(v1 - v0) < 0.006:
                uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
                rect = (max(0.0, uc - 0.015), max(0.0, vc - 0.015), min(1.0, uc + 0.015), min(1.0, vc + 0.015))
            else:
                rect = (min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1))
            self._sampled_rect = rect
            self.patchSelected.emit(rect)
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        pr = self._pixmap_rect()
        if self._sampled_rect is not None:
            u0, v0, u1, v1 = self._sampled_rect
            rx = pr.x() + int(u0 * pr.width())
            ry = pr.y() + int(v0 * pr.height())
            rw = max(2, int((u1 - u0) * pr.width()))
            rh = max(2, int((v1 - v0) * pr.height()))
            painter.setPen(QtGui.QPen(QtGui.QColor(80, 255, 160), 2, Qt.SolidLine))
            painter.drawRect(rx, ry, rw, rh)
        if self._drag_start is not None and self._drag_end is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 220, 60), 2, Qt.DashLine))
            painter.drawRect(QtCore.QRect(self._drag_start, self._drag_end).normalized())
        painter.end()


class MacbethWBDialog(QtWidgets.QDialog):
    """Load a Macbeth ColorChecker reference photo and sample a neutral gray patch for HDRI WB."""

    def __init__(
        self,
        parent=None,
        initial_path: str = "",
        initial_colorspace: str = "Auto",
        initial_gains: tuple = (1.0, 1.0, 1.0),
    ):
        super().__init__(parent)
        self.setWindowTitle("Macbeth ColorChecker White Balance Sampler")
        self.resize(780, 580)
        if parent is not None and parent.styleSheet():
            self.setStyleSheet(parent.styleSheet())

        self._pixels: Optional[np.ndarray] = None
        self._sampled_gains: tuple = tuple(float(v) for v in initial_gains)
        self._sampled_cct: float = 6500.0
        self._sampled_duv: float = 0.0
        self._sampled_lum: float = 0.18
        self._apply_requested: bool = False
        self._match_exposure: bool = False
        self._ev_delta: float = 0.0

        layout = QtWidgets.QVBoxLayout(self)

        # File & colorspace row
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Macbeth Image:"))
        self.path_edit = QtWidgets.QLineEdit(initial_path)
        self.path_edit.setPlaceholderText("Load Macbeth ColorChecker photo (.exr, .hdr, .jpg, .png, .tif)...")
        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        top_row.addWidget(self.path_edit, stretch=1)
        top_row.addWidget(browse_btn)

        top_row.addWidget(QtWidgets.QLabel("Input Space:"))
        self.cs_combo = QtWidgets.QComboBox()
        self.cs_combo.addItems(["Auto", "Linear Rec.709", "ACEScg", "sRGB"])
        self.cs_combo.setCurrentText(initial_colorspace or "Auto")
        self.cs_combo.currentIndexChanged.connect(lambda *_: self._load_image(self.path_edit.text().strip()))
        top_row.addWidget(self.cs_combo)
        layout.addLayout(top_row)

        # Preview EV row
        ev_row = QtWidgets.QHBoxLayout()
        ev_row.addWidget(QtWidgets.QLabel("Chart Preview EV:"))
        self.ev_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.ev_slider.setRange(-6, 6)
        self.ev_slider.setValue(0)
        self.ev_lbl = QtWidgets.QLabel("+0 EV")
        self.ev_slider.valueChanged.connect(self._on_ev_changed)
        ev_row.addWidget(self.ev_slider, stretch=1)
        ev_row.addWidget(self.ev_lbl)
        layout.addLayout(ev_row)

        # Image canvas
        self.canvas = MacbethCanvasLabel(
            "Load a Macbeth ColorChecker image above, then click or drag over a neutral gray patch (e.g. 18% Gray)."
        )
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setMinimumSize(520, 320)
        self.canvas.setStyleSheet("background-color: #111317; border: 1px solid #3a3f4b;")
        self.canvas.patchSelected.connect(self.sample_patch_rect)
        layout.addWidget(self.canvas, stretch=1)

        # Readout & exposure option
        self.readout_label = QtWidgets.QLabel(
            f"Current WB Gains: R={self._sampled_gains[0]:.3f}, G={self._sampled_gains[1]:.3f}, B={self._sampled_gains[2]:.3f}"
        )
        self.readout_label.setWordWrap(True)
        layout.addWidget(self.readout_label)

        self.match_exp_chk = QtWidgets.QCheckBox("Also match 18% Gray Exposure (0.18 linear) when applying")
        self.match_exp_chk.setChecked(False)
        layout.addWidget(self.match_exp_chk)

        # Action buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply WB to HDRI")
        self.apply_btn.setProperty("accent", "true")
        self.apply_btn.clicked.connect(self._on_apply)
        reset_btn = QtWidgets.QPushButton("Reset WB (1, 1, 1)")
        reset_btn.clicked.connect(self._on_reset)
        close_btn = QtWidgets.QPushButton("Cancel")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.apply_btn, stretch=1)
        btn_row.addWidget(reset_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        if initial_path and os.path.isfile(initial_path):
            self._load_image(initial_path)

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Macbeth ColorChecker Reference Image",
            self.path_edit.text().strip(),
            "Images (*.exr *.hdr *.jpg *.jpeg *.png *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.path_edit.setText(path)
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            return
        try:
            plate = read_plate(path, colorspace=self.cs_combo.currentText(), max_dim=1600)
            self._pixels = np.maximum(plate.pixels, 0.0).astype(np.float32)
            self._update_canvas_pixmap()
        except Exception as exc:
            self.readout_label.setText(f"Failed to load Macbeth image: {exc}")

    def _on_ev_changed(self, val: int) -> None:
        self.ev_lbl.setText(f"{val:+d} EV")
        self._update_canvas_pixmap()

    def _update_canvas_pixmap(self) -> None:
        if self._pixels is None:
            return
        srgb_u8 = render_preview_srgb(
            self._pixels,
            preview_ev=float(self.ev_slider.value()),
            tonemap="ACES",
        )
        h, w, _ = srgb_u8.shape
        qimg = QtGui.QImage(srgb_u8.data, w, h, w * 3, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.canvas.setPixmap(pix)

    def sample_patch_rect(self, rect: tuple) -> Optional[tuple]:
        """Sample linear RGB from `rect`=(u0,v0,u1,v1) on the loaded Macbeth chart."""
        if self._pixels is None:
            return None
        h, w, _ = self._pixels.shape
        u0, v0, u1, v1 = rect
        x0 = int(np.clip(min(u0, u1) * w, 0, max(0, w - 2)))
        x1 = int(np.clip(max(u0, u1) * w, x0 + 1, w))
        y0 = int(np.clip(min(v0, v1) * h, 0, max(0, h - 2)))
        y1 = int(np.clip(max(v0, v1) * h, y0 + 1, h))
        patch = self._pixels[y0:y1, x0:x1]
        mean_rgb = tuple(float(v) for v in np.mean(patch, axis=(0, 1)))
        gains, cct, duv, lum = compute_neutral_wb_gains(mean_rgb)
        self._sampled_gains = gains
        self._sampled_cct = cct
        self._sampled_duv = duv
        self._sampled_lum = lum
        self._ev_delta = float(np.clip(np.log2(0.18 / max(lum, 1e-4)), -6.0, 6.0))
        self.canvas.set_sampled_rect(rect)
        self.readout_label.setText(
            f"Sampled RGB: ({mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}) | "
            f"Measured: {cct:.0f} K (Duv {duv:+.4f}) | "
            f"WB Gains: (R {gains[0]:.3f}, G {gains[1]:.3f}, B {gains[2]:.3f})"
        )
        return gains

    def _on_apply(self) -> None:
        self._apply_requested = True
        self._match_exposure = self.match_exp_chk.isChecked()
        self.accept()

    def _on_reset(self) -> None:
        self._sampled_gains = (1.0, 1.0, 1.0)
        self._ev_delta = 0.0
        self._apply_requested = True
        self._match_exposure = False
        self.accept()

    def done(self, result: int) -> None:
        self._pixels = None
        self.canvas.setPixmap(QtGui.QPixmap())
        super().done(result)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_canvas_pixmap()


class DebugLogDialog(QtWidgets.QDialog):
    """Non-modal live debug log viewer showing runtime events, timings, and full tracebacks."""

    def __init__(self, parent=None, log_path: str = DEBUG_LOG_PATH):
        super().__init__(parent)
        self.setWindowTitle("HDRI Breaker - Debug Log")
        self.resize(780, 460)
        self._log_path = log_path

        layout = QtWidgets.QVBoxLayout(self)
        path_lbl = QtWidgets.QLabel(f"Log file: {log_path}")
        path_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(path_lbl)

        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QtGui.QFont("Consolas", 9))
        self.text_edit.setStyleSheet("background-color: #111318; color: #e2e8f0; border: 1px solid #3a3a3a;")
        layout.addWidget(self.text_edit, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        copy_btn = QtWidgets.QPushButton("Copy All")
        copy_btn.clicked.connect(self._on_copy)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def set_lines(self, lines: List[str]) -> None:
        self.text_edit.setPlainText("\n".join(lines))
        sb = self.text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def append_line(self, line: str) -> None:
        self.text_edit.appendPlainText(line)
        sb = self.text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_copy(self) -> None:
        cb = QtWidgets.QApplication.clipboard()
        if cb is not None:
            cb.setText(self.text_edit.toPlainText())

    def _on_clear(self) -> None:
        self.text_edit.clear()
        p = self.parent()
        if p is not None and hasattr(p, "_debug_lines"):
            p._debug_lines.clear()


class HDRIMakerWidget(QtWidgets.QWidget):
    """Main HDRI Breaker widget for Houdini Solaris."""

    def __init__(self, parent=None, restore_session: bool = True):
        super().__init__(parent)
        self.setWindowTitle("HDRI Breaker — Solaris 32-bit Float Lat/Long Studio")
        self.resize(1240, 800)
        icon_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "houdini", "config", "Icons", "hdri_breaker.png")
        )
        if os.path.isfile(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        self._profile: Optional[LightingProfile] = None
        self._frame_profiles: List[LightingProfile] = []
        self._views: List[PlateViewInput] = []
        self._hdri_pixels: Optional[np.ndarray] = None
        self._sky_rect: Optional[tuple] = None
        self._ground_rect: Optional[tuple] = None
        self._light_regions: List[Dict[str, Any]] = []
        self._extracted_lights: List[tuple] = []
        self._brush_strokes: List[Dict[str, Any]] = []
        self._brush_undo_stack: List[np.ndarray] = []
        self._pre_brush_hdri: Optional[np.ndarray] = None
        self._baked_canvas_hdri: Optional[np.ndarray] = None
        self._active_brush_stroke: Optional[Dict[str, Any]] = None
        self._brush_clone_src_uv: Optional[Tuple[float, float]] = None
        self._brush_clone_offset_uv: Optional[Tuple[float, float]] = None
        self._last_exr_path: str = ""
        self._last_hdri_full: Optional[np.ndarray] = None
        self._last_light_exr_paths: List[str] = []
        self._last_dome_node = None
        self._last_vshift: float = 0.0
        self._last_yaw: float = 0.0
        self._wb_gains: tuple = (1.0, 1.0, 1.0)
        self._macbeth_path: str = ""
        self._macbeth_colorspace: str = "Auto"
        self._thumb_cache: Dict[tuple, QtGui.QIcon] = {}
        base_pt = self.font().pointSizeF()
        self._base_font_pt: float = float(base_pt) if base_pt > 0 else 9.0
        self._suppress_auto_preview: bool = False
        self._debug_lines: List[str] = []
        self._debug_dialog: Optional[DebugLogDialog] = None
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._regenerate_preview)
        self._display_timer = QtCore.QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.setInterval(30)
        self._display_timer.timeout.connect(self._update_preview_pixmap)
        self._rough_cache: Optional[tuple] = None
        self._raw_rough_cache: Optional[tuple] = None

        self._build_ui()
        QShortcutCls = getattr(QtGui, "QShortcut", None) or getattr(QtWidgets, "QShortcut", None)
        if QShortcutCls is not None:
            self._undo_shortcut = QShortcutCls(QtGui.QKeySequence.Undo, self)
            self._undo_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            self._undo_shortcut.activated.connect(self._on_undo_brush)
        self._apply_ui_scale(self.ui_scale_slider.value())
        if restore_session:
            self._auto_load_session()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setMinimumSize(480, 360)
        root_layout = QtWidgets.QHBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # Left Panel: Controls + Analysis (freely resizable with scrollbars if narrow)
        self._left_scroll = QtWidgets.QScrollArea()
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setMinimumWidth(240)
        self._left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 8, 4)
        left_layout.setSpacing(8)

        left_layout.addWidget(self._build_plate_group())
        left_layout.addWidget(self._build_analysis_group())
        left_layout.addWidget(self._build_synthesis_group())
        left_layout.addStretch(1)
        self._left_scroll.setWidget(left_container)

        # Right Panel: Vertical Splitter between Live Preview and Solaris Output
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        solaris_scroll = QtWidgets.QScrollArea()
        solaris_scroll.setWidgetResizable(True)
        solaris_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        solaris_scroll.setMinimumHeight(140)
        solaris_scroll.setWidget(self._build_solaris_group())

        right_splitter = QtWidgets.QSplitter(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(self._build_preview_group())
        right_splitter.addWidget(solaris_scroll)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([480, 280])
        right_layout.addWidget(right_splitter, stretch=1)

        status_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready. Load a plate or image sequence to begin.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #9ecbff; padding: 4px;")
        save_btn = QtWidgets.QPushButton("Save State...")
        save_btn.setToolTip("Save current plates, positions, tuning, and inpaint areas to a JSON preset")
        save_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        save_btn.clicked.connect(self._on_save_state)

        load_btn = QtWidgets.QPushButton("Load State...")
        load_btn.setToolTip("Load a saved HDRI Maker JSON preset")
        load_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        load_btn.clicked.connect(self._on_load_state)

        reload_btn = QtWidgets.QPushButton("Reload Code")
        reload_btn.setToolTip("Hot-reload all hdri_maker Python modules without restarting Houdini")
        reload_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        reload_btn.clicked.connect(self._on_reload_code)

        debug_btn = QtWidgets.QPushButton("Debug Log")
        debug_btn.setToolTip(f"Open live Debug Log window (also written to {DEBUG_LOG_PATH})")
        debug_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        debug_btn.clicked.connect(self._on_open_debug_log)

        theme_lbl = QtWidgets.QLabel("Theme:")
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(list(UI_THEMES.keys()))
        self.theme_combo.setToolTip("Select HDRI Breaker UI color theme for high-contrast button visibility")
        self.theme_combo.currentIndexChanged.connect(lambda *_: self._apply_ui_scale(self.ui_scale_slider.value()))

        ui_scale_lbl = QtWidgets.QLabel("UI Scale:")
        self.ui_scale_slider = SliderResetControl(0.60, 1.40, 1.00, 0.05, decimals=2)
        self.ui_scale_slider.setToolTip("Scale the HDRI Breaker UI fonts, buttons, and row heights (0.60x - 1.40x)")
        self.ui_scale_slider.setMaximumWidth(210)
        self.ui_scale_slider.valueChanged.connect(self._apply_ui_scale)

        status_row.addWidget(self.status_label, stretch=1)
        status_row.addWidget(theme_lbl)
        status_row.addWidget(self.theme_combo)
        status_row.addWidget(ui_scale_lbl)
        status_row.addWidget(self.ui_scale_slider)
        status_row.addWidget(save_btn)
        status_row.addWidget(load_btn)
        status_row.addWidget(debug_btn)
        status_row.addWidget(reload_btn)
        right_layout.addLayout(status_row)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([560, 760])
        splitter.splitterMoved.connect(lambda *_: self._update_preview_pixmap())
        root_layout.addWidget(splitter)

    def _build_plate_group(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("1. Plate / Sequence Input")
        form = QtWidgets.QFormLayout(grp)

        # Path row
        path_row = QtWidgets.QHBoxLayout()
        self.plate_path_edit = QtWidgets.QLineEdit()
        self.plate_path_edit.setPlaceholderText("e.g. D:/plates/shot.$F4.exr or plate.jpg")
        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_plate)
        merge_hdr_btn = QtWidgets.QPushButton("Merge Brackets...")
        merge_hdr_btn.setToolTip(
            "Select 2+ bracketed LDR/HDR exposures (-2 EV, 0 EV, +2 EV) to merge into a 32-bit float HDR EXR"
        )
        merge_hdr_btn.clicked.connect(self._on_merge_brackets_clicked)
        path_row.addWidget(self.plate_path_edit)
        path_row.addWidget(browse_btn)
        path_row.addWidget(merge_hdr_btn)
        form.addRow("Plate Path:", path_row)

        # Colorspace & Sequence mode
        mode_row = QtWidgets.QHBoxLayout()
        self.colorspace_combo = QtWidgets.QComboBox()
        self.colorspace_combo.addItems(["Auto", "sRGB", "Linear Rec.709", "ACEScg"])
        self.seq_mode_combo = QtWidgets.QComboBox()
        self.seq_mode_combo.addItems([
            "Single Frame",
            "Sequence Master (Stabilized)",
            "Per-Frame Sequence",
        ])
        mode_row.addWidget(QtWidgets.QLabel("Color:"))
        mode_row.addWidget(self.colorspace_combo)
        mode_row.addWidget(QtWidgets.QLabel("Mode:"))
        mode_row.addWidget(self.seq_mode_combo)
        form.addRow("Input Mode:", mode_row)

        # Frame range
        fr_row = QtWidgets.QHBoxLayout()
        self.start_frame_spin = QtWidgets.QSpinBox()
        self.start_frame_spin.setRange(-999999, 999999)
        self.start_frame_spin.setValue(1)
        self.end_frame_spin = QtWidgets.QSpinBox()
        self.end_frame_spin.setRange(-999999, 999999)
        self.end_frame_spin.setValue(100)
        self.step_frame_spin = QtWidgets.QSpinBox()
        self.step_frame_spin.setRange(1, 1000)
        self.step_frame_spin.setValue(1)
        self.use_range_chk = QtWidgets.QCheckBox("Limit Range")
        fr_row.addWidget(self.use_range_chk)
        fr_row.addWidget(QtWidgets.QLabel("Start:"))
        fr_row.addWidget(self.start_frame_spin)
        fr_row.addWidget(QtWidgets.QLabel("End:"))
        fr_row.addWidget(self.end_frame_spin)
        fr_row.addWidget(QtWidgets.QLabel("Step:"))
        fr_row.addWidget(self.step_frame_spin)
        form.addRow("Frames:", fr_row)

        # Camera mapping & hotspot settings
        cam_row = QtWidgets.QHBoxLayout()
        self.fov_spin = QtWidgets.QDoubleSpinBox()
        self.fov_spin.setRange(10.0, 360.0)
        self.fov_spin.setValue(90.0)
        self.cam_az_spin = QtWidgets.QDoubleSpinBox()
        self.cam_az_spin.setRange(-180.0, 180.0)
        self.cam_az_spin.setValue(0.0)
        self.cam_el_spin = QtWidgets.QDoubleSpinBox()
        self.cam_el_spin.setRange(-85.0, 85.0)
        self.cam_el_spin.setValue(0.0)
        cam_row.addWidget(QtWidgets.QLabel("FOV°:"))
        cam_row.addWidget(self.fov_spin)
        cam_row.addWidget(QtWidgets.QLabel("Az°:"))
        cam_row.addWidget(self.cam_az_spin)
        cam_row.addWidget(QtWidgets.QLabel("El°:"))
        cam_row.addWidget(self.cam_el_spin)
        self.link_cam_btn = QtWidgets.QPushButton("Link Scene Cam...")
        self.link_cam_btn.setToolTip("Read FOV, Azimuth, Elevation, and Roll from a Houdini /stage USD Camera or /obj Camera")
        self.link_cam_btn.clicked.connect(self._on_link_scene_camera)
        cam_row.addWidget(self.link_cam_btn)
        form.addRow("Camera Map:", cam_row)

        hs_row = QtWidgets.QHBoxLayout()
        self.max_hotspots_spin = QtWidgets.QSpinBox()
        self.max_hotspots_spin.setRange(1, 12)
        self.max_hotspots_spin.setValue(5)
        self.ldr_boost_spin = QtWidgets.QDoubleSpinBox()
        self.ldr_boost_spin.setRange(1.0, 100.0)
        self.ldr_boost_spin.setValue(4.0)
        hs_row.addWidget(QtWidgets.QLabel("Max Hotspots:"))
        hs_row.addWidget(self.max_hotspots_spin)
        hs_row.addWidget(QtWidgets.QLabel("LDR Boost:"))
        hs_row.addWidget(self.ldr_boost_spin)
        form.addRow("Detection:", hs_row)

        # Multi-Position Plate List (for multiple photos taken from different angles)
        mv_btn_grid = QtWidgets.QGridLayout()
        add_multi_btn = QtWidgets.QPushButton("Add Plates...")
        add_multi_btn.setToolTip("Browse and add one or more plate images to the list")
        add_multi_btn.clicked.connect(self._on_add_multiview_plates)
        add_curr_btn = QtWidgets.QPushButton("Add Current")
        add_curr_btn.setToolTip("Add the current Plate Path above to the list")
        add_curr_btn.clicked.connect(self._on_add_current_plate)
        rm_mv_btn = QtWidgets.QPushButton("Remove")
        rm_mv_btn.setToolTip("Remove selected plate(s) from the list (Del)")
        rm_mv_btn.clicked.connect(self._on_remove_selected_multiview_plates)
        auto_360_btn = QtWidgets.QPushButton("Auto 360°")
        auto_360_btn.setToolTip("Evenly distribute Azimuth° across 360° for all listed plates")
        auto_360_btn.clicked.connect(self._on_auto_distribute_360)
        align_btn = QtWidgets.QPushButton("Auto-Align (Spherical)")
        align_btn.setToolTip("Refine overlapping plate Azimuth°/Elevation° via coarse-to-fine spherical image correlation")
        align_btn.clicked.connect(self._on_auto_align_plates)
        clear_mv_btn = QtWidgets.QPushButton("Clear")
        clear_mv_btn.setToolTip("Remove all plates from the list")
        clear_mv_btn.clicked.connect(self._on_clear_multiview_plates)
        mv_btn_grid.addWidget(add_multi_btn, 0, 0)
        mv_btn_grid.addWidget(add_curr_btn, 0, 1)
        mv_btn_grid.addWidget(rm_mv_btn, 0, 2)
        mv_btn_grid.addWidget(auto_360_btn, 1, 0)
        mv_btn_grid.addWidget(align_btn, 1, 1)
        mv_btn_grid.addWidget(clear_mv_btn, 1, 2)
        eq_ev_btn = QtWidgets.QPushButton("Equalize EV")
        eq_ev_btn.setToolTip("Automatically match exposure across overlapping plates to eliminate brightness seams")
        eq_ev_btn.clicked.connect(self._on_equalize_plate_exposures)
        mv_btn_grid.addWidget(eq_ev_btn, 2, 0, 1, 3)
        form.addRow("Multi-Position:", mv_btn_grid)

        self.multiview_table = QtWidgets.QTableWidget(0, 6)
        self.multiview_table.setHorizontalHeaderLabels(
            ["Plate File", "Az°", "El°", "FOV°", "Color", ""]
        )
        self.multiview_table.setIconSize(QtCore.QSize(68, 44))
        self.multiview_table.verticalHeader().setDefaultSectionSize(50)
        self.multiview_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.multiview_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.multiview_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.multiview_table.customContextMenuRequested.connect(self._on_multiview_context_menu)
        mv_hdr = self.multiview_table.horizontalHeader()
        mv_hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for col in (1, 2, 3, 4, 5):
            mv_hdr.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.multiview_table.setMinimumHeight(140)
        self.multiview_table.setMaximumHeight(240)
        self.multiview_table.setToolTip(
            "Optional: Add multiple plates taken from different camera positions/angles.\n"
            "Double-click Az°, El°, or FOV° to adjust each photo's orientation.\n"
            "Select rows and press Del or click × / Remove to remove plates."
        )
        shortcut_cls = getattr(QtGui, "QShortcut", None) or getattr(QtWidgets, "QShortcut", None)
        if shortcut_cls is not None:
            del_sc = shortcut_cls(QtGui.QKeySequence.Delete, self.multiview_table)
            del_sc.setContext(Qt.WidgetWithChildrenShortcut)
            del_sc.activated.connect(self._on_remove_selected_multiview_plates)
        form.addRow(self.multiview_table)

        self.analyze_btn = QtWidgets.QPushButton("Analyze Plate(s) & Generate Preview")
        self.analyze_btn.setMinimumHeight(32)
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        form.addRow(self.analyze_btn)

        return grp

    def _build_analysis_group(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("2. Extracted Lighting Data")
        layout = QtWidgets.QVBoxLayout(grp)

        self.metrics_label = QtWidgets.QLabel("No plate analyzed yet.")
        self.metrics_label.setWordWrap(True)
        layout.addWidget(self.metrics_label)

        # Tonal swatches
        swatch_row = QtWidgets.QHBoxLayout()
        self._swatches = {}
        for name in ("Zenith", "Sky", "Horizon", "Ground", "Nadir"):
            lbl = QtWidgets.QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumHeight(26)
            lbl.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
            swatch_row.addWidget(lbl)
            self._swatches[name] = lbl
        layout.addLayout(swatch_row)

        # Hotspots table
        self.hotspot_table = QtWidgets.QTableWidget(0, 6)
        self.hotspot_table.setHorizontalHeaderLabels(
            ["#", "Az°", "El°", "Radius°", "Peak HDR", "CCT (K)"]
        )
        self.hotspot_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.hotspot_table.setMaximumHeight(130)
        layout.addWidget(self.hotspot_table)

        return grp

    def _build_synthesis_group(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("3. 32-bit Lat/Long Synthesis Tuning")
        form = QtWidgets.QFormLayout(grp)

        res_row = QtWidgets.QHBoxLayout()
        self.res_combo = QtWidgets.QComboBox()
        self.res_combo.addItems(["1024 x 512", "2048 x 1024", "4096 x 2048", "8192 x 4096"])
        self.res_combo.setCurrentIndex(1)
        self.res_combo.setToolTip("Output resolution for the synthesized 32-bit float Lat/Long Dome HDRI (up to 8K)")
        self.out_cs_combo = QtWidgets.QComboBox()
        self.out_cs_combo.addItems(["Linear Rec.709", "ACEScg"])
        res_row.addWidget(self.res_combo)
        res_row.addWidget(QtWidgets.QLabel("Out Space:"))
        res_row.addWidget(self.out_cs_combo)
        form.addRow("Resolution:", res_row)

        self.ev_spin = SliderResetControl(-10.0, 10.0, 0.0, 0.05)
        self.hs_boost_spin = SliderResetControl(0.0, 20.0, 0.0, 0.1)
        self.hs_soft_spin = SliderResetControl(0.2, 5.0, 1.0, 0.05)
        self.wrap_spin = SliderResetControl(0.0, 1.0, 0.55, 0.01)
        self.sky_gain_spin = SliderResetControl(0.0, 10.0, 1.0, 0.05)
        self.gnd_gain_spin = SliderResetControl(0.0, 10.0, 1.0, 0.05)
        self.sat_spin = SliderResetControl(0.0, 2.0, 1.0, 0.01)
        self.temp_k_spin = SliderResetControl(1800.0, 20000.0, 6500.0, 50.0, decimals=0)
        self.temp_blend_spin = SliderResetControl(0.0, 1.0, 0.0, 0.01)
        self.plate_blend_spin = SliderResetControl(0.0, 1.0, 1.0, 0.01)
        self.yaw_spin = SliderResetControl(-180.0, 180.0, 0.0, 1.0, decimals=1)
        self.vshift_spin = SliderResetControl(-85.0, 85.0, 0.0, 0.5, decimals=1)
        self.stitch_blend_spin = SliderResetControl(0.03, 1.0, 0.35, 0.01)
        self.sky_blend_slider = SliderResetControl(0.0, 1.0, 0.35, 0.01)
        self.sky_blend_slider.setToolTip("Vertical feather blend between stitched plates and sky fill")
        self.gnd_blend_slider = SliderResetControl(0.0, 1.0, 0.35, 0.01)
        self.gnd_blend_slider.setToolTip("Vertical feather blend between stitched plates and ground fill")
        self.sky_blur_slider = SliderResetControl(0.0, 1.0, 0.0, 0.01)
        self.sky_blur_slider.setToolTip("2D wrap-safe blur for the synthesized sky dome to remove horizontal stripes")
        self.gnd_blur_slider = SliderResetControl(0.0, 1.0, 0.0, 0.01)
        self.gnd_blur_slider.setToolTip("2D wrap-safe blur for the synthesized ground dome to remove horizontal stripes")
        self.hdri_blur_slider = SliderResetControl(0.0, 1.0, 0.0, 0.01)
        self.hdri_blur_slider.setToolTip("360-deg wrap-safe Gaussian-approximate blur for the output HDRI")
        self.sky_fill_slider = SliderResetControl(0.0, 1.0, 0.0, 0.02)
        self.sky_fill_slider.setToolTip("Synthesize a physical Rayleigh/Mie sky gradient & HDR sun disc in uncovered upper-hemisphere gaps")
        self.highlight_recovery_slider = SliderResetControl(0.0, 10.0, 0.0, 0.25)
        self.highlight_recovery_slider.setToolTip("Reconstruct smooth physical HDR peak energy inside clipped LDR highlight cores (sun/softboxes)")
        self.nadir_cleanup_slider = SliderResetControl(0.0, 1.0, 0.0, 0.02)
        self.nadir_cleanup_slider.setToolTip("Seam-aware radial clone & pole smoothing to clean up tripods or shadows at the nadir")
        self.temporal_stab_slider = SliderResetControl(0.0, 1.0, 0.0, 0.05)
        self.temporal_stab_slider.setToolTip("Temporal EV & white-balance smoothing across exported sequence ($F4) frames")

        self.inpaint_chk = QtWidgets.QCheckBox("Inpaint Sky/Ground")
        self.inpaint_chk.setChecked(True)
        self.neutralize_env_chk = QtWidgets.QCheckBox("Neutralize Environment (Turntable WB)")
        self.neutralize_env_chk.setToolTip("Strip heavy ambient color cast while preserving contrast and key/fill ratios for asset lookdev")
        self.neutralize_env_chk.setChecked(False)

        vshift_row = QtWidgets.QHBoxLayout()
        vshift_row.setContentsMargins(0, 0, 0, 0)
        vshift_row.addWidget(self.vshift_spin, stretch=1)
        auto_horiz_btn = QtWidgets.QPushButton("Auto-Level")
        auto_horiz_btn.setToolTip("Automatically detect the sky/ground horizon line and level Horizon Shift")
        auto_horiz_btn.clicked.connect(self._on_auto_level_horizon)
        vshift_row.addWidget(auto_horiz_btn)

        form.addRow("Exposure (EV):", self.ev_spin)
        form.addRow("Hotspot Boost:", self.hs_boost_spin)
        form.addRow("Hotspot Soft:", self.hs_soft_spin)
        form.addRow("Ambient Wrap:", self.wrap_spin)
        form.addRow("Sky Gain:", self.sky_gain_spin)
        form.addRow("Ground Gain:", self.gnd_gain_spin)
        form.addRow("Saturation:", self.sat_spin)
        form.addRow("Target CCT (K):", self.temp_k_spin)
        form.addRow("CCT Blend:", self.temp_blend_spin)
        form.addRow("Plate Proj Blend:", self.plate_blend_spin)
        form.addRow("HDRI Yaw (°):", self.yaw_spin)
        form.addRow("Horizon Shift (°):", vshift_row)
        form.addRow("Stitch Blend:", self.stitch_blend_spin)
        form.addRow("Sky Blend:", self.sky_blend_slider)
        form.addRow("Ground Blend:", self.gnd_blend_slider)
        form.addRow("Sky Blur:", self.sky_blur_slider)
        form.addRow("Ground Blur:", self.gnd_blur_slider)
        form.addRow("HDRI Blur:", self.hdri_blur_slider)
        form.addRow("Physical Sky Fill:", self.sky_fill_slider)
        form.addRow("Highlight Recovery:", self.highlight_recovery_slider)
        form.addRow("Nadir Cleanup:", self.nadir_cleanup_slider)
        form.addRow("Temporal Smooth:", self.temporal_stab_slider)
        form.addRow("", self.inpaint_chk)
        form.addRow("", self.neutralize_env_chk)

        regen_btn = QtWidgets.QPushButton("Update Preview")
        regen_btn.clicked.connect(self._regenerate_preview)
        form.addRow(regen_btn)

        # Dynamically update HDRI viewer whenever any tuning parameter changes
        self.res_combo.currentIndexChanged.connect(self._schedule_preview_update)
        self.out_cs_combo.currentIndexChanged.connect(self._schedule_preview_update)
        self.inpaint_chk.toggled.connect(self._schedule_preview_update)
        self.neutralize_env_chk.toggled.connect(self._schedule_preview_update)
        for ctrl in (
            self.ev_spin,
            self.hs_boost_spin,
            self.hs_soft_spin,
            self.wrap_spin,
            self.sky_gain_spin,
            self.gnd_gain_spin,
            self.sat_spin,
            self.temp_k_spin,
            self.temp_blend_spin,
            self.plate_blend_spin,
            self.yaw_spin,
            self.vshift_spin,
            self.stitch_blend_spin,
            self.sky_blend_slider,
            self.gnd_blend_slider,
            self.sky_blur_slider,
            self.gnd_blur_slider,
            self.hdri_blur_slider,
            self.sky_fill_slider,
            self.highlight_recovery_slider,
            self.nadir_cleanup_slider,
        ):
            ctrl.valueChanged.connect(self._schedule_preview_update)
        self.vshift_spin.valueChanged.connect(self._on_vshift_changed)
        self.yaw_spin.valueChanged.connect(self._on_yaw_changed)

        return grp

    @staticmethod
    def _shift_rect_v(rect: Optional[tuple], dv: float) -> Optional[tuple]:
        """Shift a normalized (u0, v0, u1, v1) rect vertically by dv, clamping inside [0, 1]."""
        if rect is None or abs(dv) < 1e-7:
            return rect
        u0, v0, u1, v1 = rect
        lo = min(v0, v1)
        hi = max(v0, v1)
        span = hi - lo
        new_lo = float(np.clip(lo + dv, 0.0, max(0.0, 1.0 - span)))
        new_hi = float(np.clip(new_lo + span, span, 1.0))
        return (u0, new_lo, u1, new_hi)

    @staticmethod
    def _shift_rect_u(rect: Optional[tuple], du: float) -> Optional[tuple]:
        """Shift a normalized (u0, v0, u1, v1) rect horizontally by du (wrapped mod 1.0, clamped inside [0, 1])."""
        if rect is None or abs(du) < 1e-7:
            return rect
        u0, v0, u1, v1 = rect
        lo = min(u0, u1)
        hi = max(u0, u1)
        span = min(1.0, hi - lo)
        center = (0.5 * (lo + hi) + du) % 1.0
        new_lo = float(np.clip(center - 0.5 * span, 0.0, max(0.0, 1.0 - span)))
        new_hi = float(np.clip(new_lo + span, span, 1.0))
        return (new_lo, v0, new_hi, v1)

    def _on_vshift_changed(self, new_val: float) -> None:
        """Keep Sky, Ground, and Light selection boxes aligned when Horizon Shift changes."""
        if self._suppress_auto_preview:
            self._last_vshift = float(new_val)
            return
        delta_deg = float(new_val) - float(self._last_vshift)
        self._last_vshift = float(new_val)
        if abs(delta_deg) < 1e-6:
            return
        dv = -delta_deg / 180.0
        self._sky_rect = self._shift_rect_v(self._sky_rect, dv)
        self._ground_rect = self._shift_rect_v(self._ground_rect, dv)
        for region in self._light_regions:
            if "rect" in region and region["rect"] is not None:
                region["rect"] = self._shift_rect_v(region["rect"], dv)
        self.preview_label.set_regions(self._sky_rect, self._ground_rect, self._light_regions)
        self._update_preview_pixmap()

    def _on_yaw_changed(self, new_val: float) -> None:
        """Keep Sky, Ground, and Light selection boxes aligned when HDRI Yaw changes."""
        if self._suppress_auto_preview:
            self._last_yaw = float(new_val)
            return
        delta_deg = float(new_val) - float(self._last_yaw)
        self._last_yaw = float(new_val)
        if abs(delta_deg) < 1e-6:
            return
        du = delta_deg / 360.0
        self._sky_rect = self._shift_rect_u(self._sky_rect, du)
        self._ground_rect = self._shift_rect_u(self._ground_rect, du)
        for region in self._light_regions:
            if "rect" in region and region["rect"] is not None:
                region["rect"] = self._shift_rect_u(region["rect"], du)
        self.preview_label.set_regions(self._sky_rect, self._ground_rect, self._light_regions)
        self._update_preview_pixmap()

    def _schedule_preview_update(self, *_args) -> None:
        """Debounce live preview regeneration when synthesis parameters change."""
        if self._profile is not None and not self._suppress_auto_preview:
            self._preview_timer.start()

    def _apply_ui_scale(self, scale: float) -> None:
        """Apply selected UI theme and scale fonts, button heights, and table rows inside HDRIMakerWidget."""
        s = float(np.clip(scale, 0.60, 1.40))
        pt = max(5.5, self._base_font_pt * s)
        pad_v = max(2, int(round(4 * s)))
        pad_h = max(4, int(round(8 * s)))
        font = self.font()
        font.setPointSizeF(pt)
        self.setFont(font)

        theme_name = self.theme_combo.currentText() if hasattr(self, "theme_combo") else "Breaker Dark (Amber)"
        t = UI_THEMES.get(theme_name, UI_THEMES["Breaker Dark (Amber)"])
        self.setStyleSheet(
            f"QWidget {{ background-color: {t['bg']}; color: {t['fg']}; font-size: {pt:.1f}pt; }}\n"
            f"QGroupBox {{ background-color: {t['panel_bg']}; border: 1px solid {t['btn_border']}; border-radius: 4px; margin-top: 8px; padding-top: 8px; font-weight: bold; }}\n"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {t['accent']}; }}\n"
            f"QPushButton, QToolButton {{ background-color: {t['btn_bg']}; color: {t['fg']}; border: 1px solid {t['btn_border']}; border-radius: 3px; padding: {pad_v}px {pad_h}px; font-weight: 500; }}\n"
            f"QPushButton:hover, QToolButton:hover {{ background-color: {t['btn_hover']}; border: 1px solid {t['accent']}; }}\n"
            f"QPushButton:pressed, QPushButton:checked, QToolButton:pressed, QToolButton:checked {{ background-color: {t['accent']}; color: {t['accent_fg']}; border: 1px solid {t['accent']}; font-weight: bold; }}\n"
            f"QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTableWidget {{ background-color: {t['input_bg']}; color: {t['fg']}; border: 1px solid {t['btn_border']}; border-radius: 2px; selection-background-color: {t['accent']}; selection-color: {t['accent_fg']}; }}\n"
            f"QHeaderView::section {{ background-color: {t['btn_bg']}; color: {t['fg']}; border: 1px solid {t['btn_border']}; padding: 2px 4px; font-weight: bold; }}\n"
        )
        if hasattr(self, "status_label"):
            self.status_label.setStyleSheet(f"color: {t['status_fg']}; padding: 4px; font-weight: bold;")
        btn_h = max(20, int(round(34 * s)))
        for b in (self.export_btn, self.dome_btn, self.rig_btn, self.update_lookdev_btn):
            b.setMinimumHeight(btn_h)
        self.multiview_table.setIconSize(
            QtCore.QSize(max(36, int(round(68 * s))), max(24, int(round(44 * s))))
        )
        self.multiview_table.verticalHeader().setDefaultSectionSize(max(28, int(round(50 * s))))
        self.setMinimumSize(max(300, int(round(480 * min(s, 1.0)))), max(240, int(round(360 * min(s, 1.0)))))
        self._left_scroll.setMinimumWidth(max(150, int(round(240 * min(s, 1.0)))))

    def _build_preview_group(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("Lat/Long 2:1 HDRI Preview")
        layout = QtWidgets.QVBoxLayout(grp)

        self.preview_label = PreviewLabel("Analyze a plate to preview generated 32-bit Lat/Long HDRI")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(180, 90)
        self.preview_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored
        )
        self.preview_label.setStyleSheet("background-color: #161616; border: 1px solid #3a3a3a;")
        self.preview_label.rectSelected.connect(self._on_preview_rect_selected)
        self.preview_label.regionEdited.connect(self._on_preview_region_edited)
        self.preview_label.pixelHovered.connect(self._on_preview_pixel_hovered)
        self.preview_label.plateSelected.connect(self._on_viewer_plate_selected)
        self.preview_label.plateDragMoved.connect(self._on_viewer_plate_dragged)
        self.preview_label.plateDragFinished.connect(self._on_viewer_plate_drag_finished)
        self.preview_label.plateWheelScaled.connect(self._on_viewer_plate_wheel_scaled)
        self._selected_plate_idx: int = 0
        self._updating_plate_ui: bool = False
        layout.addWidget(self.preview_label, stretch=1)

        self.inspector_label = QtWidgets.QLabel(
            "Inspector: Hover over preview to inspect Az°, El°, Linear RGB, and EV stops | Drag existing boxes to move/resize"
        )
        self.inspector_label.setStyleSheet("font-family: Consolas, monospace; padding: 2px 4px;")
        layout.addWidget(self.inspector_label)

        ev_row = QtWidgets.QHBoxLayout()
        ev_row.addWidget(QtWidgets.QLabel("View:"))
        self.preview_tonemap_combo = QtWidgets.QComboBox()
        self.preview_tonemap_combo.addItems(["ACES", "False Color (EV)", "Reinhard", "Linear", "Overlap Diff"])
        self.preview_tonemap_combo.setToolTip(
            "Preview display transform (does not affect exported 32-bit float linear EXR)"
        )
        self.preview_tonemap_combo.currentIndexChanged.connect(self._update_preview_pixmap)
        ev_row.addWidget(self.preview_tonemap_combo)

        ev_row.addWidget(QtWidgets.QLabel("Preview EV:"))
        self.preview_ev_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.preview_ev_slider.setRange(-60, 60)
        self.preview_ev_slider.setValue(0)
        self.preview_ev_slider.valueChanged.connect(self._schedule_display_update)
        ev_row.addWidget(self.preview_ev_slider)

        self.preview_ev_val = QtWidgets.QLabel("0.0 EV")
        ev_row.addWidget(self.preview_ev_val)
        self.preview_ev_reset_btn = QtWidgets.QPushButton("R")
        self.preview_ev_reset_btn.setMaximumWidth(24)
        self.preview_ev_reset_btn.setToolTip("Reset Preview EV to 0.0 EV")
        self.preview_ev_reset_btn.clicked.connect(lambda: self.preview_ev_slider.setValue(0))
        ev_row.addWidget(self.preview_ev_reset_btn)

        ev_row.addWidget(QtWidgets.QLabel("Rough:"))
        self.preview_roughness_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.preview_roughness_slider.setRange(0, 100)
        self.preview_roughness_slider.setValue(0)
        self.preview_roughness_slider.setMaximumWidth(80)
        self.preview_roughness_slider.setToolTip("Simulate glossy-to-diffuse reflection roughness (0.0 .. 1.0) in the 2:1 preview")
        self.preview_roughness_slider.valueChanged.connect(self._schedule_display_update)
        ev_row.addWidget(self.preview_roughness_slider)
        self.preview_roughness_val = QtWidgets.QLabel("0.00")
        ev_row.addWidget(self.preview_roughness_val)
        self.preview_roughness_reset_btn = QtWidgets.QPushButton("R")
        self.preview_roughness_reset_btn.setMaximumWidth(24)
        self.preview_roughness_reset_btn.setToolTip("Reset Roughness Preview to 0.00")
        self.preview_roughness_reset_btn.clicked.connect(lambda: self.preview_roughness_slider.setValue(0))
        ev_row.addWidget(self.preview_roughness_reset_btn)

        self.ab_split_btn = QtWidgets.QPushButton("A/B Split")
        self.ab_split_btn.setCheckable(True)
        self.ab_split_btn.setToolTip("Compare Raw Plate Projection (left) vs Synthesized HDRI (right) with a draggable divider")
        self.ab_split_btn.toggled.connect(self._on_toggle_ab_split)
        ev_row.addWidget(self.ab_split_btn)

        self.show_markers_chk = QtWidgets.QCheckBox("Hotspot Markers")
        self.show_markers_chk.setChecked(False)
        self.show_markers_chk.toggled.connect(self._update_preview_pixmap)
        ev_row.addWidget(self.show_markers_chk)

        self.edit_plates_btn = QtWidgets.QPushButton("Move / Align Plates")
        self.edit_plates_btn.setCheckable(True)
        self.edit_plates_btn.setStyleSheet(
            "QPushButton:checked { background-color: #0e7490; color: #ffffff; font-weight: bold; }"
        )
        self.edit_plates_btn.setToolTip(
            "PTGui-style interactive plate alignment in viewer:\n"
            "• Click plate to select\n"
            "• Drag to move Az° / El°\n"
            "• Shift+Drag or orange top handle to Roll°\n"
            "• Drag cyan corner handles or Shift+Wheel to Scale FOV°\n"
            "• Ctrl+Drag to adjust Lens Warp (k1)"
        )
        self.edit_plates_btn.toggled.connect(self._toggle_plate_edit_mode)
        ev_row.addWidget(self.edit_plates_btn)

        zoom_out_btn = QtWidgets.QPushButton("-")
        zoom_out_btn.setMaximumWidth(28)
        zoom_out_btn.setToolTip("Zoom Out (or scroll mouse wheel down)")
        zoom_out_btn.clicked.connect(lambda: self.preview_label.set_zoom(self.preview_label.zoom_factor() / 1.25))
        self.zoom_reset_btn = QtWidgets.QPushButton("1.0x")
        self.zoom_reset_btn.setMaximumWidth(56)
        self.zoom_reset_btn.setToolTip("Reset Viewer Zoom to Fit (or double-click viewer). Middle/Right-drag to pan when zoomed.")
        self.zoom_reset_btn.clicked.connect(self.preview_label.reset_zoom)
        zoom_in_btn = QtWidgets.QPushButton("+")
        zoom_in_btn.setMaximumWidth(28)
        zoom_in_btn.setToolTip("Zoom In (or scroll mouse wheel up)")
        zoom_in_btn.clicked.connect(lambda: self.preview_label.set_zoom(self.preview_label.zoom_factor() * 1.25))
        self.preview_label.zoomChanged.connect(lambda z: self.zoom_reset_btn.setText(f"{z:.1f}x"))
        ev_row.addWidget(zoom_out_btn)
        ev_row.addWidget(self.zoom_reset_btn)
        ev_row.addWidget(zoom_in_btn)
        layout.addLayout(ev_row)

        ctrl_row = QtWidgets.QHBoxLayout()
        self.pick_sky_btn = QtWidgets.QPushButton("Pick Sky Area")
        self.pick_sky_btn.setCheckable(True)
        self.pick_sky_btn.toggled.connect(lambda c: self._toggle_pick_mode("sky", c))
        self.pick_gnd_btn = QtWidgets.QPushButton("Pick Ground Area")
        self.pick_gnd_btn.setCheckable(True)
        self.pick_gnd_btn.toggled.connect(lambda c: self._toggle_pick_mode("ground", c))
        self.pick_light_btn = QtWidgets.QPushButton("Pick Light Area")
        self.pick_light_btn.setCheckable(True)
        self.pick_light_btn.setToolTip("Drag on the HDRI to add multiple Area Lights (Rectangle or Circle)")
        self.pick_light_btn.toggled.connect(lambda c: self._toggle_pick_mode("light", c))
        self.pick_wb_btn = QtWidgets.QPushButton("Pick Neutral WB")
        self.pick_wb_btn.setCheckable(True)
        self.pick_wb_btn.setToolTip(
            "Drag over a neutral gray / Macbeth patch in the HDRI preview to auto-calibrate White Balance"
        )
        self.pick_wb_btn.toggled.connect(lambda c: self._toggle_pick_mode("wb", c))
        self.macbeth_wb_btn = QtWidgets.QPushButton("Macbeth WB...")
        self.macbeth_wb_btn.setToolTip(
            "Load a separate Macbeth ColorChecker reference image and sample any neutral gray patch for White Balance"
        )
        self.macbeth_wb_btn.clicked.connect(self._on_open_macbeth_wb)
        self.reset_wb_btn = QtWidgets.QPushButton("Reset WB")
        self.reset_wb_btn.setToolTip("Reset sampled Macbeth / Neutral White Balance gains back to (1.0, 1.0, 1.0)")
        self.reset_wb_btn.clicked.connect(self._on_reset_wb)
        self.light_shape_combo = QtWidgets.QComboBox()
        self.light_shape_combo.addItems(["Rectangle", "Circle"])
        self.light_shape_combo.currentIndexChanged.connect(self._on_light_shape_changed)
        self.remove_hotspot_chk = QtWidgets.QCheckBox("Inpaint Light Out")
        self.remove_hotspot_chk.setChecked(True)
        self.remove_hotspot_chk.toggled.connect(self._regenerate_preview)
        undo_light_btn = QtWidgets.QPushButton("Undo Light")
        undo_light_btn.setToolTip("Remove the last picked Area Light")
        undo_light_btn.clicked.connect(self._on_undo_light)
        clear_area_btn = QtWidgets.QPushButton("Clear Areas")
        clear_area_btn.clicked.connect(self._clear_inpaint_areas)

        ctrl_row.addWidget(self.pick_sky_btn)
        ctrl_row.addWidget(self.pick_gnd_btn)
        ctrl_row.addWidget(self.pick_light_btn)
        ctrl_row.addWidget(self.pick_wb_btn)
        ctrl_row.addWidget(self.macbeth_wb_btn)
        ctrl_row.addWidget(self.reset_wb_btn)
        ctrl_row.addWidget(self.light_shape_combo)
        ctrl_row.addWidget(self.remove_hotspot_chk)
        ctrl_row.addWidget(undo_light_btn)
        ctrl_row.addWidget(clear_area_btn)
        layout.addLayout(ctrl_row)

        # Interactive 32-bit Float Canvas Brush Toolbar
        brush_row = QtWidgets.QHBoxLayout()
        self.brush_btn = QtWidgets.QPushButton("Brush Paint")
        self.brush_btn.setCheckable(True)
        self.brush_btn.setStyleSheet(
            "QPushButton:checked { background-color: #b45309; color: #ffffff; font-weight: bold; }"
        )
        self.brush_btn.setToolTip(
            "Paint directly on the 32-bit float 2:1 HDRI canvas:\n"
            "• Heal / Inpaint: Clean spots/tripods from surrounding ring\n"
            "• Clone Stamp: Ctrl+Click to pick source, then paint (360° wrap-aware)\n"
            "• Blur / Burn / Dodge / Desaturate: Smooth seams, dim fireflies, or boost highlights"
        )
        self.brush_btn.toggled.connect(self._on_toggle_brush_mode)
        self.brush_mode_combo = QtWidgets.QComboBox()
        self.brush_mode_combo.addItems([
            "Heal / Inpaint",
            "Clone Stamp",
            "Blur / Smooth",
            "Burn (Dim HDR)",
            "Dodge (Boost HDR)",
            "Desaturate",
        ])
        self.brush_mode_combo.currentIndexChanged.connect(self._on_brush_config_changed)
        self.brush_size_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(5, 120)
        self.brush_size_slider.setValue(25)
        self.brush_size_slider.setMaximumWidth(60)
        self.brush_size_slider.valueChanged.connect(self._on_brush_config_changed)
        self.brush_strength_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.brush_strength_slider.setRange(10, 100)
        self.brush_strength_slider.setValue(85)
        self.brush_strength_slider.setMaximumWidth(58)
        self.brush_strength_slider.valueChanged.connect(self._on_brush_config_changed)
        self.brush_soft_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.brush_soft_slider.setRange(10, 100)
        self.brush_soft_slider.setValue(65)
        self.brush_soft_slider.setMaximumWidth(58)
        self.brush_soft_slider.valueChanged.connect(self._on_brush_config_changed)
        undo_brush_btn = QtWidgets.QPushButton("Undo (Ctrl+Z)")
        undo_brush_btn.setToolTip("Undo last brush stroke (Ctrl+Z)")
        undo_brush_btn.clicked.connect(self._on_undo_brush)
        clear_brush_btn = QtWidgets.QPushButton("Clear Brush")
        clear_brush_btn.clicked.connect(self._on_clear_brush)
        self.bake_edits_btn = QtWidgets.QPushButton("Apply / Collapse")
        self.bake_edits_btn.setStyleSheet(
            "QPushButton { background-color: #0e7490; color: #ffffff; font-weight: bold; padding: 3px 8px; }"
            "QPushButton:hover { background-color: #0891b2; }"
        )
        self.bake_edits_btn.setToolTip(
            "Apply & collapse plate stitching, sky/ground/light inpainting, and brush strokes into a fast 32-bit canvas\n"
            "so tweaking Exposure, CCT, Saturation, Blur, etc. becomes instant."
        )
        self.bake_edits_btn.clicked.connect(self._on_bake_edits_to_canvas)
        self.unbake_edits_btn = QtWidgets.QPushButton("Unbake")
        self.unbake_edits_btn.setEnabled(False)
        self.unbake_edits_btn.setToolTip("Return to live multi-plate synthesis pipeline")
        self.unbake_edits_btn.clicked.connect(self._on_unbake_edits)

        brush_row.addWidget(self.brush_btn)
        brush_row.addWidget(self.brush_mode_combo)
        brush_row.addWidget(QtWidgets.QLabel("Size:"))
        brush_row.addWidget(self.brush_size_slider)
        brush_row.addWidget(QtWidgets.QLabel("Strength:"))
        brush_row.addWidget(self.brush_strength_slider)
        brush_row.addWidget(QtWidgets.QLabel("Soft:"))
        brush_row.addWidget(self.brush_soft_slider)
        brush_row.addWidget(undo_brush_btn)
        brush_row.addWidget(clear_brush_btn)
        brush_row.addWidget(self.bake_edits_btn)
        brush_row.addWidget(self.unbake_edits_btn)
        brush_row.addStretch(1)
        layout.addLayout(brush_row)

        self.preview_label.brushCloneSourcePicked.connect(self._on_brush_clone_src_picked)
        self.preview_label.brushStrokeStarted.connect(self._on_brush_stroke_started)
        self.preview_label.brushStrokeMoved.connect(self._on_brush_stroke_moved)
        self.preview_label.brushStrokeFinished.connect(self._on_brush_stroke_finished)
        self.preview_label.brushParamAdjusted.connect(self._on_brush_param_adjusted)

        self.plate_transform_bar = QtWidgets.QWidget()
        pt_layout = QtWidgets.QHBoxLayout(self.plate_transform_bar)
        pt_layout.setContentsMargins(2, 2, 2, 2)
        pt_layout.setSpacing(6)
        self.plate_sel_combo = QtWidgets.QComboBox()
        self.plate_sel_combo.setMinimumWidth(110)
        self.plate_sel_combo.currentIndexChanged.connect(self._on_plate_combo_changed)
        pt_layout.addWidget(QtWidgets.QLabel("Plate:"))
        pt_layout.addWidget(self.plate_sel_combo)

        def _mk_spin(lo: float, hi: float, val: float, step: float, dec: int = 1) -> QtWidgets.QDoubleSpinBox:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(val)
            sb.setSingleStep(step)
            sb.setDecimals(dec)
            sb.setMaximumWidth(70)
            sb.valueChanged.connect(self._on_plate_spin_changed)
            return sb

        self.pt_az_spin = _mk_spin(-360.0, 360.0, 0.0, 1.0, 1)
        self.pt_el_spin = _mk_spin(-89.0, 89.0, 0.0, 1.0, 1)
        self.pt_roll_spin = _mk_spin(-180.0, 180.0, 0.0, 1.0, 1)
        self.pt_fov_spin = _mk_spin(10.0, 165.0, 90.0, 1.0, 1)
        self.pt_sx_spin = _mk_spin(0.3, 3.0, 1.0, 0.02, 2)
        self.pt_sy_spin = _mk_spin(0.3, 3.0, 1.0, 0.02, 2)
        self.pt_warp_spin = _mk_spin(-0.45, 0.45, 0.0, 0.01, 2)
        self.pt_ev_spin = _mk_spin(-4.0, 4.0, 0.0, 0.05, 2)
        self.pt_vig_spin = _mk_spin(-0.5, 1.5, 0.0, 0.05, 2)
        self.pt_lens_combo = QtWidgets.QComboBox()
        self.pt_lens_combo.addItems(["rectilinear", "fisheye_equidistant", "fisheye_equisolid", "chrome_ball"])
        self.pt_lens_combo.currentIndexChanged.connect(self._on_plate_spin_changed)
        self.pt_auto_vig_btn = QtWidgets.QPushButton("Auto Vig")
        self.pt_auto_vig_btn.setToolTip("Estimate radial corner vignetting for the selected plate")
        self.pt_auto_vig_btn.clicked.connect(self._on_auto_vignette_plate)
        self.pt_mask_btn = QtWidgets.QPushButton("Mask Area")
        self.pt_mask_btn.setCheckable(True)
        self.pt_mask_btn.setToolTip("Drag a box on the selected plate in the viewer to exclude tripod/moving objects")
        self.pt_mask_btn.toggled.connect(self._on_toggle_plate_mask_mode)
        self.pt_clear_mask_btn = QtWidgets.QPushButton("Clear Masks")
        self.pt_clear_mask_btn.setToolTip("Clear all excluded mask regions on the selected plate")
        self.pt_clear_mask_btn.clicked.connect(self._on_clear_plate_masks)
        self.pt_reset_btn = QtWidgets.QPushButton("Reset")
        self.pt_reset_btn.setToolTip("Reset Roll, Scale X/Y, Lens Warp, EV, and Vignette for selected plate")
        self.pt_reset_btn.clicked.connect(self._on_reset_plate_transform)

        for lbl, w in (
            ("Az°:", self.pt_az_spin),
            ("El°:", self.pt_el_spin),
            ("Roll°:", self.pt_roll_spin),
            ("FOV°:", self.pt_fov_spin),
            ("ScaleX:", self.pt_sx_spin),
            ("ScaleY:", self.pt_sy_spin),
            ("Warp:", self.pt_warp_spin),
            ("EV:", self.pt_ev_spin),
            ("Vig:", self.pt_vig_spin),
        ):
            pt_layout.addWidget(QtWidgets.QLabel(lbl))
            pt_layout.addWidget(w)
        pt_layout.addWidget(self.pt_lens_combo)
        pt_layout.addWidget(self.pt_auto_vig_btn)
        pt_layout.addWidget(self.pt_mask_btn)
        pt_layout.addWidget(self.pt_clear_mask_btn)
        pt_layout.addWidget(self.pt_reset_btn)
        pt_layout.addStretch(1)
        self.plate_transform_bar.setVisible(False)
        layout.addWidget(self.plate_transform_bar)

        light_row = QtWidgets.QHBoxLayout()
        light_row.addWidget(QtWidgets.QLabel("Light Area Blur:"))
        self.light_blur_slider = SliderResetControl(0.0, 1.0, 0.0, 0.01, decimals=2)
        self.light_blur_slider.setToolTip("Smoothly blur the picked Light Area region(s) in the HDRI")
        self.light_blur_slider.valueChanged.connect(self._schedule_preview_update)
        light_row.addWidget(self.light_blur_slider, stretch=1)
        light_row.addSpacing(12)
        light_row.addWidget(QtWidgets.QLabel("Light Tex Res:"))
        self.light_res_combo = QtWidgets.QComboBox()
        self.light_res_combo.addItems(["256 x 256", "512 x 512", "1024 x 1024", "2048 x 2048"])
        self.light_res_combo.setCurrentIndex(1)
        self.light_res_combo.setToolTip("Output texture resolution for extracted Solaris Area Light EXR(s), independent of Dome HDRI resolution")
        light_row.addWidget(self.light_res_combo)
        layout.addLayout(light_row)

        return grp

    def _build_solaris_group(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("4. Export 32-bit EXR & Solaris (/stage) Wiring")
        layout = QtWidgets.QVBoxLayout(grp)

        out_row = QtWidgets.QHBoxLayout()
        default_out = "$HIP/tex/hdri_maker.exr" if HAS_HOU else os.path.abspath("hdri_maker.exr")
        self.out_path_edit = QtWidgets.QLineEdit(default_out)
        out_browse = QtWidgets.QPushButton("Browse...")
        out_browse.clicked.connect(self._on_browse_output)
        out_row.addWidget(QtWidgets.QLabel("Output EXR:"))
        out_row.addWidget(self.out_path_edit)
        out_row.addWidget(out_browse)
        layout.addLayout(out_row)

        self.lookdev_chk = QtWidgets.QCheckBox(
            "Create Lookdev Tripod (Chrome/18% Gray/ACEScg Macbeth) + Circular Ground Projection (/stage)"
        )
        self.lookdev_chk.setChecked(True)
        layout.addWidget(self.lookdev_chk)

        lookdev_form = QtWidgets.QFormLayout()
        self.tripod_y_slider = SliderResetControl(0.2, 10.0, 1.5, 0.05, decimals=2)
        self.tripod_y_slider.setToolTip("Tripod Y height (meters) used for the Lookdev Tripod and ground projection camera")
        self.ground_radius_slider = SliderResetControl(1.0, 100.0, 10.0, 0.5, decimals=1)
        self.ground_radius_slider.setToolTip("Radius (meters) of the circular ground projection disc")
        self.ground_proj_blur_slider = SliderResetControl(0.0, 1.0, 0.0, 0.01, decimals=2)
        self.ground_proj_blur_slider.setToolTip("Blur the projected HDRI ground texture")
        self.ground_feather_slider = SliderResetControl(0.0, 1.0, 0.45, 0.01, decimals=2)
        self.ground_feather_slider.setToolTip("Radial opacity feather width for the circular ground projection (0 = sharp edge, 1 = full radius fade)")
        self.ground_emission_slider = SliderResetControl(0.0, 2.0, 0.5, 0.05, decimals=2)
        self.ground_emission_slider.setToolTip("Self-illumination / emission multiplier for the projected ground texture (helps match HDRI ground brightness)")
        self.render_engine_combo = QtWidgets.QComboBox()
        self.render_engine_combo.addItems([
            "Karma (MaterialX)",
            "Redshift",
            "Arnold",
        ])
        self.render_engine_combo.setToolTip(
            "Target Solaris render engine for the Material Library (/stage/hdri_maker_materials).\n"
            "Creates native shaders for the selected renderer and assigns them to the Lookdev Tripod and Ground Disc."
        )
        self.render_engine_combo.currentIndexChanged.connect(self._on_render_engine_changed)
        self.ground_shader_combo = QtWidgets.QComboBox()
        self.ground_shader_combo.addItems([
            "Emissive + Diffuse",
            "Emissive Only (No Diffuse)",
            "Diffuse Only (No Emission)",
            "Shadow Catcher (Matte)",
        ])
        self.ground_shader_combo.setToolTip(
            "Ground projection shader mode for Karma, Arnold, Redshift, and PreviewSurface:\n"
            "- Emissive + Diffuse: Self-illuminated + receives scene lighting/shadows\n"
            "- Emissive Only (No Diffuse): Pure unlit projection matching the HDRI (no double-lighting)\n"
            "- Diffuse Only (No Emission): Lit only by scene lights/DomeLight\n"
            "- Shadow Catcher (Matte): Configures Karma/Redshift/Arnold matte shadow catcher primvars"
        )
        lookdev_form.addRow("Tripod Y Height:", self.tripod_y_slider)
        lookdev_form.addRow("Ground Radius:", self.ground_radius_slider)
        lookdev_form.addRow("Ground Proj Blur:", self.ground_proj_blur_slider)
        lookdev_form.addRow("Ground Feather:", self.ground_feather_slider)
        lookdev_form.addRow("Ground Emission:", self.ground_emission_slider)
        lookdev_form.addRow("Render Engine:", self.render_engine_combo)
        lookdev_form.addRow("Ground Shader:", self.ground_shader_combo)
        self.diffuse_split_chk = QtWidgets.QCheckBox("Export Companion Diffuse EXR (_diffuse.exr)")
        self.diffuse_split_chk.setToolTip("Write a firefly-free clamped & smoothed diffuse/irradiance EXR alongside the full-range specular EXR")
        self.diffuse_split_chk.setChecked(False)
        lookdev_form.addRow("", self.diffuse_split_chk)
        layout.addLayout(lookdev_form)

        btn_row = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Generate 32-bit Float EXR")
        self.export_btn.setMinimumHeight(34)
        self.export_btn.clicked.connect(self._on_export_exr)

        self.dome_btn = QtWidgets.QPushButton("Create / Update Solaris Dome Light")
        self.dome_btn.setMinimumHeight(34)
        self.dome_btn.clicked.connect(self._on_create_domelight)

        self.rig_btn = QtWidgets.QPushButton("Create / Update Area Light (/stage)")
        self.rig_btn.setMinimumHeight(34)
        self.rig_btn.clicked.connect(self._on_create_hotspot_rig)

        self.sun_btn = QtWidgets.QPushButton("Create / Update Sun Light (/stage)")
        self.sun_btn.setMinimumHeight(34)
        self.sun_btn.setToolTip("Extract Hotspot #1 as a Solaris Distant Light (Sun) aimed toward origin")
        self.sun_btn.clicked.connect(self._on_create_sun_light)

        self.update_lookdev_btn = QtWidgets.QPushButton("Update Lookdev / Ground (/stage)")
        self.update_lookdev_btn.setMinimumHeight(34)
        self.update_lookdev_btn.clicked.connect(self._on_update_lookdev_clicked)

        self.export_seq_btn = QtWidgets.QPushButton("Export Sequence ($F4)")
        self.export_seq_btn.setMinimumHeight(34)
        self.export_seq_btn.setToolTip("Export per-frame 32-bit float Lat/Long EXRs for an analyzed sequence and set $F4 path")
        self.export_seq_btn.clicked.connect(self._on_export_sequence_exr)

        self.export_usd_btn = QtWidgets.QPushButton("Export USD Rig (.usda)")
        self.export_usd_btn.setMinimumHeight(34)
        self.export_usd_btn.setToolTip("Export the active /stage HDRI Maker dome, lights, and lookdev rig to a standalone .usda/.usd sublayer")
        self.export_usd_btn.clicked.connect(self._on_export_usd_rig)

        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.export_seq_btn)
        btn_row.addWidget(self.dome_btn)
        btn_row.addWidget(self.rig_btn)
        btn_row.addWidget(self.sun_btn)
        btn_row.addWidget(self.update_lookdev_btn)
        btn_row.addWidget(self.export_usd_btn)
        layout.addLayout(btn_row)

        return grp

    @staticmethod
    def _make_dspin(lo: float, hi: float, val: float, step: float) -> QtWidgets.QDoubleSpinBox:
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setSingleStep(step)
        sb.setDecimals(2)
        return sb

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _maybe_update_default_output_path(self, plate_path: str) -> None:
        """
        If the user hasn't customized Output EXR and the Houdini scene is still
        unsaved (untitled.hip where $HIP == $HOME), default the output EXR next
        to the loaded plate directory instead of C:/Users/<user>/tex.
        """
        curr = self.out_path_edit.text().strip()
        is_default = curr in ("$HIP/tex/hdri_maker.exr", "") or curr.endswith("hdri_maker.exr") and os.path.dirname(curr) in ("", os.getcwd())
        if not is_default:
            return
        is_unsaved = True
        if HAS_HOU:
            try:
                is_unsaved = hou.hipFile.basename().lower() == "untitled.hip"
            except Exception:
                pass
        if is_unsaved and plate_path:
            plate_dir = os.path.dirname(os.path.abspath(plate_path))
            if plate_dir:
                self.out_path_edit.setText(os.path.join(plate_dir, "hdri_maker.exr").replace("\\", "/"))

    def _on_browse_plate(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Plate Image or Sequence Frame",
            "",
            "Images (*.exr *.hdr *.jpg *.jpeg *.png *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.plate_path_edit.setText(path)
            self._maybe_update_default_output_path(path)

    def _make_plate_thumbnail(
        self, path: str, cs: str = "Auto", thumb_w: int = 68, thumb_h: int = 44
    ) -> QtGui.QIcon:
        """Load and cache a compact letterboxed QIcon thumbnail for a plate file."""
        key = (path, cs, thumb_w, thumb_h)
        if key in self._thumb_cache:
            return self._thumb_cache[key]

        icon = QtGui.QIcon()
        try:
            target = path
            if any(tok in path for tok in ("$F", "#")):
                seq = resolve_plate_sequence(path)
                if seq:
                    target = seq[0]
            if not target or not os.path.isfile(target):
                self._thumb_cache[key] = icon
                return icon

            ext = os.path.splitext(target)[1].lower()
            src_pix: Optional[QtGui.QPixmap] = None
            if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
                reader = QtGui.QImageReader(target)
                sz = reader.size()
                if sz.isValid() and sz.width() > 0 and sz.height() > 0:
                    scale = min(thumb_w / float(sz.width()), thumb_h / float(sz.height()))
                    sw = max(1, int(round(sz.width() * scale)))
                    sh = max(1, int(round(sz.height() * scale)))
                    reader.setScaledSize(QtCore.QSize(sw, sh))
                    img = reader.read()
                    if not img.isNull():
                        src_pix = QtGui.QPixmap.fromImage(img)

            if src_pix is None:
                plate = read_plate(target, colorspace=cs, max_dim=160)
                srgb = render_preview_srgb(plate.pixels, preview_ev=0.0, tonemap="aces")
                h, w, _ = srgb.shape
                qimg = QtGui.QImage(srgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
                src_pix = QtGui.QPixmap.fromImage(qimg.copy()).scaled(
                    thumb_w,
                    thumb_h,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )

            canvas = QtGui.QPixmap(thumb_w, thumb_h)
            canvas.fill(QtGui.QColor(24, 24, 28))
            painter = QtGui.QPainter(canvas)
            ox = (thumb_w - src_pix.width()) // 2
            oy = (thumb_h - src_pix.height()) // 2
            painter.drawPixmap(ox, oy, src_pix)
            painter.end()
            icon = QtGui.QIcon(canvas)
        except Exception:
            icon = QtGui.QIcon()

        self._thumb_cache[key] = icon
        return icon

    def _add_multiview_row(
        self,
        path: str,
        az: float,
        el: float,
        fov: float,
        cs: str,
        roll: float = 0.0,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        warp_k1: float = 0.0,
        lens_type: str = "rectilinear",
        exposure_ev: float = 0.0,
        vignette_v1: float = 0.0,
        exclude_rects: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> None:
        row = self.multiview_table.rowCount()
        self.multiview_table.insertRow(row)

        file_item = QtWidgets.QTableWidgetItem(os.path.basename(path))
        file_item.setData(Qt.UserRole, path)
        file_item.setData(
            Qt.UserRole + 1,
            {
                "roll": float(roll),
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "warp_k1": float(warp_k1),
                "lens_type": str(lens_type or "rectilinear"),
                "exposure_ev": float(exposure_ev),
                "vignette_v1": float(vignette_v1),
                "exclude_rects": [list(r) for r in (exclude_rects or [])],
            },
        )
        file_item.setToolTip(path)
        file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
        thumb_icon = self._make_plate_thumbnail(path, cs)
        if not thumb_icon.isNull():
            file_item.setIcon(thumb_icon)
        self.multiview_table.setItem(row, 0, file_item)

        for col, val in ((1, f"{az:.1f}"), (2, f"{el:.1f}"), (3, f"{fov:.1f}")):
            item = QtWidgets.QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.multiview_table.setItem(row, col, item)

        cs_item = QtWidgets.QTableWidgetItem(cs)
        cs_item.setTextAlignment(Qt.AlignCenter)
        self.multiview_table.setItem(row, 4, cs_item)

        rm_btn = QtWidgets.QToolButton()
        rm_btn.setText("×")
        rm_btn.setToolTip("Remove this plate from the list")
        rm_btn.setAutoRaise(True)
        rm_btn.clicked.connect(lambda _checked=False, b=rm_btn: self._remove_multiview_row_by_button(b))
        self.multiview_table.setCellWidget(row, 5, rm_btn)

    def _remove_multiview_row_by_button(self, btn: QtWidgets.QToolButton) -> None:
        for r in range(self.multiview_table.rowCount()):
            if self.multiview_table.cellWidget(r, 5) is btn:
                self.multiview_table.removeRow(r)
                self._sync_after_multiview_list_change(removed_indices=[r])
                self.status_label.setText(
                    f"Removed plate ({self.multiview_table.rowCount()} remaining)."
                )
                return

    def _on_remove_selected_multiview_plates(self) -> None:
        rows = sorted(
            {idx.row() for idx in self.multiview_table.selectedIndexes()},
            reverse=True,
        )
        if not rows:
            curr = self.multiview_table.currentRow()
            if curr >= 0:
                rows = [curr]
        if not rows:
            self.status_label.setText("Select plate row(s) to remove.")
            return
        for r in rows:
            self.multiview_table.removeRow(r)
        self._sync_after_multiview_list_change(removed_indices=rows)
        self.status_label.setText(
            f"Removed {len(rows)} plate(s) ({self.multiview_table.rowCount()} remaining)."
        )

    def _sync_after_multiview_list_change(self, removed_indices: Optional[List[int]] = None) -> None:
        """Update cached views, lighting profile, and preview canvas after removing/clearing plates."""
        n_rows = self.multiview_table.rowCount()
        self._baked_canvas_hdri = None
        if hasattr(self, "unbake_edits_btn"):
            self.unbake_edits_btn.setEnabled(False)
            self.bake_edits_btn.setText("Apply / Collapse")
        if n_rows == 0:
            self._views = []
            self._profile = None
            self._frame_profiles = []
            self._hdri_pixels = None
            self._raw_hdri_pixels = None
            self._rough_cache = None
            self._raw_rough_cache = None
            self._selected_plate_idx = -1
            self._refresh_plate_transform_bar()
            self.hotspot_table.setRowCount(0)
            self.preview_label.set_ab_split(False)
            self.preview_label.set_plate_overlay([], -1, self.yaw_spin.value(), self.vshift_spin.value())
            self.preview_label.set_source_pixmap(None)
            self.preview_label.setText("Analyze a plate or multi-view set to preview the 2:1 Lat/Long HDRI.")
            return

        if removed_indices and self._views:
            for r in sorted(set(removed_indices), reverse=True):
                if 0 <= r < len(self._views):
                    self._views.pop(r)

        if len(self._views) == n_rows and self._views:
            self._selected_plate_idx = min(max(0, self._selected_plate_idx), len(self._views) - 1)
            self._refresh_plate_transform_bar()
            self._profile = analyze_multiview_plates(
                self._views,
                max_hotspots=self.max_hotspots_spin.value(),
                ldr_highlight_boost=self.ldr_boost_spin.value(),
            )
            self._populate_analysis_ui(self._profile, len(self._views))
            self._raw_hdri_pixels = None
            self._regenerate_preview()
        elif self._profile is not None or self._hdri_pixels is not None:
            self._on_analyze_clicked()

    def _on_multiview_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self.multiview_table)
        act_add = menu.addAction("Add Plates...")
        act_rm = menu.addAction("Remove Selected Plate(s)")
        act_use = menu.addAction("Set as Single Plate Path")
        menu.addSeparator()
        act_clear = menu.addAction("Clear All")

        has_sel = len(self.multiview_table.selectedIndexes()) > 0 or self.multiview_table.currentRow() >= 0
        act_rm.setEnabled(has_sel)
        act_use.setEnabled(has_sel)
        act_clear.setEnabled(self.multiview_table.rowCount() > 0)

        chosen = menu.exec_(self.multiview_table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_add:
            self._on_add_multiview_plates()
        elif chosen == act_rm:
            self._on_remove_selected_multiview_plates()
        elif chosen == act_use:
            row = self.multiview_table.currentRow()
            if row >= 0:
                item = self.multiview_table.item(row, 0)
                if item is not None:
                    self.plate_path_edit.setText(str(item.data(Qt.UserRole) or ""))
        elif chosen == act_clear:
            self._on_clear_multiview_plates()

    def _on_add_multiview_plates(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Multiple Plate Images (Different Positions)",
            "",
            "Images (*.exr *.hdr *.jpg *.jpeg *.png *.tif *.tiff);;All Files (*)",
        )
        if not paths:
            return
        start_idx = self.multiview_table.rowCount()
        total = start_idx + len(paths)
        fov = self.fov_spin.value()
        el = self.cam_el_spin.value()
        cs = self.colorspace_combo.currentText()
        for i, p in enumerate(paths):
            # Suggest evenly spaced azimuths if adding a batch
            az = ((start_idx + i) * (360.0 / max(total, 1)))
            if az > 180.0:
                az -= 360.0
            meta = read_plate_metadata(p)
            plate_fov = meta["fov_h_deg"] if meta.get("fov_h_deg") is not None else fov
            plate_roll = float(meta.get("roll_deg", 0.0))
            self._add_multiview_row(p, az, el, plate_fov, cs, roll=plate_roll)
        self._maybe_update_default_output_path(paths[0])
        self.status_label.setText(f"Added {len(paths)} plate(s) to multi-view list.")

    def _on_add_current_plate(self) -> None:
        p = self.plate_path_edit.text().strip()
        if not p:
            return
        meta = read_plate_metadata(p)
        plate_fov = meta["fov_h_deg"] if meta.get("fov_h_deg") is not None else self.fov_spin.value()
        plate_roll = float(meta.get("roll_deg", 0.0))
        self._add_multiview_row(
            p,
            self.cam_az_spin.value(),
            self.cam_el_spin.value(),
            plate_fov,
            self.colorspace_combo.currentText(),
            roll=plate_roll,
        )

    def _on_auto_distribute_360(self) -> None:
        n = self.multiview_table.rowCount()
        if n == 0:
            return
        for i in range(n):
            az = i * (360.0 / float(n))
            if az > 180.0:
                az -= 360.0
            item = self.multiview_table.item(i, 1)
            if item is not None:
                item.setText(f"{az:.1f}")

    def _on_clear_multiview_plates(self) -> None:
        self.multiview_table.setRowCount(0)
        self._sync_after_multiview_list_change()
        self.status_label.setText("Cleared all multi-position plates.")

    def _on_auto_align_plates(self) -> None:
        n_mv = self.multiview_table.rowCount()
        if n_mv < 2:
            self.status_label.setText("Add at least 2 overlapping plates to auto-align.")
            return
        try:
            views: List[PlateViewInput] = []
            for row in range(n_mv):
                fpath = self.multiview_table.item(row, 0).data(Qt.UserRole)
                az = float(self.multiview_table.item(row, 1).text())
                el = float(self.multiview_table.item(row, 2).text())
                fov = float(self.multiview_table.item(row, 3).text())
                cs = self.multiview_table.item(row, 4).text() or "Auto"
                plate = read_plate(fpath, colorspace=cs, max_dim=512)
                views.append(
                    PlateViewInput(
                        pixels=plate.pixels,
                        is_hdr=plate.is_hdr,
                        source_path=fpath,
                        fov_h_deg=fov,
                        cam_azimuth_deg=az,
                        cam_elevation_deg=el,
                    )
                )
            aligned = auto_align_adjacent_plates(views)
            for row, v in enumerate(aligned):
                self.multiview_table.item(row, 1).setText(f"{v.cam_azimuth_deg:.1f}")
                self.multiview_table.item(row, 2).setText(f"{v.cam_elevation_deg:.1f}")
                self.multiview_table.item(row, 3).setText(f"{v.fov_h_deg:.1f}")
                item0 = self.multiview_table.item(row, 0)
                if item0 is not None:
                    extra = dict(item0.data(Qt.UserRole + 1) or {})
                    extra["roll"] = float(v.cam_roll_deg)
                    extra["warp_k1"] = float(v.warp_k1)
                    item0.setData(Qt.UserRole + 1, extra)
            if self._views and len(self._views) == len(aligned):
                for idx, v in enumerate(aligned):
                    self._views[idx].cam_azimuth_deg = v.cam_azimuth_deg
                    self._views[idx].cam_elevation_deg = v.cam_elevation_deg
                    self._views[idx].cam_roll_deg = v.cam_roll_deg
                    self._views[idx].fov_h_deg = v.fov_h_deg
                    self._views[idx].warp_k1 = v.warp_k1
                self._refresh_plate_transform_bar()
                self._regenerate_preview()
            self.status_label.setText(f"Auto-aligned {len(aligned)} overlapping plates (Az, El, Roll, FOV, Warp).")
        except Exception as exc:
            self.status_label.setText(f"Auto-align error: {exc}")

    def _on_equalize_plate_exposures(self) -> None:
        if len(self._views) < 2:
            self.status_label.setText("Analyze at least 2 overlapping plates before equalizing EV.")
            return
        try:
            eq = auto_equalize_plate_exposures(self._views)
            for idx, v in enumerate(eq):
                self._views[idx].exposure_ev = v.exposure_ev
                self._sync_view_to_table_and_bar(idx)
            self._regenerate_preview()
            self.status_label.setText(f"Equalized exposure across {len(eq)} overlapping plates.")
        except Exception as exc:
            self._log_debug(f"Equalize EV error: {exc}\n{traceback.format_exc()}")
            self.status_label.setText(f"Equalize EV error: {exc}")

    def _on_browse_output(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save 32-bit Float Lat/Long EXR",
            self.out_path_edit.text(),
            "OpenEXR (*.exr)",
        )
        if path:
            if not path.lower().endswith(".exr"):
                path += ".exr"
            self.out_path_edit.setText(path)

    def _current_synthesis_params(self) -> HDRISynthesisParams:
        res_text = self.res_combo.currentText()
        w_str, h_str = res_text.split("x")
        return HDRISynthesisParams(
            width=int(w_str.strip()),
            height=int(h_str.strip()),
            exposure_ev=self.ev_spin.value(),
            hotspot_boost=self.hs_boost_spin.value(),
            hotspot_softness=self.hs_soft_spin.value(),
            sky_gain=self.sky_gain_spin.value(),
            ground_gain=self.gnd_gain_spin.value(),
            ambient_wrap=self.wrap_spin.value(),
            temperature_kelvin=self.temp_k_spin.value(),
            temperature_blend=self.temp_blend_spin.value(),
            wb_gains=self._wb_gains,
            saturation=self.sat_spin.value(),
            plate_projection_blend=self.plate_blend_spin.value(),
            ldr_highlight_boost=self.ldr_boost_spin.value(),
            azimuth_offset_deg=self.yaw_spin.value(),
            vertical_shift_deg=self.vshift_spin.value(),
            stitch_blend=self.stitch_blend_spin.value(),
            enable_inpaint=self.inpaint_chk.isChecked(),
            sky_blend=self.sky_blend_slider.value(),
            ground_blend=self.gnd_blend_slider.value(),
            sky_blur=self.sky_blur_slider.value(),
            ground_blur=self.gnd_blur_slider.value(),
            hdri_blur=self.hdri_blur_slider.value(),
            sky_rect=self._sky_rect,
            ground_rect=self._ground_rect,
            light_rect=self._light_regions[0]["rect"] if self._light_regions else None,
            remove_hotspot_from_hdri=self.remove_hotspot_chk.isChecked(),
            physical_sky_fill=self.sky_fill_slider.value(),
            highlight_recovery=self.highlight_recovery_slider.value(),
            nadir_cleanup=self.nadir_cleanup_slider.value(),
            neutralize_environment=self.neutralize_env_chk.isChecked(),
            temporal_stabilize=self.temporal_stab_slider.value(),
        )

    def _current_light_shape(self) -> str:
        return "circle" if self.light_shape_combo.currentText().lower() == "circle" else "rect"

    def _current_light_tex_size(self) -> tuple:
        txt = self.light_res_combo.currentText()
        w_str, h_str = txt.split("x")
        return int(w_str.strip()), int(h_str.strip())

    def _on_light_shape_changed(self) -> None:
        if self.pick_light_btn.isChecked():
            self.preview_label.set_pick_mode("light", shape=self._current_light_shape())

    def _toggle_pick_mode(self, mode: str, checked: bool) -> None:
        if checked:
            if hasattr(self, "edit_plates_btn") and self.edit_plates_btn.isChecked():
                self.edit_plates_btn.setChecked(False)
            if mode != "sky":
                self.pick_sky_btn.setChecked(False)
            if mode != "ground":
                self.pick_gnd_btn.setChecked(False)
            if mode != "light":
                self.pick_light_btn.setChecked(False)
            if mode != "wb" and hasattr(self, "pick_wb_btn"):
                self.pick_wb_btn.setChecked(False)
            shape = self._current_light_shape() if mode == "light" else "rect"
            self.preview_label.set_pick_mode(mode, shape=shape)
            if mode == "light":
                self.status_label.setText(
                    f"Drag ({shape.upper()}) on preview to add Area Light #{len(self._light_regions) + 1}."
                )
            elif mode == "wb":
                self.status_label.setText(
                    "Drag a box over a neutral gray or Macbeth patch to auto-calibrate White Balance & Exposure."
                )
            else:
                self.status_label.setText(f"Drag a box on the preview to select the {mode.upper()} area.")
        else:
            wb_on = hasattr(self, "pick_wb_btn") and self.pick_wb_btn.isChecked()
            if not (self.pick_sky_btn.isChecked() or self.pick_gnd_btn.isChecked() or self.pick_light_btn.isChecked() or wb_on):
                self.preview_label.set_pick_mode(None)

    def _toggle_plate_edit_mode(self, checked: bool) -> None:
        self._log_debug(f"Move/Align Plates toggled: {checked} (views={len(self._views)})")
        if checked:
            for btn in (self.pick_sky_btn, self.pick_gnd_btn, self.pick_light_btn):
                btn.setChecked(False)
            if hasattr(self, "pick_wb_btn"):
                self.pick_wb_btn.setChecked(False)
            if hasattr(self, "pt_mask_btn"):
                self.pt_mask_btn.setChecked(False)
            self.preview_label.set_pick_mode(None)
            if self._views and not (0 <= self._selected_plate_idx < len(self._views)):
                self._selected_plate_idx = 0
            self._refresh_plate_transform_bar()
            self.status_label.setText(
                "Move / Align Plates: Click plate to select | Drag=Az/El | Shift+Drag or orange dot=Roll | "
                "Corner handle or Shift+Wheel=FOV Scale | Ctrl+Drag=Lens Warp"
            )
        self.plate_transform_bar.setVisible(bool(checked))
        self.preview_label.set_plate_edit_mode(checked)
        self._update_preview_pixmap()

    def _refresh_plate_transform_bar(self) -> None:
        self._updating_plate_ui = True
        try:
            self.plate_sel_combo.clear()
            for i, v in enumerate(self._views):
                name = os.path.basename(v.source_path) if v.source_path else f"Plate #{i + 1}"
                self.plate_sel_combo.addItem(f"#{i + 1}: {name}")
            has_sel = 0 <= self._selected_plate_idx < len(self._views)
            if has_sel:
                self.plate_sel_combo.setCurrentIndex(self._selected_plate_idx)
                v = self._views[self._selected_plate_idx]
                self.pt_az_spin.setValue(float(v.cam_azimuth_deg))
                self.pt_el_spin.setValue(float(v.cam_elevation_deg))
                self.pt_roll_spin.setValue(float(v.cam_roll_deg))
                self.pt_fov_spin.setValue(float(v.fov_h_deg))
                self.pt_sx_spin.setValue(float(v.scale_x))
                self.pt_sy_spin.setValue(float(v.scale_y))
                self.pt_warp_spin.setValue(float(v.warp_k1))
                self.pt_ev_spin.setValue(float(v.exposure_ev))
                self.pt_vig_spin.setValue(float(v.vignette_v1))
                self.pt_lens_combo.setCurrentText(str(v.lens_type or "rectilinear"))
            for w in (
                self.pt_az_spin,
                self.pt_el_spin,
                self.pt_roll_spin,
                self.pt_fov_spin,
                self.pt_sx_spin,
                self.pt_sy_spin,
                self.pt_warp_spin,
                self.pt_ev_spin,
                self.pt_vig_spin,
                self.pt_lens_combo,
                self.pt_mask_btn,
                self.pt_clear_mask_btn,
                self.pt_reset_btn,
            ):
                w.setEnabled(has_sel)
        finally:
            self._updating_plate_ui = False

    def _sync_view_to_table_and_bar(self, idx: int) -> None:
        if not (0 <= idx < len(self._views)):
            return
        v = self._views[idx]
        if idx < self.multiview_table.rowCount():
            item0 = self.multiview_table.item(idx, 0)
            if item0 is not None:
                item0.setData(
                    Qt.UserRole + 1,
                    {
                        "roll": float(v.cam_roll_deg),
                        "scale_x": float(v.scale_x),
                        "scale_y": float(v.scale_y),
                        "warp_k1": float(v.warp_k1),
                        "lens_type": str(v.lens_type or "rectilinear"),
                        "exposure_ev": float(v.exposure_ev),
                        "vignette_v1": float(v.vignette_v1),
                        "exclude_rects": [list(r) for r in (v.exclude_rects or [])],
                    },
                )
            for col, text in (
                (1, f"{v.cam_azimuth_deg:.1f}"),
                (2, f"{v.cam_elevation_deg:.1f}"),
                (3, f"{v.fov_h_deg:.1f}"),
            ):
                it = self.multiview_table.item(idx, col)
                if it is not None:
                    it.setText(text)
        elif len(self._views) == 1 and idx == 0:
            self.cam_az_spin.setValue(float(v.cam_azimuth_deg))
            self.cam_el_spin.setValue(float(v.cam_elevation_deg))
            self.fov_spin.setValue(float(v.fov_h_deg))
        self._refresh_plate_transform_bar()

    def _on_plate_combo_changed(self, idx: int) -> None:
        if self._updating_plate_ui:
            return
        self._selected_plate_idx = int(idx)
        self._refresh_plate_transform_bar()
        self._update_preview_pixmap()

    def _on_plate_spin_changed(self) -> None:
        if self._updating_plate_ui:
            return
        idx = self._selected_plate_idx
        if not (0 <= idx < len(self._views)):
            return
        v = self._views[idx]
        v.cam_azimuth_deg = float(self.pt_az_spin.value())
        v.cam_elevation_deg = float(self.pt_el_spin.value())
        v.cam_roll_deg = float(self.pt_roll_spin.value())
        v.fov_h_deg = float(self.pt_fov_spin.value())
        v.scale_x = float(self.pt_sx_spin.value())
        v.scale_y = float(self.pt_sy_spin.value())
        v.warp_k1 = float(self.pt_warp_spin.value())
        v.exposure_ev = float(self.pt_ev_spin.value())
        v.vignette_v1 = float(self.pt_vig_spin.value())
        v.lens_type = self.pt_lens_combo.currentText()
        self._sync_view_to_table_and_bar(idx)
        self._schedule_preview_update()

    def _on_reset_plate_transform(self) -> None:
        idx = self._selected_plate_idx
        if not (0 <= idx < len(self._views)):
            return
        v = self._views[idx]
        v.cam_roll_deg = 0.0
        v.scale_x = 1.0
        v.scale_y = 1.0
        v.warp_k1 = 0.0
        v.exposure_ev = 0.0
        v.vignette_v1 = 0.0
        v.lens_type = "rectilinear"
        self._sync_view_to_table_and_bar(idx)
        self._regenerate_preview(fast=False)

    def _on_toggle_plate_mask_mode(self, checked: bool) -> None:
        if checked:
            self.preview_label.set_pick_mode("plate_mask", shape="rect")
            self.status_label.setText(
                f"Drag a box over Plate #{self._selected_plate_idx + 1} in the viewer to exclude that region."
            )
        else:
            self.preview_label.set_pick_mode(None)

    def _on_clear_plate_masks(self) -> None:
        idx = self._selected_plate_idx
        if not (0 <= idx < len(self._views)):
            return
        self._views[idx].exclude_rects = []
        self._sync_view_to_table_and_bar(idx)
        self._regenerate_preview(fast=False)
        self.status_label.setText(f"Cleared exclude masks on Plate #{idx + 1}.")

    def _on_viewer_plate_selected(self, idx: int) -> None:
        self._selected_plate_idx = int(idx)
        self._refresh_plate_transform_bar()
        self._update_preview_pixmap()

    def _on_viewer_plate_dragged(self, idx: int, du: float, dv: float, mode: str) -> None:
        if not (0 <= idx < len(self._views)):
            return
        v = self._views[idx]
        if mode == "move":
            az = (v.cam_azimuth_deg + du * 360.0 + 180.0) % 360.0 - 180.0
            el = float(np.clip(v.cam_elevation_deg - dv * 180.0, -89.0, 89.0))
            v.cam_azimuth_deg = az
            v.cam_elevation_deg = el
        elif mode == "roll":
            roll = (v.cam_roll_deg + du * 240.0 + 180.0) % 360.0 - 180.0
            v.cam_roll_deg = roll
        elif mode == "scale":
            v.fov_h_deg = float(np.clip(v.fov_h_deg + (du - dv) * 140.0, 10.0, 165.0))
        elif mode == "warp":
            v.warp_k1 = float(np.clip(v.warp_k1 + du * 1.0, -0.45, 0.45))
        self._refresh_plate_transform_bar()
        self._regenerate_preview(fast=True)

    def _on_viewer_plate_drag_finished(self, idx: int) -> None:
        if 0 <= idx < len(self._views):
            v = self._views[idx]
            self._log_debug(
                f"Plate #{idx + 1} aligned: Az={v.cam_azimuth_deg:.1f}, El={v.cam_elevation_deg:.1f}, "
                f"Roll={v.cam_roll_deg:.1f}, FOV={v.fov_h_deg:.1f}, Warp={v.warp_k1:.2f}"
            )
        self._sync_view_to_table_and_bar(idx)
        self._regenerate_preview(fast=False)

    def _on_viewer_plate_wheel_scaled(self, idx: int, delta: float) -> None:
        if not (0 <= idx < len(self._views)):
            return
        v = self._views[idx]
        step = 2.0 if delta > 0 else -2.0
        v.fov_h_deg = float(np.clip(v.fov_h_deg + step, 10.0, 165.0))
        self._sync_view_to_table_and_bar(idx)
        self._schedule_preview_update()

    def _on_undo_light(self) -> None:
        if self._light_regions:
            self._light_regions.pop()
            self._extracted_lights = []
            self._last_light_exr_paths = []
            self._regenerate_preview()
            self.status_label.setText(f"Removed last light. {len(self._light_regions)} light(s) selected.")

    def _clear_inpaint_areas(self) -> None:
        self._sky_rect = None
        self._ground_rect = None
        self._light_regions = []
        self._extracted_lights = []
        self._last_light_exr_paths = []
        self.pick_sky_btn.setChecked(False)
        self.pick_gnd_btn.setChecked(False)
        self.pick_light_btn.setChecked(False)
        self.preview_label.set_pick_mode(None)
        self._regenerate_preview()
        self.status_label.setText("Cleared custom Sky/Ground/Light areas.")

    def _on_preview_rect_selected(self, mode: str, rect: tuple) -> None:
        if mode == "sky":
            self._sky_rect = rect
            self.pick_sky_btn.setChecked(False)
            self.preview_label.set_pick_mode(None)
            self.status_label.setText("Updated SKY selection.")
        elif mode == "ground":
            self._ground_rect = rect
            self.pick_gnd_btn.setChecked(False)
            self.preview_label.set_pick_mode(None)
            self.status_label.setText("Updated GROUND selection.")
        elif mode == "wb":
            if hasattr(self, "pick_wb_btn"):
                self.pick_wb_btn.setChecked(False)
            self.preview_label.set_pick_mode(None)
            self._apply_neutral_wb_patch(rect)
            return
        elif mode == "plate_mask":
            if hasattr(self, "pt_mask_btn"):
                self.pt_mask_btn.setChecked(False)
            self.preview_label.set_pick_mode(None)
            idx = self._selected_plate_idx
            if 0 <= idx < len(self._views):
                mapped = equirect_rect_to_plate_rect(
                    self._views[idx],
                    rect,
                    azimuth_offset_deg=self.yaw_spin.value(),
                    vertical_shift_deg=self.vshift_spin.value(),
                )
                if mapped is not None:
                    self._views[idx].exclude_rects.append(mapped)
                    self._sync_view_to_table_and_bar(idx)
                    self._regenerate_preview(fast=False)
                    self.status_label.setText(
                        f"Added exclude mask to Plate #{idx + 1} ({len(self._views[idx].exclude_rects)} total)."
                    )
            return
        else:
            shape = self._current_light_shape()
            self._light_regions.append({"rect": rect, "shape": shape})
            self._last_light_exr_paths = []
            self.status_label.setText(
                f"Added Light #{len(self._light_regions)} ({shape.upper()}). Drag another or uncheck Pick Light Area."
            )
        self.preview_label.set_regions(self._sky_rect, self._ground_rect, self._light_regions)
        self._regenerate_preview()

    def _on_preview_region_edited(self, kind: str, idx: int, new_rect: tuple) -> None:
        if kind == "sky":
            self._sky_rect = new_rect
        elif kind == "ground":
            self._ground_rect = new_rect
        elif kind == "light" and 0 <= idx < len(self._light_regions):
            self._light_regions[idx]["rect"] = new_rect
            self._last_light_exr_paths = []
        self.preview_label.set_regions(self._sky_rect, self._ground_rect, self._light_regions)
        self._update_preview_pixmap()
        self._schedule_preview_update()

    def _on_preview_pixel_hovered(self, u: float, v: float) -> None:
        if self._hdri_pixels is None or not hasattr(self, "inspector_label"):
            return
        h, w, _ = self._hdri_pixels.shape
        px = int(np.clip(u * (w - 1), 0, w - 1))
        py = int(np.clip(v * (h - 1), 0, h - 1))
        r, g, b = [float(c) for c in self._hdri_pixels[py, px]]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        ev_18 = float(np.log2(max(lum, 1e-6) / 0.18))
        az_deg = (0.5 - u) * 360.0
        el_deg = (0.5 - v) * 180.0
        self.inspector_label.setText(
            f"Az: {az_deg:+6.1f} deg | El: {el_deg:+5.1f} deg | "
            f"RGB: ({r:.3f}, {g:.3f}, {b:.3f}) | Lum: {lum:.3f} ({ev_18:+.2f} EV vs 18% gray)"
        )

    def _apply_neutral_wb_patch(self, rect: tuple) -> None:
        """Auto-calibrate 3-channel White Balance gains and 18% Gray Exposure from a selected neutral patch."""
        if self._hdri_pixels is None:
            return
        h, w, _ = self._hdri_pixels.shape
        u0, v0, u1, v1 = rect
        x0 = int(np.clip(min(u0, u1) * w, 0, w - 2))
        x1 = int(np.clip(max(u0, u1) * w, x0 + 1, w))
        y0 = int(np.clip(min(v0, v1) * h, 0, h - 2))
        y1 = int(np.clip(max(v0, v1) * h, y0 + 1, h))
        patch = self._hdri_pixels[y0:y1, x0:x1]
        mean_rgb = tuple(float(v) for v in np.mean(patch, axis=(0, 1)))
        gains, patch_cct, duv, lum = compute_neutral_wb_gains(mean_rgb)
        ev_delta = float(np.clip(np.log2(0.18 / max(lum, 1e-4)), -6.0, 6.0))
        self._wb_gains = (
            float(np.clip(self._wb_gains[0] * gains[0], 0.1, 10.0)),
            float(np.clip(self._wb_gains[1] * gains[1], 0.1, 10.0)),
            float(np.clip(self._wb_gains[2] * gains[2], 0.1, 10.0)),
        )
        self._suppress_auto_preview = True
        try:
            self.ev_spin.setValue(float(np.clip(self.ev_spin.value() + 0.5 * ev_delta, -10.0, 10.0)))
        finally:
            self._suppress_auto_preview = False
        self._regenerate_preview()
        self.status_label.setText(
            f"Calibrated neutral patch ({patch_cct:.0f} K, Duv {duv:+.4f}, Lum {lum:.3f}) -> "
            f"WB Gains ({self._wb_gains[0]:.3f}, {self._wb_gains[1]:.3f}, {self._wb_gains[2]:.3f})."
        )

    def _on_open_macbeth_wb(self) -> None:
        dlg = MacbethWBDialog(
            parent=self,
            initial_path=self._macbeth_path,
            initial_colorspace=self._macbeth_colorspace,
            initial_gains=self._wb_gains,
        )
        exec_fn = getattr(dlg, "exec", None) or getattr(dlg, "exec_", None)
        if exec_fn() == QtWidgets.QDialog.Accepted and dlg._apply_requested:
            self._macbeth_path = dlg.path_edit.text().strip()
            self._macbeth_colorspace = dlg.cs_combo.currentText()
            self.apply_wb_gains(
                dlg._sampled_gains,
                ev_delta=dlg._ev_delta if dlg._match_exposure else 0.0,
                source_label=f"Macbeth ({os.path.basename(self._macbeth_path)})" if self._macbeth_path else "Macbeth",
            )

    def apply_wb_gains(
        self,
        gains: tuple,
        ev_delta: float = 0.0,
        source_label: str = "Macbeth",
    ) -> None:
        """Apply sampled 3-channel White Balance gains (and optional EV delta) to HDRI synthesis."""
        self._wb_gains = (
            float(np.clip(gains[0], 0.1, 10.0)),
            float(np.clip(gains[1], 0.1, 10.0)),
            float(np.clip(gains[2], 0.1, 10.0)),
        )
        if abs(ev_delta) > 1e-4:
            self._suppress_auto_preview = True
            try:
                self.ev_spin.setValue(float(np.clip(self.ev_spin.value() + ev_delta, -10.0, 10.0)))
            finally:
                self._suppress_auto_preview = False
        self._regenerate_preview()
        self.status_label.setText(
            f"Applied {source_label} WB Gains: "
            f"(R={self._wb_gains[0]:.3f}, G={self._wb_gains[1]:.3f}, B={self._wb_gains[2]:.3f})."
        )

    def _on_reset_wb(self) -> None:
        self._wb_gains = (1.0, 1.0, 1.0)
        self._regenerate_preview()
        self.status_label.setText("Reset White Balance gains to (1.000, 1.000, 1.000).")

    def _on_auto_level_horizon(self) -> None:
        if self._profile is None:
            self.status_label.setText("Analyze a plate first before auto-leveling the horizon.")
            return
        s_map = self._profile.spherical_plate_map
        s_cov = self._profile.spherical_coverage_mask
        if s_map is None and self._hdri_pixels is not None:
            s_map = self._hdri_pixels
        if s_map is None:
            self.status_label.setText("No spherical map available to estimate horizon.")
            return
        est_shift = estimate_horizon_shift_deg(s_map, s_cov)
        self.vshift_spin.setValue(est_shift)
        roll_msg = ""
        idx = self._selected_plate_idx
        if 0 <= idx < len(self._views):
            est_roll = estimate_horizon_roll_deg(self._views[idx].pixels)
            if abs(est_roll) > 0.25:
                self._views[idx].cam_roll_deg = est_roll
                self._sync_view_to_table_and_bar(idx)
                roll_msg = f" | Plate #{idx + 1} Roll: {est_roll:+.1f} deg"
        self.status_label.setText(f"Auto-leveled Horizon Shift to {est_shift:+.1f} deg{roll_msg}.")

    def _on_link_scene_camera(self) -> None:
        if not HAS_HOU:
            self.status_label.setText("Scene camera linking is only available inside Houdini.")
            return
        cams = list_scene_cameras()
        if not cams:
            self.status_label.setText("No /stage USD cameras or /obj cameras found in the scene.")
            return
        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Link Scene Camera",
            "Select a Solaris (/stage) or /obj Camera:",
            cams,
            0,
            False,
        )
        if not ok or not choice:
            return
        try:
            cam_p = read_scene_camera_params(choice)
            self.fov_spin.setValue(cam_p["fov_h_deg"])
            self.cam_az_spin.setValue(cam_p["azimuth_deg"])
            self.cam_el_spin.setValue(cam_p["elevation_deg"])
            idx = self._selected_plate_idx
            if 0 <= idx < len(self._views):
                v = self._views[idx]
                v.fov_h_deg = cam_p["fov_h_deg"]
                v.cam_azimuth_deg = cam_p["azimuth_deg"]
                v.cam_elevation_deg = cam_p["elevation_deg"]
                v.cam_roll_deg = cam_p["roll_deg"]
                self._sync_view_to_table_and_bar(idx)
                self._regenerate_preview()
            self.status_label.setText(
                f"Linked {choice}: FOV={cam_p['fov_h_deg']:.1f}°, Az={cam_p['azimuth_deg']:+.1f}°, "
                f"El={cam_p['elevation_deg']:+.1f}°, Roll={cam_p['roll_deg']:+.1f}°."
            )
        except Exception as exc:
            self.status_label.setText(f"Camera link error: {exc}")

    def _on_auto_vignette_plate(self) -> None:
        idx = self._selected_plate_idx
        if not (0 <= idx < len(self._views)):
            self.status_label.setText("Select a plate first to estimate radial vignetting.")
            return
        v1 = estimate_radial_vignette(self._views[idx].pixels)
        self._views[idx].vignette_v1 = v1
        self._sync_view_to_table_and_bar(idx)
        self._regenerate_preview()
        self.status_label.setText(f"Estimated Plate #{idx + 1} Vignette lift = {v1:.2f}.")

    def _on_toggle_ab_split(self, checked: bool) -> None:
        if checked and self._profile is not None:
            if getattr(self, "_raw_hdri_pixels", None) is None:
                p = self._current_synthesis_params()
                raw_p = HDRISynthesisParams(
                    width=min(p.width, 1024),
                    height=min(p.height, 512),
                    azimuth_offset_deg=p.azimuth_offset_deg,
                    vertical_shift_deg=p.vertical_shift_deg,
                    plate_projection_blend=1.0,
                    stitch_blend=p.stitch_blend,
                    enable_inpaint=False,
                    ldr_highlight_boost=1.0,
                )
                self._raw_hdri_pixels = generate_latlong_hdri(
                    self._profile,
                    raw_p,
                    views=self._views if len(self._views) <= 12 else self._views[:1],
                )
            self._update_preview_pixmap()
        else:
            self.preview_label.set_ab_split(False)
            self._update_preview_pixmap()

    def _schedule_display_update(self) -> None:
        ev = self.preview_ev_slider.value() / 10.0
        self.preview_ev_val.setText(f"{ev:+.1f} EV")
        if hasattr(self, "preview_roughness_val"):
            self.preview_roughness_val.setText(f"{self.preview_roughness_slider.value() / 100.0:.2f}")
        if self._suppress_auto_preview:
            return
        self._display_timer.start()

    def _current_brush_mode(self) -> str:
        txt = self.brush_mode_combo.currentText().lower()
        if "clone" in txt:
            return "clone"
        if "blur" in txt:
            return "blur"
        if "burn" in txt:
            return "burn"
        if "dodge" in txt:
            return "dodge"
        if "desat" in txt:
            return "desat"
        return "heal"

    def _on_toggle_brush_mode(self, checked: bool) -> None:
        if checked:
            for btn in (self.pick_sky_btn, self.pick_gnd_btn, self.pick_light_btn):
                btn.setChecked(False)
            if hasattr(self, "pick_wb_btn"):
                self.pick_wb_btn.setChecked(False)
            if hasattr(self, "edit_plates_btn") and self.edit_plates_btn.isChecked():
                self.edit_plates_btn.setChecked(False)
            self.preview_label.set_pick_mode(None)
            m = self._current_brush_mode()
            if m == "clone":
                self.status_label.setText("Brush (Clone Stamp): Ctrl+Click to set Clone Source, then drag to paint.")
            else:
                self.status_label.setText(f"Brush ({self.brush_mode_combo.currentText()}): Drag on canvas to paint in 32-bit float.")
        self._on_brush_config_changed()

    def _on_brush_config_changed(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        self.preview_label.set_brush_state(
            enabled=self.brush_btn.isChecked(),
            mode=self._current_brush_mode(),
            radius_u=self.brush_size_slider.value() / 1000.0,
            softness=self.brush_soft_slider.value() / 100.0,
            clone_src_uv=self._brush_clone_src_uv,
        )

    def _on_brush_param_adjusted(self, param: str, dx: float) -> None:
        step = int(round(dx * 0.5))
        if step == 0 and abs(dx) > 0.2:
            step = 1 if dx > 0 else -1
        hud = ""
        if param == "size":
            val = int(np.clip(self.brush_size_slider.value() + step, self.brush_size_slider.minimum(), self.brush_size_slider.maximum()))
            self.brush_size_slider.setValue(val)
            hud = f"Size: {val / 10.0:.1f}%"
        elif param == "strength":
            val = int(np.clip(self.brush_strength_slider.value() + step, self.brush_strength_slider.minimum(), self.brush_strength_slider.maximum()))
            self.brush_strength_slider.setValue(val)
            hud = f"Strength: {val}%"
        elif param == "soft":
            val = int(np.clip(self.brush_soft_slider.value() + step, self.brush_soft_slider.minimum(), self.brush_soft_slider.maximum()))
            self.brush_soft_slider.setValue(val)
            hud = f"Soft: {val}%"
        if hud:
            self.preview_label.set_brush_state(
                enabled=self.brush_btn.isChecked(),
                mode=self._current_brush_mode(),
                radius_u=self.brush_size_slider.value() / 1000.0,
                softness=self.brush_soft_slider.value() / 100.0,
                clone_src_uv=self._brush_clone_src_uv,
                hud_text=hud,
            )

    def _on_brush_clone_src_picked(self, u: float, v: float) -> None:
        self._brush_clone_src_uv = (float(u), float(v))
        self.status_label.setText(f"Clone Source set at ({u:.3f}, {v:.3f}). Drag anywhere to clone-stamp.")

    def _on_brush_stroke_started(self, u: float, v: float) -> None:
        if self._hdri_pixels is None:
            return
        mode = self._current_brush_mode()
        src_off = None
        if mode == "clone":
            if self._brush_clone_src_uv is None:
                self.status_label.setText("Ctrl+Click on the canvas first to set a Clone Source.")
                return
            src_off = (self._brush_clone_src_uv[0] - u, self._brush_clone_src_uv[1] - v)
            self._brush_clone_offset_uv = src_off
        else:
            self._brush_clone_offset_uv = None

        radius_u = self.brush_size_slider.value() / 1000.0
        softness = self.brush_soft_slider.value() / 100.0
        strength = self.brush_strength_slider.value() / 100.0
        if self._pre_brush_hdri is None:
            self._pre_brush_hdri = self._hdri_pixels.copy()
        self._brush_undo_stack.append(self._hdri_pixels.copy())
        if len(self._brush_undo_stack) > 15:
            self._brush_undo_stack.pop(0)
        self._active_brush_stroke = {
            "mode": mode,
            "radius_u": radius_u,
            "softness": softness,
            "strength": strength,
            "src_offset": list(src_off) if src_off is not None else None,
            "points": [[float(u), float(v)]],
        }
        apply_brush_dab(
            self._hdri_pixels,
            u,
            v,
            mode=mode,
            radius_u=radius_u,
            softness=softness,
            strength=strength,
            src_offset_uv=src_off,
            inplace=True,
        )
        self._rough_cache = None
        self._update_preview_pixmap()

    def _on_brush_stroke_moved(self, u: float, v: float) -> None:
        if self._hdri_pixels is None or self._active_brush_stroke is None:
            return
        pts = self._active_brush_stroke["points"]
        last_u, last_v = pts[-1]
        if ((u - last_u) ** 2 + (v - last_v) ** 2) < (self._active_brush_stroke["radius_u"] * 0.25) ** 2:
            return
        pts.append([float(u), float(v)])
        src_off = self._brush_clone_offset_uv
        apply_brush_dab(
            self._hdri_pixels,
            u,
            v,
            mode=self._active_brush_stroke["mode"],
            radius_u=self._active_brush_stroke["radius_u"],
            softness=self._active_brush_stroke["softness"],
            strength=self._active_brush_stroke["strength"],
            src_offset_uv=src_off,
            inplace=True,
        )
        self._rough_cache = None
        self._update_preview_pixmap()

    def _on_brush_stroke_finished(self) -> None:
        if self._active_brush_stroke is not None:
            self._brush_strokes.append(self._active_brush_stroke)
            self._active_brush_stroke = None
            self._brush_clone_offset_uv = None
            self.status_label.setText(f"Applied brush stroke ({len(self._brush_strokes)} total).")

    def _on_undo_brush(self) -> None:
        if not self._brush_strokes:
            self.status_label.setText("No brush strokes to undo.")
            return
        self._brush_strokes.pop()
        if self._brush_undo_stack:
            self._hdri_pixels = self._brush_undo_stack.pop()
            self._rough_cache = None
            self._update_preview_pixmap()
        elif self._pre_brush_hdri is not None:
            self._hdri_pixels = apply_brush_strokes(self._pre_brush_hdri, self._brush_strokes)
            self._rough_cache = None
            self._update_preview_pixmap()
        else:
            self._regenerate_preview()
        self.status_label.setText(f"Undid last brush stroke ({len(self._brush_strokes)} remaining).")

    def _on_clear_brush(self) -> None:
        if not self._brush_strokes:
            return
        self._brush_strokes.clear()
        self._brush_undo_stack.clear()
        if self._pre_brush_hdri is not None:
            self._hdri_pixels = self._pre_brush_hdri.copy()
            self._rough_cache = None
            self._update_preview_pixmap()
        else:
            self._regenerate_preview()
        self.status_label.setText("Cleared all brush strokes.")

    def _on_bake_edits_to_canvas(self) -> None:
        if self._profile is None:
            self.status_label.setText("Analyze a plate first before baking edits.")
            return
        try:
            p = self._current_synthesis_params()
            # Bake geometry, stitching, sky/ground/light inpainting, and brush strokes at up to 4K
            # while leaving global post-adjustments (EV, CCT, Saturation, Blur, etc.) live.
            bake_p = HDRISynthesisParams(
                width=min(p.width, 4096),
                height=min(p.height, 2048),
                hotspot_boost=p.hotspot_boost,
                hotspot_softness=p.hotspot_softness,
                ambient_wrap=p.ambient_wrap,
                plate_projection_blend=p.plate_projection_blend,
                vertical_shift_deg=p.vertical_shift_deg,
                stitch_blend=p.stitch_blend,
                ldr_highlight_boost=p.ldr_highlight_boost,
                enable_inpaint=p.enable_inpaint,
                sky_blend=p.sky_blend,
                ground_blend=p.ground_blend,
                sky_blur=p.sky_blur,
                ground_blur=p.ground_blur,
                sky_rect=self._sky_rect,
                ground_rect=self._ground_rect,
                light_rect=p.light_rect,
                remove_hotspot_from_hdri=self.remove_hotspot_chk.isChecked(),
                physical_sky_fill=p.physical_sky_fill,
            )
            if self._baked_canvas_hdri is not None:
                baked = self._baked_canvas_hdri.copy()
            else:
                baked = generate_latlong_hdri(
                    self._profile,
                    bake_p,
                    views=self._views if len(self._views) <= 12 else self._views[:1],
                )
            if self._light_regions:
                baked, extracted = extract_and_inpaint_multiple_hotspots(
                    baked,
                    self._light_regions,
                    inpaint_hdri=self.remove_hotspot_chk.isChecked(),
                    region_blur=self.light_blur_slider.value(),
                )
                self._extracted_lights = extracted
            if self._brush_strokes:
                baked = apply_brush_strokes(baked, self._brush_strokes)

            self._baked_canvas_hdri = np.ascontiguousarray(baked, dtype=np.float32)
            self._sky_rect = None
            self._ground_rect = None
            self._light_regions.clear()
            self._brush_strokes.clear()
            self._brush_undo_stack.clear()
            self._pre_brush_hdri = None
            self.unbake_edits_btn.setEnabled(True)
            self.bake_edits_btn.setText("Re-Apply / Collapse")
            self._regenerate_preview()
            self.status_label.setText(
                f"Applied & collapsed all edits into fast {baked.shape[1]}x{baked.shape[0]} 32-bit canvas. "
                "Parameter tweaks are now instant!"
            )
        except Exception as exc:
            self.status_label.setText(f"Bake error: {exc}")

    def _on_unbake_edits(self) -> None:
        if self._baked_canvas_hdri is None:
            return
        self._baked_canvas_hdri = None
        self.unbake_edits_btn.setEnabled(False)
        self.bake_edits_btn.setText("Apply / Collapse")
        self._regenerate_preview()
        self.status_label.setText("Unbaked: returned to live multi-plate synthesis pipeline.")

    def _on_merge_brackets_clicked(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Bracketed Exposure Plates to Merge into 32-bit HDR EXR",
            "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.exr *.hdr);;All Files (*)",
        )
        if not paths or len(paths) < 2:
            if paths:
                self.status_label.setText("Select at least 2 bracketed exposures to merge.")
            return
        try:
            cs = self.colorspace_combo.currentText()
            merged_hdr = merge_ldr_brackets_to_hdr(paths, colorspace=cs)
            stem, _ = os.path.splitext(paths[0])
            merged_out = f"{stem}_merged_hdr.exr"
            written = write_exr_32f(merged_out, merged_hdr, output_colorspace="Linear Rec.709")
            self.plate_path_edit.setText(written)
            self._maybe_update_default_output_path(written)
            self._on_analyze_clicked()
            self.status_label.setText(
                f"Merged {len(paths)} brackets into 32-bit float HDR: {written}"
            )
        except Exception as exc:
            self.status_label.setText(f"Bracket merge error: {exc}")

    def _on_analyze_clicked(self) -> None:
        n_mv = self.multiview_table.rowCount()
        self._log_debug(f"Analyze clicked: multiview_rows={n_mv}")
        if n_mv > 0:
            try:
                views: List[PlateViewInput] = []
                for row in range(n_mv):
                    item0 = self.multiview_table.item(row, 0)
                    fpath = item0.data(Qt.UserRole)
                    extra = item0.data(Qt.UserRole + 1) or {}
                    az = float(self.multiview_table.item(row, 1).text())
                    el = float(self.multiview_table.item(row, 2).text())
                    fov = float(self.multiview_table.item(row, 3).text())
                    cs = self.multiview_table.item(row, 4).text() or "Auto"
                    self._log_debug(f"Reading multiview plate #{row + 1}: {fpath} (az={az}, el={el}, fov={fov}, cs={cs})")
                    plate = read_plate(fpath, colorspace=cs, max_dim=4096)
                    self._log_debug(f"Loaded plate #{row + 1}: shape={plate.pixels.shape}, is_hdr={plate.is_hdr}")
                    views.append(
                        PlateViewInput(
                            pixels=plate.pixels,
                            is_hdr=plate.is_hdr,
                            source_path=fpath,
                            fov_h_deg=fov,
                            cam_azimuth_deg=az,
                            cam_elevation_deg=el,
                            cam_roll_deg=float(extra.get("roll", 0.0)),
                            scale_x=float(extra.get("scale_x", 1.0)),
                            scale_y=float(extra.get("scale_y", 1.0)),
                            warp_k1=float(extra.get("warp_k1", 0.0)),
                            lens_type=str(extra.get("lens_type", "rectilinear")),
                            exposure_ev=float(extra.get("exposure_ev", 0.0)),
                            vignette_v1=float(extra.get("vignette_v1", 0.0)),
                            exclude_rects=[tuple(r) for r in (extra.get("exclude_rects") or [])],
                        )
                    )
                self._views = views
                self._selected_plate_idx = min(max(0, self._selected_plate_idx), len(views) - 1) if views else -1
                self._refresh_plate_transform_bar()
                self._frame_profiles = []
                self._log_debug("Running analyze_multiview_plates...")
                self._profile = analyze_multiview_plates(
                    views,
                    max_hotspots=self.max_hotspots_spin.value(),
                    ldr_highlight_boost=self.ldr_boost_spin.value(),
                )
                self._log_debug(f"Analysis done: CCT={self._profile.overall_cct_kelvin:.0f}K, hotspots={len(self._profile.hotspots)}")
                self._populate_analysis_ui(self._profile, len(views))
                self._regenerate_preview()
                self.status_label.setText(
                    f"Analyzed {len(views)} multi-position plate(s). "
                    f"Overall CCT: {self._profile.overall_cct_kelvin:.0f} K, "
                    f"Hotspots: {len(self._profile.hotspots)}."
                )
            except Exception as exc:
                self._log_debug(f"Multi-view analysis error: {exc}\n{traceback.format_exc()}")
                self.status_label.setText(f"Multi-view analysis error: {exc}")
            return

        pattern = self.plate_path_edit.text().strip()
        if not pattern:
            self.status_label.setText("Please enter or browse to a plate file or sequence.")
            return

        mode = self.seq_mode_combo.currentText()
        auto_seq = mode != "Single Frame"
        start_f = self.start_frame_spin.value() if self.use_range_chk.isChecked() else None
        end_f = self.end_frame_spin.value() if self.use_range_chk.isChecked() else None
        step_f = self.step_frame_spin.value() if self.use_range_chk.isChecked() else 1

        try:
            frames = resolve_plate_sequence(
                pattern,
                start_frame=start_f,
                end_frame=end_f,
                step=step_f,
                auto_detect_sequence=auto_seq,
            )
            if not frames:
                self.status_label.setText(f"No matching plate files found for: {pattern}")
                return

            if mode == "Single Frame":
                frames = frames[:1]

            self._frame_profiles = []
            self._views = []
            cs = self.colorspace_combo.currentText()
            fov = self.fov_spin.value()
            az = self.cam_az_spin.value()
            el = self.cam_el_spin.value()
            load_max_dim = 4096 if len(frames) <= 8 else 2048
            for fnum, fpath in frames:
                plate = read_plate(fpath, colorspace=cs, frame=fnum, max_dim=load_max_dim)
                view = PlateViewInput(
                    pixels=plate.pixels,
                    is_hdr=plate.is_hdr,
                    source_path=fpath,
                    fov_h_deg=fov,
                    cam_azimuth_deg=az,
                    cam_elevation_deg=el,
                )
                self._views.append(view)
                prof = analyze_plate(
                    plate.pixels,
                    is_hdr=plate.is_hdr,
                    frame=fnum,
                    source_path=fpath,
                    fov_h_deg=fov,
                    cam_azimuth_deg=az,
                    cam_elevation_deg=el,
                    max_hotspots=self.max_hotspots_spin.value(),
                    ldr_highlight_boost=self.ldr_boost_spin.value(),
                )
                self._frame_profiles.append(prof)

            self._profile = (
                self._frame_profiles[0]
                if len(self._frame_profiles) == 1
                else merge_lighting_profiles(self._frame_profiles)
            )
            self._selected_plate_idx = 0 if self._views else -1
            self._refresh_plate_transform_bar()
            self._populate_analysis_ui(self._profile, len(self._frame_profiles))
            self._regenerate_preview()
            self._auto_save_session()
            self.status_label.setText(
                f"Analyzed {len(self._frame_profiles)} frame(s). "
                f"Overall CCT: {self._profile.overall_cct_kelvin:.0f} K, "
                f"Hotspots: {len(self._profile.hotspots)}."
            )
        except Exception as exc:
            self._log_debug(f"Analysis error: {exc}\n{traceback.format_exc()}")
            self.status_label.setText(f"Analysis error: {exc}")

    def _populate_analysis_ui(self, prof: LightingProfile, n_frames: int) -> None:
        self.metrics_label.setText(
            f"<b>Frames:</b> {n_frames} | "
            f"<b>DR:</b> {prof.dynamic_range_stops:.1f} stops | "
            f"<b>Key/Fill:</b> {prof.key_to_fill_ratio:.1f}:1 | "
            f"<b>EV100:</b> {prof.estimated_ev100:.1f}<br>"
            f"<b>Overall CCT:</b> {prof.overall_cct_kelvin:.0f} K (Duv {prof.overall_tint_duv:+.4f}) | "
            f"<b>Key:</b> {prof.key_cct_kelvin:.0f} K | "
            f"<b>Sky:</b> {prof.sky_cct_kelvin:.0f} K | "
            f"<b>Ground:</b> {prof.ground_cct_kelvin:.0f} K"
        )

        # Update tonal swatches
        swatch_map = {
            "Zenith": prof.zenith_color,
            "Sky": prof.sky_color,
            "Horizon": prof.horizon_color,
            "Ground": prof.ground_color,
            "Nadir": prof.nadir_color,
        }
        for name, rgb in swatch_map.items():
            srgb = linear_to_srgb(np.array(rgb, dtype=np.float32))
            r, g, b = [int(np.clip(c * 255.0, 0, 255)) for c in srgb]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            fg = "#111111" if lum > 130 else "#f0f0f0"
            self._swatches[name].setStyleSheet(
                f"background-color: rgb({r},{g},{b}); color: {fg}; "
                f"border: 1px solid #555; font-weight: bold;"
            )

        # Update hotspots table
        self.hotspot_table.setRowCount(len(prof.hotspots))
        for row, hs in enumerate(prof.hotspots):
            vals = [
                str(hs.index + 1),
                f"{hs.azimuth_deg:+.1f}",
                f"{hs.elevation_deg:+.1f}",
                f"{hs.radius_deg:.1f}",
                f"{hs.peak_intensity:.2f}",
                f"{hs.cct_kelvin:.0f}",
            ]
            for col, val in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.hotspot_table.setItem(row, col, item)

    def _regenerate_preview(self, fast: bool = False) -> None:
        if self._profile is None:
            return
        try:
            # Crisp 2K preview (or 640x320 during interactive plate dragging)
            p = self._current_synthesis_params()
            if fast:
                p.width = min(p.width, 400)
                p.height = min(p.height, 200)
                p.enable_inpaint = False
            else:
                p.width = min(p.width, 2048)
                p.height = min(p.height, 1024)
            if not fast:
                self._log_debug(f"Regenerating preview ({p.width}x{p.height}), views={len(self._views)}")
            if self._baked_canvas_hdri is not None:
                hdri = apply_post_adjustments_to_hdri(self._baked_canvas_hdri, p, self._profile)
            else:
                hdri = generate_latlong_hdri(
                    self._profile,
                    p,
                    views=self._views if len(self._views) <= 12 else self._views[:1],
                )
            if not fast:
                self._log_debug(f"generate_latlong_hdri finished: shape={hdri.shape}")
            if (not fast) and self._light_regions:
                hdri, extracted = extract_and_inpaint_multiple_hotspots(
                    hdri,
                    self._light_regions,
                    inpaint_hdri=self.remove_hotspot_chk.isChecked(),
                    region_blur=self.light_blur_slider.value(),
                )
                self._extracted_lights = extracted
            elif not self._light_regions:
                self._extracted_lights = []
            if not fast:
                self._pre_brush_hdri = hdri.copy()
                self._brush_undo_stack.clear()
                if self._brush_strokes:
                    hdri = apply_brush_strokes(hdri, self._brush_strokes)
            self._hdri_pixels = hdri
            if hasattr(self, "ab_split_btn") and self.ab_split_btn.isChecked() and not fast:
                raw_p = HDRISynthesisParams(
                    width=p.width,
                    height=p.height,
                    azimuth_offset_deg=p.azimuth_offset_deg,
                    vertical_shift_deg=p.vertical_shift_deg,
                    plate_projection_blend=1.0,
                    stitch_blend=p.stitch_blend,
                    enable_inpaint=False,
                    ldr_highlight_boost=1.0,
                )
                self._raw_hdri_pixels = generate_latlong_hdri(
                    self._profile,
                    raw_p,
                    views=self._views if len(self._views) <= 12 else self._views[:1],
                )
            elif not hasattr(self, "ab_split_btn") or not self.ab_split_btn.isChecked():
                self._raw_hdri_pixels = None
            if not fast:
                self._log_debug("Updating preview pixmap...")
            self._update_preview_pixmap()
            if not fast:
                self._log_debug("Preview pixmap updated.")
        except Exception as exc:
            self._log_debug(f"Preview generation error: {exc}\n{traceback.format_exc()}")
            self.status_label.setText(f"Preview error: {exc}")

    def _update_preview_pixmap(self) -> None:
        if self._hdri_pixels is None:
            return
        ev = self.preview_ev_slider.value() / 10.0
        self.preview_ev_val.setText(f"{ev:+.1f} EV")
        tm_mode = self.preview_tonemap_combo.currentText() if hasattr(self, "preview_tonemap_combo") else "aces"
        if tm_mode == "Overlap Diff" and self._views:
            diff_rgb = compute_overlap_difference_map(
                self._views if len(self._views) <= 12 else self._views[:1],
                out_width=min(1024, self._hdri_pixels.shape[1]),
                out_height=min(512, self._hdri_pixels.shape[0]),
                azimuth_offset_deg=self.yaw_spin.value(),
                vertical_shift_deg=self.vshift_spin.value(),
            )
            srgb = np.clip(diff_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            rough = (self.preview_roughness_slider.value() / 100.0) if hasattr(self, "preview_roughness_slider") else 0.0
            if hasattr(self, "preview_roughness_val"):
                self.preview_roughness_val.setText(f"{rough:.2f}")

            def _get_display_src(arr: np.ndarray, cache_attr: str, rough_val: float) -> np.ndarray:
                cached = getattr(self, cache_attr, None)
                arr_id = id(arr)
                if cached is not None and cached[0] == arr_id and abs(cached[1] - rough_val) < 1e-4:
                    return cached[2]
                sh, sw = arr.shape[:2]
                # Cap display buffer at 1024x512 (or 320x160 when roughness blur > 0) for instant EV/Rough dragging
                max_w = 320 if rough_val > 1e-3 else 1024
                if sw > max_w:
                    step = max(1, sw // max_w)
                    disp = arr[::step, ::step]
                else:
                    disp = arr
                if rough_val > 1e-3:
                    from .hdri_generator import blur_latlong_hdri
                    disp = blur_latlong_hdri(disp, blur_amount=min(1.0, (rough_val ** 1.45) * 0.85))
                setattr(self, cache_attr, (arr_id, rough_val, disp))
                return disp

            disp_src = _get_display_src(self._hdri_pixels, "_rough_cache", rough)
            srgb = render_preview_srgb(
                disp_src,
                preview_ev=ev,
                tonemap=tm_mode,
                roughness_preview=0.0,
            )
            if (
                hasattr(self, "ab_split_btn")
                and self.ab_split_btn.isChecked()
                and getattr(self, "_raw_hdri_pixels", None) is not None
            ):
                raw_disp_src = _get_display_src(self._raw_hdri_pixels, "_raw_rough_cache", rough)
                raw_srgb = render_preview_srgb(
                    raw_disp_src,
                    preview_ev=ev,
                    tonemap=tm_mode,
                    roughness_preview=0.0,
                )
                rh, rw, _ = raw_srgb.shape
                rqimg = QtGui.QImage(raw_srgb.data, rw, rh, 3 * rw, QtGui.QImage.Format_RGB888)
                self.preview_label.set_ab_split(True, QtGui.QPixmap.fromImage(rqimg.copy()))
            else:
                self.preview_label.set_ab_split(False)
        self.preview_label.set_regions(self._sky_rect, self._ground_rect, self._light_regions)
        self.preview_label.set_plate_overlay(
            self._views if len(self._views) <= 12 else self._views[:1],
            self._selected_plate_idx,
            self.yaw_spin.value(),
            self.vshift_spin.value(),
        )
        h, w, _ = srgb.shape
        qimg = QtGui.QImage(srgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg.copy())

        if (
            (self.show_markers_chk.isChecked() and self._profile is not None)
            or self._sky_rect is not None
            or self._ground_rect is not None
            or len(self._light_regions) > 0
        ):
            painter = QtGui.QPainter(pix)
            if self._sky_rect is not None:
                u0, v0, u1, v1 = self._sky_rect
                painter.setPen(QtGui.QPen(QtGui.QColor(80, 220, 255), 2, Qt.DashLine))
                painter.drawRect(int(u0 * w), int(v0 * h), int((u1 - u0) * w), int((v1 - v0) * h))
                painter.drawText(int(u0 * w) + 4, int(v0 * h) + 14, "SKY")
            if self._ground_rect is not None:
                u0, v0, u1, v1 = self._ground_rect
                painter.setPen(QtGui.QPen(QtGui.QColor(90, 240, 120), 2, Qt.DashLine))
                painter.drawRect(int(u0 * w), int(v0 * h), int((u1 - u0) * w), int((v1 - v0) * h))
                painter.drawText(int(u0 * w) + 4, int(v0 * h) + 14, "GROUND")
            for idx, lreg in enumerate(self._light_regions, start=1):
                u0, v0, u1, v1 = lreg["rect"]
                shape = lreg.get("shape", "rect")
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 200, 60), 2, Qt.DashLine))
                rx, ry = int(u0 * w), int(v0 * h)
                rw, rh = int((u1 - u0) * w), int((v1 - v0) * h)
                if shape in ("circle", "ellipse"):
                    painter.drawEllipse(rx, ry, rw, rh)
                else:
                    painter.drawRect(rx, ry, rw, rh)
                painter.drawText(rx + 4, ry + 14, f"LIGHT #{idx}")
            if self.show_markers_chk.isChecked() and self._profile is not None:
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 220, 60), 2))
                for hs in self._profile.hotspots:
                    px = int(((hs.azimuth_deg / 360.0) + 0.5) * w) % w
                    py = int((0.5 - (hs.elevation_deg / 180.0)) * h)
                    painter.drawEllipse(QtCore.QPoint(px, py), 8, 8)
                    painter.drawText(px + 10, py + 4, f"#{hs.index + 1}")
            painter.end()

        self.preview_label.set_source_pixmap(pix)

    def _on_export_sequence_exr(self) -> Optional[str]:
        if not self._frame_profiles:
            self.status_label.setText("Analyze a sequence first (Mode != Single Frame) before exporting sequence EXRs.")
            return None
        out_path = self.out_path_edit.text().strip()
        if not out_path:
            self.status_label.setText("Specify an Output EXR path first.")
            return None
        if "$F" not in out_path and "#" not in out_path:
            stem, ext = os.path.splitext(out_path)
            out_path = f"{stem}.$F4{ext or '.exr'}"
        try:
            p = self._current_synthesis_params()
            cs = self.out_cs_combo.currentText()
            pattern, written = export_hdri_sequence(
                self._frame_profiles,
                self._views,
                p,
                output_pattern=out_path,
                output_colorspace=cs,
                light_regions=self._light_regions,
                inpaint_lights=self.remove_hotspot_chk.isChecked(),
                light_blur=self.light_blur_slider.value(),
                temporal_stabilize=p.temporal_stabilize,
                export_diffuse_split=self.diffuse_split_chk.isChecked(),
            )
            self.out_path_edit.setText(pattern)
            self.status_label.setText(f"Exported {len(written)} 32-bit float EXR frame(s): {pattern}")
            self._log_debug(f"Exported sequence ({len(written)} frames): {pattern}")
            return pattern
        except Exception as exc:
            self._log_debug(f"Sequence export error: {exc}\n{traceback.format_exc()}")
            self.status_label.setText(f"Sequence export error: {exc}")
            return None

    def _on_export_exr(self) -> Optional[str]:
        if self._profile is None:
            self.status_label.setText("Analyze a plate first before exporting.")
            return None

        out_path = self.out_path_edit.text().strip()
        if not out_path:
            self.status_label.setText("Please specify an output .exr path.")
            return None

        params = self._current_synthesis_params()
        out_cs = self.out_cs_combo.currentText()
        mode = self.seq_mode_combo.currentText()

        try:
            if mode == "Per-Frame Sequence" and len(self._frame_profiles) > 1:
                written_paths = []
                last_hdri = None
                for idx, prof in enumerate(self._frame_profiles):
                    hdri = generate_latlong_hdri(prof, params, views=None)
                    if self._light_regions:
                        hdri, extracted = extract_and_inpaint_multiple_hotspots(
                            hdri,
                            self._light_regions,
                            light_tex_size=self._current_light_tex_size(),
                            inpaint_hdri=self.remove_hotspot_chk.isChecked(),
                            region_blur=self.light_blur_slider.value(),
                        )
                        self._extracted_lights = extracted
                        stem, _ = os.path.splitext(out_path)
                        self._last_light_exr_paths = []
                        for l_idx, (light_tex, light_info) in enumerate(extracted, start=1):
                            light_exr_out = f"{stem}_light_{l_idx}.exr"
                            written_l = write_exr_32f(
                                light_exr_out,
                                light_tex,
                                output_colorspace=out_cs,
                                metadata={
                                    "hdri_maker:light_index": str(l_idx),
                                    "hdri_maker:light_shape": light_info.shape,
                                    "hdri_maker:light_azimuth_deg": f"{light_info.azimuth_deg:.2f}",
                                    "hdri_maker:light_elevation_deg": f"{light_info.elevation_deg:.2f}",
                                    "hdri_maker:light_cct_kelvin": f"{light_info.cct_kelvin:.1f}",
                                },
                                frame=prof.frame,
                            )
                            self._last_light_exr_paths.append(written_l)
                    meta = {
                        "hdri_maker:cct_kelvin": f"{prof.overall_cct_kelvin:.1f}",
                        "hdri_maker:dynamic_range_stops": f"{prof.dynamic_range_stops:.2f}",
                        "hdri_maker:source_plate": prof.source_path,
                    }
                    p_written = write_exr_32f(
                        out_path,
                        hdri,
                        output_colorspace=out_cs,
                        metadata=meta,
                        frame=prof.frame,
                    )
                    written_paths.append(p_written)
                    last_hdri = hdri
                self._last_exr_path = out_path
                self._last_hdri_full = self._downsample_for_lookdev(last_hdri)
                lookdev_msg = ""
                if self.lookdev_chk.isChecked() and self._last_hdri_full is not None:
                    lookdev_msg = self._bake_and_wire_lookdev(self._last_hdri_full, written_paths[-1], out_cs)
                self.status_label.setText(
                    f"Exported {len(written_paths)} 32-bit float Lat/Long EXR frames (last: {written_paths[-1]}){lookdev_msg}."
                )
                return out_path
            else:
                if self._baked_canvas_hdri is not None:
                    hdri = apply_post_adjustments_to_hdri(self._baked_canvas_hdri, params, self._profile)
                else:
                    hdri = generate_latlong_hdri(
                        self._profile,
                        params,
                        views=self._views if self._views else None,
                    )
                light_msg = ""
                self._last_light_exr_paths = []
                if self._light_regions:
                    hdri, extracted = extract_and_inpaint_multiple_hotspots(
                        hdri,
                        self._light_regions,
                        light_tex_size=self._current_light_tex_size(),
                        inpaint_hdri=self.remove_hotspot_chk.isChecked(),
                        region_blur=self.light_blur_slider.value(),
                    )
                    self._extracted_lights = extracted
                    stem, _ = os.path.splitext(out_path)
                    for idx, (light_tex, light_info) in enumerate(extracted, start=1):
                        light_exr_out = f"{stem}_light_{idx}.exr"
                        written_l = write_exr_32f(
                            light_exr_out,
                            light_tex,
                            output_colorspace=out_cs,
                            metadata={
                                "hdri_maker:light_index": str(idx),
                                "hdri_maker:light_shape": light_info.shape,
                                "hdri_maker:light_azimuth_deg": f"{light_info.azimuth_deg:.2f}",
                                "hdri_maker:light_elevation_deg": f"{light_info.elevation_deg:.2f}",
                                "hdri_maker:light_cct_kelvin": f"{light_info.cct_kelvin:.1f}",
                            },
                        )
                        self._last_light_exr_paths.append(written_l)
                    light_msg = f" + {len(self._last_light_exr_paths)} Light EXR(s)"
                if self._brush_strokes:
                    hdri = apply_brush_strokes(hdri, self._brush_strokes)

                meta = {
                    "hdri_maker:cct_kelvin": f"{self._profile.overall_cct_kelvin:.1f}",
                    "hdri_maker:dynamic_range_stops": f"{self._profile.dynamic_range_stops:.2f}",
                    "hdri_maker:source_plate": self._profile.source_path,
                }
                p_written = write_exr_32f(
                    out_path,
                    hdri,
                    output_colorspace=out_cs,
                    metadata=meta,
                )
                if self.diffuse_split_chk.isChecked():
                    stem_d, ext_d = os.path.splitext(p_written)
                    diff_out = f"{stem_d}_diffuse{ext_d or '.exr'}"
                    diff_hdri = generate_diffuse_irradiance_hdri(hdri)
                    write_exr_32f(diff_out, diff_hdri, output_colorspace=out_cs, metadata=meta)
                    light_msg += " + Diffuse EXR"
                self._last_exr_path = p_written
                self._last_hdri_full = self._downsample_for_lookdev(hdri)
                lookdev_msg = ""
                if self.lookdev_chk.isChecked() and self._last_hdri_full is not None:
                    lookdev_msg = self._bake_and_wire_lookdev(self._last_hdri_full, p_written, out_cs)
                del hdri

                self.status_label.setText(f"Exported 32-bit float Lat/Long EXR: {p_written}{light_msg}{lookdev_msg}")
                return p_written
        except Exception as exc:
            self.status_label.setText(f"Export error: {exc}")
            return None

    def _bake_and_wire_lookdev(self, hdri: np.ndarray, exr_path: str, out_cs: str) -> str:
        stem, _ = os.path.splitext(exr_path)
        tripod_h = self.tripod_y_slider.value()
        g_radius = self.ground_radius_slider.value()
        feather_start = 1.0 - float(np.clip(self.ground_feather_slider.value(), 0.02, 1.0))
        g_rgb, g_op = bake_planar_ground_texture(
            hdri,
            ground_radius=g_radius,
            tripod_height=tripod_h,
            resolution=1024,
            feather_start=feather_start,
            proj_blur=self.ground_proj_blur_slider.value(),
        )
        g_tex_path = write_exr_32f(f"{stem}_ground_proj.exr", g_rgb, output_colorspace=out_cs)
        g_op_path = write_exr_32f(f"{stem}_ground_opacity.exr", g_op, output_colorspace="Linear Rec.709")
        if HAS_HOU:
            create_or_update_lookdev_and_ground(
                ground_tex_exr_path=g_tex_path,
                ground_opacity_exr_path=g_op_path,
                parent_node=self._last_dome_node,
                colorspace=out_cs,
                ground_radius=g_radius,
                tripod_height=tripod_h,
                ground_emission=self.ground_emission_slider.value(),
                ground_shader_mode=self.ground_shader_combo.currentText(),
                render_engine=self.render_engine_combo.currentText(),
            )
            return " + Lookdev Tripod & Ground (/lookdev)"
        return f" + Ground Proj EXR: {g_tex_path}"

    def _on_render_engine_changed(self) -> None:
        if HAS_HOU and self._profile is not None and self._last_exr_path and self.lookdev_chk.isChecked():
            self._on_update_lookdev_clicked()

    def _on_update_lookdev_clicked(self) -> None:
        if self._profile is None:
            self.status_label.setText("Analyze a plate first before updating Lookdev / Ground.")
            return
        hdri_src = getattr(self, "_last_hdri_full", None)
        if hdri_src is None or not self._last_exr_path:
            self._on_export_exr()
            return
        try:
            out_cs = self.out_cs_combo.currentText()
            msg = self._bake_and_wire_lookdev(hdri_src, self._last_exr_path, out_cs)
            self.status_label.setText(f"Updated{msg} (Tripod Y={self.tripod_y_slider.value():.2f}m)")
        except Exception as exc:
            self.status_label.setText(f"Lookdev / Ground update error: {exc}")

    def _on_create_domelight(self) -> None:
        if not HAS_HOU:
            self.status_label.setText("Solaris /stage wiring is only available inside Houdini.")
            return

        exr_path = self._last_exr_path or self._on_export_exr()
        if not exr_path:
            return

        try:
            dome = create_or_update_domelight(
                hdri_exr_path=exr_path,
                profile=self._profile,
            )
            self._last_dome_node = dome
            self.status_label.setText(
                f"Solaris Dome Light ready at {dome.path()} -> {exr_path}"
            )
        except Exception as exc:
            self.status_label.setText(f"Solaris Dome Light error: {exc}")

    def _on_create_hotspot_rig(self) -> None:
        if not HAS_HOU:
            self.status_label.setText("Solaris /stage wiring is only available inside Houdini.")
            return
        if self._profile is None:
            self.status_label.setText("Analyze a plate first before creating the Solaris Area Light.")
            return

        # If user hasn't picked any Light Area yet, auto-create one around hotspot #1
        if not self._light_regions:
            if not self._profile.hotspots:
                self.status_label.setText("No hotspot detected. Use 'Pick Light Area' to select light(s) on the preview.")
                return
            hs = self._profile.hotspots[0]
            u_c = ((hs.azimuth_deg / 360.0) + 0.5) % 1.0
            v_c = 0.5 - (hs.elevation_deg / 180.0)
            du = max(0.03, hs.radius_deg / 360.0)
            dv = max(0.04, hs.radius_deg / 180.0)
            self._light_regions.append(
                {
                    "rect": (max(0.0, u_c - du), max(0.0, v_c - dv), min(1.0, u_c + du), min(1.0, v_c + dv)),
                    "shape": self._current_light_shape(),
                }
            )
            self._regenerate_preview()

        # Always export/refresh HDRI EXR (inpainted) + 32-bit Light Texture EXR(s)
        self._on_export_exr()

        if not self._extracted_lights or len(self._last_light_exr_paths) != len(self._light_regions):
            self.status_label.setText("Failed to extract 32-bit float light texture EXR(s).")
            return

        try:
            import hou

            stage = hou.node("/stage")
            # Remove stale area lights beyond current count
            if stage is not None:
                for k in range(len(self._extracted_lights) + 1, 32):
                    stale = stage.node(f"hdri_maker_area_{k}")
                    if stale is not None:
                        stale.destroy()

            prev_node = self._last_dome_node
            if self._last_exr_path:
                prev_node = create_or_update_domelight(
                    hdri_exr_path=self._last_exr_path,
                    profile=self._profile,
                    set_display_flag=False,
                )
                self._last_dome_node = prev_node
            created = []
            for idx, ((_tex, info), tex_path) in enumerate(
                zip(self._extracted_lights, self._last_light_exr_paths), start=1
            ):
                is_last = idx == len(self._extracted_lights)
                area_light = create_or_update_area_light_from_hotspot(
                    info,
                    light_tex_exr_path=tex_path,
                    parent_dome=prev_node,
                    node_name=f"hdri_maker_area_{idx}",
                    prim_path=f"/lights/hdri_maker_area_{idx}",
                    set_display_flag=is_last,
                )
                prev_node = area_light
                created.append(area_light.path())

            tail_msg = f"last: {created[-1]}"
            if stage is not None and stage.node("hdri_maker_lookdev") is not None:
                tail_msg = f"{created[-1]} -> /stage/hdri_maker_lookdev"
            self.status_label.setText(
                f"Created {len(created)} Solaris Area Light(s) on /stage ({tail_msg})."
            )
        except Exception as exc:
            self.status_label.setText(f"Solaris Area Light error: {exc}")

    def _on_create_sun_light(self) -> None:
        if not HAS_HOU:
            self.status_label.setText("Solaris /stage wiring is only available inside Houdini.")
            return
        if self._profile is None or not self._profile.hotspots:
            self.status_label.setText("Analyze a plate with at least 1 detected hotspot before creating a Sun Light.")
            return
        try:
            hs = self._profile.hotspots[0]
            parent = self._last_dome_node
            if parent is None and self._last_exr_path:
                parent = create_or_update_domelight(
                    hdri_exr_path=self._last_exr_path,
                    profile=self._profile,
                    set_display_flag=False,
                )
                self._last_dome_node = parent
            sun = create_or_update_distant_sun_from_hotspot(hs, parent_node=parent)
            self.status_label.setText(
                f"Solaris Distant Sun ready at {sun.path()} (Az={hs.azimuth_deg:+.1f}°, El={hs.elevation_deg:+.1f}°)."
            )
        except Exception as exc:
            self.status_label.setText(f"Solaris Sun Light error: {exc}")

    def _on_export_usd_rig(self) -> None:
        if not HAS_HOU:
            self.status_label.setText("USD Rig export requires Houdini /stage.")
            return
        default_dir = os.path.dirname(self.out_path_edit.text().strip()) or os.path.expanduser("~")
        usd_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Solaris HDRI & Lookdev Rig (.usda / .usd)",
            os.path.join(default_dir, "hdri_maker_rig.usda"),
            "USD ASCII (*.usda);;USD Binary (*.usd);;USD Crate (*.usdc)",
        )
        if not usd_path:
            return
        try:
            written = export_solaris_rig_usd(usd_path)
            self.status_label.setText(f"Exported standalone Solaris USD rig: {written}")
        except Exception as exc:
            self.status_label.setText(f"USD export error: {exc}")

    @staticmethod
    def _downsample_for_lookdev(hdri: Optional[np.ndarray], max_width: int = 2048) -> Optional[np.ndarray]:
        """Cap cached full-export HDRI to max_width (e.g. 2K instead of 8K 384MB) for ground baking."""
        if hdri is None:
            return None
        w = hdri.shape[1]
        if w <= max_width:
            return hdri
        step = max(1, w // max_width)
        return np.ascontiguousarray(hdri[::step, ::step], dtype=np.float32)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

    def closeEvent(self, event) -> None:
        global _FLOATING_WINDOW
        self._auto_save_session()
        self._hdri_pixels = None
        self._last_hdri_full = None
        self._views = []
        self._extracted_lights = []
        self._frame_profiles = []
        self._profile = None
        self._thumb_cache.clear()
        self.preview_label.set_source_pixmap(None)
        if _FLOATING_WINDOW is self:
            _FLOATING_WINDOW = None
        super().closeEvent(event)

    def _log_debug(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] {msg}"
        self._debug_lines.append(line)
        if len(self._debug_lines) > 2000:
            self._debug_lines = self._debug_lines[-1500:]
        if self._debug_dialog is not None and self._debug_dialog.isVisible():
            self._debug_dialog.append_line(line)
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except Exception:
            pass

    def _on_open_debug_log(self) -> None:
        if self._debug_dialog is None:
            self._debug_dialog = DebugLogDialog(parent=self, log_path=DEBUG_LOG_PATH)
        self._debug_dialog.set_lines(self._debug_lines)
        self._debug_dialog.show()
        self._debug_dialog.raise_()
        self._debug_dialog.activateWindow()

    def _on_reload_code(self) -> None:
        self._auto_save_session()
        import hdri_maker
        hdri_maker.reload_all(relaunch=True)

    # ------------------------------------------------------------------
    # State Save / Load & Session Persistence
    # ------------------------------------------------------------------
    def get_state_dict(self) -> Dict[str, Any]:
        """Serialize all UI settings, multi-view plates, and inpaint rects."""
        mv_rows = []
        for row in range(self.multiview_table.rowCount()):
            item0 = self.multiview_table.item(row, 0)
            extra = item0.data(Qt.UserRole + 1) or {}
            mv_rows.append(
                {
                    "path": item0.data(Qt.UserRole),
                    "az": float(self.multiview_table.item(row, 1).text()),
                    "el": float(self.multiview_table.item(row, 2).text()),
                    "fov": float(self.multiview_table.item(row, 3).text()),
                    "colorspace": self.multiview_table.item(row, 4).text(),
                    "roll": float(extra.get("roll", 0.0)),
                    "scale_x": float(extra.get("scale_x", 1.0)),
                    "scale_y": float(extra.get("scale_y", 1.0)),
                    "warp_k1": float(extra.get("warp_k1", 0.0)),
                    "lens_type": str(extra.get("lens_type", "rectilinear")),
                    "exposure_ev": float(extra.get("exposure_ev", 0.0)),
                    "vignette_v1": float(extra.get("vignette_v1", 0.0)),
                    "exclude_rects": [list(r) for r in (extra.get("exclude_rects") or [])],
                }
            )
        return {
            "version": 1,
            "plate_path": self.plate_path_edit.text(),
            "colorspace": self.colorspace_combo.currentText(),
            "seq_mode": self.seq_mode_combo.currentText(),
            "use_range": self.use_range_chk.isChecked(),
            "start_frame": self.start_frame_spin.value(),
            "end_frame": self.end_frame_spin.value(),
            "step_frame": self.step_frame_spin.value(),
            "fov": self.fov_spin.value(),
            "cam_az": self.cam_az_spin.value(),
            "cam_el": self.cam_el_spin.value(),
            "max_hotspots": self.max_hotspots_spin.value(),
            "ldr_boost": self.ldr_boost_spin.value(),
            "multiview_plates": mv_rows,
            "resolution": self.res_combo.currentText(),
            "out_colorspace": self.out_cs_combo.currentText(),
            "exposure_ev": self.ev_spin.value(),
            "hotspot_boost": self.hs_boost_spin.value(),
            "hotspot_softness": self.hs_soft_spin.value(),
            "ambient_wrap": self.wrap_spin.value(),
            "sky_gain": self.sky_gain_spin.value(),
            "ground_gain": self.gnd_gain_spin.value(),
            "saturation": self.sat_spin.value(),
            "target_cct": self.temp_k_spin.value(),
            "cct_blend": self.temp_blend_spin.value(),
            "wb_gains": list(self._wb_gains),
            "macbeth_path": self._macbeth_path,
            "macbeth_colorspace": self._macbeth_colorspace,
            "plate_proj_blend": self.plate_blend_spin.value(),
            "yaw": self.yaw_spin.value(),
            "vertical_shift": self.vshift_spin.value(),
            "stitch_blend": self.stitch_blend_spin.value(),
            "enable_inpaint": self.inpaint_chk.isChecked(),
            "sky_blend": self.sky_blend_slider.value(),
            "ground_blend": self.gnd_blend_slider.value(),
            "sky_blur": self.sky_blur_slider.value(),
            "ground_blur": self.gnd_blur_slider.value(),
            "hdri_blur": self.hdri_blur_slider.value(),
            "physical_sky_fill": self.sky_fill_slider.value(),
            "highlight_recovery": self.highlight_recovery_slider.value(),
            "nadir_cleanup": self.nadir_cleanup_slider.value(),
            "temporal_stabilize": self.temporal_stab_slider.value(),
            "neutralize_environment": self.neutralize_env_chk.isChecked(),
            "diffuse_split": self.diffuse_split_chk.isChecked(),
            "preview_roughness": self.preview_roughness_slider.value(),
            "ab_split": self.ab_split_btn.isChecked(),
            "brush_strokes": self._brush_strokes,
            "sky_rect": list(self._sky_rect) if self._sky_rect else None,
            "ground_rect": list(self._ground_rect) if self._ground_rect else None,
            "light_regions": [
                {"rect": list(r["rect"]), "shape": r.get("shape", "rect")}
                for r in self._light_regions
            ],
            "light_shape": self.light_shape_combo.currentText(),
            "light_resolution": self.light_res_combo.currentText(),
            "light_region_blur": self.light_blur_slider.value(),
            "remove_hotspot": self.remove_hotspot_chk.isChecked(),
            "create_lookdev": self.lookdev_chk.isChecked(),
            "tripod_y": self.tripod_y_slider.value(),
            "ground_radius": self.ground_radius_slider.value(),
            "ground_proj_blur": self.ground_proj_blur_slider.value(),
            "ground_feather": self.ground_feather_slider.value(),
            "ground_emission": self.ground_emission_slider.value(),
            "render_engine": self.render_engine_combo.currentText(),
            "ground_shader_mode": self.ground_shader_combo.currentText(),
            "preview_ev": self.preview_ev_slider.value(),
            "preview_tonemap": self.preview_tonemap_combo.currentText(),
            "show_markers": self.show_markers_chk.isChecked(),
            "ui_scale": self.ui_scale_slider.value(),
            "ui_theme": self.theme_combo.currentText(),
            "out_path": self.out_path_edit.text(),
        }

    def apply_state_dict(self, data: Dict[str, Any], auto_analyze: bool = True) -> None:
        """Restore UI settings, multi-view plates, and inpaint rects from dict."""
        if not isinstance(data, dict):
            return
        self._suppress_auto_preview = True
        try:
            self._apply_state_dict_inner(data)
        finally:
            self._suppress_auto_preview = False

        if auto_analyze and (self.multiview_table.rowCount() > 0 or self.plate_path_edit.text().strip()):
            self._on_analyze_clicked()

    def _apply_state_dict_inner(self, data: Dict[str, Any]) -> None:
        self.plate_path_edit.setText(str(data.get("plate_path", "")))
        if "colorspace" in data:
            self.colorspace_combo.setCurrentText(str(data["colorspace"]))
        if "seq_mode" in data:
            self.seq_mode_combo.setCurrentText(str(data["seq_mode"]))
        self.use_range_chk.setChecked(bool(data.get("use_range", False)))
        self.start_frame_spin.setValue(int(data.get("start_frame", 1)))
        self.end_frame_spin.setValue(int(data.get("end_frame", 100)))
        self.step_frame_spin.setValue(int(data.get("step_frame", 1)))
        self.fov_spin.setValue(float(data.get("fov", 90.0)))
        self.cam_az_spin.setValue(float(data.get("cam_az", 0.0)))
        self.cam_el_spin.setValue(float(data.get("cam_el", 0.0)))
        self.max_hotspots_spin.setValue(int(data.get("max_hotspots", 5)))
        self.ldr_boost_spin.setValue(float(data.get("ldr_boost", 4.0)))

        self.multiview_table.setRowCount(0)
        for row_d in data.get("multiview_plates", []):
            p = row_d.get("path", "")
            if p:
                self._add_multiview_row(
                    p,
                    float(row_d.get("az", 0.0)),
                    float(row_d.get("el", 0.0)),
                    float(row_d.get("fov", 90.0)),
                    str(row_d.get("colorspace", "Auto")),
                    roll=float(row_d.get("roll", 0.0)),
                    scale_x=float(row_d.get("scale_x", 1.0)),
                    scale_y=float(row_d.get("scale_y", 1.0)),
                    warp_k1=float(row_d.get("warp_k1", 0.0)),
                    lens_type=str(row_d.get("lens_type", "rectilinear")),
                    exposure_ev=float(row_d.get("exposure_ev", 0.0)),
                    vignette_v1=float(row_d.get("vignette_v1", 0.0)),
                    exclude_rects=[tuple(r) for r in (row_d.get("exclude_rects") or [])],
                )

        if "resolution" in data:
            self.res_combo.setCurrentText(str(data["resolution"]))
        if "out_colorspace" in data:
            self.out_cs_combo.setCurrentText(str(data["out_colorspace"]))
        self.ev_spin.setValue(float(data.get("exposure_ev", 0.0)))
        self.hs_boost_spin.setValue(float(data.get("hotspot_boost", 0.0)))
        self.hs_soft_spin.setValue(float(data.get("hotspot_softness", 1.0)))
        self.wrap_spin.setValue(float(data.get("ambient_wrap", 0.55)))
        self.sky_gain_spin.setValue(float(data.get("sky_gain", 1.0)))
        self.gnd_gain_spin.setValue(float(data.get("ground_gain", 1.0)))
        self.sat_spin.setValue(float(data.get("saturation", 1.0)))
        self.temp_k_spin.setValue(float(data.get("target_cct", 6500.0)))
        self.temp_blend_spin.setValue(float(data.get("cct_blend", 0.0)))
        wbg = data.get("wb_gains")
        if isinstance(wbg, (list, tuple)) and len(wbg) == 3:
            self._wb_gains = tuple(float(v) for v in wbg)
        else:
            self._wb_gains = (1.0, 1.0, 1.0)
        self._macbeth_path = str(data.get("macbeth_path", ""))
        self._macbeth_colorspace = str(data.get("macbeth_colorspace", "Auto"))
        self.plate_blend_spin.setValue(float(data.get("plate_proj_blend", 1.0)))
        self.yaw_spin.setValue(float(data.get("yaw", 0.0)))
        self.vshift_spin.setValue(float(data.get("vertical_shift", 0.0)))
        self.stitch_blend_spin.setValue(float(data.get("stitch_blend", 0.35)))
        self.inpaint_chk.setChecked(bool(data.get("enable_inpaint", True)))
        self.sky_blend_slider.setValue(float(data.get("sky_blend", 0.35)))
        self.gnd_blend_slider.setValue(float(data.get("ground_blend", 0.35)))
        self.sky_blur_slider.setValue(float(data.get("sky_blur", 0.0)))
        self.gnd_blur_slider.setValue(float(data.get("ground_blur", 0.0)))
        self.hdri_blur_slider.setValue(float(data.get("hdri_blur", 0.0)))
        self.sky_fill_slider.setValue(float(data.get("physical_sky_fill", 0.0)))
        self.highlight_recovery_slider.setValue(float(data.get("highlight_recovery", 0.0)))
        self.nadir_cleanup_slider.setValue(float(data.get("nadir_cleanup", 0.0)))
        self.temporal_stab_slider.setValue(float(data.get("temporal_stabilize", 0.0)))
        self.neutralize_env_chk.setChecked(bool(data.get("neutralize_environment", False)))
        self.diffuse_split_chk.setChecked(bool(data.get("diffuse_split", False)))
        self.preview_roughness_slider.setValue(int(data.get("preview_roughness", 0)))
        self.ab_split_btn.setChecked(bool(data.get("ab_split", False)))
        self._brush_strokes = list(data.get("brush_strokes") or [])

        sr = data.get("sky_rect")
        self._sky_rect = tuple(sr) if (isinstance(sr, (list, tuple)) and len(sr) == 4) else None
        gr = data.get("ground_rect")
        self._ground_rect = tuple(gr) if (isinstance(gr, (list, tuple)) and len(gr) == 4) else None
        self._light_regions = []
        if isinstance(data.get("light_regions"), list):
            for entry in data["light_regions"]:
                if isinstance(entry, dict) and isinstance(entry.get("rect"), (list, tuple)) and len(entry["rect"]) == 4:
                    self._light_regions.append(
                        {"rect": tuple(float(v) for v in entry["rect"]), "shape": str(entry.get("shape", "rect"))}
                    )
        elif isinstance(data.get("light_rect"), (list, tuple)) and len(data["light_rect"]) == 4:
            self._light_regions.append(
                {"rect": tuple(float(v) for v in data["light_rect"]), "shape": "rect"}
            )
        if "light_shape" in data:
            self.light_shape_combo.setCurrentText(str(data["light_shape"]))
        if "light_resolution" in data:
            self.light_res_combo.setCurrentText(str(data["light_resolution"]))
        self.light_blur_slider.setValue(float(data.get("light_region_blur", 0.0)))
        self.remove_hotspot_chk.setChecked(bool(data.get("remove_hotspot", True)))
        self.lookdev_chk.setChecked(bool(data.get("create_lookdev", True)))
        self.tripod_y_slider.setValue(float(data.get("tripod_y", 1.5)))
        self.ground_radius_slider.setValue(float(data.get("ground_radius", 10.0)))
        self.ground_proj_blur_slider.setValue(float(data.get("ground_proj_blur", 0.0)))
        self.ground_feather_slider.setValue(float(data.get("ground_feather", 0.45)))
        self.ground_emission_slider.setValue(float(data.get("ground_emission", 0.5)))
        if "render_engine" in data:
            self.render_engine_combo.setCurrentText(str(data["render_engine"]))
        if "ground_shader_mode" in data:
            self.ground_shader_combo.setCurrentText(str(data["ground_shader_mode"]))

        self.preview_ev_slider.setValue(int(data.get("preview_ev", 0)))
        if "preview_tonemap" in data:
            self.preview_tonemap_combo.setCurrentText(str(data["preview_tonemap"]))
        self.show_markers_chk.setChecked(bool(data.get("show_markers", False)))
        if "ui_theme" in data and str(data["ui_theme"]) in UI_THEMES:
            self.theme_combo.setCurrentText(str(data["ui_theme"]))
        self.ui_scale_slider.setValue(float(data.get("ui_scale", 1.0)))
        self._apply_ui_scale(self.ui_scale_slider.value())
        if data.get("out_path"):
            self.out_path_edit.setText(str(data["out_path"]))

    def _auto_save_session(self) -> None:
        try:
            state_json = json.dumps(self.get_state_dict(), indent=2)
            with open(SESSION_STATE_PATH, "w", encoding="utf-8") as f:
                f.write(state_json)
            if HAS_HOU:
                try:
                    root_node = hou.node("/")
                    if root_node is not None:
                        root_node.setUserData("hdri_maker_state", state_json)
                except Exception:
                    pass
        except Exception:
            pass

    def _auto_analyze_restored_plates(self) -> None:
        if self.multiview_table.rowCount() > 0 or self.plate_path_edit.text().strip():
            self._on_analyze_clicked()

    def _auto_load_session(self) -> None:
        data = None
        if HAS_HOU:
            try:
                root_node = hou.node("/")
                if root_node is not None:
                    raw = root_node.userData("hdri_maker_state")
                    if raw:
                        data = json.loads(raw)
            except Exception:
                data = None
        if data is None and os.path.isfile(SESSION_STATE_PATH):
            try:
                with open(SESSION_STATE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = None
        if not isinstance(data, dict):
            return
        try:
            self.apply_state_dict(data, auto_analyze=False)
            has_plates = self.multiview_table.rowCount() > 0 or bool(self.plate_path_edit.text().strip())
            if has_plates:
                self.status_label.setText("Restored scene session; reloading plates and preview...")
                QtCore.QTimer.singleShot(50, self._auto_analyze_restored_plates)
            else:
                self.status_label.setText("Restored last session settings.")
        except Exception:
            pass

    def _on_save_state(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save HDRI Maker State",
            "hdri_maker_state.json",
            "HDRI Maker State (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.get_state_dict(), f, indent=2)
            self.status_label.setText(f"Saved state to: {path}")
        except Exception as exc:
            self.status_label.setText(f"Save state error: {exc}")

    def _on_load_state(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load HDRI Maker State",
            "",
            "HDRI Maker State (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.apply_state_dict(data, auto_analyze=True)
            self.status_label.setText(f"Loaded state from: {path}")
        except Exception as exc:
            self.status_label.setText(f"Load state error: {exc}")


_FLOATING_WINDOW = None


def launch_hdri_maker():
    """Launch HDRI Maker as a floating dialog parented to Houdini's main window."""
    global _FLOATING_WINDOW
    parent = hou.qt.mainWindow() if (HAS_HOU and hasattr(hou, "qt")) else None
    if _FLOATING_WINDOW is not None:
        try:
            _FLOATING_WINDOW.close()
        except Exception:
            pass
    _FLOATING_WINDOW = HDRIMakerWidget(parent=parent)
    _FLOATING_WINDOW.setAttribute(Qt.WA_DeleteOnClose, True)
    _FLOATING_WINDOW.setWindowFlags(Qt.Window)
    _FLOATING_WINDOW.resize(1340, 860)
    _FLOATING_WINDOW.show()
    return _FLOATING_WINDOW
