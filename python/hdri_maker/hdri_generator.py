"""
True 32-bit float Lat/Long (Equirectangular 2:1) HDRI synthesizer.

Synthesizes distortion-free spherical sky/ground gradients, azimuthal color wrap,
and unclamped physical HDR hotspots from a LightingProfile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .lighting_analyzer import (
    LightingProfile,
    PlateViewInput,
    cct_to_rgb,
    compute_luminance,
    project_multiview_plates_to_latlong,
    rgb_to_cct_kelvin,
)
from .plate_io import linear_to_srgb


@dataclass
class HDRISynthesisParams:
    """User-tunable parameters for 32-bit float Lat/Long HDRI generation."""

    width: int = 2048
    height: int = 1024
    exposure_ev: float = 0.0
    hotspot_boost: float = 0.0
    hotspot_softness: float = 1.0
    sky_gain: float = 1.0
    ground_gain: float = 1.0
    ambient_wrap: float = 0.55
    horizon_softness: float = 0.25
    temperature_kelvin: float = 0.0  # 0 = use plate CCT; >0 = override/blend target K
    temperature_blend: float = 0.0  # [0, 1] blend toward temperature_kelvin
    wb_gains: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Exact RGB white balance multiplier
    saturation: float = 1.0
    azimuth_offset_deg: float = 0.0
    vertical_shift_deg: float = 0.0
    plate_projection_blend: float = 1.0
    stitch_blend: float = 0.35
    ldr_highlight_boost: float = 4.0
    enable_inpaint: bool = True
    sky_blend: float = 0.35
    ground_blend: float = 0.35
    sky_blur: float = 0.0
    ground_blur: float = 0.0
    hdri_blur: float = 0.0
    sky_rect: Optional[Tuple[float, float, float, float]] = None
    ground_rect: Optional[Tuple[float, float, float, float]] = None
    light_rect: Optional[Tuple[float, float, float, float]] = None
    remove_hotspot_from_hdri: bool = True
    physical_sky_fill: float = 0.0
    highlight_recovery: float = 0.0
    nadir_cleanup: float = 0.0
    neutralize_environment: bool = False
    temporal_stabilize: float = 0.0
    sun_reconstruct: float = 0.0


@dataclass
class ExtractedAreaLightInfo:
    """Extracted Solaris Area Light geometry, energy, and baked 32-bit texture metadata."""

    azimuth_deg: float
    elevation_deg: float
    angular_width_deg: float
    angular_height_deg: float
    mean_intensity: float
    peak_intensity: float
    color_rgb: Tuple[float, float, float]
    cct_kelvin: float
    rect_uv: Tuple[float, float, float, float]
    shape: str = "rect"


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _smooth_horizontal_wrap(row: np.ndarray, radius: int) -> np.ndarray:
    """Fast wrap-around 1D box blur on a (W, 3) row using cumulative sum."""
    if radius <= 1:
        return row
    w = row.shape[0]
    r = min(radius, w // 2)
    padded = np.concatenate([row[-r:], row, row[:r]], axis=0)
    csum = np.cumsum(np.insert(padded, 0, 0.0, axis=0), axis=0)
    return ((csum[2 * r + 1 :] - csum[:w]) / float(2 * r + 1)).astype(np.float32)


def _separable_box_blur(
    pixels: np.ndarray,
    rx: int,
    ry: int,
    wrap_x: bool = True,
) -> np.ndarray:
    """Fast 3-pass separable box blur approximating a Gaussian kernel (with multi-scale proxy for large radii)."""
    h, w, c = pixels.shape
    rx = max(0, min(w // 2 - 1, int(rx)))
    ry = max(0, min(h // 2 - 1, int(ry)))
    if rx == 0 and ry == 0:
        return pixels

    # For large blur radii on high-res (4K/8K) arrays, downsample 2x-4x before box-blurring
    # and bilinearly upsample back (4x-16x faster & uses 1/16th RAM with identical smooth output).
    min_r = min(rx if rx > 0 else 9999, ry if ry > 0 else 9999)
    ds = 4 if (min_r >= 24 and min(h, w) >= 256) else (2 if (min_r >= 8 and min(h, w) >= 128) else 1)
    if ds > 1:
        dh = max(8, h // ds)
        dw = max(8, w // ds)
        ys_d = np.linspace(0.0, h - 1.0, dh, dtype=np.float32)
        xs_d = np.linspace(0.0, w - 1.0, dw, dtype=np.float32)
        y0_d = np.floor(ys_d).astype(np.int32)
        x0_d = np.floor(xs_d).astype(np.int32)
        small = pixels[np.ix_(y0_d, x0_d)]
        blurred_small = _separable_box_blur(
            small,
            max(1, int(round(rx / float(ds)))) if rx > 0 else 0,
            max(1, int(round(ry / float(ds)))) if ry > 0 else 0,
            wrap_x=wrap_x,
        )
        yy = np.linspace(0.0, dh - 1.0, h, dtype=np.float32)
        xx = np.linspace(0.0, dw - 1.0, w, dtype=np.float32)
        y0 = np.floor(yy).astype(np.int32)
        x0 = np.floor(xx).astype(np.int32)
        y1 = np.minimum(y0 + 1, dh - 1)
        x1 = np.minimum(x0 + 1, dw - 1)
        fy = (yy - y0)[:, None, None]
        fx = (xx - x0)[None, :, None]
        up = (
            blurred_small[np.ix_(y0, x0)] * (1.0 - fx) * (1.0 - fy)
            + blurred_small[np.ix_(y0, x1)] * fx * (1.0 - fy)
            + blurred_small[np.ix_(y1, x0)] * (1.0 - fx) * fy
            + blurred_small[np.ix_(y1, x1)] * fx * fy
        )
        return np.ascontiguousarray(np.maximum(up, 0.0), dtype=np.float32)

    out = pixels.astype(np.float64, copy=True)
    for _ in range(3):
        if rx >= 1:
            if wrap_x:
                pad_x = np.concatenate([out[:, -rx:, :], out, out[:, :rx, :]], axis=1)
            else:
                pad_x = np.pad(out, ((0, 0), (rx, rx), (0, 0)), mode="edge")
            zeros_x = np.zeros((h, 1, c), dtype=np.float64)
            cs_x = np.cumsum(np.concatenate([zeros_x, pad_x], axis=1), axis=1)
            out = (cs_x[:, 2 * rx + 1 :, :] - cs_x[:, :w, :]) / float(2 * rx + 1)
        if ry >= 1:
            pad_y = np.pad(out, ((ry, ry), (0, 0), (0, 0)), mode="edge")
            zeros_y = np.zeros((1, w, c), dtype=np.float64)
            cs_y = np.cumsum(np.concatenate([zeros_y, pad_y], axis=0), axis=0)
            out = (cs_y[2 * ry + 1 :, :, :] - cs_y[:h, :, :]) / float(2 * ry + 1)

    return np.ascontiguousarray(np.maximum(out, 0.0), dtype=np.float32)


def blur_latlong_hdri(hdri_pixels: np.ndarray, blur_amount: float) -> np.ndarray:
    """
    Fast, energy-preserving 2D separable Gaussian-approximate blur for Lat/Long HDRIs.
    Wraps seamlessly across 360 deg horizontally and clamps cleanly at zenith/nadir.
    """
    b = float(np.clip(blur_amount, 0.0, 1.0))
    if b <= 1e-4:
        return hdri_pixels

    h, w, _ = hdri_pixels.shape
    rx = max(1, int(round(b * (w / 24.0))))
    ry = max(1, int(round(b * (h / 24.0))))
    return _separable_box_blur(hdri_pixels, rx, ry, wrap_x=True)


def _build_seamless_tile(patch: np.ndarray, mirror_vertical: bool = False) -> np.ndarray:
    """Create a horizontally seamless (optionally 2x2 ping-pong) tile from an (H, W, 3) patch."""
    h_mir = np.concatenate([patch, patch[:, ::-1, :]], axis=1)
    if mirror_vertical:
        return np.concatenate([h_mir, h_mir[::-1, :, :]], axis=0)
    return h_mir


def _tile_fill_region(
    tile: np.ndarray,
    target_h: int,
    target_w: int,
    flip_vertical: bool = False,
) -> np.ndarray:
    """Stretch/tile a seamless 2D patch into (target_h, target_w, 3)."""
    th, tw, _ = tile.shape
    if th < 1 or tw < 1 or target_h < 1 or target_w < 1:
        return np.zeros((max(target_h, 1), max(target_w, 1), 3), dtype=np.float32)

    # Stretch vertically to fill cap height, tile horizontally across 360 deg
    ys = np.linspace(0, th - 1, target_h, dtype=np.int32)
    if flip_vertical:
        ys = ys[::-1]
    n_repeats = max(1, int(round(target_w / float(max(tw, 1)))))
    xs = (np.arange(target_w, dtype=np.int32) * max(1, (tw * n_repeats) // target_w)) % tw
    return tile[np.ix_(ys, xs)].astype(np.float32)


def _fill_horizontal_gaps_crossblend(
    proj_rgb: np.ndarray,
    cov_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bridge all horizontal gaps between multiple stitched plates on each covered
    row by mirroring and cross-fading adjacent plate edges across the gap.
    """
    h, w, _ = proj_rgb.shape
    out = proj_rgb.copy()
    out_cov = cov_mask.copy()

    valid_mask = cov_mask > 0.25
    rows_to_fill = np.where(np.any(valid_mask, axis=1) & ~np.all(valid_mask, axis=1))[0]

    for y in rows_to_fill:
        row_c = cov_mask[y]
        valid = valid_mask[y]

        # Find contiguous uncovered gaps (wrap-around aware)
        inv = ~valid
        # Roll so index 0 is inside a valid pixel to avoid split wrap-around logic
        first_valid = int(np.argmax(valid))
        inv_r = np.roll(inv, -first_valid)
        row_r = np.roll(proj_rgb[y], -first_valid, axis=0)
        c_r = np.roll(row_c, -first_valid)

        padded = np.concatenate([[False], inv_r, [False]])
        starts = np.where(~padded[:-1] & padded[1:])[0]
        ends = np.where(padded[:-1] & ~padded[1:])[0] - 1

        filled_r = row_r.copy()
        filled_c = c_r.copy()

        for gs, ge in zip(starts, ends):
            glen = (ge - gs) + 1
            xl = (gs - 1) % w
            xr = (ge + 1) % w
            # Mirror-sample from left covered side and right covered side
            offsets = np.arange(1, glen + 1, dtype=np.int32)
            left_idx = (xl - offsets) % w
            right_idx = (xr + (glen - offsets)) % w

            left_samp = row_r[left_idx]
            right_samp = row_r[right_idx]
            t = (offsets / float(glen + 1)).astype(np.float32)
            t = (t * t * (3.0 - 2.0 * t))[:, None]

            filled_r[gs : ge + 1] = left_samp * (1.0 - t) + right_samp * t
            filled_c[gs : ge + 1] = min(float(c_r[xl]), float(c_r[xr]))

        out[y] = np.roll(filled_r, first_valid, axis=0)
        out_cov[y] = np.roll(filled_c, first_valid, axis=0)

    return out, out_cov


def _smooth_pole_cap(
    cap: np.ndarray,
    is_zenith: bool,
    blur_amount: float,
) -> np.ndarray:
    """
    Apply 2D wrap-safe smoothing and converge the zenith/nadir pole row
    smoothly to a uniform pole color so chrome spheres show zero concentric rings.
    """
    ch, cw, _ = cap.shape
    if ch < 2:
        return cap

    b = float(np.clip(blur_amount, 0.0, 1.0))
    # Always apply a mild baseline vertical+horizontal smoothing to eliminate
    # latitude banding from stretched tile rows; scale up with Sky/Ground Blur.
    eff_b = max(0.18, b)
    rx = max(2, int(round(eff_b * (cw / 16.0))))
    ry = max(2, int(round(eff_b * max(8, ch) * 0.28)))
    smoothed = _separable_box_blur(cap, rx, ry, wrap_x=True)

    # Smoothly converge toward uniform zenith/nadir pole color over outer 75% of cap
    sample_rows = max(2, ch // 4)
    if is_zenith:
        pole_rgb = np.mean(smoothed[:sample_rows], axis=(0, 1), keepdims=True)
        dist = np.arange(ch, dtype=np.float32)
    else:
        pole_rgb = np.mean(smoothed[-sample_rows:], axis=(0, 1), keepdims=True)
        dist = np.arange(ch - 1, -1, -1, dtype=np.float32)

    pole_w = (1.0 - _smoothstep(0.0, max(2.0, ch * (0.55 + 0.35 * b)), dist))[:, None, None]
    return (smoothed * (1.0 - pole_w) + pole_rgb * pole_w).astype(np.float32)


def _mirror_extend_rows(
    proj_rgb: np.ndarray,
    cov_mask: np.ndarray,
    base_hdri: np.ndarray,
    enable_inpaint: bool = True,
    sky_rect: Optional[Tuple[float, float, float, float]] = None,
    ground_rect: Optional[Tuple[float, float, float, float]] = None,
    sky_blend: float = 0.35,
    ground_blend: float = 0.35,
    sky_blur: float = 0.0,
    ground_blur: float = 0.0,
) -> np.ndarray:
    """
    Bridge horizontal gaps between stitched plates and fill top (zenith) and
    bottom (nadir) completely with zero dark spaces—using either selected
    sky/ground texture inpainting or stretched/smoothed sky/ground edges,
    with configurable vertical sky_blend / ground_blend feathering.
    """
    h, w, _ = proj_rgb.shape
    row_has_cov = np.any(cov_mask > 0.15, axis=1)
    if not np.any(row_has_cov):
        return base_hdri.copy()

    # 1. Cross-blend all horizontal gaps between plates so every covered row is 360 deg
    filled, filled_cov = _fill_horizontal_gaps_crossblend(proj_rgb, cov_mask)

    ys_cov = np.where(row_has_cov)[0]
    y_top, y_bot = int(ys_cov[0]), int(ys_cov[-1])
    band_h = max(1, y_bot - y_top + 1)
    sky_b = float(np.clip(sky_blend, 0.0, 1.0))
    gnd_b = float(np.clip(ground_blend, 0.0, 1.0))

    # Feather inward into the plate band so there is zero hard horizontal seam at y_top / y_bot
    blend_top_end = min(y_bot, y_top + max(6, int(band_h * (0.04 + 0.45 * sky_b))))
    blend_bot_start = max(y_top, y_bot - max(6, int(band_h * (0.04 + 0.45 * gnd_b))))

    # 2. Extract Sky and Ground texture patches (user-selected rect or auto-band)
    def _extract_patch(rect, default_y0, default_y1):
        if rect is not None:
            u0, v0, u1, v1 = rect
            x0 = int(np.clip(min(u0, u1) * w, 0, w - 2))
            x1 = int(np.clip(max(u0, u1) * w, x0 + 2, w))
            py0 = int(np.clip(min(v0, v1) * h, 0, h - 2))
            py1 = int(np.clip(max(v0, v1) * h, py0 + 2, h))
            return filled[py0:py1, x0:x1]
        return filled[default_y0:default_y1, :]

    y_top_in = min(y_bot, y_top + 2)
    y_bot_in = max(y_top, y_bot - 2)

    if blend_top_end > 0:
        cap_h = blend_top_end + 1
        sky_cap = np.empty((cap_h, w, 3), dtype=np.float32)
        if enable_inpaint:
            auto_strip = max(4, min(16, band_h // 16))
            sky_patch = _extract_patch(
                sky_rect,
                y_top_in,
                min(y_bot, y_top_in + auto_strip),
            )
            sky_tile = _build_seamless_tile(sky_patch, mirror_vertical=False)
            sky_fill = _tile_fill_region(sky_tile, cap_h, w, flip_vertical=False)
            top_edge_smooth = _smooth_horizontal_wrap(filled[y_top_in], max(8, w // 20))
            for y in range(cap_h):
                pole_t = (y / float(max(blend_top_end, 1))) ** 0.8
                blur_r = max(4, int((1.0 - pole_t) * (w // 10) + sky_b * (w // 18)))
                row_tex = _smooth_horizontal_wrap(sky_fill[y], blur_r)
                seam_t = min(1.0, float(y) / float(max(y_top_in, 1)))
                seam_w = 0.35 * sky_b * (seam_t * seam_t * (3.0 - 2.0 * seam_t))
                sky_cap[y] = row_tex * (1.0 - seam_w) + top_edge_smooth * seam_w
        else:
            anchor = _smooth_horizontal_wrap(filled[y_top_in], max(8, w // 32))
            for y in range(cap_h):
                t = (y / float(max(blend_top_end, 1))) ** 1.2
                sky_cap[y] = _smooth_horizontal_wrap(anchor, max(4, int((1.0 - t) * (w // 8))))

        sky_cap = _smooth_pole_cap(sky_cap, is_zenith=True, blur_amount=sky_blur)

        filled[: y_top_in + 1] = sky_cap[: y_top_in + 1]
        if blend_top_end > y_top_in:
            ys_f = np.arange(y_top_in + 1, cap_h, dtype=np.float32)
            edge_t = _smoothstep(float(y_top_in), float(blend_top_end), ys_f)[:, None, None]
            filled[y_top_in + 1 : cap_h] = (
                sky_cap[y_top_in + 1 : cap_h] * (1.0 - edge_t)
                + filled[y_top_in + 1 : cap_h] * edge_t
            )

    if blend_bot_start < h - 1:
        nadir_h = h - blend_bot_start
        gnd_cap = np.empty((nadir_h, w, 3), dtype=np.float32)
        if enable_inpaint:
            auto_strip = max(4, min(16, band_h // 16))
            gnd_patch = _extract_patch(
                ground_rect,
                max(y_top, y_bot_in - auto_strip),
                y_bot_in,
            )
            gnd_tile = _build_seamless_tile(gnd_patch, mirror_vertical=False)
            gnd_fill = _tile_fill_region(gnd_tile, nadir_h, w, flip_vertical=False)
            bot_edge_smooth = _smooth_horizontal_wrap(filled[y_bot_in], max(8, w // 20))
            for idx, y in enumerate(range(blend_bot_start, h)):
                pole_t = (1.0 - idx / float(max(nadir_h - 1, 1))) ** 0.8
                blur_r = max(4, int((1.0 - pole_t) * (w // 10) + gnd_b * (w // 18)))
                row_tex = _smooth_horizontal_wrap(gnd_fill[idx], blur_r)
                seam_t = min(1.0, float(h - 1 - y) / float(max(h - 1 - y_bot_in, 1)))
                seam_w = 0.35 * gnd_b * (seam_t * seam_t * (3.0 - 2.0 * seam_t))
                gnd_cap[idx] = row_tex * (1.0 - seam_w) + bot_edge_smooth * seam_w
        else:
            anchor = _smooth_horizontal_wrap(filled[y_bot_in], max(8, w // 32))
            for idx in range(nadir_h):
                t = (1.0 - idx / float(max(nadir_h - 1, 1))) ** 1.2
                gnd_cap[idx] = _smooth_horizontal_wrap(anchor, max(4, int((1.0 - t) * (w // 8))))

        gnd_cap = _smooth_pole_cap(gnd_cap, is_zenith=False, blur_amount=ground_blur)

        split_idx = y_bot_in - blend_bot_start
        filled[y_bot_in:] = gnd_cap[split_idx:]
        if split_idx > 0:
            ys_f = np.arange(blend_bot_start, y_bot_in, dtype=np.float32)
            edge_t = (1.0 - _smoothstep(float(blend_bot_start), float(y_bot_in), ys_f))[:, None, None]
            filled[blend_bot_start:y_bot_in] = (
                gnd_cap[:split_idx] * (1.0 - edge_t)
                + filled[blend_bot_start:y_bot_in] * edge_t
            )

    return filled


def apply_sky_ground_inpaint_to_hdri(
    hdri: np.ndarray,
    sky_rect: Optional[Tuple[float, float, float, float]] = None,
    ground_rect: Optional[Tuple[float, float, float, float]] = None,
    sky_blend: float = 0.35,
    ground_blend: float = 0.35,
    sky_blur: float = 0.0,
    ground_blur: float = 0.0,
) -> np.ndarray:
    """
    Apply newly selected Sky and/or Ground region inpainting directly onto an
    existing baked 32-bit float Lat/Long HDRI without discarding prior baked edits.
    """
    if sky_rect is None and ground_rect is None:
        return hdri
    out = np.asarray(hdri, dtype=np.float32).copy()
    h, w, _ = out.shape

    def _extract_patch(rect: Tuple[float, float, float, float]) -> np.ndarray:
        u0, v0, u1, v1 = rect
        x0 = int(np.clip(min(u0, u1) * w, 0, w - 2))
        x1 = int(np.clip(max(u0, u1) * w, x0 + 2, w))
        py0 = int(np.clip(min(v0, v1) * h, 0, h - 2))
        py1 = int(np.clip(max(v0, v1) * h, py0 + 2, h))
        return out[py0:py1, x0:x1]

    if sky_rect is not None:
        sky_b = float(np.clip(sky_blend, 0.0, 1.0))
        cap_h = max(8, int(round(h * (0.18 + 0.27 * sky_b))))
        cap_h = min(h // 2, cap_h)
        sky_patch = _extract_patch(sky_rect)
        sky_tile = _build_seamless_tile(sky_patch, mirror_vertical=False)
        sky_fill = _tile_fill_region(sky_tile, cap_h, w, flip_vertical=False)
        sky_cap = _smooth_pole_cap(sky_fill, is_zenith=True, blur_amount=sky_blur)
        ys = np.arange(cap_h, dtype=np.float32)
        blend_start = max(0, int(round(cap_h * 0.45)))
        alpha = (1.0 - _smoothstep(float(blend_start), float(max(blend_start + 1, cap_h - 1)), ys))[:, None, None]
        out[:cap_h] = out[:cap_h] * (1.0 - alpha) + sky_cap * alpha

    if ground_rect is not None:
        gnd_b = float(np.clip(ground_blend, 0.0, 1.0))
        nadir_h = max(8, int(round(h * (0.18 + 0.27 * gnd_b))))
        nadir_h = min(h // 2, nadir_h)
        start_y = h - nadir_h
        gnd_patch = _extract_patch(ground_rect)
        gnd_tile = _build_seamless_tile(gnd_patch, mirror_vertical=False)
        gnd_fill = _tile_fill_region(gnd_tile, nadir_h, w, flip_vertical=False)
        gnd_cap = _smooth_pole_cap(gnd_fill, is_zenith=False, blur_amount=ground_blur)
        ys = np.arange(nadir_h, dtype=np.float32)
        blend_end = min(nadir_h - 1, int(round(nadir_h * 0.55)))
        alpha = _smoothstep(0.0, float(max(1, blend_end)), ys)[:, None, None]
        out[start_y:] = out[start_y:] * (1.0 - alpha) + gnd_cap * alpha

    return np.ascontiguousarray(np.maximum(out, 0.0), dtype=np.float32)


_PROJ_CACHE: Optional[Tuple[tuple, np.ndarray, np.ndarray]] = None


def generate_latlong_hdri(
    profile: LightingProfile,
    params: HDRISynthesisParams | None = None,
    views: Optional[List[PlateViewInput]] = None,
) -> np.ndarray:
    """
    Generate a 32-bit float (H, W, 3) Lat/Long (2:1) HDRI in Linear Rec.709.

    Args:
        profile: Extracted LightingProfile from plate analysis.
        params: Optional synthesis tuning parameters.

    Returns:
        np.ndarray of shape (height, width, 3), dtype np.float32, unclamped HDR.
    """
    p = params or HDRISynthesisParams()
    w = int(p.width)
    h = int(p.height)

    # Equirectangular spherical coordinates
    u = (np.arange(w, dtype=np.float32) + 0.5) / float(w)
    v = (np.arange(h, dtype=np.float32) + 0.5) / float(h)

    phi = (u - 0.5) * (2.0 * math.pi) - math.radians(p.azimuth_offset_deg)  # [-pi, pi]
    theta = (0.5 - v) * math.pi  # [+pi/2 (zenith), -pi/2 (nadir)]

    sin_theta = np.sin(theta)[:, None]  # (H, 1)
    cos_theta = np.cos(theta)[:, None]  # (H, 1)
    sin_phi = np.sin(phi)[None, :]      # (1, W)
    cos_phi = np.cos(phi)[None, :]      # (1, W)

    # 1. Vertical hemispherical gradient (Zenith -> Sky -> Horizon -> Ground -> Nadir)
    zenith = np.array(profile.zenith_color, dtype=np.float32) * p.sky_gain
    sky = np.array(profile.sky_color, dtype=np.float32) * p.sky_gain
    horizon = np.array(profile.horizon_color, dtype=np.float32)
    ground = np.array(profile.ground_color, dtype=np.float32) * p.ground_gain
    nadir = np.array(profile.nadir_color, dtype=np.float32) * p.ground_gain

    elev = theta[:, None] / (0.5 * math.pi)  # +1 zenith, 0 horizon, -1 nadir
    soft = max(0.05, float(p.horizon_softness))

    # Build vertical profile (H, 3)
    w_zen = _smoothstep(0.30, 0.95, elev)[..., None]
    w_sky = _smoothstep(0.0, soft, elev)[..., None]
    w_gnd = _smoothstep(0.0, soft, -elev)[..., None]
    w_nad = _smoothstep(0.30, 0.95, -elev)[..., None]

    upper = sky * (1.0 - w_zen) + zenith * w_zen
    lower = ground * (1.0 - w_nad) + nadir * w_nad

    # Blend around horizon
    hor_blend = horizon * (1.0 - w_sky - w_gnd)
    # Normalize weights cleanly
    total_w = (1.0 - w_sky) * (1.0 - w_gnd) + w_sky + w_gnd
    vert_rgb = (horizon * ((1.0 - w_sky) * (1.0 - w_gnd)) + upper * w_sky + lower * w_gnd) / np.maximum(total_w, 1e-5)

    hdri = np.broadcast_to(vert_rgb, (h, w, 3)).copy()

    # 2. Azimuthal environment color wrap (gives directional ambient bounce)
    if p.ambient_wrap > 0.0 and profile.azimuthal_strip is not None and len(profile.azimuthal_strip) > 0:
        strip = np.asarray(profile.azimuthal_strip, dtype=np.float32)
        n_strip = len(strip)
        # Map u in [0, 1] seamlessly around azimuth using cosine-weighted wrap
        # Center of plate (u=0.5) maps to strip center; back side wraps smoothly
        az_u = ((u - p.azimuth_offset_deg / 360.0) % 1.0)
        # Mirror/wrap smoothly so there is no discontinuity at back seam
        sym_u = 1.0 - np.abs(2.0 * az_u - 1.0)  # 0 at back, 1 at front
        sample_idx = sym_u * (n_strip - 1)
        i0 = np.floor(sample_idx).astype(np.int32)
        i1 = np.minimum(i0 + 1, n_strip - 1)
        frac = (sample_idx - i0)[:, None]
        az_colors = strip[i0] * (1.0 - frac) + strip[i1] * frac  # (W, 3)

        # Concentrate azimuthal wrap near horizon/mid-elevation
        band_mask = np.exp(-0.5 * (elev / 0.55) ** 2)[..., None]  # (H, 1, 1)
        wrap_alpha = float(np.clip(p.ambient_wrap, 0.0, 1.0)) * band_mask
        hdri = hdri * (1.0 - wrap_alpha) + az_colors[None, :, :] * wrap_alpha

    # 2b. Full-resolution spherical plate stitching & out-of-frame extension
    if p.plate_projection_blend > 0.0:
        proj_rgb = None
        cov_w = None
        if views:
            global _PROJ_CACHE
            cache_key = None
            if w <= 2048:
                views_sig = tuple(
                    (
                        id(v.pixels),
                        float(v.cam_azimuth_deg),
                        float(v.cam_elevation_deg),
                        float(v.fov_h_deg),
                        float(v.cam_roll_deg),
                        float(v.scale_x),
                        float(v.scale_y),
                        float(v.warp_k1),
                        str(getattr(v, "lens_type", "rectilinear")),
                        float(getattr(v, "exposure_ev", 0.0)),
                        float(getattr(v, "vignette_v1", 0.0)),
                        tuple(getattr(v, "exclude_rects", ()) or ()),
                    )
                    for v in views
                )
                cache_key = (
                    views_sig,
                    w,
                    h,
                    round(float(p.ldr_highlight_boost), 4),
                    round(float(p.stitch_blend), 4),
                    round(float(p.vertical_shift_deg), 4),
                )
            if cache_key is not None and _PROJ_CACHE is not None and _PROJ_CACHE[0] == cache_key:
                proj_rgb = _PROJ_CACHE[1].copy()
                cov_w = _PROJ_CACHE[2].copy()
            else:
                proj_rgb, cov_w = project_multiview_plates_to_latlong(
                    views,
                    out_width=w,
                    out_height=h,
                    ldr_highlight_boost=p.ldr_highlight_boost,
                    edge_feather=p.stitch_blend,
                    vertical_shift_deg=p.vertical_shift_deg,
                )
                if cache_key is not None:
                    _PROJ_CACHE = (cache_key, proj_rgb.copy(), cov_w.copy())
        elif profile.spherical_plate_map is not None and profile.spherical_coverage_mask is not None:
            s_map = profile.spherical_plate_map
            s_cov = profile.spherical_coverage_mask
            sh, sw = s_map.shape[:2]
            yy_raw = np.linspace(0, sh - 1, h, dtype=np.float32) + (float(p.vertical_shift_deg) / 180.0) * max(1, sh - 1)
            valid_y = (yy_raw >= 0.0) & (yy_raw <= (sh - 1))
            yy = np.clip(yy_raw, 0.0, sh - 1)
            xx = np.linspace(0, sw - 1, w, dtype=np.float32)
            y0 = np.floor(yy).astype(np.int32)
            x0 = np.floor(xx).astype(np.int32)
            y1 = np.minimum(y0 + 1, sh - 1)
            x1 = np.minimum(x0 + 1, sw - 1)
            fy = (yy - y0)[:, None, None]
            fx = (xx - x0)[None, :, None]
            proj_rgb = (
                s_map[np.ix_(y0, x0)] * (1.0 - fx) * (1.0 - fy)
                + s_map[np.ix_(y0, x1)] * fx * (1.0 - fy)
                + s_map[np.ix_(y1, x0)] * (1.0 - fx) * fy
                + s_map[np.ix_(y1, x1)] * fx * fy
            )
            cov_w = (
                s_cov[np.ix_(y0, x0)] * (1.0 - fx[..., 0]) * (1.0 - fy[..., 0])
                + s_cov[np.ix_(y0, x1)] * fx[..., 0] * (1.0 - fy[..., 0])
                + s_cov[np.ix_(y1, x0)] * (1.0 - fx[..., 0]) * fy[..., 0]
                + s_cov[np.ix_(y1, x1)] * fx[..., 0] * fy[..., 0]
            ) * valid_y[:, None].astype(np.float32)

        if proj_rgb is not None and cov_w is not None:
            if abs(p.azimuth_offset_deg) > 1e-4:
                shift_px = int(round((p.azimuth_offset_deg / 360.0) * w)) % w
                if shift_px != 0:
                    proj_rgb = np.roll(proj_rgb, shift_px, axis=1)
                    cov_w = np.roll(cov_w, shift_px, axis=1)

            stitched = _mirror_extend_rows(
                proj_rgb,
                cov_w,
                hdri,
                enable_inpaint=p.enable_inpaint,
                sky_rect=p.sky_rect,
                ground_rect=p.ground_rect,
                sky_blend=p.sky_blend,
                ground_blend=p.ground_blend,
                sky_blur=p.sky_blur,
                ground_blur=p.ground_blur,
            )
            # Apply Sky Gain, Ground Gain, and subtle Ambient Wrap to stitched HDRI too
            # so these sliders always have a clear, physical effect even at Plate Proj Blend = 1.0
            if abs(p.sky_gain - 1.0) > 1e-4:
                sky_mask = _smoothstep(0.0, 0.25, elev)[..., None]
                stitched = stitched * (1.0 + (float(p.sky_gain) - 1.0) * sky_mask)
            if abs(p.ground_gain - 1.0) > 1e-4:
                gnd_mask = _smoothstep(0.0, 0.25, -elev)[..., None]
                stitched = stitched * (1.0 + (float(p.ground_gain) - 1.0) * gnd_mask)
            if abs(p.ambient_wrap - 0.55) > 1e-4 and profile.azimuthal_strip is not None and len(profile.azimuthal_strip) > 0:
                wrap_delta = float(np.clip(p.ambient_wrap, 0.0, 1.0)) - 0.55
                band_mask = np.exp(-0.5 * (elev / 0.55) ** 2)[..., None]
                if wrap_delta > 0.0:
                    stitched = stitched * (1.0 - 0.35 * wrap_delta * band_mask) + az_colors[None, :, :] * (0.35 * wrap_delta * band_mask)
                else:
                    row_mean = np.mean(stitched, axis=1, keepdims=True)
                    stitched = stitched * (1.0 + 0.5 * wrap_delta * band_mask) + row_mean * (-0.5 * wrap_delta * band_mask)

            blend = float(np.clip(p.plate_projection_blend, 0.0, 1.0))
            hdri = hdri * (1.0 - blend) + stitched * blend
            if p.physical_sky_fill > 1e-4:
                sky_fill_amt = float(np.clip(p.physical_sky_fill, 0.0, 1.0))
                uncovered_sky = (1.0 - np.clip(cov_w, 0.0, 1.0)) * _smoothstep(-0.04, 0.12, elev) * sky_fill_amt
                if np.max(uncovered_sky) > 1e-4:
                    dir_x_s = cos_theta * sin_phi
                    dir_y_s = np.broadcast_to(sin_theta, (h, w))
                    dir_z_s = cos_theta * cos_phi
                    zenith_t = np.clip(np.sin(np.maximum(elev, 0.0)), 0.0, 1.0)
                    zen_col = np.array(profile.zenith_color, dtype=np.float32)
                    hor_col = np.array(profile.horizon_color, dtype=np.float32)
                    # Rayleigh-like horizon-to-zenith sky gradient
                    rayleigh = hor_col * np.power(1.0 - zenith_t, 2.2)[..., None] + zen_col * (1.0 - np.power(1.0 - zenith_t, 2.2))[..., None]
                    if profile.hotspots:
                        sun = profile.hotspots[0]
                        s_az = math.radians(sun.azimuth_deg)
                        s_el = math.radians(max(5.0, sun.elevation_deg))
                        sun_col = np.array(cct_to_rgb(sun.cct_kelvin), dtype=np.float32)
                        sun_peak = float(max(sun.peak_intensity, 4.0))
                    else:
                        s_az = math.radians(p.azimuth_offset_deg)
                        s_el = math.radians(45.0)
                        sun_col = np.array(cct_to_rgb(profile.key_cct_kelvin), dtype=np.float32)
                        sun_peak = 8.0
                    sx, sy, sz = math.cos(s_el) * math.sin(s_az), math.sin(s_el), math.cos(s_el) * math.cos(s_az)
                    cos_sun = np.clip(dir_x_s * sx + dir_y_s * sy + dir_z_s * sz, -1.0, 1.0)
                    mie_halo = np.power(np.maximum(cos_sun, 0.0), 12.0)[..., None] * sun_col * 0.65
                    sun_disc = np.exp(-0.5 * (np.arccos(np.clip(cos_sun, -1.0, 1.0)) / math.radians(1.2)) ** 2)[..., None] * sun_col * sun_peak
                    phys_sky = rayleigh + mie_halo + sun_disc
                    hdri = hdri * (1.0 - uncovered_sky[..., None]) + phys_sky * uncovered_sky[..., None]
            del stitched, proj_rgb, cov_w
    elif abs(p.azimuth_offset_deg) > 1e-4:
        # When plate_projection_blend == 0, az_colors and dir_x/y/z already incorporate azimuth_offset_deg
        pass

    # 3. Optional Extra 3D Spherical Hotspot Glow (only when hotspot_boost > 0)
    if p.hotspot_boost > 1e-4:
        dir_x = cos_theta * sin_phi
        dir_y = np.broadcast_to(sin_theta, (h, w))
        dir_z = cos_theta * cos_phi
        for hs in profile.hotspots:
            az_rad = math.radians(hs.azimuth_deg)
            el_rad = math.radians(hs.elevation_deg)
            lx = math.cos(el_rad) * math.sin(az_rad)
            ly = math.sin(el_rad)
            lz = math.cos(el_rad) * math.cos(az_rad)

            cos_gamma = np.clip(dir_x * lx + dir_y * ly + dir_z * lz, -1.0, 1.0)
            gamma = np.arccos(cos_gamma)

            sigma = math.radians(max(0.8, hs.radius_deg * p.hotspot_softness))
            core = np.exp(-0.5 * (gamma / sigma) ** 2)
            halo = 0.16 * np.exp(-0.5 * (gamma / (sigma * 3.2)) ** 2)

            energy = float(hs.peak_intensity) * float(p.hotspot_boost)
            hs_rgb = np.array(hs.color_rgb, dtype=np.float32) * energy
            hdri += (core + halo)[..., None] * hs_rgb[None, None, :]

    # 4. Temperature shift / white-balance adjustment
    if p.temperature_kelvin > 0.0 and p.temperature_blend > 0.0:
        target_rgb = np.array(cct_to_rgb(p.temperature_kelvin), dtype=np.float32)
        src_rgb = np.array(cct_to_rgb(profile.overall_cct_kelvin), dtype=np.float32)
        ratio = target_rgb / np.maximum(src_rgb, 1e-4)
        # Normalize luminance of ratio so temperature shift doesn't change EV
        ratio_lum = float(0.2126 * ratio[0] + 0.7152 * ratio[1] + 0.0722 * ratio[2])
        if ratio_lum > 1e-5:
            ratio /= ratio_lum
        blend = float(np.clip(p.temperature_blend, 0.0, 1.0))
        mult = (1.0 - blend) + blend * ratio
        hdri *= mult[None, None, :]

    # 4b. Macbeth / Neutral Patch 3-channel White Balance gains (luminance-preserving)
    if p.wb_gains is not None and len(p.wb_gains) == 3:
        wb = np.array(p.wb_gains, dtype=np.float32)
        if float(np.max(np.abs(wb - 1.0))) > 1e-4:
            hdri *= wb[None, None, :]

    # 4c. Optional Clipped Highlight / Sun Core Recovery
    if p.highlight_recovery > 1e-4:
        hdri = recover_clipped_highlights(hdri, strength=p.highlight_recovery)

    # 4d. Optional Seam-Aware Nadir / Tripod Cleanup
    if p.nadir_cleanup > 1e-4:
        hdri = _apply_nadir_cleanup(hdri, strength=p.nadir_cleanup)

    # 4e. Optional One-Click Neutralize Environment (turntable WB)
    if p.neutralize_environment:
        hdri = neutralize_hdri_environment(hdri)

    # 5. Saturation adjustment in linear space
    if abs(p.saturation - 1.0) > 1e-4:
        lum = compute_luminance(hdri)[..., None]
        hdri = lum + (hdri - lum) * float(p.saturation)
        hdri = np.maximum(hdri, 0.0)

    # 6. Exposure offset (EV stops)
    if abs(p.exposure_ev) > 1e-5:
        hdri *= float(2.0 ** p.exposure_ev)

    # 7. Optional output HDRI blur (360-deg wrap-safe, energy-preserving)
    if p.hdri_blur > 1e-4:
        hdri = blur_latlong_hdri(hdri, p.hdri_blur)

    if getattr(p, "sun_reconstruct", 0.0) > 1e-4:
        hdri = reconstruct_physical_sun_disk(hdri, strength=float(p.sun_reconstruct))

    return np.ascontiguousarray(np.maximum(hdri, 0.0), dtype=np.float32)


def recover_clipped_highlights(
    hdri_pixels: np.ndarray,
    strength: float = 1.0,
    threshold: float = 0.94,
) -> np.ndarray:
    """
    Reconstruct smooth physical HDR peak energy inside clipped LDR highlight cores
    (such as clamped suns or softboxes) using blurred core-distance falloff.
    """
    s = float(max(0.0, strength))
    if s <= 1e-4:
        return hdri_pixels
    img = hdri_pixels.astype(np.float32, copy=True)
    h, w = img.shape[:2]
    max_ch = np.max(img, axis=-1)
    clipped = (max_ch >= float(threshold)).astype(np.float32)
    if float(np.max(clipped)) < 0.5:
        return img

    # Smooth the binary clipped mask to obtain an interior-peaking distance proxy
    rx = max(2, w // 96)
    ry = max(2, h // 48)
    dist_proxy = _separable_box_blur(clipped[..., None], rx, ry, wrap_x=True)[..., 0]
    core = np.clip(dist_proxy * clipped, 0.0, 1.0)
    # Reconstruct peak energy while preserving local highlight chromaticity
    boost = 1.0 + s * (core * core) * 4.0
    return np.ascontiguousarray(img * boost[..., None], dtype=np.float32)


def _apply_nadir_cleanup(hdri_pixels: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Seam-aware nadir/tripod cleanup: blends a smoothly wrapped, pole-converged
    ring sampled just above the nadir zone down across the bottom cap.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-4:
        return hdri_pixels
    img = hdri_pixels.astype(np.float32, copy=True)
    h, w = img.shape[:2]
    y_ring0 = max(0, int(h * 0.72))
    y_ring1 = max(y_ring0 + 2, int(h * 0.84))
    y_nadir_start = int(h * 0.80)
    cap_h = h - y_nadir_start
    if cap_h < 4:
        return img

    ring = img[y_ring0:y_ring1]
    tile = _build_seamless_tile(ring)
    patch = _tile_fill_region(tile, cap_h, w, flip_vertical=False)
    patch = _smooth_pole_cap(patch, is_zenith=False, blur_amount=0.45 + 0.45 * s)

    ys = np.arange(cap_h, dtype=np.float32)
    w_blend = _smoothstep(0.0, max(2.0, cap_h * 0.45), ys)[:, None, None] * s
    img[y_nadir_start:] = img[y_nadir_start:] * (1.0 - w_blend) + patch * w_blend
    return np.ascontiguousarray(img, dtype=np.float32)


def generate_diffuse_irradiance_hdri(
    hdri_pixels: np.ndarray,
    clamp_percentile: float = 99.5,
) -> np.ndarray:
    """
    Generate a firefly-free diffuse/irradiance companion Lat/Long HDRI by softly
    clamping extreme specular spikes and applying wide wrap-safe spherical blur.
    """
    img = np.maximum(hdri_pixels.astype(np.float32, copy=False), 0.0)
    lum = compute_luminance(img)
    p_hi = float(np.percentile(lum, float(np.clip(clamp_percentile, 90.0, 99.95))))
    p_hi = max(p_hi, 1e-3)
    # Soft-knee rolloff above p_hi to prevent path-tracer fireflies while keeping energy
    excess = np.maximum(lum - p_hi, 0.0)
    target_lum = p_hi + p_hi * np.tanh(excess / p_hi)
    scale = np.where(lum > p_hi, target_lum / np.maximum(lum, 1e-6), 1.0)
    clamped = img * scale[..., None]
    return blur_latlong_hdri(clamped, blur_amount=0.45)


def render_preview_srgb(
    hdri_pixels: np.ndarray,
    preview_ev: float = 0.0,
    tonemap: str = "aces",
    roughness_preview: float = 0.0,
    exposure_ev: Optional[float] = None,
) -> np.ndarray:
    """
    Tone-map a 32-bit float HDRI into 8-bit sRGB (H, W, 3) uint8 for UI display.
    Supports 'aces', 'false_color', 'reinhard', and 'linear', plus optional
    roughness mip-blur simulation in [0, 1].
    """
    if exposure_ev is not None:
        preview_ev = exposure_ev
    src = hdri_pixels.astype(np.float32)
    r_prev = float(np.clip(roughness_preview, 0.0, 1.0))
    if r_prev > 1e-3:
        sh, sw = src.shape[:2]
        if sw > 320:
            step_x = max(1, sw // 320)
            step_y = max(1, sh // 160)
            src = src[::step_y, ::step_x]
        src = blur_latlong_hdri(src, blur_amount=min(1.0, (r_prev ** 1.45) * 0.85))
    img = src * float(2.0 ** preview_ev)
    tm = str(tonemap or "aces").strip().lower()

    if "false" in tm:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        # Stops relative to 18% middle gray (0.18)
        ev_stops = np.log2(np.maximum(lum, 1e-6) / 0.18)
        knots = np.array([-6.0, -4.0, -2.0, -0.5, 0.5, 2.0, 4.0, 6.5], dtype=np.float32)
        palette = np.array(
            [
                [0.18, 0.05, 0.35],  # <-6 EV: Deep purple (crushed blacks)
                [0.10, 0.25, 0.85],  # -4 EV: Blue (deep shadows)
                [0.10, 0.75, 0.85],  # -2 EV: Cyan (shadows)
                [0.15, 0.78, 0.25],  # -0.5 EV: Green (18% middle gray low)
                [0.45, 0.88, 0.20],  # +0.5 EV: Green-Yellow (18% middle gray high)
                [0.95, 0.88, 0.15],  # +2 EV: Yellow (bright diffuse / skin / sky)
                [0.98, 0.45, 0.08],  # +4 EV: Orange (highlights)
                [0.98, 0.10, 0.15],  # >+6.5 EV: Red (HDR specular / sun hotspot)
            ],
            dtype=np.float32,
        )
        out_rgb = np.empty_like(img)
        for ch in range(3):
            out_rgb[..., ch] = np.interp(ev_stops, knots, palette[:, ch])
        return np.ascontiguousarray(( np.clip(out_rgb, 0.0, 1.0) * 255.0 + 0.5 ).astype(np.uint8))

    if "aces" in tm:
        # Narkowicz ACES Filmic approximation
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        img = (img * (a * img + b)) / (img * (c * img + d) + e)
        img = np.clip(img, 0.0, 1.0)
    elif "reinhard" in tm:
        img = img / (1.0 + img)
    else:
        img = np.clip(img, 0.0, 1.0)

    srgb = linear_to_srgb(img)
    return np.ascontiguousarray((srgb * 255.0 + 0.5).astype(np.uint8))


def extract_and_inpaint_hotspot(
    hdri_pixels: np.ndarray,
    light_rect: Tuple[float, float, float, float],
    light_tex_size: Tuple[int, int] = (512, 512),
    inpaint_hdri: bool = True,
    shape: str = "rect",
    region_blur: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, ExtractedAreaLightInfo]:
    """
    Extract a box- or circle-selected hotspot region (u0, v0, u1, v1) from a
    32-bit float Lat/Long HDRI into a dedicated 32-bit float Area Light texture,
    compute its exact spherical position/size/energy, and optionally inpaint/remove
    the hotspot from the HDRI map so the Dome Light does not double-light.
    """
    from .lighting_analyzer import rgb_to_cct_and_tint

    h, w, _ = hdri_pixels.shape
    u0, v0, u1, v1 = light_rect
    u_min, u_max = float(np.clip(min(u0, u1), 0.0, 1.0)), float(np.clip(max(u0, u1), 0.0, 1.0))
    v_min, v_max = float(np.clip(min(v0, v1), 0.0, 1.0)), float(np.clip(max(v0, v1), 0.0, 1.0))

    x0 = int(np.clip(round(u_min * w), 0, w - 2))
    x1 = int(np.clip(round(u_max * w), x0 + 2, w))
    y0 = int(np.clip(round(v_min * h), 0, h - 2))
    y1 = int(np.clip(round(v_max * h), y0 + 2, h))

    patch = hdri_pixels[y0:y1, x0:x1].astype(np.float32)

    # Resample extracted hotspot patch to 32-bit float light texture (H_tex, W_tex, 3)
    tw, th = light_tex_size
    yy = np.linspace(0, patch.shape[0] - 1, th, dtype=np.float32)
    xx = np.linspace(0, patch.shape[1] - 1, tw, dtype=np.float32)
    py0 = np.floor(yy).astype(np.int32)
    px0 = np.floor(xx).astype(np.int32)
    py1 = np.minimum(py0 + 1, patch.shape[0] - 1)
    px1 = np.minimum(px0 + 1, patch.shape[1] - 1)
    fy = (yy - py0)[:, None, None]
    fx = (xx - px0)[None, :, None]

    light_tex = (
        patch[np.ix_(py0, px0)] * (1.0 - fx) * (1.0 - fy)
        + patch[np.ix_(py0, px1)] * fx * (1.0 - fy)
        + patch[np.ix_(py1, px0)] * (1.0 - fx) * fy
        + patch[np.ix_(py1, px1)] * fx * fy
    ).astype(np.float32)

    # Store wide C1-smooth feathering in Alpha (for Redshift/Arnold opacity) while
    # keeping full edge-to-edge RGB (so Karma, which ignores light texture alpha,
    # never renders black corners).
    uy = np.linspace(-1.0, 1.0, th, dtype=np.float32)[:, None]
    ux = np.linspace(-1.0, 1.0, tw, dtype=np.float32)[None, :]
    norm_shape = "circle" if str(shape).lower() in ("circle", "ellipse", "disk") else "rect"
    if norm_shape == "circle":
        rad = np.sqrt(ux * ux + uy * uy)
        alpha = (1.0 - _smoothstep(0.55, 1.0, rad)).astype(np.float32)
        # Mask RGB into a soft disk so UsdLuxRectLight emits & reflects as a true disk in Karma & Redshift
        light_tex = (light_tex * alpha[..., None]).astype(np.float32)
    else:
        vig_x = (1.0 - _smoothstep(0.45, 1.0, np.abs(ux))).astype(np.float32)
        vig_y = (1.0 - _smoothstep(0.45, 1.0, np.abs(uy))).astype(np.float32)
        alpha = (vig_x * vig_y).astype(np.float32)
    light_tex = np.concatenate([light_tex, alpha[..., None]], axis=-1).astype(np.float32)

    # Compute spherical direction & angular size of box center
    # Match USD/Karma UsdLuxDomeLight latlong at ry=0: u=0.5 -> +Z, u=0.25 -> +X, u=0.75 -> -X
    u_center = 0.5 * (u_min + u_max)
    v_center = 0.5 * (v_min + v_max)
    az_deg = (0.5 - u_center) * 360.0
    el_deg = (0.5 - v_center) * 180.0
    ang_w = max(2.0, (u_max - u_min) * 360.0 * max(0.15, math.cos(math.radians(el_deg))))
    ang_h = max(2.0, (v_max - v_min) * 180.0)

    lum = compute_luminance(patch)
    mean_int = float(np.mean(lum))
    peak_int = float(np.max(lum))
    mean_rgb = np.mean(patch, axis=(0, 1))
    m_c = float(np.max(mean_rgb))
    norm_rgb = (
        (float(mean_rgb[0] / m_c), float(mean_rgb[1] / m_c), float(mean_rgb[2] / m_c))
        if m_c > 1e-6
        else (1.0, 1.0, 1.0)
    )
    cct, _ = rgb_to_cct_and_tint((float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])))

    info = ExtractedAreaLightInfo(
        azimuth_deg=float(az_deg),
        elevation_deg=float(el_deg),
        angular_width_deg=float(ang_w),
        angular_height_deg=float(ang_h),
        mean_intensity=mean_int,
        peak_intensity=peak_int,
        color_rgb=norm_rgb,
        cct_kelvin=float(cct),
        rect_uv=(u_min, v_min, u_max, v_max),
        shape=norm_shape,
    )

    cleaned = hdri_pixels.copy()
    bh = y1 - y0
    bw = x1 - x0
    if norm_shape == "circle":
        cy = np.linspace(-1.0, 1.0, bh, dtype=np.float32)[:, None]
        cx = np.linspace(-1.0, 1.0, bw, dtype=np.float32)[None, :]
        ell_r = np.sqrt(cx * cx + cy * cy)
        mask = np.clip((1.0 - ell_r) / 0.18, 0.0, 1.0)[..., None]
    else:
        fy_b = np.clip(np.minimum(np.arange(bh), bh - 1 - np.arange(bh)) / max(1.0, bh * 0.12), 0.0, 1.0)[:, None, None]
        fx_b = np.clip(np.minimum(np.arange(bw), bw - 1 - np.arange(bw)) / max(1.0, bw * 0.12), 0.0, 1.0)[None, :, None]
        mask = fy_b * fx_b

    if inpaint_hdri:
        # Sample surrounding border strips (top, bottom, left, right)
        top_row = np.nan_to_num(hdri_pixels[max(0, y0 - 2) : max(1, y0), x0:x1].mean(axis=0, keepdims=True))
        bot_row = np.nan_to_num(hdri_pixels[min(h - 1, y1) : min(h, y1 + 2), x0:x1].mean(axis=0, keepdims=True))
        left_col = np.nan_to_num(hdri_pixels[y0:y1, (x0 - 1) % w][:, None, :])
        right_col = np.nan_to_num(hdri_pixels[y0:y1, (x1) % w][:, None, :])

        ty = np.linspace(0.0, 1.0, bh, dtype=np.float64)[:, None, None]
        tx = np.linspace(0.0, 1.0, bw, dtype=np.float64)[None, :, None]

        v_interp = top_row * (1.0 - ty) + bot_row * ty
        h_interp = left_col * (1.0 - tx) + right_col * tx

        # Weight toward nearest border (transfinite Coons-like blend)
        dy_edge = np.maximum(np.minimum(ty, 1.0 - ty), 1e-3)
        dx_edge = np.maximum(np.minimum(tx, 1.0 - tx), 1e-3)
        wv = 1.0 / dy_edge
        wh = 1.0 / dx_edge
        inpainted = ((v_interp * wv + h_interp * wh) / (wv + wh)).astype(np.float32)

        cleaned[y0:y1, x0:x1] = cleaned[y0:y1, x0:x1] * (1.0 - mask) + inpainted * mask

    rb = float(np.clip(region_blur, 0.0, 1.0))
    if rb > 1e-4:
        r_px = max(1, int(round(rb * max(bw, bh) * 0.22)))
        pad = min(max(8, r_px * 2), max(h, w) // 4)
        py0 = max(0, y0 - pad)
        py1 = min(h, y1 + pad)
        px0 = max(0, x0 - pad)
        px1 = min(w, x1 + pad)
        blurred_crop = _separable_box_blur(cleaned[py0:py1, px0:px1], r_px, r_px, wrap_x=False)
        local_patch = blurred_crop[(y0 - py0) : (y1 - py0), (x0 - px0) : (x1 - px0)]
        cleaned[y0:y1, x0:x1] = cleaned[y0:y1, x0:x1] * (1.0 - mask) + local_patch * mask

    return np.ascontiguousarray(cleaned, dtype=np.float32), np.ascontiguousarray(light_tex, dtype=np.float32), info


def extract_and_inpaint_multiple_hotspots(
    hdri_pixels: np.ndarray,
    light_regions: List[Dict[str, object]],
    light_tex_size: Tuple[int, int] = (512, 512),
    inpaint_hdri: bool = True,
    region_blur: float = 0.0,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, ExtractedAreaLightInfo]]]:
    """
    Extract multiple picked light regions (each with 'rect' and optional 'shape')
    from the original HDRI into 32-bit float textures and cumulatively inpaint
    all selected regions out of the HDRI map.
    """
    cleaned = hdri_pixels.copy()
    results: List[Tuple[np.ndarray, ExtractedAreaLightInfo]] = []
    for region in light_regions:
        rect = tuple(region["rect"])  # type: ignore[arg-type]
        shape = str(region.get("shape", "rect"))
        # Extract each light texture from the original un-inpainted HDRI
        _, light_tex, info = extract_and_inpaint_hotspot(
            hdri_pixels,
            rect,  # type: ignore[arg-type]
            light_tex_size=light_tex_size,
            inpaint_hdri=False,
            shape=shape,
        )
        if inpaint_hdri or float(region_blur) > 1e-4:
            cleaned, _, _ = extract_and_inpaint_hotspot(
                cleaned,
                rect,  # type: ignore[arg-type]
                light_tex_size=light_tex_size,
                inpaint_hdri=inpaint_hdri,
                shape=shape,
                region_blur=region_blur,
            )
        results.append((light_tex, info))
    return np.ascontiguousarray(cleaned, dtype=np.float32), results


def bake_planar_ground_texture(
    hdri_pixels: np.ndarray,
    ground_radius: float = 10.0,
    tripod_height: float = 1.5,
    resolution: int = 1024,
    feather_start: float = 0.55,
    proj_blur: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bake a top-down planar rectilinear 32-bit float ground projection texture
    and a matching 3-channel radial soft-blending opacity mask from the Lat/Long HDRI.

    Samples the equirectangular HDRI looking down from (0, tripod_height, 0) over
    [-ground_radius, ground_radius] in X and Z (matching DomeBreaker's ground projection).
    """
    R = max(0.5, float(ground_radius))
    H_cam = max(0.1, float(tripod_height))
    res = max(64, int(resolution))

    u_lin = np.linspace(0.0, 1.0, res, dtype=np.float32)
    v_lin = np.linspace(1.0, 0.0, res, dtype=np.float32)
    uu, vv = np.meshgrid(u_lin, v_lin)

    # World XZ coordinates on y = 0 plane matching UV (0..1, 0..1)
    dx = (uu - 0.5) * (2.0 * R)
    dz = (0.5 - vv) * (2.0 * R)
    dy = np.full_like(dx, -H_cam)

    dist = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-4)
    u_eq = (0.5 - (np.arctan2(dx, dz) / (2.0 * np.pi))) % 1.0
    v_eq = np.clip(0.5 - (np.arcsin(np.clip(dy / dist, -1.0, 1.0)) / np.pi), 0.0, 1.0)

    h_img, w_img = hdri_pixels.shape[:2]
    px = u_eq * (w_img - 1)
    py = v_eq * (h_img - 1)
    x0 = np.floor(px).astype(np.int32)
    x1 = (x0 + 1) % w_img
    y0 = np.floor(py).astype(np.int32)
    y1 = np.minimum(y0 + 1, h_img - 1)
    fx = (px - x0)[..., None]
    fy = (py - y0)[..., None]

    c00 = hdri_pixels[y0, x0]
    c10 = hdri_pixels[y0, x1]
    c01 = hdri_pixels[y1, x0]
    c11 = hdri_pixels[y1, x1]
    ground_rgb = (
        (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy)
        + (c01 * (1.0 - fx) + c11 * fx) * fy
    ).astype(np.float32)

    pb = float(np.clip(proj_blur, 0.0, 1.0))
    if pb > 1e-4:
        r_px = max(1, int(round(pb * (res / 10.0))))
        ground_rgb = _separable_box_blur(ground_rgb, r_px, r_px, wrap_x=False)
        ground_rgb = _separable_box_blur(ground_rgb, r_px, r_px, wrap_x=False)

    # Radial soft-blending transparency mask (1.0 in center, smoothstep to 0.0 at edge)
    f_start = float(np.clip(feather_start, 0.0, 0.99))
    r_norm = np.sqrt(dx * dx + dz * dz) / R
    alpha = 1.0 - _smoothstep(f_start, 1.0, r_norm)
    ground_opacity = np.repeat(alpha[..., None], 3, axis=-1).astype(np.float32)

    return np.ascontiguousarray(ground_rgb, dtype=np.float32), np.ascontiguousarray(ground_opacity, dtype=np.float32)


def compute_neutral_wb_gains(
    mean_rgb: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], float, float, float]:
    """
    Compute luminance-preserving 3-channel white balance gains from a sampled
    neutral gray / Macbeth ColorChecker patch in linear RGB.

    Returns:
        (wb_gains_rgb, measured_cct_kelvin, measured_duv, patch_luminance)
    """
    r = max(float(mean_rgb[0]), 1e-5)
    g = max(float(mean_rgb[1]), 1e-5)
    b = max(float(mean_rgb[2]), 1e-5)
    lum = max(0.2126 * r + 0.7152 * g + 0.0722 * b, 1e-5)

    cct, duv = rgb_to_cct_kelvin((r, g, b))

    gr = lum / r
    gg = lum / g
    gb = lum / b

    gr = float(np.clip(gr, 0.1, 10.0))
    gg = float(np.clip(gg, 0.1, 10.0))
    gb = float(np.clip(gb, 0.1, 10.0))
    return (gr, gg, gb), float(cct), float(duv), float(lum)


def export_hdri_sequence(
    frame_profiles: List[LightingProfile],
    frame_views: List[PlateViewInput],
    params: HDRISynthesisParams,
    output_pattern: str,
    output_colorspace: str = "ACEScg",
    light_regions: Optional[List[dict]] = None,
    inpaint_lights: bool = False,
    light_blur: float = 0.0,
    temporal_stabilize: Optional[float] = None,
    export_diffuse_split: bool = False,
) -> Tuple[str, List[str]]:
    """
    Generate and export a per-frame sequence of 32-bit float Lat/Long EXR files.
    Supports optional temporal EV/WB stabilization and diffuse irradiance split EXRs.

    Returns:
        (solaris_pattern_path, list_of_written_exr_paths)
    """
    import os
    from .plate_io import format_frame_path, write_exr_32f

    written: List[str] = []
    if not frame_profiles:
        raise ValueError("At least one LightingProfile is required for sequence export.")

    t_stab = float(
        np.clip(
            params.temporal_stabilize if temporal_stabilize is None else temporal_stabilize,
            0.0,
            1.0,
        )
    )
    n_frames = len(frame_profiles)
    ev_mults = [1.0] * n_frames
    wb_mults = [(1.0, 1.0, 1.0)] * n_frames

    if t_stab > 1e-4 and n_frames > 1:
        log_lums = np.array(
            [math.log(max(float(p.mean_luminance), 1e-5)) for p in frame_profiles],
            dtype=np.float32,
        )
        cols = np.array([p.overall_color for p in frame_profiles], dtype=np.float32)
        radius = max(1, int(round(1 + t_stab * 4)))
        sigma = max(0.75, radius * 0.55)
        for i in range(n_frames):
            lo = max(0, i - radius)
            hi = min(n_frames, i + radius + 1)
            offsets = np.arange(lo, hi, dtype=np.float32) - float(i)
            w = np.exp(-0.5 * (offsets / sigma) ** 2)
            w /= np.sum(w)
            target_log_lum = float(np.sum(log_lums[lo:hi] * w))
            ev_mults[i] = float(np.clip(math.exp((target_log_lum - float(log_lums[i])) * t_stab), 0.25, 4.0))

            target_col = np.sum(cols[lo:hi] * w[:, None], axis=0)
            src_col = np.maximum(cols[i], 1e-4)
            ratio = np.clip(target_col / src_col, 0.5, 2.0)
            r_lum = max(0.2126 * float(ratio[0]) + 0.7152 * float(ratio[1]) + 0.0722 * float(ratio[2]), 1e-5)
            ratio /= r_lum
            blended = (1.0 - t_stab) + t_stab * ratio
            wb_mults[i] = (float(blended[0]), float(blended[1]), float(blended[2]))

    for idx, prof in enumerate(frame_profiles):
        fnum = prof.frame if prof.frame is not None else (idx + 1)
        view_slice = [frame_views[idx]] if idx < len(frame_views) else None
        hdri = generate_latlong_hdri(prof, params, views=view_slice)
        if t_stab > 1e-4 and n_frames > 1:
            wb = np.array(wb_mults[idx], dtype=np.float32) * float(ev_mults[idx])
            hdri = hdri * wb[None, None, :]
        if light_regions:
            hdri, _ = extract_and_inpaint_multiple_hotspots(
                hdri,
                light_regions,
                inpaint_hdri=inpaint_lights,
                region_blur=light_blur,
            )
        out_file = format_frame_path(output_pattern, int(fnum))
        written.append(
            write_exr_32f(
                out_file,
                hdri,
                output_colorspace=output_colorspace,
                frame=int(fnum),
            )
        )
        if export_diffuse_split:
            stem, ext = os.path.splitext(out_file)
            diff_file = f"{stem}_diffuse{ext or '.exr'}"
            diff_hdri = generate_diffuse_irradiance_hdri(hdri)
            write_exr_32f(
                diff_file,
                diff_hdri,
                output_colorspace=output_colorspace,
                frame=int(fnum),
            )
    return output_pattern, written


def apply_brush_dab(
    hdri: np.ndarray,
    u: float,
    v: float,
    mode: str = "heal",
    radius_u: float = 0.025,
    softness: float = 0.65,
    strength: float = 0.85,
    src_offset_uv: Optional[Tuple[float, float]] = None,
    inplace: bool = False,
) -> np.ndarray:
    """
    Apply a single 32-bit float wrap-aware brush dab onto a 2:1 Lat/Long HDRI.

    Supported modes:
        - "clone": Clone from `(u + du, v + dv)` with 360-deg horizontal wrap
        - "heal": Smoothly inpaint from the surrounding clean ring
        - "blur": Local 32-bit float smoothing
        - "burn": Attenuate hot speculars/fireflies (`* 2^(-strength * w)`)
        - "dodge": Boost HDR energy (`* 2^(+strength * w)`)
        - "desat": Neutralize local chroma cast toward luminance
    """
    out = hdri if inplace else hdri.astype(np.float32, copy=True)
    h, w = out.shape[:2]
    ru_px = max(2, int(round(float(np.clip(radius_u, 0.002, 0.25)) * w)))
    rv_px = ru_px  # 2:1 aspect ratio -> equal pixel radius in x and y
    cx = int(round((u % 1.0) * (w - 1)))
    cy = int(round(float(np.clip(v, 0.0, 1.0)) * (h - 1)))

    # Local patch bounds with 4px padding for blur/heal context
    pad = max(4, ru_px // 2) if mode in ("heal", "blur") else 0
    rad = ru_px + pad
    y0 = max(0, cy - rad)
    y1 = min(h, cy + rad + 1)
    if y1 <= y0:
        return out

    xs = (np.arange(cx - rad, cx + rad + 1, dtype=np.int32)) % w
    ys = np.arange(y0, y1, dtype=np.int32)
    patch = out[np.ix_(ys, xs)].copy()

    # Radial distance in brush-radius units inside the patch
    gx = np.arange(cx - rad, cx + rad + 1, dtype=np.float32)[None, :] - float(cx)
    gy = ys.astype(np.float32)[:, None] - float(cy)
    r_norm = np.hypot(gx, gy) / float(max(1, ru_px))

    soft = float(np.clip(softness, 0.05, 1.0))
    inner = max(0.0, 1.0 - soft)
    # Smooth falloff: 1 at center, 0 at r_norm >= 1.0
    falloff = 1.0 - _smoothstep(inner, 1.0, r_norm)
    w_mask = (falloff * float(np.clip(strength, 0.02, 1.0)))[..., None]

    m = str(mode or "heal").strip().lower()
    if m == "clone" and src_offset_uv is not None:
        du_px = int(round(float(src_offset_uv[0]) * w))
        dv_px = int(round(float(src_offset_uv[1]) * h))
        src_xs = (xs + du_px) % w
        src_ys = np.clip(ys + dv_px, 0, h - 1)
        src_patch = out[np.ix_(src_ys, src_xs)]
        blended = patch * (1.0 - w_mask) + src_patch * w_mask
    elif m == "heal":
        # Compute mean color of the outer clean ring (1.05 <= r_norm <= 1.55)
        ring_mask = (r_norm >= 1.02) & (r_norm <= 1.55)
        if np.any(ring_mask):
            ring_rgb = np.median(patch[ring_mask], axis=0, keepdims=True)[None, :, :]
        else:
            ring_rgb = np.mean(patch, axis=(0, 1), keepdims=True)
        # Smooth the patch + blend toward ring_rgb for natural textureless fill
        rx_b = max(2, ru_px // 2)
        blurred = _separable_box_blur(patch, rx_b, rx_b, wrap_x=False)
        fill = blurred * 0.45 + ring_rgb * 0.55
        blended = patch * (1.0 - w_mask) + fill * w_mask
    elif m == "blur":
        rx_b = max(2, ru_px // 3)
        blurred = _separable_box_blur(patch, rx_b, rx_b, wrap_x=False)
        blended = patch * (1.0 - w_mask) + blurred * w_mask
    elif m == "burn":
        mult = np.power(2.0, -w_mask * 1.5)
        blended = patch * mult
    elif m == "dodge":
        mult = np.power(2.0, w_mask * 1.5)
        blended = patch * mult
    elif m == "desat":
        lum = compute_luminance(patch)[..., None]
        blended = patch * (1.0 - w_mask) + lum * w_mask
    else:
        return out

    out[np.ix_(ys, xs)] = np.maximum(blended, 0.0)
    return out


def apply_brush_strokes(
    hdri: np.ndarray,
    strokes: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Replay a list of resolution-independent 32-bit float brush strokes onto an HDRI.
    """
    if not strokes:
        return hdri
    out = hdri.astype(np.float32, copy=True)
    for st in strokes:
        mode = str(st.get("mode", "heal"))
        radius_u = float(st.get("radius_u", 0.025))
        softness = float(st.get("softness", 0.65))
        strength = float(st.get("strength", 0.85))
        src_off = st.get("src_offset")
        src_offset_uv = (float(src_off[0]), float(src_off[1])) if src_off is not None else None
        pts = st.get("points") or []
        for pt in pts:
            if len(pt) >= 2:
                apply_brush_dab(
                    out,
                    float(pt[0]),
                    float(pt[1]),
                    mode=mode,
                    radius_u=radius_u,
                    softness=softness,
                    strength=strength,
                    src_offset_uv=src_offset_uv,
                    inplace=True,
                )
    return np.ascontiguousarray(out, dtype=np.float32)


def apply_post_adjustments_to_hdri(
    base_hdri: np.ndarray,
    params: HDRISynthesisParams | None = None,
    profile: Optional[LightingProfile] = None,
) -> np.ndarray:
    """
    Fast path for a baked/collapsed 32-bit float Lat/Long canvas: resizes if needed
    and applies global post-adjustments (Yaw, Sky/Ground Gain, CCT/WB, Highlight
    Recovery, Nadir Cleanup, Neutralize, Saturation, Exposure EV, HDRI Blur)
    without re-running plate projection, stitching, or inpainting.
    """
    p = params or HDRISynthesisParams()
    h = max(64, int(p.height))
    w = max(128, int(p.width))
    src = np.asarray(base_hdri, dtype=np.float32)
    sh, sw = src.shape[:2]
    if (sh, sw) != (h, w):
        ys = np.linspace(0.0, sh - 1.0, h, dtype=np.float32)
        xs = np.linspace(0.0, sw - 1.0, w, dtype=np.float32)
        y0 = np.floor(ys).astype(np.int32)
        x0 = np.floor(xs).astype(np.int32)
        y1 = np.minimum(y0 + 1, sh - 1)
        x1 = (x0 + 1) % sw
        fy = (ys - y0)[:, None, None]
        fx = (xs - x0)[None, :, None]
        hdri = (
            (src[np.ix_(y0, x0)] * (1.0 - fx) + src[np.ix_(y0, x1)] * fx) * (1.0 - fy)
            + (src[np.ix_(y1, x0)] * (1.0 - fx) + src[np.ix_(y1, x1)] * fx) * fy
        ).astype(np.float32)
    else:
        hdri = src.copy()

    if abs(p.azimuth_offset_deg) > 1e-4:
        shift_px = int(round((p.azimuth_offset_deg / 360.0) * w)) % w
        if shift_px != 0:
            hdri = np.roll(hdri, shift_px, axis=1)

    if abs(p.sky_gain - 1.0) > 1e-4 or abs(p.ground_gain - 1.0) > 1e-4:
        v = (np.arange(h, dtype=np.float32) + 0.5) / float(h)
        elev = (0.5 - v) * math.pi
        if abs(p.sky_gain - 1.0) > 1e-4:
            sky_mask = _smoothstep(0.0, 0.25, elev)[:, None, None]
            hdri = hdri * (1.0 + (float(p.sky_gain) - 1.0) * sky_mask)
        if abs(p.ground_gain - 1.0) > 1e-4:
            gnd_mask = _smoothstep(0.0, 0.25, -elev)[:, None, None]
            hdri = hdri * (1.0 + (float(p.ground_gain) - 1.0) * gnd_mask)

    if p.temperature_kelvin > 0.0 and p.temperature_blend > 0.0:
        src_cct = profile.overall_cct_kelvin if profile is not None else 6500.0
        target_rgb = np.array(cct_to_rgb(p.temperature_kelvin), dtype=np.float32)
        src_rgb = np.array(cct_to_rgb(src_cct), dtype=np.float32)
        ratio = target_rgb / np.maximum(src_rgb, 1e-4)
        ratio_lum = float(0.2126 * ratio[0] + 0.7152 * ratio[1] + 0.0722 * ratio[2])
        if ratio_lum > 1e-5:
            ratio /= ratio_lum
        blend = float(np.clip(p.temperature_blend, 0.0, 1.0))
        hdri *= ((1.0 - blend) + blend * ratio)[None, None, :]

    if p.wb_gains is not None and len(p.wb_gains) == 3:
        wb = np.array(p.wb_gains, dtype=np.float32)
        if float(np.max(np.abs(wb - 1.0))) > 1e-4:
            hdri *= wb[None, None, :]

    if p.highlight_recovery > 1e-4:
        hdri = recover_clipped_highlights(hdri, strength=p.highlight_recovery)

    if p.nadir_cleanup > 1e-4:
        hdri = _apply_nadir_cleanup(hdri, strength=p.nadir_cleanup)

    if p.neutralize_environment:
        hdri = neutralize_hdri_environment(hdri)

    if abs(p.saturation - 1.0) > 1e-4:
        lum = compute_luminance(hdri)[..., None]
        hdri = np.maximum(lum + (hdri - lum) * float(p.saturation), 0.0)

    if abs(p.exposure_ev) > 1e-5:
        hdri *= float(2.0 ** p.exposure_ev)

    if p.hdri_blur > 1e-4:
        hdri = blur_latlong_hdri(hdri, p.hdri_blur)

    if getattr(p, "sun_reconstruct", 0.0) > 1e-4:
        hdri = reconstruct_physical_sun_disk(hdri, strength=float(p.sun_reconstruct))

    return np.ascontiguousarray(np.maximum(hdri, 0.0), dtype=np.float32)


def neutralize_hdri_environment(
    hdri_pixels: np.ndarray,
    wb_strength: float = 0.90,
) -> np.ndarray:
    """
    Estimate and remove global illuminant color cast for turntable lookdev without
    desaturating true scene colors or warming outdoor blue-sky HDRIs.

    Estimates the illuminant from the lowest-chroma mid-tone pixels (weighted by
    solid angle sin(theta)), so saturated sky/foliage do not bias the white balance.
    """
    img = np.maximum(hdri_pixels.astype(np.float32, copy=False), 0.0)
    h = img.shape[0]
    lum = compute_luminance(img)
    p95 = float(np.percentile(lum, 95.0))
    theta = (np.arange(h, dtype=np.float32) + 0.5) * (np.pi / float(max(h, 1)))
    sin_w = np.sin(theta)[:, None]

    # Measure per-pixel normalized chroma so saturated sky/grass/objects don't skew WB
    max_c = np.max(img, axis=2)
    min_c = np.min(img, axis=2)
    chroma = (max_c - min_c) / np.maximum(lum, 1e-4)

    mid_mask = (lum > 1e-4) & (lum <= max(p95, 1e-3))
    chroma_w = 1.0 / (1.0 + (chroma * 8.0) ** 2)
    weights = np.where(mid_mask, sin_w * chroma_w, 0.0)
    w_sum = float(np.sum(weights))
    if w_sum > 1e-6:
        mean_rgb = np.sum(img * weights[..., None], axis=(0, 1)) / w_sum
    else:
        mean_rgb = np.mean(img, axis=(0, 1))

    mr = max(float(mean_rgb[0]), 1e-5)
    mg = max(float(mean_rgb[1]), 1e-5)
    mb = max(float(mean_rgb[2]), 1e-5)
    mlum = max(0.2126 * mr + 0.7152 * mg + 0.0722 * mb, 1e-5)

    raw_gains = np.array([mlum / mr, mlum / mg, mlum / mb], dtype=np.float32)
    gains = np.clip(1.0 + (raw_gains - 1.0) * float(wb_strength), 0.30, 3.50)
    out = img * gains[None, None, :]
    return np.ascontiguousarray(np.maximum(out, 0.0), dtype=np.float32)


def reconstruct_physical_sun_disk(
    hdri_pixels: np.ndarray,
    strength: float = 1.0,
    sun_uv: Optional[Tuple[float, float]] = None,
    sun_radius_deg: float = 0.75,
) -> np.ndarray:
    """
    Synthesize a physically scaled limb-darkened solar disk + circumsolar Mie aureole
    at the brightest sky peak (or `sun_uv`) to restore crisp directional sun shadows
    from clipped LDR or phone skies.
    """
    s = float(max(0.0, strength))
    if s <= 1e-4:
        return hdri_pixels
    img = np.maximum(hdri_pixels.astype(np.float32, copy=True), 0.0)
    h, w = img.shape[:2]

    if sun_uv is None:
        sky_rows = max(4, int(h * 0.48))
        lum_sky = compute_luminance(img[:sky_rows])
        py, px = np.unravel_index(int(np.argmax(lum_sky)), lum_sky.shape)
        su = (px + 0.5) / float(w)
        sv = (py + 0.5) / float(h)
    else:
        su, sv = float(sun_uv[0]), float(sun_uv[1])
        px = int(np.clip(su * w, 0, w - 1))
        py = int(np.clip(sv * h, 0, h - 1))

    # Sun direction unit vector (matches equirectangular convention)
    az_s = (0.5 - su) * 2.0 * math.pi
    el_s = (0.5 - sv) * math.pi
    sx = math.cos(el_s) * math.sin(az_s)
    sy = math.sin(el_s)
    sz = math.cos(el_s) * math.cos(az_s)

    # Sample local sun chromaticity (normalized warm tint)
    y0, y1 = max(0, py - 2), min(h, py + 3)
    x0, x1 = max(0, px - 2), min(w, px + 3)
    local_rgb = np.mean(img[y0:y1, x0:x1], axis=(0, 1))
    local_lum = max(float(0.2126 * local_rgb[0] + 0.7152 * local_rgb[1] + 0.0722 * local_rgb[2]), 1e-4)
    sun_tint = np.clip(local_rgb / local_lum, 0.4, 2.0).astype(np.float32)

    u = (np.arange(w, dtype=np.float32) + 0.5) / float(w)
    v = (np.arange(h, dtype=np.float32) + 0.5) / float(h)
    az = (0.5 - u) * (2.0 * math.pi)
    el = (0.5 - v) * math.pi
    cos_el = np.cos(el)[:, None]
    dx = cos_el * np.sin(az)[None, :]
    dy = np.sin(el)[:, None]
    dz = cos_el * np.cos(az)[None, :]

    dot = np.clip(dx * sx + dy * sy + dz * sz, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(dot))

    r_disk = max(0.35, float(sun_radius_deg))
    # Smooth anti-aliased limb-darkened solar disk + circumsolar aureole
    disk_mask = 1.0 - _smoothstep(r_disk * 0.75, r_disk * 1.45, theta_deg)
    limb = np.sqrt(np.maximum(1.0 - np.minimum(theta_deg / max(r_disk * 1.45, 1e-3), 1.0) ** 2, 0.0))
    aureole = np.exp(-theta_deg / 4.2) * 0.025
    sun_energy = s * (160.0 * disk_mask * (0.65 + 0.35 * limb) + 40.0 * aureole)

    img += sun_energy[..., None] * sun_tint[None, None, :]
    return np.ascontiguousarray(img, dtype=np.float32)


def render_lookdev_sphere_preview_srgb(
    hdri_pixels: np.ndarray,
    width: int = 800,
    height: int = 400,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    exposure_ev: float = 0.0,
    tonemap: str = "ACES",
) -> np.ndarray:
    """
    Render a real-time pure-numpy 3D Lookdev Sphere preview (left: 100% Chrome Mirror Ball,
    right: 18% Gray Diffuse Ball) lit by the current 32-bit float Lat/Long HDRI.
    """
    w = max(256, int(width))
    h = max(128, int(height))
    src = np.maximum(np.asarray(hdri_pixels, dtype=np.float32), 0.0)
    sh, sw = src.shape[:2]

    # Build a fast blurred diffuse irradiance mip for the 18% Gray ball
    diff_src = _separable_box_blur(
        src[:: max(1, sh // 64), :: max(1, sw // 128)],
        rx=max(3, (sw // max(1, sw // 128)) // 8),
        ry=max(3, (sh // max(1, sh // 64)) // 8),
        wrap_x=True,
    )
    dh, dw = diff_src.shape[:2]

    # Studio background
    canvas = np.full((h, w, 3), 0.035, dtype=np.float32)

    xs = np.arange(w, dtype=np.float32) + 0.5
    ys = np.arange(h, dtype=np.float32) + 0.5
    gx, gy = np.meshgrid(xs, ys)

    radius = min(w * 0.21, h * 0.42)
    cy = h * 0.50
    cx_chrome = w * 0.28
    cx_gray = w * 0.72

    yaw_rad = math.radians(float(yaw_deg))
    pitch_rad = math.radians(float(np.clip(pitch_deg, -75.0, 75.0)))
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)

    def _sample_equirect(img_arr: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
        ih, iw = img_arr.shape[:2]
        # Apply pitch (X-axis) then yaw (Y-axis)
        vy1 = vy * cos_p - vz * sin_p
        vz1 = vy * sin_p + vz * cos_p
        vx2 = vx * cos_y + vz1 * sin_y
        vz2 = -vx * sin_y + vz1 * cos_y
        az = np.arctan2(vx2, vz2)
        el = np.arcsin(np.clip(vy1, -1.0, 1.0))
        u = (0.5 - az / (2.0 * math.pi)) % 1.0
        v = np.clip(0.5 - el / math.pi, 0.0, 1.0 - 1e-5)
        fx = u * (iw - 1)
        fy = v * (ih - 1)
        x0 = np.floor(fx).astype(np.int32)
        y0 = np.floor(fy).astype(np.int32)
        x1 = (x0 + 1) % iw
        y1 = np.minimum(y0 + 1, ih - 1)
        wx = (fx - x0)[..., None]
        wy = (fy - y0)[..., None]
        return (
            (img_arr[y0, x0] * (1.0 - wx) + img_arr[y0, x1] * wx) * (1.0 - wy)
            + (img_arr[y1, x0] * (1.0 - wx) + img_arr[y1, x1] * wx) * wy
        )

    # 1. Chrome Mirror Ball (left)
    nx_c = (gx - cx_chrome) / radius
    ny_c = (cy - gy) / radius
    r2_c = nx_c * nx_c + ny_c * ny_c
    mask_c = r2_c <= 1.0
    if np.any(mask_c):
        nz_c = np.sqrt(np.maximum(1.0 - r2_c[mask_c], 0.0))
        nx_m = nx_c[mask_c]
        ny_m = ny_c[mask_c]
        # View vector V = (0, 0, 1) toward camera -> R = 2*(N.V)*N - V = (2*nz*nx, 2*nz*ny, 2*nz^2 - 1)
        rx = 2.0 * nz_c * nx_m
        ry = 2.0 * nz_c * ny_m
        rz = 2.0 * nz_c * nz_c - 1.0
        canvas[mask_c] = _sample_equirect(src, rx, ry, rz) * 0.96

    # 2. 18% Gray Diffuse Ball (right)
    nx_g = (gx - cx_gray) / radius
    ny_g = (cy - gy) / radius
    r2_g = nx_g * nx_g + ny_g * ny_g
    mask_g = r2_g <= 1.0
    if np.any(mask_g):
        nz_g = np.sqrt(np.maximum(1.0 - r2_g[mask_g], 0.0))
        nx_m = nx_g[mask_g]
        ny_m = ny_g[mask_g]
        canvas[mask_g] = _sample_equirect(diff_src, nx_m, ny_m, nz_g) * 0.36

    return render_preview_srgb(canvas, preview_ev=exposure_ev, tonemap=tonemap)
