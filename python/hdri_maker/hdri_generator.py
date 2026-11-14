"""
True 32-bit float Lat/Long (Equirectangular 2:1) HDRI synthesizer.

Synthesizes distortion-free spherical sky/ground gradients, azimuthal color wrap,
and unclamped physical HDR hotspots from a LightingProfile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def _build_seamless_tile(patch: np.ndarray) -> np.ndarray:
    """Create a 2x2 ping-pong mirrored seamless tile from an (H, W, 3) patch."""
    h_mir = np.concatenate([patch, patch[:, ::-1, :]], axis=1)
    return np.concatenate([h_mir, h_mir[::-1, :, :]], axis=0)


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
            sky_patch = _extract_patch(
                sky_rect,
                y_top_in,
                min(y_bot, y_top_in + max(12, band_h // 5)),
            )
            sky_tile = _build_seamless_tile(sky_patch)
            sky_fill = _tile_fill_region(sky_tile, cap_h, w, flip_vertical=True)
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
            gnd_patch = _extract_patch(
                ground_rect,
                max(y_top, y_bot_in - max(12, band_h // 5)),
                y_bot_in,
            )
            gnd_tile = _build_seamless_tile(gnd_patch)
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
                    (id(v.pixels), float(v.cam_azimuth_deg), float(v.cam_elevation_deg), float(v.fov_h_deg))
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

    return np.ascontiguousarray(np.maximum(hdri, 0.0), dtype=np.float32)


def render_preview_srgb(
    hdri_pixels: np.ndarray,
    preview_ev: float = 0.0,
    tonemap: str = "aces",
) -> np.ndarray:
    """
    Tone-map a 32-bit float HDRI into 8-bit sRGB (H, W, 3) uint8 for UI display.
    Supports 'aces', 'false_color', 'reinhard', and 'linear'.
    """
    img = hdri_pixels.astype(np.float32) * float(2.0 ** preview_ev)
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
