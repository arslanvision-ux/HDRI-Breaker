"""
Plate lighting, hotspot, tone, and color temperature (CCT) analyzer.

Pure-numpy implementation compatible with Houdini 20+ (no scipy/cv2 required).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# Linear Rec.709 / sRGB -> CIE XYZ (D65)
MAT_REC709_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

# CIE XYZ -> Linear Rec.709 / sRGB (D65)
MAT_XYZ_TO_REC709 = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)


@dataclass
class HotspotLight:
    """Detected light hotspot mapped from plate space to spherical HDRI space."""

    index: int
    plate_u: float  # [0, 1] left -> right
    plate_v: float  # [0, 1] top -> bottom
    azimuth_deg: float  # [-180, 180], 0 = +Z / front, +90 = +X / right
    elevation_deg: float  # [-90, 90], +90 = zenith, -90 = nadir
    radius_deg: float  # angular radius in degrees
    peak_intensity: float  # linear HDR peak luminance
    mean_intensity: float  # linear HDR mean luminance
    color_rgb: Tuple[float, float, float]  # normalized linear RGB tint
    cct_kelvin: float
    is_clipped_ldr: bool = False


@dataclass
class LightingProfile:
    """Complete lighting, tone, and temperature analysis of a plate (or sequence)."""

    frame: Optional[int] = None
    source_path: str = ""
    is_hdr_source: bool = False

    # Luminance & dynamic range
    mean_luminance: float = 0.0
    median_luminance: float = 0.0
    p10_luminance: float = 0.0
    p90_luminance: float = 0.0
    p99_luminance: float = 0.0
    max_luminance: float = 0.0
    dynamic_range_stops: float = 0.0
    key_to_fill_ratio: float = 1.0
    estimated_ev100: float = 0.0

    # Color temperature & tint
    overall_cct_kelvin: float = 6500.0
    overall_tint_duv: float = 0.0
    key_cct_kelvin: float = 6500.0
    sky_cct_kelvin: float = 7500.0
    ground_cct_kelvin: float = 5500.0

    # Linear RGB tonal regions (preserving relative luminance)
    overall_color: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    zenith_color: Tuple[float, float, float] = (0.18, 0.24, 0.35)
    sky_color: Tuple[float, float, float] = (0.25, 0.30, 0.38)
    horizon_color: Tuple[float, float, float] = (0.35, 0.34, 0.32)
    ground_color: Tuple[float, float, float] = (0.10, 0.09, 0.08)
    nadir_color: Tuple[float, float, float] = (0.06, 0.055, 0.05)

    # Azimuthal ambient strip sampled from plate (shape: N x 3)
    azimuthal_strip: np.ndarray = field(
        default_factory=lambda: np.full((64, 3), 0.2, dtype=np.float32)
    )

    # Extracted light hotspots sorted by energy descending
    hotspots: List[HotspotLight] = field(default_factory=list)

    # Optional multi-view spherical plate projection (H, W, 3) & coverage mask (H, W)
    spherical_plate_map: Optional[np.ndarray] = None
    spherical_coverage_mask: Optional[np.ndarray] = None


@dataclass
class PlateViewInput:
    """Single plate image with its camera orientation and FOV for multi-view HDRI."""

    pixels: np.ndarray
    is_hdr: bool = False
    source_path: str = ""
    fov_h_deg: float = 90.0
    cam_azimuth_deg: float = 0.0
    cam_elevation_deg: float = 0.0
    weight: float = 1.0


def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute Rec.709 linear luminance Y from (..., 3) RGB."""
    return (
        0.2126729 * rgb[..., 0]
        + 0.7151522 * rgb[..., 1]
        + 0.0721750 * rgb[..., 2]
    ).astype(np.float32)


def rgb_to_cct_and_tint(rgb: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Estimate Correlated Color Temperature (Kelvin) and Duv tint from linear Rec.709 RGB.

    Uses McCamy's cubic approximation in CIE xy and CIE 1960 UCS for Duv.
    """
    vec = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    if np.sum(vec) < 1e-8:
        return 6500.0, 0.0

    xyz = MAT_REC709_TO_XYZ @ vec
    xyz_sum = float(np.sum(xyz))
    if xyz_sum < 1e-8:
        return 6500.0, 0.0

    x = float(xyz[0] / xyz_sum)
    y = float(xyz[1] / xyz_sum)

    # McCamy's CCT formula (accurate ~2000K to 15000K+)
    denom = 0.1858 - y
    if abs(denom) < 1e-6:
        denom = 1e-6
    n = (x - 0.3320) / denom
    cct = 449.0 * (n ** 3) + 3525.0 * (n ** 2) + 6823.3 * n + 5520.33
    cct = float(np.clip(cct, 1800.0, 25000.0))

    # CIE 1960 UCS (u, v) for green/magenta Duv tint relative to Planckian locus
    ucs_denom = -2.0 * x + 12.0 * y + 3.0
    if abs(ucs_denom) < 1e-6:
        return cct, 0.0
    u = 4.0 * x / ucs_denom
    v = 6.0 * y / ucs_denom

    # Planckian locus approximation in CIE 1960 UCS (Krystek)
    t = cct
    u_bb = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t) / (
        1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t * t
    )
    v_bb = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t) / (
        1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t * t
    )
    duv = math.copysign(math.hypot(u - u_bb, v - v_bb), v - v_bb)
    return cct, float(np.clip(duv, -0.05, 0.05))


# Backward compatibility alias
rgb_to_cct_kelvin = rgb_to_cct_and_tint


def cct_to_rgb(kelvin: float, tint_duv: float = 0.0) -> Tuple[float, float, float]:
    """
    Convert Correlated Color Temperature (Kelvin) and optional green/magenta tint
    to normalized linear Rec.709 RGB (with luminance ~ 1.0).
    """
    t = float(np.clip(kelvin, 1800.0, 25000.0))
    # CIE D-illuminant / Planckian locus xy approximation
    if t <= 4000.0:
        x = (
            -0.2661239e9 / (t ** 3)
            - 0.2343589e6 / (t ** 2)
            + 0.8776956e3 / t
            + 0.179910
        )
    else:
        x = (
            -3.0258469e9 / (t ** 3)
            + 2.1070379e6 / (t ** 2)
            + 0.2226347e3 / t
            + 0.240390
        )

    if t <= 2222.0:
        y = -1.1063814 * (x ** 3) - 1.34811020 * (x ** 2) + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * (x ** 3) - 1.37418593 * (x ** 2) + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * (x ** 3) - 5.87338670 * (x ** 2) + 3.75112997 * x - 0.37001483

    y = max(y + tint_duv * 1.5, 1e-4)
    Y = 1.0
    X = (x / y) * Y
    Z = ((1.0 - x - y) / y) * Y
    rgb = MAT_XYZ_TO_REC709 @ np.array([X, Y, Z], dtype=np.float64)
    rgb = np.maximum(rgb, 0.0)
    max_c = float(np.max(rgb))
    if max_c > 1e-6:
        rgb = rgb / max_c
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _plate_uv_to_spherical(
    u: float,
    v: float,
    fov_h_deg: float,
    aspect: float,
    cam_azimuth_deg: float,
    cam_elevation_deg: float,
) -> Tuple[float, float]:
    """
    Map normalized plate coordinates (u in [0,1], v in [0,1]) to
    equirectangular spherical (azimuth_deg, elevation_deg).
    """
    fov_v_deg = fov_h_deg / max(aspect, 0.1)
    az = cam_azimuth_deg + (u - 0.5) * fov_h_deg
    # Wrap azimuth to [-180, 180]
    az = ((az + 180.0) % 360.0) - 180.0
    el = cam_elevation_deg + (0.5 - v) * fov_v_deg
    el = float(np.clip(el, -89.0, 89.0))
    return float(az), float(el)


def _angular_distance_deg(az1: float, el1: float, az2: float, el2: float) -> float:
    """Great-circle angular distance in degrees between two spherical directions."""
    p1 = math.radians(el1)
    p2 = math.radians(el2)
    dl = math.radians(az2 - az1)
    cos_d = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_d))))


def _normalize_color(rgb: np.ndarray) -> Tuple[float, float, float]:
    """Normalize an RGB triplet so its max channel is 1.0 (preserving chromaticity)."""
    arr = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    m = float(np.max(arr))
    if m < 1e-7:
        return (1.0, 1.0, 1.0)
    arr = arr / m
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def analyze_plate(
    pixels: np.ndarray,
    is_hdr: bool = False,
    frame: Optional[int] = None,
    source_path: str = "",
    fov_h_deg: float = 90.0,
    cam_azimuth_deg: float = 0.0,
    cam_elevation_deg: float = 0.0,
    max_hotspots: int = 5,
    ldr_highlight_boost: float = 8.0,
) -> LightingProfile:
    """
    Analyze a linear RGB plate image and extract comprehensive lighting data.

    Args:
        pixels: (H, W, 3) float32 linear Rec.709 image.
        is_hdr: Whether source was HDR (EXR/HDR) or LDR (JPG/PNG).
        frame: Optional frame number.
        source_path: Source file path for reporting.
        fov_h_deg: Horizontal field of view of the plate camera in degrees.
        cam_azimuth_deg: Camera yaw offset in degrees.
        cam_elevation_deg: Camera pitch offset in degrees.
        max_hotspots: Maximum number of distinct light hotspots to extract.
        ldr_highlight_boost: Multiplier applied to clipped LDR highlights (>0.92).
    """
    img = np.asarray(pixels, dtype=np.float32)
    h, w = img.shape[:2]
    aspect = w / float(max(h, 1))

    lum = compute_luminance(img)
    p10, p25, p50, p90, p95, p99, p999 = np.percentile(
        lum, [10.0, 25.0, 50.0, 90.0, 95.0, 99.0, 99.9]
    )
    mean_lum = float(np.mean(lum))
    max_lum = float(np.max(lum))
    dr_stops = float(math.log2(max(float(p999), 1e-5) / max(float(p10), 1e-5)))
    key_to_fill = float(p95 / max(float(p25), 1e-5))
    ev100 = float(math.log2(max(mean_lum * 100.0 / 12.5, 1e-5)))

    # Regional tonal extraction (zenith, sky, horizon, ground, nadir)
    y_zenith = max(1, int(h * 0.12))
    y_sky_end = max(2, int(h * 0.35))
    y_hor_start = int(h * 0.38)
    y_hor_end = max(y_hor_start + 1, int(h * 0.62))
    y_gnd_start = int(h * 0.65)
    y_nadir_start = min(h - 1, int(h * 0.88))

    def _robust_mean_rgb(region: np.ndarray) -> Tuple[float, float, float]:
        # Clip extreme 1% specular spikes so ambient sky/ground tone isn't skewed
        reg_lum = compute_luminance(region)
        cap = float(np.percentile(reg_lum, 98.0))
        mask = reg_lum <= max(cap, 1e-5)
        if np.any(mask):
            m = np.mean(region[mask], axis=0)
        else:
            m = np.mean(region, axis=(0, 1))
        return (float(m[0]), float(m[1]), float(m[2]))

    overall_rgb = _robust_mean_rgb(img)
    zenith_rgb = _robust_mean_rgb(img[:y_zenith, :])
    sky_rgb = _robust_mean_rgb(img[:y_sky_end, :])
    horizon_rgb = _robust_mean_rgb(img[y_hor_start:y_hor_end, :])
    ground_rgb = _robust_mean_rgb(img[y_gnd_start:, :])
    nadir_rgb = _robust_mean_rgb(img[y_nadir_start:, :])

    overall_cct, overall_duv = rgb_to_cct_and_tint(overall_rgb)
    sky_cct, _ = rgb_to_cct_and_tint(sky_rgb)
    ground_cct, _ = rgb_to_cct_and_tint(ground_rgb)

    # Azimuthal ambient strip (64 bins across plate width, averaged vertically)
    n_az = 64
    az_strip = np.zeros((n_az, 3), dtype=np.float32)
    bin_edges = np.linspace(0, w, n_az + 1).astype(np.int32)
    for i in range(n_az):
        x0, x1 = bin_edges[i], max(bin_edges[i] + 1, bin_edges[i + 1])
        col_slice = img[:, x0:x1, :]
        az_strip[i] = np.median(col_slice, axis=(0, 1))

    # Hotspot detection on a fast analysis grid
    grid_w = min(w, 128)
    grid_h = max(16, int(round(grid_w / aspect)))
    y_idx = np.linspace(0, h - 1, grid_h).astype(np.int32)
    x_idx = np.linspace(0, w - 1, grid_w).astype(np.int32)
    small_img = img[np.ix_(y_idx, x_idx)]
    small_lum = compute_luminance(small_img)

    # Adaptive hotspot threshold: top bright energy above ambient
    thresh = max(float(np.percentile(small_lum, 96.5)), float(p50) * 2.5, 0.35)
    cand_y, cand_x = np.where(small_lum >= thresh)

    hotspots: List[HotspotLight] = []
    if len(cand_y) > 0:
        # Sort candidates by luminance descending and apply angular non-maximum suppression
        cand_lums = small_lum[cand_y, cand_x]
        order = np.argsort(-cand_lums)
        min_sep_deg = max(10.0, fov_h_deg * 0.14)

        for idx in order:
            gy = int(cand_y[idx])
            gx = int(cand_x[idx])
            u = (gx + 0.5) / float(grid_w)
            v = (gy + 0.5) / float(grid_h)
            az, el = _plate_uv_to_spherical(
                u, v, fov_h_deg, aspect, cam_azimuth_deg, cam_elevation_deg
            )

            # Check separation from already accepted hotspots
            if any(
                _angular_distance_deg(az, el, hs.azimuth_deg, hs.elevation_deg)
                < min_sep_deg
                for hs in hotspots
            ):
                continue

            # Measure local patch around peak
            pr = max(2, int(round(grid_w * 0.04)))
            y0, y1 = max(0, gy - pr), min(grid_h, gy + pr + 1)
            x0, x1 = max(0, gx - pr), min(grid_w, gx + pr + 1)
            patch = small_img[y0:y1, x0:x1]
            patch_lum = small_lum[y0:y1, x0:x1]

            peak_val = float(small_lum[gy, gx])
            half_mask = patch_lum >= (peak_val * 0.5)
            area_frac = float(np.sum(half_mask)) / float(grid_w * grid_h)
            # Estimate angular radius from area fraction
            radius_deg = float(
                np.clip(
                    math.sqrt(max(area_frac, 1e-4) / math.pi) * fov_h_deg * 1.4,
                    1.5,
                    28.0,
                )
            )

            # Check if LDR highlight is clipped near 1.0
            peak_rgb = small_img[gy, gx]
            is_clipped = (not is_hdr) and (float(np.max(peak_rgb)) >= 0.88)

            # Boost LDR clipped highlights into true HDR energy
            hdr_scale = ldr_highlight_boost if is_clipped else 1.0
            mean_patch_rgb = np.mean(patch[half_mask], axis=0) if np.any(half_mask) else peak_rgb
            hs_cct, _ = rgb_to_cct_and_tint(
                (float(mean_patch_rgb[0]), float(mean_patch_rgb[1]), float(mean_patch_rgb[2]))
            )

            hotspots.append(
                HotspotLight(
                    index=len(hotspots),
                    plate_u=u,
                    plate_v=v,
                    azimuth_deg=az,
                    elevation_deg=el,
                    radius_deg=radius_deg,
                    peak_intensity=peak_val * hdr_scale,
                    mean_intensity=float(np.mean(patch_lum[half_mask])) * hdr_scale
                    if np.any(half_mask)
                    else peak_val * hdr_scale,
                    color_rgb=_normalize_color(mean_patch_rgb),
                    cct_kelvin=hs_cct,
                    is_clipped_ldr=is_clipped,
                )
            )
            if len(hotspots) >= max_hotspots:
                break

    # Key light CCT (from brightest hotspot if found, else p95 highlight average)
    if hotspots:
        key_cct = hotspots[0].cct_kelvin
    else:
        hi_mask = lum >= float(p95)
        hi_rgb = np.mean(img[hi_mask], axis=0) if np.any(hi_mask) else np.array(overall_rgb)
        key_cct, _ = rgb_to_cct_and_tint((float(hi_rgb[0]), float(hi_rgb[1]), float(hi_rgb[2])))

    return LightingProfile(
        frame=frame,
        source_path=source_path,
        is_hdr_source=is_hdr,
        mean_luminance=mean_lum,
        median_luminance=float(p50),
        p10_luminance=float(p10),
        p90_luminance=float(p90),
        p99_luminance=float(p99),
        max_luminance=max_lum,
        dynamic_range_stops=dr_stops,
        key_to_fill_ratio=key_to_fill,
        estimated_ev100=ev100,
        overall_cct_kelvin=overall_cct,
        overall_tint_duv=overall_duv,
        key_cct_kelvin=key_cct,
        sky_cct_kelvin=sky_cct,
        ground_cct_kelvin=ground_cct,
        overall_color=overall_rgb,
        zenith_color=zenith_rgb,
        sky_color=sky_rgb,
        horizon_color=horizon_rgb,
        ground_color=ground_rgb,
        nadir_color=nadir_rgb,
        azimuthal_strip=az_strip,
        hotspots=hotspots,
    )


def merge_lighting_profiles(profiles: List[LightingProfile]) -> LightingProfile:
    """
    Temporally stabilize/merge multiple per-frame LightingProfiles into one master profile.
    """
    if not profiles:
        raise ValueError("Cannot merge an empty list of LightingProfiles.")
    if len(profiles) == 1:
        return profiles[0]

    def _med_attr(attr: str) -> float:
        return float(np.median([getattr(p, attr) for p in profiles]))

    def _med_rgb(attr: str) -> Tuple[float, float, float]:
        arr = np.median([getattr(p, attr) for p in profiles], axis=0)
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    az_stack = np.median([p.azimuthal_strip for p in profiles], axis=0).astype(np.float32)

    # Use representative middle-frame hotspots averaged with matching directions
    mid = profiles[len(profiles) // 2]

    return LightingProfile(
        frame=None,
        source_path=f"{profiles[0].source_path} ({len(profiles)} frames)",
        is_hdr_source=any(p.is_hdr_source for p in profiles),
        mean_luminance=_med_attr("mean_luminance"),
        median_luminance=_med_attr("median_luminance"),
        p10_luminance=_med_attr("p10_luminance"),
        p90_luminance=_med_attr("p90_luminance"),
        p99_luminance=_med_attr("p99_luminance"),
        max_luminance=float(np.max([p.max_luminance for p in profiles])),
        dynamic_range_stops=_med_attr("dynamic_range_stops"),
        key_to_fill_ratio=_med_attr("key_to_fill_ratio"),
        estimated_ev100=_med_attr("estimated_ev100"),
        overall_cct_kelvin=_med_attr("overall_cct_kelvin"),
        overall_tint_duv=_med_attr("overall_tint_duv"),
        key_cct_kelvin=_med_attr("key_cct_kelvin"),
        sky_cct_kelvin=_med_attr("sky_cct_kelvin"),
        ground_cct_kelvin=_med_attr("ground_cct_kelvin"),
        overall_color=_med_rgb("overall_color"),
        zenith_color=_med_rgb("zenith_color"),
        sky_color=_med_rgb("sky_color"),
        horizon_color=_med_rgb("horizon_color"),
        ground_color=_med_rgb("ground_color"),
        nadir_color=_med_rgb("nadir_color"),
        azimuthal_strip=az_stack,
        hotspots=list(mid.hotspots),
    )


def project_multiview_plates_to_latlong(
    views: List[PlateViewInput],
    out_width: int = 2048,
    out_height: int = 1024,
    ldr_highlight_boost: float = 4.0,
    edge_feather: float = 0.25,
    vertical_shift_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project multiple rectilinear plate views taken from different camera
    orientations (azimuth, elevation, FOV) onto a 2:1 Lat/Long canvas with
    smooth cosine/edge-feathered blending and optional LDR highlight expansion.

    Returns:
        (rgb_map (H, W, 3) float32, coverage_mask (H, W) float32 in [0, 1])
    """
    u = (np.arange(out_width, dtype=np.float32) + 0.5) / float(out_width)
    v = (np.arange(out_height, dtype=np.float32) + 0.5) / float(out_height)
    phi = (u - 0.5) * (2.0 * math.pi)
    theta = (0.5 - v) * math.pi

    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    dx = cos_t * np.sin(phi)[None, :]
    dy = np.broadcast_to(sin_t, (out_height, out_width))
    dz = cos_t * np.cos(phi)[None, :]

    accum_rgb = np.zeros((out_height, out_width, 3), dtype=np.float32)
    accum_w = np.zeros((out_height, out_width), dtype=np.float32)
    max_cov = np.zeros((out_height, out_width), dtype=np.float32)

    for view in views:
        img = np.asarray(view.pixels, dtype=np.float32)
        ih, iw = img.shape[:2]
        aspect = iw / float(max(ih, 1))
        fov_h_rad = math.radians(np.clip(view.fov_h_deg, 5.0, 165.0) * 0.5)
        tan_half_fov_x = math.tan(fov_h_rad)
        tan_half_fov_y = tan_half_fov_x / max(aspect, 0.1)
        fov_v_deg = math.degrees(math.atan(tan_half_fov_y))

        az = math.radians(view.cam_azimuth_deg)
        el_deg = float(np.clip(view.cam_elevation_deg + vertical_shift_deg, -89.0, 89.0))
        el = math.radians(el_deg)

        # Restrict evaluation to the latitude row band covered by this view (+margin)
        diag_half_deg = math.degrees(math.atan(math.hypot(tan_half_fov_x, tan_half_fov_y))) + 8.0
        el_hi = el_deg + diag_half_deg
        el_lo = el_deg - diag_half_deg
        if el_hi >= 89.0 or el_lo <= -89.0:
            y_start, y_end = 0, out_height
        else:
            y_start = max(0, int(math.floor((0.5 - el_hi / 180.0) * out_height)))
            y_end = min(out_height, int(math.ceil((0.5 - el_lo / 180.0) * out_height)) + 1)
        if y_end <= y_start:
            continue

        fwd_x = math.cos(el) * math.sin(az)
        fwd_y = math.sin(el)
        fwd_z = math.cos(el) * math.cos(az)

        right_x = math.cos(az)
        right_y = 0.0
        right_z = -math.sin(az)

        up_x = fwd_y * right_z - fwd_z * right_y
        up_y = fwd_z * right_x - fwd_x * right_z
        up_z = fwd_x * right_y - fwd_y * right_x

        dx_s = dx[y_start:y_end]
        dy_s = dy[y_start:y_end]
        dz_s = dz[y_start:y_end]
        cam_z = dx_s * fwd_x + dy_s * fwd_y + dz_s * fwd_z
        cam_x = dx_s * right_x + dy_s * right_y + dz_s * right_z
        cam_y = dx_s * up_x + dy_s * up_y + dz_s * up_z

        in_front = cam_z > 1e-4
        safe_z = np.where(in_front, cam_z, 1.0)
        ndc_x = (cam_x / safe_z) / tan_half_fov_x
        ndc_y = (cam_y / safe_z) / tan_half_fov_y

        inside = in_front & (np.abs(ndc_x) <= 1.0) & (np.abs(ndc_y) <= 1.0)
        iy_loc, ix = np.where(inside)
        if len(iy_loc) == 0:
            continue
        iy = iy_loc + y_start

        nx_in = ndc_x[iy_loc, ix]
        ny_in = ndc_y[iy_loc, ix]

        # Smooth edge feathering so overlapping views blend cleanly
        f_w = max(0.03, float(edge_feather))
        edge_x = np.clip((1.0 - np.abs(nx_in)) / f_w, 0.0, 1.0)
        edge_y = np.clip((1.0 - np.abs(ny_in)) / f_w, 0.0, 1.0)
        feather = (edge_x * edge_y).astype(np.float32)
        w_in = feather * float(max(view.weight, 0.01))

        # Bilinear sample only at covered pixels
        px_f = ((nx_in + 1.0) * 0.5) * (iw - 1)
        py_f = ((1.0 - ny_in) * 0.5) * (ih - 1)
        x0 = np.clip(np.floor(px_f).astype(np.int32), 0, iw - 1)
        y0 = np.clip(np.floor(py_f).astype(np.int32), 0, ih - 1)
        x1 = np.minimum(x0 + 1, iw - 1)
        y1 = np.minimum(y0 + 1, ih - 1)
        fx = (px_f - x0)[:, None]
        fy = (py_f - y0)[:, None]

        sampled = (
            img[y0, x0] * (1.0 - fx) * (1.0 - fy)
            + img[y0, x1] * fx * (1.0 - fy)
            + img[y1, x0] * (1.0 - fx) * fy
            + img[y1, x1] * fx * fy
        )

        # Inverse tone-mapping highlight expansion for LDR plates (preserves cloud/sun detail)
        if (not view.is_hdr) and ldr_highlight_boost > 1.0:
            lum_in = (
                0.2126729 * sampled[:, 0]
                + 0.7151522 * sampled[:, 1]
                + 0.0721750 * sampled[:, 2]
            )
            hi = np.clip((lum_in - 0.72) / 0.28, 0.0, 1.0)
            boost = 1.0 + (float(ldr_highlight_boost) - 1.0) * np.power(hi, 1.8)
            sampled = sampled * boost[:, None]

        accum_rgb[iy, ix] += sampled * w_in[:, None]
        accum_w[iy, ix] += w_in
        max_cov[iy, ix] = np.maximum(max_cov[iy, ix], feather)

    valid = accum_w > 1e-6
    inv_w = np.where(valid, 1.0 / np.maximum(accum_w, 1e-6), 0.0).astype(np.float32)
    accum_rgb *= inv_w[..., None]
    return accum_rgb, np.clip(max_cov, 0.0, 1.0).astype(np.float32)


def auto_align_adjacent_plates(views: List[PlateViewInput]) -> List[PlateViewInput]:
    """
    Coarse-to-fine spherical overlap refinement for consecutive plates.

    Searches azimuth/elevation offsets around each plate's initial pose
    to maximize normalized cross-correlation in overlapping border regions.
    """
    if len(views) < 2:
        return views

    def _ncc_score(prev_lum: np.ndarray, prev_cov: np.ndarray, cand: PlateViewInput) -> float:
        c_map, c_cov = project_multiview_plates_to_latlong([cand], 256, 128, ldr_highlight_boost=1.0)
        overlap = (prev_cov > 0.25) & (c_cov > 0.25)
        if np.sum(overlap) < 40:
            return -1e9
        a = prev_lum[overlap]
        b = compute_luminance(c_map)[overlap]
        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        return float(np.sum(a * b) / denom) if denom > 1e-6 else -1e9

    aligned = [views[0]]
    for i in range(1, len(views)):
        prev = aligned[-1]
        curr = views[i]
        best_az = curr.cam_azimuth_deg
        best_el = curr.cam_elevation_deg
        best_score = -1e9

        prev_map, prev_cov = project_multiview_plates_to_latlong([prev], 256, 128, ldr_highlight_boost=1.0)
        prev_lum = compute_luminance(prev_map)

        # Stage 1: Coarse search (+-18 deg az, +-8 deg el)
        for d_az in (-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0):
            for d_el in (-8.0, -4.0, 0.0, 4.0, 8.0):
                cand = PlateViewInput(
                    pixels=curr.pixels,
                    is_hdr=curr.is_hdr,
                    source_path=curr.source_path,
                    fov_h_deg=curr.fov_h_deg,
                    cam_azimuth_deg=curr.cam_azimuth_deg + d_az,
                    cam_elevation_deg=float(np.clip(curr.cam_elevation_deg + d_el, -85.0, 85.0)),
                    weight=curr.weight,
                )
                score = _ncc_score(prev_lum, prev_cov, cand)
                if score > best_score:
                    best_score = score
                    best_az = cand.cam_azimuth_deg
                    best_el = cand.cam_elevation_deg

        # Stage 2: Fine search (+-3 deg az, +-2 deg el in 1 deg steps)
        coarse_az, coarse_el = best_az, best_el
        for d_az in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
            for d_el in (-2.0, -1.0, 0.0, 1.0, 2.0):
                cand = PlateViewInput(
                    pixels=curr.pixels,
                    is_hdr=curr.is_hdr,
                    source_path=curr.source_path,
                    fov_h_deg=curr.fov_h_deg,
                    cam_azimuth_deg=coarse_az + d_az,
                    cam_elevation_deg=float(np.clip(coarse_el + d_el, -85.0, 85.0)),
                    weight=curr.weight,
                )
                score = _ncc_score(prev_lum, prev_cov, cand)
                if score > best_score:
                    best_score = score
                    best_az = cand.cam_azimuth_deg
                    best_el = cand.cam_elevation_deg

        aligned.append(
            PlateViewInput(
                pixels=curr.pixels,
                is_hdr=curr.is_hdr,
                source_path=curr.source_path,
                fov_h_deg=curr.fov_h_deg,
                cam_azimuth_deg=best_az,
                cam_elevation_deg=best_el,
                weight=curr.weight,
            )
        )
    return aligned


def analyze_multiview_plates(
    views: List[PlateViewInput],
    max_hotspots: int = 8,
    ldr_highlight_boost: float = 8.0,
) -> LightingProfile:
    """
    Analyze multiple plate images taken from different camera positions/angles
    and fuse them into a unified 360-degree LightingProfile.
    """
    if not views:
        raise ValueError("At least one PlateViewInput is required.")
    if len(views) == 1:
        v = views[0]
        prof = analyze_plate(
            v.pixels,
            is_hdr=v.is_hdr,
            source_path=v.source_path,
            fov_h_deg=v.fov_h_deg,
            cam_azimuth_deg=v.cam_azimuth_deg,
            cam_elevation_deg=v.cam_elevation_deg,
            max_hotspots=max_hotspots,
            ldr_highlight_boost=ldr_highlight_boost,
        )
        s_map, s_cov = project_multiview_plates_to_latlong(views)
        prof.spherical_plate_map = s_map
        prof.spherical_coverage_mask = s_cov
        return prof

    per_view_profiles: List[LightingProfile] = []
    all_hotspots: List[HotspotLight] = []

    for v in views:
        p = analyze_plate(
            v.pixels,
            is_hdr=v.is_hdr,
            source_path=v.source_path,
            fov_h_deg=v.fov_h_deg,
            cam_azimuth_deg=v.cam_azimuth_deg,
            cam_elevation_deg=v.cam_elevation_deg,
            max_hotspots=max_hotspots,
            ldr_highlight_boost=ldr_highlight_boost,
        )
        per_view_profiles.append(p)
        all_hotspots.extend(p.hotspots)

    # Deduplicate hotspots across overlapping views in 3D spherical space
    all_hotspots.sort(key=lambda h: -h.peak_intensity)
    merged_hotspots: List[HotspotLight] = []
    for hs in all_hotspots:
        if any(
            _angular_distance_deg(hs.azimuth_deg, hs.elevation_deg, ex.azimuth_deg, ex.elevation_deg)
            < 12.0
            for ex in merged_hotspots
        ):
            continue
        merged_hotspots.append(
            HotspotLight(
                index=len(merged_hotspots),
                plate_u=hs.plate_u,
                plate_v=hs.plate_v,
                azimuth_deg=hs.azimuth_deg,
                elevation_deg=hs.elevation_deg,
                radius_deg=hs.radius_deg,
                peak_intensity=hs.peak_intensity,
                mean_intensity=hs.mean_intensity,
                color_rgb=hs.color_rgb,
                cct_kelvin=hs.cct_kelvin,
                is_clipped_ldr=hs.is_clipped_ldr,
            )
        )
        if len(merged_hotspots) >= max_hotspots:
            break

    merged = merge_lighting_profiles(per_view_profiles)
    merged.source_path = f"{len(views)} multi-view plates"
    merged.hotspots = merged_hotspots

    s_map, s_cov = project_multiview_plates_to_latlong(views)
    merged.spherical_plate_map = s_map
    merged.spherical_coverage_mask = s_cov
    return merged


def estimate_horizon_shift_deg(
    spherical_map: np.ndarray,
    coverage_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Estimate the vertical horizon offset (in degrees) from a spherical Lat/Long map
    by locating the strongest sky-to-ground vertical transition (luminance + sky blue/ground warm)
    within [-40 deg, +40 deg] of the equator.

    Returns the recommended `vertical_shift_deg` to center the horizon at 0 deg (v = 0.5).
    """
    if spherical_map is None or spherical_map.ndim != 3:
        return 0.0
    h, w, _ = spherical_map.shape
    if h < 16 or w < 16:
        return 0.0

    y_lo = max(1, int(round(0.28 * h)))
    y_hi = min(h - 2, int(round(0.72 * h)))
    if y_hi <= y_lo + 4:
        return 0.0

    lum = (
        0.2126 * spherical_map[..., 0]
        + 0.7152 * spherical_map[..., 1]
        + 0.0722 * spherical_map[..., 2]
    )
    log_lum = np.log2(np.maximum(lum, 1e-4))
    tot = np.maximum(spherical_map[..., 0] + spherical_map[..., 1] + spherical_map[..., 2], 1e-4)
    sky_blue_ratio = (spherical_map[..., 2] - 0.5 * spherical_map[..., 0]) / tot

    if coverage_mask is not None and coverage_mask.shape == (h, w):
        valid = (coverage_mask > 0.25).astype(np.float32)
    else:
        valid = (lum > 1e-5).astype(np.float32)

    # Vertical derivative: sky above (smaller y) is brighter/bluer than ground below (larger y)
    dl_dy = log_lum[:-1, :] - log_lum[1:, :]
    db_dy = sky_blue_ratio[:-1, :] - sky_blue_ratio[1:, :]
    edge_score = np.maximum(0.0, dl_dy + 1.5 * db_dy)
    pair_valid = valid[:-1, :] * valid[1:, :]

    band_edge = edge_score[y_lo:y_hi, :] * pair_valid[y_lo:y_hi, :]
    row_weights = np.sum(pair_valid[y_lo:y_hi, :], axis=1)
    row_profile = np.sum(band_edge, axis=1) / np.maximum(row_weights, 1.0)
    row_profile = np.where(row_weights > (0.05 * w), row_profile, 0.0)

    if float(np.max(row_profile)) <= 1e-5:
        return 0.0

    # Smooth 1D profile across 3 rows to avoid single-pixel noise spikes
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    smoothed = np.convolve(row_profile, kernel, mode="same")
    best_idx = int(np.argmax(smoothed))
    horizon_y = float(y_lo + best_idx) + 0.5
    v_horizon = horizon_y / float(h)

    # Elevation of detected horizon in degrees: +90 at v=0, 0 at v=0.5, -90 at v=1
    detected_el_deg = (0.5 - v_horizon) * 180.0
    return float(np.clip(-detected_el_deg, -40.0, 40.0))
