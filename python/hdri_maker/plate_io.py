"""
Plate & image sequence I/O and true 32-bit float EXR writer.

Uses OpenImageIO (bundled with Houdini 20+) with a PIL fallback for LDR plates.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import OpenImageIO as oiio
    HAS_OIIO = True
except ImportError:
    oiio = None
    HAS_OIIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    import hou
    HAS_HOU = True
except ImportError:
    hou = None
    HAS_HOU = False


LDR_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tga"}
HDR_EXTENSIONS = {".exr", ".hdr", ".pic", ".rat"}

# Exact 3x3 chromaticity conversion matrices (D60 <-> D65 Bradford adapted)
# ACEScg (AP1, D60) -> Linear sRGB / Rec.709 (D65)
MAT_ACESCG_TO_REC709 = np.array(
    [
        [1.70505099, -0.62179212, -0.08325887],
        [-0.13025642, 1.14080474, -0.01054832],
        [-0.02400336, -0.12896898, 1.15297233],
    ],
    dtype=np.float32,
)

# Linear sRGB / Rec.709 (D65) -> ACEScg (AP1, D60)
MAT_REC709_TO_ACESCG = np.array(
    [
        [0.61309740, 0.33952315, 0.04737945],
        [0.07019372, 0.91635388, 0.01345240],
        [0.02061559, 0.10956977, 0.86981464],
    ],
    dtype=np.float32,
)


@dataclass
class PlateFrame:
    """Loaded plate image in 32-bit float linear RGB."""

    pixels: np.ndarray  # shape (H, W, 3), dtype float32
    path: str
    frame: Optional[int] = None
    source_colorspace: str = "Linear"
    is_hdr: bool = False


def expand_houdini_path(path: str, frame: Optional[int] = None) -> str:
    """Expand Houdini environment variables and frame tokens ($F, $F4, ####)."""
    result = path.strip().strip('"').strip("'")
    if HAS_HOU:
        try:
            if frame is not None:
                # Replace $F<N> or $F explicitly for arbitrary frame evaluation
                result = re.sub(
                    r"\$F(\d+)",
                    lambda m: f"{frame:0{int(m.group(1))}d}",
                    result,
                )
                result = result.replace("$F", str(frame))
            result = hou.text.expandString(result)
        except Exception:
            pass
    else:
        if frame is not None:
            result = re.sub(
                r"\$F(\d+)",
                lambda m: f"{frame:0{int(m.group(1))}d}",
                result,
            )
            result = result.replace("$F", str(frame))
        result = os.path.expandvars(result)

    if frame is not None:
        # Handle #### or %04d style tokens
        result = re.sub(
            r"(#+)",
            lambda m: f"{frame:0{len(m.group(1))}d}",
            result,
        )
        if "%" in result:
            try:
                result = result % frame
            except Exception:
                pass
    return os.path.normpath(result)


def resolve_plate_sequence(
    path_pattern: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    step: int = 1,
    auto_detect_sequence: bool = True,
) -> List[Tuple[Optional[int], str]]:
    """
    Resolve a single plate path or sequence pattern into sorted (frame, path) pairs.

    Supports:
    - Houdini tokens: $F, $F4, ####, %04d
    - Concrete frame file (e.g. shot_0001.exr) when auto_detect_sequence=True
    - Single static image
    """
    raw = path_pattern.strip().strip('"').strip("'")
    has_token = bool(
        re.search(r"(\$F\d*|\$F|#+|%0?\d*d)", raw)
    )

    if has_token:
        # Preserve frame token across hou.text.expandString (which would expand $F4 to frame 1)
        placeholder = "__HDRIMAKER_FRAME__"
        tokenized = re.sub(r"(\$F\d*|\$F|#+|%0?\d*d)", placeholder, raw)
        probe = expand_houdini_path(tokenized, frame=None)
        directory = os.path.dirname(probe) or "."
        base = os.path.basename(probe)
        regex_str = re.escape(base).replace(placeholder, r"(\d+)")
        pattern = re.compile(f"^{regex_str}$", re.IGNORECASE)

        matches: List[Tuple[Optional[int], str]] = []
        if os.path.isdir(directory):
            for fname in os.listdir(directory):
                m = pattern.match(fname)
                if m:
                    fnum = int(m.group(1))
                    matches.append((fnum, os.path.join(directory, fname)))
        matches.sort(key=lambda x: x[0] if x[0] is not None else 0)
    else:
        expanded = expand_houdini_path(raw)
        if not auto_detect_sequence:
            return [(None, expanded)] if os.path.isfile(expanded) else []

        directory = os.path.dirname(expanded) or "."
        base = os.path.basename(expanded)
        m = re.match(r"^(.*?)(\d+)(\.[^.]+)$", base)
        if not m or not os.path.isdir(directory):
            return [(None, expanded)] if os.path.isfile(expanded) else []

        prefix, digits, ext = m.groups()
        pattern = re.compile(
            f"^{re.escape(prefix)}(\\d+){re.escape(ext)}$", re.IGNORECASE
        )
        matches = []
        for fname in os.listdir(directory):
            fm = pattern.match(fname)
            if fm:
                matches.append((int(fm.group(1)), os.path.join(directory, fname)))
        matches.sort(key=lambda x: x[0] if x[0] is not None else 0)
        if not matches and os.path.isfile(expanded):
            matches = [(None, expanded)]

    if start_frame is not None or end_frame is not None:
        filtered: List[Tuple[Optional[int], str]] = []
        for fnum, fpath in matches:
            if fnum is None:
                filtered.append((fnum, fpath))
                continue
            if start_frame is not None and fnum < start_frame:
                continue
            if end_frame is not None and fnum > end_frame:
                continue
            filtered.append((fnum, fpath))
        matches = filtered

    if step > 1 and len(matches) > 1:
        matches = matches[::step]

    return matches


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (IEC 61966-2-1) values to linear Rec.709."""
    arr = np.asarray(rgb, dtype=np.float32)
    sign = np.sign(arr)
    abs_arr = np.abs(arr)
    linear = np.where(
        abs_arr <= 0.04045,
        abs_arr / 12.92,
        np.power((abs_arr + 0.055) / 1.055, 2.4),
    )
    return (sign * linear).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear Rec.709 values to sRGB (IEC 61966-2-1)."""
    arr = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, None)
    srgb = np.where(
        arr <= 0.0031308,
        arr * 12.92,
        1.055 * np.power(arr, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def apply_color_matrix(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 color matrix to an (H, W, 3) RGB image."""
    return np.einsum("ij,hwj->hwi", matrix, rgb).astype(np.float32)


def read_plate(
    path: str,
    colorspace: str = "Auto",
    frame: Optional[int] = None,
    max_dim: Optional[int] = None,
) -> PlateFrame:
    """
    Read a plate image into 32-bit float Linear Rec.709 RGB.

    Args:
        path: File path to image (.exr, .hdr, .jpg, .png, .tif, etc.).
        colorspace: 'Auto', 'sRGB', 'Linear Rec.709', or 'ACEScg'.
        frame: Optional frame number associated with this plate.
        max_dim: Optional max width/height for fast analysis downsampling.
    """
    resolved = expand_houdini_path(path, frame=frame)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Plate file not found: {resolved}")

    ext = os.path.splitext(resolved)[1].lower()
    is_hdr = ext in HDR_EXTENSIONS
    pixels: Optional[np.ndarray] = None
    detected_cs = "Linear" if is_hdr else "sRGB"

    if HAS_OIIO:
        inp = oiio.ImageInput.open(resolved)
        if inp is None:
            raise RuntimeError(f"OpenImageIO failed to open {resolved}: {oiio.geterror()}")
        try:
            spec = inp.spec()
            data = inp.read_image(oiio.FLOAT)
            if data is None:
                raise RuntimeError(f"Failed to read pixels from {resolved}: {inp.geterror()}")
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            elif arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[2] >= 3:
                arr = arr[:, :, :3]
            pixels = np.ascontiguousarray(arr, dtype=np.float32)

            oiio_cs = spec.get_string_attribute("oiio:ColorSpace", "")
            if oiio_cs:
                detected_cs = oiio_cs
            if spec.format in (oiio.HALF, oiio.FLOAT, oiio.DOUBLE):
                is_hdr = True
        finally:
            inp.close()
    elif HAS_PIL:
        with Image.open(resolved) as img:
            rgb_img = img.convert("RGB")
            pixels = np.asarray(rgb_img, dtype=np.float32) / 255.0
    else:
        raise RuntimeError("Neither OpenImageIO nor PIL is available to read plates.")

    # Sanitize NaNs/Infs before any downsampling so a single bad EXR pixel cannot poison neighbors
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=65504.0, neginf=0.0)

    if max_dim and max(pixels.shape[0], pixels.shape[1]) > max_dim:
        pixels = downsample_image(pixels, max_dim=max_dim)

    # Normalize to Linear Rec.709
    cs_mode = colorspace.strip()
    if cs_mode == "Auto":
        if ext in LDR_EXTENSIONS or (not is_hdr and "linear" not in detected_cs.lower()):
            pixels = srgb_to_linear(pixels)
            source_cs = "sRGB"
        elif "acescg" in detected_cs.lower():
            pixels = apply_color_matrix(pixels, MAT_ACESCG_TO_REC709)
            source_cs = "ACEScg"
        else:
            source_cs = "Linear Rec.709"
    elif cs_mode == "sRGB":
        pixels = srgb_to_linear(pixels)
        source_cs = "sRGB"
    elif cs_mode == "ACEScg":
        pixels = apply_color_matrix(pixels, MAT_ACESCG_TO_REC709)
        source_cs = "ACEScg"
    else:
        source_cs = "Linear Rec.709"

    # Sanitize NaNs/Infs while preserving >1.0 HDR values
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=65504.0, neginf=0.0)
    pixels = np.maximum(pixels, 0.0).astype(np.float32)

    return PlateFrame(
        pixels=pixels,
        path=resolved,
        frame=frame,
        source_colorspace=source_cs,
        is_hdr=is_hdr,
    )


def downsample_image(pixels: np.ndarray, max_dim: int = 512) -> np.ndarray:
    """Anti-aliased area-prefiltered + bilinear downsampling in pure float32 numpy."""
    h, w = pixels.shape[:2]
    scale = max(h, w) / float(max_dim)
    if scale <= 1.0:
        return pixels
    work = pixels.astype(np.float32, copy=False)
    # Integer block-averaging when downsampling >= 2x so high-frequency edges don't alias
    bin_f = int(scale)
    if bin_f >= 2:
        bh = (h // bin_f) * bin_f
        bw = (w // bin_f) * bin_f
        if bh >= bin_f and bw >= bin_f:
            work = work[:bh, :bw].reshape(bh // bin_f, bin_f, bw // bin_f, bin_f, -1).mean(axis=(1, 3))
            h, w = work.shape[:2]
    new_h = max(1, int(round(h / (max(h, w) / float(max_dim)) if max(h, w) > max_dim else h)))
    new_w = max(1, int(round(w / (max(h, w) / float(max_dim)) if max(h, w) > max_dim else w)))
    if new_h == h and new_w == w:
        return np.ascontiguousarray(work, dtype=np.float32)
    yy = np.linspace(0.0, h - 1.0, new_h, dtype=np.float32)
    xx = np.linspace(0.0, w - 1.0, new_w, dtype=np.float32)
    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (yy - y0)[:, None, None]
    fx = (xx - x0)[None, :, None]
    out = work[np.ix_(y0, x0)] * ((1.0 - fx) * (1.0 - fy))
    out += work[np.ix_(y0, x1)] * (fx * (1.0 - fy))
    out += work[np.ix_(y1, x0)] * ((1.0 - fx) * fy)
    out += work[np.ix_(y1, x1)] * (fx * fy)
    return np.ascontiguousarray(out, dtype=np.float32)


def write_exr_32f(
    output_path: str,
    pixels: np.ndarray,
    output_colorspace: str = "Linear Rec.709",
    compression: str = "piz",
    metadata: Optional[Dict[str, str]] = None,
    frame: Optional[int] = None,
) -> str:
    """
    Write a true 32-bit float (oiio.FLOAT) Lat/Long EXR file.

    Args:
        output_path: Destination .exr path (supports $HIP/$F4 tokens).
        pixels: (H, W, 3) float32 array in Linear Rec.709.
        output_colorspace: 'Linear Rec.709' or 'ACEScg'.
        compression: EXR compression ('piz', 'zip', 'zips', 'none').
        metadata: Optional string metadata attributes written into EXR header.
        frame: Optional frame number for token expansion.

    Returns:
        Resolved output path written to disk.
    """
    if not HAS_OIIO:
        raise RuntimeError(
            "OpenImageIO is required to write true 32-bit float EXR files."
        )

    resolved = expand_houdini_path(output_path, frame=frame)
    if not resolved.lower().endswith(".exr"):
        resolved += ".exr"

    parent_dir = os.path.dirname(resolved)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    arr = np.asarray(pixels, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected (H, W, 3) or (H, W, 4) array, got shape {arr.shape}")

    if output_colorspace == "ACEScg":
        if arr.shape[2] == 4:
            rgb_acg = apply_color_matrix(arr[..., :3], MAT_REC709_TO_ACESCG)
            arr = np.concatenate([rgb_acg, arr[..., 3:4]], axis=-1)
        else:
            arr = apply_color_matrix(arr, MAT_REC709_TO_ACESCG)

    arr = np.ascontiguousarray(np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=0.0), dtype=np.float32)
    height, width, channels = arr.shape

    spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
    spec.attribute("compression", compression)
    spec.attribute(
        "oiio:ColorSpace",
        "ACEScg" if output_colorspace == "ACEScg" else "Linear",
    )
    spec.attribute("Software", "Houdini Solaris HDRI Breaker")
    spec.attribute("textureformat", "LatLong")
    spec.attribute("hdri_maker:envmap", "latlong")

    if metadata:
        for k, v in metadata.items():
            spec.attribute(str(k), str(v))

    out = oiio.ImageOutput.create(resolved)
    if out is None:
        raise RuntimeError(f"OpenImageIO could not create {resolved}: {oiio.geterror()}")

    try:
        if not out.open(resolved, spec):
            raise RuntimeError(f"Failed to open {resolved} for writing: {out.geterror()}")
        if not out.write_image(arr):
            raise RuntimeError(f"Failed to write 32-bit EXR pixels: {out.geterror()}")
    finally:
        out.close()

    return resolved


def merge_ldr_brackets_to_hdr(
    paths: List[str],
    ev_offsets: Optional[List[float]] = None,
    colorspace: str = "Auto",
) -> np.ndarray:
    """
    Merge multiple bracketed LDR/HDR exposures of the same camera view into a
    single unclamped 32-bit float Linear Rec.709 HDR image using hat-weighted
    Debevec/Robertson irradiance weighting.

    If `ev_offsets` is None, relative EV stops are automatically estimated from
    log-median midtone luminance ratios (or fallback 2.0 EV spacing).
    """
    if not paths:
        raise ValueError("At least one image path is required to merge brackets.")
    frames = [read_plate(p, colorspace=colorspace) for p in paths]
    if len(frames) == 1:
        return frames[0].pixels.copy()

    target_h, target_w = frames[0].pixels.shape[:2]
    imgs: List[np.ndarray] = []
    med_lums: List[float] = []
    for f in frames:
        px = f.pixels
        if px.shape[0] != target_h or px.shape[1] != target_w:
            y_idx = np.linspace(0, px.shape[0] - 1, target_h).astype(np.int32)
            x_idx = np.linspace(0, px.shape[1] - 1, target_w).astype(np.int32)
            px = px[np.ix_(y_idx, x_idx)]
        imgs.append(px)
        lum = 0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]
        mid = lum[(lum > 0.005) & (lum < 0.92)]
        med_lums.append(float(np.median(mid)) if mid.size > 32 else float(np.mean(lum) + 1e-4))

    # Sort exposures from darkest (shortest shutter) to brightest (longest shutter)
    order = np.argsort(med_lums)
    imgs = [imgs[i] for i in order]
    med_lums = [med_lums[i] for i in order]

    if ev_offsets is not None and len(ev_offsets) == len(imgs):
        evs = sorted(float(v) for v in ev_offsets)
    else:
        mid_idx = len(imgs) // 2
        ref_lum = max(med_lums[mid_idx], 1e-4)
        evs = []
        for idx, ml in enumerate(med_lums):
            ratio = max(ml, 1e-5) / ref_lum
            est_ev = float(np.log2(ratio))
            if abs(est_ev) < 0.15 and idx != mid_idx:
                est_ev = float((idx - mid_idx) * 2.0)
            evs.append(est_ev)

    accum_hdr = np.zeros((target_h, target_w, 3), dtype=np.float32)
    accum_w = np.zeros((target_h, target_w, 1), dtype=np.float32)

    n_exp = len(imgs)
    for i, (img, ev) in enumerate(zip(imgs, evs)):
        # Exposure scale factor: brighter bracket (ev > 0) is divided by 2^ev
        scale = float(2.0 ** (-ev))
        srgb_approx = np.clip(np.power(np.clip(img, 0.0, 1.0), 1.0 / 2.2), 0.0, 1.0)
        peak_c = np.max(srgb_approx, axis=-1, keepdims=True)
        min_c = np.min(srgb_approx, axis=-1, keepdims=True)

        # Hat weight peaking at 0.5
        w = 1.0 - 2.0 * np.abs(srgb_approx - 0.5)
        w = np.mean(np.clip(w, 0.02, 1.0), axis=-1, keepdims=True) ** 2

        if i == 0:
            # Darkest bracket preserves bright highlights without clipping penalty
            w = np.where(peak_c > 0.5, np.maximum(w, 0.85), w)
        else:
            # Penalize clipped highlights in brighter brackets
            w *= np.clip((0.98 - peak_c) / 0.15, 0.0, 1.0)

        if i == n_exp - 1:
            # Brightest bracket preserves deep shadows
            w = np.where(min_c < 0.5, np.maximum(w, 0.85), w)
        else:
            w *= np.clip((min_c - 0.01) / 0.08, 0.0, 1.0)

        w = np.maximum(w, 1e-4)
        accum_hdr += (img * scale) * w
        accum_w += w

    merged = accum_hdr / np.maximum(accum_w, 1e-5)
    return np.ascontiguousarray(np.maximum(merged, 0.0), dtype=np.float32)


def read_plate_metadata(path: str) -> Dict[str, Any]:
    """
    Extract camera FOV (horizontal degrees) and roll orientation (degrees) from
    EXIF / OpenImageIO metadata when available.

    Returns:
        {"fov_h_deg": Optional[float], "roll_deg": float, "focal_mm": Optional[float]}
    """
    result: Dict[str, Any] = {"fov_h_deg": None, "roll_deg": 0.0, "focal_mm": None}
    if not path or not os.path.isfile(path):
        return result

    focal_35: Optional[float] = None
    focal_mm: Optional[float] = None
    orient: int = 1
    width: int = 0
    height: int = 0

    if HAS_OIIO:
        try:
            inp = oiio.ImageInput.open(path)
            if inp is not None:
                spec = inp.spec()
                width, height = int(spec.width), int(spec.height)
                for key in ("Exif:FocalLengthIn35mmFilm", "FocalLengthIn35mmFilm"):
                    val = spec.get_float_attribute(key, 0.0)
                    if val > 1.0:
                        focal_35 = float(val)
                        break
                for key in ("Exif:FocalLength", "FocalLength", "cameraFocalLength"):
                    val = spec.get_float_attribute(key, 0.0)
                    if val > 1.0:
                        focal_mm = float(val)
                        break
                orient = int(spec.get_int_attribute("Orientation", 1))
                inp.close()
        except Exception:
            pass

    if focal_35 is None and HAS_PIL:
        try:
            with Image.open(path) as im:
                width, height = im.size
                exif = im.getexif() if hasattr(im, "getexif") else None
                if exif:
                    # 41989 = FocalLengthIn35mmFilm, 37386 = FocalLength, 274 = Orientation
                    f35 = exif.get(41989)
                    if f35 and float(f35) > 1.0:
                        focal_35 = float(f35)
                    fl = exif.get(37386)
                    if fl is not None:
                        focal_mm = float(fl)
                    ori = exif.get(274)
                    if ori is not None:
                        orient = int(ori)
        except Exception:
            pass

    if orient == 6:
        result["roll_deg"] = -90.0
    elif orient == 8:
        result["roll_deg"] = 90.0
    elif orient == 3:
        result["roll_deg"] = 180.0

    # Only trust FocalLengthIn35mmFilm, or raw focal_mm >= 14mm (avoid phone 4mm lens / 36mm sensor mismatch)
    used_focal = focal_35 if focal_35 is not None else (focal_mm if (focal_mm is not None and focal_mm >= 14.0) else None)
    if used_focal is not None and used_focal > 1.0:
        # 36mm full-frame sensor width baseline; adjust if portrait aspect
        sensor_w = 36.0 if width >= height or height == 0 else 24.0
        fov_rad = 2.0 * math.atan(sensor_w / (2.0 * used_focal))
        result["fov_h_deg"] = float(np.clip(math.degrees(fov_rad), 10.0, 165.0))
        result["focal_mm"] = float(used_focal)
    return result


def format_frame_path(pattern: str, frame: int) -> str:
    """Expand Houdini frame tokens ($F4, $F, ####) in an output path for a given frame."""
    if "$F" in pattern:
        def _repl_f(m: re.Match) -> str:
            pad = int(m.group(1)) if m.group(1) else 1
            return str(int(frame)).zfill(pad)
        return re.sub(r"\$F(\d*)", _repl_f, pattern)
    if "#" in pattern:
        return re.sub(r"#+", lambda m: str(int(frame)).zfill(len(m.group(0))), pattern)
    stem, ext = os.path.splitext(pattern)
    return f"{stem}.{int(frame):04d}{ext or '.exr'}"
