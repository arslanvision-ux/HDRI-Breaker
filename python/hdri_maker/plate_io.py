"""
Plate & image sequence I/O and true 32-bit float EXR writer.

Uses OpenImageIO (bundled with Houdini 20+) with a PIL fallback for LDR plates.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
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

try:
    import PyOpenColorIO as OCIO
    HAS_OCIO = True
except ImportError:
    OCIO = None
    HAS_OCIO = False


LDR_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tga", ".tif", ".tiff"}
HDR_EXTENSIONS = {".exr", ".hdr", ".pic", ".rat"}
RAW_EXTENSIONS = {
    ".dng", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".raf", ".orf", ".rw2", ".pef", ".srw", ".rwl", ".3fr", ".iiq",
    ".gpr", ".raw",
}
HEIF_EXTENSIONS = {".heic", ".heif"}


def _read_exif_from_tiff_container(path: str) -> Dict[str, Any]:
    """Pure-Python TIFF/EXIF IFD parser for DNG/CR2/NEF/ARW/TIFF Orientation & FocalLength."""
    out: Dict[str, Any] = {}
    try:
        with open(path, "rb") as f:
            hdr = f.read(8)
            if len(hdr) < 8:
                return out
            if hdr[:2] == b"II":
                bo = "<"
            elif hdr[:2] == b"MM":
                bo = ">"
            else:
                return out
            magic = struct.unpack(bo + "H", hdr[2:4])[0]
            if magic != 42:
                return out
            ifd0_off = struct.unpack(bo + "I", hdr[4:8])[0]

            def _read_ifd(offset: int) -> Dict[int, Any]:
                tags: Dict[int, Any] = {}
                if offset <= 0:
                    return tags
                f.seek(offset)
                raw_cnt = f.read(2)
                if len(raw_cnt) < 2:
                    return tags
                cnt = struct.unpack(bo + "H", raw_cnt)[0]
                if cnt > 512:
                    return tags
                entries = f.read(cnt * 12)
                for i in range(0, len(entries) - 11, 12):
                    tag, typ, nval, val_off = struct.unpack(bo + "HHII", entries[i : i + 12])
                    if typ == 3 and nval == 1:  # SHORT
                        v = struct.unpack(bo + "H", entries[i + 8 : i + 10])[0]
                        tags[tag] = v
                    elif typ == 4 and nval == 1:  # LONG
                        tags[tag] = val_off
                    elif typ == 2 and nval > 4 and nval <= 64:  # ASCII
                        cur = f.tell()
                        f.seek(val_off)
                        raw_s = f.read(nval)
                        f.seek(cur)
                        tags[tag] = raw_s.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
                    elif typ == 5 and nval == 1:  # RATIONAL
                        cur = f.tell()
                        f.seek(val_off)
                        rb = f.read(8)
                        f.seek(cur)
                        if len(rb) == 8:
                            num, den = struct.unpack(bo + "II", rb)
                            if den != 0:
                                tags[tag] = float(num) / float(den)
                return tags

            ifd0 = _read_ifd(ifd0_off)
            if 274 in ifd0:
                out["orient"] = int(ifd0[274])
            exif_off = ifd0.get(34665)
            if exif_off:
                exif_ifd = _read_ifd(int(exif_off))
                if 41989 in exif_ifd:
                    out["focal_35"] = float(exif_ifd[41989])
                if 37386 in exif_ifd:
                    out["focal_mm"] = float(exif_ifd[37386])
                if 33434 in exif_ifd:
                    out["exposure_time"] = float(exif_ifd[33434])
                if 33437 in exif_ifd:
                    out["f_number"] = float(exif_ifd[33437])
                if 34855 in exif_ifd:
                    out["iso"] = int(exif_ifd[34855])
                if 36867 in exif_ifd:
                    out["datetime"] = str(exif_ifd[36867])
    except Exception:
        pass
    return out


def _decode_with_windows_wic(path: str) -> Optional[np.ndarray]:
    """Decode DNG/RAW/HEIC via Windows Imaging Component (PresentationCore) if codecs are installed."""
    if os.name != "nt":
        return None
    tmp_png = None
    try:
        fd, tmp_png = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        ps_script = (
            "Add-Type -AssemblyName PresentationCore; "
            f"$src = '{path.replace(chr(39), chr(39)*2)}'; "
            f"$dst = '{tmp_png.replace(chr(39), chr(39)*2)}'; "
            "$stream = [System.IO.File]::OpenRead($src); "
            "try { "
            "$dec = [System.Windows.Media.Imaging.BitmapDecoder]::Create($stream, "
            "[System.Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat, "
            "[System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad); "
            "$enc = New-Object System.Windows.Media.Imaging.PngBitmapEncoder; "
            "$enc.Frames.Add($dec.Frames[0]); "
            "$out = [System.IO.File]::Create($dst); "
            "try { $enc.Save($out) } finally { $out.Dispose() } "
            "} finally { $stream.Dispose() }"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            timeout=25,
        )
        if res.returncode == 0 and os.path.isfile(tmp_png) and os.path.getsize(tmp_png) > 0:
            if HAS_OIIO:
                inp = oiio.ImageInput.open(tmp_png)
                if inp is not None:
                    try:
                        data = inp.read_image(oiio.FLOAT)
                        if data is not None:
                            arr = np.asarray(data, dtype=np.float32)
                            if arr.ndim == 2:
                                arr = np.repeat(arr[:, :, None], 3, axis=2)
                            elif arr.shape[2] >= 3:
                                arr = arr[:, :, :3]
                            return np.ascontiguousarray(arr, dtype=np.float32)
                    finally:
                        inp.close()
            elif HAS_PIL:
                with Image.open(tmp_png) as im:
                    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        pass
    finally:
        if tmp_png and os.path.isfile(tmp_png):
            try:
                os.remove(tmp_png)
            except OSError:
                pass
    return None


def _read_raw_or_heif_pixels(resolved: str, ext: str) -> Tuple[np.ndarray, bool, str]:
    """
    Decode camera/phone RAW (.dng, ProRAW, .cr2, .nef, .arw, etc.) or .heic/.heif
    into (pixels_float32_rgb, is_hdr, detected_colorspace).
    """
    if ext in RAW_EXTENSIONS:
        # 1. rawpy (LibRaw) -> true 16-bit linear Rec.709 / sRGB primaries
        try:
            import rawpy

            with rawpy.imread(resolved) as raw:
                rgb16 = raw.postprocess(
                    output_bps=16,
                    no_auto_bright=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    gamma=(1, 1),
                    use_camera_wb=True,
                )
            arr = np.asarray(rgb16, dtype=np.float32) / 65535.0
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            elif arr.shape[2] >= 3:
                arr = arr[:, :, :3]
            return np.ascontiguousarray(arr, dtype=np.float32), True, "Linear"
        except ImportError:
            pass

        # 2. dcraw CLI -> 16-bit linear TIFF
        dcraw_bin = shutil.which("dcraw")
        if dcraw_bin:
            tmp_tif = None
            try:
                res = subprocess.run([dcraw_bin, "-c", "-w", "-4", "-T", resolved], capture_output=True, timeout=45)
                if res.returncode == 0 and res.stdout:
                    fd, tmp_tif = tempfile.mkstemp(suffix=".tif")
                    os.close(fd)
                    with open(tmp_tif, "wb") as tf:
                        tf.write(res.stdout)
                    if HAS_OIIO:
                        inp = oiio.ImageInput.open(tmp_tif)
                        if inp is not None:
                            try:
                                data = inp.read_image(oiio.FLOAT)
                                if data is not None:
                                    arr = np.asarray(data, dtype=np.float32)[:, :, :3]
                                    return np.ascontiguousarray(arr, dtype=np.float32), True, "Linear"
                            finally:
                                inp.close()
            except Exception:
                pass
            finally:
                if tmp_tif and os.path.isfile(tmp_tif):
                    try:
                        os.remove(tmp_tif)
                    except OSError:
                        pass

        # 3. ImageMagick / darktable-cli -> temp 32-bit EXR
        for cli_name in ("magick", "darktable-cli"):
            cli_bin = shutil.which(cli_name)
            if not cli_bin:
                continue
            tmp_exr = None
            try:
                fd, tmp_exr = tempfile.mkstemp(suffix=".exr")
                os.close(fd)
                cmd = (
                    [cli_bin, resolved, "-auto-orient", tmp_exr]
                    if cli_name == "magick"
                    else [cli_bin, resolved, tmp_exr]
                )
                res = subprocess.run(cmd, capture_output=True, timeout=60)
                if res.returncode == 0 and os.path.isfile(tmp_exr) and os.path.getsize(tmp_exr) > 0 and HAS_OIIO:
                    inp = oiio.ImageInput.open(tmp_exr)
                    if inp is not None:
                        try:
                            data = inp.read_image(oiio.FLOAT)
                            if data is not None:
                                arr = np.asarray(data, dtype=np.float32)[:, :, :3]
                                return np.ascontiguousarray(arr, dtype=np.float32), True, "Linear"
                        finally:
                            inp.close()
            except Exception:
                pass
            finally:
                if tmp_exr and os.path.isfile(tmp_exr):
                    try:
                        os.remove(tmp_exr)
                    except OSError:
                        pass

        # 4. Windows WIC (supports DNG/RAW if Windows Raw Image Extension is installed)
        wic_arr = _decode_with_windows_wic(resolved)
        if wic_arr is not None:
            return wic_arr, False, "sRGB"

        # 5. PIL fallback (works for some uncompressed/Linear DNG IFDs)
        if HAS_PIL:
            try:
                with Image.open(resolved) as img:
                    try:
                        from PIL import ImageOps

                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        pass
                    if min(img.size) >= 256:
                        rgb_img = img.convert("RGB")
                        arr = np.asarray(rgb_img, dtype=np.float32) / 255.0
                        return np.ascontiguousarray(arr, dtype=np.float32), False, "sRGB"
            except Exception:
                pass

        raise RuntimeError(
            f"Cannot decode camera/phone RAW file '{os.path.basename(resolved)}' with built-in Houdini OIIO/PIL. "
            "Install 'rawpy' into Houdini Python (`hython -m pip install rawpy`) or install ImageMagick / dcraw."
        )

    if ext in HEIF_EXTENSIONS:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
            if HAS_PIL:
                with Image.open(resolved) as img:
                    try:
                        from PIL import ImageOps

                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        pass
                    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                    return np.ascontiguousarray(arr, dtype=np.float32), False, "sRGB"
        except ImportError:
            pass

        cli_bin = shutil.which("magick")
        if cli_bin:
            tmp_png = None
            try:
                fd, tmp_png = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                res = subprocess.run([cli_bin, resolved, "-auto-orient", tmp_png], capture_output=True, timeout=45)
                if res.returncode == 0 and os.path.isfile(tmp_png) and HAS_PIL:
                    with Image.open(tmp_png) as img:
                        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                        return np.ascontiguousarray(arr, dtype=np.float32), False, "sRGB"
            except Exception:
                pass
            finally:
                if tmp_png and os.path.isfile(tmp_png):
                    try:
                        os.remove(tmp_png)
                    except OSError:
                        pass

        wic_arr = _decode_with_windows_wic(resolved)
        if wic_arr is not None:
            return wic_arr, False, "sRGB"

        raise RuntimeError(
            f"Cannot decode HEIC/HEIF file '{os.path.basename(resolved)}'. "
            "Install 'pillow-heif' into Houdini Python (`hython -m pip install pillow-heif`) or ImageMagick."
        )

    raise RuntimeError(f"Unsupported RAW/HEIF extension: {ext}")

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


# Built-in pure-numpy camera log & cinema colorspace decoders (zero external deps)
BUILTIN_INPUT_COLORSPACES: Tuple[str, ...] = (
    "Auto",
    "sRGB",
    "Linear Rec.709",
    "ACEScg",
    "ARRI LogC3",
    "ARRI LogC4",
    "Sony S-Log3",
    "RED Log3G10",
    "Canon Log 2",
    "ACEScct",
)

# Approximate wide-gamut -> Linear Rec.709 matrices (D65)
MAT_AWG3_TO_REC709 = np.array(
    [
        [1.617523, -0.537287, -0.080237],
        [-0.070573, 1.334613, -0.264040],
        [-0.021102, -0.226954, 1.248056],
    ],
    dtype=np.float32,
)
MAT_AWG4_TO_REC709 = np.array(
    [
        [1.810412, -0.713163, -0.097249],
        [-0.129269, 1.430117, -0.300848],
        [0.015648, -0.273664, 1.258016],
    ],
    dtype=np.float32,
)


def list_available_input_colorspaces() -> List[str]:
    """Return built-in input colorspaces plus any extra spaces from Houdini's active OCIO config."""
    spaces = list(BUILTIN_INPUT_COLORSPACES)
    if HAS_OCIO:
        try:
            cfg = OCIO.GetCurrentConfig()
            for cs in cfg.getColorSpaces():
                name = str(cs.getName())
                if name and name not in spaces:
                    spaces.append(f"OCIO: {name}")
        except Exception:
            pass
    return spaces


def _logc3_to_linear(x: np.ndarray) -> np.ndarray:
    """ARRI LogC3 (EI 800) to Linear AWG3 -> Rec.709."""
    arr = np.asarray(x, dtype=np.float32)
    cut = 0.010591 * 5.367655 + 0.092809  # 0.1496582
    lin = np.where(
        arr > cut,
        (np.power(10.0, (arr - 0.385537) / 0.247190) - 0.052272) / 5.555556,
        (arr - 0.092809) / 5.367655,
    )
    return apply_color_matrix(lin.astype(np.float32), MAT_AWG3_TO_REC709)


def _logc4_to_linear(x: np.ndarray) -> np.ndarray:
    """ARRI LogC4 to Linear AWG4 -> Rec.709."""
    arr = np.asarray(x, dtype=np.float32)
    a = (2.0 ** 18 - 16.0) / 117.45
    b = (1023.0 - 95.0) / 1023.0
    c = 95.0 / 1023.0
    s = (7.0 * math.log(2.0) * (2.0 ** (7.0 - 14.0 * c / b))) / (a * b)
    t = (2.0 ** (14.0 * (-c / b) + 6.0) - 64.0) / a
    lin = np.where(
        arr >= 0.0,
        (np.power(2.0, 14.0 * ((arr - c) / b) + 6.0) - 64.0) / a,
        arr * s + t,
    )
    return apply_color_matrix(lin.astype(np.float32), MAT_AWG4_TO_REC709)


def _slog3_to_linear(x: np.ndarray) -> np.ndarray:
    """Sony S-Log3 to scene-linear."""
    arr = np.asarray(x, dtype=np.float32)
    cut = 171.2102946929 / 1023.0
    lin = np.where(
        arr >= cut,
        np.power(10.0, (arr * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01,
        (arr * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0),
    )
    return lin.astype(np.float32)


def _red_log3g10_to_linear(x: np.ndarray) -> np.ndarray:
    """RED Log3G10 to scene-linear."""
    arr = np.asarray(x, dtype=np.float32)
    a = 0.224282
    b = 155.975327
    c = 0.01
    g = 15.1927
    lin = np.where(
        arr < 0.0,
        arr / g - c,
        (np.power(10.0, arr / a) - 1.0) / b - c,
    )
    return lin.astype(np.float32)


def _canon_log2_to_linear(x: np.ndarray) -> np.ndarray:
    """Canon Log 2 to scene-linear."""
    arr = np.asarray(x, dtype=np.float32)
    c_off = 0.092864125
    lin = np.where(
        arr >= c_off,
        (np.power(10.0, (arr - c_off) / 0.24136077) - 1.0) / 87.099375,
        -(np.power(10.0, (c_off - arr) / 0.24136077) - 1.0) / 87.099375,
    )
    return lin.astype(np.float32)


def _acescct_to_linear_rec709(x: np.ndarray) -> np.ndarray:
    """ACEScct (AP1 log) to Linear Rec.709."""
    arr = np.asarray(x, dtype=np.float32)
    lin_ap1 = np.where(
        arr <= 0.155251141552511,
        (arr - 0.0729055341958355) / 10.5402377416545,
        np.power(2.0, arr * 17.52 - 9.72),
    )
    return apply_color_matrix(lin_ap1.astype(np.float32), MAT_ACESCG_TO_REC709)


def convert_colorspace_to_linear_rec709(pixels: np.ndarray, cs_name: str) -> Tuple[np.ndarray, str]:
    """
    Convert an (H, W, 3) float32 array from `cs_name` to Linear Rec.709.
    Supports built-in camera log spaces and Houdini's active PyOpenColorIO config.
    """
    cs = (cs_name or "").strip()
    if cs == "sRGB":
        return srgb_to_linear(pixels), "sRGB"
    if cs == "ACEScg":
        return apply_color_matrix(pixels, MAT_ACESCG_TO_REC709), "ACEScg"
    if cs == "ARRI LogC3":
        return _logc3_to_linear(pixels), "ARRI LogC3"
    if cs == "ARRI LogC4":
        return _logc4_to_linear(pixels), "ARRI LogC4"
    if cs == "Sony S-Log3":
        return _slog3_to_linear(pixels), "Sony S-Log3"
    if cs == "RED Log3G10":
        return _red_log3g10_to_linear(pixels), "RED Log3G10"
    if cs == "Canon Log 2":
        return _canon_log2_to_linear(pixels), "Canon Log 2"
    if cs == "ACEScct":
        return _acescct_to_linear_rec709(pixels), "ACEScct"
    if cs in ("Linear Rec.709", "Linear", "Raw"):
        return pixels, "Linear Rec.709"

    # Try PyOpenColorIO if available (handles "OCIO: <name>" or raw OCIO colorspace name)
    ocio_name = cs[6:].strip() if cs.startswith("OCIO:") else cs
    if HAS_OCIO and ocio_name:
        try:
            cfg = OCIO.GetCurrentConfig()
            dst_name = "Linear Rec.709 (sRGB)"
            avail = {str(c.getName()) for c in cfg.getColorSpaces()}
            if dst_name not in avail:
                dst_name = cfg.getCanonicalName(OCIO.ROLE_SCENE_LINEAR)
            if ocio_name in avail and dst_name:
                proc = cfg.getProcessor(ocio_name, dst_name)
                cpu = proc.getDefaultCPUProcessor()
                buf = np.ascontiguousarray(pixels, dtype=np.float32).copy()
                cpu.applyRGB(buf)
                return buf, ocio_name
        except Exception:
            pass
    return pixels, "Linear Rec.709"


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

    if ext in RAW_EXTENSIONS or ext in HEIF_EXTENSIONS:
        pixels, is_hdr, detected_cs = _read_raw_or_heif_pixels(resolved, ext)
    elif HAS_OIIO:
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

            # Respect EXIF Orientation (1..8) for LDR photos so upside-down/rotated shots load upright
            if not is_hdr:
                orient = int(spec.get_int_attribute("Orientation", 1) or 1)
                if orient == 2:
                    pixels = pixels[:, ::-1, :]
                elif orient == 3:
                    pixels = pixels[::-1, ::-1, :]
                elif orient == 4:
                    pixels = pixels[::-1, :, :]
                elif orient == 5:
                    pixels = np.transpose(pixels, (1, 0, 2))
                elif orient == 6:
                    pixels = np.rot90(pixels, k=3)
                elif orient == 7:
                    pixels = np.rot90(pixels, k=3)[:, ::-1, :]
                elif orient == 8:
                    pixels = np.rot90(pixels, k=1)
                if orient != 1:
                    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        finally:
            inp.close()
    elif HAS_PIL:
        with Image.open(resolved) as img:
            try:
                from PIL import ImageOps

                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
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
    else:
        pixels, source_cs = convert_colorspace_to_linear_rec709(pixels, cs_mode)

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


def is_equirectangular_image(path: str, width: int, height: int) -> bool:
    """
    Detect whether an input plate is already an unwrapped 360° Lat/Long
    (equirectangular, 2:1 aspect ratio) environment map.
    """
    if height > 0 and width > 0:
        aspect = float(width) / float(height)
        if 1.95 <= aspect <= 2.05:
            return True
    if path and os.path.isfile(path) and HAS_OIIO:
        try:
            inp = oiio.ImageInput.open(path)
            if inp is not None:
                spec = inp.spec()
                tex_fmt = str(spec.get_string_attribute("textureformat", "") or "").lower()
                env_attr = str(spec.get_string_attribute("hdri_maker:envmap", "") or "").lower()
                proj_attr = str(spec.get_string_attribute("GPano:ProjectionType", "") or "").lower()
                inp.close()
                if "latlong" in tex_fmt or "latlong" in env_attr or "equirectangular" in proj_attr:
                    return True
        except Exception:
            pass
    return False


def _phase_correlate_shift(ref_gray: np.ndarray, mov_gray: np.ndarray, max_shift: int = 48) -> Tuple[int, int]:
    """Estimate 2D integer pixel shift (dy, dx) to align `mov_gray` onto `ref_gray` via FFT phase correlation."""
    h, w = ref_gray.shape[:2]
    if h < 16 or w < 16:
        return 0, 0
    # Downsample to <= 512px for fast, noise-robust alignment
    scale = 1
    if max(h, w) > 512:
        scale = max(1, max(h, w) // 512)
        ref_s = ref_gray[::scale, ::scale]
        mov_s = mov_gray[::scale, ::scale]
    else:
        ref_s = ref_gray
        mov_s = mov_gray

    # Normalize log-luminance so different exposures share identical edge structure
    r_log = np.log1p(np.maximum(ref_s, 0.0) * 16.0)
    m_log = np.log1p(np.maximum(mov_s, 0.0) * 16.0)
    r_log = r_log - float(np.mean(r_log))
    m_log = m_log - float(np.mean(m_log))

    win = np.outer(np.hanning(r_log.shape[0]), np.hanning(r_log.shape[1])).astype(np.float32)
    fr = np.fft.rfft2(r_log * win)
    fm = np.fft.rfft2(m_log * win)
    cross = fr * np.conj(fm)
    mag = np.abs(cross)
    corr = np.fft.irfft2(cross / np.maximum(mag, 1e-8), s=r_log.shape)

    sh, sw = corr.shape
    max_s = max(2, int(round(max_shift / float(scale))))
    # Restrict search window around (0, 0) wrapped
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = int(iy if iy <= sh // 2 else iy - sh)
    dx = int(ix if ix <= sw // 2 else ix - sw)
    if abs(dy) > max_s or abs(dx) > max_s:
        return 0, 0
    return int(dy * scale), int(dx * scale)


def _shift_image_clamp(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift an (H, W, 3) image by (dy, dx) pixels with edge clamping."""
    if dy == 0 and dx == 0:
        return img
    h, w = img.shape[:2]
    ys = np.clip(np.arange(h, dtype=np.int32) - dy, 0, h - 1)
    xs = np.clip(np.arange(w, dtype=np.int32) - dx, 0, w - 1)
    return np.ascontiguousarray(img[np.ix_(ys, xs)], dtype=np.float32)


def inspect_bracket_paths(
    paths: List[str],
    colorspace: str = "Auto",
) -> List[Dict[str, Any]]:
    """
    Inspect candidate bracket exposure files for the Interactive Bracket Merge Dialog.
    Returns a list of dicts (darkest-to-brightest) with EXIF fields and estimated EV.
    """
    rows: List[Dict[str, Any]] = []
    lums: List[np.ndarray] = []
    for idx, p in enumerate(paths):
        meta = read_plate_metadata(p)
        frame = read_plate(p, colorspace=colorspace, max_dim=512)
        px = frame.pixels
        lum = (0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]).astype(np.float32)
        lums.append(lum)
        rows.append(
            {
                "orig_index": idx,
                "path": p,
                "filename": os.path.basename(p),
                "exposure_time": meta.get("exposure_time"),
                "f_number": meta.get("f_number"),
                "iso": meta.get("iso"),
                "ev100": meta.get("ev100"),
                "mean_lum": float(np.mean(lum)),
                "ev_offset": 0.0,
                "ev_source": "Estimated",
            }
        )

    order = np.argsort([r["mean_lum"] for r in rows])
    rows = [rows[int(i)] for i in order]
    lums = [lums[int(i)] for i in order]

    exif_evs = [r["ev100"] for r in rows]
    if rows and all(e is not None for e in exif_evs):
        med_ev = float(np.median([float(e) for e in exif_evs]))
        for r, e in zip(rows, exif_evs):
            r["ev_offset"] = round(med_ev - float(e), 2)
            r["ev_source"] = "EXIF"
    elif len(rows) > 1:
        cum_evs = [0.0]
        for idx in range(len(lums) - 1):
            la, lb = lums[idx], lums[idx + 1]
            if la.shape != lb.shape:
                y_i = np.linspace(0, lb.shape[0] - 1, la.shape[0]).astype(np.int32)
                x_i = np.linspace(0, lb.shape[1] - 1, la.shape[1]).astype(np.int32)
                lb = lb[np.ix_(y_i, x_i)]
            overlap = (la > 0.015) & (la < 0.75) & (lb > 0.015) & (lb < 0.75)
            if int(np.count_nonzero(overlap)) >= 64:
                ratios = lb[overlap] / np.maximum(la[overlap], 1e-6)
                delta_ev = float(np.median(np.log2(np.maximum(ratios, 1e-4))))
                if not math.isfinite(delta_ev) or delta_ev < 0.15:
                    delta_ev = 2.0
                delta_ev = float(np.clip(delta_ev, 0.25, 6.0))
            else:
                delta_ev = 2.0
            cum_evs.append(cum_evs[-1] + delta_ev)
        mid_ev = cum_evs[len(cum_evs) // 2]
        for r, e in zip(rows, cum_evs):
            r["ev_offset"] = round(e - mid_ev, 2)
            r["ev_source"] = "Estimated"
    return rows


def merge_ldr_brackets_to_hdr(
    paths: List[str],
    ev_offsets: Optional[List[float]] = None,
    colorspace: str = "Auto",
    align: bool = False,
    deghost: bool = False,
    crf_calibrate: bool = False,
) -> np.ndarray:
    """
    Merge multiple bracketed LDR/HDR exposures of the same camera view into a
    single unclamped 32-bit float Linear Rec.709 HDR image using hat-weighted
    Debevec/Robertson irradiance weighting.

    Args:
        paths: List of bracketed exposure image paths.
        ev_offsets: Optional explicit EV offset per path (same order as `paths`).
        colorspace: Input colorspace ('Auto', 'sRGB', 'ARRI LogC3', etc.).
        align: Align handheld/shifted brackets to the middle exposure via phase correlation.
        deghost: Suppress moving objects (people, clouds, foliage, water) via irradiance consensus.
        crf_calibrate: Estimate & refine camera response gamma from overlapping midtones.
    """
    if not paths:
        raise ValueError("At least one image path is required to merge brackets.")
    frames = [read_plate(p, colorspace=colorspace) for p in paths]
    if len(frames) == 1:
        return frames[0].pixels.copy()

    target_h, target_w = frames[0].pixels.shape[:2]
    imgs: List[np.ndarray] = []
    lums: List[np.ndarray] = []
    sort_scores: List[float] = []
    for f in frames:
        px = f.pixels
        if px.shape[0] != target_h or px.shape[1] != target_w:
            y_idx = np.linspace(0, px.shape[0] - 1, target_h).astype(np.int32)
            x_idx = np.linspace(0, px.shape[1] - 1, target_w).astype(np.int32)
            px = px[np.ix_(y_idx, x_idx)]
        imgs.append(px)
        lum = (0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]).astype(np.float32)
        lums.append(lum)
        # Mean luminance across the whole frame is strictly monotonic with exposure
        # (even in night/low-key scenes where most pixels are near black).
        sort_scores.append(float(np.mean(lum)))

    # Sort exposures from darkest (shortest shutter) to brightest (longest shutter)
    order = np.argsort(sort_scores)
    imgs = [imgs[int(i)] for i in order]
    lums = [lums[int(i)] for i in order]

    # Optional handheld bracket alignment to middle reference frame
    if align and len(imgs) >= 2:
        ref_idx = len(imgs) // 2
        ref_lum = lums[ref_idx]
        for idx in range(len(imgs)):
            if idx == ref_idx:
                continue
            dy, dx = _phase_correlate_shift(ref_lum, lums[idx])
            if dy != 0 or dx != 0:
                imgs[idx] = _shift_image_clamp(imgs[idx], dy, dx)
                lums[idx] = (
                    0.2126 * imgs[idx][..., 0]
                    + 0.7152 * imgs[idx][..., 1]
                    + 0.0722 * imgs[idx][..., 2]
                ).astype(np.float32)

    if ev_offsets is not None and len(ev_offsets) == len(imgs):
        evs = [float(ev_offsets[int(i)]) for i in order]
    else:
        exif_evs = [read_plate_metadata(paths[int(i)]).get("ev100") for i in order]
        if all(e is not None for e in exif_evs):
            med_ev = float(np.median([float(e) for e in exif_evs]))
            evs = [med_ev - float(e) for e in exif_evs]
        else:
            # Pairwise matched-pixel EV estimation between adjacent sorted exposures:
            # avoids night-scene bias where unclipped midtone pixel sets differ per exposure.
            cum_evs = [0.0]
            for idx in range(len(lums) - 1):
                la = lums[idx]
                lb = lums[idx + 1]
                overlap = (la > 0.015) & (la < 0.75) & (lb > 0.015) & (lb < 0.75)
                if int(np.count_nonzero(overlap)) >= 64:
                    ratios = lb[overlap] / np.maximum(la[overlap], 1e-6)
                    delta_ev = float(np.median(np.log2(np.maximum(ratios, 1e-4))))
                    if not math.isfinite(delta_ev) or delta_ev < 0.15:
                        delta_ev = 2.0
                    delta_ev = float(np.clip(delta_ev, 0.25, 6.0))
                else:
                    delta_ev = 2.0
                cum_evs.append(cum_evs[-1] + delta_ev)
            mid_ev = cum_evs[len(cum_evs) // 2]
            evs = [e - mid_ev for e in cum_evs]

    # Optional Camera Response Function (CRF) gamma refinement for LDR brackets:
    # Fits residual power-law gamma exponent `g_corr` so log(L_b / L_a) == delta_ev * log(2)
    if crf_calibrate and len(imgs) >= 2:
        g_estimates = []
        for idx in range(len(imgs) - 1):
            dev = float(evs[idx + 1] - evs[idx])
            if dev < 0.35:
                continue
            la, lb = lums[idx], lums[idx + 1]
            overlap = (la > 0.02) & (la < 0.65) & (lb > 0.04) & (lb < 0.85)
            if int(np.count_nonzero(overlap)) >= 128:
                obs_ev = float(np.median(np.log2(np.maximum(lb[overlap], 1e-6) / np.maximum(la[overlap], 1e-6))))
                if obs_ev > 0.15:
                    g_estimates.append(dev / obs_ev)
        if g_estimates:
            g_corr = float(np.clip(np.median(g_estimates), 0.72, 1.38))
            if abs(g_corr - 1.0) > 0.025:
                imgs = [np.power(np.maximum(im, 0.0), g_corr).astype(np.float32) for im in imgs]

    n_exp = len(imgs)
    scaled_imgs: List[np.ndarray] = []
    weights: List[np.ndarray] = []

    for i, (img, ev) in enumerate(zip(imgs, evs)):
        # Exposure scale factor: brighter bracket (ev > 0) is divided by 2^ev
        scale = float(2.0 ** (-ev))
        scaled_imgs.append(img * scale)
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

        weights.append(np.maximum(w, 1e-4).astype(np.float32))

    # Optional deghosting: suppress brackets that strongly disagree with the middle reference
    # exposure where the reference itself is well-exposed (neither clipped nor crushed).
    if deghost and n_exp >= 3:
        ref_idx = n_exp // 2
        ref_rad = scaled_imgs[ref_idx]
        ref_lum_raw = lums[ref_idx]
        ref_valid = ((ref_lum_raw > 0.015) & (ref_lum_raw < 0.85))[..., None]
        ref_rad_lum = (0.2126 * ref_rad[..., 0] + 0.7152 * ref_rad[..., 1] + 0.0722 * ref_rad[..., 2])[..., None]
        for i in range(n_exp):
            if i == ref_idx:
                continue
            cur_rad = scaled_imgs[i]
            cur_rad_lum = (0.2126 * cur_rad[..., 0] + 0.7152 * cur_rad[..., 1] + 0.0722 * cur_rad[..., 2])[..., None]
            # Log-ratio deviation in EV stops
            ev_diff = np.abs(np.log2(np.maximum(cur_rad_lum, 1e-5) / np.maximum(ref_rad_lum, 1e-5)))
            # Smoothly suppress weight where deviation exceeds 1.1 stops
            ghost_suppress = np.exp(-0.5 * np.maximum(ev_diff - 0.9, 0.0) ** 2 / (0.65 ** 2)).astype(np.float32)
            weights[i] = np.where(ref_valid, weights[i] * ghost_suppress, weights[i])

    accum_hdr = np.zeros((target_h, target_w, 3), dtype=np.float32)
    accum_w = np.zeros((target_h, target_w, 1), dtype=np.float32)
    for sc, w in zip(scaled_imgs, weights):
        accum_hdr += sc * w
        accum_w += w

    merged = accum_hdr / np.maximum(accum_w, 1e-5)
    return np.ascontiguousarray(np.maximum(merged, 0.0), dtype=np.float32)


def read_plate_metadata(path: str) -> Dict[str, Any]:
    """
    Extract camera FOV (horizontal degrees), roll orientation (degrees), and
    EXIF exposure/timestamp metadata when available.

    Returns:
        {
            "fov_h_deg": Optional[float],
            "roll_deg": float,
            "focal_mm": Optional[float],
            "exposure_time": Optional[float],
            "f_number": Optional[float],
            "iso": Optional[int],
            "ev100": Optional[float],
            "timestamp": Optional[float],
        }
    """
    result: Dict[str, Any] = {
        "fov_h_deg": None,
        "roll_deg": 0.0,
        "focal_mm": None,
        "exposure_time": None,
        "f_number": None,
        "iso": None,
        "ev100": None,
        "timestamp": None,
    }
    if not path or not os.path.isfile(path):
        return result

    focal_35: Optional[float] = None
    focal_mm: Optional[float] = None
    exposure_time: Optional[float] = None
    f_number: Optional[float] = None
    iso_val: Optional[int] = None
    dt_str: str = ""
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
                for key in ("ExposureTime", "Exif:ExposureTime"):
                    val = spec.get_float_attribute(key, 0.0)
                    if val > 0.0:
                        exposure_time = float(val)
                        break
                for key in ("FNumber", "Exif:FNumber"):
                    val = spec.get_float_attribute(key, 0.0)
                    if val > 0.0:
                        f_number = float(val)
                        break
                for key in ("Exif:ISOSpeedRatings", "ISOSpeedRatings", "Exif:PhotographicSensitivity"):
                    val = spec.get_int_attribute(key, 0)
                    if val > 0:
                        iso_val = int(val)
                        break
                for key in ("Exif:DateTimeOriginal", "DateTime"):
                    s = spec.get_string_attribute(key, "")
                    if s:
                        dt_str = str(s)
                        break
                orient = int(spec.get_int_attribute("Orientation", 1))
                inp.close()
        except Exception:
            pass

    if (focal_35 is None or exposure_time is None or f_number is None) and HAS_PIL:
        try:
            with Image.open(path) as im:
                if width == 0:
                    width, height = im.size
                exif = im.getexif() if hasattr(im, "getexif") else None
                if exif:
                    f35 = exif.get(41989)
                    if focal_35 is None and f35 and float(f35) > 1.0:
                        focal_35 = float(f35)
                    fl = exif.get(37386)
                    if focal_mm is None and fl is not None:
                        focal_mm = float(fl)
                    et = exif.get(33434)
                    if exposure_time is None and et is not None and float(et) > 0.0:
                        exposure_time = float(et)
                    fn = exif.get(33437)
                    if f_number is None and fn is not None and float(fn) > 0.0:
                        f_number = float(fn)
                    iv = exif.get(34855)
                    if iso_val is None and iv is not None:
                        if isinstance(iv, (tuple, list)):
                            iv = iv[0] if iv else None
                        if iv and int(iv) > 0:
                            iso_val = int(iv)
                    if not dt_str:
                        dt_raw = exif.get(36867) or exif.get(306)
                        if dt_raw:
                            dt_str = str(dt_raw)
                    ori = exif.get(274)
                    if ori is not None and orient == 1:
                        orient = int(ori)
        except Exception:
            pass

    if focal_35 is None or exposure_time is None or f_number is None:
        tiff_exif = _read_exif_from_tiff_container(path)
        if tiff_exif:
            if focal_35 is None and "focal_35" in tiff_exif and float(tiff_exif["focal_35"]) > 1.0:
                focal_35 = float(tiff_exif["focal_35"])
            if focal_mm is None and "focal_mm" in tiff_exif and float(tiff_exif["focal_mm"]) > 1.0:
                focal_mm = float(tiff_exif["focal_mm"])
            if exposure_time is None and "exposure_time" in tiff_exif and float(tiff_exif["exposure_time"]) > 0.0:
                exposure_time = float(tiff_exif["exposure_time"])
            if f_number is None and "f_number" in tiff_exif and float(tiff_exif["f_number"]) > 0.0:
                f_number = float(tiff_exif["f_number"])
            if iso_val is None and "iso" in tiff_exif and int(tiff_exif["iso"]) > 0:
                iso_val = int(tiff_exif["iso"])
            if not dt_str and "datetime" in tiff_exif:
                dt_str = str(tiff_exif["datetime"])
            if "orient" in tiff_exif and orient == 1:
                orient = int(tiff_exif["orient"])

    if orient == 6:
        result["roll_deg"] = -90.0
    elif orient == 8:
        result["roll_deg"] = 90.0
    elif orient == 3:
        result["roll_deg"] = 180.0

    used_focal = focal_35 if focal_35 is not None else (focal_mm if (focal_mm is not None and focal_mm >= 14.0) else None)
    if used_focal is not None and used_focal > 1.0:
        sensor_w = 36.0 if width >= height or height == 0 else 24.0
        fov_rad = 2.0 * math.atan(sensor_w / (2.0 * used_focal))
        result["fov_h_deg"] = float(np.clip(math.degrees(fov_rad), 10.0, 165.0))
        result["focal_mm"] = float(used_focal)

    result["exposure_time"] = exposure_time
    result["f_number"] = f_number
    result["iso"] = iso_val
    if exposure_time and exposure_time > 0.0 and f_number and f_number > 0.0:
        iso_eff = float(iso_val) if (iso_val and iso_val > 0) else 100.0
        result["ev100"] = float(math.log2((f_number ** 2) / exposure_time) - math.log2(iso_eff / 100.0))

    ts: Optional[float] = None
    if dt_str:
        try:
            import datetime as _dt

            ts = _dt.datetime.strptime(dt_str[:19], "%Y:%m:%d %H:%M:%S").timestamp()
        except Exception:
            ts = None
    if ts is None:
        try:
            ts = float(os.path.getmtime(path))
        except Exception:
            ts = None
    result["timestamp"] = ts
    return result


def group_bracket_sequences(
    paths: List[str],
    frames_per_angle: Optional[int] = None,
) -> List[List[str]]:
    """
    Group a flat list of bracketed photos from a 360° shoot into per-angle bracket groups.

    If `frames_per_angle` (>= 2) is given, chunks deterministically into groups of that size.
    Otherwise, groups by EXIF timestamp gaps (> 3.0s) or EV cycle resets.
    """
    clean = [p for p in paths if p and os.path.isfile(p)]
    if not clean:
        return []
    metas = [(p, read_plate_metadata(p)) for p in clean]
    metas.sort(key=lambda item: (item[1].get("timestamp") or 0.0, item[0]))

    if frames_per_angle is not None and frames_per_angle >= 2:
        n = int(frames_per_angle)
        return [[p for p, _ in metas[i : i + n]] for i in range(0, len(metas), n) if len(metas[i : i + n]) >= 1]

    groups: List[List[str]] = [[metas[0][0]]]
    prev_ts = metas[0][1].get("timestamp")
    prev_ev = metas[0][1].get("ev100")
    ev_dir: Optional[int] = None  # +1 ascending, -1 descending

    for p, m in metas[1:]:
        ts = m.get("timestamp")
        ev = m.get("ev100")
        split = False
        if ts is not None and prev_ts is not None and (ts - prev_ts) > 3.5:
            split = True
        elif ev is not None and prev_ev is not None:
            diff = ev - prev_ev
            if abs(diff) > 0.25:
                cur_dir = 1 if diff > 0 else -1
                if ev_dir is not None and cur_dir != ev_dir and len(groups[-1]) >= 2:
                    split = True
                else:
                    ev_dir = cur_dir
        if split:
            groups.append([p])
            ev_dir = None
        else:
            groups[-1].append(p)
        prev_ts = ts
        prev_ev = ev
    return groups


def detect_circular_fisheye(pixels: np.ndarray) -> bool:
    """Return True if an image exhibits dark circular-fisheye corners with a lit center."""
    if pixels is None or pixels.ndim < 3:
        return False
    ih, iw = pixels.shape[:2]
    if ih < 16 or iw < 16:
        return False
    c_pts = np.array(
        [pixels[2, 2], pixels[2, iw - 3], pixels[ih - 3, 2], pixels[ih - 3, iw - 3]],
        dtype=np.float32,
    )
    c_lum = float(np.max(0.2126 * c_pts[:, 0] + 0.7152 * c_pts[:, 1] + 0.0722 * c_pts[:, 2]))
    mid_p = pixels[ih // 2, iw // 2]
    m_lum = float(0.2126 * mid_p[0] + 0.7152 * mid_p[1] + 0.0722 * mid_p[2])
    return c_lum < 0.015 and m_lum > 0.02


def batch_merge_bracket_groups(
    groups: List[List[str]],
    out_dir: Optional[str] = None,
    colorspace: str = "Auto",
) -> List[str]:
    """Merge each bracket group into a 32-bit float Linear Rec.709 EXR and return output paths."""
    out_paths: List[str] = []
    for idx, grp in enumerate(groups, start=1):
        if not grp:
            continue
        merged = merge_ldr_brackets_to_hdr(grp, colorspace=colorspace)
        base_dir = out_dir or os.path.dirname(grp[0]) or "."
        stem = os.path.splitext(os.path.basename(grp[0]))[0]
        dst = os.path.join(base_dir, f"angle_{idx:02d}_{stem}_merged_hdr.exr")
        written = write_exr_32f(dst, merged, output_colorspace="Linear Rec.709")
        out_paths.append(written)
    return out_paths


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
