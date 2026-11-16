[← Back to Tutorial Hub](../TUTORIAL.md) | [Next: 1. Plate / Sequence Input →](02_plate_input_and_alignment.md)

# Exposure Brackets Setup, Merging & Phone/Camera RAW

Shooting and merging exposure brackets is the best way to capture **true unclamped HDR dynamic range** (sun, sky, and deep shadows) using a standard DSLR, mirrorless camera, or smartphone.

---

## 1. Camera / Phone Bracket Shooting Rules

To prevent ghosting, color shifts, and brightness mismatches during merging:

- **Lock Aperture** (e.g., `f/5.6`–`f/8`): Never change aperture between brackets; it alters depth of field and lens vignetting.
- **Lock ISO**: Use base ISO (`ISO 100` or `ISO 50`).
- **Lock White Balance**: Set a fixed Kelvin or `Daylight (5500K)` preset — **never Auto WB**.
- **Lock Focus**: Switch to Manual Focus (`MF`) or lock `AF/AE`.
- **Vary Shutter Speed Only**:
  - **Overcast / Indoor**: 3 brackets at `-2 EV`, `0 EV`, `+2 EV`
  - **Direct Sun / High Contrast**: 5–7 brackets from `-6 EV` to `+2 EV` in `2 EV` steps (ensure the darkest frame keeps the sun disk unclipped and the brightest frame has clean shadow detail).
- **Multi-Angle 360° Panorama**: Shoot the exact same bracket set at each camera angle around the horizon.

---

## 2. Phone / iPhone RAW (`.dng` / ProRAW, `.heic`) & Camera RAW

HDRI Breaker supports:
- **Phone**: `.dng` (Apple ProRAW / Android RAW), `.heic`, `.heif`
- **Camera RAW**: `.cr2`, `.cr3`, `.nef`, `.nrw`, `.arw`, `.raf`, `.orf`, `.rw2`, `.pef`, `.srw`

> **Enabling full 16-bit linear RAW decoding in Houdini (`hython`)**:  
> Houdini's built-in OpenImageIO does not ship with `libraw` or `libheif`. HDRI Breaker automatically checks for `rawpy`, `pillow-heif`, `dcraw`, `magick`, `darktable-cli`, and Windows WIC. For native 16-bit linear sensor decoding inside Houdini, install once via:
> ```sh
> hython -m pip install rawpy pillow-heif
> ```

---

## 3. Merging Brackets with `Merge Brackets...`

1. In **1. Plate / Sequence Input**, click **Merge Brackets...**.
2. Select the **2+ bracketed files for ONE camera angle**.
3. HDRI Breaker merges them using irradiance weighting into a single unclamped **32-bit float Linear Rec.709 `.exr`** saved next to your source files and places it in **Plate Path**.
4. Keep **Input Mode → Color** on `Auto`.
5. *(For multi-angle 360° shoots)* Repeat **Merge Brackets...** for each angle, then load all the resulting `*.exr` files in **[Section 1: Plate / Sequence Input](02_plate_input_and_alignment.md)**.

---

[← Back to Tutorial Hub](../TUTORIAL.md) | [Next: 1. Plate / Sequence Input →](02_plate_input_and_alignment.md)
