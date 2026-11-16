# HDRI Breaker — Complete Workflow Tutorial

This guide walks through the full **HDRI Breaker** workflow in Houdini 20.0 / 20.5 / 21.0+ Solaris (`/stage`), from preparing camera/phone shots and exposure brackets to synthesizing a 32-bit float Lat/Long `.exr` and wiring a Solaris lookdev rig.

---

## 1. Preparing Your Images

HDRI Breaker accepts:
- **HDR formats**: `.exr`, `.hdr`, `.pic`, `.rat`
- **LDR formats**: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.tga`
- **Phone / Camera RAW & HEIC**: `.dng` (iPhone ProRAW / Android RAW), `.heic`, `.heif`, `.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`

> **Tip for RAW / HEIC in Houdini (`hython`)**:  
> Houdini's built-in OpenImageIO does not ship with `libraw` or `libheif`. HDRI Breaker automatically checks for `rawpy` / `pillow-heif`, `dcraw` / `magick` / `darktable-cli`, or Windows WIC codecs. For full 16-bit linear decoding of `.dng` / `.cr2` / `.nef` / `.arw` / `.heic`, you can install them once into Houdini Python:
> ```sh
> hython -m pip install rawpy pillow-heif
> ```

---

## 2. Step 1 — Merge Exposure Brackets (Optional, Recommended for Sun/Practicals)

If you shot **bracketed exposures** (e.g., `-2 EV`, `0 EV`, `+2 EV`) at the same camera angle:

1. In **1. Plate / Sequence Input**, click **Merge Brackets...**.
2. Select the 2+ bracketed files for that angle (`.jpg`, `.png`, `.tif`, `.dng`, etc.).
3. HDRI Breaker merges them using Debevec/Robertson irradiance weighting into a single unclamped **32-bit float Linear Rec.709 `.exr`** and places it in the **Plate Path** field.
4. If you have multiple camera angles around the room/outdoor horizon, merge each angle's brackets first, then add the resulting `.exr` files via **Add Plates...**.

---

## 3. Step 2 — Load Plates or Sequences

Choose one of three ways to load your shots:
- **Multi-Angle 360° Set**: Click **Add Plates...** (or **drag and drop** multiple image files directly onto the preview viewport).
- **Single Plate**: Click **Browse...** next to **Plate Path**, or drop a single file onto the viewport.
- **Image Sequence**: Set **Plate Path** to a tokenized path like `D:/plates/shot.$F4.exr` and switch **Input Mode → Mode** to **Sequence Master (Stabilized)** or **Per-Frame Sequence**.

### Managing the Multi-Position List
- **Checkboxes**: Uncheck a plate row to temporarily exclude it without deleting it.
- **`↑` / `↓` buttons**: Reorder plates in the blend stack.
- **Color**: Leave on `Auto` (automatically applies sRGB→Linear for LDR/sRGB and keeps EXR/RAW linear).

---

## 4. Step 3 — Set Initial Angles, FOV & Analyze

When multi-angle plates are first added, their `Az°` and `El°` start at `0.0`:

1. **Spread Angles**: Click **Auto 360°** to distribute all listed plates evenly around the 360° horizon *(or double-click `Az°`, `El°`, or `FOV°` cells in the table to type exact angles)*.
2. **Match Lens FOV (if needed)**: If all photos were shot with the same lens but one EXIF tag read differently, select the reference plate row and click **Unify FOV**.
3. **Scene Camera (optional)**: Click **Link Scene Cam...** to pull `FOV°`, `Az°`, and `El°` directly from a `/stage` USD Camera or `/obj` Camera.
4. Click the blue **Analyze Plate(s) & Generate Preview** button to build the initial 2:1 Lat/Long canvas and extract lighting metrics in section **2. Extracted Lighting Data**.

---

## 5. Step 4 — Align Seams & Match Exposure

1. **Automatic Alignment**: Click **Auto-Align (Spherical)** to refine overlapping `Az°` / `El°` positions via coarse-to-fine spherical correlation.
2. **Manual Fine-Tuning (`E`)**: Toggle **Move / Align Plates (`E`)** on the preview toolbar:
   - **Click** a plate to select it.
   - **Drag** to move `Az° / El°`.
   - **Drag cyan corner handles** (or `Shift + Wheel`) to scale `FOV°`.
   - **`Shift + Drag`** (or drag the orange top handle) to adjust **Roll°**.
   - **`Ctrl + Drag`** to adjust radial **Lens Warp (`k1`)**.
   - Use **Mask Area** in the Plate Transform Bar to exclude unwanted objects (e.g. a hand or tripod leg) from a specific plate.
3. **Match Brightness**: Click **Equalize EV** to balance exposure differences across overlapping plates and eliminate brightness seams.

---

## 6. Step 5 — Fill Sky/Ground, White Balance & 32-bit Brush Cleanup

### Sky & Ground Inpainting
- Click **Pick Sky Area** and drag a box over clean sky in the preview to synthesize/fill missing zenith sky.
- Click **Pick Ground Area** and drag a box over clean ground to fill missing nadir ground.
- Tune **Sky Blend**, **Ground Blend**, **Sky Blur**, and **Ground Blur** in section **3. 32-bit Lat/Long Synthesis Tuning**.
- Existing boxes can be dragged to move or resized by their edges directly in the viewport.

### White Balance Calibration
- **Pick Neutral WB**: Drag a box over a neutral gray/white surface (concrete, gray card, cloud) to auto-calibrate 3-channel WB gains and exposure.
- **Macbeth WB...**: Load a separate ColorChecker reference shot and click any neutral gray patch.
- **Reset WB**: Resets WB gains back to `(1.0, 1.0, 1.0)` *(undoable via `Ctrl+Z`)*.

### Interactive 32-bit Float Brush (`B`)
Toggle **Brush Paint (`B`)** to reveal the contextual brush toolbar:
- **Heal / Inpaint**: Clean spots, tripods, or people from the surrounding ring.
- **Clone Stamp**: `Ctrl + Click` to pick a source point, then drag to paint (360° seam-wrap aware).
- **Blur / Smooth**, **Burn (Dim HDR)**, **Dodge (Boost HDR)**, **Desaturate**: Smooth seams or tame/boost highlights.
- **Shortcuts**: `[` / `]` (or `Ctrl + Left-Drag`) adjusts Brush Size; `Ctrl + Right-Drag` adjusts Strength; `Shift + Left-Drag` adjusts Softness.

### Apply / Collapse (`Ctrl+Enter`)
Once your plate alignment, sky/ground fills, and brush cleanup look solid, click **Apply / Collapse** (`Ctrl+Enter`):
- Bakes stitching, inpainting, and brush strokes into a fast 32-bit canvas so global sliders (**Exposure**, **CCT**, **Saturation**, **Blur**, **Yaw**) respond instantaneously.
- You can continue layering new **Sky/Ground/Light** picks or **Brush** strokes on top of the collapsed canvas.
- **Undo (`Ctrl+Z`)** undoes edits made *after* collapsing without undoing your **Apply / Collapse**; use **Unbake (`N`)** whenever you explicitly want to step back through saved bake checkpoints.

---

## 7. Step 6 — Lookdev Tuning & Global Undo

- In **3. 32-bit Lat/Long Synthesis Tuning**, dial in **Output Resolution** (up to `8192 x 4096`), **Out Space** (`Linear Rec.709` or `ACEScg`), **Exposure (EV)**, **Target CCT (K)**, **Saturation**, **Hotspot Boost**, **HDRI Yaw°**, and **Horizon Shift°** (or click **Auto-Level**).
- **Global Undo (`Ctrl+Z`)**: Undoes parameter changes, region picks, plate edits, WB changes, and brush strokes. The footer **Undo** button displays the exact action that will be undone next.
- **Viewport Resizing**: Drag the horizontal or vertical splitters to resize the viewport, or click **`⤢`** next to the zoom controls to expand/restore the viewport.
- **Reset All**: Clears all plates, edits, and Apply/Collapse history to start fresh (with confirmation).

---

## 8. Step 7 — Export 32-bit EXR & Wire Solaris (`/stage`)

1. **(Optional) Extract Area / Sun Lights**:
   - Toggle **Pick Light Area** (`Rectangle` or `Circle`) and drag over the sun or bright practicals.
   - Keep **Inpaint Light Out** checked to remove the light from the dome texture so it isn't double-counted.
2. In **4. Export 32-bit EXR & Solaris (`/stage`) Wiring**:
   - Set **Output EXR** (e.g. `$HIP/tex/hdri_maker.exr`).
   - Select your **Render Engine** (`Karma (MaterialX)`, `Redshift`, or `Arnold`) and **Ground Shader** mode.
   - Click **Generate 32-bit Float EXR** (or **Export Sequence ($F4)** for sequences).
   - Click **Create / Update Solaris Dome Light**, and optionally **Create / Update Area Light (/stage)**, **Create / Update Sun Light (/stage)**, **Update Lookdev / Ground (/stage)**, or **Export USD Rig (.usda)**.

---

## 9. Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl+Z` | Global Undo (parameters, regions, WB, brush strokes) |
| `Ctrl+Enter` | Apply / Collapse edits into fast 32-bit canvas |
| `B` | Toggle 32-bit Brush Paint mode |
| `E` | Toggle Move / Align Plates mode |
| `F` or `H` | Fit / Reset preview zoom and pan |
| `[` / `]` | Decrease / Increase brush size |
| `Del` | Remove selected plate row(s) from Multi-Position table |
| `Middle / Right-Drag` | Pan preview when zoomed |
| `Wheel` | Zoom preview (`Shift+Wheel` scales selected plate FOV in Edit mode) |
