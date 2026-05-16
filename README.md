# Gridfinity Pic-to-Bin

Generate 3D-printable gridfinity bins with custom tool cutouts from phone camera photos.

A printed ArUco marker template handles perspective correction and automatic scale calibration. The rest of the pipeline — SAM2 segmentation, vectorization, layout packing, and Fusion 360 bin generation — runs automatically.

## Workflow overview

```
1. Print the ArUco template (one-time setup)
2. Place tool on template, photograph from above
3. Run pic-to-bin → get a Fusion 360 config JSON
4. Click the "Gridfinity Pic-to-Bin" button in Fusion → get a parametric STL/STEP
```

---

## Installation

```bash
pip install gridfinity-pic-to-bin
```

Dependencies (installed automatically): `ultralytics` (SAM2), `opencv-python`, `numpy`, `ezdxf`, `potracer`, `pyclipper`, `matplotlib`, `Pillow`, `pillow-heif`.

---

## Step 1: Print the template

Generate and print a template for your paper size. **Print at exactly 100% scale — no fit-to-page.**

```bash
generate-phone-template --paper-size letter --output template.pdf
```

```
Options:
  --paper-size {a4,letter,legal}   Paper size (default: a4)
  --output PATH                    Output PDF path (default: phone_template_<size>.pdf)
```

The template places 8 ArUco markers (IDs 0–7) — 4 corners and 4 edge midpoints — around a dotted placement zone. The markers are 20 mm squares with ~20 mm margins from the paper edge.

**Placement zone sizes by paper:**

| Paper   | Placement zone (W × H) |
|---------|------------------------|
| A4      | 130 × 217 mm           |
| Letter  | 136 × 199 mm           |
| Legal   | 136 × 275 mm           |

---

## Step 2: Take the photo

1. Lay the printed template on a flat surface.
2. Place the tool inside the dotted placement zone.
3. Photograph from above. Moderate angles are fine — perspective is corrected automatically.
4. All 8 markers should be visible (3 minimum, 8 ideal).
5. Measure the tool's **depth** with calipers — you'll need it for `--tool-height`.

**Lighting tips:** Use diffuse overhead light. Avoid harsh shadows across the markers. Do not cover markers with the tool.

**Format:** JPEG, PNG, or HEIC/HEIF (iPhone) are all supported.

---

## Step 3: Run the pipeline

### Single tool

```bash
pic-to-bin photo.jpg --tool-height 17
```

### Multiple tools (one photo each)

```bash
pic-to-bin a.jpg b.jpg --tool-height 0=17 --tool-height 1=14
```

Tool indices correspond to image order. Each tool's DXF is traced separately and packed into one bin.

### Specifying paper size

```bash
pic-to-bin photo.jpg --tool-height 17 --paper-size a4
```

The paper size must match what you printed. Default is `letter`.

### Shallow drawer (no stacking lip)

```bash
pic-to-bin photo.jpg --tool-height 17 --stacking false
```

Drops the 4.4 mm stacking lip for shorter bins in shallow drawers. Pocket depth is unchanged.

### All `pic-to-bin` options

```
positional:
  images                        Photo files to process (default: all PNG/JPG in cwd)

required:
  --tool-height VALUE           Tool depth in mm. Use INDEX=VALUE per tool
                                (e.g. --tool-height 0=17 --tool-height 1=14)

optional:
  --paper-size {a4,letter,legal}  Template paper size (default: legal)
  --tolerance MM                  Extra clearance on top of a 2 mm baseline
                                  (default: 0 = 2 mm physical clearance).
                                  Positive = looser fit, negative = tighter,
                                  -2 = exact-trace match.
  --axial-tolerance MM            Extra clearance only along the tool's
                                  principal axis (default: 'auto'; 2 mm
                                  floor + taper-proportional bonus).
                                  Compensates for SAM2 length under-
                                  detection.
  --phone-height MM               Camera height above template, mm (default: 480).
                                  Drives the parallax-compensation scale-down.
  --gap MM                        Minimum gap between tools in layout, mm (default: 3.0)
  --bin-margin MM                 Extra clearance from tool extent to bin wall (default: 0)
  --min-units-x N                 Minimum X grid size in units (default: 1)
  --min-units-y N                 Minimum Y grid size in units (default: 1)
  --min-units-z N                 Minimum Z grid size in height units (default: 1).
                                  Floor on the auto height; ignored when
                                  --height-units is set.
  --max-units N                   Max gridfinity grid size per axis (default: 7)
  --height-units N                Force bin height in gridfinity units (default: auto)
  --stacking BOOL                 Generate stacking lip (default: true). Set
                                  false for a shorter bin without the lip.
  --slots BOOL                    Generate finger-access slots (default: true)
  --output-dir DIR                Output directory (default: generated/)
  --straighten-threshold DEG      Max degrees to auto-straighten trace (default: 45, 0=off)
  --max-refine-iterations N       SAM2 cleanup iterations (default: 5)
  --max-concavity-depth MM        Max acceptable concavity loss, mm (default: 3.0)
  --mask-erode MM                 Post-SAM mask erosion (default: 0). Use 0.3-0.5
                                  only if your photos have a clear shadow halo.
  --sam-model WEIGHTS             SAM2 model file (default: sam2.1_l.pt)
  --skip-trace                    Skip tracing, reuse existing DXFs in generated/
```

### Bin sizing logic

- The bin auto-sizes to the smallest gridfinity unit count that fits the tool: `ceil((tool_height + 1mm) / 7mm)` height units.
- The pocket floor sits 1 mm above the bin floor.
- The deck rises to half the tool's height — the upper half of the tool stands proud for finger access; the lower half is buried in the pocket.
- The combined cutout (pocket + finger slot) is centered in the bin floor; slack from rounding up to whole gridfinity units is distributed evenly on all four sides.

### Output files

```
generated/
  <stem>/
    <stem>_rectified.png       Perspective-corrected image (scanner equivalent)
    <stem>_trace.dxf           Tool outline DXF (inner + tolerance + finger slot)
    <stem>_trace.svg           SVG preview of the trace
  combined_layout.dxf          All tools packed into a bin footprint
  layout_preview.png           Screen-viewable layout preview (matplotlib, 150 DPI)
  layout_actual_size.pdf       1:1 scale fit-test drawing (PDF page = bin footprint)
  layout_actual_size.svg       1:1 scale fit-test drawing (SVG width/height in mm)
  bin_config.json              Fusion 360 input config
```

The `layout_actual_size.pdf` and `.svg` files are sized to the bin's exact
mm dimensions — print at "Actual size" / 100% scale (NOT "Fit to page") and
lay your real tool on top to verify the fit before committing to a 3D print.

After Fusion runs:

```
generated/
  gridfinity_bin.stl
  gridfinity_bin.step
  gridfinity_bin_preview.png  Viewport screenshot of the finished bin
```

---

## Web app (browser frontend)

A FastAPI + Lit web wrapper exposes the same pipeline through a browser. Multi-user
ready: per-job UUID directories, GPU semaphore around SAM2 so concurrent submissions
queue rather than fight over the GPU, SSE-streamed progress.

```bash
pip install -e ".[web]"
pic-to-bin-web --port 8000
```

Open http://localhost:8000, drag in a photo, fill in the tool height, watch
the step tracker, review the layout preview (with print-at-actual-size PDF /
SVG downloads to test fit), then click Proceed to generate the bin config.

The browser back button navigates between screens (form → progress → preview
→ downloads). The form fields all have an `(i)` info button next to their
label that opens a modal with a multi-paragraph explanation.

To replace the default `esm.sh` Lit import with a vendored local copy:

```bash
python -m pic_to_bin.web.vendor_lit
```

---

## Running individual steps

You can run each stage of the pipeline independently.

### Preprocess a photo

Detect markers, correct perspective, and save the rectified image:

```bash
preprocess-phone photo.jpg --paper-size letter --output-dir generated/photo
```

```
positional:
  image                         Phone photo file

optional:
  --paper-size {a4,letter,legal}  Template paper size (default: a4)
  --output-dir DIR                Output directory (default: generated/<stem>)
```

Outputs `<stem>_rectified.png` at the computed effective DPI (typically 100–250).

### Trace a rectified image

Run SAM2 segmentation + vectorization on a rectified image:

```bash
trace-tool rectified.png --dpi 150
```

### Pack tool DXFs into a layout

```bash
layout-tools tool1.dxf tool2.dxf --gap 3 --max-units 5
```

### Generate the Fusion 360 config

```bash
prepare-bin generated/combined_layout.dxf --tool-height 17
```

---

## Fusion 360 integration

`pic-to-bin` ships in two flavors for Fusion 360 — a **toolbar add-in** (recommended) and a classic **script**. One install command sets up both:

```bash
pic-to-bin-fusion install
```

This copies the add-in to `…/API/AddIns/pic_to_bin/` and the script to `…/API/Scripts/pic_to_bin/`, sharing the build code (`_bin_builder.py`) between them.

### Add-in (recommended) — toolbar button

1. Open Fusion 360.
2. Press **Shift+S → Add-Ins tab**.
3. Select **pic_to_bin → Run** (toggle **Run on Startup** to keep the button available every session).
4. In a Design workspace, the **Solid > Create** panel now contains a **Gridfinity Pic-to-Bin** button.
5. Click the button. The script auto-loads `<project>/generated/bin_config.json` if it exists; otherwise a file dialog opens defaulting to your Desktop.
6. Bin gets built, exported as STL + STEP, and a viewport screenshot is saved alongside `bin_config.json`.

### Script form (alternate)

If you prefer the classic Scripts dialog:

1. Press **Shift+S → Scripts tab**.
2. Select **pic_to_bin → Run**.

The behavior is identical to the add-in button.

### What gets built

The bin is generated in a fresh document with these timeline groups for easy navigation:

- **Bin Body** — outer rectangular block. Appearance: ABS (White).
- **Stacking Lip** (if enabled) — solid block + corner fillets + base-profile mating cutout + 0.6 mm top recess.
- **Deck** — recessed surface around the pocket.
- **Tool Pockets** — one cut per tool.
- **Finger Slots** — one cut per tool, same floor as the pocket.
- **Base Pads** — gridfinity baseplate-mating geometry, one Join extrude for all wide pads, one for the narrow posts, plus the two chamfers.

### Reinstalling and reload

Re-running `pic-to-bin-fusion install` overwrites both folders. The script and add-in entry points reload `_bin_builder.py` from disk on every invocation, so most code changes land on the next button click without restarting Fusion. Only changes to the entry-point files themselves (`pic_to_bin_script/pic_to_bin.py` or `pic_to_bin_addin/pic_to_bin.py`) require a Stop/Run on the add-in or a Fusion restart.

### Uninstall

```bash
pic-to-bin-fusion uninstall
```

Removes both the script and the add-in.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `MarkerDetectionError: No markers detected` | Template not visible or too blurry | Ensure all markers are in frame; hold camera steadier |
| `MarkerDetectionError: Only N markers (need ≥3)` | Markers obscured or overexposed | Improve lighting; don't cover markers with tool |
| `ScaleInconsistencyError: H/V scales differ >5%` | Template not printed at 100% | Reprint with fit-to-page disabled |
| `WARNING: Low effective DPI (<100)` | Camera too far away | Hold phone closer; use higher resolution mode |
| Tools don't fit in grid | Tools too large for `--max-units` | Increase `--max-units` |
| Fusion freezes building pockets | Stale cached `_bin_builder` after editing | The reload is already wired in — just click the button again. If still stuck, restart Fusion. |
| Pocket fits too loose | Default `--tolerance 0` produces 2 mm physical clearance + ≥2 mm at each tip | Lower with `--tolerance -0.5` and/or `--axial-tolerance 1.0` |
| Pocket fits too tight at the tool's tips only | SAM2 under-detected the tapered ends | Increase `--axial-tolerance` (default 'auto', 2 mm floor) |
| Pocket fits too tight everywhere | Trace itself is short (shadow halo, parallax, mask erosion) | First try `--tolerance 1` (= 3 mm physical). If still tight, check `--phone-height` matches your shooting distance |

### Common photo issues

- **Markers partially cut off**: Keep all 8 markers visible. At 3 minimum the pipeline will run but accuracy drops.
- **Blurry markers**: Tap to focus on the template before shooting; avoid camera shake.
- **Shadows on markers**: Use diffuse or overhead lighting, not a side lamp.
- **HEIC images (iPhone)**: Supported natively — the pipeline converts automatically via `pillow-heif`.

---

## Running tests

```bash
python -m pytest tests/ -v
```
