# Gridfinity Pic-to-Bin

Generate 3D-printable gridfinity bins with custom tool cutouts from phone camera photos.

A printed ArUco marker template handles perspective correction and automatic scale calibration. The rest of the pipeline — SAM2 segmentation, vectorization, layout packing, and Fusion 360 config generation — runs automatically.

## Workflow overview

```
1. Print the ArUco template (one-time setup)
2. Place tool on template, photograph from above
3. Run pic-to-bin → get a Fusion 360 config JSON
4. Run the Fusion 360 script → get a parametric STL/STEP
```

---

## Installation

```bash
pip install gridfinity-pic-to-bin
```

Dependencies (installed automatically): `ultralytics` (SAM2), `opencv-python`, `numpy`, `ezdxf`, `potracer`, `pyclipper`, `matplotlib`, `pillow-heif`.

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

### All `pic-to-bin` options

```
positional:
  images                        Photo files to process (default: all PNG/JPG in cwd)

required:
  --tool-height VALUE           Tool depth in mm. Use INDEX=VALUE per tool
                                (e.g. --tool-height 0=17 --tool-height 1=14)

optional:
  --paper-size {a4,letter,legal}  Template paper size (default: letter)
  --tolerance MM                  Tolerance offset added around trace, mm (default: 1.5)
  --gap MM                        Minimum gap between tools in layout, mm (default: 3.0)
  --max-units N                   Max gridfinity grid size per axis (default: 7)
  --height-units N                Force bin height in gridfinity units (default: auto)
  --output-dir DIR                Output directory (default: generated/)
  --straighten-threshold DEG      Max degrees to auto-straighten trace (default: 45, 0=off)
  --max-refine-iterations N       SAM2 cleanup iterations (default: 5)
  --max-concavity-depth MM        Max acceptable concavity loss, mm (default: 3.0)
  --sam-model WEIGHTS             SAM2 model file (default: sam2.1_l.pt)
  --skip-trace                    Skip tracing, reuse existing DXFs in generated/
```

### Output files

```
generated/
  <stem>/
    <stem>_rectified.png      Perspective-corrected image (scanner equivalent)
    <stem>_trace.dxf          Tool outline DXF (inner + tolerance + finger slot)
    <stem>_trace.svg          SVG preview of the trace
  combined_layout.dxf         All tools packed into a bin footprint
  combined_layout_preview.png Layout preview image
  bin_config.json             Fusion 360 input config
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

Install the bundled script into Fusion 360's Scripts directory:

```bash
pic-to-bin-fusion install
```

Then in Fusion 360: **Utilities → Scripts and Add-Ins → Scripts → pic_to_bin → Run**.

Point it at `generated/bin_config.json`. The script generates the parametric bin body with pockets, tolerance offsets, and finger slots.

To uninstall:

```bash
pic-to-bin-fusion uninstall
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `MarkerDetectionError: No markers detected` | Template not visible or too blurry | Ensure all markers are in frame; hold camera steadier |
| `MarkerDetectionError: Only N markers (need ≥3)` | Markers obscured or overexposed | Improve lighting; don't cover markers with tool |
| `ScaleInconsistencyError: H/V scales differ >5%` | Template not printed at 100% | Reprint with fit-to-page disabled |
| `WARNING: Low effective DPI (<100)` | Camera too far away | Hold phone closer; use higher resolution mode |
| Tools don't fit in grid | Tools too large for `--max-units` | Increase `--max-units` or use a smaller tolerance |

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
