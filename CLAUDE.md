# Gridfinity Pic-to-Bin — Phone Camera to Gridfinity Bin

## Project Overview

Generate 3D-printable gridfinity bins with custom tool cutouts from **phone camera photos**. Uses a printed ArUco marker template for perspective correction and automatic scale calibration. After preprocessing, the same SAM2 segmentation and vectorization pipeline from gridfinity-scan-to-bin handles the tool outline extraction.

## Architecture — Phone-Camera Pipeline

```
1. Print ArUco template (one-time)
2. Place tool on template, take phone photo
         ↓
Phone Preprocessing (phone_preprocess.py)
    [ArUco detection → homography → perspective warp → scale calibration → crop]
         ↓  Rectified image + effective DPI
Trace Generation (trace_tool.py, refine_trace.py)
    [SAM2 segmentation → iterative cleanup → potrace vectorization → SVG/DXF]
         ↓  SVG + DXF (inner trace + tolerance outline + finger slot)
Layout Packing (layout_tools.py)
    [DXF reading → rotation + mirror → polygon collision packing → combined DXF]
         ↓
Bin Config Generation (prepare_bin.py)
    [JSON config → Fusion 360 script → parametric bin + pockets → STL/STEP]
```

## Directory Layout

```
gridfinity-pic-to-bin/
    pyproject.toml
    pic_to_bin/                      # Python package
        __init__.py
        phone_template.py            # ArUco template PDF generation
        phone_preprocess.py          # Marker detection, homography, warping
        pipeline.py                  # Main orchestrator (pic-to-bin CLI)
        trace_tool.py                # SAM2 segmentation + cleanup + vectorize
        refine_trace.py              # Iterative cleanup refinement
        trace_export.py              # SVG/DXF export
        validate_trace.py            # Trace validation
        layout_tools.py              # Layout packing
        prepare_bin.py               # Fusion 360 config generation
        fusion_install.py            # Fusion 360 script installer
        fusion_bin_script/           # Fusion 360 script
            fusion_bin_script.py
            fusion_bin_script.manifest
    tests/
        test_phone_template.py
        test_phone_preprocess.py
    generated/                       # Pipeline output (auto-created)
```

## Key Files

| File | Purpose |
|------|---------|
| `phone_template.py` | Generate printable ArUco marker template PDF. CLI: `generate-phone-template` |
| `phone_preprocess.py` | Detect ArUco markers, compute homography, warp to frontal view. CLI: `preprocess-phone` |
| `pipeline.py` | Main orchestrator. CLI: `pic-to-bin <images> --tool-height VALUE` |
| `trace_tool.py` | SAM2 segmentation + mask cleanup + vectorization (copied from scan-to-bin) |
| `refine_trace.py` | Iterative cleanup refinement (copied from scan-to-bin) |
| `trace_export.py` | SVG/DXF export with dual outlines + finger slot (copied from scan-to-bin) |
| `layout_tools.py` | Layout packing (copied from scan-to-bin) |
| `prepare_bin.py` | Fusion 360 JSON config (copied from scan-to-bin) |

## Template Design

- 8 ArUco markers from `DICT_4X4_50` (IDs 0-7)
- 4 corner markers (IDs 0-3) + 4 edge midpoint markers (IDs 4-7) for redundancy
- Marker size: 20mm square
- ~20mm margin from paper edge
- Supports: **A4** (210x297mm), **US Letter** (215.9x279.4mm), **US Legal** (215.9x355.6mm)
- Placement zone (tool area): ~130-136mm wide, ~199-276mm tall depending on paper

## Phone Preprocessing Details

### ArUco Detection
- Uses `cv2.aruco.ArucoDetector` with `DICT_4X4_50`
- Sub-pixel corner refinement (`CORNER_REFINE_SUBPIX`)
- Adaptive thresholding tuned for variable phone lighting
- Filters to template IDs 0-7 only

### Homography Computation
- Maps each marker's 4 pixel corners to known mm positions (up to 32 point pairs for 8 markers)
- `cv2.findHomography()` with RANSAC for outlier rejection
- Computes effective DPI from marker spacing
- Cross-validates horizontal vs vertical scale (warn >2%, error >5%)
- Requires at least 3 detected markers

### Perspective Correction
- `cv2.warpPerspective()` warps phone photo to frontal view
- Output at computed effective DPI (typically 100-250 depending on phone distance)
- Crops to placement zone (inside marker ring)
- Result is scanner-equivalent: flat, known scale, white background

## Usage

```bash
# 1. Generate and print the template
generate-phone-template --paper-size letter --output template.pdf

# 2. Place tool on template, take phone photo

# 3. Run the full pipeline
pic-to-bin photo.jpg --tool-height 17
pic-to-bin photo.jpg --tool-height 17 --paper-size legal
pic-to-bin a.jpg b.jpg --tool-height 0=17 --tool-height 1=14

# Individual steps:
preprocess-phone photo.jpg --paper-size letter
trace-tool rectified.png --dpi 150
layout-tools tool1.dxf tool2.dxf
prepare-bin generated/combined_layout.dxf --tool-height 17.0

# Install Fusion 360 script
pic-to-bin-fusion install

# Run tests
python -m pytest tests/ -v
```

## Error Handling

| Scenario | Exception | Guidance |
|----------|-----------|----------|
| <3 markers detected | `MarkerDetectionError` | Ensure template is visible, well-lit, not too far |
| H/V scale mismatch >5% | `ScaleInconsistencyError` | Reprint template at 100% (no fit-to-page) |
| Effective DPI <100 | Warning | Hold camera closer or use higher resolution |
| No markers at all | `MarkerDetectionError` | Check template is in frame and not too blurry |

## Photo Best Practices

- Print template at **100% scale** (no fit-to-page) on white paper
- Use **good overhead lighting** — diffuse light reduces shadows
- Photograph from above; moderate angles are OK (perspective is corrected)
- Keep all 8 markers visible if possible (3 minimum, 8 ideal)
- Tool should fit within the dotted placement zone
- Tool depth measured separately with calipers (manual `--tool-height` input)

## Dependencies

Same as gridfinity-scan-to-bin — all installed via `pip install gridfinity-pic-to-bin`:
- `ultralytics` (SAM2), `opencv-python` (ArUco + image processing), `numpy`
- `potracer`, `svgpathtools`, `ezdxf`, `pyclipper`, `matplotlib`

Note: `cv2.aruco` is included in standard `opencv-python` (4.13+). No need for `opencv-contrib-python`.

## Technical Notes

- **ArUco dictionary**: `DICT_4X4_50` — 4x4 bit grid, 50 unique markers. Simple and robust for large printed markers.
- **Scale accuracy**: Sub-pixel ArUco corners give ~0.1px precision. Over 160mm marker span, scale error is <0.5mm — within the 0.8mm default tolerance.
- **Homography safety**: RANSAC rejects outlier corners from wrinkled paper or partially occluded markers. 8 markers give up to 32 point pairs for a 4-DOF homography.
- **White border fill**: Warped image uses white border (`borderValue=255,255,255`) so areas outside the template appear as white background, which SAM2 handles correctly.
- **Effective DPI**: Computed from the homography inverse — how many pixels per mm at the template center. Typically 100-250 for phone photos.

## Relationship to gridfinity-scan-to-bin

This is a standalone project. Core pipeline files (trace_tool.py, trace_export.py, layout_tools.py, prepare_bin.py, etc.) were copied from gridfinity-scan-to-bin with import paths updated from `gridfinity_scan_to_bin.*` to `pic_to_bin.*`. The phone preprocessing modules (phone_template.py, phone_preprocess.py) are new.
