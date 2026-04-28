# Gridfinity Pic-to-Bin — Phone Camera to Gridfinity Bin

## Project Overview

Generate 3D-printable gridfinity bins with custom tool cutouts from **phone camera photos**. Uses a printed ArUco marker template for perspective correction and automatic scale calibration. After preprocessing, SAM2 segmentation and vectorization extract the tool outline; layout packing and a Fusion 360 add-in/script turn that into a parametric bin with pockets, finger slots, and a stacking-lip-compatible base.

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
         ↓  SVG + DXF (inner trace + simplified tolerance outline + finger slot)
Layout Packing (layout_tools.py)
    [DXF reading → rotation + mirror → polygon collision packing → combined DXF]
         ↓
Bin Config Generation (prepare_bin.py)
    [center cutout in bin → JSON config]
         ↓
Fusion 360 Add-In / Script (pic_to_bin_addin/ or pic_to_bin_script/)
    [parametric bin body + ABS-white appearance + pockets + slots + base pads]
         ↓  STL / STEP / PNG preview
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
        trace_export.py              # SVG/DXF export (always-simplified tolerance poly)
        validate_trace.py            # Trace validation
        layout_tools.py              # Layout packing
        prepare_bin.py               # Centering + Fusion config generation
        fusion_install.py            # Installs script AND add-in into Fusion
        pic_to_bin_script/           # Fusion 360 script form
            pic_to_bin.py            # script entry point
            pic_to_bin.manifest      # type: "script"
            _bin_builder.py          # shared build logic (also copied into addin)
        pic_to_bin_addin/            # Fusion 360 add-in form (toolbar button)
            pic_to_bin.py            # add-in entry (registers Solid > Create button)
            pic_to_bin.manifest      # type: "addin"
            resources/pic_to_bin/    # toolbar icons
                16x16.png
                32x32.png
                64x64.png
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
| `trace_tool.py` | SAM2 segmentation + mask cleanup + vectorization |
| `refine_trace.py` | Iterative cleanup refinement |
| `trace_export.py` | SVG/DXF export. Always emits a Douglas-Peucker–simplified TOLERANCE polygon (the one Fusion cuts) so the inner high-res potrace polys never reach the CAD step |
| `layout_tools.py` | Layout packing (rotation + collision-pack, packing bbox = inner+tolerance, slot allowed to overhang) |
| `prepare_bin.py` | Centers the combined cutout (inner + tolerance + slot) in the bin, then writes the Fusion JSON config |
| `pic_to_bin_script/_bin_builder.py` | Shared Fusion build code — sketch consolidation, named timeline groups, ABS (White) appearance, STL/STEP/PNG export |
| `pic_to_bin_script/pic_to_bin.py` | Thin script entry — picks `bin_config.json` and calls `_bin_builder.build_bin()` |
| `pic_to_bin_addin/pic_to_bin.py` | Add-in entry — registers a "Gridfinity Pic-to-Bin" button in Solid > Create |
| `fusion_install.py` | `pic-to-bin-fusion install` copies the script into `…/API/Scripts/pic_to_bin/` AND the add-in into `…/API/AddIns/pic_to_bin/`, copying `_bin_builder.py` into both |

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

## Bin Sizing & Cutout Geometry

- **Pocket floor at z=1 mm** above the bin floor (1 mm of solid material below the pocket). The bin auto-sizes to `ceil((tool_height + 1mm) / 7mm)` gridfinity height units — i.e., the smallest count that fits the tool.
- **Deck rises to half the tool height** above the pocket floor. Upper half of the tool stands proud for finger access; lower half is buried in the pocket.
- **Finger slot is centered along the principal axis** of the tool (PCA-derived). The candidate-position sort uses the global axis center, not the narrow-band midpoint, so asymmetric tools (screwdriver, wrench) don't end up with the slot pushed toward one end.
- **Cutout is centered in the bin** during `prepare_bin`. The combined (inner + tolerance + slot) bbox is shifted so its center coincides with the bin center; this redistributes the slack from rounding up to whole gridfinity units evenly on all four sides. Runs before thin-wall snapping and slot-clip-to-boundary so those operate on final positions.
- **Tolerance polygon is always simplified.** `trace_export` emits the TOLERANCE layer for any offset value (including 0): `_offset_polygons` (no-op at 0, expand on positive, shrink on negative) → Douglas-Peucker simplify (ε=0.3 mm) → corner rounding. This keeps Fusion sketches under ~50 points per tool instead of hundreds.

## Fusion 360 Build Geometry

The Fusion side (`_bin_builder.py`) builds the bin in a new document with these phases, each wrapped in a named timeline group:

1. **Bin Body** — single rectangular extrude. Body gets the Fusion appearance "ABS (White)" applied immediately.
2. **Stacking Lip** (optional, toggle via `--stacking false`) — solid block + 4 mm fillet + base-profile inverse cutout + 0.6 mm top recess.
3. **Deck** — single rectangular cut inside the wall+inset perimeter, lowered to expose the upper half of the tallest tool.
4. **Tool Pockets** — one sketch per tool containing all that tool's tolerance polys, single Cut extrude per tool. Cut starts above the lip so edge-reaching profiles cut the lip too.
5. **Finger Slots** — one sketch per tool containing all slot polys, single Cut extrude per tool, same depth as the pocket.
6. **Base Pads** — single sketch with all wide-pad rounded rects + single Join extrude, then chamfer; same pattern for narrow posts. For an N×M grid this is 2 sketches + 2 extrudes + 2 chamfers regardless of N×M.

Both entry points (`pic_to_bin_script/pic_to_bin.py` and `pic_to_bin_addin/pic_to_bin.py`) call `importlib.reload(_bin_builder)` on each invocation so edits to `_bin_builder.py` land on the next click without restarting Fusion. The file dialog defaults to the user's Desktop (Windows `%USERPROFILE%\Desktop` / macOS `~/Desktop`) when no `<project>/generated/bin_config.json` is found.

## Usage

```bash
# 1. Generate and print the template
generate-phone-template --paper-size letter --output template.pdf

# 2. Place tool on template, take phone photo

# 3. Run the full pipeline
pic-to-bin photo.jpg --tool-height 17
pic-to-bin photo.jpg --tool-height 17 --paper-size legal
pic-to-bin a.jpg b.jpg --tool-height 0=17 --tool-height 1=14
pic-to-bin photo.jpg --tool-height 17 --stacking false   # shorter bin, no lip

# Individual steps:
preprocess-phone photo.jpg --paper-size letter
trace-tool rectified.png --dpi 150
layout-tools tool1.dxf tool2.dxf
prepare-bin generated/combined_layout.dxf --tool-height 17.0

# Install Fusion 360 script + add-in (idempotent)
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
- `Pillow`, `pillow-heif`

Note: `cv2.aruco` is included in standard `opencv-python` (4.13+). No need for `opencv-contrib-python`.

## Technical Notes

- **ArUco dictionary**: `DICT_4X4_50` — 4x4 bit grid, 50 unique markers. Simple and robust for large printed markers.
- **Scale accuracy**: Sub-pixel ArUco corners give ~0.1px precision. Over 160mm marker span, scale error is <0.5mm.
- **Homography safety**: RANSAC rejects outlier corners from wrinkled paper or partially occluded markers. 8 markers give up to 32 point pairs for a 4-DOF homography.
- **White border fill**: Warped image uses white border (`borderValue=255,255,255`) so areas outside the template appear as white background, which SAM2 handles correctly.
- **Effective DPI**: Computed from the homography inverse — how many pixels per mm at the template center. Typically 100-250 for phone photos.
- **Tolerance default = 0**: The pocket matches the trace exactly. Positive `--tolerance` expands the pocket past the trace (clearance fit); negative shrinks it (interference fit). The tolerance polygon is always Douglas-Peucker simplified at ε=0.3 mm, so the cut stays light on points regardless of offset value.

## Relationship to gridfinity-scan-to-bin

This is a standalone project. Core pipeline files (`trace_tool.py`, `trace_export.py`, `layout_tools.py`, `prepare_bin.py`, etc.) were originally copied from gridfinity-scan-to-bin with import paths updated from `gridfinity_scan_to_bin.*` to `pic_to_bin.*`, and have since diverged (centering, tolerance handling, slot placement, pocket-floor sizing). The phone preprocessing modules (`phone_template.py`, `phone_preprocess.py`) are new. The Fusion side has been refactored from a single script into a shared `_bin_builder.py` plus parallel script and add-in entry points.
