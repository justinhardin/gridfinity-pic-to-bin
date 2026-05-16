# Gridfinity Pic-to-Bin — Phone Camera to Gridfinity Bin

## Current state (2026-05-01 EOD)

Active branch: **`web-app`** (HEAD `e4782f1`). Pushed to `origin/web-app` —
the master branch is unchanged from before the web work started. To resume:

```bash
git checkout web-app
pip install -e ".[web]"          # pulls fastapi, uvicorn[standard], sse-starlette, python-multipart
pic-to-bin-web --port 8000       # http://localhost:8000
```

**What's working end-to-end on `web-app`:**
- CLI pipeline (`pic-to-bin`) plus Python-callable `pipeline.run_pipeline()`
  with progress callbacks and a `stop_after="layout"` checkpoint.
- FastAPI web wrapper (`pic_to_bin/web/`) with multi-user job queue, SSE
  progress streaming, two-phase layout-then-bin flow with cheap re-do.
- Lit-based frontend with drag-drop dropzone, per-image tool-height inputs,
  info-modal field documentation, browser-back navigation between screens.
- 1:1-scale fit-test PDF + SVG outputs alongside the screen-PNG preview so
  users can print and verify fit before 3D printing.
- 2 mm uniform tolerance baseline + 'auto' axial-tolerance default
  (2 mm floor + taper-proportional bonus, PCA-stretched along the
  tool's principal axis to compensate for SAM2 tip under-detection).

**Last session's open thread:** none — everything committed and tested
(>100 passing). Security hardening complete for public hosting:
- 30 MiB / 8 photo / 120 MiB upload limits + client-side guards
- "check with LLM" feature is now opt-in only (`--enable-llm` / `PIC_TO_BIN_ENABLE_LLM`); disabled by default with clear warnings
- Security headers (CSP, X-Frame-Options, etc.) + hardened error handler
- Server-side param validation + defensive checks in JobManager
- Comprehensive NGINX/Apache guidance + updated web/README.md

The app is now safe to put behind a public NGINX/Apache reverse proxy on 127.0.0.1:8000. LLM costs and classic DoS vectors are neutralised while preserving the full tuned pipeline quality.

## Project Overview

Generate 3D-printable gridfinity bins with custom tool cutouts from **phone camera photos**. Uses a printed ArUco marker template for perspective correction and automatic scale calibration. After preprocessing, SAM2 segmentation and vectorization extract the tool outline; layout packing and a Fusion 360 add-in/script turn that into a parametric bin with pockets, finger slots, and a stacking-lip-compatible base. A FastAPI + Lit web wrapper exposes the same pipeline through a browser.

## Architecture — Phone-Camera Pipeline

```
1. Print ArUco template (one-time)
2. Place tool on template, take phone photo
         ↓
Phone Preprocessing (phone_preprocess.py)
    [ArUco detection → homography → perspective warp → scale calibration → crop]
         ↓  Rectified image + effective DPI
Trace Generation (trace_tool.py, refine_trace.py, trace_export.py)
    [SAM2 segmentation → iterative cleanup → potrace → uniform offset →
     axial PCA stretch → Douglas-Peucker → SVG/DXF]
         ↓  SVG + DXF (inner trace + simplified tolerance outline + finger slot)
Layout Packing (layout_tools.py)
    [DXF reading → rotation + mirror → polygon collision packing → combined DXF
     → layout_preview.png + 1:1-scale layout_actual_size.pdf / .svg for fit-testing]
         ↓
Bin Config Generation (prepare_bin.py)
    [center cutout in bin → JSON config]
         ↓
EITHER:  Fusion 360 Add-In (pic_to_bin_addin/)
    [parametric bin body + ABS-white appearance + pockets + slots + base pads]
         ↓  STL / STEP / PNG preview
OR:     Web app (pic_to_bin/web/) — FastAPI + Lit
    [drag-drop photos → SSE-streamed progress → preview → proceed/redo →
     downloads (PNG, PDF/SVG fit-test, DXF, JSON for the Fusion add-in)]
```

## Directory Layout

```
gridfinity-pic-to-bin/
    pyproject.toml                   # [web] optional dep group; pic-to-bin-web script
    pic_to_bin/                      # Python package
        __init__.py
        phone_template.py            # ArUco template PDF generation
        phone_preprocess.py          # Marker detection, homography, warping
        pipeline.py                  # CLI main() + library run_pipeline()
        trace_tool.py                # SAM2 segmentation + cleanup + vectorize
        refine_trace.py              # Iterative cleanup refinement
        trace_export.py              # SVG/DXF export, axial-stretch tolerance
        validate_trace.py            # Trace validation
        layout_tools.py              # Layout packing + fit-test PDF/SVG generation
        prepare_bin.py               # Centering + Fusion config generation
        fusion_install.py            # Installs the add-in into Fusion's AddIns dir
        web/                         # FastAPI + Lit web wrapper
            __init__.py
            jobs.py                  # JobManager: UUID registry, GPU semaphore, SSE
            server.py                # FastAPI routes + uvicorn cli() + Fusion ZIP builder
            vendor_lit.py            # `python -m … vendor_lit` to vendor Lit locally
            static/
                home.html            # Public landing page at /
                index.html           # Importmap-based Lit loader at /app
                app.js               # PicApp / Form / Progress / Preview / Downloads
                styles.css
        pic_to_bin_addin/            # Fusion 360 add-in (only Fusion entry point)
            pic_to_bin.py            # add-in entry (registers Solid > Create button)
            _bin_builder.py          # build logic — sketches, timeline groups, exports
            pic_to_bin.manifest      # type: "addin"
            resources/pic_to_bin/    # toolbar icons
                16x16.png
                32x32.png
                64x64.png
    tests/
        conftest.py
        test_phone_template.py
        test_phone_preprocess.py
        test_web_jobs.py             # JobManager smoke tests with mocked run_pipeline
    generated/                       # CLI pipeline output (auto-created, gitignored)
    web_jobs/                        # Web app per-job UUID dirs (gitignored)
```

## Key Files

| File | Purpose |
|------|---------|
| `phone_template.py` | Generate printable ArUco marker template PDF. CLI: `generate-phone-template` |
| `phone_preprocess.py` | Detect ArUco markers, compute homography, warp to frontal view. CLI: `preprocess-phone` |
| `pipeline.py` | CLI `pic-to-bin` + library-callable `run_pipeline()`; defines `ProgressEvent`, `TOLERANCE_BASELINE_MM`, `DEFAULT_PHONE_HEIGHT_MM` |
| `trace_tool.py` | SAM2 segmentation + mask cleanup + vectorization |
| `refine_trace.py` | Iterative cleanup refinement |
| `trace_export.py` | SVG/DXF export. Always emits a Douglas-Peucker–simplified TOLERANCE polygon (the one Fusion cuts). `_axial_stretch_polygons` adds extra clearance along the tool's principal axis only |
| `layout_tools.py` | Layout packing + `generate_preview` (PNG) + `generate_fit_test_drawing` (PDF/SVG at 1:1 mm scale for printing) |
| `prepare_bin.py` | Centers the combined cutout in the bin, writes Fusion JSON config |
| `web/jobs.py` | `JobManager`: UUID registry, ThreadPoolExecutor, GPU semaphore around SAM2, async SSE event fan-out, TTL sweep |
| `web/server.py` | FastAPI routes + `pic-to-bin-web` uvicorn launcher; whitelisted artifact serving |
| `web/static/app.js` | Lit components: `pic-app` (root, owns modal + history), `pic-form`, `pic-progress`, `pic-preview` (with fit-test card), `pic-downloads`. `FIELD_INFO` map drives the (i) info modals |
| `web/vendor_lit.py` | `python -m pic_to_bin.web.vendor_lit` downloads `lit-all.min.js` into `static/` and rewrites the `index.html` import map |
| `pic_to_bin_addin/pic_to_bin.py` | Add-in entry — registers a "Gridfinity Pic-to-Bin" button in Solid > Create |
| `pic_to_bin_addin/_bin_builder.py` | Fusion build code — sketch consolidation, named timeline groups, ABS (White) appearance, STL/STEP/PNG export |
| `fusion_install.py` | `pic-to-bin-fusion install` copies the add-in into `…/API/AddIns/pic_to_bin/`. Uninstall also cleans up the legacy `…/API/Scripts/pic_to_bin/` folder for users who installed pre-consolidation. |

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

The Fusion side (`pic_to_bin_addin/_bin_builder.py`) builds the bin in a new document with these phases, each wrapped in a named timeline group:

1. **Bin Body** — single rectangular extrude. Body gets the Fusion appearance "ABS (White)" applied immediately.
2. **Stacking Lip** (optional, toggle via `--stacking false`) — solid block + 4 mm fillet + base-profile inverse cutout + 0.6 mm top recess.
3. **Deck** — single rectangular cut inside the wall+inset perimeter, lowered to expose the upper half of the tallest tool.
4. **Tool Pockets** — one sketch per tool containing all that tool's tolerance polys, single Cut extrude per tool. Cut starts above the lip so edge-reaching profiles cut the lip too.
5. **Finger Slots** — one sketch per tool containing all slot polys, single Cut extrude per tool, same depth as the pocket.
6. **Base Pads** — single sketch with all wide-pad rounded rects + single Join extrude, then chamfer; same pattern for narrow posts. For an N×M grid this is 2 sketches + 2 extrudes + 2 chamfers regardless of N×M.

The add-in entry point (`pic_to_bin_addin/pic_to_bin.py`) calls `importlib.reload(_bin_builder)` on every click so edits to `_bin_builder.py` land on the next button press without restarting Fusion. The file dialog defaults to the user's Desktop (Windows `%USERPROFILE%\Desktop` / macOS `~/Desktop`) when no `<project>/generated/bin_config.json` is found.

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

# Web app (multi-user; FastAPI + Lit frontend)
pip install -e ".[web]"
pic-to-bin-web --port 8000           # opens at http://localhost:8000
python -m pic_to_bin.web.vendor_lit  # optional: vendor Lit locally

# Run tests
python -m pytest tests/ -v
```

## Web App

`pic_to_bin/web/` is a FastAPI wrapper around `run_pipeline()` (the
library-callable form of the CLI's `main()`). Designed for multi-user from
day one:

- **Per-job UUID dirs** under `web_jobs/<uuid>/` so concurrent submissions
  cannot collide. Files survive server restart so the layout-preview / re-do
  flow still works after a reboot. A background sweep deletes terminal jobs
  older than `--job-ttl-hours` (default 24h).
- **GPU semaphore** (`threading.Semaphore(1)` in `JobManager`) serializes the
  SAM2 step across concurrent jobs so the second user's submission queues
  rather than OOMing the GPU. SAM2 weights load once at process startup and
  stay resident.
- **Two-phase pipeline**: submit runs Phase A (preprocess + trace + layout) and
  stops with `stop_after="layout"`; the user reviews `layout_preview.png` and
  clicks Proceed (Phase B = `prepare_bin`) or Re-do. Layout-only re-runs use
  `skip_trace=True` to re-use cached per-tool DXFs (cheap; seconds). Re-runs
  that change trace-affecting params re-trace from the photos.
- **Server-Sent Events** stream `ProgressEvent`s from worker threads to the
  browser via `loop.call_soon_threadsafe`. The job's event log is replayed on
  late connections.
- **Frontend stack**: Lit components served as static files. Default
  `index.html` import map points at `esm.sh`; running
  `python -m pic_to_bin.web.vendor_lit` downloads `lit-all.min.js` into
  `static/` and rewrites the import map for a fully self-contained deploy.
- **Browser back navigates between screens** instead of leaving the site.
  `PicApp` captures each screen change via `history.pushState` in `updated()`
  and restores from `popstate`. Initial state is `replaceState` so back from
  the form still exits.
- **Preview cache key**: artifact URLs use `?v=<artifactKey>` where
  `artifactKey` is bumped exactly once when fresh artifacts arrive
  (`_onLayoutReady` and `_onComplete`). Without this, embedding `Date.now()`
  in the template literal caused the preview image to refetch on every
  Lit re-render — visible flicker.
- **Field info modals**: `FIELD_INFO` map in `app.js` has a `title`/`hint`/
  `body[]` entry for every form field plus `tool_height` and `fit_test`. A
  bubbling `show-info` event bubbles up to `PicApp` which renders the modal,
  so any screen can open it.
- **Future hooks designed in**: OAuth (FastAPI dependency `get_current_user`
  default-stubbed to anonymous), LLM-driven re-do (`/jobs/{id}/redo` body has
  `mode: "params" | "llm"` slot, only `"params"` implemented), printables.com
  search (form has `part_number` / `description` slots, no endpoint yet).

Endpoints: `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/events` (SSE),
`POST /jobs/{id}/proceed`, `POST /jobs/{id}/redo`,
`GET /jobs/{id}/artifacts/{name}` — whitelisted artifacts:
`layout_preview.png`, `layout_actual_size.pdf`, `layout_actual_size.svg`,
`combined_layout.dxf`, `bin_config.json`.

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

Core (always installed):
- `ultralytics` (SAM2), `opencv-python` (ArUco + image processing), `numpy`
- `potracer`, `svgpathtools`, `ezdxf`, `pyclipper`, `matplotlib`
- `Pillow`, `pillow-heif`

Web (`pip install -e ".[web]"`):
- `fastapi`, `uvicorn[standard]`, `python-multipart`, `sse-starlette`

Dev (`pip install -e ".[dev]"`):
- `pytest`, `httpx`

Note: `cv2.aruco` is included in standard `opencv-python` (4.13+). No need for `opencv-contrib-python`.

## Technical Notes

- **ArUco dictionary**: `DICT_4X4_50` — 4x4 bit grid, 50 unique markers. Simple and robust for large printed markers.
- **Scale accuracy**: Sub-pixel ArUco corners give ~0.1px precision. Over 160mm marker span, scale error is <0.5mm.
- **Homography safety**: RANSAC rejects outlier corners from wrinkled paper or partially occluded markers. 8 markers give up to 32 point pairs for a 4-DOF homography.
- **White border fill**: Warped image uses white border (`borderValue=255,255,255`) so areas outside the template appear as white background, which SAM2 handles correctly.
- **Effective DPI**: Computed from the homography inverse — how many pixels per mm at the template center. Typically 100-250 for phone photos.
- **Tolerance baseline = 2 mm**: The pipeline silently adds 2 mm of clearance to whatever the user passes via `--tolerance` (constant `pipeline.py:TOLERANCE_BASELINE_MM`). The CLI/form default is `0`, which produces a 2 mm physical clearance — calibrated for typical FDM tolerances. `--tolerance -2` recovers an exact-trace match; more negative produces an interference fit. The tolerance polygon is always Douglas-Peucker simplified at ε=0.3 mm.
- **Mask erosion default = 0 mm**: Uniform mask erosion disproportionately shrinks tapered tool tips (a 0.3 mm erosion can lose meaningful coverage at a screwdriver tip while barely affecting a wide handle). Default is 0; users can re-enable when a photo has a clearly fat shadow halo via `--mask-erode 0.3`.
- **Axial tolerance (default 'auto')**: After the uniform offset, the tolerance polygon is stretched along the tool's PCA principal axis so each end gets `--axial-tolerance` mm of additional clearance (perpendicular extent unchanged). Compensates for SAM2's tendency to under-detect tool length — present even on rounded/square ends, larger on tapered tips. `'auto'` uses `2.0 + 0.014 × axial_length × taper` (taper from per-bin width analysis along the principal axis), so square-ended tools get the 2 mm floor (matching the perpendicular baseline) and sharply tapered tools get more. Implemented in `trace_export._axial_stretch_polygons` via SVD on the polygon point cloud + a linear-ramp scale in the rotated frame. Set to 0 for fully uniform tolerance.

## Relationship to gridfinity-scan-to-bin

This is a standalone project. Core pipeline files (`trace_tool.py`, `trace_export.py`, `layout_tools.py`, `prepare_bin.py`, etc.) were originally copied from gridfinity-scan-to-bin with import paths updated from `gridfinity_scan_to_bin.*` to `pic_to_bin.*`, and have since diverged (centering, tolerance handling, slot placement, pocket-floor sizing). The phone preprocessing modules (`phone_template.py`, `phone_preprocess.py`) are new. The Fusion side has been refactored from a single script into a shared `_bin_builder.py` plus parallel script and add-in entry points.
