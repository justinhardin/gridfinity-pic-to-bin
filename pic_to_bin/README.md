# `pic_to_bin/` — main Python package

Holds every module for the photo → printable Gridfinity bin pipeline.
Console scripts (`pic-to-bin`, `preprocess-phone`, `trace-tool`,
`layout-tools`, `prepare-bin`, `generate-phone-template`,
`pic-to-bin-fusion`, `pic-to-bin-web`) are wired in `pyproject.toml` to
entry points in these modules.

| File | Purpose |
|------|---------|
| `phone_template.py` | Generate the printable ArUco template PDF. |
| `phone_preprocess.py` | Detect markers, compute homography, warp the photo to a flat scanner-equivalent view. |
| `pipeline.py` | CLI `pic-to-bin` and library `run_pipeline()` — orchestrates the full pipeline. |
| `trace_tool.py` | SAM2 segmentation + mask cleanup + potrace vectorization. |
| `refine_trace.py` | Iterative cleanup refinement around `trace_tool`. |
| `trace_export.py` | SVG/DXF export, tolerance polygon, axial-stretch logic. |
| `validate_trace.py` | Trace-quality sanity checks. |
| `layout_tools.py` | Polygon-pack tools into a Gridfinity bin; emits combined DXF and a 1:1 PDF/SVG fit-test. |
| `prepare_bin.py` | Center the cutout, write the `bin_config.json` consumed by Fusion. |
| `fusion_install.py` | `pic-to-bin-fusion install` — copies the add-in and script into Fusion's user dirs. |

Subpackages:

- `web/` — FastAPI + Lit browser app (see `web/README.md`)
- `pic_to_bin_script/` — Fusion 360 Script entry (see `pic_to_bin_script/README.md`)
- `pic_to_bin_addin/` — Fusion 360 Add-In entry (see `pic_to_bin_addin/README.md`)

Runtime output goes to `../generated/` (CLI) or `../web_jobs/<uuid>/`
(web app); both are gitignored.
