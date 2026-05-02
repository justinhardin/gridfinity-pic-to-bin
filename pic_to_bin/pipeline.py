#!/usr/bin/env python3
"""
Pic-to-Bin: One-command gridfinity bin generation from phone camera photos.

Orchestrates the full pipeline:
  0. Preprocess phone photos (ArUco marker detection, perspective correction)
  1. Trace tool outlines from rectified images  (trace_tool / refine_trace)
  2. Pack traces into an optimal bin layout      (layout_tools)
  3. Generate Fusion 360 config JSON             (prepare_bin)

Minimal usage:
    pic-to-bin photo.jpg --tool-height 17

Full usage:
    pic-to-bin photo1.jpg photo2.jpg --tool-height 0=17 --tool-height 1=14 --paper-size letter
    (default paper size is legal)

Library usage (web app, tests, notebooks):
    from pic_to_bin.pipeline import run_pipeline, ProgressEvent
    result = run_pipeline(
        image_paths=[Path("photo.jpg")],
        tool_heights=17.0,
        output_dir=Path("web_jobs/abc123"),
        progress_cb=lambda ev: print(ev),
    )
"""

import argparse
import os
import shutil
import stat
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Literal, Optional

from pic_to_bin.phone_preprocess import (
    preprocess_phone_image,
    MarkerDetectionError,
    ScaleInconsistencyError,
)
from pic_to_bin.trace_tool import _collect_images
from pic_to_bin.refine_trace import refine_trace
from pic_to_bin.layout_tools import layout_tools, GridSizeError
from pic_to_bin.prepare_bin import prepare_bin


DEFAULT_PHONE_HEIGHT_MM = 482.0

# Baseline tolerance offset (mm) added to whatever the user requests before
# the trace is offset. Empirically, prints come out too tight without it;
# 2 mm of baseline clearance produces a comfortable clearance fit on typical
# FDM prints. The user-facing default is 0 (= 2 mm physical clearance);
# `--tolerance -2` recovers an exact-trace match, more negative produces an
# interference fit.
TOLERANCE_BASELINE_MM = 2.0


# ---------------------------------------------------------------------------
# Progress events (for web UI / programmatic callers)
# ---------------------------------------------------------------------------

@dataclass
class ProgressEvent:
    """Structured progress signal emitted by run_pipeline().

    step is one of: "preprocess", "trace", "layout", "bin_config", "done",
    or "error" for per-image failures that the pipeline recovered from.
    """
    step: str
    message: str = ""
    fraction: float = 0.0
    image_index: Optional[int] = None
    image_total: Optional[int] = None
    image_name: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


ProgressCallback = Callable[[ProgressEvent], None]
StopAfter = Literal["layout", "all"]


# ---------------------------------------------------------------------------
# Programmatic entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    image_paths,
    tool_heights,
    *,
    output_dir,
    paper_size: str = "legal",
    tolerance: float = 0.0,
    axial_tolerance: float = 1.0,
    phone_height: float = DEFAULT_PHONE_HEIGHT_MM,
    gap: float = 3.0,
    bin_margin: float = 0.0,
    max_units: int = 7,
    min_units: int = 1,
    height_units: Optional[int] = None,
    stacking: bool = True,
    slots: bool = True,
    straighten_threshold: float = 45.0,
    max_refine_iterations: int = 5,
    max_concavity_depth: float = 3.0,
    mask_erode: float = 0.0,
    sam_model: str = "sam2.1_l.pt",
    skip_trace: bool = False,
    stop_after: StopAfter = "all",
    progress_cb: Optional[ProgressCallback] = None,
) -> dict:
    """Run the pic-to-bin pipeline non-interactively.

    Args mirror the CLI flags. ``stop_after="layout"`` stops after the
    layout-preview step (used by the web app's preview-then-proceed flow).
    Re-running with ``skip_trace=True`` re-uses cached per-tool DXFs.

    Returns a dict::

        {
          "dxf_paths":      [...],   # per-tool trace DXFs
          "combined_dxf":   "...",   # combined_layout.dxf
          "layout_preview": "...",   # layout_preview.png
          "layout_result":  {...},   # full dict from layout_tools()
          "grid_units_x":   int,
          "grid_units_y":   int,
          "bin_config":     "..." | None,  # bin_config.json (None if stop_after=="layout")
        }

    Raises ``GridSizeError`` (tools don't fit in max_units), ``RuntimeError``
    (no DXFs produced), and unexpected exceptions from layout/prepare_bin.
    Per-image preprocessing/trace errors are reported via the progress
    callback as ``step="error"`` events and the pipeline continues with the
    surviving images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [Path(p) for p in image_paths]

    def _emit(event: ProgressEvent) -> None:
        if progress_cb is not None:
            progress_cb(event)

    # --- Steps 1-2: Preprocess + Trace -------------------------------------
    dxf_paths: list[str] = []
    if skip_trace:
        print("\n-- Skipping trace (--skip-trace), looking for existing DXFs --")
        _emit(ProgressEvent(step="trace", message="Reusing cached traces", fraction=0.5))
        for img in image_paths:
            tool_dir = output_dir / img.stem
            # Phone preprocessing renames the rectified image to "{stem}_rectified.png",
            # so the trace_tool output is "{stem}_rectified_trace.dxf". Glob to match
            # both the phone-pipeline name and the legacy "{stem}_trace.dxf".
            candidates = sorted(tool_dir.glob("*_trace.dxf"))
            if candidates:
                dxf_paths.append(str(candidates[0]))
                print(f"  Found: {candidates[0]}")
            else:
                print(f"  WARNING: no *_trace.dxf in {tool_dir}, skipping {img.name}")
    else:
        print("\n" + "=" * 60)
        print("STEP 1/3: Phone preprocessing + Tracing tool outlines")
        print("=" * 60)
        n = len(image_paths)
        for idx, img in enumerate(image_paths):
            tool_output_dir = output_dir / img.stem
            try:
                # Phone preprocessing
                print(f"\n--- Preprocessing {img.name} ---")
                _emit(ProgressEvent(
                    step="preprocess",
                    message=f"Preprocessing {img.name}",
                    image_index=idx, image_total=n, image_name=img.name,
                    fraction=idx / max(n, 1),
                ))
                pp = preprocess_phone_image(
                    str(img),
                    paper_size=paper_size,
                    output_dir=str(tool_output_dir),
                )
                rectified_img = Path(pp["rectified_image_path"])
                dpi = round(pp["effective_dpi"])
                print(f"  Phone mode: {pp['markers_detected']}/8 markers, "
                      f"effective DPI: {dpi}")

                # Trace
                print(f"\n--- Tracing {img.name} ---")
                _emit(ProgressEvent(
                    step="trace",
                    message=f"Tracing {img.name}",
                    image_index=idx, image_total=n, image_name=img.name,
                    fraction=(idx + 0.5) / max(n, 1),
                ))
                tool_height_mm = _resolve_tool_height(tool_heights, idx)
                result = refine_trace(
                    image_path=str(rectified_img),
                    dpi=dpi,
                    tolerance_mm=tolerance + TOLERANCE_BASELINE_MM,
                    axial_tolerance_mm=axial_tolerance,
                    straighten_threshold=straighten_threshold,
                    output_dir=str(tool_output_dir),
                    max_iterations=max_refine_iterations,
                    max_concavity_depth_mm=max_concavity_depth,
                    sam_model=sam_model,
                    mask_erode_mm=mask_erode,
                    tool_height_mm=tool_height_mm,
                    phone_height_mm=phone_height,
                    finger_slots=slots,
                )
                dxf_paths.append(result["dxf_path"])
                iters = result.get("refinement_iterations", 1)
                conv = result.get("refinement_converged", True)
                if iters > 1:
                    status = "converged" if conv else "max iterations"
                    print(f"  Refined in {iters} iterations ({status})")

            except MarkerDetectionError as e:
                msg = f"ERROR preprocessing {img.name}: {e}"
                print(f"\n{msg}\n")
                _emit(ProgressEvent(
                    step="error", message=msg,
                    image_index=idx, image_total=n, image_name=img.name,
                ))
            except ScaleInconsistencyError as e:
                msg = f"ERROR (scale) {img.name}: {e}"
                print(f"\n{msg}\n")
                _emit(ProgressEvent(
                    step="error", message=msg,
                    image_index=idx, image_total=n, image_name=img.name,
                ))
            except Exception as e:
                msg = f"ERROR processing {img.name}: {e}"
                print(f"\n{msg}\n")
                _emit(ProgressEvent(
                    step="error", message=msg,
                    image_index=idx, image_total=n, image_name=img.name,
                ))

    if not dxf_paths:
        raise RuntimeError(
            "No DXF traces produced. Check that photos contain a visible, "
            "well-lit ArUco template and that the template was printed at "
            "100% scale."
        )

    # --- Step 3: Layout packing -------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2/3: Packing layout")
    print("=" * 60)
    _emit(ProgressEvent(
        step="layout",
        message=f"Packing {len(dxf_paths)} tool(s) into bin grid",
        fraction=0.0,
    ))
    layout_result = layout_tools(
        dxf_paths=dxf_paths,
        gap_mm=gap,
        max_units=max_units,
        min_units=min_units,
        bin_margin_mm=bin_margin,
        output_dir=str(output_dir),
    )
    _emit(ProgressEvent(
        step="layout",
        message=(
            f"Layout: {layout_result['grid_units_x']}x"
            f"{layout_result['grid_units_y']} gridfinity units"
        ),
        fraction=1.0,
    ))

    result = {
        "dxf_paths": dxf_paths,
        "combined_dxf": layout_result["combined_dxf_path"],
        "layout_preview": layout_result["preview_path"],
        "layout_result": layout_result,
        "grid_units_x": layout_result["grid_units_x"],
        "grid_units_y": layout_result["grid_units_y"],
        "bin_config": None,
    }

    if stop_after == "layout":
        return result

    # --- Step 4: Bin config ------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3/3: Generating bin config")
    print("=" * 60)
    _emit(ProgressEvent(step="bin_config", message="Generating bin config", fraction=0.5))
    config_path = prepare_bin(
        dxf_path=layout_result["combined_dxf_path"],
        tool_heights=tool_heights,
        height_units=height_units,
        stacking_lip=stacking,
        output_path=str(output_dir / "bin_config.json"),
    )
    result["bin_config"] = config_path
    _emit(ProgressEvent(step="done", message="Done", fraction=1.0))
    return result


# ---------------------------------------------------------------------------
# Existing-output helpers (CLI-only; web caller manages its own job dirs)
# ---------------------------------------------------------------------------

def check_existing_output(output_dir: Path, skip_trace: bool) -> bool:
    """Check if generated output exists and prompt user to confirm deletion."""
    if skip_trace:
        return True
    if not output_dir.exists():
        return True
    contents = list(output_dir.iterdir())
    if not contents:
        return True

    print(f"Output directory '{output_dir}' already contains "
          f"{len(contents)} item(s):")
    for item in sorted(contents)[:10]:
        label = "dir " if item.is_dir() else "file"
        print(f"  [{label}] {item.name}")
    if len(contents) > 10:
        print(f"  ... and {len(contents) - 10} more")

    try:
        answer = input("\nDelete existing output and continue? [y/N] ").strip().lower()
    except EOFError:
        answer = ""

    if answer != "y":
        print("Cancelled.")
        return False

    clear_generated_dir(output_dir)
    return True


def _rmtree_onerror(func, path, exc_info):
    """Handle permission errors on Windows by clearing read-only flag."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clear_generated_dir(output_dir: Path) -> None:
    """Remove all contents of the output directory."""
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item, onexc=_rmtree_onerror)
        else:
            item.unlink()
    print(f"Cleared {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a gridfinity bin config from phone camera photos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  pic-to-bin photo.jpg --tool-height 17
  pic-to-bin a.jpg b.jpg --tool-height 0=17 --tool-height 1=14
  pic-to-bin photo.jpg --tool-height 17 --paper-size letter
""",
    )

    parser.add_argument(
        "images", nargs="*", default=None,
        help="Phone photo files to process (default: all PNG/JPG in cwd)")
    parser.add_argument(
        "--tool-height", action="append", dest="tool_heights", required=True,
        help="Tool height in mm (required). Use INDEX=VALUE for per-tool "
             "(e.g. --tool-height 0=17 --tool-height 1=14)")
    parser.add_argument(
        "--paper-size", choices=["a4", "letter", "legal"], default="legal",
        help="Template paper size used for photos (default: legal)")
    parser.add_argument(
        "--tolerance", type=float, default=0.0,
        help=f"Extra tolerance in mm relative to the standard fit "
             f"(default: 0). The pipeline always adds {TOLERANCE_BASELINE_MM} "
             f"mm of baseline clearance so the printed pocket fits typical "
             f"FDM tolerances; --tolerance is added on top of that. Positive "
             f"= looser fit, negative = tighter, "
             f"--tolerance -{TOLERANCE_BASELINE_MM} = exact trace match.")
    parser.add_argument(
        "--axial-tolerance", type=float, default=1.0,
        help="Extra clearance (mm) along the tool's principal axis only "
             "(default: 1.0). Each end of the tool gets pushed outward by "
             "this amount, leaving the perpendicular extent unchanged. "
             "Compensates for SAM2 mask under-detection at tapered tool "
             "tips. Set to 0 for fully uniform tolerance.")
    parser.add_argument(
        "--phone-height", type=float, default=None,
        help="Phone camera height above the template in mm (default: 482). "
             "Used to compensate parallax: a tool sitting above the paper "
             "appears larger in the photo than it really is, by a factor of "
             "phone_height / (phone_height - tool_height/2). Lower values "
             "apply more compensation; 0 disables compensation.")
    parser.add_argument(
        "--gap", type=float, default=3.0,
        help="Minimum gap between tools in mm (default: 3.0)")
    parser.add_argument(
        "--bin-margin", type=float, default=0.0,
        help="Extra clearance in mm between the tool extent and the bin "
             "boundary, applied before snap-to-grid (default: 0). With a "
             "non-zero --tolerance plus the natural slack from rounding the "
             "bin size up to a whole gridfinity unit, no extra padding is "
             "usually needed. Set this >0 to force the bin one unit larger "
             "when the tool gets within bin_margin_mm of the wall.")
    parser.add_argument(
        "--max-units", type=int, default=7,
        help="Maximum grid size in gridfinity units per axis (default: 7)")
    parser.add_argument(
        "--min-units", type=int, default=1,
        help="Minimum grid size in gridfinity units per axis (default: 1). "
             "Forces the bin to at least min_units x min_units even if the "
             "tools would fit in a smaller grid. Useful to match an existing "
             "drawer slot or other bins in the same set.")
    parser.add_argument(
        "--height-units", type=int, default=None,
        help="Force bin height in gridfinity units (default: auto)")
    parser.add_argument(
        "--stacking", type=_parse_bool, default=True,
        metavar="true|false",
        help="Generate stacking lip (default: true). Set false to remove the "
             "lip and reduce overall bin height for shallow drawers.")
    parser.add_argument(
        "--slots", type=_parse_bool, default=True,
        metavar="true|false",
        help="Generate finger-access slots in the pocket (default: true). "
             "Set false to omit slots entirely — useful for tools that don't "
             "need a finger pull or where the slot intrudes awkwardly.")
    parser.add_argument(
        "--output-dir", type=str, default="generated",
        help="Output directory (default: generated/)")
    parser.add_argument(
        "--straighten-threshold", type=float, default=45.0,
        help="Max degrees to auto-straighten (default: 45). 0 to disable.")
    parser.add_argument(
        "--max-refine-iterations", type=int, default=5,
        help="Maximum cleanup refinement iterations (default: 5)")
    parser.add_argument(
        "--max-concavity-depth", type=float, default=3.0,
        help="Maximum acceptable concavity depth loss in mm (default: 3.0)")
    parser.add_argument(
        "--mask-erode", type=float, default=0.0,
        help="Post-SAM mask erosion in mm to counter shadow halos (default: 0). "
             "Uniform erosion disproportionately shrinks thin/tapered tool "
             "tips, so leave at 0 unless your photo has a strong shadow halo "
             "that bleeds into the mask. 0.3-0.5 mm is a reasonable starting "
             "value when needed.")
    parser.add_argument(
        "--sam-model", type=str, default="sam2.1_l.pt",
        help="SAM2 model weights (default: sam2.1_l.pt)")
    parser.add_argument(
        "--skip-trace", action="store_true",
        help="Skip tracing, reuse existing DXFs in generated/")

    args = parser.parse_args()

    # Confirm parallax-compensation default if --phone-height was omitted.
    if args.phone_height is None:
        response = input(
            f"You haven't specified --phone-height. "
            f"Is the default of {DEFAULT_PHONE_HEIGHT_MM:.0f} mm ok? [y/N] "
        )
        if response.strip().lower() != "y":
            print("Aborted. Re-run with --phone-height <mm> to specify.")
            sys.exit(1)
        args.phone_height = DEFAULT_PHONE_HEIGHT_MM

    output_dir = Path(args.output_dir)

    if not check_existing_output(output_dir, args.skip_trace):
        sys.exit(0)

    image_sources = args.images if args.images else ["all_images"]
    image_paths = _collect_images(image_sources)
    print(f"Found {len(image_paths)} image(s): "
          + ", ".join(p.name for p in image_paths))

    tool_heights = _parse_tool_height_args(args.tool_heights)

    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        image_paths=image_paths,
        tool_heights=tool_heights,
        output_dir=output_dir,
        paper_size=args.paper_size,
        tolerance=args.tolerance,
        axial_tolerance=args.axial_tolerance,
        phone_height=args.phone_height,
        gap=args.gap,
        bin_margin=args.bin_margin,
        max_units=args.max_units,
        min_units=args.min_units,
        height_units=args.height_units,
        stacking=args.stacking,
        slots=args.slots,
        straighten_threshold=args.straighten_threshold,
        max_refine_iterations=args.max_refine_iterations,
        max_concavity_depth=args.max_concavity_depth,
        mask_erode=args.mask_erode,
        sam_model=args.sam_model,
        skip_trace=args.skip_trace,
    )

    try:
        result = run_pipeline(**kwargs)
    except GridSizeError as e:
        required = max(e.required_x, e.required_y)
        print(f"\nTools don't fit in a {args.max_units}x{args.max_units} grid. "
              f"Smallest layout found: {e.required_x}x{e.required_y} units "
              f"({e.required_x * 42}x{e.required_y * 42}mm).")
        try:
            answer = input(f"Proceed with up to {required}x{required}? [y/N] "
                           ).strip().lower()
        except EOFError:
            answer = ""
        if answer != "y":
            print("Cancelled.")
            sys.exit(0)
        kwargs["max_units"] = required
        # Re-using cached DXFs — we already ran preprocess+trace successfully.
        kwargs["skip_trace"] = True
        result = run_pipeline(**kwargs)
    except RuntimeError as e:
        print(f"\n{e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Traces:  {len(result['dxf_paths'])} tool(s)")
    print(f"  Layout:  {result['grid_units_x']}x"
          f"{result['grid_units_y']} gridfinity units")
    print(f"  Config:  {result['bin_config']}")
    print(f"  Preview: {result['layout_preview']}")
    print(f"\nNext: Open Fusion 360 -> Scripts -> Run fusion_bin_script")


def _parse_bool(value: str) -> bool:
    """Parse a true/false string for argparse."""
    v = value.strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(
        f"Expected true/false, got: {value!r}")


def _resolve_tool_height(tool_heights, idx: int, default: float = 0.0) -> float:
    """Look up the tool height for image index *idx* from the parsed args."""
    if isinstance(tool_heights, (int, float)):
        return float(tool_heights)
    if idx in tool_heights:
        return float(tool_heights[idx])
    if "default" in tool_heights:
        return float(tool_heights["default"])
    return float(default)


def _parse_tool_height_args(height_strs: list[str]) -> dict | float:
    """Parse --tool-height arguments into a float or index->float dict."""
    if len(height_strs) == 1 and "=" not in height_strs[0]:
        return float(height_strs[0])
    heights = {}
    for s in height_strs:
        if "=" in s:
            key, val = s.split("=", 1)
            try:
                heights[int(key)] = float(val)
            except ValueError:
                heights[key] = float(val)
        else:
            heights["default"] = float(s)
    return heights


if __name__ == "__main__":
    main()
