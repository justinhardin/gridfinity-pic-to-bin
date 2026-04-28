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
"""

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

from pic_to_bin.phone_preprocess import (
    preprocess_phone_image,
    MarkerDetectionError,
    ScaleInconsistencyError,
)
from pic_to_bin.trace_tool import _collect_images
from pic_to_bin.refine_trace import refine_trace
from pic_to_bin.layout_tools import layout_tools, GridSizeError
from pic_to_bin.prepare_bin import prepare_bin


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a gridfinity bin config from phone camera photos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  pic-to-bin photo.jpg --tool-height 17
  pic-to-bin a.jpg b.jpg --tool-height 0=17 --tool-height 1=14
  pic-to-bin photo.jpg --tool-height 17 --paper-size legal
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
        "--paper-size", choices=["a4", "letter", "legal"], default="letter",
        help="Template paper size used for photos (default: letter)")
    parser.add_argument(
        "--tolerance", type=float, default=0.0,
        help="Tolerance outline offset in mm (default: 0). Positive "
             "expands the pocket past the trace; negative shrinks it.")
    parser.add_argument(
        "--gap", type=float, default=3.0,
        help="Minimum gap between tools in mm (default: 3.0)")
    parser.add_argument(
        "--max-units", type=int, default=7,
        help="Maximum grid size in gridfinity units per axis (default: 7)")
    parser.add_argument(
        "--height-units", type=int, default=None,
        help="Force bin height in gridfinity units (default: auto)")
    parser.add_argument(
        "--stacking", type=_parse_bool, default=True,
        metavar="true|false",
        help="Generate stacking lip (default: true). Set false to remove the "
             "lip and reduce overall bin height for shallow drawers.")
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
        "--mask-erode", type=float, default=0.3,
        help="Post-SAM mask erosion in mm to counter shadow halos (default: 0.3, "
             "0 to disable). Increase if handles still read wide.")
    parser.add_argument(
        "--sam-model", type=str, default="sam2.1_l.pt",
        help="SAM2 model weights (default: sam2.1_l.pt)")
    parser.add_argument(
        "--skip-trace", action="store_true",
        help="Skip tracing, reuse existing DXFs in generated/")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ── Check for existing output ──────────────────────────────────────
    if not check_existing_output(output_dir, args.skip_trace):
        sys.exit(0)

    # ── Step 0: Collect images ─────────────────────────────────────────
    image_sources = args.images if args.images else ["all_images"]
    image_paths = _collect_images(image_sources)
    print(f"Found {len(image_paths)} image(s): "
          + ", ".join(p.name for p in image_paths))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Steps 1-2: Preprocess + Trace each image ──────────────────────
    dxf_paths = []
    if args.skip_trace:
        print("\n-- Skipping trace (--skip-trace), looking for existing DXFs --")
        for img in image_paths:
            dxf = output_dir / img.stem / f"{img.stem}_trace.dxf"
            if dxf.exists():
                dxf_paths.append(str(dxf))
                print(f"  Found: {dxf}")
            else:
                print(f"  WARNING: {dxf} not found, skipping {img.name}")
    else:
        print("\n" + "=" * 60)
        print("STEP 1/3: Phone preprocessing + Tracing tool outlines")
        print("=" * 60)
        for img in image_paths:
            tool_output_dir = output_dir / img.stem
            try:
                # Step 1: Phone preprocessing
                print(f"\n--- Preprocessing {img.name} ---")
                pp = preprocess_phone_image(
                    str(img),
                    paper_size=args.paper_size,
                    output_dir=str(tool_output_dir),
                )
                rectified_img = Path(pp["rectified_image_path"])
                dpi = round(pp["effective_dpi"])
                print(f"  Phone mode: {pp['markers_detected']}/8 markers, "
                      f"effective DPI: {dpi}")

                # Step 2: Trace (SAM2 segmentation on rectified image)
                print(f"\n--- Tracing {img.name} ---")
                result = refine_trace(
                    image_path=str(rectified_img),
                    dpi=dpi,
                    tolerance_mm=args.tolerance,
                    straighten_threshold=args.straighten_threshold,
                    output_dir=str(tool_output_dir),
                    max_iterations=args.max_refine_iterations,
                    max_concavity_depth_mm=args.max_concavity_depth,
                    sam_model=args.sam_model,
                    mask_erode_mm=args.mask_erode,
                )
                dxf_paths.append(result["dxf_path"])
                iters = result.get("refinement_iterations", 1)
                conv = result.get("refinement_converged", True)
                if iters > 1:
                    status = "converged" if conv else "max iterations"
                    print(f"  Refined in {iters} iterations ({status})")

            except MarkerDetectionError as e:
                print(f"\nERROR preprocessing {img.name}: {e}\n")
            except ScaleInconsistencyError as e:
                print(f"\nERROR (scale) {img.name}: {e}\n")
            except Exception as e:
                print(f"\nERROR processing {img.name}: {e}\n")

    if not dxf_paths:
        print("No DXF traces produced. Exiting.")
        sys.exit(1)

    # ── Step 3: Layout packing ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2/3: Packing layout")
    print("=" * 60)

    max_units = args.max_units
    try:
        layout_result = layout_tools(
            dxf_paths=dxf_paths,
            gap_mm=args.gap,
            max_units=max_units,
            output_dir=str(output_dir),
        )
    except GridSizeError as e:
        required = max(e.required_x, e.required_y)
        print(f"\nTools don't fit in a {max_units}x{max_units} grid. "
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
        layout_result = layout_tools(
            dxf_paths=dxf_paths,
            gap_mm=args.gap,
            max_units=required,
            output_dir=str(output_dir),
        )

    combined_dxf = layout_result["combined_dxf_path"]

    # ── Step 4: Generate bin config ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3/3: Generating bin config")
    print("=" * 60)

    tool_heights = _parse_tool_height_args(args.tool_heights)

    config_path = prepare_bin(
        dxf_path=combined_dxf,
        tool_heights=tool_heights,
        height_units=args.height_units,
        stacking_lip=args.stacking,
        output_path=str(output_dir / "bin_config.json"),
    )

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Traces:  {len(dxf_paths)} tool(s)")
    print(f"  Layout:  {layout_result['grid_units_x']}x"
          f"{layout_result['grid_units_y']} gridfinity units")
    print(f"  Config:  {config_path}")
    print(f"  Preview: {layout_result['preview_path']}")
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
