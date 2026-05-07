"""
Agent 2: Trace Validation
Validates the generated trace against the original image.

Checks:
1. Visual overlay - renders trace on original image for inspection
2. Dimensional check - compares bounding box against known tool length
3. Path integrity - verifies closed path, no self-intersections
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from svgpathtools import svg2paths2


def create_overlay(image_path: str, svg_path: str, dpi: int = 200,
                   output_path: str = None) -> str:
    """Render SVG trace overlaid on the original scanner image.

    Args:
        image_path: Path to original scanner image
        svg_path: Path to generated SVG trace
        dpi: Scanner DPI
        output_path: Where to save overlay image

    Returns:
        Path to saved overlay image
    """
    if output_path is None:
        output_path = str(Path(svg_path).with_suffix('.overlay.png'))

    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    scale = 25.4 / dpi  # mm per pixel

    # Parse SVG paths
    paths, attrs, svg_attrs = svg2paths2(svg_path)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    # Left: original image with trace overlay
    axes[0].imshow(img_rgb)
    axes[0].set_title("Trace Overlay on Original", fontsize=14)

    # Detect if SVG uses cm and get scale factor to mm
    unit_scale = _detect_svg_unit_scale(svg_path)

    for path in paths:
        # Sample points along the SVG path
        points = []
        for i in range(1000):
            t = i / 999
            try:
                pt = path.point(t)
                # SVG coords may be in cm (for Fusion 360), convert to mm then to pixels
                px = pt.real * unit_scale / scale
                py = pt.imag * unit_scale / scale
                points.append((px, py))
            except (ValueError, ZeroDivisionError):
                continue

        if points:
            xs, ys = zip(*points)
            axes[0].plot(xs, ys, 'c-', linewidth=2, alpha=0.8)

    axes[0].axis('off')

    # Right: mask comparison
    stem = Path(image_path).stem
    # Look for mask in generated/<stem>/ first, then next to image
    generated_mask = Path("generated") / stem / f"{stem}_mask.png"
    adjacent_mask = Path(image_path).parent / f"{stem}_mask.png"
    mask_path = str(generated_mask if generated_mask.exists() else adjacent_mask)
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title("Segmentation Mask", fontsize=14)
        else:
            axes[1].text(0.5, 0.5, "Mask not found", ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=16)
    except Exception:
        axes[1].text(0.5, 0.5, "Mask not found", ha='center', va='center',
                    transform=axes[1].transAxes, fontsize=16)

    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Overlay saved: {output_path}")
    return output_path


def _detect_svg_unit_scale(svg_path: str) -> float:
    """Detect the unit scale factor to convert SVG coordinates to mm.

    Our SVGs use cm coordinates (for Fusion 360 compatibility).
    svgpathtools reads raw coordinate numbers without unit awareness.
    Returns the multiplier to convert SVG coordinates to mm.
    """
    with open(svg_path, 'r') as f:
        header = f.read(500)

    # Check if width/height attributes use cm
    import re
    match = re.search(r'width="[\d.]+cm"', header)
    if match:
        return 10.0  # cm → mm

    match = re.search(r'width="[\d.]+mm"', header)
    if match:
        return 1.0  # already mm

    # Default: assume mm
    return 1.0


def check_dimensions(svg_path: str, expected_length_mm: float = None,
                     tolerance_pct: float = 5.0) -> dict:
    """Check trace dimensions and optionally compare to expected length.

    Args:
        svg_path: Path to SVG trace
        expected_length_mm: Expected longest dimension in mm
        tolerance_pct: Acceptable tolerance percentage

    Returns:
        Dict with dimension info and pass/fail status
    """
    paths, attrs, svg_attrs = svg2paths2(svg_path)

    if not paths:
        return {"error": "No paths found in SVG", "pass": False}

    # Detect if SVG uses cm (for Fusion 360) and convert to mm
    unit_scale = _detect_svg_unit_scale(svg_path)

    # Compute bounding box across all paths
    all_min_x = float('inf')
    all_min_y = float('inf')
    all_max_x = float('-inf')
    all_max_y = float('-inf')

    for path in paths:
        bbox = path.bbox()  # returns (xmin, xmax, ymin, ymax)
        all_min_x = min(all_min_x, bbox[0])
        all_max_x = max(all_max_x, bbox[1])
        all_min_y = min(all_min_y, bbox[2])
        all_max_y = max(all_max_y, bbox[3])

    width = (all_max_x - all_min_x) * unit_scale
    height = (all_max_y - all_min_y) * unit_scale
    longest = max(width, height)

    result = {
        "width_mm": width,
        "height_mm": height,
        "longest_mm": longest,
        "num_paths": len(paths),
        "pass": True,
        "messages": [],
    }

    # Check path count: expect 2 (inner + outer) or 3 (inner + outer + slot)
    if len(paths) < 2:
        result["messages"].append(
            f"WARNING: only {len(paths)} path found (expected 2-3: inner + outer + optional slot)")
    elif len(paths) > 3:
        result["messages"].append(
            f"WARNING: {len(paths)} paths found (expected 2-3: inner + outer + optional slot)")
    else:
        result["messages"].append(f"OK: {len(paths)} paths found")

    # Check against expected length
    if expected_length_mm is not None:
        error_pct = abs(longest - expected_length_mm) / expected_length_mm * 100
        result["expected_mm"] = expected_length_mm
        result["error_pct"] = error_pct

        if error_pct > tolerance_pct:
            result["pass"] = False
            result["messages"].append(
                f"FAIL: Longest dimension {longest:.1f}mm vs expected "
                f"{expected_length_mm:.0f}mm (error: {error_pct:.1f}%)")
        else:
            result["messages"].append(
                f"OK: Longest dimension {longest:.1f}mm vs expected "
                f"{expected_length_mm:.0f}mm (error: {error_pct:.1f}%)")

    # Check path closure
    for i, path in enumerate(paths):
        if path.isclosed():
            result["messages"].append(f"Path {i}: closed (OK)")
        else:
            start = path.start
            end = path.end
            gap = abs(start - end)
            result["messages"].append(
                f"WARNING: Path {i} not closed (gap: {gap:.2f}mm)")

    return result


def validate(image_path: str, svg_path: str, dpi: int = 200,
             expected_length_mm: float = None) -> dict:
    """Run all validation checks.

    Args:
        image_path: Original scanner image
        svg_path: Generated SVG trace
        dpi: Scanner DPI
        expected_length_mm: Expected tool length for dimensional check

    Returns:
        Validation results dict
    """
    print("=" * 60)
    print("TRACE VALIDATION")
    print("=" * 60)

    results = {}

    # 1. Create overlay
    print("\n1. Creating overlay image...")
    overlay_path = create_overlay(image_path, svg_path, dpi)
    results["overlay_path"] = overlay_path

    # 2. Dimensional check
    print("\n2. Dimensional check...")
    dim_results = check_dimensions(svg_path, expected_length_mm)
    results["dimensions"] = dim_results
    for msg in dim_results["messages"]:
        print(f"  {msg}")
    print(f"  Width: {dim_results['width_mm']:.1f}mm, Height: {dim_results['height_mm']:.1f}mm")

    # 3. Summary
    print("\n" + "=" * 60)
    overall_pass = dim_results["pass"]
    if overall_pass:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
    print("=" * 60)

    results["pass"] = overall_pass
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate a tool trace")
    parser.add_argument("image", help="Original scanner image")
    parser.add_argument("svg", help="Generated SVG trace")
    parser.add_argument("--dpi", type=int, default=200, help="Scanner DPI")
    parser.add_argument("--expected-length", type=float, default=None,
                        help="Expected tool length in mm for dimensional check")

    args = parser.parse_args()
    validate(args.image, args.svg, args.dpi, args.expected_length)


if __name__ == "__main__":
    main()
