"""
Iterative trace refinement: automatically adjusts cleanup parameters
to preserve significant concavities while removing scanner artifacts.

Wraps trace_tool's segmentation and cleanup pipeline in a feedback loop
that compares the cleaned mask to the raw segmentation mask. If cleanup
fills concavities deeper than a threshold (e.g. the gap between pliers
handles), parameters are reduced and cleanup is re-run.
"""

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from pic_to_bin.trace_tool import (
    segment_tool, cleanup_mask, trace_from_mask, _fill_mask_holes,
    erode_mask_mm,
)


# ---------------------------------------------------------------------------
# Cleanup parameters
# ---------------------------------------------------------------------------

@dataclass
class CleanupParams:
    """Mutable cleanup parameter set for iterative refinement. All in mm."""
    kernel_mm: float = 0.9
    smooth_radius_mm: float = 0.9
    shadow_kernel_mm: float = 0.0
    contour_smooth_sigma_mm: float = 0.6
    notch_fill_mm: float = 2.4

    def to_dict(self) -> dict:
        return {
            "kernel_mm": self.kernel_mm,
            "smooth_radius_mm": self.smooth_radius_mm,
            "shadow_kernel_mm": self.shadow_kernel_mm,
            "contour_smooth_sigma_mm": self.contour_smooth_sigma_mm,
            "notch_fill_mm": self.notch_fill_mm,
        }

    def summary(self) -> str:
        return (f"notch_fill={self.notch_fill_mm:.2f}mm, "
                f"contour_smooth={self.contour_smooth_sigma_mm:.2f}mm, "
                f"kernel={self.kernel_mm:.2f}mm, "
                f"smooth_r={self.smooth_radius_mm:.2f}mm")


# ---------------------------------------------------------------------------
# Mask comparison
# ---------------------------------------------------------------------------

def compare_masks(raw_mask: np.ndarray, cleaned_mask: np.ndarray,
                  scale: float, min_depth_mm: float = 2.0) -> dict:
    """Compare raw segmentation mask to cleaned mask to detect cleanup distortion.

    Detects both lost pixels (raw & ~cleaned, from erosion/open) and gained
    pixels (cleaned & ~raw, from close/contour-fill). The pliers handle gap
    scenario is a "gained" case — cleanup fills the concavity, adding pixels
    that weren't in the raw mask.

    Args:
        raw_mask: Binary mask (uint8, 0/255) from SAM2 before cleanup
        cleaned_mask: Binary mask after cleanup_mask()
        scale: mm per pixel (25.4 / dpi)
        min_depth_mm: Minimum concavity depth to consider significant

    Returns:
        Dict with:
            lost_pixels: pixels in raw but not cleaned (erosion)
            gained_pixels: pixels in cleaned but not raw (fill)
            max_depth_mm: max depth across both lost and gained regions
            significant_concavities: count of deep distortion regions
            diff_mask: binary mask of all changed pixels (uint8)
            depth_map: distance transform of changed regions (float32)
    """
    raw_bool = raw_mask > 128
    cleaned_bool = cleaned_mask > 128

    lost = raw_bool & ~cleaned_bool       # pixels removed by cleanup
    gained = cleaned_bool & ~raw_bool     # pixels added by cleanup (filled concavities)
    changed = lost | gained               # any distortion
    changed_uint8 = changed.astype(np.uint8) * 255

    lost_pixels = int(np.count_nonzero(lost))
    gained_pixels = int(np.count_nonzero(gained))
    total_changed = lost_pixels + gained_pixels

    if total_changed == 0:
        return {
            "lost_pixels": 0,
            "gained_pixels": 0,
            "max_depth_mm": 0.0,
            "significant_concavities": 0,
            "diff_mask": changed_uint8,
            "depth_map": np.zeros_like(raw_mask, dtype=np.float32),
        }

    # Distance transform on all changed regions
    dist = cv2.distanceTransform(changed_uint8, cv2.DIST_L2, 5)
    dist_mm = dist * scale

    max_depth_mm = float(dist_mm.max())

    # Count significant distortion clusters
    n_labels, labels = cv2.connectedComponents(changed_uint8)
    significant = 0
    for label_id in range(1, n_labels):
        region = labels == label_id
        if dist_mm[region].max() >= min_depth_mm:
            significant += 1

    return {
        "lost_pixels": lost_pixels,
        "gained_pixels": gained_pixels,
        "max_depth_mm": max_depth_mm,
        "significant_concavities": significant,
        "diff_mask": changed_uint8,
        "depth_map": dist_mm,
    }


# ---------------------------------------------------------------------------
# Comparison image
# ---------------------------------------------------------------------------

def generate_comparison_image(raw_mask: np.ndarray, cleaned_mask: np.ndarray,
                              diff_mask: np.ndarray, depth_map: np.ndarray,
                              iteration: int, params: CleanupParams,
                              metrics: dict, output_path: str) -> str:
    """Generate a 4-panel comparison image for one iteration.

    Panels:
        Top-left: Raw mask (SAM2 output)
        Top-right: Cleaned mask (this iteration)
        Bottom-left: Changed regions overlaid on raw mask
                     (red = gained/filled, blue = lost/eroded)
        Bottom-right: Depth heatmap of changed regions

    Returns:
        Path to saved comparison image
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Top-left: Raw mask
    axes[0, 0].imshow(raw_mask, cmap='gray')
    axes[0, 0].set_title("Raw Mask (SAM2)")
    axes[0, 0].axis('off')

    # Top-right: Cleaned mask
    axes[0, 1].imshow(cleaned_mask, cmap='gray')
    axes[0, 1].set_title(f"Cleaned Mask (iter {iteration})")
    axes[0, 1].axis('off')

    # Bottom-left: Changed regions overlaid on raw mask
    raw_bool = raw_mask > 128
    cleaned_bool = cleaned_mask > 128
    overlay = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2RGB)
    gained = cleaned_bool & ~raw_bool
    lost = raw_bool & ~cleaned_bool
    overlay[gained] = [255, 0, 0]    # red = filled concavities
    overlay[lost] = [0, 0, 255]      # blue = eroded pixels
    total_changed = metrics["gained_pixels"] + metrics["lost_pixels"]
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(
        f"Changes: {metrics['gained_pixels']} filled (red), "
        f"{metrics['lost_pixels']} eroded (blue)")
    axes[1, 0].axis('off')

    # Bottom-right: Depth heatmap
    if depth_map.max() > 0:
        im = axes[1, 1].imshow(depth_map, cmap='hot', interpolation='nearest')
        plt.colorbar(im, ax=axes[1, 1], label='Depth (mm)')
    else:
        axes[1, 1].imshow(np.zeros_like(raw_mask), cmap='gray')
    axes[1, 1].set_title(
        f"Distortion Depth — max {metrics['max_depth_mm']:.1f} mm, "
        f"{metrics['significant_concavities']} significant")
    axes[1, 1].axis('off')

    fig.suptitle(
        f"Iteration {iteration}: {params.summary()}\n"
        f"Max depth: {metrics['max_depth_mm']:.1f} mm | "
        f"Significant concavities: {metrics['significant_concavities']}",
        fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Parameter reduction
# ---------------------------------------------------------------------------

def _reduce_params(params: CleanupParams) -> CleanupParams:
    """Reduce cleanup aggressiveness by one step. All values in mm.

    Priority order (reduces one param per call):
    1. notch_fill_mm: halve (2.4 -> 1.2 -> 0.6 -> 0.4, floor 0.4mm)
    2. contour_smooth_sigma_mm: x0.6 (0.6 -> 0.36 -> 0.22 -> 0.13, floor 0.13mm, disable below)
    3. kernel_mm: halve (0.9 -> 0.45, floor 0.4mm)
    4. smooth_radius_mm: halve (0.9 -> 0.45, floor 0.4mm)

    Returns a new CleanupParams, or the same object if nothing can be reduced.
    """
    FLOOR_MM = 0.4
    SIGMA_FLOOR_MM = 0.13

    # Priority 1: Reduce notch_fill (primary concavity filler)
    if params.notch_fill_mm > FLOOR_MM + 1e-6:
        new_nf = max(FLOOR_MM, round(params.notch_fill_mm / 2, 2))
        return CleanupParams(
            kernel_mm=params.kernel_mm,
            smooth_radius_mm=params.smooth_radius_mm,
            shadow_kernel_mm=params.shadow_kernel_mm,
            contour_smooth_sigma_mm=params.contour_smooth_sigma_mm,
            notch_fill_mm=new_nf,
        )

    # Priority 2: Reduce contour smoothing
    if params.contour_smooth_sigma_mm > SIGMA_FLOOR_MM + 1e-6:
        new_cs = round(params.contour_smooth_sigma_mm * 0.6, 2)
        if new_cs < SIGMA_FLOOR_MM:
            new_cs = 0.0  # disable entirely below floor
        return CleanupParams(
            kernel_mm=params.kernel_mm,
            smooth_radius_mm=params.smooth_radius_mm,
            shadow_kernel_mm=params.shadow_kernel_mm,
            contour_smooth_sigma_mm=new_cs,
            notch_fill_mm=params.notch_fill_mm,
        )

    # Priority 3: Reduce kernel_mm
    if params.kernel_mm > FLOOR_MM + 1e-6:
        return CleanupParams(
            kernel_mm=max(FLOOR_MM, round(params.kernel_mm / 2, 2)),
            smooth_radius_mm=params.smooth_radius_mm,
            shadow_kernel_mm=params.shadow_kernel_mm,
            contour_smooth_sigma_mm=params.contour_smooth_sigma_mm,
            notch_fill_mm=params.notch_fill_mm,
        )

    # Priority 4: Reduce smooth_radius_mm
    if params.smooth_radius_mm > FLOOR_MM + 1e-6:
        return CleanupParams(
            kernel_mm=params.kernel_mm,
            smooth_radius_mm=max(FLOOR_MM, round(params.smooth_radius_mm / 2, 2)),
            shadow_kernel_mm=params.shadow_kernel_mm,
            contour_smooth_sigma_mm=params.contour_smooth_sigma_mm,
            notch_fill_mm=params.notch_fill_mm,
        )

    # Nothing left to reduce — return same object to signal convergence
    return params


# ---------------------------------------------------------------------------
# Main feedback loop
# ---------------------------------------------------------------------------

def refine_trace(
    image_path: str,
    dpi: int = 200,
    clearance_mm: float = 0.0,
    tolerance_mm: float = 0.0,
    axial_tolerance_mm: float = 0.0,
    alphamax: float = 1.2,
    turdsize: int = 50,
    opttolerance: float = 2.0,
    straighten_threshold: float = 45.0,
    output_dir: str = None,
    max_iterations: int = 5,
    max_concavity_depth_mm: float = 3.0,
    sam_model: str = "sam2.1_l.pt",
    mask_erode_mm: float = 0.3,
    tool_height_mm: float = 0.0,
    phone_height_mm: float = 0.0,
    tool_taper: str = "top",
    finger_slots: bool = True,
    display_smooth_sigma_mm: float = 2.5,
) -> dict:
    """Run trace_tool iteratively, adjusting cleanup params to preserve concavities.

    1. Segments the image once with SAM2
    2. Iteratively runs cleanup_mask with decreasing aggressiveness
    3. Compares each cleaned mask to the raw mask to detect lost concavities
    4. When concavity loss is acceptable, runs the full export pipeline

    Args:
        image_path..output_dir: Same as trace_tool()
        max_iterations: Maximum refinement iterations (default 5).
            Set to 1 to disable iterative refinement.
        max_concavity_depth_mm: Maximum acceptable concavity depth loss in mm.
            Concavities deeper than this trigger parameter reduction. Default 3.0.
        sam_model: SAM2 model weight name (default: sam2.1_l.pt)

    Returns:
        Dict with same keys as trace_tool() plus:
            raw_mask_path: Path to raw SAM2 mask
            refinement_iterations: Number of iterations run
            refinement_converged: Whether trace met quality threshold
            comparison_paths: List of comparison image paths
    """
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = Path("generated") / image_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    scale = 25.4 / dpi

    print(f"Processing: {image_path.name}")
    print(f"  DPI: {dpi}, scale: {scale:.4f} mm/pixel")

    # Step 1: Segmentation (runs once)
    print("Step 1: Segmentation...")
    raw_mask = segment_tool(str(image_path), sam_model=sam_model)

    # Save raw mask (original, pre-erosion — useful for debugging)
    raw_mask_path = output_dir / f"{stem}_raw_mask.png"
    cv2.imwrite(str(raw_mask_path), raw_mask)

    # Post-SAM erosion: counter shadow-halo / soft-edge bias. Applied once so
    # every subsequent step (iteration loop + final export) sees the corrected
    # mask, and the concavity comparison baseline stays consistent.
    if mask_erode_mm > 0:
        print(f"  Post-SAM erosion: {mask_erode_mm:.2f}mm")
        raw_mask = erode_mask_mm(raw_mask, mask_erode_mm, dpi)

    # Pre-fill holes in the raw mask so comparison only measures cleanup
    # distortion, not the intentional hole-filling that cleanup_mask applies.
    filled_raw = _fill_mask_holes(raw_mask.copy())

    # Iterative cleanup refinement
    params = CleanupParams()
    comparison_paths = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Refinement iteration {iteration}/{max_iterations} ---")
        print(f"  Params: {params.summary()}")

        # Run cleanup (no straighten yet — compare at same dimensions as raw)
        cleaned = cleanup_mask(raw_mask.copy(), dpi=dpi, **params.to_dict())

        # Compare against hole-filled raw mask so that intentional hole
        # filling doesn't register as distortion
        metrics = compare_masks(filled_raw, cleaned, scale)
        print(f"  Gained: {metrics['gained_pixels']} px, "
              f"lost: {metrics['lost_pixels']} px, "
              f"max depth: {metrics['max_depth_mm']:.1f} mm, "
              f"significant: {metrics['significant_concavities']}")

        # Generate comparison image (use filled_raw as baseline)
        comp_path = str(output_dir / f"{stem}_comparison_iter{iteration}.png")
        generate_comparison_image(
            filled_raw, cleaned, metrics["diff_mask"], metrics["depth_map"],
            iteration, params, metrics, comp_path)
        comparison_paths.append(comp_path)
        print(f"  Saved comparison: {comp_path}")

        # Check convergence
        if metrics["max_depth_mm"] <= max_concavity_depth_mm:
            print(f"  Converged! Max depth {metrics['max_depth_mm']:.1f} mm "
                  f"<= threshold {max_concavity_depth_mm:.1f} mm")
            converged = True
            break

        # Try reducing parameters
        new_params = _reduce_params(params)
        if new_params is params:
            print("  Cannot reduce parameters further. Accepting current result.")
            break

        params = new_params

    # Run full pipeline (cleanup + straighten + vectorize + export)
    print(f"\n--- Final export with: {params.summary()} ---")
    result = trace_from_mask(
        raw_mask, stem, dpi=dpi,
        clearance_mm=clearance_mm, tolerance_mm=tolerance_mm,
        axial_tolerance_mm=axial_tolerance_mm,
        alphamax=alphamax, turdsize=turdsize, opttolerance=opttolerance,
        straighten_threshold=straighten_threshold,
        output_dir=str(output_dir),
        tool_height_mm=tool_height_mm,
        phone_height_mm=phone_height_mm,
        tool_taper=tool_taper,
        finger_slots=finger_slots,
        display_smooth_sigma_mm=display_smooth_sigma_mm,
        **params.to_dict(),
    )

    result["raw_mask_path"] = str(raw_mask_path)
    result["refinement_iterations"] = iteration
    result["refinement_converged"] = converged
    result["comparison_paths"] = comparison_paths
    return result
