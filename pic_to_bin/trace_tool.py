"""
Agent 1: Tool Trace Generator
Converts a flatbed scanner image of a tool into an SVG/DXF outline.

Pipeline:
1. Segmentation (SAM2 neural network via ultralytics)
2. Mask cleanup (morphological operations)
3. Vectorization (potrace - bitmap to Bezier curves + straight lines)
4. Scale to real-world mm using scanner DPI
5. Add clearance offset for 3D printing tolerance
6. Export SVG and DXF
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import potrace
from ultralytics import SAM

from pic_to_bin.trace_export import potrace_to_svg, potrace_to_dxf, compute_finger_slot


# ---------------------------------------------------------------------------
# SAM model cache
# ---------------------------------------------------------------------------

_sam_model = None
_sam_model_name = None


def _get_sam_model(model_name: str = "sam2.1_l.pt"):
    """Load and cache the SAM2 model (downloads weights on first use)."""
    global _sam_model, _sam_model_name
    if _sam_model is None or _sam_model_name != model_name:
        print(f"  Loading SAM model: {model_name}")
        _sam_model = SAM(model_name)
        _sam_model_name = model_name
    return _sam_model


def segment_tool(image_path: str, sam_model: str = "sam2.1_l.pt") -> np.ndarray:
    """Extract binary mask of the tool from the scanner image using SAM2.

    Uses a two-step approach:
    1. Quick Otsu threshold to find the approximate tool bounding box
    2. SAM2 with bounding box prompt for precise neural segmentation

    For dark backgrounds (lid-open scanning), interior gap refinement
    carves out handle gaps that SAM may fill.

    Args:
        image_path: Path to the scanner image
        sam_model: SAM2 model weight name (default: sam2.1_l.pt)

    Returns:
        Binary mask (uint8, 0 or 255) where 255 = tool
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape

    # --- Step 1: Find approximate tool location ---
    # Background detection from corners (gray is fine here — we only need to
    # know if the surround is light or dark to pick the right "non-bg" score).
    margin = min(10, h // 20, w // 20)
    corners = np.concatenate([
        gray[:margin, :margin].ravel(),
        gray[:margin, -margin:].ravel(),
        gray[-margin:, :margin].ravel(),
        gray[-margin:, -margin:].ravel(),
    ])
    bg_median = float(np.median(corners))

    # "Distance from background" score, then Otsu. On light backgrounds
    # plain gray-Otsu can split multi-tone tools (e.g. yellow body + black
    # grips) into many disconnected dark fragments because the bright body
    # gets binned with the paper. Using max(saturation, 255 - value) instead
    # treats both saturated *and* dark pixels as foreground, so the whole
    # tool comes through as one contiguous component.
    sat = hsv[..., 1]
    val = hsv[..., 2]
    if bg_median > 128:
        score = np.maximum(sat, 255 - val)
    else:
        score = np.maximum(sat, val)
    _, rough_mask = cv2.threshold(
        score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean rough mask to get a stable bounding box
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_CLOSE, k)
    rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(
        rough_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = None
    if contours:
        # Multi-tone tools (yellow body + black grips, etc.) on a light
        # background can produce several disconnected Otsu components — one
        # per dark feature. Union the bboxes of every tool-sized contour so
        # SAM2's bbox prompt spans the whole tool, not just the largest
        # fragment. Falls through to the single-contour case naturally.
        img_area = h * w
        tool_contours = [c for c in contours
                         if cv2.contourArea(c) / img_area > 0.01]
        if tool_contours:
            rects = [cv2.boundingRect(c) for c in tool_contours]
            bx = min(r[0] for r in rects)
            by = min(r[1] for r in rects)
            bx2 = max(r[0] + r[2] for r in rects)
            by2 = max(r[1] + r[3] for r in rects)
            pad = max(30, int(0.02 * max(h, w)))
            bbox = [
                max(0, bx - pad),
                max(0, by - pad),
                min(w, bx2 + pad),
                min(h, by2 + pad),
            ]

    # --- Step 2: SAM2 segmentation ---
    model = _get_sam_model(sam_model)

    if bbox is not None:
        print(f"  SAM2 bbox prompt: [{bbox[0]}, {bbox[1]}, "
              f"{bbox[2]}, {bbox[3]}]")
        results = model(image_path, bboxes=[bbox], verbose=False)
    else:
        print("  SAM2 auto-segmentation (no bbox found from threshold)")
        results = model(image_path, verbose=False)

    mask = _extract_best_mask(results, h, w)

    # Fallback: if SAM produced an empty mask (can happen on synthetic
    # images or very unusual scans), use the rough threshold mask instead.
    if np.count_nonzero(mask) == 0 and rough_mask is not None:
        print("  WARNING: SAM produced empty mask, falling back to "
              "threshold segmentation")
        # Re-derive rough_mask (it was modified in-place above for bbox)
        if bg_median > 128:
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(f"  Segmentation: SAM2 ({sam_model})")
    print(f"  Mask shape: {mask.shape}, tool pixels: {np.count_nonzero(mask)}")

    # On bright backgrounds, recover dark tool regions (e.g. black metal
    # handles) that SAM2 may exclude.  Uses Otsu thresholding to find all
    # non-background pixels and adds threshold-connected regions back.
    mask = _recover_bright_bg_missed(mask, gray)

    # Refine mask using original image brightness to carve out interior
    # gaps (e.g. handle gaps on pliers) that SAM may fill.
    # Only runs on dark backgrounds (lid-open scanning).
    mask = _refine_mask_with_image(mask, gray)

    return mask


def _extract_best_mask(results, img_h: int, img_w: int) -> np.ndarray:
    """Extract the best tool mask from SAM2 results.

    For bbox-prompted results: takes the first (highest quality) mask.
    For auto-segmentation results: takes the largest non-background mask.

    Args:
        results: ultralytics Results list from SAM inference
        img_h: Original image height
        img_w: Original image width

    Returns:
        Binary mask (uint8, 0 or 255)
    """
    if (not results or results[0].masks is None
            or len(results[0].masks.data) == 0):
        print("  WARNING: SAM produced no masks, returning empty mask")
        return np.zeros((img_h, img_w), dtype=np.uint8)

    masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
    total_pixels = img_h * img_w

    if len(masks) == 1:
        mask = masks[0]
    else:
        # Multiple masks: pick the largest that isn't background
        best_idx = 0
        best_area = 0
        for i, m in enumerate(masks):
            area = float(m.sum())
            if area / total_pixels > 0.80:
                continue  # skip background-sized masks
            if area > best_area:
                best_area = area
                best_idx = i
        mask = masks[best_idx]

    mask = (mask > 0.5).astype(np.uint8) * 255

    # Resize if SAM returned a different resolution
    if mask.shape[0] != img_h or mask.shape[1] != img_w:
        mask = cv2.resize(mask, (img_w, img_h),
                          interpolation=cv2.INTER_NEAREST)

    return mask


def _recover_bright_bg_missed(mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Recover dark tool regions that SAM2 missed on bright backgrounds.

    On white/bright backgrounds (e.g. white paper under the scanner lid),
    SAM2 may exclude dark surfaces (black metal handles, pivot mechanisms)
    that contrast differently from the main tool body (orange/colored grips).
    These dark surfaces are clearly non-background in a simple threshold,
    but SAM2's neural segmentation doesn't associate them with the object.

    Strategy:
    1. Detect bright background (median > 128).
    2. Otsu threshold finds all non-background pixels (both coloured and dark).
    3. Connected components of the threshold mask identify contiguous
       foreground blobs.
    4. Any threshold component that overlaps the SAM2 mask is part of the
       tool — its full extent (including dark areas SAM2 missed) is added.
    5. Safety: if recovery would grow the mask by more than 50%, skip
       (suggests threshold noise connecting to scanner-edge artifacts).

    Only runs on bright backgrounds.  Dark-background scans use
    ``_refine_mask_with_image`` instead (which carves interior gaps).

    Args:
        mask: Binary mask (uint8, 0/255) from SAM2
        gray: Grayscale original image (same dimensions as mask)

    Returns:
        Mask with recovered dark regions (or original mask if not applicable)
    """
    tool_region = mask > 128
    bg_region = ~tool_region

    if np.count_nonzero(bg_region) == 0:
        return mask

    bg_median = float(np.median(gray[bg_region]))
    if bg_median <= 60:
        return mask  # dark background — handled by _refine_mask_with_image

    # Otsu threshold: all non-background pixels
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Light cleanup — close small gaps, remove noise specks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)

    # Connected components of threshold foreground
    n_labels, labels = cv2.connectedComponents(thresh)

    # Find which components overlap with the SAM2 mask
    overlapping = set(np.unique(labels[tool_region]))
    overlapping.discard(0)  # background label

    # Identify pixels to add: in overlapping threshold components but not in SAM2
    recovery = np.zeros(mask.shape, dtype=bool)
    for lid in overlapping:
        component = labels == lid
        recovery |= (component & ~tool_region)

    added = int(np.count_nonzero(recovery))
    if added == 0:
        return mask

    original_size = int(np.count_nonzero(tool_region))
    ratio = added / original_size if original_size > 0 else 0

    if ratio > 0.5:
        print(f"  Bright-bg recovery: SKIPPED — would add {added} px "
              f"({100 * ratio:.0f}% of mask, exceeds 50% limit)")
        return mask

    recovered = mask.copy()
    recovered[recovery] = 255
    print(f"  Bright-bg recovery: added {added} px "
          f"({100 * ratio:.1f}% of mask)")
    return recovered


def _refine_mask_with_image(mask: np.ndarray, gray: np.ndarray,
                            dark_threshold: int = 25,
                            min_area: int = 500,
                            min_depth_px: int = 100) -> np.ndarray:
    """Carve out dark interior regions that extend deep into the tool body.

    SAM sometimes fills narrow gaps (e.g., between pliers handles) because
    the neural network treats them as foreground. This function detects truly
    dark pixels inside the mask (scanner bed showing through a gap) and
    removes them if they form a deep region extending far from the mask edge.

    Shallow dark regions (like the gap between open pliers jaws) are kept
    because the tool physically covers that area and must fit in the pocket.
    Deep dark regions (like the handle gap) are carved out because the
    handles are truly separate.

    Only affects scans on dark backgrounds (lid-open scanning).

    Args:
        mask: Binary mask (uint8, 0/255)
        gray: Grayscale original image (same dimensions as mask)
        dark_threshold: Pixels darker than this inside the mask are suspect.
            Scanner bed is ~0-10, dark metal is ~60+. Default 25.
        min_area: Minimum pixel area for a gap region to be carved out.
            Prevents noise dots from being removed. Default 500.
        min_depth_px: Minimum depth in pixels that a dark region must reach
            into the tool body to be carved out. Shallow features like jaw
            gaps (~10mm) are kept; deep features like handle gaps (~100mm)
            are carved. At 200 DPI, 100 px ~ 12.7mm. Default 100.

    Returns:
        Refined mask with deep interior gaps carved out
    """
    tool_region = mask > 128

    # Only run on dark-background scans (lid-open scanning).  On bright or
    # coloured backgrounds the dark-pixel heuristic mistakes the dark tool
    # surface for the scanner bed, punching holes in the mask.
    bg_region = ~tool_region
    if np.count_nonzero(bg_region) > 0:
        bg_median = float(np.median(gray[bg_region]))
        if bg_median > 60:
            print(f"  Interior gap refinement: skipped "
                  f"(background median {bg_median:.0f} > 60, not lid-open)")
            return mask

    # Find very dark pixels inside the tool mask (scanner bed visible through gap)
    dark_inside = tool_region & (gray < dark_threshold)
    dark_uint8 = dark_inside.astype(np.uint8) * 255

    if np.count_nonzero(dark_inside) == 0:
        return mask

    # Light dilation to bridge pixel-level noise in the dark gap
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_bridged = cv2.dilate(dark_uint8, kernel, iterations=1)

    # Find connected components of dark pixels WITHIN the mask only.
    # This separates the jaw gap from the handle gap because they are
    # disconnected dark blobs inside the tool body (not connected through
    # the external background).
    dark_interior = (dark_bridged > 0) & tool_region
    dark_interior_uint8 = dark_interior.astype(np.uint8) * 255
    n_dark_labels, dark_labels = cv2.connectedComponents(dark_interior_uint8)

    # Check which dark components actually connect to the background
    background = (~tool_region).astype(np.uint8) * 255
    combined = cv2.bitwise_or(background, dark_bridged)
    n_bg_labels, bg_labels = cv2.connectedComponents(combined)
    bg_label = bg_labels[0, 0]
    bg_connected = bg_labels == bg_label

    # Distance transform: how deep each mask pixel is from the edge
    mask_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Filter each dark interior component independently:
    # - Must connect to background (not a fully enclosed dark feature)
    # - Must be large enough (not noise)
    # - Must reach deep into the tool body (not shallow like jaw gap)
    # Tool bounding box for relative sizing
    tool_rows = np.any(tool_region, axis=1)
    tool_rmin, tool_rmax = np.where(tool_rows)[0][[0, -1]]
    tool_height = tool_rmax - tool_rmin

    gap_filtered = np.zeros_like(dark_interior_uint8)
    for label_id in range(1, n_dark_labels):
        component = dark_labels == label_id
        # Must connect to background
        if not np.any(bg_connected & component):
            continue
        area = np.count_nonzero(component)
        if area < min_area:
            continue
        # Must span a significant portion of the tool length (>20%).
        # The handle gap runs most of the tool length; jaw gaps and
        # pivot openings are localized to a small area.
        comp_rows = np.any(component, axis=1)
        comp_rmin, comp_rmax = np.where(comp_rows)[0][[0, -1]]
        comp_span = comp_rmax - comp_rmin
        span_ratio = comp_span / tool_height if tool_height > 0 else 0
        if span_ratio < 0.20:
            continue
        gap_filtered[component] = 255

    if np.count_nonzero(gap_filtered) == 0:
        return mask

    # Erode to be conservative (undo dilation + safety margin)
    gap_final = cv2.erode(gap_filtered, kernel, iterations=2)

    refined = mask.copy()
    refined[gap_final > 128] = 0

    # Smooth jagged edges from the carving. The brightness threshold can
    # eat into handle edges where shadows fall below the threshold,
    # creating a serrated boundary. A morphological close fills those
    # small notches without affecting the overall gap shape.
    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, smooth_k, iterations=1)

    carved = np.count_nonzero(mask) - np.count_nonzero(refined)
    print(f"  Interior gap refinement: carved {carved} px "
          f"(threshold={dark_threshold})")

    return refined


def _smooth_contour_coords(contour: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth the (x, y) coordinates of a closed contour.

    Uses wrap-around (circular) convolution so the start/end join is seamless.
    This removes bumps from the polygon outline without touching pixel values,
    preserving large-scale shape while eliminating shadow-induced protrusions.

    Args:
        contour: cv2 contour, shape (N, 1, 2), integer pixels
        sigma:   Gaussian sigma in pixels; radius ≈ 3*sigma

    Returns:
        Smoothed contour, same shape and dtype
    """
    pts = contour.squeeze(1).astype(np.float64)  # (N, 2)
    n = len(pts)
    if n < 3:
        return contour

    radius = int(np.ceil(sigma * 3))
    ksize = 2 * radius + 1
    x = np.arange(ksize, dtype=np.float64) - radius
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    smoothed = np.empty_like(pts)
    for col in range(2):
        data = pts[:, col]
        # Wrap-around padding for a closed contour
        padded = np.concatenate([data[-radius:], data, data[:radius]])
        smoothed[:, col] = np.convolve(padded, kernel, mode='valid')

    return smoothed.astype(np.int32).reshape(-1, 1, 2)


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes in a binary mask.

    SAM masks for dark tools may have internal holes (jaw mechanisms,
    reflective surfaces) connected to the exterior through hairline cracks.
    A simple flood-fill misses these because they technically touch the
    border.

    Strategy:
    1. Close the mask with a large kernel to seal thin boundary cracks.
       The kernel (25px ~ 3.2mm at 200 DPI) seals hairline gaps but is
       far too small to bridge the handle gap on pliers (~30-50mm).
    2. Find background components in the sealed mask.  Any component NOT
       touching the image border is an internal hole.
    3. Fill those hole regions in the original mask so the original
       boundary is preserved (the close is only used for hole detection).
    """
    # Seal thin boundary cracks for hole detection
    seal_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    sealed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, seal_k)

    inv = cv2.bitwise_not(sealed)
    n_labels, labels = cv2.connectedComponents(inv)

    h, w = labels.shape
    border_labels = set()
    border_labels.update(labels[0, :].tolist())       # top row
    border_labels.update(labels[h - 1, :].tolist())   # bottom row
    border_labels.update(labels[:, 0].tolist())        # left col
    border_labels.update(labels[:, w - 1].tolist())    # right col

    # Build mask of all internal holes (not touching border in sealed mask)
    internal = np.zeros(mask.shape, dtype=bool)
    for label_id in range(n_labels):
        if label_id not in border_labels:
            internal |= (labels == label_id)

    # Fill internal holes in the original mask (preserves original boundary)
    filled = mask.copy()
    holes_filled = int(np.count_nonzero(internal & (mask == 0)))
    filled[internal] = 255

    if holes_filled > 0:
        print(f"  Filled {holes_filled} hole pixels in mask interior")

    return filled


def erode_mask_mm(mask: np.ndarray, amount_mm: float, dpi: float) -> np.ndarray:
    """Shrink a binary mask inward by amount_mm on every edge.

    Counters SAM2's tendency to include soft shadow/out-of-focus pixels at
    tool edges — on phone photos this adds ~0.3-1mm per edge, making
    handles read noticeably wider than the physical part. Applied once
    between segmentation and cleanup.
    """
    if amount_mm <= 0:
        return mask
    radius_px = max(1, round(amount_mm * dpi / 25.4))
    ksize = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask, kernel, iterations=1)


def _mm_to_odd_px(mm_value: float, dpi: float) -> int:
    """Convert mm to an odd pixel count. Returns 0 when mm_value <= 0.

    cv2 structuring-element kernels must be odd-sized so the kernel has a
    well-defined center pixel.
    """
    if mm_value <= 0:
        return 0
    px = max(1, round(mm_value * dpi / 25.4))
    if px % 2 == 0:
        px += 1
    return px


def _mm_to_px(mm_value: float, dpi: float) -> int:
    """Convert mm to a non-negative integer pixel count."""
    if mm_value <= 0:
        return 0
    return max(1, round(mm_value * dpi / 25.4))


def cleanup_mask(mask: np.ndarray, dpi: float,
                 kernel_mm: float = 0.9,
                 smooth_radius_mm: float = 0.9,
                 shadow_kernel_mm: float = 0.0,
                 contour_smooth_sigma_mm: float = 0.6,
                 notch_fill_mm: float = 2.4) -> np.ndarray:
    """Clean up the binary mask with morphological operations.

    All feature sizes are expressed in mm and converted to pixels using
    ``dpi`` so cleanup behaves consistently whether the input is a 200 DPI
    scanner or a ~130 DPI phone photo.

    Pipeline:
    0. Hole fill: fill internal background regions enclosed by foreground
    1. Fine close: fill texture gaps (jaw teeth, knurling)
    2. Open: remove noise specks
    3. Contour fill: keep largest external contour only (all boundary points)
    3.25. Contour-coord smoothing: Gaussian-smooth the polygon vertices to
          eliminate bumps from 3D shadows / pivot geometry while keeping the
          large-scale tool shape intact
    3.5. Notch fill: morphological close with medium kernel — fills inward
         concavities (shadow notches at pivot/slider mechanisms)
    3.75. Shadow suppression: open with larger kernel to remove outward
          protrusions caused by scanner shadows
    4. Smooth: Gaussian blur for clean 3D-printable outlines
    """
    kernel_px = _mm_to_odd_px(kernel_mm, dpi)
    smooth_radius_px = _mm_to_px(smooth_radius_mm, dpi)
    shadow_kernel_px = _mm_to_odd_px(shadow_kernel_mm, dpi)
    notch_fill_px = _mm_to_odd_px(notch_fill_mm, dpi)
    contour_sigma_px = contour_smooth_sigma_mm * dpi / 25.4

    # Step 0: Fill internal holes — enclosed background regions become
    # foreground.  Gaps connected to the image border (e.g. handle gap) are
    # kept.  This prevents noisy masks from fragmenting during the
    # morphological open in step 2.
    mask = _fill_mask_holes(mask)

    if kernel_px >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
        # Step 1: Close small gaps — merge fine textures (knurling, jaw teeth)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Step 2: Remove noise specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 3: Keep only the largest external contour.
    # Use CHAIN_APPROX_NONE so every boundary pixel is present — needed for
    # smooth Gaussian convolution in step 3.25.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        largest = max(contours, key=cv2.contourArea)

        # Step 3.25: Smooth contour coordinates to remove localised outward bumps.
        if contour_sigma_px > 0.5:
            largest = _smooth_contour_coords(largest, contour_sigma_px)

        mask_clean = np.zeros_like(mask)
        cv2.drawContours(mask_clean, [largest], -1, 255, cv2.FILLED)
        mask = mask_clean

    # Step 3.5: Notch fill — morphological close with a medium kernel.
    # Dilate then erode: fills inward concavities caused by shadows at pivot
    # slots / adjustment mechanisms. The handle gap (much wider than the kernel)
    # is unaffected; small shadow notches are bridged closed.
    if notch_fill_px >= 3:
        notch_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (notch_fill_px, notch_fill_px)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, notch_k, iterations=1)

    # Step 3.75: Shadow suppression — open with a larger kernel to remove
    # outward protrusions caused by scanner shadows along handle edges.
    if shadow_kernel_px >= 3:
        shadow_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (shadow_kernel_px, shadow_kernel_px)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, shadow_k, iterations=1)

    # Step 4: Smooth edges for clean, 3D-printable outlines
    if smooth_radius_px >= 1:
        blur_size = smooth_radius_px * 2 + 1  # must be odd
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    print(f"  Cleaned mask: tool pixels: {np.count_nonzero(mask)}")
    return mask


def straighten_mask(mask: np.ndarray, angle_threshold: float = 45.0) -> np.ndarray:
    """Auto-straighten a tool mask by snapping the principal axis to 0° or 90°.

    Uses PCA on the largest contour to find the dominant orientation, computes
    the small rotation needed to align to the nearest cardinal direction, and
    rotates the mask.  Only applies corrections smaller than *angle_threshold*.

    Args:
        mask: Binary mask (uint8, 0 or 255)
        angle_threshold: Maximum correction in degrees (default 45). Set to 0
            to disable straightening entirely. Max useful value is 45 (the
            largest possible snap-to-90° correction).

    Returns:
        Straightened binary mask (may be larger than input to avoid clipping)
    """
    if angle_threshold <= 0:
        return mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask

    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze(1).astype(np.float64)  # (N, 2)

    # PCA — same method used by compute_finger_slot
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # Angle of principal axis in image coords (y-down).
    # Normalize to [0, 180) — the axis is undirected.
    angle = np.degrees(np.arctan2(principal[1], principal[0])) % 180

    # Snap to nearest 90° increment (0, 90, or 180)
    snap_angle = round(angle / 90.0) * 90.0
    correction = angle - snap_angle  # rotation to apply via getRotationMatrix2D

    if abs(correction) > angle_threshold or abs(correction) < 0.5:
        if abs(correction) >= 0.5:
            print(f"  Straighten: skipped ({correction:+.1f}° exceeds "
                  f"{angle_threshold}° threshold)")
        return mask

    # Rotate the mask, expanding the canvas so nothing is clipped
    h, w = mask.shape[:2]
    center = (w / 2.0, h / 2.0)

    rad = np.radians(abs(correction))
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    new_w = int(np.ceil(h * sin_a + w * cos_a))
    new_h = int(np.ceil(h * cos_a + w * sin_a))

    M = cv2.getRotationMatrix2D(center, correction, 1.0)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rotated = cv2.warpAffine(mask, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR, borderValue=0)
    _, rotated = cv2.threshold(rotated, 128, 255, cv2.THRESH_BINARY)

    print(f"  Straightened: {correction:+.1f}° correction "
          f"(principal axis was {angle:.1f}° from horizontal)")
    return rotated


def vectorize_mask(mask: np.ndarray, alphamax: float = 1.2,
                   turdsize: int = 15, opttolerance: float = 0.2) -> potrace.Path:
    """Convert binary mask to vector paths using potrace.

    Args:
        mask: Binary mask (uint8, 0 or 255)
        alphamax: Corner detection threshold (0=all corners, 1.34=all curves)
        turdsize: Minimum feature size in pixels (removes small artifacts)
        opttolerance: Curve optimization tolerance

    Returns:
        potrace.Path object with Bezier curves and line segments
    """
    # potrace expects a 2D boolean array where True = foreground
    bmp_data = (mask > 128).astype(np.bool_)

    # Create bitmap and trace
    bmp = potrace.Bitmap(bmp_data)
    path = bmp.trace(
        turdsize=turdsize,
        alphamax=alphamax,
        opticurve=True,
        opttolerance=opttolerance,
    )

    # Count segments
    n_curves = 0
    n_corners = 0
    for curve in path:
        for segment in curve:
            if segment.is_corner:
                n_corners += 1
            else:
                n_curves += 1

    print(f"  Vectorization: {n_curves} Bezier curves, {n_corners} corners")
    return path


def trace_from_mask(
    raw_mask: np.ndarray,
    stem: str,
    dpi: int = 200,
    clearance_mm: float = 0.0,
    tolerance_mm: float = 0.0,
    axial_tolerance_mm: float = 0.0,
    alphamax: float = 1.2,
    turdsize: int = 50,
    opttolerance: float = 2.0,
    kernel_mm: float = 0.9,
    smooth_radius_mm: float = 0.9,
    shadow_kernel_mm: float = 0.0,
    contour_smooth_sigma_mm: float = 0.6,
    notch_fill_mm: float = 2.4,
    straighten_threshold: float = 45.0,
    output_dir: str = "generated",
    tool_height_mm: float = 0.0,
    phone_height_mm: float = 0.0,
    finger_slots: bool = True,
) -> dict:
    """Run cleanup → straighten → vectorize → export on a pre-segmented mask.

    This is the core pipeline without the segmentation step, allowing
    the feedback loop to re-run with different cleanup parameters without
    re-running the expensive neural network segmentation.

    Args:
        raw_mask: Binary mask (uint8, 0 or 255) from segment_tool()
        stem: Base filename for outputs (e.g. 'cobra_350')
        dpi: Scanner/photo DPI (for mm-to-pixel conversion)
        clearance_mm: Offset to add around the trace for printing tolerance
        tolerance_mm: Additional offset beyond clearance for smoothed outer perimeter
        alphamax: potrace corner sensitivity
        turdsize: potrace minimum feature size
        opttolerance: potrace curve optimization tolerance
        kernel_mm: Morphological kernel size (mm) for texture/noise cleanup
        smooth_radius_mm: Gaussian blur radius (mm) for edge smoothing
        shadow_kernel_mm: Kernel size (mm) for shadow suppression open (0 disables)
        contour_smooth_sigma_mm: Gaussian sigma (mm) for contour smoothing
        notch_fill_mm: Kernel size (mm) for notch-fill close
        straighten_threshold: Max degrees to auto-straighten (0 to disable)
        output_dir: Output directory

    Returns:
        Dict with paths to output files and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = 25.4 / dpi

    # Parallax compensation: a tool of thickness tool_height_mm photographed
    # from height phone_height_mm appears larger than reality because its top
    # surface sits closer to the camera than the marker plane. The homography
    # corrects perspective on the marker plane only; objects above z=0 still
    # get inflated by H/(H-z). For a flat-bottomed tool with vertical-ish
    # sides (most hand tools — screwdrivers, pliers, hammers, wrenches), the
    # silhouette as seen from overhead is bounded by the top face: the bottom
    # edge is hidden behind it, so the visible outline tracks z=tool_height,
    # not the mid-height. Using half-height under-compensates by ~2-3% per
    # axis, which is invisible on short dimensions but several mm on long
    # ones. Shrink mm-per-pixel by the inverse so every downstream polygon
    # (inner, tolerance, slot) lands at true mm.
    if phone_height_mm > 0 and tool_height_mm > 0:
        z = tool_height_mm
        if z < phone_height_mm:
            parallax_factor = (phone_height_mm - z) / phone_height_mm
            scale *= parallax_factor
            print(f"  Parallax compensation: phone={phone_height_mm:.0f}mm, "
                  f"tool={tool_height_mm:.1f}mm, factor={parallax_factor:.4f} "
                  f"({(1 - parallax_factor) * 100:.1f}% shrink)")

    if abs(tolerance_mm) > 0.001:
        sign = "expand" if tolerance_mm > 0 else "shrink"
        print(f"  Tolerance: {sign} pocket by {abs(tolerance_mm):.2f}mm "
              f"(written to TOLERANCE layer)")

    # Cleanup
    print("  Mask cleanup...")
    mask = cleanup_mask(raw_mask.copy(), dpi=dpi,
                        kernel_mm=kernel_mm,
                        smooth_radius_mm=smooth_radius_mm,
                        shadow_kernel_mm=shadow_kernel_mm,
                        contour_smooth_sigma_mm=contour_smooth_sigma_mm,
                        notch_fill_mm=notch_fill_mm)

    # Auto-straighten
    if straighten_threshold > 0:
        print("  Auto-straighten...")
        mask = straighten_mask(mask, angle_threshold=straighten_threshold)

    # Save mask
    mask_path = output_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"  Saved mask: {mask_path}")

    # Vectorize
    print("  Vectorization (potrace)...")
    path = vectorize_mask(mask, alphamax=alphamax, turdsize=turdsize,
                          opttolerance=opttolerance)

    # Finger slot
    if finger_slots:
        print("  Finger slot placement...")
        slot_polygon = compute_finger_slot(path, scale, clearance_mm=clearance_mm,
                                           img_shape=mask.shape)
    else:
        print("  Finger slot disabled (--slots false)")
        slot_polygon = None

    # Export SVG
    print("  Export SVG...")
    svg_path = output_dir / f"{stem}_trace.svg"
    bbox = potrace_to_svg(path, str(svg_path), scale=scale,
                          clearance_mm=clearance_mm, tolerance_mm=tolerance_mm,
                          img_shape=mask.shape, slot_polygon=slot_polygon)
    print(f"  Saved SVG: {svg_path}")
    print(f"  Bounding box: {bbox['width_mm']:.1f} x {bbox['height_mm']:.1f} mm")

    # Export DXF
    print("  Export DXF...")
    dxf_path = output_dir / f"{stem}_trace.dxf"
    potrace_to_dxf(path, str(dxf_path), scale=scale,
                   clearance_mm=clearance_mm, tolerance_mm=tolerance_mm,
                   axial_tolerance_mm=axial_tolerance_mm,
                   img_shape=mask.shape, slot_polygon=slot_polygon)
    print(f"  Saved DXF: {dxf_path}")

    print(f"  Done! Tool dimensions: {bbox['width_mm']:.1f} x {bbox['height_mm']:.1f} mm")

    return {
        "svg_path": str(svg_path),
        "dxf_path": str(dxf_path),
        "mask_path": str(mask_path),
        "width_mm": bbox["width_mm"],
        "height_mm": bbox["height_mm"],
        "scale": scale,
        "dpi": dpi,
        "clearance_mm": clearance_mm,
        "tolerance_mm": tolerance_mm,
    }


def trace_tool(
    image_path: str,
    dpi: int = 200,
    clearance_mm: float = 0.0,
    tolerance_mm: float = 0.0,
    alphamax: float = 1.2,
    turdsize: int = 50,
    opttolerance: float = 2.0,
    kernel_mm: float = 0.9,
    smooth_radius_mm: float = 0.9,
    shadow_kernel_mm: float = 0.0,
    contour_smooth_sigma_mm: float = 0.6,
    notch_fill_mm: float = 2.4,
    mask_erode_mm: float = 0.3,
    straighten_threshold: float = 45.0,
    output_dir: str = None,
    sam_model: str = "sam2.1_l.pt",
) -> dict:
    """Full pipeline: scanner image → SVG + DXF trace.

    Runs SAM2 segmentation then delegates to trace_from_mask() for
    cleanup, vectorization, and export.

    Args:
        image_path: Path to flatbed scanner image
        dpi: Scanner/photo DPI (for mm-to-pixel conversion)
        clearance_mm: Offset to add around the trace for printing tolerance
        tolerance_mm: Additional offset beyond clearance for a smoothed outer perimeter
        alphamax: potrace corner sensitivity
        turdsize: potrace minimum feature size
        opttolerance: potrace curve optimization tolerance
        kernel_mm: Morphological kernel size (mm) for texture/noise cleanup
        smooth_radius_mm: Gaussian blur radius (mm) for edge smoothing
        shadow_kernel_mm: Kernel size (mm) for shadow suppression open (0 disables)
        contour_smooth_sigma_mm: Gaussian sigma (mm) for contour smoothing
        notch_fill_mm: Kernel size (mm) for notch-fill close
        mask_erode_mm: Post-segmentation inward erosion (mm) to counter SAM's
            shadow-halo bias on phone photos (default 0.3)
        straighten_threshold: Max degrees to auto-straighten (0 to disable)
        output_dir: Output directory (defaults to generated/<stem>)
        sam_model: SAM2 model weight name (default: sam2.1_l.pt)

    Returns:
        Dict with paths to output files and metadata
    """
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = Path("generated") / image_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    print(f"Processing: {image_path.name}")
    print(f"  DPI: {dpi}, scale: {25.4 / dpi:.4f} mm/pixel")

    # Step 1: Segmentation
    print("Step 1: Segmentation...")
    raw_mask = segment_tool(str(image_path), sam_model=sam_model)

    # Save raw mask for comparison / feedback loop
    raw_mask_path = output_dir / f"{stem}_raw_mask.png"
    cv2.imwrite(str(raw_mask_path), raw_mask)

    # Post-SAM erosion: counter shadow-halo / soft-edge bias
    if mask_erode_mm > 0:
        print(f"  Post-SAM erosion: {mask_erode_mm:.2f}mm")
        raw_mask = erode_mask_mm(raw_mask, mask_erode_mm, dpi)

    # Steps 2-6: Cleanup → Straighten → Vectorize → Export
    print("Steps 2-6: Cleanup -> Export...")
    result = trace_from_mask(
        raw_mask, stem, dpi=dpi,
        clearance_mm=clearance_mm, tolerance_mm=tolerance_mm,
        alphamax=alphamax, turdsize=turdsize, opttolerance=opttolerance,
        kernel_mm=kernel_mm, smooth_radius_mm=smooth_radius_mm,
        shadow_kernel_mm=shadow_kernel_mm,
        contour_smooth_sigma_mm=contour_smooth_sigma_mm,
        notch_fill_mm=notch_fill_mm,
        straighten_threshold=straighten_threshold,
        output_dir=str(output_dir),
    )
    result["raw_mask_path"] = str(raw_mask_path)
    return result


def _collect_images(args_images: list[str]) -> list[Path]:
    """Resolve image arguments into a list of image paths.

    Accepts explicit file paths or the keyword ``all_images`` which expands to
    every .png/.jpg/.jpeg in the current working directory (excluding
    subdirectories and generated *_mask.png files).
    """
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".heif"}

    paths: list[Path] = []
    for arg in args_images:
        if arg.lower() == "all_images":
            for p in sorted(Path.cwd().iterdir()):
                if (
                    p.is_file()
                    and p.suffix.lower() in IMAGE_EXTENSIONS
                    and not p.name.endswith("_mask.png")
                ):
                    paths.append(p)
        else:
            paths.append(Path(arg))

    if not paths:
        print("No images found.")
        sys.exit(1)

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate SVG/DXF trace of a tool from a flatbed scanner image"
    )
    parser.add_argument("images", nargs="+",
                        help="Image paths, or 'all_images' to process every "
                             "PNG/JPG in the current directory")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Scanner DPI (default: 200)")
    parser.add_argument("--clearance", type=float, default=0.0,
                        help="Clearance offset in mm applied to inner outline (default: 0)")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="Tolerance perimeter offset in mm beyond inner outline (default: 0). "
                             "Positive expands the pocket; negative shrinks it.")
    parser.add_argument("--alphamax", type=float, default=1.2,
                        help="potrace corner sensitivity 0-1.34 (default: 1.2)")
    parser.add_argument("--turdsize", type=int, default=50,
                        help="potrace min feature size in pixels (default: 50)")
    parser.add_argument("--opttolerance", type=float, default=2.0,
                        help="potrace curve optimization tolerance (default: 0.8)")
    parser.add_argument("--kernel-mm", type=float, default=0.9,
                        help="Morphological kernel size for mask cleanup in mm "
                             "(default: 0.9). Set to 0 to disable.")
    parser.add_argument("--smooth-radius-mm", type=float, default=0.9,
                        help="Gaussian blur radius for edge smoothing in mm (default: 0.9)")
    parser.add_argument("--contour-smooth-mm", type=float, default=0.6,
                        help="Gaussian sigma for contour-coordinate smoothing in mm "
                             "(default: 0.6). Removes localised bumps from shadows / "
                             "pivot geometry. Set to 0 to disable.")
    parser.add_argument("--notch-fill-mm", type=float, default=2.4,
                        help="Kernel size for notch-fill close in mm (default: 2.4). "
                             "Fills inward shadow concavities (pivot slots, adjustment "
                             "mechanisms). Set to 0 to disable.")
    parser.add_argument("--shadow-kernel-mm", type=float, default=0.0,
                        help="Kernel size for shadow suppression in mm (default: 0.0, "
                             "disabled). Removes thin protrusions from shadows.")
    parser.add_argument("--mask-erode-mm", type=float, default=0.3,
                        help="Post-segmentation mask erosion in mm (default: 0.3). "
                             "Counters SAM's tendency to include shadow halos at edges. "
                             "Set to 0 to disable.")
    parser.add_argument("--straighten-threshold", type=float, default=45.0,
                        help="Max degrees to auto-straighten scanned tools (default: 45). "
                             "Snaps the tool's principal axis to nearest 90°. "
                             "Set to 0 to disable.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto per image)")
    parser.add_argument("--sam-model", type=str, default="sam2.1_l.pt",
                        help="SAM2 model weights (default: sam2.1_l.pt). "
                             "Options: sam2.1_t.pt, sam2.1_s.pt, sam2.1_b.pt, sam2.1_l.pt")

    args = parser.parse_args()

    image_paths = _collect_images(args.images)
    print(f"Processing {len(image_paths)} image(s)...\n")

    results = []
    for image_path in image_paths:
        try:
            result = trace_tool(
                image_path=str(image_path),
                dpi=args.dpi,
                clearance_mm=args.clearance,
                tolerance_mm=args.tolerance,
                alphamax=args.alphamax,
                turdsize=args.turdsize,
                opttolerance=args.opttolerance,
                kernel_mm=args.kernel_mm,
                smooth_radius_mm=args.smooth_radius_mm,
                shadow_kernel_mm=args.shadow_kernel_mm,
                contour_smooth_sigma_mm=args.contour_smooth_mm,
                notch_fill_mm=args.notch_fill_mm,
                mask_erode_mm=args.mask_erode_mm,
                straighten_threshold=args.straighten_threshold,
                output_dir=args.output_dir,
                sam_model=args.sam_model,
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR processing {image_path.name}: {e}\n")

    print(f"\n{'='*40}")
    print(f"Processed {len(results)}/{len(image_paths)} images successfully.")
    for r in results:
        print(f"  {Path(r['svg_path']).parent.name}: "
              f"{r['width_mm']:.1f} x {r['height_mm']:.1f} mm")

    return results


if __name__ == "__main__":
    main()
