"""
Phone-photo preprocessing: detect ArUco markers, compute perspective
correction and scale, warp the image to a scanner-equivalent frontal view.

After preprocessing the rectified image can be fed directly into the
existing trace pipeline (SAM2 segmentation -> cleanup -> export).

CLI:
    preprocess-phone <image> [--paper-size a4|letter|legal] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from pic_to_bin.phone_template import (
    ARUCO_DICT_ID,
    MARKER_SIZE_MM,
    PAPER_SIZES,
    get_marker_positions,
    get_placement_zone,
)


# ---------------------------------------------------------------------------
# HEIC conversion
# ---------------------------------------------------------------------------

HEIC_EXTENSIONS = {".heic", ".heif"}


def convert_heic_to_png(image_path: str | Path) -> Path:
    """Convert a HEIC/HEIF image to PNG, returning the new path.

    If the file is not HEIC/HEIF, returns the original path unchanged.
    The PNG is saved alongside the original with the same stem.

    Args:
        image_path: Path to the image file

    Returns:
        Path to the PNG file (or original path if not HEIC)
    """
    image_path = Path(image_path)
    if image_path.suffix.lower() not in HEIC_EXTENSIONS:
        return image_path

    from pillow_heif import register_heif_opener
    from PIL import Image

    register_heif_opener()

    png_path = image_path.with_suffix(".png")
    print(f"  Converting {image_path.name} -> {png_path.name}")
    img = Image.open(image_path)
    img.save(png_path)

    return png_path


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class MarkerDetectionError(ValueError):
    """Raised when insufficient markers are detected."""

    def __init__(self, message: str, detected_count: int,
                 detected_ids: list[int]):
        super().__init__(message)
        self.detected_count = detected_count
        self.detected_ids = detected_ids


class ScaleInconsistencyError(ValueError):
    """Raised when horizontal and vertical scales differ too much."""
    pass


# ---------------------------------------------------------------------------
# Marker detection
# ---------------------------------------------------------------------------

def detect_markers(image: np.ndarray,
                   ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect ArUco markers in a phone photo.

    Uses DICT_4X4_50 with sub-pixel corner refinement and adaptive
    thresholding tuned for variable phone lighting.

    Args:
        image: BGR image (as from cv2.imread)

    Returns:
        (ids, corners) where:
            ids: shape (N, 1) array of marker IDs, or None if no markers
            corners: list of N arrays, each shape (1, 4, 2) — the 4 corner
                     pixel coordinates of each detected marker
        Returns (None, None) if no markers are found.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()

    # Adaptive thresholding for variable lighting
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 25
    params.adaptiveThreshWinSizeStep = 5

    # Sub-pixel corner refinement for accuracy
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50

    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, rejected = detector.detectMarkers(image)

    if ids is None or len(ids) == 0:
        return None, None

    # Filter to template IDs (0-7) only
    valid = ids.flatten() < 8
    if not np.any(valid):
        return None, None

    ids = ids[valid]
    corners = [corners[i] for i in range(len(valid)) if valid[i]]

    return ids, corners


# ---------------------------------------------------------------------------
# Homography computation
# ---------------------------------------------------------------------------

def _marker_corner_mm(cx_mm: float, cy_mm: float, marker_size_mm: float,
                      ) -> np.ndarray:
    """Compute the 4 corner positions (mm) of a marker given its center.

    Corner order matches ArUco convention: top-left, top-right,
    bottom-right, bottom-left.

    Returns:
        (4, 2) array of (x_mm, y_mm)
    """
    half = marker_size_mm / 2.0
    return np.array([
        [cx_mm - half, cy_mm - half],  # top-left
        [cx_mm + half, cy_mm - half],  # top-right
        [cx_mm + half, cy_mm + half],  # bottom-right
        [cx_mm - half, cy_mm + half],  # bottom-left
    ], dtype=np.float64)


def compute_homography(
    ids: np.ndarray,
    corners: list[np.ndarray],
    paper_size: str = "legal",
) -> tuple[np.ndarray, float, dict]:
    """Compute perspective transform and effective DPI from detected markers.

    Maps each marker's 4 detected pixel corners to their known mm positions
    on the template.  Uses RANSAC for outlier rejection.

    Args:
        ids: Detected marker IDs, shape (N, 1)
        corners: Detected corners, list of N arrays each (1, 4, 2)
        paper_size: Template paper size

    Returns:
        (H, effective_dpi, diagnostics) where:
            H: 3x3 homography matrix (pixel -> mm)
            effective_dpi: computed DPI (dots per inch)
            diagnostics: dict with reprojection_error, h_scale, v_scale, etc.

    Raises:
        MarkerDetectionError: If fewer than 3 markers are available
        ScaleInconsistencyError: If horizontal/vertical scales differ by >5%
    """
    flat_ids = ids.flatten().tolist()

    if len(flat_ids) < 3:
        raise MarkerDetectionError(
            f"Only {len(flat_ids)} marker(s) detected (need at least 3). "
            f"Ensure the template is fully visible, well-lit, and not too "
            f"far from the camera.",
            detected_count=len(flat_ids),
            detected_ids=flat_ids,
        )

    # Build point correspondences: pixel corners -> mm corners
    marker_positions = {m[0]: (m[1], m[2])
                        for m in get_marker_positions(paper_size)}

    src_pts = []  # pixel coordinates
    dst_pts = []  # mm coordinates

    for i, mid in enumerate(flat_ids):
        if mid not in marker_positions:
            continue
        cx_mm, cy_mm = marker_positions[mid]
        mm_corners = _marker_corner_mm(cx_mm, cy_mm, MARKER_SIZE_MM)
        px_corners = corners[i].reshape(4, 2)

        for j in range(4):
            src_pts.append(px_corners[j])
            dst_pts.append(mm_corners[j])

    src_pts = np.array(src_pts, dtype=np.float64)
    dst_pts = np.array(dst_pts, dtype=np.float64)

    # Compute homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    if H is None:
        raise MarkerDetectionError(
            "Could not compute homography from detected markers. "
            "The markers may be too distorted or the photo too blurry.",
            detected_count=len(flat_ids),
            detected_ids=flat_ids,
        )

    # Reprojection error
    inliers = mask.ravel() == 1
    if np.any(inliers):
        projected = cv2.perspectiveTransform(
            src_pts[inliers].reshape(-1, 1, 2), H)
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - dst_pts[inliers], axis=1)
        reproj_error = float(np.mean(errors))
    else:
        reproj_error = float("inf")

    # Compute effective scale (mm per pixel) and DPI
    # Use the inverse homography: mm -> pixel, measure how many pixels
    # per mm in both axes
    H_inv = np.linalg.inv(H)

    # Horizontal scale: measure pixels between two horizontal markers
    pw_mm, ph_mm = PAPER_SIZES[paper_size.lower()]
    # Sample a 1mm horizontal segment at the template center
    center_mm = np.array([[pw_mm / 2, ph_mm / 2]], dtype=np.float64)
    right_mm = np.array([[pw_mm / 2 + 1.0, ph_mm / 2]], dtype=np.float64)
    down_mm = np.array([[pw_mm / 2, ph_mm / 2 + 1.0]], dtype=np.float64)

    center_px = cv2.perspectiveTransform(
        center_mm.reshape(-1, 1, 2), H_inv).reshape(2)
    right_px = cv2.perspectiveTransform(
        right_mm.reshape(-1, 1, 2), H_inv).reshape(2)
    down_px = cv2.perspectiveTransform(
        down_mm.reshape(-1, 1, 2), H_inv).reshape(2)

    h_px_per_mm = float(np.linalg.norm(right_px - center_px))
    v_px_per_mm = float(np.linalg.norm(down_px - center_px))
    avg_px_per_mm = (h_px_per_mm + v_px_per_mm) / 2.0

    effective_dpi = avg_px_per_mm * 25.4

    # Scale consistency check
    scale_diff = abs(h_px_per_mm - v_px_per_mm) / avg_px_per_mm
    if scale_diff > 0.05:
        raise ScaleInconsistencyError(
            f"Horizontal and vertical scales differ by {scale_diff * 100:.1f}%. "
            f"The template may not have been printed at 100% scale "
            f"(no fit-to-page). H: {h_px_per_mm:.2f} px/mm, "
            f"V: {v_px_per_mm:.2f} px/mm.")

    if scale_diff > 0.02:
        print(f"  WARNING: H/V scale difference {scale_diff * 100:.1f}% "
              f"(H: {h_px_per_mm:.2f}, V: {v_px_per_mm:.2f} px/mm)")

    diagnostics = {
        "reprojection_error_mm": reproj_error,
        "h_px_per_mm": h_px_per_mm,
        "v_px_per_mm": v_px_per_mm,
        "scale_diff_pct": scale_diff * 100,
        "inlier_count": int(np.sum(inliers)),
        "total_points": len(src_pts),
    }

    return H, effective_dpi, diagnostics


# ---------------------------------------------------------------------------
# Image warping
# ---------------------------------------------------------------------------

def warp_image(image: np.ndarray, H: np.ndarray,
               output_size_mm: tuple[float, float],
               dpi: float) -> np.ndarray:
    """Warp a phone image to a frontal, rectified view.

    The output image is at the given DPI, with coordinates corresponding
    to mm positions on the template.  This makes it equivalent to a
    flatbed scanner image.

    Args:
        image: Input BGR image
        H: 3x3 homography matrix (pixel -> mm)
        output_size_mm: (width_mm, height_mm) of the output region
        dpi: Target DPI for the output image

    Returns:
        Rectified BGR image
    """
    px_per_mm = dpi / 25.4
    out_w = int(round(output_size_mm[0] * px_per_mm))
    out_h = int(round(output_size_mm[1] * px_per_mm))

    # Build a combined transform: pixel -> mm -> output_pixel
    # H maps input pixels to mm.  We need mm -> output pixels.
    # output_px = mm * px_per_mm  (simple scaling, no offset for now)
    S = np.array([
        [px_per_mm, 0,          0],
        [0,         px_per_mm,  0],
        [0,         0,          1],
    ], dtype=np.float64)

    H_combined = S @ H  # input pixel -> mm -> output pixel

    rectified = cv2.warpPerspective(
        image, H_combined, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),  # white border (matches template bg)
    )

    return rectified


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def preprocess_phone_image(
    image_path: str,
    paper_size: str = "legal",
    output_dir: str | None = None,
) -> dict:
    """Full phone-photo preprocessing: detect markers, correct perspective,
    compute scale, crop to placement zone.

    Args:
        image_path: Path to phone photo
        paper_size: Template paper size used
        output_dir: Directory to save rectified image (default: generated/<stem>)

    Returns:
        Dict with:
            rectified_image_path: path to saved rectified image
            effective_dpi: computed DPI (float)
            markers_detected: number of markers found
            detected_ids: list of detected marker IDs
            reprojection_error_mm: mean reprojection error in mm
            diagnostics: full diagnostics dict from compute_homography
    """
    image_path = Path(image_path)

    # Convert HEIC/HEIF to PNG if needed
    image_path = convert_heic_to_png(image_path)

    if output_dir is None:
        output_dir = Path("generated") / image_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    print(f"Phone preprocessing: {image_path.name}")

    # Load image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.stat().st_size == 0:
        raise ValueError(
            f"Image file is empty (0 bytes): {image_path}. "
            f"Re-transfer the photo from your phone.")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(
            f"Could not decode image: {image_path}. "
            f"The file may be corrupt or in an unsupported format.")
    print(f"  Image: {img.shape[1]}x{img.shape[0]} px")

    # Detect markers
    print("  Detecting ArUco markers...")
    ids, corners = detect_markers(img)

    if ids is None:
        raise MarkerDetectionError(
            "No ArUco markers detected in the image. "
            "Ensure the template is visible and well-lit.",
            detected_count=0,
            detected_ids=[],
        )

    n_markers = len(ids)
    detected_ids = sorted(ids.flatten().tolist())
    print(f"  Detected {n_markers}/8 markers: {detected_ids}")

    if n_markers < 8:
        print(f"  WARNING: Only {n_markers}/8 markers found. "
              f"Missing: {sorted(set(range(8)) - set(detected_ids))}")

    # Compute homography and scale
    print("  Computing perspective correction...")
    H, effective_dpi, diagnostics = compute_homography(
        ids, corners, paper_size)

    print(f"  Effective DPI: {effective_dpi:.0f}")
    print(f"  Scale: {diagnostics['h_px_per_mm']:.2f} x "
          f"{diagnostics['v_px_per_mm']:.2f} px/mm")
    print(f"  Reprojection error: {diagnostics['reprojection_error_mm']:.3f} mm")

    if effective_dpi < 100:
        print(f"  WARNING: Low effective DPI ({effective_dpi:.0f}). "
              f"Hold the camera closer or use a higher resolution.")

    # Warp to frontal view — output covers the full paper
    pw_mm, ph_mm = PAPER_SIZES[paper_size.lower()]
    print(f"  Warping to frontal view at {effective_dpi:.0f} DPI...")
    rectified = warp_image(img, H, (pw_mm, ph_mm), effective_dpi)

    # Crop to placement zone
    zone = get_placement_zone(paper_size)
    zx0, zy0, zx1, zy1 = zone
    px_per_mm = effective_dpi / 25.4
    crop_x0 = int(round(zx0 * px_per_mm))
    crop_y0 = int(round(zy0 * px_per_mm))
    crop_x1 = int(round(zx1 * px_per_mm))
    crop_y1 = int(round(zy1 * px_per_mm))
    cropped = rectified[crop_y0:crop_y1, crop_x0:crop_x1]

    zone_w_mm = zx1 - zx0
    zone_h_mm = zy1 - zy0
    print(f"  Cropped to placement zone: {cropped.shape[1]}x{cropped.shape[0]} px "
          f"({zone_w_mm:.0f}x{zone_h_mm:.0f} mm)")

    # Save rectified image
    rectified_path = output_dir / f"{stem}_rectified.png"
    cv2.imwrite(str(rectified_path), cropped)
    print(f"  Saved: {rectified_path}")

    return {
        "rectified_image_path": str(rectified_path),
        "effective_dpi": effective_dpi,
        "markers_detected": n_markers,
        "detected_ids": detected_ids,
        "reprojection_error_mm": diagnostics["reprojection_error_mm"],
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a phone photo with ArUco template: "
                    "detect markers, correct perspective, compute scale")
    parser.add_argument(
        "image", help="Phone photo with ArUco template visible")
    parser.add_argument(
        "--paper-size", choices=["a4", "letter", "legal"], default="legal",
        help="Template paper size (default: legal)")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: generated/<stem>)")

    args = parser.parse_args()

    try:
        result = preprocess_phone_image(
            args.image, paper_size=args.paper_size,
            output_dir=args.output_dir)
        print(f"\nDone! Effective DPI: {result['effective_dpi']:.0f}, "
              f"markers: {result['markers_detected']}/8")
    except (MarkerDetectionError, ScaleInconsistencyError) as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
