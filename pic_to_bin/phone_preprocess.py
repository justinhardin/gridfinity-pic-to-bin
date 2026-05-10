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
from typing import Optional

import cv2
import numpy as np

from pic_to_bin.phone_template import (
    ARUCO_DICT_ID,
    MARKER_QUIET_PAD_MM,
    MARKER_SIZE_MM,
    PAPER_SIZES,
    get_marker_positions,
    get_placement_zone,
)


def _mask_marker_quiet_pads(cropped: np.ndarray, paper_size: str,
                            px_per_mm: float,
                            safety_margin_mm: float = 8.0,
                            edge_band_mm: float = 2.5) -> np.ndarray:
    """Repaint the white quiet-zone pads, marker ID labels, and dashed
    placement-zone border that bleed into the cropped image on green-bg
    templates.

    Three sources of bright artefacts at the cropped edges:

    * The MARKER_QUIET_PAD_MM-wide pads behind each ArUco marker —
      protrude ~4 mm into the placement zone.
    * The "ID 0", "ID 1" … text labels printed 3 mm below each top /
      edge-mid marker, drawn WHITE on green-bg templates so they sit a
      few millimetres inside the placement zone.
    * The dashed placement-zone border itself, drawn at the zone edge.

    Strategy: paint a generous rectangle around each marker (pad +
    safety margin to cover labels and homography drift) AND a thin band
    along every cropped-image edge (covers the dashed border and any
    sliver the per-marker rectangles miss). Then refill with the median
    of the rest of the image so SAM2 sees uniform background where the
    artefacts used to be. No-op on white-bg templates — pad / label /
    border are all the same colour as the page.
    """
    h, w = cropped.shape[:2]
    pad_half_mm = MARKER_SIZE_MM / 2.0 + MARKER_QUIET_PAD_MM + safety_margin_mm
    zx0, zy0, _, _ = get_placement_zone(paper_size)

    pad_mask = np.zeros((h, w), dtype=np.uint8)
    for _mid, mcx, mcy in get_marker_positions(paper_size):
        cx0 = (mcx - pad_half_mm - zx0) * px_per_mm
        cy0 = (mcy - pad_half_mm - zy0) * px_per_mm
        cx1 = (mcx + pad_half_mm - zx0) * px_per_mm
        cy1 = (mcy + pad_half_mm - zy0) * px_per_mm
        rx0 = max(0, int(round(cx0)))
        ry0 = max(0, int(round(cy0)))
        rx1 = min(w, int(round(cx1)))
        ry1 = min(h, int(round(cy1)))
        if rx1 > rx0 and ry1 > ry0:
            pad_mask[ry0:ry1, rx0:rx1] = 255

    # Belt-and-suspenders edge band — covers the dashed placement-zone
    # border and any per-marker artefact wider than the per-marker rect.
    band_px = max(1, int(round(edge_band_mm * px_per_mm)))
    pad_mask[:band_px, :] = 255
    pad_mask[h - band_px:, :] = 255
    pad_mask[:, :band_px] = 255
    pad_mask[:, w - band_px:] = 255

    if not pad_mask.any():
        return cropped

    # Sample the bg from non-pad pixels via median (robust to whatever
    # tool happens to be in the rest of the image).
    non_pad = pad_mask == 0
    if not non_pad.any():
        return cropped
    if cropped.ndim == 3:
        bg_color = np.median(cropped[non_pad], axis=0).astype(cropped.dtype)
    else:
        bg_color = np.uint8(np.median(cropped[non_pad]))

    out = cropped.copy()
    out[pad_mask > 0] = bg_color
    pad_px = int(np.count_nonzero(pad_mask))
    print(f"  Repainted {pad_px} px of marker quiet-zone pads + edge band "
          f"with bg color {tuple(int(c) for c in np.atleast_1d(bg_color))}")
    return out


# ---------------------------------------------------------------------------
# HEIC conversion
# ---------------------------------------------------------------------------

HEIC_EXTENSIONS = {".heic", ".heif"}


def convert_heic_to_png(image_path: str | Path) -> Path:
    """Convert a HEIC/HEIF image to PNG, returning the new path.

    If the file is not HEIC/HEIF, returns the original path unchanged.
    The PNG is saved alongside the original with the same stem.

    EXIF metadata (FocalLengthIn35mmFilm in particular, which we use for
    parallax auto-calibration) is preserved on the output PNG. Most viewers
    don't surface PNG EXIF, but PIL reads it back fine.

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
    exif_bytes = img.info.get("exif")
    save_kwargs = {}
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes
    img.save(png_path, **save_kwargs)

    return png_path


# ---------------------------------------------------------------------------
# EXIF-driven camera-height estimation
# ---------------------------------------------------------------------------
#
# A photo of a planar marker grid is fundamentally scale-degenerate: without
# knowing the camera's focal length, you cannot recover absolute distance to
# the plane (any focal length / distance pair that preserves f/D produces the
# same image). EXIF gives us that focal length when present.
#
# The relevant EXIF tag is FocalLengthIn35mmFilm (0xA405), which expresses
# the lens's field of view as if it were on a 35 mm full-frame sensor
# (36 mm × 24 mm). Pixel focal length recovers as
#
#     f_px = (focal_35mm / 36 mm) × image_width_px
#
# Once we have f_px we build a pinhole K, run solvePnP against the marker
# correspondences, and read the camera height off the recovered translation.
# Without EXIF — JPGs that have been re-exported, screenshots, etc. — we
# fall back to whatever the user (or DEFAULT_PHONE_HEIGHT_MM) provided.

# Standard EXIF tag IDs (from the Exif specification).
_EXIF_TAG_FOCAL_LENGTH_35MM = 0xA405  # FocalLengthIn35mmFilm — integer mm
_EXIF_IFD_TAG = 0x8769                # ExifIFD pointer (where most lens tags live)

# Diagonal of a 35mm full-frame sensor (36 × 24 mm). FocalLengthIn35mmFilm
# is defined to match the *diagonal* field of view of a 35mm full-frame
# camera, so the conversion from f_35 to a pixel focal length scales with
# the image diagonal — not the width or height — and is invariant to the
# actual sensor's aspect ratio (4:3 phones, 3:2 DSLRs, etc.).
_FULL_FRAME_DIAG_MM = (36.0 ** 2 + 24.0 ** 2) ** 0.5  # ≈ 43.267 mm


def read_focal_length_35mm(image_path: str | Path) -> Optional[float]:
    """Return the photo's 35 mm-equivalent focal length, or None if unset.

    Reads the original file's EXIF (HEIC, JPEG, or PNG with eXIf chunk).
    Returns ``float`` mm, e.g. ``26.0`` for an iPhone main wide shot.
    """
    image_path = Path(image_path)
    try:
        from PIL import Image
        if image_path.suffix.lower() in HEIC_EXTENSIONS:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        with Image.open(image_path) as im:
            exif = im.getexif()
            if not exif:
                return None
            ifd = exif.get_ifd(_EXIF_IFD_TAG) if hasattr(exif, "get_ifd") else {}
            value = ifd.get(_EXIF_TAG_FOCAL_LENGTH_35MM)
            if value is None:
                value = exif.get(_EXIF_TAG_FOCAL_LENGTH_35MM)
            if value is None:
                return None
            return float(value)
    except Exception:
        return None


def estimate_camera_height_mm(
    ids: np.ndarray,
    corners: list[np.ndarray],
    paper_size: str,
    image_shape,
    focal_35mm: float,
) -> Optional[float]:
    """Estimate camera height above the paper from EXIF focal length + markers.

    Uses solvePnP with a pinhole K built from ``focal_35mm`` and the photo's
    pixel dimensions. Returns the camera's perpendicular distance from the
    marker plane in mm, or ``None`` if the result is implausible (suggesting
    the photo was cropped, the focal length tag was wrong, or solvePnP
    failed).

    Args:
        ids: Detected marker IDs, shape (N, 1) — same as detect_markers().
        corners: Per-marker (1, 4, 2) pixel-corner arrays.
        paper_size: Template paper size (matches compute_homography).
        image_shape: Original photo shape (h, w, ...) — pre-rectification.
        focal_35mm: 35 mm-equivalent focal length from EXIF.
    """
    flat_ids = ids.flatten().tolist()
    marker_positions = {m[0]: (m[1], m[2])
                        for m in get_marker_positions(paper_size)}

    obj_pts: list[list[float]] = []
    img_pts: list[list[float]] = []
    for i, mid in enumerate(flat_ids):
        if mid not in marker_positions:
            continue
        cx_mm, cy_mm = marker_positions[mid]
        mm_corners = _marker_corner_mm(cx_mm, cy_mm, MARKER_SIZE_MM)
        px_corners = corners[i].reshape(4, 2)
        for j in range(4):
            obj_pts.append([float(mm_corners[j, 0]), float(mm_corners[j, 1]), 0.0])
            img_pts.append([float(px_corners[j, 0]), float(px_corners[j, 1])])

    if len(obj_pts) < 4:
        return None

    obj_arr = np.array(obj_pts, dtype=np.float64)
    img_arr = np.array(img_pts, dtype=np.float64)

    h, w = image_shape[:2]
    # Pixel focal length from the 35mm-equivalent value: scale by the image
    # diagonal vs. the 35mm full-frame diagonal (43.27 mm). This works
    # regardless of orientation (portrait/landscape) and aspect ratio
    # (4:3 phone vs 3:2 DSLR) — using w/36 instead would under-estimate
    # f_px by ~50% on a portrait phone shot, since the photo's narrow
    # dimension corresponds to the sensor's short edge, not the 36 mm long
    # edge of a 35mm full-frame.
    image_diag = (float(w) ** 2 + float(h) ** 2) ** 0.5
    f_px = float(focal_35mm) * image_diag / _FULL_FRAME_DIAG_MM
    K = np.array([
        [f_px, 0.0,   w / 2.0],
        [0.0,  f_px,  h / 2.0],
        [0.0,  0.0,   1.0],
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    # Planar-scene PnP is two-fold ambiguous: rotating the camera 180° about
    # an axis in the plane gives the same projection. solvePnPGeneric+IPPE
    # returns both candidates with their reprojection errors; the
    # geometrically valid one has near-zero error (one of the two reprojects
    # the points exactly), so we just pick that. Plain cv2.solvePnP often
    # returns the mirrored solution.
    #
    # tvec[2] is the depth of the world origin in camera coordinates. For a
    # camera that's roughly perpendicular to the marker plane (typical
    # phone-overhead shots), this equals the camera-to-plane perpendicular
    # distance — exactly what we want as ``phone_height``. Mild tilt
    # (≤30°) inflates this by 1/cos(tilt) which is ≤15%; well within the
    # other sources of error in this estimate.
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj_arr, img_arr, K, dist, flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None
    if not n:
        return None

    # Pick the lowest-error candidate.
    best_idx = 0
    if errs is not None and len(errs) > 1:
        flat_errs = [float(e[0]) if hasattr(e, "__len__") else float(e) for e in errs]
        best_idx = int(np.argmin(flat_errs))
    height_mm = float(abs(tvecs[best_idx][2, 0]))

    # Plausibility clamp: phone shots range from "right above the paper"
    # (~150 mm) to "standing over a desk" (~1200 mm). Outside that range,
    # something else is wrong (EXIF focal length is for a different lens
    # than the one used, photo was cropped breaking the pixel-width
    # assumption, etc.) — better to fall back than to ship a bad number.
    if height_mm < 150.0 or height_mm > 1500.0:
        return None
    return height_mm


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

    # Read EXIF from the original (HEIC/JPG/etc.) BEFORE conversion.
    # convert_heic_to_png preserves it on the PNG output now too, but
    # going to the original is one fewer thing to depend on.
    focal_35mm_exif = read_focal_length_35mm(image_path)

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

    # Camera-height auto-estimate from EXIF (used for parallax compensation
    # when the user doesn't pass --phone-height). Only viable when the
    # FocalLengthIn35mmFilm tag survived to our copy of the file.
    camera_height_mm: Optional[float] = None
    if focal_35mm_exif is not None:
        camera_height_mm = estimate_camera_height_mm(
            ids, corners, paper_size, img.shape, focal_35mm_exif,
        )
        if camera_height_mm is not None:
            print(f"  Camera height (from EXIF f={focal_35mm_exif:.0f}mm): "
                  f"{camera_height_mm:.0f} mm")
        else:
            print(f"  EXIF focal length present ({focal_35mm_exif:.0f}mm) "
                  f"but camera-height estimate was implausible — falling "
                  f"back to user/default value.")
    else:
        print("  No FocalLengthIn35mmFilm in EXIF — using user/default "
              "phone height for parallax.")

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
    cropped = _mask_marker_quiet_pads(cropped, paper_size, px_per_mm)

    zone_w_mm = zx1 - zx0
    zone_h_mm = zy1 - zy0
    print(f"  Cropped to placement zone: {cropped.shape[1]}x{cropped.shape[0]} px "
          f"({zone_w_mm:.0f}x{zone_h_mm:.0f} mm)")

    # Save rectified image
    rectified_path = output_dir / f"{stem}_rectified.png"
    cv2.imwrite(str(rectified_path), cropped)
    print(f"  Saved: {rectified_path}")

    # Sidecar metadata: enough to reconstruct the mm↔pixel mapping later
    # without re-running the homography. The LLM overlay step (and any
    # future post-trace tooling) reads this to draw the trace polygons
    # back onto the rectified photo at the right scale.
    import json as _json
    meta_path = output_dir / f"{stem}_rectified.json"
    meta_path.write_text(_json.dumps({
        "effective_dpi": float(effective_dpi),
        "paper_size": paper_size.lower(),
        "image_width_px": int(cropped.shape[1]),
        "image_height_px": int(cropped.shape[0]),
        "placement_zone_mm": [float(zx0), float(zy0), float(zx1), float(zy1)],
        "exif_focal_length_35mm": (
            float(focal_35mm_exif) if focal_35mm_exif is not None else None
        ),
        "camera_height_mm": (
            float(camera_height_mm) if camera_height_mm is not None else None
        ),
    }, indent=2), encoding="utf-8")

    return {
        "rectified_image_path": str(rectified_path),
        "effective_dpi": effective_dpi,
        "markers_detected": n_markers,
        "detected_ids": detected_ids,
        "reprojection_error_mm": diagnostics["reprojection_error_mm"],
        "diagnostics": diagnostics,
        "exif_focal_length_35mm": focal_35mm_exif,
        "camera_height_mm": camera_height_mm,
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
