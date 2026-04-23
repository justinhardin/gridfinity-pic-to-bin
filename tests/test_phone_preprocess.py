"""Tests for phone_preprocess.py — ArUco detection, homography, warping."""

import numpy as np
import cv2
import pytest

from pic_to_bin.phone_preprocess import (
    detect_markers,
    compute_homography,
    warp_image,
    preprocess_phone_image,
    MarkerDetectionError,
    ScaleInconsistencyError,
    _marker_corner_mm,
)
from pic_to_bin.phone_template import (
    get_marker_positions,
    get_placement_zone,
    PAPER_SIZES,
    MARKER_SIZE_MM,
    ARUCO_DICT_ID,
    _generate_marker_image,
)


# ---------------------------------------------------------------------------
# Helpers — render synthetic "phone photos" of the template
# ---------------------------------------------------------------------------

def _render_template_bgr(paper_size: str = "letter",
                         dpi: int = 150) -> np.ndarray:
    """Render the marker template to a BGR image at a known DPI.

    This is a "perfect" frontal photo — no perspective distortion.
    """
    pw_mm, ph_mm = PAPER_SIZES[paper_size]
    px_per_mm = dpi / 25.4
    w = int(round(pw_mm * px_per_mm))
    h = int(round(ph_mm * px_per_mm))

    # White canvas
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    half = MARKER_SIZE_MM / 2.0
    for mid, cx_mm, cy_mm in get_marker_positions(paper_size):
        marker = _generate_marker_image(mid, 200)
        # Convert to BGR
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        # Destination rectangle in pixels
        x0 = int(round((cx_mm - half) * px_per_mm))
        y0 = int(round((cy_mm - half) * px_per_mm))
        x1 = int(round((cx_mm + half) * px_per_mm))
        y1 = int(round((cy_mm + half) * px_per_mm))
        mw, mh = x1 - x0, y1 - y0
        if mw > 0 and mh > 0:
            resized = cv2.resize(marker_bgr, (mw, mh),
                                 interpolation=cv2.INTER_NEAREST)
            img[y0:y1, x0:x1] = resized

    return img


def _apply_perspective(img: np.ndarray,
                       angles_deg: tuple[float, float, float] = (10, 5, 0),
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Apply a synthetic perspective warp to simulate a phone camera angle.

    Returns (warped_image, H_forward) where H_forward maps original
    coords to warped coords.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Simple 4-point perspective: shift the corners to simulate tilt
    rx, ry, rz = [np.radians(a) for a in angles_deg]

    # Just do a simple perspective shift of corners
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    # Shift top corners inward and down, shift bottom corners outward
    shift_x = w * 0.05 * np.sin(ry)
    shift_y = h * 0.05 * np.sin(rx)
    dst = np.array([
        [0 + shift_x, 0 + shift_y],
        [w - shift_x, 0 + shift_y * 0.5],
        [w + shift_x * 0.3, h - shift_y * 0.3],
        [0 - shift_x * 0.3, h - shift_y * 0.5],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (w, h),
                                  borderValue=(200, 200, 200))
    return warped, H


# ---------------------------------------------------------------------------
# Tests: _marker_corner_mm
# ---------------------------------------------------------------------------

class TestMarkerCornerMm:
    def test_corners_at_origin(self):
        corners = _marker_corner_mm(0, 0, 20.0)
        assert corners.shape == (4, 2)
        assert np.allclose(corners[0], [-10, -10])
        assert np.allclose(corners[2], [10, 10])

    def test_corners_offset(self):
        corners = _marker_corner_mm(100, 50, 20.0)
        assert np.allclose(corners[0], [90, 40])
        assert np.allclose(corners[1], [110, 40])
        assert np.allclose(corners[2], [110, 60])
        assert np.allclose(corners[3], [90, 60])


# ---------------------------------------------------------------------------
# Tests: detect_markers
# ---------------------------------------------------------------------------

class TestDetectMarkers:
    def test_detect_all_8_frontal(self):
        img = _render_template_bgr("letter", dpi=150)
        ids, corners = detect_markers(img)
        assert ids is not None
        detected = sorted(ids.flatten().tolist())
        assert detected == [0, 1, 2, 3, 4, 5, 6, 7]
        assert len(corners) == 8

    def test_detect_with_perspective(self):
        img = _render_template_bgr("letter", dpi=150)
        warped, _ = _apply_perspective(img)
        ids, corners = detect_markers(warped)
        assert ids is not None
        assert len(ids) >= 6  # most markers should still be detected

    def test_no_markers_in_blank_image(self):
        blank = np.full((500, 500, 3), 255, dtype=np.uint8)
        ids, corners = detect_markers(blank)
        assert ids is None
        assert corners is None

    def test_filters_out_of_range_ids(self):
        """Markers with ID >= 8 should be filtered out."""
        img = _render_template_bgr("a4", dpi=150)
        ids, corners = detect_markers(img)
        assert ids is not None
        for mid in ids.flatten():
            assert mid < 8


# ---------------------------------------------------------------------------
# Tests: compute_homography
# ---------------------------------------------------------------------------

class TestComputeHomography:
    def test_frontal_dpi_matches_expected(self):
        """A frontal render at known DPI should produce matching effective DPI."""
        dpi = 150
        img = _render_template_bgr("letter", dpi=dpi)
        ids, corners = detect_markers(img)
        H, effective_dpi, diag = compute_homography(ids, corners, "letter")

        # Effective DPI should be close to the render DPI
        assert abs(effective_dpi - dpi) < 5, \
            f"Expected ~{dpi} DPI, got {effective_dpi:.1f}"

    def test_reprojection_error_small(self):
        img = _render_template_bgr("letter", dpi=150)
        ids, corners = detect_markers(img)
        H, dpi, diag = compute_homography(ids, corners, "letter")
        assert diag["reprojection_error_mm"] < 1.0

    def test_scale_consistency(self):
        img = _render_template_bgr("a4", dpi=200)
        ids, corners = detect_markers(img)
        H, dpi, diag = compute_homography(ids, corners, "a4")
        assert diag["scale_diff_pct"] < 2.0

    def test_raises_on_too_few_markers(self):
        # Create an image with only 1 marker
        img = np.full((500, 500, 3), 255, dtype=np.uint8)
        marker = _generate_marker_image(0, 100)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        img[50:150, 50:150] = marker_bgr

        ids, corners = detect_markers(img)
        if ids is not None and len(ids) > 0:
            with pytest.raises(MarkerDetectionError):
                compute_homography(ids, corners, "letter")

    def test_works_with_perspective(self):
        """Homography should correct perspective and give reasonable DPI."""
        dpi = 150
        img = _render_template_bgr("letter", dpi=dpi)
        warped, _ = _apply_perspective(img, angles_deg=(8, 5, 0))
        ids, corners = detect_markers(warped)
        assert ids is not None and len(ids) >= 3

        H, effective_dpi, diag = compute_homography(ids, corners, "letter")
        # DPI won't match exactly due to perspective, but should be reasonable
        assert 80 < effective_dpi < 300

    def test_all_paper_sizes(self):
        for size in PAPER_SIZES:
            img = _render_template_bgr(size, dpi=150)
            ids, corners = detect_markers(img)
            H, dpi, diag = compute_homography(ids, corners, size)
            assert abs(dpi - 150) < 10


# ---------------------------------------------------------------------------
# Tests: warp_image
# ---------------------------------------------------------------------------

class TestWarpImage:
    def test_output_dimensions(self):
        img = _render_template_bgr("letter", dpi=150)
        ids, corners = detect_markers(img)
        H, dpi, _ = compute_homography(ids, corners, "letter")
        pw, ph = PAPER_SIZES["letter"]
        rectified = warp_image(img, H, (pw, ph), dpi)

        expected_w = int(round(pw * dpi / 25.4))
        expected_h = int(round(ph * dpi / 25.4))
        assert abs(rectified.shape[1] - expected_w) <= 1
        assert abs(rectified.shape[0] - expected_h) <= 1

    def test_rectified_markers_detectable(self):
        """After warping a perspective-distorted image, markers should
        still be detectable in the rectified output."""
        img = _render_template_bgr("letter", dpi=150)
        warped, _ = _apply_perspective(img)
        ids, corners = detect_markers(warped)
        H, dpi, _ = compute_homography(ids, corners, "letter")
        pw, ph = PAPER_SIZES["letter"]
        rectified = warp_image(warped, H, (pw, ph), dpi)

        # Detect markers in rectified image
        ids2, corners2 = detect_markers(rectified)
        assert ids2 is not None
        assert len(ids2) >= 6


# ---------------------------------------------------------------------------
# Tests: preprocess_phone_image (integration)
# ---------------------------------------------------------------------------

class TestPreprocessPhoneImage:
    def test_full_pipeline(self, tmp_path):
        """Full preprocess: render template -> save -> preprocess -> check."""
        dpi = 150
        img = _render_template_bgr("letter", dpi=dpi)
        img_path = str(tmp_path / "phone_photo.png")
        cv2.imwrite(img_path, img)

        result = preprocess_phone_image(
            img_path, paper_size="letter", output_dir=str(tmp_path / "out"))

        assert result["markers_detected"] == 8
        assert abs(result["effective_dpi"] - dpi) < 10
        assert result["reprojection_error_mm"] < 1.0

        import os
        assert os.path.isfile(result["rectified_image_path"])

    def test_rectified_image_is_cropped_to_zone(self, tmp_path):
        dpi = 150
        img = _render_template_bgr("letter", dpi=dpi)
        img_path = str(tmp_path / "photo.png")
        cv2.imwrite(img_path, img)

        result = preprocess_phone_image(
            img_path, paper_size="letter", output_dir=str(tmp_path / "out"))

        # Load rectified image and check it's roughly the placement zone size
        rect = cv2.imread(result["rectified_image_path"])
        zone = get_placement_zone("letter")
        zone_w_mm = zone[2] - zone[0]
        zone_h_mm = zone[3] - zone[1]
        eff_dpi = result["effective_dpi"]
        px_per_mm = eff_dpi / 25.4

        expected_w = int(round(zone_w_mm * px_per_mm))
        expected_h = int(round(zone_h_mm * px_per_mm))
        assert abs(rect.shape[1] - expected_w) < 5
        assert abs(rect.shape[0] - expected_h) < 5

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            preprocess_phone_image(
                str(tmp_path / "nonexistent.jpg"), paper_size="a4")

    def test_no_markers_raises(self, tmp_path):
        blank = np.full((500, 500, 3), 255, dtype=np.uint8)
        img_path = str(tmp_path / "blank.png")
        cv2.imwrite(img_path, blank)

        with pytest.raises(MarkerDetectionError):
            preprocess_phone_image(img_path, paper_size="a4")
