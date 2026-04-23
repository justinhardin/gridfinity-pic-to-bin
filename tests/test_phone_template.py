"""Tests for phone_template.py — ArUco marker template generation."""

import os
import numpy as np
import cv2
import pytest

from pic_to_bin.phone_template import (
    get_marker_positions,
    get_placement_zone,
    generate_template,
    PAPER_SIZES,
    MARKER_SIZE_MM,
    MARGIN_MM,
    ARUCO_DICT_ID,
    _generate_marker_image,
)


class TestGetMarkerPositions:
    def test_returns_8_markers(self):
        for size in PAPER_SIZES:
            markers = get_marker_positions(size)
            assert len(markers) == 8

    def test_ids_are_0_through_7(self):
        markers = get_marker_positions("a4")
        ids = sorted(m[0] for m in markers)
        assert ids == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_all_within_paper_bounds(self):
        for size, (pw, ph) in PAPER_SIZES.items():
            markers = get_marker_positions(size)
            half = MARKER_SIZE_MM / 2.0
            for mid, cx, cy in markers:
                assert cx - half >= 0, f"Marker {mid} exceeds left edge on {size}"
                assert cy - half >= 0, f"Marker {mid} exceeds top edge on {size}"
                assert cx + half <= pw, f"Marker {mid} exceeds right edge on {size}"
                assert cy + half <= ph, f"Marker {mid} exceeds bottom edge on {size}"

    def test_corner_markers_form_rectangle(self):
        markers = get_marker_positions("letter")
        corners = {m[0]: (m[1], m[2]) for m in markers if m[0] < 4}
        # Top-left and top-right have same Y
        assert corners[0][1] == corners[1][1]
        # Bottom-left and bottom-right have same Y
        assert corners[2][1] == corners[3][1]
        # Top-left and bottom-left have same X
        assert corners[0][0] == corners[3][0]
        # Top-right and bottom-right have same X
        assert corners[1][0] == corners[2][0]

    def test_unknown_size_raises(self):
        with pytest.raises(ValueError, match="Unknown paper size"):
            get_marker_positions("tabloid")

    def test_legal_taller_than_letter(self):
        letter = get_marker_positions("letter")
        legal = get_marker_positions("legal")
        # Bottom markers should be further down on legal
        letter_bottom_y = max(m[2] for m in letter)
        legal_bottom_y = max(m[2] for m in legal)
        assert legal_bottom_y > letter_bottom_y


class TestGetPlacementZone:
    def test_zone_inside_paper(self):
        for size, (pw, ph) in PAPER_SIZES.items():
            x0, y0, x1, y1 = get_placement_zone(size)
            assert x0 > 0
            assert y0 > 0
            assert x1 < pw
            assert y1 < ph

    def test_zone_dimensions_reasonable(self):
        x0, y0, x1, y1 = get_placement_zone("letter")
        w = x1 - x0
        h = y1 - y0
        assert w > 100  # at least 100mm wide
        assert h > 150  # at least 150mm tall


class TestGenerateMarkerImage:
    def test_returns_correct_size(self):
        img = _generate_marker_image(0, 200)
        assert img.shape == (200, 200)

    def test_is_binary(self):
        img = _generate_marker_image(0, 200)
        unique = set(np.unique(img))
        assert unique.issubset({0, 255})

    def test_different_ids_different_images(self):
        img0 = _generate_marker_image(0, 200)
        img1 = _generate_marker_image(1, 200)
        assert not np.array_equal(img0, img1)


class TestGenerateTemplate:
    def test_creates_pdf(self, tmp_path):
        out = str(tmp_path / "test_template.pdf")
        result = generate_template(out, paper_size="a4")
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_all_paper_sizes(self, tmp_path):
        for size in PAPER_SIZES:
            out = str(tmp_path / f"template_{size}.pdf")
            result = generate_template(out, paper_size=size)
            assert os.path.isfile(result)


class TestMarkersDetectableInRenderedTemplate:
    """Render the template to an image and verify all 8 markers are detected."""

    def _render_template_to_image(self, paper_size: str,
                                  dpi: int = 150) -> np.ndarray:
        """Render template to a numpy image at the given DPI."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pic_to_bin.phone_template import (
            get_marker_positions, get_placement_zone,
            _generate_marker_image, PAPER_SIZES, MARGIN_MM,
        )

        pw, ph = PAPER_SIZES[paper_size]
        fig, ax = plt.subplots(1, 1, figsize=(pw / 25.4, ph / 25.4))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(0, pw)
        ax.set_ylim(ph, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), pw, ph,
                                    facecolor="white", edgecolor="none"))

        half = MARKER_SIZE_MM / 2.0
        for mid, cx, cy in get_marker_positions(paper_size):
            img = _generate_marker_image(mid, 200)
            ax.imshow(img, cmap="gray", interpolation="nearest",
                      extent=[cx - half, cx + half, cy + half, cy - half],
                      zorder=2)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)
        plt.close(fig)
        # Convert RGBA to grayscale
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        return gray

    def test_all_8_markers_detected_a4(self):
        img = self._render_template_to_image("a4")
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(img)
        assert ids is not None, "No markers detected"
        detected = sorted(ids.flatten().tolist())
        assert detected == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_all_8_markers_detected_letter(self):
        img = self._render_template_to_image("letter")
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(img)
        assert ids is not None, "No markers detected"
        detected = sorted(ids.flatten().tolist())
        assert detected == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_all_8_markers_detected_legal(self):
        img = self._render_template_to_image("legal")
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(img)
        assert ids is not None, "No markers detected"
        detected = sorted(ids.flatten().tolist())
        assert detected == [0, 1, 2, 3, 4, 5, 6, 7]
