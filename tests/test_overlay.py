"""Tests for the LLM-input overlay generator.

The overlay step renders the trace polygons on top of the rectified
photo at the same mm scale, so the LLM can judge fit by inspecting one
image instead of mentally aligning two coordinate systems.

These tests verify:
- Sidecar DPI metadata is read when the caller doesn't pass dpi=.
- The mm→pixel conversion has the expected sign/scale (inverse of the
  Y-flip done by `trace_export._potrace_curves_to_polygons`).
- A bare DXF + a small synthetic rectified image produces a valid PNG
  at sensible dimensions.
- Missing DPI raises a clear error.
"""

from __future__ import annotations

import json
from pathlib import Path

import ezdxf
import numpy as np
import pytest
from PIL import Image

from pic_to_bin.web import overlay


def _write_minimal_rectified(path: Path, dpi: float, *, w_mm=40.0, h_mm=80.0):
    """Synthesize a rectified PNG (white background) plus its DPI sidecar."""
    px_per_mm = dpi / 25.4
    w_px = int(round(w_mm * px_per_mm))
    h_px = int(round(h_mm * px_per_mm))
    img = np.full((h_px, w_px, 3), 240, dtype=np.uint8)
    Image.fromarray(img).save(path)

    # Sidecar mirrors what phone_preprocess writes.
    meta_path = path.with_suffix("").parent / (path.stem + ".json")
    meta_path.write_text(json.dumps({
        "effective_dpi": dpi,
        "paper_size": "letter",
        "image_width_px": w_px,
        "image_height_px": h_px,
        "placement_zone_mm": [0.0, 0.0, w_mm, h_mm],
    }, indent=2), encoding="utf-8")
    return w_px, h_px


def _write_minimal_dxf(path: Path):
    """Synthesize a DXF with a small inner rectangle on layer 0 and an
    expanded tolerance rectangle on layer TOLERANCE — enough geometry
    that overlay rendering exercises both layers."""
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    msp = doc.modelspace()
    inner = [(10.0, 20.0), (30.0, 20.0), (30.0, 60.0), (10.0, 60.0), (10.0, 20.0)]
    msp.add_lwpolyline(inner, close=True)
    doc.layers.add("TOLERANCE", color=3)
    tolerance = [(8.0, 18.0), (32.0, 18.0), (32.0, 62.0), (8.0, 62.0), (8.0, 18.0)]
    msp.add_lwpolyline(tolerance, close=True,
                       dxfattribs={"layer": "TOLERANCE"})
    doc.saveas(str(path))


# ---------------------------------------------------------------------------


def test_mm_to_image_pixels_inverts_y_flip():
    """``_mm_to_image_pixels`` is the inverse of the Y-flip done in
    ``trace_export._potrace_curves_to_polygons``: a polygon point with
    mm_y near the maximum should land near image y=0 (top of image)."""
    # 100 px tall image, 100 dpi → 25.4 mm tall (Y-up mm).
    h_px = 100
    dpi = 100.0
    # Polygon point at mm_y = 25.4 (top of CAD-Y range) → image y_px ≈ 0.
    pts_top_mm = [(0.0, 25.4), (10.0, 25.4)]
    pixels = overlay._mm_to_image_pixels(pts_top_mm, h_px, dpi)
    assert pixels[0, 1] == pytest.approx(0.0, abs=0.5)

    # Polygon point at mm_y = 0 → image y_px ≈ h_px.
    pts_bot_mm = [(0.0, 0.0)]
    pixels_bot = overlay._mm_to_image_pixels(pts_bot_mm, h_px, dpi)
    assert pixels_bot[0, 1] == pytest.approx(h_px, abs=0.5)


def test_mm_to_image_pixels_undoes_straighten_round_trip():
    """Round-trip: take a known pixel in the original rectified frame,
    apply the same straightening transform that ``trace_tool.straighten_mask``
    would produce, convert that to mm in the post-straighten frame, then
    pass through ``_mm_to_image_pixels`` with the matching straighten_info.
    The output must land back near the original pixel."""
    import cv2

    # Original mask is 200×100 (h×w). Pretend we straightened by +12°.
    h_orig, w_orig = 200, 100
    correction_deg = 12.0
    rad = np.radians(abs(correction_deg))
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    w_new = int(np.ceil(h_orig * sin_a + w_orig * cos_a))
    h_new = int(np.ceil(h_orig * cos_a + w_orig * sin_a))

    M = cv2.getRotationMatrix2D(
        (w_orig / 2.0, h_orig / 2.0), correction_deg, 1.0
    )
    M[0, 2] += (w_new - w_orig) / 2.0
    M[1, 2] += (h_new - h_orig) / 2.0

    # A test pixel in the ORIGINAL frame (a feature of the photo we want
    # to land on after the inverse).
    orig_x, orig_y = 30.0, 80.0
    # Apply forward transform to get the pixel in the post-straighten frame.
    post = M @ np.array([orig_x, orig_y, 1.0])
    post_x, post_y = float(post[0]), float(post[1])

    # Now express this post-frame pixel as polygon mm coords (Y-flipped
    # by h_new — same convention `_potrace_curves_to_polygons` uses).
    dpi = 100.0
    px_per_mm = dpi / 25.4
    mm_x = post_x / px_per_mm
    mm_y = (h_new - post_y) / px_per_mm

    straighten_info = {
        "applied": True,
        "correction_deg": correction_deg,
        "original_shape": [h_orig, w_orig],
        "new_shape": [h_new, w_new],
    }

    # Pass through and check we land near the original pixel.
    pixels = overlay._mm_to_image_pixels(
        [(mm_x, mm_y)],
        image_height_px=h_orig,  # photo dimensions, not used when applied=True
        dpi=dpi,
        straighten_info=straighten_info,
    )
    assert pixels[0, 0] == pytest.approx(orig_x, abs=0.5)
    assert pixels[0, 1] == pytest.approx(orig_y, abs=0.5)


def test_mm_to_image_pixels_no_straighten_when_applied_false():
    """``applied=False`` → behave exactly like the unrotated case."""
    h_px = 200
    dpi = 100.0
    pts = [(10.0, 5.0)]
    info = {
        "applied": False,
        "correction_deg": 0.0,
        "original_shape": [h_px, 100],
        "new_shape": [h_px, 100],
    }
    a = overlay._mm_to_image_pixels(pts, h_px, dpi, straighten_info=info)
    b = overlay._mm_to_image_pixels(pts, h_px, dpi, straighten_info=None)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_mm_to_image_pixels_undoes_parallax():
    """``parallax_factor < 1`` means the polygon mm coords are shrunken
    to real-world size; the overlay must scale them back UP by 1/factor
    to land on the parallax-inflated photo silhouette."""
    h_px = 200
    dpi = 100.0
    # A polygon point at (10 mm, 5 mm) when parallax_factor=0.5 in mm
    # corresponds to (20 mm, 10 mm) in photo-frame mm. With dpi=100 →
    # px_per_mm ≈ 3.937. Photo-frame pixel x ≈ 78.74. Y is flipped from
    # the photo-frame mm height: y_px ≈ h_px − 10·px_per_mm ≈ 161.
    info = {
        "applied": False, "correction_deg": 0.0,
        "original_shape": [h_px, 100], "new_shape": [h_px, 100],
        "parallax_factor": 0.5,
    }
    out = overlay._mm_to_image_pixels([(10.0, 5.0)], h_px, dpi, info)
    px_per_mm = dpi / 25.4
    expected_x = (10.0 / 0.5) * px_per_mm
    expected_y = h_px - (5.0 / 0.5) * px_per_mm
    assert out[0, 0] == pytest.approx(expected_x, abs=0.5)
    assert out[0, 1] == pytest.approx(expected_y, abs=0.5)


def test_mm_to_image_pixels_parallax_one_is_noop():
    """``parallax_factor = 1.0`` (or absent) must not change anything."""
    h_px = 200
    dpi = 100.0
    pts = [(10.0, 5.0)]
    info_no_pf = {
        "applied": False, "correction_deg": 0.0,
        "original_shape": [h_px, 100], "new_shape": [h_px, 100],
    }
    info_pf_one = {**info_no_pf, "parallax_factor": 1.0}
    a = overlay._mm_to_image_pixels(pts, h_px, dpi, info_no_pf)
    b = overlay._mm_to_image_pixels(pts, h_px, dpi, info_pf_one)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_mm_to_image_pixels_combines_straighten_and_parallax():
    """Round-trip: take a known photo-frame pixel, forward-rotate it to
    post-straighten frame, scale DOWN by parallax_factor to get the
    polygon's stored mm, then run through ``_mm_to_image_pixels`` —
    must land back at the original photo pixel."""
    import cv2

    h_orig, w_orig = 200, 100
    correction_deg = 12.0
    parallax_factor = 0.85

    rad = np.radians(abs(correction_deg))
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    w_new = int(np.ceil(h_orig * sin_a + w_orig * cos_a))
    h_new = int(np.ceil(h_orig * cos_a + w_orig * sin_a))

    M = cv2.getRotationMatrix2D(
        (w_orig / 2.0, h_orig / 2.0), correction_deg, 1.0
    )
    M[0, 2] += (w_new - w_orig) / 2.0
    M[1, 2] += (h_new - h_orig) / 2.0

    orig_x, orig_y = 30.0, 80.0
    post = M @ np.array([orig_x, orig_y, 1.0])
    post_x, post_y = float(post[0]), float(post[1])

    dpi = 100.0
    px_per_mm = dpi / 25.4
    # Photo-frame mm at this post-straighten pixel
    photo_mm_x = post_x / px_per_mm
    photo_mm_y = (h_new - post_y) / px_per_mm
    # Polygon mm = photo mm × parallax_factor (stored shrunken)
    poly_mm_x = photo_mm_x * parallax_factor
    poly_mm_y = photo_mm_y * parallax_factor

    info = {
        "applied": True, "correction_deg": correction_deg,
        "original_shape": [h_orig, w_orig], "new_shape": [h_new, w_new],
        "parallax_factor": parallax_factor,
    }
    out = overlay._mm_to_image_pixels(
        [(poly_mm_x, poly_mm_y)], image_height_px=h_orig, dpi=dpi,
        straighten_info=info,
    )
    assert out[0, 0] == pytest.approx(orig_x, abs=0.5)
    assert out[0, 1] == pytest.approx(orig_y, abs=0.5)


def test_read_straighten_metadata(tmp_path):
    dxf = tmp_path / "img_a_rectified_trace.dxf"
    dxf.write_text("placeholder", encoding="utf-8")
    sidecar = tmp_path / "img_a_rectified_trace_straighten.json"
    sidecar.write_text(json.dumps({
        "applied": True, "correction_deg": 7.5,
        "original_shape": [100, 100], "new_shape": [110, 110],
    }), encoding="utf-8")
    info = overlay._read_straighten_metadata(dxf)
    assert info is not None
    assert info["applied"] is True
    assert info["correction_deg"] == 7.5


def test_read_straighten_metadata_missing(tmp_path):
    dxf = tmp_path / "img_a_rectified_trace.dxf"
    dxf.write_text("placeholder", encoding="utf-8")
    assert overlay._read_straighten_metadata(dxf) is None


def test_read_dpi_metadata_returns_value(tmp_path):
    img = tmp_path / "tool_rectified.png"
    img.write_bytes(b"")  # placeholder — only the sidecar matters here
    (tmp_path / "tool_rectified.json").write_text(
        json.dumps({"effective_dpi": 412.5}), encoding="utf-8"
    )
    assert overlay._read_dpi_metadata(img) == pytest.approx(412.5)


def test_read_dpi_metadata_missing_returns_none(tmp_path):
    img = tmp_path / "tool_rectified.png"
    img.write_bytes(b"")
    assert overlay._read_dpi_metadata(img) is None


def test_read_dpi_metadata_corrupt_returns_none(tmp_path):
    img = tmp_path / "tool_rectified.png"
    img.write_bytes(b"")
    (tmp_path / "tool_rectified.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    assert overlay._read_dpi_metadata(img) is None


def test_generate_overlay_uses_sidecar_dpi(tmp_path):
    """End-to-end: rectified PNG + trace DXF + sidecar DPI metadata →
    overlay PNG written, non-empty, openable as an image."""
    rect = tmp_path / "tool_rectified.png"
    dxf = tmp_path / "tool_rectified_trace.dxf"
    out = tmp_path / "tool_rectified_overlay.png"

    _write_minimal_rectified(rect, dpi=200.0)
    _write_minimal_dxf(dxf)

    result = overlay.generate_overlay_image(
        rectified_path=rect,
        trace_dxf_path=dxf,
        output_path=out,
    )
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 1024  # PNG is more than just headers
    # Verify the output is a real PNG, not a 0-byte file masquerading.
    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.size[0] > 50 and img.size[1] > 50


def test_generate_overlay_explicit_dpi_overrides_sidecar(tmp_path):
    """An explicit dpi= argument overrides the sidecar metadata."""
    rect = tmp_path / "tool_rectified.png"
    dxf = tmp_path / "tool_rectified_trace.dxf"
    out = tmp_path / "tool_rectified_overlay.png"
    _write_minimal_rectified(rect, dpi=200.0)
    _write_minimal_dxf(dxf)

    # Pass a different dpi explicitly. Should not crash; should write
    # a valid PNG.
    overlay.generate_overlay_image(
        rectified_path=rect,
        trace_dxf_path=dxf,
        output_path=out,
        dpi=300.0,
    )
    assert out.exists()


def test_generate_overlay_missing_dpi_raises(tmp_path):
    """No sidecar and no explicit dpi → clear ValueError."""
    rect = tmp_path / "tool_rectified.png"
    dxf = tmp_path / "tool_rectified_trace.dxf"
    out = tmp_path / "overlay.png"

    # Synthesize the rectified image without writing the sidecar.
    img = np.full((100, 50, 3), 240, dtype=np.uint8)
    Image.fromarray(img).save(rect)
    _write_minimal_dxf(dxf)

    with pytest.raises(ValueError, match="effective_dpi"):
        overlay.generate_overlay_image(
            rectified_path=rect,
            trace_dxf_path=dxf,
            output_path=out,
        )


def test_read_trace_polygons_layer_routing(tmp_path):
    """DXF layer "0" → "inner", "TOLERANCE" → "tolerance",
    "SLOT" → "slot"; other layers ignored."""
    dxf = tmp_path / "x.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (1, 0), (1, 1), (0, 1)], close=True)
    doc.layers.add("TOLERANCE", color=3)
    msp.add_lwpolyline([(2, 0), (3, 0), (3, 1)], close=True,
                       dxfattribs={"layer": "TOLERANCE"})
    doc.layers.add("SLOT", color=5)
    msp.add_lwpolyline([(4, 0), (5, 0), (5, 1)], close=True,
                       dxfattribs={"layer": "SLOT"})
    doc.layers.add("UNRELATED", color=7)
    msp.add_lwpolyline([(6, 0), (7, 0)], close=False,
                       dxfattribs={"layer": "UNRELATED"})
    doc.saveas(str(dxf))

    result = overlay._read_trace_polygons(dxf)
    assert len(result["inner"]) == 1
    assert len(result["tolerance"]) == 1
    assert len(result["slot"]) == 1
    # "UNRELATED" should not appear under any of the three keys.
    total = sum(len(v) for v in result.values())
    assert total == 3


def test_cap_image_size_to_jpeg_already_small(tmp_path):
    """A tiny source image needs no shrinking; we still re-save as JPEG
    so the call always produces a known format the LLM accepts."""
    from pic_to_bin.web import overlay as _overlay
    src = tmp_path / 'src.png'
    Image.new('RGB', (50, 50), color='blue').save(src)
    dst = tmp_path / 'small.jpg'
    out = _overlay.cap_image_size_to_jpeg(src, dst, max_bytes=1_000_000)
    assert out == dst
    assert dst.exists()
    assert dst.stat().st_size <= 1_000_000
    with Image.open(dst) as im:
        assert im.format == 'JPEG'
        assert im.size == (50, 50)


def test_cap_image_size_to_jpeg_shrinks_to_fit(tmp_path):
    """A large noisy image (incompressible content, big pixel count) must
    end up under the cap, even if quality reduction alone isn't enough —
    the function falls back to dimension reduction."""
    import numpy as _np
    from pic_to_bin.web import overlay as _overlay
    rng = _np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(2000, 2000, 3), dtype=_np.uint8)
    src = tmp_path / 'noisy.png'
    Image.fromarray(arr, 'RGB').save(src)
    cap = 100_000  # 100 KB — well below what 4 MP of noise compresses to
    dst = tmp_path / 'capped.jpg'
    _overlay.cap_image_size_to_jpeg(src, dst, max_bytes=cap)
    assert dst.exists()
    assert dst.stat().st_size <= cap, (
        f'cap not honored: {dst.stat().st_size} > {cap}'
    )
    with Image.open(dst) as im:
        # Must have reduced dimensions to fit such an aggressive cap.
        assert max(im.size) < 2000


def test_cap_image_size_to_jpeg_converts_rgba(tmp_path):
    """PNGs with alpha (matplotlib's overlay output is often RGBA) must
    convert cleanly to JPEG (no alpha) without raising."""
    from pic_to_bin.web import overlay as _overlay
    src = tmp_path / 'rgba.png'
    Image.new('RGBA', (100, 100), color=(255, 0, 0, 128)).save(src)
    dst = tmp_path / 'rgba.jpg'
    _overlay.cap_image_size_to_jpeg(src, dst, max_bytes=500_000)
    with Image.open(dst) as im:
        assert im.format == 'JPEG'
        assert im.mode == 'RGB'
