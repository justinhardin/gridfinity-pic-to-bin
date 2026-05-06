"""Tests for trace_export's smoothing and corner-rounding helpers.

These cover the curvature-aware smoothing and convex-skipping rounding
added to fix sharp-cornered tools (brackets/braces) being smoothed into
unfit-able pockets.
"""

import math

import numpy as np
import pytest

from pic_to_bin.trace_export import (
    _round_sharp_corners,
    _signed_area,
    _smooth_polygon_coords,
)


def _polygon_axis_extent(polygon, axis_index=0):
    arr = np.array(polygon)
    return float(arr[:, axis_index].max() - arr[:, axis_index].min())


def _is_axis_aligned_square(polygon, side, tol=1e-6):
    """Check whether `polygon` is the four corners of a side×side axis-aligned
    square (with vertices at (0,0), (side,0), (side,side), (0,side) — order
    doesn't matter as long as the set matches)."""
    expected = {(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)}
    got = {(round(x, 6), round(y, 6)) for x, y in polygon}
    return expected == got


# ---------------------------------------------------------------------------
# _signed_area
# ---------------------------------------------------------------------------


def test_signed_area_ccw_square_positive():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert _signed_area(sq) == pytest.approx(100.0)


def test_signed_area_cw_square_negative():
    sq = [(0, 0), (0, 10), (10, 10), (10, 0)]
    assert _signed_area(sq) == pytest.approx(-100.0)


def test_signed_area_degenerate_returns_zero():
    assert _signed_area([(0, 0), (1, 1)]) == 0.0  # too few points


# ---------------------------------------------------------------------------
# _round_sharp_corners
# ---------------------------------------------------------------------------


def test_concave_only_skips_convex_90deg_corners():
    """A square (all-convex 90° corners) must come out unchanged with
    concave_only=True. This is the bracket case: convex protrusions
    must not get filleted into the pocket."""
    sq = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    out = _round_sharp_corners(sq, concave_only=True)
    assert _is_axis_aligned_square(out, 10.0), out


def test_concave_only_rounds_concave_notch():
    """An L-shape has one concave corner (inside of the L). With
    concave_only=True, the convex outer corners stay sharp, and the
    concave inner corner gets filleted.

    The concave notch is cut at a 60° interior angle so the corner trips
    `_round_sharp_corners`'s `min_angle_deg=90` threshold (corners at or
    above 90° aren't filleted by design — see the inequality in the
    function)."""
    # CCW L-shape with a sharp (~60°) concave notch cut into the top edge.
    # Vertex order: outer corners CCW; the notch points inward toward
    # the interior.
    L = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (60.0, 100.0),
        (50.0, 50.0),    # CONCAVE inner corner; ~60° interior
        (40.0, 100.0),
        (0.0, 100.0),
    ]
    out = _round_sharp_corners(L, concave_only=True, radius=2.0)
    out_set = {(round(x, 4), round(y, 4)) for x, y in out}

    # The four convex outer corners stay verbatim.
    for v in [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]:
        assert (round(v[0], 4), round(v[1], 4)) in out_set, (
            f"Convex outer corner {v} was modified"
        )
    # The concave notch tip at (50, 50) gets replaced by an arc.
    assert (50.0, 50.0) not in out_set, (
        "Concave corner should have been filleted but is unchanged"
    )


def test_legacy_rounds_all_corners_when_concave_only_false():
    """concave_only=False is the legacy behavior — sharp corners get
    filleted regardless of convex/concave direction.

    Use a thin diamond where the side tips at x=0 and x=20 are sharp
    (~5.7° interior) and trip the rounding threshold, while the top/
    bottom corners at (10, ±1) are gentle (~168°) and stay. With
    concave_only=True the convex sharp tips would be skipped; with
    concave_only=False they get filleted."""
    diamond = [(0.0, 0.0), (10.0, 1.0), (20.0, 0.0), (10.0, -1.0)]
    sharp_corners = [(0.0, 0.0), (20.0, 0.0)]
    gentle_corners = [(10.0, 1.0), (10.0, -1.0)]

    # With concave_only=True, all four corners are convex and stay.
    out_concave = _round_sharp_corners(diamond, concave_only=True, radius=0.3)
    out_concave_set = {(round(x, 4), round(y, 4)) for x, y in out_concave}
    for corner in sharp_corners:
        assert (round(corner[0], 4), round(corner[1], 4)) in out_concave_set

    # With concave_only=False, the sharp convex tips get filleted.
    out_legacy = _round_sharp_corners(diamond, concave_only=False, radius=0.3)
    out_legacy_set = {(round(x, 4), round(y, 4)) for x, y in out_legacy}
    for corner in sharp_corners:
        assert (round(corner[0], 4), round(corner[1], 4)) not in out_legacy_set, (
            f"Sharp tip {corner} should have been filleted with "
            f"concave_only=False"
        )
    # Gentle corners (>90°) are above the rounding threshold and stay.
    for corner in gentle_corners:
        assert (round(corner[0], 4), round(corner[1], 4)) in out_legacy_set


def test_handles_clockwise_polygon():
    """A CW square: the convex/concave detection has to flip sign with
    winding direction. All corners are still convex, so concave_only=True
    must leave them alone."""
    sq = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    out = _round_sharp_corners(sq, concave_only=True)
    assert _is_axis_aligned_square(out, 10.0), out


# ---------------------------------------------------------------------------
# _smooth_polygon_coords
# ---------------------------------------------------------------------------


def _make_long_thin_with_noise_and_tip():
    """Build a synthetic test polygon: a 2 mm × 100 mm thin strip pointing
    along +X, with a ±0.5 mm sinusoidal wave on the long sides AND a sharp
    triangular tip protruding +5 mm at the right end.

    Used to verify two simultaneous properties of the smoother:
        - Wave noise on the sides gets flattened.
        - The sharp axial tip stays sharp (axial extent preserved within
          a small tolerance).
    """
    rng = np.random.default_rng(0)
    n_per_side = 200
    xs_top = np.linspace(0, 100, n_per_side)
    ys_top = 1.0 + 0.5 * np.sin(xs_top * 2.0)  # 2 mm wave wavelength

    # Sharp tip at x = 100 protruding to (105, 0)
    tip = [(100.0, 1.0), (105.0, 0.0), (100.0, -1.0)]

    xs_bot = xs_top[::-1]
    ys_bot = -1.0 - 0.5 * np.sin(xs_bot * 2.0)

    polygon = (
        [(float(x), float(y)) for x, y in zip(xs_top, ys_top)]
        + tip
        + [(float(x), float(y)) for x, y in zip(xs_bot, ys_bot)]
    )
    return polygon


def test_smoothing_preserves_axial_extent_with_sharp_tip():
    """The principal axis of this synthetic strip is X. The polygon's
    rightmost point sits at x=105 (the sharp tip). After anisotropic +
    curvature-aware smoothing, that tip must still reach within 0.5 mm
    of x=105 — the previous isotropic smoothing was pulling tips inward
    by several mm."""
    polygon = _make_long_thin_with_noise_and_tip()
    raw_extent_x = _polygon_axis_extent(polygon, 0)
    smoothed = _smooth_polygon_coords(polygon, sigma_mm=4.0)
    smoothed_extent_x = _polygon_axis_extent(smoothed, 0)
    # Axial extent (X) must shrink by at most ~0.5 mm.
    assert smoothed_extent_x >= raw_extent_x - 0.5, (
        f"Axial extent shrank too much: {raw_extent_x:.3f} → "
        f"{smoothed_extent_x:.3f}"
    )


def test_smoothing_partially_flattens_perpendicular_wave_noise():
    """Side wave noise (perpendicular to the principal axis) has 0.5 mm
    amplitude in the input. After smoothing with sigma=4 mm, the wave
    should be partially attenuated — but the smoother now applies an
    outward bias (no point may shrink below the input's outward extent
    in the PCA frame), so wave PEAKS are preserved while VALLEYS get
    filled in. That makes the residual std about half the raw std rather
    than the previous ~⅕, which is the cost of guaranteeing the trace
    never clips real tool features inward.

    We still expect a meaningful (>20 %) reduction; if the wave were
    fully preserved (no smoothing) the test would fail."""
    polygon = _make_long_thin_with_noise_and_tip()
    raw = np.array(polygon)
    in_top_body = (np.abs(raw[:, 0] - 50) < 30) & (raw[:, 1] > 0.5)
    raw_std = float(raw[in_top_body, 1].std())
    assert raw_std > 0.2, "Test fixture is missing the expected wave noise"

    smoothed = _smooth_polygon_coords(polygon, sigma_mm=4.0)
    smoothed_arr = np.array(smoothed)
    in_top_body_s = (
        (np.abs(smoothed_arr[:, 0] - 50) < 30) & (smoothed_arr[:, 1] > 0)
    )
    smoothed_std = float(smoothed_arr[in_top_body_s, 1].std())
    assert smoothed_std < raw_std * 0.8, (
        f"Wave noise wasn't flattened at all: std {raw_std:.3f} → "
        f"{smoothed_std:.3f}"
    )


def test_smoothing_never_shrinks_below_input_extent():
    """Outward bias guarantee: every smoothed point's |perpendicular
    offset from the centroid (in PCA frame)| is ≥ the corresponding input
    point's. This is what prevents the trace from clipping tool features
    when a curve gets pulled inward by the smoothing kernel."""
    polygon = _make_long_thin_with_noise_and_tip()
    raw = np.array(polygon)
    smoothed = np.array(_smooth_polygon_coords(polygon, sigma_mm=4.0))

    # Compute the input polygon's PCA frame so we can compare outward
    # extents component-wise.
    centroid = raw.mean(axis=0)
    centered = raw - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    secondary = eigvecs[:, 1 - int(np.argmax(eigvals))]
    perp_raw = np.abs(centered @ secondary)
    sm_centered = smoothed - centroid
    perp_sm = np.abs(sm_centered @ secondary)

    # Smoothed has a different point count after resampling; for each
    # input vertex check that the closest smoothed vertex's outward
    # extent isn't notably smaller. Tolerance ~ resample_step (0.3 mm)
    # accounts for the fact that the resampled polygon may not include
    # the exact original vertex if it falls between two arc-length samples.
    tol = 0.3
    for i, pt in enumerate(raw):
        d = np.linalg.norm(smoothed - pt, axis=1)
        j = int(np.argmin(d))
        assert perp_sm[j] >= perp_raw[i] - tol, (
            f"Smoothed extent shrank below input at vertex {i}: "
            f"raw |perp|={perp_raw[i]:.3f}, smoothed |perp|={perp_sm[j]:.3f}"
        )


def test_smoothing_with_zero_sigma_is_identity_on_triangle():
    """Zero sigma on both axes returns the input unchanged (early-exit
    path; no resampling, no PCA)."""
    poly = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
    out = _smooth_polygon_coords(poly, sigma_mm=0.0, sigma_axial_mm=0.0)
    assert out == poly


def test_smoothing_handles_small_polygon():
    """Two-point degenerate input falls through to identity rather than
    blowing up the resampler."""
    poly = [(0.0, 0.0), (1.0, 0.0)]
    out = _smooth_polygon_coords(poly, sigma_mm=2.0)
    assert out == poly


def test_smoothing_preserves_sharp_convex_corners():
    """Convex sharp corners (outward-pointing, like the corner of a
    trigger guard) must not be pulled inward by the smoother. Earlier
    behavior used a per-vertex Gaussian falloff that left ~10 % smoothing
    creep at a 90° vertex AND fully smoothed neighbors within ~3·sigma,
    visibly clipping the feature (see trigger_rounded.png).

    Use a CCW triangle so the principal axis is well-defined and all
    three vertices are convex. After smoothing, each original corner
    must still have a smoothed point within ~1 mm — i.e. the corner
    is not rounded off by more than 1 mm of arc."""
    size = 50.0
    corners = [
        (0.0, 0.0),
        (size, 0.0),
        (size / 2, size * (3.0 ** 0.5) / 2.0),
    ]
    n_edge = 60
    polygon = []
    for i in range(3):
        p0 = corners[i]
        p1 = corners[(i + 1) % 3]
        for t in np.linspace(0, 1, n_edge, endpoint=False):
            polygon.append((p0[0] + t * (p1[0] - p0[0]),
                            p0[1] + t * (p1[1] - p0[1])))

    smoothed = _smooth_polygon_coords(polygon, sigma_mm=2.5)
    smoothed_arr = np.array(smoothed)

    for cx, cy in corners:
        d = np.linalg.norm(smoothed_arr - np.array([cx, cy]), axis=1)
        min_d = float(d.min())
        assert min_d < 1.0, (
            f"Convex corner ({cx},{cy}) clipped by smoothing: closest "
            f"smoothed point is {min_d:.2f} mm away (expected < 1 mm)."
        )


def test_smoothing_does_not_widen_polygon_at_long_edges():
    """The convex-corner-preservation change must not bleed into ordinary
    long edges — the wave-flattening on a noisy thin strip should still
    work and the polygon's perpendicular extent should not balloon out.

    This is the symmetric counterpart to
    ``test_smoothing_preserves_sharp_convex_corners``: that one asserts
    sharp protrusions stay sharp; this one asserts the rest of the
    polygon doesn't drift outward as a side effect."""
    polygon = _make_long_thin_with_noise_and_tip()
    smoothed = _smooth_polygon_coords(polygon, sigma_mm=4.0)
    smoothed_arr = np.array(smoothed)
    raw_arr = np.array(polygon)

    # Pick a body region (away from the tip) on the +Y side and check
    # that the smoothed top edge sits inside or at the original outer
    # extent, not pushed outward.
    in_body_top = (np.abs(smoothed_arr[:, 0] - 50) < 30) & (smoothed_arr[:, 1] > 0)
    raw_in_body_top = (np.abs(raw_arr[:, 0] - 50) < 30) & (raw_arr[:, 1] > 0)
    raw_max_y = float(raw_arr[raw_in_body_top, 1].max())
    smoothed_max_y = float(smoothed_arr[in_body_top, 1].max())
    assert smoothed_max_y <= raw_max_y + 0.1, (
        f"Smoothing pushed the top edge outward: raw max y={raw_max_y:.3f}, "
        f"smoothed max y={smoothed_max_y:.3f}"
    )
