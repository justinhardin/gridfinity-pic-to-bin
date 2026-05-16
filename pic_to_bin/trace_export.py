"""
Export potrace paths to SVG and DXF formats.
Handles scaling from pixels to mm, clearance offset, and tolerance outline.
"""

import math

import ezdxf
import numpy as np
import pyclipper


def _is_boundary_curve(curve, img_h: int, img_w: int, tolerance: float = 0.02) -> bool:
    """Check if a potrace curve is just the full-image boundary rectangle.

    potrace sometimes generates a rectangular path spanning the entire bitmap.
    This is an artifact, not a real traced feature. Detect it by checking if
    the curve's bounding box matches the image dimensions (within tolerance).
    """
    xs = [curve.start_point.x]
    ys = [curve.start_point.y]
    for seg in curve:
        xs.append(seg.end_point.x)
        ys.append(seg.end_point.y)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    curve_w = max_x - min_x
    curve_h = max_y - min_y

    # Check if this curve spans the full image (within tolerance)
    w_ratio = curve_w / img_w if img_w > 0 else 0
    h_ratio = curve_h / img_h if img_h > 0 else 0

    return w_ratio > (1 - tolerance) and h_ratio > (1 - tolerance)


def _filter_curves(path, img_shape: tuple):
    """Return list of potrace curves, excluding full-image boundary rectangles.

    Args:
        path: potrace.Path object
        img_shape: (height, width) of the source mask image
    """
    img_h, img_w = img_shape[:2]
    filtered = []
    for curve in path:
        if _is_boundary_curve(curve, img_h, img_w):
            continue
        filtered.append(curve)
    return filtered


def _potrace_curves_to_polygons(path, scale: float,
                                img_shape: tuple = None) -> list[list[tuple[float, float]]]:
    """Convert potrace path curves to polygon point lists (scaled to mm).

    Each curve in the potrace path becomes a polygon (list of (x, y) points).
    Bezier curves are sampled into line segments for polygon representation.

    The Y axis is flipped from image convention (Y-down) to CAD convention
    (Y-up), so that the output orientation matches the input photo when
    viewed in Y-up CAD tools (DXF viewers, Fusion 360, matplotlib).
    """
    polygons = []

    # Y-flip: image_h * scale - y, so top-of-photo ends up at high Y in CAD.
    h_mm = img_shape[0] * scale if img_shape else 0.0
    fy = (lambda y: h_mm - y) if img_shape else (lambda y: y)

    curves = _filter_curves(path, img_shape) if img_shape else path
    for curve in curves:
        points = []
        start_x = curve.start_point.x * scale
        start_y = fy(curve.start_point.y * scale)
        points.append((start_x, start_y))

        for segment in curve:
            if segment.is_corner:
                # Corner: two line segments via a corner point
                cx = segment.c.x * scale
                cy = fy(segment.c.y * scale)
                ex = segment.end_point.x * scale
                ey = fy(segment.end_point.y * scale)
                points.append((cx, cy))
                points.append((ex, ey))
            else:
                # Bezier curve: cubic bezier from current point through c1, c2 to end
                x0, y0 = points[-1]
                c1x = segment.c1.x * scale
                c1y = fy(segment.c1.y * scale)
                c2x = segment.c2.x * scale
                c2y = fy(segment.c2.y * scale)
                ex = segment.end_point.x * scale
                ey = fy(segment.end_point.y * scale)

                # Sample the bezier curve
                n_samples = 10
                for i in range(1, n_samples + 1):
                    t = i / n_samples
                    t2 = t * t
                    t3 = t2 * t
                    mt = 1 - t
                    mt2 = mt * mt
                    mt3 = mt2 * mt

                    x = mt3 * x0 + 3 * mt2 * t * c1x + 3 * mt * t2 * c2x + t3 * ex
                    y = mt3 * y0 + 3 * mt2 * t * c1y + 3 * mt * t2 * c2y + t3 * ey
                    points.append((x, y))

        polygons.append(points)

    return polygons


def _offset_polygons(polygons: list[list[tuple[float, float]]],
                     offset_mm: float) -> list[list[tuple[float, float]]]:
    """Offset polygons outward by offset_mm using Clipper library."""
    if abs(offset_mm) < 0.001:
        return polygons

    # Clipper works with integer coordinates, so scale up
    CLIPPER_SCALE = 1000  # 1mm = 1000 units -> 0.001mm precision

    pco = pyclipper.PyclipperOffset()

    for poly in polygons:
        scaled_poly = [(int(x * CLIPPER_SCALE), int(y * CLIPPER_SCALE))
                       for x, y in poly]
        pco.AddPath(scaled_poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    offset_scaled = int(offset_mm * CLIPPER_SCALE)
    result = pco.Execute(offset_scaled)

    # Scale back to mm
    offset_polygons = []
    for poly in result:
        offset_polygons.append([(x / CLIPPER_SCALE, y / CLIPPER_SCALE)
                                for x, y in poly])

    return offset_polygons


def _principal_axis_angle(polygons: list[list[tuple[float, float]]]) -> float:
    """Return the angle (radians) of the principal axis of the union of all
    points in `polygons`, computed via PCA. 0 means horizontal."""
    pts = [p for poly in polygons for p in poly]
    if len(pts) < 2:
        return 0.0
    arr = np.array(pts, dtype=float)
    centered = arr - arr.mean(axis=0)
    # SVD of the (centered) point cloud — first right-singular vector is the
    # principal axis direction.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    return float(np.arctan2(axis[1], axis[0]))


def _auto_axial_tolerance_mm(
    polygons: list[list[tuple[float, float]]],
) -> float:
    """Compute an axial tolerance based on the tool's tip taper.

    SAM2 under-detects tool length on every shape — even rounded or
    square ends lose a few pixels at the axial extremes — and tapered
    tips lose more. The 2 mm floor matches the perpendicular baseline
    so length and width get the same minimum clearance; tapered shapes
    add proportional extra on top.

    Algorithm:
      1. PCA principal axis from the polygon point cloud (existing helper).
      2. Project all points to the rotated frame; bin along the axis.
      3. In each bin, perpendicular extent = max − min of cross-axis
         coordinate. Empty / tiny bins are skipped.
      4. tip_width  = mean of width in the outer 10% of bins (each end).
         body_width = median of width across the middle 80% of bins.
      5. taper = clamp(1 − tip_width / body_width, 0, 1).
      6. axial_tol = 2.0 + 0.014 × axial_length × taper.

    Returns a non-negative float. On a degenerate polygon (no points,
    zero extent) returns the 2.0 mm floor.
    """
    floor = 2.0
    if not polygons:
        return floor

    # Pick the largest polygon by point count as the outer outline —
    # smaller polygons (interior holes) don't shape the tip taper.
    outer = max(polygons, key=len)
    if len(outer) < 8:
        return floor

    angle = _principal_axis_angle([outer])
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    pts = np.array(outer, dtype=float) @ rot.T  # (N, 2): col 0 axial, col 1 perp

    axial = pts[:, 0]
    perp = pts[:, 1]
    axial_min, axial_max = float(axial.min()), float(axial.max())
    axial_length = axial_max - axial_min
    if axial_length < 1e-3:
        return floor

    # Bin along the axis. 30 bins is enough resolution for a typical
    # hand tool (axial_length ~ 50–300 mm) without making each bin so
    # narrow that it captures only a handful of polygon points.
    n_bins = 30
    edges = np.linspace(axial_min, axial_max, n_bins + 1)
    widths = np.full(n_bins, np.nan)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask_in = (axial >= lo) & (axial <= hi)
        if mask_in.sum() < 2:
            continue
        widths[i] = float(perp[mask_in].max() - perp[mask_in].min())
    valid = ~np.isnan(widths)
    if valid.sum() < 5:
        return floor

    # Tip widths: take the median of the outermost 2 bins on EACH end,
    # then use the minimum of the two ends. Asymmetric tools (e.g. a
    # screwdriver: sharp tip + chunky handle, or pruning shears: sharp
    # blades + thick pivot/handle) need the formula driven by the
    # sharper end — that's where SAM2 under-detects, and the symmetric
    # axial stretch will give the square end a bit of bonus clearance,
    # which is harmless.
    # Body widths: median of the middle 80% of bins, robust to the
    # occasional fat handle/grip bulge.
    n_tip = max(1, int(round(0.10 * n_bins)))
    n_outer = min(n_tip, 2)  # median of the outermost 2 bins per end
    left_end = widths[:n_outer]
    left_end = left_end[~np.isnan(left_end)]
    right_end = widths[-n_outer:]
    right_end = right_end[~np.isnan(right_end)]
    body_widths = widths[n_tip:-n_tip]
    body_widths = body_widths[~np.isnan(body_widths)]
    if left_end.size == 0 or right_end.size == 0 or body_widths.size == 0:
        return floor
    tip_w = float(min(np.median(left_end), np.median(right_end)))
    body_w = float(np.median(body_widths))
    if body_w < 1e-3:
        return floor

    taper = max(0.0, min(1.0, 1.0 - tip_w / body_w))
    axial_tol = floor + 0.014 * axial_length * taper

    print(f"  Auto axial tolerance: length={axial_length:.1f} mm, "
          f"tip_w={tip_w:.1f} mm (sharper end), body_w={body_w:.1f} mm, "
          f"taper={taper:.2f} → {axial_tol:.2f} mm")
    return float(axial_tol)


def _axial_stretch_polygons(polygons: list[list[tuple[float, float]]],
                            axial_extra_mm: float,
                            ) -> list[list[tuple[float, float]]]:
    """Add extra clearance along the principal axis only.

    Compensates for SAM2 mask under-detection at tapered tool tips: the
    uniform offset shrinks tip coverage proportionally more than handle
    coverage, so the printed pocket fits the wide section of the tool but
    is too tight at the ends. This stretches the polygon along its principal
    axis so each end gets `axial_extra_mm` of additional clearance, leaving
    the centerline (perpendicular extent) unchanged.

    The stretch is a linear ramp in the rotated frame: a point at
    fractional axial distance t (t ∈ [-1, 1] across the bbox) shifts
    outward by t * axial_extra_mm. This means features along the axis
    stretch slightly too — fine for typical hand tools, less ideal for
    tools with internal axis-parallel features (rare).
    """
    if abs(axial_extra_mm) < 0.001 or not polygons:
        return polygons

    angle = _principal_axis_angle(polygons)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Rotate-to-axis matrix: R^T  (so axis aligns with X in rotated frame).
    rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    inv_rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Project all points to the rotated frame to find centroid + extent.
    all_pts = np.array([p for poly in polygons for p in poly], dtype=float)
    rotated = all_pts @ rot.T
    cx = float(rotated[:, 0].mean())
    half_extent = float(max(rotated[:, 0].max() - cx, cx - rotated[:, 0].min()))
    if half_extent < 1e-6:
        return polygons

    # Stretch factor: a point at half_extent gets pushed outward by
    # axial_extra_mm. (half_extent + axial_extra) / half_extent is the
    # multiplier on the centered axial coordinate.
    scale_x = (half_extent + axial_extra_mm) / half_extent

    out: list[list[tuple[float, float]]] = []
    for poly in polygons:
        arr = np.array(poly, dtype=float)
        rot_pts = arr @ rot.T
        rot_pts[:, 0] = cx + (rot_pts[:, 0] - cx) * scale_x
        back = rot_pts @ inv_rot.T
        out.append([(float(x), float(y)) for x, y in back])
    return out


def _resample_polygon(polygon: list[tuple[float, float]],
                       step: float) -> np.ndarray:
    """Resample a closed polygon at uniform arc-length spacing.

    Returns an (N, 2) float array. Used as a prerequisite for Gaussian
    contour smoothing — the convolution kernel is a function of arc
    length, so the input must have uniform spacing.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    if len(pts) < 3:
        return pts
    closed = np.vstack([pts, pts[:1]])
    seg = np.diff(closed, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    perimeter = float(arc[-1])
    if perimeter < 1e-9:
        return pts
    n = max(8, int(np.ceil(perimeter / step)))
    targets = np.linspace(0.0, perimeter, n, endpoint=False)
    out = np.empty((n, 2), dtype=np.float64)
    out[:, 0] = np.interp(targets, arc, closed[:, 0])
    out[:, 1] = np.interp(targets, arc, closed[:, 1])
    return out


def _smooth_polygon_coords(polygon: list[tuple[float, float]],
                            sigma_mm: float,
                            sigma_axial_mm: float = 0.5,
                            resample_step_mm: float = 0.3,
                            corner_preserve_deg: float = 60.0,
                            convex_corner_threshold_deg: float = 30.0,
                            ) -> list[tuple[float, float]]:
    """Anisotropic, curvature-aware Gaussian-smooth a closed polygon.

    Decomposes each point's offset from the centroid into a component along
    the polygon's principal axis (axial) and a component perpendicular to
    it, then smooths each component with its own sigma along arc length
    (wrap-around so the start/end join stays seamless).

    The asymmetry exists to remove wave noise on the sides of an elongated
    tool (mostly perpendicular displacement) without shortening sharp tips
    (which want their axial extent preserved). A second mechanism — local
    curvature gating — blends the smoothed result back toward the original
    at sharp corners, so brackets/braces with intentional convex
    protrusions don't get rounded off the way reflective scissor blades
    get smoothed.

    Args:
        sigma_mm: Perpendicular Gaussian sigma in mm. Larger = more side
            smoothing. Set to 0 to disable.
        sigma_axial_mm: Along-principal-axis sigma in mm. Kept small
            (default 0.5) to preserve tip-to-tip length.
        resample_step_mm: Uniform arc-length spacing used during the
            convolution.
        corner_preserve_deg: Turning-angle threshold above which a vertex
            is treated as a "real" corner and the original position is
            preserved. The blend uses a Gaussian falloff in the turning
            angle so the transition is smooth, not binary.
        convex_corner_threshold_deg: Lower threshold (default 30°) used
            specifically to detect convex sharp corners on the original
            input polygon for full neighborhood preservation. Set lower
            than ``corner_preserve_deg`` because real-world tools have
            corners that aren't always 90° (e.g. a trigger guard's
            angle is closer to 50–60° turn) and the user-visible cost
            of clipping a convex feature is much worse than the cost of
            slightly under-smoothing a curve.

    Falls back to the input on degenerate input (under 3 points).
    """
    perp_sigma = max(0.0, sigma_mm)
    axial_sigma = max(0.0, sigma_axial_mm)
    if (perp_sigma <= 0 and axial_sigma <= 0) or len(polygon) < 3:
        return polygon

    pts = _resample_polygon(polygon, resample_step_mm)
    n = len(pts)
    if n < 3:
        return polygon

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA: principal axis is the eigenvector with the larger eigenvalue.
    cov = np.cov(centered.T)
    if not np.all(np.isfinite(cov)):
        return polygon
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = int(np.argmax(eigvals))
    principal = eigvecs[:, idx]
    secondary = eigvecs[:, 1 - idx]

    axial = centered @ principal     # (n,) signed offsets along principal axis
    perp = centered @ secondary      # (n,) signed offsets perpendicular to it

    def _smooth_1d(data: np.ndarray, sigma_mm_: float) -> np.ndarray:
        if sigma_mm_ <= 0:
            return data
        sigma_samples = sigma_mm_ / resample_step_mm
        radius = max(1, int(np.ceil(sigma_samples * 3)))
        if radius >= n // 2:
            radius = max(1, n // 2 - 1)
        x = np.arange(2 * radius + 1, dtype=np.float64) - radius
        kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
        kernel /= kernel.sum()
        padded = np.concatenate([data[-radius:], data, data[:radius]])
        return np.convolve(padded, kernel, mode="valid")

    axial_s = _smooth_1d(axial, axial_sigma)
    perp_s = _smooth_1d(perp, perp_sigma)

    # Per-point corner-preservation blend. Compute the turning angle
    # twice — once on the resampled polygon (for the mild Gaussian
    # falloff at every vertex) and once on the original input polygon
    # (so we can detect convex sharp corners robustly).
    #
    # Why detect on the original: ``_resample_polygon`` interpolates
    # linearly along arc length without preserving the input's
    # vertices, so a single 90° corner ends up split across two or three
    # adjacent ~45° subvertices in the resampled output. Each is below
    # the default 60° threshold individually, so a per-resampled-vertex
    # detector misses the corner entirely (the shape ends up rounded by
    # the perpendicular smoothing). The original polygon, however, has
    # the corner as one vertex with a clean 90° turn.
    #
    # Why convex vs concave matters: at convex (outward-pointing) sharp
    # corners — the corner of a trigger guard, a hammer claw, etc. —
    # smoothing pulls the polygon inward and visibly clips the feature.
    # Concave sharp corners (inward notches) get the existing turn-angle
    # Gaussian falloff; gentle rounding there is desirable for tool
    # clearance.
    if corner_preserve_deg > 0:
        threshold_rad = np.deg2rad(corner_preserve_deg)

        # --- Resampled-vertex turn angles → base smoothing weight ---
        seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)  # (n, 2)
        seg_len = np.linalg.norm(seg, axis=1, keepdims=True)
        seg_len = np.where(seg_len < 1e-9, 1.0, seg_len)
        seg_unit = seg / seg_len
        prev_unit = np.roll(seg_unit, 1, axis=0)
        cos_turn = np.clip(
            (prev_unit * seg_unit).sum(axis=1), -1.0, 1.0
        )
        turn_rad = np.arccos(cos_turn)
        # Gaussian falloff in turn angle: weight 1 at turn=0, weight ≈0.05
        # at turn=corner_preserve_deg, ≈0 well above.
        smooth_weight = np.exp(-(turn_rad / threshold_rad) ** 2)

        # --- Original-polygon convex-sharp detection ---
        # Per-vertex turn angle alone can't distinguish a real tool corner
        # from a wave-noise peak: SAM2 traces of reflective surfaces have
        # 0.5 mm-amplitude waves with ≈3 mm wavelength, and sampled
        # densely enough each peak hits ~90° turn at the apex vertex.
        # The discriminator is *integrated* signed turn over a small arc-
        # length window: a real corner concentrates its turn into one
        # vertex with smooth shoulders, so the integral over a few mm
        # stays close to the corner angle. A wave peak alternates with
        # an adjacent concave valley, so the signed integral over ≥ one
        # wavelength averages near zero.
        orig_pts = np.asarray(polygon, dtype=np.float64)
        if len(orig_pts) >= 3:
            orig_seg = np.diff(np.vstack([orig_pts, orig_pts[:1]]), axis=0)
            orig_seg_len_arr = np.linalg.norm(orig_seg, axis=1)
            orig_seg_len_safe = np.where(
                orig_seg_len_arr < 1e-9, 1.0, orig_seg_len_arr
            )
            orig_seg_unit = orig_seg / orig_seg_len_safe[:, None]
            orig_prev_unit = np.roll(orig_seg_unit, 1, axis=0)
            orig_cos_turn = np.clip(
                (orig_prev_unit * orig_seg_unit).sum(axis=1), -1.0, 1.0
            )
            orig_turn_rad = np.arccos(orig_cos_turn)
            orig_cross_z = (
                orig_prev_unit[:, 0] * orig_seg_unit[:, 1]
                - orig_prev_unit[:, 1] * orig_seg_unit[:, 0]
            )
            ccw_polygon = _signed_area(polygon) >= 0.0
            orig_is_convex = (
                (orig_cross_z > 0) if ccw_polygon else (orig_cross_z < 0)
            )
            convex_threshold_rad = np.deg2rad(convex_corner_threshold_deg)

            # Signed turn: positive for convex (left turns on CCW), negative
            # for concave (right turns). Integrate over a 5 mm window —
            # wider than typical SAM2 wave wavelength (≈3 mm on
            # reflective surfaces) so peaks and valleys within the wave
            # cancel, narrower than tool-feature spacing so adjacent real
            # corners don't confuse each other.
            orig_signed_turn = np.where(
                orig_is_convex, orig_turn_rad, -orig_turn_rad
            )
            arc_at_vertex = np.concatenate(
                [[0.0], np.cumsum(orig_seg_len_arr)[:-1]]
            )
            perimeter = float(orig_seg_len_arr.sum())
            window_mm = 5.0
            half_w = window_mm / 2.0
            # Triple-tile arc length + signed turn for wrap-around.
            arc_3x = np.concatenate([
                arc_at_vertex - perimeter,
                arc_at_vertex,
                arc_at_vertex + perimeter,
            ])
            turn_3x = np.tile(orig_signed_turn, 3)
            cum_turn_3x = np.concatenate([[0.0], np.cumsum(turn_3x)])
            integrated_turn = np.zeros(len(orig_pts))
            for i in range(len(orig_pts)):
                center = arc_at_vertex[i]
                lo = int(np.searchsorted(
                    arc_3x, center - half_w, side="left"
                ))
                hi = int(np.searchsorted(
                    arc_3x, center + half_w, side="right"
                ))
                integrated_turn[i] = (
                    cum_turn_3x[hi] - cum_turn_3x[lo]
                )

            # A vertex is a "real" convex sharp corner when:
            #   - its integrated signed turn over the window is at least
            #     the threshold (so SAM2 wave noise is filtered out), AND
            #   - its own per-vertex turn is convex AND above the threshold
            #     (so we have a clean local apex to anchor the preservation
            #     zone on).
            orig_convex_sharp_idx = np.where(
                (integrated_turn >= convex_threshold_rad)
                & (orig_turn_rad >= convex_threshold_rad)
                & orig_is_convex
            )[0]
        else:
            orig_convex_sharp_idx = np.array([], dtype=int)
            orig_pts = np.empty((0, 2), dtype=np.float64)

        # Map each original-polygon convex-sharp vertex onto its nearest
        # resampled vertex, then spread that single-point indicator
        # along arc length by the same sigma we use for the smoothing
        # kernel — the resulting "preserve zone" exactly matches the
        # region whose neighbors would otherwise be pulled inward by the
        # kernel. Normalize the zone so a single isolated corner peaks
        # at 1.0 (otherwise Gaussian-convolution normalization scales
        # the peak down by ~1/(σ·√(2π)·samples), which would barely
        # dent the smooth weight).
        convex_sharp_indicator = np.zeros(n, dtype=np.float64)
        for idx in orig_convex_sharp_idx:
            corner_pos = orig_pts[idx]
            d = np.linalg.norm(pts - corner_pos, axis=1)
            convex_sharp_indicator[int(np.argmin(d))] = 1.0

        if convex_sharp_indicator.any() and perp_sigma > 0:
            # Spread the corner indicator with a sigma WIDER than the
            # smoothing kernel itself. The smoothing kernel averages
            # points within ~3·perp_sigma of arc length; if the zone
            # Gaussian has the SAME sigma, neighbors still inside the
            # smoothing reach but outside the zone's effective support
            # get fully smoothed and pull the corner inward despite the
            # vertex being preserved. 2.5×perp_sigma keeps the zone ≥0.5
            # out to 3×perp_sigma, which is roughly the distance over
            # which a sharp corner needs its neighbors held in place to
            # actually look sharp.
            zone_sigma = perp_sigma * 2.5
            zone = _smooth_1d(convex_sharp_indicator, zone_sigma)
            z_max = float(zone.max())
            if z_max > 1e-9:
                zone = zone / z_max
            # Square-root the normalized zone for a flatter plateau at
            # the corner (zone^0.5 stays close to 1 over a wider range
            # before falling off), so the preserved region is shaped
            # less like a single Gaussian peak and more like a held
            # neighborhood with smooth shoulders.
            zone = np.sqrt(np.clip(zone, 0.0, 1.0))
            # Inside the zone, force smooth_weight toward 0 (full
            # preservation). 1 − zone is 0 at the corner and rises back
            # to 1 outside the smoothing kernel's reach.
            smooth_weight = smooth_weight * (1.0 - zone)

        # Blend per-component: w·smoothed + (1-w)·original
        axial_s = smooth_weight * axial_s + (1.0 - smooth_weight) * axial
        perp_s = smooth_weight * perp_s + (1.0 - smooth_weight) * perp

    # Outward bias: ensure the smoothed polygon never shrinks below the
    # original's outward extent in the PCA frame. For each vertex, if
    # smoothing pulled |axial| or |perp| toward zero (i.e. toward the
    # centroid), revert that component to the original. This guarantees
    # the trace covers everything the input polygon covered — preventing
    # the trace from clipping the actual tool, which was the user's
    # primary complaint when reflective trim or curved features were
    # being cut off by the smoothing.
    #
    # The trade-off: SAM2 wave noise on long edges (e.g. scissor blades)
    # has its outward peaks preserved instead of fully averaged out, so
    # the trace shows mild bumps where it used to be perfectly flat. In
    # practice the bumps are sub-millimeter and the bin pocket just ends
    # up a hair wider in those regions, which is preferable to clipping
    # a real tool feature.
    # Use the sign of the ORIGINAL component for the bias clamp. The
    # smoothed component might have flipped sign (rare, only happens
    # with aggressive smoothing across the principal axis); when that
    # occurs we still want to preserve the original outward direction.
    abs_axial_orig = np.abs(axial)
    abs_perp_orig = np.abs(perp)
    axial_s = np.where(
        np.abs(axial_s) < abs_axial_orig,
        np.sign(axial) * abs_axial_orig,
        axial_s,
    )
    perp_s = np.where(
        np.abs(perp_s) < abs_perp_orig,
        np.sign(perp) * abs_perp_orig,
        perp_s,
    )

    out = (centroid
           + np.outer(axial_s, principal)
           + np.outer(perp_s, secondary))
    return [(float(x), float(y)) for x, y in out]


def _simplify_polygon(polygon: list[tuple[float, float]],
                       epsilon: float = 0.3) -> list[tuple[float, float]]:
    """Simplify polygon using the Douglas-Peucker algorithm.

    Removes points within epsilon (mm) of the simplified line,
    smoothing out small curves and bumps while preserving overall shape.
    """
    if len(polygon) < 3:
        return polygon

    points = np.array(polygon, dtype=np.float64)

    def _rdp(pts, eps):
        if len(pts) < 3:
            return pts

        start = pts[0]
        end = pts[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            dists = np.linalg.norm(pts - start, axis=1)
        else:
            line_unit = line_vec / line_len
            vecs = pts - start
            proj = np.dot(vecs, line_unit)
            proj = np.clip(proj, 0, line_len)
            closest = start + np.outer(proj, line_unit)
            dists = np.linalg.norm(pts - closest, axis=1)

        max_idx = np.argmax(dists)
        max_dist = dists[max_idx]

        if max_dist > eps:
            left = _rdp(pts[:max_idx + 1], eps)
            right = _rdp(pts[max_idx:], eps)
            return np.vstack([left[:-1], right])
        else:
            return np.array([start, end])

    simplified = _rdp(points, epsilon)
    return [(float(x), float(y)) for x, y in simplified]


def _signed_area(polygon: list[tuple[float, float]]) -> float:
    """Signed area via the shoelace formula. Positive = CCW, negative = CW."""
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s * 0.5


def _round_sharp_corners(polygon: list[tuple[float, float]],
                          radius: float = 1.5,
                          min_angle_deg: float = 90,
                          n_arc_points: int = 8,
                          concave_only: bool = True,
                          ) -> list[tuple[float, float]]:
    """Replace sharp corners with fillet arcs.

    Any vertex where the interior angle is less than min_angle_deg gets
    replaced by an arc of the given radius, smoothing the sharp point.

    Args:
        polygon: List of (x, y) tuples
        radius: Fillet radius in mm (default 1.5)
        min_angle_deg: Threshold — corners sharper than this are rounded
        n_arc_points: Number of points to generate along each fillet arc
        concave_only: When True (default), skip convex corners. Convex
            corners are outward protrusions; rounding them shrinks the
            polygon and clips features (e.g. bracket corners that the
            physical tool actually has). Concave corners are inward
            notches; rounding those just adds clearance and helps Fusion
            cut without sub-tool-radius inside corners.
    """
    if len(polygon) < 3:
        return polygon

    # Determine winding so the convex/concave sign convention is stable.
    # For CCW polygons, the cross product of (curr→next) × (curr→prev) is
    # positive at convex vertices. We flip the sign for CW input so the
    # downstream convex-detection logic stays simple.
    ccw = _signed_area(polygon) >= 0.0

    result = []
    n = len(polygon)

    for i in range(n):
        p_prev = np.array(polygon[(i - 1) % n])
        p_curr = np.array(polygon[i])
        p_next = np.array(polygon[(i + 1) % n])

        v1 = p_prev - p_curr  # toward previous vertex
        v2 = p_next - p_curr  # toward next vertex

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < 1e-10 or len2 < 1e-10:
            result.append(polygon[i])
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1, 1)
        angle = np.arccos(cos_angle)  # interior angle at this vertex

        if np.degrees(angle) >= min_angle_deg:
            result.append(polygon[i])
            continue

        # Skip convex corners when concave_only is set. Cross of the
        # incoming edge direction × outgoing edge direction: a left
        # turn (positive cross on a CCW polygon) is a convex vertex.
        # Sign flips for CW input.
        #
        # v1 = prev - curr  (toward previous vertex, opposite of incoming direction)
        # v2 = next - curr  (already the outgoing direction, since outgoing = next - curr)
        if concave_only:
            e_in = -v1   # incoming edge direction = curr - prev
            e_out = v2   # outgoing edge direction = next - curr
            cross = float(e_in[0] * e_out[1] - e_in[1] * e_out[0])
            is_convex = cross > 0 if ccw else cross < 0
            if is_convex:
                result.append(polygon[i])
                continue

        # Sharp corner — compute fillet arc
        half_angle = angle / 2
        tan_ha = np.tan(half_angle)
        sin_ha = np.sin(half_angle)
        if tan_ha < 1e-10 or sin_ha < 1e-10:
            result.append(polygon[i])
            continue

        d = radius / tan_ha  # tangent distance along each edge

        # Clamp if tangent length exceeds available edge
        max_d = min(len1, len2) * 0.4
        if d > max_d:
            d = max_d
            actual_r = d * tan_ha
        else:
            actual_r = radius

        u1 = v1 / len1
        u2 = v2 / len2
        t1 = p_curr + d * u1  # tangent point on edge toward previous
        t2 = p_curr + d * u2  # tangent point on edge toward next

        # Arc center along the angle bisector, offset inward
        bisector = u1 + u2
        bis_len = np.linalg.norm(bisector)
        if bis_len < 1e-10:
            result.append(polygon[i])
            continue
        bisector_unit = bisector / bis_len

        center_dist = actual_r / np.sin(half_angle)
        center = p_curr + center_dist * bisector_unit

        # Determine arc sweep from t1 to t2
        a_start = np.arctan2(t1[1] - center[1], t1[0] - center[0])
        a_end = np.arctan2(t2[1] - center[1], t2[0] - center[0])

        # The correct arc is on the opposite side of center from the vertex.
        # Check which sweep direction (CW or CCW) passes through that side.
        TWO_PI = 2 * np.pi
        mid_angle = np.arctan2(-bisector_unit[1], -bisector_unit[0])
        delta_ccw = (a_end - a_start) % TWO_PI

        if ((mid_angle - a_start) % TWO_PI) < delta_ccw:
            delta = delta_ccw
        else:
            delta = -(TWO_PI - delta_ccw)

        for j in range(n_arc_points + 1):
            t = j / n_arc_points
            a = a_start + t * delta
            px = center[0] + actual_r * np.cos(a)
            py = center[1] + actual_r * np.sin(a)
            result.append((float(px), float(py)))

    return result


def _compute_bbox(polygons: list[list[tuple[float, float]]]) -> dict:
    """Compute bounding box of all polygons."""
    all_x = []
    all_y = []
    for poly in polygons:
        for x, y in poly:
            all_x.append(x)
            all_y.append(y)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width_mm": max_x - min_x,
        "height_mm": max_y - min_y,
    }


def _generate_stadium(center: tuple[float, float],
                       axis_dir: tuple[float, float],
                       length: float, width: float,
                       n_arc: int = 12) -> list[tuple[float, float]]:
    """Generate a stadium (obround) polygon — a rectangle with semicircular ends.

    Args:
        center: (x, y) center point
        axis_dir: (dx, dy) direction along the long axis (will be normalized)
        length: total end-to-end length including semicircles
        width: total width (diameter of semicircular ends)
        n_arc: number of points per semicircle
    """
    R = width / 2
    S = max((length - width) / 2, 0)
    ax, ay = float(axis_dir[0]), float(axis_dir[1])
    norm = math.sqrt(ax * ax + ay * ay)
    if norm < 1e-10:
        return []
    ax, ay = ax / norm, ay / norm
    cx, cy = float(center[0]), float(center[1])

    local_points = []
    # Right semicircle (positive end of long axis)
    for i in range(n_arc + 1):
        angle = -math.pi / 2 + math.pi * i / n_arc
        local_points.append((S + R * math.cos(angle), R * math.sin(angle)))
    # Left semicircle (negative end of long axis)
    for i in range(n_arc + 1):
        angle = math.pi / 2 + math.pi * i / n_arc
        local_points.append((-S + R * math.cos(angle), R * math.sin(angle)))

    # Rotate and translate to global coordinates
    points = []
    for lx, ly in local_points:
        gx = cx + lx * ax - ly * ay
        gy = cy + lx * ay + ly * ax
        points.append((gx, gy))
    return points


def compute_finger_slot(path, scale: float, clearance_mm: float = 0.0,
                        img_shape: tuple = None, gridfinity_unit: float = 42.0,
                        min_slot_width: float = 20.0,
                        max_slot_width: float = 40.0) -> list[tuple[float, float]] | None:
    """Compute a finger-access slot for lifting the tool from a gridfinity bin.

    Analyzes the tool polygon to find the handle/shaft region (the narrowest
    contiguous section) and places a stadium-shaped slot there. The slot is
    sized to stay within the tool's current gridfinity unit footprint.

    Args:
        path: potrace.Path object
        scale: mm per pixel
        clearance_mm: clearance offset applied to inner outline
        img_shape: (height, width) of source mask
        gridfinity_unit: base unit size in mm (default 42)
        min_slot_width: minimum slot narrow dimension in mm (default 20)
        max_slot_width: maximum slot narrow dimension in mm (default 40)

    Returns:
        List of (x, y) tuples for the slot polygon, or None if placement fails.
    """
    polygons = _potrace_curves_to_polygons(path, scale, img_shape=img_shape)
    if clearance_mm > 0.001:
        polygons = _offset_polygons(polygons, clearance_mm)
    if not polygons:
        return None

    # Use the largest polygon (main tool outline)
    polygon = max(polygons, key=len)
    points = np.array(polygon)
    n_pts = len(polygon)

    # --- Step 1: Principal axis via PCA ---
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = int(np.argmax(eigenvalues))
    principal = eigenvectors[:, idx]
    secondary = eigenvectors[:, 1 - idx]

    # --- Step 2: Width profile via cross-section ray casting ---
    proj_along = centered @ principal
    p_min, p_max = float(proj_along.min()), float(proj_along.max())

    n_slices = 80
    positions = np.linspace(p_min + 1, p_max - 1, n_slices)
    total_widths = np.zeros(n_slices)
    slice_regions = [[] for _ in range(n_slices)]  # [(perp_center, width), ...]

    for k in range(n_slices):
        pos = positions[k]
        perp_ints = []
        for i in range(n_pts):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n_pts]
            v1 = np.array([x1, y1]) - centroid
            v2 = np.array([x2, y2]) - centroid
            p1 = float(v1 @ principal)
            p2 = float(v2 @ principal)
            if (p1 - pos) * (p2 - pos) > 0:
                continue
            dp = p2 - p1
            if abs(dp) < 1e-10:
                continue
            t = (pos - p1) / dp
            inter = np.array([x1, y1]) + t * (np.array([x2, y2]) - np.array([x1, y1]))
            perp_ints.append(float((inter - centroid) @ secondary))

        perp_ints.sort()
        if len(perp_ints) >= 2:
            total_widths[k] = perp_ints[-1] - perp_ints[0]
            for j in range(0, len(perp_ints) - 1, 2):
                w = perp_ints[j + 1] - perp_ints[j]
                c = (perp_ints[j + 1] + perp_ints[j]) / 2
                slice_regions[k].append((c, w))

    # --- Step 3: Find handle zone ---
    valid = total_widths > 0
    if not np.any(valid):
        return None

    # Count material regions per slice — 2+ regions means separated handles (pliers)
    region_counts = np.array([len(r) for r in slice_regions])

    # Primary detector: slices with 2+ material regions (e.g., two pliers handles)
    multi_region_mask = valid & (region_counts >= 2)

    if np.sum(multi_region_mask) >= 5:
        # Pliers-type tool: use multi-region detection
        handle_mask = multi_region_mask
    else:
        # Single-handle tool (screwdriver, wrench): fall back to width threshold
        median_w = float(np.median(total_widths[valid]))
        threshold = median_w * 0.65
        handle_mask = valid & (total_widths < threshold)

        if np.sum(handle_mask) < 3:
            sorted_w = np.sort(total_widths[valid])
            cutoff = max(1, int(len(sorted_w) * 0.4))
            threshold = float(sorted_w[cutoff])
            handle_mask = valid & (total_widths <= threshold)

    # Longest contiguous run of narrow slices
    runs = []
    start = None
    for i in range(len(handle_mask)):
        if handle_mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, len(handle_mask) - 1))

    if not runs:
        return None

    best_run = max(runs, key=lambda r: r[1] - r[0])
    run_start, run_end = best_run
    run_mid = (run_start + run_end) // 2

    # Anchor placement at the global center along the principal axis.
    axis_center = (p_min + p_max) / 2.0
    center_idx = int(np.argmin(np.abs(positions - axis_center)))

    # --- Step 4–6: Place slot, trying positions from geometric center outward ---
    bbox_tool = _compute_bbox([polygon])
    bin_w = math.ceil(bbox_tool['width_mm'] / gridfinity_unit)
    bin_h = math.ceil(bbox_tool['height_mm'] / gridfinity_unit)

    # Slot is always parallel to the X axis (horizontal) and must extend
    # at least min_overhang beyond the tool outline on each side.
    # Ideal overhang is larger; bin expansion is allowed to meet the minimum.
    is_multi_region = np.sum(multi_region_mask) >= 5
    axis = (1.0, 0.0)  # always horizontal
    min_overhang = 10.0   # mm — hard minimum, expand bin if needed
    ideal_overhang = 20.0  # mm — preferred if it fits

    # slot_width = narrow dimension (vertical)
    # slot_length = long dimension (horizontal)
    slot_width = float(np.clip(min_slot_width, min_slot_width, max_slot_width))

    # Measure tool width at handle zone to size the slot
    handle_total_w = float(total_widths[run_mid]) if total_widths[run_mid] > 0 else 30
    ideal_length = handle_total_w + 2 * ideal_overhang
    min_length = handle_total_w + 2 * min_overhang

    # Search any valid slice along the principal axis, sorted by distance to
    # the geometric midpoint. The narrow-run logic above informs slot sizing
    # and the multi-region (pliers) detector, but we don't restrict candidate
    # positions to the run — for tools whose only narrow region sits at one
    # end (scissors: heart-shaped head + long blade), constraining candidates
    # to the run forced the slot away from the tool's middle.
    candidate_indices = sorted(
        [i for i in range(n_slices) if valid[i]],
        key=lambda i: abs(i - center_idx)
    )

    slot_poly = None
    for pos_idx in candidate_indices:
        regions = slice_regions[pos_idx]
        if not regions:
            continue

        if is_multi_region and len(regions) >= 2:
            # Place slot in the gap between handles
            sorted_r = sorted(regions, key=lambda r: r[0])
            g_start = sorted_r[0][0] + sorted_r[0][1] / 2
            g_end = sorted_r[1][0] - sorted_r[1][1] / 2
            perp_center = (g_start + g_end) / 2
        else:
            # Center on the widest region
            best_region = max(regions, key=lambda r: r[1])
            perp_center = best_region[0]

        cand_center = centroid + positions[pos_idx] * principal + perp_center * secondary
        ctr = (float(cand_center[0]), float(cand_center[1]))

        # Try ideal length first, shrink toward min_length, allow bin expansion
        for trial_l in np.arange(ideal_length, min_length - 1, -2):
            trial_poly = _generate_stadium(ctr, axis, float(trial_l), slot_width)
            trial_bbox = _compute_bbox([polygon, trial_poly])
            if (math.ceil(trial_bbox['width_mm'] / gridfinity_unit) <= bin_w and
                    math.ceil(trial_bbox['height_mm'] / gridfinity_unit) <= bin_h):
                slot_poly = trial_poly
                slot_length = float(trial_l)
                break

        # If ideal didn't fit without expanding, use min_length (allow bin expansion)
        if slot_poly is None:
            trial_poly = _generate_stadium(ctr, axis, min_length, slot_width)
            slot_poly = trial_poly
            slot_length = min_length

        if slot_poly is not None:
            break

    if slot_poly is None:
        print("  Finger slot: skipped (no valid handle position found)")
        return None

    print(f"  Finger slot: {slot_width:.0f}x{slot_length:.0f}mm at handle region")
    return slot_poly


def _polygons_to_svg_paths(polygons: list[list[tuple[float, float]]]) -> list[str]:
    """Convert polygon point lists to SVG path data strings."""
    svg_paths = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        d = f"M {poly[0][0]:.3f},{poly[0][1]:.3f}"
        for x, y in poly[1:]:
            d += f" L {x:.3f},{y:.3f}"
        d += " Z"
        svg_paths.append(d)
    return svg_paths




def potrace_to_svg(path, output_path: str, scale: float,
                   clearance_mm: float = 0.0, tolerance_mm: float = 0.0,
                   img_shape: tuple = None, simplify_epsilon: float = 0.3,
                   slot_polygon: list = None,
                   display_smooth_sigma_mm: float = 2.5) -> dict:
    """Export potrace path to SVG, optionally with a tolerance outline.

    When tolerance_mm is non-zero, two outlines are written:
    - Inner: the accurate tool trace (native Bezier curves when clearance=0)
    - Outer: a smoothed tolerance perimeter, tolerance_mm beyond the inner outline.
            Positive values expand the perimeter (clearance fit); negative
            values shrink it (interference fit).

    Args:
        path: potrace.Path object
        output_path: Where to save the SVG
        scale: mm per pixel
        clearance_mm: Outward offset applied to the inner outline (default 0)
        tolerance_mm: Additional outward offset for the tolerance perimeter
        img_shape: (height, width) of source mask to filter boundary artifacts
        simplify_epsilon: Douglas-Peucker simplification threshold in mm for
                          the tolerance outline (default 0.3mm)
        display_smooth_sigma_mm: Gaussian sigma (mm) for smoothing the inner
                          display polygon before DP simplification, to remove
                          SAM2 wave noise on reflective surfaces (default 1.5)

    Returns:
        Bounding box dict with width_mm, height_mm
    """
    # Build inner (tool cutout) paths. The inner trace is Gaussian-smoothed
    # along the contour and then DP-simplified so reflective surfaces don't
    # propagate SAM2 mask noise into a visibly wavy reference line on the
    # printed fit-test. The cleanup pipeline's per-mask smoothing is tuned
    # conservatively to preserve concavities (pliers handle gaps), so we
    # smooth more aggressively here at the polygon level.
    polygons = _potrace_curves_to_polygons(path, scale, img_shape=img_shape)
    if clearance_mm > 0.001:
        base_polygons = _offset_polygons(polygons, clearance_mm)
    else:
        base_polygons = polygons

    inner_polygons = [_smooth_polygon_coords(p, sigma_mm=display_smooth_sigma_mm)
                      for p in base_polygons]
    inner_polygons = [_simplify_polygon(p, epsilon=simplify_epsilon)
                      for p in inner_polygons]
    inner_polygons = [_round_sharp_corners(p) for p in inner_polygons]
    inner_svg_paths = _polygons_to_svg_paths(inner_polygons)
    bbox_polygons = list(inner_polygons)

    # Build outer tolerance paths from the smoothed inner so the dashed
    # tolerance line stays parallel to the printed inner reference. Always
    # generated (even at offset 0) so the Fusion cut consumes a DP-simplified
    # polygon — falling back to raw potrace would give hundreds of points and
    # freeze Fusion. _offset_polygons no-ops at offset 0.
    outer_polygons = _offset_polygons(inner_polygons, tolerance_mm)
    outer_polygons = [_simplify_polygon(p, epsilon=simplify_epsilon)
                      for p in outer_polygons]
    outer_polygons = [_round_sharp_corners(p) for p in outer_polygons]
    outer_svg_paths = _polygons_to_svg_paths(outer_polygons)
    bbox_polygons.extend(outer_polygons)

    # Finger access slot
    slot_svg_paths = []
    if slot_polygon is not None:
        slot_svg_paths = _polygons_to_svg_paths([slot_polygon])
        bbox_polygons.append(slot_polygon)

    bbox = _compute_bbox(bbox_polygons)
    _write_svg(inner_svg_paths, outer_svg_paths, slot_svg_paths, bbox, output_path)
    return bbox


def _write_svg(inner_paths: list[str], outer_paths: list[str],
               slot_paths: list[str], bbox: dict, output_path: str):
    """Write SVG file with inner, outer, and slot path groups.

    Inner paths (tool cutout) use a solid stroke.
    Outer paths (tolerance perimeter) use a dashed stroke.
    Slot paths (finger access) use a dotted stroke.

    Fusion 360 interprets SVG viewBox coordinates as cm (its native unit).
    All coordinates are converted from mm to cm so Fusion imports at correct scale.

    Path data is in Y-up (CAD) convention; SVG viewBox is Y-down. We apply a
    scale(1, -1) in the group transform so the shape renders right-side-up in
    browsers and matches the photo orientation.
    """
    margin_mm = 2
    # Convert everything to cm for Fusion 360 compatibility
    MM_TO_CM = 0.1
    margin = margin_mm * MM_TO_CM
    width = bbox["width_mm"] * MM_TO_CM + 2 * margin
    height = bbox["height_mm"] * MM_TO_CM + 2 * margin
    # With scale(1,-1): y_svg = tY - y_data. Choose tY so y_data=max_y maps to margin
    # and y_data=min_y maps to height-margin.
    offset_x = -bbox["min_x"] * MM_TO_CM + margin
    offset_y = bbox["max_y"] * MM_TO_CM + margin

    # Scale all path coordinates from mm to cm
    scaled_inner = [_scale_svg_path_coords(d, MM_TO_CM) for d in inner_paths]
    scaled_outer = [_scale_svg_path_coords(d, MM_TO_CM) for d in outer_paths]
    scaled_slot = [_scale_svg_path_coords(d, MM_TO_CM) for d in slot_paths]

    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.4f}cm" height="{height:.4f}cm"
     viewBox="0 0 {width:.4f} {height:.4f}">
  <g transform="translate({offset_x:.4f},{offset_y:.4f}) scale(1,-1)">
'''

    # Inner paths (tool cutout) - solid stroke
    for d in scaled_inner:
        svg_content += f'    <path d="{d}" fill="none" stroke="black" stroke-width="0.01"/>\n'

    # Outer paths (tolerance perimeter) - dashed stroke
    for d in scaled_outer:
        svg_content += f'    <path d="{d}" fill="none" stroke="black" stroke-width="0.01" stroke-dasharray="0.05,0.03"/>\n'

    # Slot paths (finger access) - dotted stroke
    for d in scaled_slot:
        svg_content += f'    <path d="{d}" fill="none" stroke="black" stroke-width="0.01" stroke-dasharray="0.02,0.02"/>\n'

    svg_content += '''  </g>
</svg>
'''

    with open(output_path, 'w') as f:
        f.write(svg_content)


def _scale_svg_path_coords(d: str, factor: float) -> str:
    """Scale all numeric coordinates in an SVG path data string by factor."""
    import re
    parts = re.split(r'([MCLZmclz])', d)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in 'MCLZmclz':
            result.append(part)
        else:
            # Scale all numbers in this segment
            nums = part.split()
            scaled = []
            for num in nums:
                # Handle comma-separated coordinate pairs like "1.234,5.678"
                coords = num.split(',')
                scaled_coords = [f"{float(c) * factor:.4f}" for c in coords]
                scaled.append(','.join(scaled_coords))
            result.append(' '.join(scaled))
    return ' '.join(result)


def potrace_to_dxf(path, output_path: str, scale: float,
                   clearance_mm: float = 0.0, tolerance_mm: float = 0.0,
                   axial_tolerance_mm=0.0,
                   img_shape: tuple = None, simplify_epsilon: float = 0.3,
                   slot_polygon: list = None,
                   display_smooth_sigma_mm: float = 2.5):
    """Export potrace path to DXF format, optionally with a tolerance outline.

    When tolerance_mm is non-zero, the inner (tool cutout) outline stays on
    the default layer "0", and the tolerance perimeter is placed on a
    "TOLERANCE" layer. Positive values expand the perimeter (clearance fit);
    negative values shrink it (interference fit).

    Args:
        path: potrace.Path object
        output_path: Where to save the DXF
        scale: mm per pixel
        clearance_mm: Outward offset for printing tolerance
        tolerance_mm: Additional outward offset for the tolerance perimeter
        axial_tolerance_mm: Extra clearance along the tool's principal axis
            (each end pushed outward by this amount), applied after the
            uniform tolerance. Compensates for SAM2 under-detection at
            tapered tool tips.
        img_shape: (height, width) of source mask to filter boundary artifacts
        simplify_epsilon: Douglas-Peucker simplification threshold in mm
        display_smooth_sigma_mm: Gaussian sigma (mm) for smoothing the inner
            display polygon before DP simplification, to remove SAM2 wave
            noise on reflective surfaces (default 1.5)
    """
    polygons = _potrace_curves_to_polygons(path, scale, img_shape=img_shape)

    if clearance_mm > 0.001:
        inner_polygons = _offset_polygons(polygons, clearance_mm)
    else:
        inner_polygons = polygons

    # Smooth + DP-simplify the inner trace for visual output. Reflective
    # surfaces (polished metal blades) make SAM2 produce wavy noise on the
    # raw potrace polygon — DP alone preserves the wave (amplitude > epsilon),
    # so we Gaussian-smooth along the contour first. The cleanup pipeline's
    # per-mask smoothing is tuned conservatively to preserve concavities
    # (pliers handle gaps), so we smooth more aggressively here at the polygon
    # level. The TOLERANCE layer below is offset from this same smoothed
    # polygon so the dashed tolerance perimeter stays parallel to the inner.
    display_inner = [_smooth_polygon_coords(p, sigma_mm=display_smooth_sigma_mm)
                     for p in inner_polygons]
    display_inner = [_simplify_polygon(p, epsilon=simplify_epsilon)
                     for p in display_inner]
    display_inner = [_round_sharp_corners(p) for p in display_inner]

    doc = ezdxf.new("R2010")
    # Set drawing units to millimeters so Fusion 360 imports at correct scale
    doc.header['$INSUNITS'] = 4  # 4 = millimeters
    doc.header['$MEASUREMENT'] = 1  # 1 = metric
    msp = doc.modelspace()

    # Inner outline on default layer
    for poly in display_inner:
        if len(poly) < 3:
            continue
        closed_poly = list(poly) + [poly[0]]
        msp.add_lwpolyline(closed_poly, close=True)

    # Outer tolerance outline on separate layer, offset from the smoothed
    # `display_inner` so the cut polygon stays parallel to the printed inner
    # reference. Always generated (even at offset 0) so prepare_bin → Fusion
    # get a DP-simplified polygon to cut against — falling back to the raw
    # potrace inner would give hundreds of points and freeze Fusion.
    # _offset_polygons no-ops at offset 0; positive expands the pocket,
    # negative shrinks it.
    doc.layers.add("TOLERANCE", color=3)  # green for visual distinction
    outer_polygons = _offset_polygons(display_inner, tolerance_mm)
    # "auto" sentinel (or empty string from a blank web form field)
    # triggers the taper-based heuristic; explicit numeric values
    # still take precedence.
    if isinstance(axial_tolerance_mm, str):
        s = axial_tolerance_mm.strip().lower()
        if s == "" or s == "auto":
            axial_tolerance_mm = _auto_axial_tolerance_mm(display_inner)
        else:
            try:
                axial_tolerance_mm = float(s)
            except ValueError:
                axial_tolerance_mm = 0.0
    elif axial_tolerance_mm is None:
        axial_tolerance_mm = _auto_axial_tolerance_mm(display_inner)
    if abs(axial_tolerance_mm) > 0.001:
        outer_polygons = _axial_stretch_polygons(outer_polygons,
                                                 axial_tolerance_mm)
    outer_polygons = [_simplify_polygon(p, epsilon=simplify_epsilon)
                      for p in outer_polygons]
    outer_polygons = [_round_sharp_corners(p) for p in outer_polygons]
    for poly in outer_polygons:
        if len(poly) < 3:
            continue
        closed_poly = list(poly) + [poly[0]]
        msp.add_lwpolyline(closed_poly, close=True,
                           dxfattribs={"layer": "TOLERANCE"})

    # Finger access slot on separate layer
    if slot_polygon is not None:
        doc.layers.add("SLOT", color=5)  # blue
        closed_slot = list(slot_polygon) + [slot_polygon[0]]
        msp.add_lwpolyline(closed_slot, close=True,
                           dxfattribs={"layer": "SLOT"})

    doc.saveas(output_path)
