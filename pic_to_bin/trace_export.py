"""
Export potrace paths to SVG and DXF formats.
Handles scaling from pixels to mm, clearance offset, and tolerance outline.
"""

import math
from typing import Optional

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


def _round_sharp_corners(polygon: list[tuple[float, float]],
                          radius: float = 1.5,
                          min_angle_deg: float = 90,
                          n_arc_points: int = 8) -> list[tuple[float, float]]:
    """Replace sharp corners with fillet arcs.

    Any vertex where the interior angle is less than min_angle_deg gets
    replaced by an arc of the given radius, smoothing the sharp point.

    Args:
        polygon: List of (x, y) tuples
        radius: Fillet radius in mm (default 1.5)
        min_angle_deg: Threshold — corners sharper than this are rounded
        n_arc_points: Number of points to generate along each fillet arc
    """
    if len(polygon) < 3:
        return polygon

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

    # --- Step 4–6: Place slot, trying positions from center of handle run outward ---
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

    # Try handle positions starting from center of run outward (most room)
    candidate_indices = sorted(
        range(run_start, run_end + 1),
        key=lambda i: abs(i - run_mid)
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


def _potrace_bezier_to_svg_paths(path, scale: float,
                                  img_shape: tuple = None) -> tuple[list[str], list[tuple[float, float]]]:
    """Convert potrace path to SVG path data strings using native Bezier curves.

    Uses potrace's native curve representation (compact, accurate) rather than
    sampling into polygon points.

    The Y axis is flipped from image convention (Y-down) to CAD convention
    (Y-up), matching _potrace_curves_to_polygons so the output orientation
    matches the input photo in Y-up viewers.

    Returns:
        (svg_paths, all_points) where all_points is a flat list of (x, y) for bbox.
    """
    svg_paths = []
    all_points = []

    h_mm = img_shape[0] * scale if img_shape else 0.0
    fy = (lambda y: h_mm - y) if img_shape else (lambda y: y)

    curves = _filter_curves(path, img_shape) if img_shape else path
    for curve in curves:
        start_x = curve.start_point.x * scale
        start_y = fy(curve.start_point.y * scale)
        all_points.append((start_x, start_y))

        d = f"M {start_x:.3f},{start_y:.3f}"

        for segment in curve:
            if segment.is_corner:
                cx = segment.c.x * scale
                cy = fy(segment.c.y * scale)
                ex = segment.end_point.x * scale
                ey = fy(segment.end_point.y * scale)
                d += f" L {cx:.3f},{cy:.3f} L {ex:.3f},{ey:.3f}"
                all_points.extend([(cx, cy), (ex, ey)])
            else:
                c1x = segment.c1.x * scale
                c1y = fy(segment.c1.y * scale)
                c2x = segment.c2.x * scale
                c2y = fy(segment.c2.y * scale)
                ex = segment.end_point.x * scale
                ey = fy(segment.end_point.y * scale)
                d += f" C {c1x:.3f},{c1y:.3f} {c2x:.3f},{c2y:.3f} {ex:.3f},{ey:.3f}"
                all_points.extend([(c1x, c1y), (c2x, c2y), (ex, ey)])

        d += " Z"
        svg_paths.append(d)

    return svg_paths, all_points


def potrace_to_svg(path, output_path: str, scale: float,
                   clearance_mm: float = 0.0, tolerance_mm: float = 0.0,
                   img_shape: tuple = None, simplify_epsilon: float = 0.3,
                   slot_polygon: list = None) -> dict:
    """Export potrace path to SVG, optionally with a tolerance outline.

    When tolerance_mm > 0, two outlines are written:
    - Inner: the accurate tool trace (native Bezier curves when clearance=0)
    - Outer: a smoothed tolerance perimeter, tolerance_mm beyond the inner outline

    Args:
        path: potrace.Path object
        output_path: Where to save the SVG
        scale: mm per pixel
        clearance_mm: Outward offset applied to the inner outline (default 0)
        tolerance_mm: Additional outward offset for the tolerance perimeter
        img_shape: (height, width) of source mask to filter boundary artifacts
        simplify_epsilon: Douglas-Peucker simplification threshold in mm for
                          the tolerance outline (default 0.3mm)

    Returns:
        Bounding box dict with width_mm, height_mm
    """
    # Build inner (tool cutout) paths
    if clearance_mm > 0.001:
        # Clearance offset requires polygon approximation
        polygons = _potrace_curves_to_polygons(path, scale, img_shape=img_shape)
        inner_polygons = _offset_polygons(polygons, clearance_mm)
        inner_svg_paths = _polygons_to_svg_paths(inner_polygons)
        bbox_polygons = list(inner_polygons)
    else:
        # No clearance — use native Bezier curves (fewer control points, accurate)
        inner_svg_paths, all_points = _potrace_bezier_to_svg_paths(path, scale, img_shape)
        bbox_polygons = [all_points]

    # Build outer tolerance paths
    outer_svg_paths = []
    if tolerance_mm > 0.001:
        # Need polygon representation for offset operation
        if clearance_mm > 0.001:
            base_polygons = inner_polygons
        else:
            base_polygons = _potrace_curves_to_polygons(path, scale, img_shape=img_shape)

        outer_polygons = _offset_polygons(base_polygons, tolerance_mm)
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
                   img_shape: tuple = None, simplify_epsilon: float = 0.3,
                   slot_polygon: list = None):
    """Export potrace path to DXF format, optionally with a tolerance outline.

    When tolerance_mm > 0, the inner (tool cutout) outline stays on the default
    layer "0", and the tolerance perimeter is placed on a "TOLERANCE" layer.

    Args:
        path: potrace.Path object
        output_path: Where to save the DXF
        scale: mm per pixel
        clearance_mm: Outward offset for printing tolerance
        tolerance_mm: Additional outward offset for the tolerance perimeter
        img_shape: (height, width) of source mask to filter boundary artifacts
        simplify_epsilon: Douglas-Peucker simplification threshold in mm
    """
    polygons = _potrace_curves_to_polygons(path, scale, img_shape=img_shape)

    if clearance_mm > 0.001:
        inner_polygons = _offset_polygons(polygons, clearance_mm)
    else:
        inner_polygons = polygons

    doc = ezdxf.new("R2010")
    # Set drawing units to millimeters so Fusion 360 imports at correct scale
    doc.header['$INSUNITS'] = 4  # 4 = millimeters
    doc.header['$MEASUREMENT'] = 1  # 1 = metric
    msp = doc.modelspace()

    # Inner outline on default layer
    for poly in inner_polygons:
        if len(poly) < 3:
            continue
        closed_poly = list(poly) + [poly[0]]
        msp.add_lwpolyline(closed_poly, close=True)

    # Outer tolerance outline on separate layer
    if tolerance_mm > 0.001:
        doc.layers.add("TOLERANCE", color=3)  # green for visual distinction
        outer_polygons = _offset_polygons(inner_polygons, tolerance_mm)
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
