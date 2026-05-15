"""
Agent 3: Layout Tools
Packs multiple tool DXF traces into a single gridfinity bin layout.

Pipeline:
1. Load DXF files (extract polylines by layer)
2. Try rotations (0/180) and mirrors for each tool
3. Polygon-based packing with actual shape collision detection
4. Snap to gridfinity grid, center horizontally
5. Write combined DXF + preview PNG
"""

import argparse
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import ezdxf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import numpy as np

from pic_to_bin.trace_export import _compute_bbox, _offset_polygons


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GridSizeError(ValueError):
    """Raised when tools cannot fit within the maximum grid size.

    Attributes:
        required_x: Minimum grid units needed in X (best combo found).
        required_y: Minimum grid units needed in Y (best combo found).
    """
    def __init__(self, message: str, required_x: int = 0, required_y: int = 0):
        super().__init__(message)
        self.required_x = required_x
        self.required_y = required_y


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ToolProfile:
    """All polygon data for one tool, extracted from a DXF file."""
    source_path: str
    name: str
    inner_polys: list  # list[list[tuple[float, float]]]
    tolerance_polys: list
    slot_polys: list
    bbox: dict = field(default_factory=dict)

    def all_polys(self):
        return self.inner_polys + self.tolerance_polys + self.slot_polys


@dataclass
class PlacedTool:
    """A tool with its final position and rotation in the layout."""
    tool: ToolProfile
    rotation_deg: int
    offset_x: float
    offset_y: float
    mirrored: bool = False


# ---------------------------------------------------------------------------
# DXF I/O
# ---------------------------------------------------------------------------

def load_tool_from_dxf(dxf_path: str) -> ToolProfile:
    """Read a DXF file and extract LWPOLYLINE entities grouped by layer.

    Normalizes coordinates so the tool's bounding box starts at (0, 0).
    """
    path = Path(dxf_path)
    if not path.exists():
        raise FileNotFoundError(f"DXF not found: {dxf_path}")

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    inner = []
    tolerance = []
    slot = []

    for entity in msp.query("LWPOLYLINE"):
        layer = entity.dxf.layer.upper()
        points = [(p[0], p[1]) for p in entity.get_points(format="xy")]
        if len(points) < 3:
            continue

        if layer == "TOLERANCE":
            tolerance.append(points)
        elif layer == "SLOT":
            slot.append(points)
        else:
            inner.append(points)

    if not inner:
        raise ValueError(f"No polylines found on layer '0' in {dxf_path}")

    # Normalize to origin using packing bbox (inner + tolerance), NOT full bbox.
    # Slots are floor cutouts that can extend beyond the tool body — they get
    # negative coordinates when they stick out, which is correct.
    packing_polys = inner + tolerance if tolerance else inner
    packing_bbox = _compute_bbox(packing_polys)
    ox, oy = packing_bbox["min_x"], packing_bbox["min_y"]
    inner = [[(x - ox, y - oy) for x, y in p] for p in inner]
    tolerance = [[(x - ox, y - oy) for x, y in p] for p in tolerance]
    slot = [[(x - ox, y - oy) for x, y in p] for p in slot]

    bbox = _compute_bbox(inner + (tolerance if tolerance else []))

    return ToolProfile(
        source_path=str(path),
        name=path.stem.replace("_trace", ""),
        inner_polys=inner,
        tolerance_polys=tolerance,
        slot_polys=slot,
        bbox=bbox,
    )


def write_combined_dxf(placed_tools: list[PlacedTool], output_path: str,
                       bin_width_mm: float, bin_height_mm: float) -> None:
    """Write a combined DXF with all tools positioned and a bin boundary."""
    doc = ezdxf.new("R2010")
    doc.header['$INSUNITS'] = 4
    doc.header['$MEASUREMENT'] = 1
    msp = doc.modelspace()

    # Ensure layers exist
    if "TOLERANCE" not in [l.dxf.name for l in doc.layers]:
        doc.layers.add("TOLERANCE", color=3)
    if "SLOT" not in [l.dxf.name for l in doc.layers]:
        doc.layers.add("SLOT", color=5)
    doc.layers.add("BIN_BOUNDARY", color=1)

    for placed in placed_tools:
        rotated = _apply_transform(placed.tool, placed.rotation_deg,
                                   placed.mirrored)
        ox, oy = placed.offset_x, placed.offset_y

        for poly in rotated.inner_polys:
            pts = [(x + ox, y + oy) for x, y in poly]
            msp.add_lwpolyline(pts, close=True)

        for poly in rotated.tolerance_polys:
            pts = [(x + ox, y + oy) for x, y in poly]
            msp.add_lwpolyline(pts, close=True,
                               dxfattribs={"layer": "TOLERANCE"})

        for poly in rotated.slot_polys:
            pts = [(x + ox, y + oy) for x, y in poly]
            msp.add_lwpolyline(pts, close=True,
                               dxfattribs={"layer": "SLOT"})

    # Bin boundary rectangle
    msp.add_lwpolyline(
        [(0, 0), (bin_width_mm, 0), (bin_width_mm, bin_height_mm),
         (0, bin_height_mm)],
        close=True,
        dxfattribs={"layer": "BIN_BOUNDARY"},
    )

    doc.saveas(output_path)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# Exact trig values for 90° increments to avoid float drift
_TRIG = {
    0:   (1, 0),
    90:  (0, 1),
    180: (-1, 0),
    270: (0, -1),
}


def rotate_polygon(polygon: list[tuple[float, float]],
                   angle_deg: int) -> list[tuple[float, float]]:
    """Rotate a polygon around the origin by angle_deg degrees."""
    angle_deg = angle_deg % 360
    if angle_deg == 0:
        return list(polygon)

    if angle_deg in _TRIG:
        cos_a, sin_a = _TRIG[angle_deg]
    else:
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a)
            for x, y in polygon]


def rotate_tool(tool: ToolProfile, angle_deg: int) -> ToolProfile:
    """Return a new ToolProfile with all polygons rotated and re-normalized."""
    angle_deg = angle_deg % 360
    if angle_deg == 0:
        return tool

    # Rotate around the center of the bbox
    cx = (tool.bbox["min_x"] + tool.bbox["max_x"]) / 2
    cy = (tool.bbox["min_y"] + tool.bbox["max_y"]) / 2

    def rotate_poly(poly):
        centered = [(x - cx, y - cy) for x, y in poly]
        rotated = rotate_polygon(centered, angle_deg)
        return [(x + cx, y + cy) for x, y in rotated]

    inner = [rotate_poly(p) for p in tool.inner_polys]
    tolerance = [rotate_poly(p) for p in tool.tolerance_polys]
    slot = [rotate_poly(p) for p in tool.slot_polys]

    # Normalize using packing bbox (inner + tolerance), not full bbox.
    packing_polys = inner + tolerance if tolerance else inner
    packing_bbox = _compute_bbox(packing_polys)
    ox, oy = packing_bbox["min_x"], packing_bbox["min_y"]
    inner = [[(x - ox, y - oy) for x, y in p] for p in inner]
    tolerance = [[(x - ox, y - oy) for x, y in p] for p in tolerance]
    slot = [[(x - ox, y - oy) for x, y in p] for p in slot]

    bbox = _compute_bbox(inner + (tolerance if tolerance else []))

    return ToolProfile(
        source_path=tool.source_path,
        name=tool.name,
        inner_polys=inner,
        tolerance_polys=tolerance,
        slot_polys=slot,
        bbox=bbox,
    )


def mirror_polygon(polygon: list[tuple[float, float]],
                   axis: str = "x") -> list[tuple[float, float]]:
    """Mirror a polygon across the given axis.

    axis='x' flips left-to-right (negate x).
    axis='y' flips top-to-bottom (negate y).
    """
    if axis == "x":
        return [(-x, y) for x, y in polygon]
    elif axis == "y":
        return [(x, -y) for x, y in polygon]
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def mirror_tool(tool: ToolProfile, axis: str = "x") -> ToolProfile:
    """Return a new ToolProfile mirrored across the given axis, re-normalized."""
    inner = [mirror_polygon(p, axis) for p in tool.inner_polys]
    tolerance = [mirror_polygon(p, axis) for p in tool.tolerance_polys]
    slot = [mirror_polygon(p, axis) for p in tool.slot_polys]

    # Re-normalize using packing bbox
    packing_polys = inner + tolerance if tolerance else inner
    packing_bbox = _compute_bbox(packing_polys)
    ox, oy = packing_bbox["min_x"], packing_bbox["min_y"]
    inner = [[(x - ox, y - oy) for x, y in p] for p in inner]
    tolerance = [[(x - ox, y - oy) for x, y in p] for p in tolerance]
    slot = [[(x - ox, y - oy) for x, y in p] for p in slot]

    bbox = _compute_bbox(inner + (tolerance if tolerance else []))

    return ToolProfile(
        source_path=tool.source_path,
        name=tool.name,
        inner_polys=inner,
        tolerance_polys=tolerance,
        slot_polys=slot,
        bbox=bbox,
    )


def _apply_transform(tool: ToolProfile, angle_deg: int,
                      mirrored: bool) -> ToolProfile:
    """Apply rotation then optional mirror to a tool."""
    result = rotate_tool(tool, angle_deg)
    if mirrored:
        result = mirror_tool(result, "x")
    return result


def snap_to_grid(dimension_mm: float, unit_mm: float = 42.0) -> int:
    """Round dimension up to the nearest whole multiple of unit_mm."""
    if dimension_mm <= 0:
        return 1
    return max(1, math.ceil(dimension_mm / unit_mm))


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def _shelf_pack(bboxes: list[dict]) -> list[tuple[float, float]]:
    """Bottom-left shelf packing. Each bbox has 'width_mm' and 'height_mm'.

    Items sorted by height (tallest first) for better shelf utilization.
    Returns list of (x, y) positions in the same order as input bboxes.
    """
    if not bboxes:
        return []

    order = sorted(range(len(bboxes)), key=lambda i: bboxes[i]["height_mm"],
                   reverse=True)

    # Each shelf: [y_offset, height, current_x]
    shelves = []
    positions = [None] * len(bboxes)

    for idx in order:
        w = bboxes[idx]["width_mm"]
        h = bboxes[idx]["height_mm"]

        placed = False
        for shelf in shelves:
            if h <= shelf[1]:
                positions[idx] = (shelf[2], shelf[0])
                shelf[2] += w
                placed = True
                break

        if not placed:
            y_off = sum(s[1] for s in shelves)
            shelves.append([y_off, h, w])
            positions[idx] = (0.0, y_off)

    return positions


# ---------------------------------------------------------------------------
# Polygon-based packing
# ---------------------------------------------------------------------------

def _make_footprint(polygons, resolution, half_gap):
    """Rasterize polygons with outward padding for collision detection.

    Returns:
        (footprint, origin_x, origin_y)
        footprint: 2D bool array
        origin_x, origin_y: tool-local coords of footprint[0, 0]
    """
    if half_gap > 0.01:
        try:
            padded = _offset_polygons(polygons, half_gap)
        except Exception:
            padded = [list(p) for p in polygons]
    else:
        padded = [list(p) for p in polygons]

    valid = [p for p in padded if len(p) >= 3]
    if not valid:
        return np.ones((1, 1), dtype=bool), 0.0, 0.0

    all_pts = [(x, y) for poly in valid for x, y in poly]
    min_x = min(x for x, y in all_pts)
    min_y = min(y for x, y in all_pts)
    max_x = max(x for x, y in all_pts)
    max_y = max(y for x, y in all_pts)

    cols = int(math.ceil((max_x - min_x) / resolution)) + 1
    rows = int(math.ceil((max_y - min_y) / resolution)) + 1

    # Limit grid size
    cols = min(cols, 500)
    rows = min(rows, 500)

    xs = np.arange(cols) * resolution + min_x
    ys = np.arange(rows) * resolution + min_y
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])

    mask = np.zeros(len(points), dtype=bool)
    for poly in valid:
        closed = list(poly) + [poly[0]]
        path = MplPath(closed)
        mask |= path.contains_points(points)

    return mask.reshape(rows, cols), min_x, min_y


def _polygon_pack(tools, gap_mm, max_width_mm, max_height_mm,
                  resolution=1.0, gridfinity_unit=42.0):
    """Bottom-left fill packing using rasterized polygon collision.

    Places tools one at a time (largest first). For each tool, scans positions
    bottom-to-top, left-to-right, and picks the position that minimises the
    tuple score ``(grid_area, -min_wall_slack)``. The secondary term is the
    smaller of (bin_width − x_extent) and (bin_height − y_extent) — i.e. how
    far the tool cluster sits from the nearest bin wall after the bin is
    rounded up to whole gridfinity units. With ties on grid area, this prefers
    layouts that leave generous wall clearance instead of squeezing one extra
    tool against the bin edge (which then forces thin-wall snapping at the
    stacking lip).

    Args:
        tools: list of ToolProfile (already transformed)
        gap_mm: minimum gap between tools in mm
        max_width_mm, max_height_mm: bin limits in mm
        resolution: grid cell size in mm
        gridfinity_unit: grid pitch in mm (used to evaluate wall clearance)

    Returns:
        list of (x, y) positions in mm, or None if placement fails
    """
    if not tools:
        return []

    half_gap = gap_mm / 2.0
    # Margin for footprints that extend past tool origin due to padding
    margin = int(math.ceil((gap_mm + 2) / resolution))

    gw = int(math.ceil(max_width_mm / resolution)) + 2 * margin
    gh = int(math.ceil(max_height_mm / resolution)) + 2 * margin

    occupied = np.zeros((gh, gw), dtype=bool)

    # Pre-compute footprints
    footprints = []
    for tool in tools:
        polys = tool.tolerance_polys if tool.tolerance_polys else tool.inner_polys
        fp, ox, oy = _make_footprint(polys, resolution, half_gap)
        footprints.append((fp, ox, oy))

    positions = []
    cur_max_x = 0.0
    cur_max_y = 0.0

    for i, tool in enumerate(tools):
        fp, fp_ox, fp_oy = footprints[i]
        fh, fw = fp.shape
        tw = tool.bbox["width_mm"]
        th = tool.bbox["height_mm"]

        best_pos = None
        best_score = (float("inf"), 0.0)

        # Search bounds
        if i == 0:
            search_max_x = 0.01  # first tool always at origin
            search_max_y = 0.01
        else:
            search_max_x = min(cur_max_x + tw + gap_mm,
                               max_width_mm - tw)
            search_max_y = min(cur_max_y + th + gap_mm,
                               max_height_mm - th)

        x_steps = int(search_max_x / resolution) + 1
        y_steps = int(search_max_y / resolution) + 1

        for wy_i in range(y_steps):
            wy = wy_i * resolution

            # Early exit: minimum possible score at this Y level
            y_extent = max(cur_max_y, wy + th)
            y_units = snap_to_grid(y_extent, gridfinity_unit)
            min_x_extent = max(cur_max_x, tw)  # best case: tool at x=0
            min_x_units = snap_to_grid(min_x_extent, gridfinity_unit)
            min_slack_x = min_x_units * gridfinity_unit - min_x_extent
            slack_y = y_units * gridfinity_unit - y_extent
            min_score = (min_x_units * y_units,
                         -min(min_slack_x, slack_y))
            if min_score >= best_score:
                break

            for wx_i in range(x_steps):
                wx = wx_i * resolution

                # Within a fixed Y, primary score (x_units * y_units) is
                # monotonically non-decreasing in X, and the secondary
                # (-min_slack) is non-decreasing within each x_units band.
                # So once score >= best, every larger X is also >= best.
                x_extent = max(cur_max_x, wx + tw)
                x_units = snap_to_grid(x_extent, gridfinity_unit)
                slack_x = x_units * gridfinity_unit - x_extent
                score = (x_units * y_units, -min(slack_x, slack_y))
                if score >= best_score:
                    break

                # Map to grid coordinates
                gx = int(round((wx + fp_ox) / resolution)) + margin
                gy = int(round((wy + fp_oy) / resolution)) + margin

                if gx < 0 or gy < 0 or gx + fw > gw or gy + fh > gh:
                    continue

                # Collision check
                if np.any(occupied[gy:gy + fh, gx:gx + fw] & fp):
                    continue

                # Valid placement — first valid X at this Y is optimal
                best_score = score
                best_pos = (wx, wy, gx, gy)
                break

        if best_pos is None:
            return None

        wx, wy, gx, gy = best_pos
        occupied[gy:gy + fh, gx:gx + fw] |= fp
        positions.append((wx, wy))
        cur_max_x = max(cur_max_x, wx + tw)
        cur_max_y = max(cur_max_y, wy + th)

    return positions


def pack_tools_greedy(tools: list[ToolProfile], gap_mm: float = 3.0,
                      gridfinity_unit: float = 42.0,
                      max_units: int = 7,
                      bin_margin_mm: float = 12.0,
                      min_units_x: int = 1,
                      min_units_y: int = 1,
                      ) -> tuple[list[PlacedTool], int, int]:
    """Pack tools into the smallest gridfinity bin.

    Two-phase approach:
    1. Score all transform combos with fast shelf packing (bounding boxes)
    2. Re-pack top candidates with polygon collision detection (actual shapes)

    Tools are centered horizontally within the final bin.

    bin_margin_mm enforces a minimum clearance between the tool extent and
    the bin's outer edge. snap-to-grid only rounds *up*, so a tool that
    barely fits a given unit count (e.g. 104mm in a 3×42=126mm bin → 11mm
    of slack on each side) would otherwise sit very close to the wall after
    accounting for the wall thickness and stacking lip. Padding the total
    extent by 2*bin_margin_mm before snap-to-grid pushes the bin one unit
    bigger when a tool is within bin_margin_mm of the boundary.

    min_units_x / min_units_y set a per-axis floor. The packer still picks
    the smallest fit, but the final grid is clamped up so the bin never comes
    out smaller than requested on either axis. Tool centering runs after
    clamping, so the extra slack distributes evenly.

    Returns:
        (placed_tools, grid_units_x, grid_units_y)
    """
    if not tools:
        raise ValueError("No tools to pack")

    half_gap = gap_mm / 2.0
    n = len(tools)
    max_w = max_units * gridfinity_unit
    max_h = max_units * gridfinity_unit
    margin_pad = 2.0 * max(0.0, bin_margin_mm)

    # Pre-compute all transforms: 2 rotations × 2 mirror states = 4 per tool
    transforms = {}
    for i, tool in enumerate(tools):
        transforms[i] = {}
        for angle in (0, 180):
            rotated = rotate_tool(tool, angle)
            transforms[i][(angle, False)] = rotated
            transforms[i][(angle, True)] = mirror_tool(rotated, "x")

    all_keys = [(a, m) for a in (0, 180) for m in (False, True)]

    # Sort tools by largest bbox area first
    tool_order = sorted(range(n), key=lambda i: max(
        transforms[i][k].bbox["width_mm"] * transforms[i][k].bbox["height_mm"]
        for k in all_keys
    ), reverse=True)

    # Generate transform combos
    if n <= 3:
        combos = list(itertools.product(all_keys, repeat=n))
    elif n <= 6:
        combos = list(itertools.product(
            [(a, False) for a in (0, 180)], repeat=n))
    else:
        best_keys = []
        for i in range(n):
            best_k = min(all_keys,
                         key=lambda k: (transforms[i][k].bbox["width_mm"] *
                                        transforms[i][k].bbox["height_mm"]))
            best_keys.append(best_k)
        combos = [tuple(best_keys)]

    # Phase 1: Score all combos with shelf packing (instant)
    candidates = []
    smallest_over_area = float("inf")
    smallest_over_ux = 0
    smallest_over_uy = 0
    for combo in combos:
        padded = []
        for idx in tool_order:
            key = combo[idx]
            r = transforms[idx][key]
            padded.append({
                "width_mm": r.bbox["width_mm"] + gap_mm,
                "height_mm": r.bbox["height_mm"] + gap_mm,
            })

        positions = _shelf_pack(padded)
        total_w = max(positions[j][0] + padded[j]["width_mm"]
                      for j in range(n))
        total_h = max(positions[j][1] + padded[j]["height_mm"]
                      for j in range(n))
        ux = snap_to_grid(total_w + margin_pad, gridfinity_unit)
        uy = snap_to_grid(total_h + margin_pad, gridfinity_unit)

        if ux <= max_units and uy <= max_units:
            candidates.append({
                "area": ux * uy,
                "combo": combo,
                "shelf_ux": ux,
                "shelf_uy": uy,
            })
        elif ux * uy < smallest_over_area:
            smallest_over_area = ux * uy
            smallest_over_ux = ux
            smallest_over_uy = uy

    if not candidates:
        raise GridSizeError(
            f"Tools cannot fit within {max_units}x{max_units} gridfinity grid",
            required_x=smallest_over_ux,
            required_y=smallest_over_uy,
        )

    candidates.sort(key=lambda c: c["area"])

    # Phase 2: Polygon pack top candidates
    top_n = min(30, len(candidates))
    best_result = None
    best_score = (float("inf"), 0.0)

    for cand in candidates[:top_n]:
        combo = cand["combo"]
        ordered_tools = [transforms[idx][combo[idx]] for idx in tool_order]

        positions = _polygon_pack(ordered_tools, gap_mm, max_w, max_h,
                                  gridfinity_unit=gridfinity_unit)
        if positions is None:
            continue

        total_w = max(positions[j][0] + ordered_tools[j].bbox["width_mm"]
                      for j in range(n))
        total_h = max(positions[j][1] + ordered_tools[j].bbox["height_mm"]
                      for j in range(n))
        ux = snap_to_grid(total_w + margin_pad, gridfinity_unit)
        uy = snap_to_grid(total_h + margin_pad, gridfinity_unit)

        if ux > max_units or uy > max_units:
            continue

        # Tiebreaker on equal grid area: prefer layouts with more wall slack,
        # matching the per-placement tiebreaker inside _polygon_pack.
        wall_slack = min(ux * gridfinity_unit - total_w,
                         uy * gridfinity_unit - total_h)
        score = (ux * uy, -wall_slack)
        if score < best_score:
            best_score = score
            best_result = (combo, positions, ux, uy)

    # Fallback: use shelf packing result from the best candidate
    if best_result is None:
        cand = candidates[0]
        combo = cand["combo"]
        padded = []
        for idx in tool_order:
            key = combo[idx]
            r = transforms[idx][key]
            padded.append({
                "width_mm": r.bbox["width_mm"] + gap_mm,
                "height_mm": r.bbox["height_mm"] + gap_mm,
            })
        shelf_positions = _shelf_pack(padded)
        positions = [(px + half_gap, py + half_gap)
                     for px, py in shelf_positions]
        best_result = (combo, positions, cand["shelf_ux"], cand["shelf_uy"])

    combo, positions, units_x, units_y = best_result

    # Apply minimum bin size before centering so the slack from any clamped
    # axis distributes evenly around the tools.
    units_x = max(units_x, max(1, int(min_units_x)))
    units_y = max(units_y, max(1, int(min_units_y)))

    # Center tools within the bin (both axes)
    bin_w = units_x * gridfinity_unit
    bin_h = units_y * gridfinity_unit
    tool_widths = [transforms[tool_order[j]][combo[tool_order[j]]].bbox["width_mm"]
                   for j in range(n)]
    tool_heights = [transforms[tool_order[j]][combo[tool_order[j]]].bbox["height_mm"]
                    for j in range(n)]
    min_x = min(positions[j][0] for j in range(n))
    max_x = max(positions[j][0] + tool_widths[j] for j in range(n))
    min_y = min(positions[j][1] for j in range(n))
    max_y = max(positions[j][1] + tool_heights[j] for j in range(n))
    content_w = max_x - min_x
    content_h = max_y - min_y
    x_shift = (bin_w - content_w) / 2 - min_x
    y_shift = (bin_h - content_h) / 2 - min_y

    placed = []
    placed_widths = []
    placed_heights = []
    for j in range(n):
        idx = tool_order[j]
        angle, mirrored = combo[idx]
        px, py = positions[j]
        placed.append(PlacedTool(
            tool=tools[idx],
            rotation_deg=angle,
            mirrored=mirrored,
            offset_x=px + x_shift,
            offset_y=py + y_shift,
        ))
        placed_widths.append(tool_widths[j])
        placed_heights.append(tool_heights[j])

    # Spread tools within their row / column to distribute slack. For each
    # group of tools sharing a y-overlap ("row"), distribute the x-slack
    # evenly so the row reads as N+1 equal gaps instead of all-pack-left
    # plus one big margin on the right. Same idea for x-overlap groups
    # along y. Collapses to centering for single-tool groups (replacing
    # the previous per-tool y_alone / x_alone re-center pass). Each
    # inter-tool gap is held at >= gap_mm so we never squeeze the
    # polygon-packer's safe spacing — only outer margins absorb the
    # remainder when slack is tight.
    _spread_within_rows(placed, placed_widths, placed_heights,
                        bin_w, bin_h, gap_mm)

    return placed, units_x, units_y


def _spread_within_rows(placed, widths, heights, bin_w, bin_h, gap_mm,
                        eps_mm: float = 0.5) -> None:
    """Distribute slack along each axis within rows / columns. In-place."""
    # X-spread within each y-overlap group.
    y_intervals = [(placed[j].offset_y, placed[j].offset_y + heights[j])
                   for j in range(len(placed))]
    for group in _overlap_groups(y_intervals, eps_mm):
        _space_evenly(
            group,
            get_pos=lambda j: placed[j].offset_x,
            set_pos=lambda j, v: setattr(placed[j], "offset_x", v),
            get_size=lambda j: widths[j],
            total=bin_w,
            min_inter_gap=gap_mm,
        )
    # Y-spread within each x-overlap group, using the just-updated x's.
    x_intervals = [(placed[j].offset_x, placed[j].offset_x + widths[j])
                   for j in range(len(placed))]
    for group in _overlap_groups(x_intervals, eps_mm):
        _space_evenly(
            group,
            get_pos=lambda j: placed[j].offset_y,
            set_pos=lambda j, v: setattr(placed[j], "offset_y", v),
            get_size=lambda j: heights[j],
            total=bin_h,
            min_inter_gap=gap_mm,
        )


def _overlap_groups(intervals: list[tuple[float, float]],
                    eps: float = 0.5) -> list[list[int]]:
    """Group indices whose intervals overlap (transitively) by more than eps."""
    n = len(intervals)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            a0, a1 = intervals[i]
            b0, b1 = intervals[j]
            if min(a1, b1) - max(a0, b0) > eps:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _space_evenly(indices, get_pos, set_pos, get_size, total,
                  min_inter_gap: float) -> None:
    """Place ``indices`` along an axis with equal gaps where possible.

    Sorted by current position to preserve packer ordering. For a single
    tool, centers it. For multiple tools, distributes total - sum(sizes)
    as (k+1) equal gaps when room allows; otherwise pins inter-tool gaps
    at ``min_inter_gap`` and splits the remainder as outer margins.
    """
    indices = sorted(indices, key=get_pos)
    sizes = [get_size(j) for j in indices]
    k = len(indices)
    slack = total - sum(sizes)
    if slack < 0:
        return  # bin too small for an even pass; leave packer's output

    if k == 1:
        set_pos(indices[0], slack / 2.0)
        return

    even = slack / (k + 1)
    if even >= min_inter_gap:
        gap_outer = gap_inter = even
    else:
        reserved = min_inter_gap * (k - 1)
        outer = (slack - reserved) / 2.0
        if outer < 0:
            return  # can't satisfy min_inter_gap; bail to packer's output
        gap_outer, gap_inter = outer, min_inter_gap

    pos = gap_outer
    for j, size in zip(indices, sizes):
        set_pos(j, pos)
        pos += size + gap_inter


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

TOOL_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#F44336", "#00BCD4", "#795548", "#607D8B",
]


def generate_preview(placed_tools: list[PlacedTool],
                     grid_units_x: int, grid_units_y: int,
                     output_path: str,
                     gridfinity_unit: float = 42.0) -> None:
    """Generate a PNG preview of the layout."""
    bin_w = grid_units_x * gridfinity_unit
    bin_h = grid_units_y * gridfinity_unit

    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * bin_h / max(bin_w, 1)))

    # Grid lines
    for i in range(grid_units_x + 1):
        ax.axvline(i * gridfinity_unit, color="#ddd", linewidth=0.5)
    for j in range(grid_units_y + 1):
        ax.axhline(j * gridfinity_unit, color="#ddd", linewidth=0.5)

    # Bin boundary
    ax.add_patch(mpatches.Rectangle((0, 0), bin_w, bin_h,
                                     fill=False, edgecolor="black",
                                     linewidth=2))

    for k, placed in enumerate(placed_tools):
        color = TOOL_COLORS[k % len(TOOL_COLORS)]
        rotated = _apply_transform(placed.tool, placed.rotation_deg,
                                   placed.mirrored)
        ox, oy = placed.offset_x, placed.offset_y

        # Inner outline — filled
        for poly in rotated.inner_polys:
            pts = np.array([(x + ox, y + oy) for x, y in poly])
            ax.fill(pts[:, 0], pts[:, 1], alpha=0.3, color=color)
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color=color, linewidth=1.5)

        # Tolerance outline — dashed
        for poly in rotated.tolerance_polys:
            pts = np.array([(x + ox, y + oy) for x, y in poly])
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color=color, linewidth=1, linestyle="--", alpha=0.6)

        # Slot — dotted
        for poly in rotated.slot_polys:
            pts = np.array([(x + ox, y + oy) for x, y in poly])
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color=color, linewidth=1, linestyle=":", alpha=0.6)

        # Label
        r_bbox = rotated.bbox
        cx = ox + r_bbox["width_mm"] / 2
        cy = oy + r_bbox["height_mm"] / 2
        mirror_str = " M" if placed.mirrored else ""
        ax.text(cx, cy,
                f"{placed.tool.name}\n{placed.rotation_deg}°{mirror_str}",
                ha="center", va="center", fontsize=7, color=color,
                fontweight="bold")

    ax.set_xlim(-5, bin_w + 5)
    ax.set_ylim(-5, bin_h + 5)
    ax.set_aspect("equal")
    ax.set_title(f"Layout: {grid_units_x}x{grid_units_y} gridfinity units "
                 f"({bin_w:.0f}x{bin_h:.0f}mm)", fontsize=14)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_fit_test_drawing(placed_tools: list[PlacedTool],
                              grid_units_x: int, grid_units_y: int,
                              output_path: str,
                              gridfinity_unit: float = 42.0) -> None:
    """Render the layout at 1:1 physical scale for printing.

    The output page is sized to the bin's exact mm dimensions, with no
    title, no axes, and no padding — so when the user prints at "Actual
    size" / "100% scale" they get a paper template they can lay physical
    tools on top of to verify fit before 3D printing.

    Saves to PDF or SVG (chosen by the file extension). Both formats encode
    physical units, so the resulting print is dimensionally accurate
    regardless of printer DPI.
    """
    bin_w = grid_units_x * gridfinity_unit  # mm
    bin_h = grid_units_y * gridfinity_unit

    fig = plt.figure(figsize=(bin_w / 25.4, bin_h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, bin_w)
    ax.set_ylim(0, bin_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # Light grid lines on each gridfinity unit boundary.
    for i in range(grid_units_x + 1):
        ax.plot([i * gridfinity_unit, i * gridfinity_unit], [0, bin_h],
                color="#cccccc", linewidth=0.25)
    for j in range(grid_units_y + 1):
        ax.plot([0, bin_w], [j * gridfinity_unit, j * gridfinity_unit],
                color="#cccccc", linewidth=0.25)

    # Bin outline.
    ax.add_patch(mpatches.Rectangle((0, 0), bin_w, bin_h,
                                     fill=False, edgecolor="black",
                                     linewidth=0.5))

    for placed in placed_tools:
        rotated = _apply_transform(placed.tool, placed.rotation_deg,
                                   placed.mirrored)
        ox, oy = placed.offset_x, placed.offset_y
        # Inner trace — thin solid line (the actual tool outline).
        for poly in rotated.inner_polys:
            pts = np.array([(x + ox, y + oy) for x, y in poly])
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color="black", linewidth=0.4)
        # Tolerance outline — dashed (shows the actual pocket shape).
        for poly in rotated.tolerance_polys:
            pts = np.array([(x + ox, y + oy) for x, y in poly])
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color="black", linewidth=0.25, linestyle=(0, (3, 2)))

    plt.savefig(output_path)
    plt.close()

    # Matplotlib writes SVG dimensions in "pt" (points). That's an absolute
    # unit, but some image viewers misinterpret it. Rewrite the root width/
    # height in "mm" so the file is unambiguous about its physical size.
    if str(output_path).lower().endswith(".svg"):
        _svg_set_mm_dimensions(output_path, bin_w, bin_h)


def _svg_set_mm_dimensions(svg_path: str, width_mm: float, height_mm: float) -> None:
    """Rewrite the root <svg> width/height attributes to mm units."""
    import re
    p = Path(svg_path)
    text = p.read_text(encoding="utf-8")
    # Only touch the first <svg ...> tag.
    def fix(match: "re.Match[str]") -> str:
        tag = match.group(0)
        tag = re.sub(r'\swidth="[^"]+"', f' width="{width_mm:.4f}mm"', tag, count=1)
        tag = re.sub(r'\sheight="[^"]+"', f' height="{height_mm:.4f}mm"', tag, count=1)
        return tag
    text = re.sub(r"<svg\b[^>]*>", fix, text, count=1)
    p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def layout_tools(dxf_paths: list[str], gap_mm: float = 3.0,
                 gridfinity_unit: float = 42.0, max_units: int = 7,
                 bin_margin_mm: float = 12.0,
                 min_units_x: int = 1,
                 min_units_y: int = 1,
                 output_dir: str = None) -> dict:
    """Main pipeline: load DXFs, pack, write combined DXF + preview."""
    if output_dir is None:
        output_dir = "generated"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tools
    print(f"Loading {len(dxf_paths)} tool(s)...")
    tools = []
    for p in dxf_paths:
        tool = load_tool_from_dxf(p)
        print(f"  {tool.name}: {tool.bbox['width_mm']:.1f} x "
              f"{tool.bbox['height_mm']:.1f} mm")
        tools.append(tool)

    # Pack
    print(f"Packing with {gap_mm}mm gap...")
    placed, units_x, units_y = pack_tools_greedy(
        tools, gap_mm=gap_mm, gridfinity_unit=gridfinity_unit,
        max_units=max_units, bin_margin_mm=bin_margin_mm,
        min_units_x=min_units_x, min_units_y=min_units_y)

    bin_w = units_x * gridfinity_unit
    bin_h = units_y * gridfinity_unit
    print(f"  Grid: {units_x}x{units_y} units ({bin_w:.0f}x{bin_h:.0f}mm)")

    for p in placed:
        mirror_str = " mirrored" if p.mirrored else ""
        print(f"  {p.tool.name}: {p.rotation_deg}°{mirror_str} at "
              f"({p.offset_x:.1f}, {p.offset_y:.1f})")

    # Write combined DXF
    dxf_path = output_dir / "combined_layout.dxf"
    print(f"Writing combined DXF: {dxf_path}")
    write_combined_dxf(placed, str(dxf_path), bin_w, bin_h)

    # Generate preview
    preview_path = output_dir / "layout_preview.png"
    print(f"Generating preview: {preview_path}")
    generate_preview(placed, units_x, units_y, str(preview_path),
                     gridfinity_unit=gridfinity_unit)

    # Generate 1:1 scale fit-test drawings for printing.
    fit_pdf = output_dir / "layout_actual_size.pdf"
    fit_svg = output_dir / "layout_actual_size.svg"
    print(f"Generating 1:1 fit-test PDF/SVG: {fit_pdf.name}, {fit_svg.name}")
    generate_fit_test_drawing(placed, units_x, units_y, str(fit_pdf),
                              gridfinity_unit=gridfinity_unit)
    generate_fit_test_drawing(placed, units_x, units_y, str(fit_svg),
                              gridfinity_unit=gridfinity_unit)

    print(f"\nDone! Layout: {units_x}x{units_y} gridfinity units")

    return {
        "combined_dxf_path": str(dxf_path),
        "preview_path": str(preview_path),
        "fit_test_pdf_path": str(fit_pdf),
        "fit_test_svg_path": str(fit_svg),
        "grid_units_x": units_x,
        "grid_units_y": units_y,
        "grid_mm_x": bin_w,
        "grid_mm_y": bin_h,
        "placements": [
            {"name": p.tool.name, "rotation": p.rotation_deg,
             "mirrored": p.mirrored, "x": p.offset_x, "y": p.offset_y}
            for p in placed
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pack multiple tool DXFs into a single gridfinity bin layout"
    )
    parser.add_argument("dxf_files", nargs="+",
                        help="DXF files from trace_tool.py output")
    parser.add_argument("--gap", type=float, default=3.0,
                        help="Minimum gap between tools in mm (default: 3.0)")
    parser.add_argument("--grid-unit", type=float, default=42.0,
                        help="Gridfinity unit size in mm (default: 42.0)")
    parser.add_argument("--max-units", type=int, default=7,
                        help="Maximum grid size in units (default: 7)")
    parser.add_argument("--min-units-x", type=int, default=1,
                        help="Minimum X grid size in units (default: 1). "
                             "Forces the bin to be at least this wide even if "
                             "the tools would fit in a smaller grid.")
    parser.add_argument("--min-units-y", type=int, default=1,
                        help="Minimum Y grid size in units (default: 1). "
                             "Forces the bin to be at least this tall even if "
                             "the tools would fit in a smaller grid.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: generated/)")

    args = parser.parse_args()

    layout_tools(
        dxf_paths=args.dxf_files,
        gap_mm=args.gap,
        gridfinity_unit=args.grid_unit,
        max_units=args.max_units,
        min_units_x=args.min_units_x,
        min_units_y=args.min_units_y,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
