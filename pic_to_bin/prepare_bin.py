"""
Agent 4 (part 1): Prepare Bin Configuration

Reads the combined layout DXF from Agent 3 (or individual tool DXFs) and
produces a JSON config file for the Fusion 360 script.

The JSON embeds polygon coordinates (mm) for each tool so the Fusion script
can draw them as sketch lines instead of importing a DXF (which loses layer
information).
"""

import argparse
import json
import math
from pathlib import Path

import ezdxf
import pyclipper

from pic_to_bin.trace_export import _compute_bbox


# ---------------------------------------------------------------------------
# Gridfinity constants
# ---------------------------------------------------------------------------

GRID_UNIT_MM = 42.0
HEIGHT_UNIT_MM = 7.0
WALL_THICKNESS_MM = 1.6
FLOOR_THICKNESS_MM = 1.2
CLEARANCE_MM = 0.5
STACKING_LIP_HEIGHT_MM = 2.6
THIN_WALL_THRESHOLD_MM = 2.0
BASE_PROFILE_HEIGHT_MM = 2.6
SLOT_FLOOR_CLEARANCE_MM = 1.0
DECK_INSET_MM = 2.0  # distance inside lip perimeter for deck lowering


# ---------------------------------------------------------------------------
# DXF loading
# ---------------------------------------------------------------------------

def load_layout_dxf(dxf_path: str) -> dict:
    """Read a combined layout DXF and extract tool polygons and bin boundary.

    Returns a dict with:
        - tools: list of dicts, each with name, inner/tolerance/slot polys,
          and bounding box (coordinates already in final position)
        - bin_boundary: dict with width_mm, height_mm
        - grid_x, grid_y: gridfinity unit counts
    """
    path = Path(dxf_path)
    if not path.exists():
        raise FileNotFoundError(f"DXF not found: {dxf_path}")

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    inner_polys = []
    tolerance_polys = []
    slot_polys = []
    bin_boundary = None

    for entity in msp.query("LWPOLYLINE"):
        layer = entity.dxf.layer.upper()
        points = [(p[0], p[1]) for p in entity.get_points(format="xy")]
        if len(points) < 3:
            continue

        if layer == "BIN_BOUNDARY":
            bbox = _compute_bbox([points])
            bin_boundary = {
                "width_mm": bbox["width_mm"],
                "height_mm": bbox["height_mm"],
            }
        elif layer == "TOLERANCE":
            tolerance_polys.append(points)
        elif layer == "SLOT":
            slot_polys.append(points)
        else:
            inner_polys.append(points)

    if bin_boundary is None:
        raise ValueError(f"No BIN_BOUNDARY layer found in {dxf_path}")
    if not inner_polys:
        raise ValueError(f"No polylines found on layer '0' in {dxf_path}")

    # Compute grid units from bin boundary
    grid_x = round(bin_boundary["width_mm"] / GRID_UNIT_MM)
    grid_y = round(bin_boundary["height_mm"] / GRID_UNIT_MM)

    # Group polygons into tools by spatial proximity.
    # Each inner polygon gets its closest tolerance and slot polygons.
    tools = _group_polygons_into_tools(inner_polys, tolerance_polys, slot_polys)

    return {
        "tools": tools,
        "bin_boundary": bin_boundary,
        "grid_x": grid_x,
        "grid_y": grid_y,
    }


def _poly_centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute the centroid of a polygon (simple average of vertices)."""
    n = len(poly)
    cx = sum(x for x, y in poly) / n
    cy = sum(y for x, y in poly) / n
    return cx, cy


def _poly_distance(poly_a: list, poly_b: list) -> float:
    """Distance between centroids of two polygons."""
    ax, ay = _poly_centroid(poly_a)
    bx, by = _poly_centroid(poly_b)
    return math.hypot(ax - bx, ay - by)


def _group_polygons_into_tools(
    inner_polys: list, tolerance_polys: list, slot_polys: list
) -> list[dict]:
    """Group polygons into tools based on spatial proximity.

    Each inner polygon defines a tool. All nearby tolerance and slot polygons
    are assigned to it (pyclipper offset can produce multiple tolerance polygons
    for tools with narrow necks or concavities).
    """
    # Each tolerance/slot poly goes to its CLOSEST tool body. Assigning by a
    # fixed distance threshold means the first tool (in iteration order) grabs
    # everything within range — and a slot is up to a full bin-diameter away
    # from a body, so a single threshold can't both reach a tool's own slot
    # and reject another tool's slot.
    tools = [
        {
            "inner_polys_mm": [[[x, y] for x, y in inner]],
            "tolerance_polys_mm": [],
            "slot_polys_mm": [],
            "bbox_mm": _compute_bbox([inner]),
        }
        for inner in inner_polys
    ]

    def _assign_to_closest(poly, target_key, max_distance):
        best_idx = -1
        best_d = float("inf")
        for ti, inner in enumerate(inner_polys):
            d = _poly_distance(inner, poly)
            if d < best_d:
                best_d = d
                best_idx = ti
        if best_idx >= 0 and best_d < max_distance:
            tools[best_idx][target_key].append([[x, y] for x, y in poly])

    for tol in tolerance_polys:
        _assign_to_closest(tol, "tolerance_polys_mm", max_distance=50)
    for sl in slot_polys:
        _assign_to_closest(sl, "slot_polys_mm", max_distance=200)

    return tools


# ---------------------------------------------------------------------------
# Bin parameter computation
# ---------------------------------------------------------------------------

def compute_bin_params(grid_x: int, grid_y: int,
                       height_units: int) -> dict:
    """Calculate bin dimensions from gridfinity parameters.

    Returns a dict with all physical dimensions in mm.
    """
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "height_units": height_units,
        "bin_width_mm": grid_x * GRID_UNIT_MM,
        "bin_height_mm": grid_y * GRID_UNIT_MM,
        "bin_depth_mm": height_units * HEIGHT_UNIT_MM,
        "inner_width_mm": grid_x * GRID_UNIT_MM - 2 * CLEARANCE_MM,
        "inner_height_mm": grid_y * GRID_UNIT_MM - 2 * CLEARANCE_MM,
        "wall_thickness_mm": WALL_THICKNESS_MM,
        "floor_thickness_mm": FLOOR_THICKNESS_MM,
        "clearance_mm": CLEARANCE_MM,
        "stacking_lip_height_mm": STACKING_LIP_HEIGHT_MM,
    }




# ---------------------------------------------------------------------------
# Thin-wall elimination
# ---------------------------------------------------------------------------

def _eliminate_thin_walls(polys_mm: list, bin_width_mm: float,
                          bin_height_mm: float,
                          threshold_mm: float = THIN_WALL_THRESHOLD_MM
                          ) -> list:
    """Snap polygon vertices near bin boundaries to eliminate thin walls.

    If any vertex coordinate is within threshold_mm of a bin edge, snap it
    to that edge so the cutout extends fully through the wall.
    """
    result = []
    for poly in polys_mm:
        new_poly = []
        for pt in poly:
            x, y = pt[0], pt[1]
            if x < threshold_mm:
                x = 0.0
            elif x > bin_width_mm - threshold_mm:
                x = bin_width_mm
            if y < threshold_mm:
                y = 0.0
            elif y > bin_height_mm - threshold_mm:
                y = bin_height_mm
            new_poly.append([x, y])
        result.append(new_poly)
    return result


# ---------------------------------------------------------------------------
# Slot clipping to bin boundary
# ---------------------------------------------------------------------------

SLOT_INSET_MM = 4.0  # slot cutouts stop this far inside the bin edge


def _center_tools_in_bin(tools: list, bin_width_mm: float,
                         bin_height_mm: float) -> None:
    """Shift every tool's polys so the combined tool-body (inner + tolerance)
    bbox sits in the middle of the bin.

    Layout packing places tools against the (0, 0) corner; rounding the bin
    up to whole gridfinity units leaves uneven slack on the high-x/high-y
    sides. This redistributes the slack evenly so the tool body is centered.
    Slot polys are translated by the same dx/dy so they keep their position
    relative to the body, but they are *not* part of the bbox calculation —
    a slot that extends past the body sideways would otherwise pull the body
    off-center toward the opposite side. Mutates `tools` in place.
    """
    body_polys = []
    for tool in tools:
        body_polys.extend(tool.get("inner_polys_mm", []))
        body_polys.extend(tool.get("tolerance_polys_mm", []))
    if not body_polys:
        return

    bbox = _compute_bbox(body_polys)
    dx = (bin_width_mm - bbox["width_mm"]) / 2.0 - bbox["min_x"]
    dy = (bin_height_mm - bbox["height_mm"]) / 2.0 - bbox["min_y"]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return

    def shift(polys):
        return [[[x + dx, y + dy] for x, y in p] for p in polys]

    for tool in tools:
        tool["inner_polys_mm"] = shift(tool.get("inner_polys_mm", []))
        tool["tolerance_polys_mm"] = shift(
            tool.get("tolerance_polys_mm", []))
        tool["slot_polys_mm"] = shift(tool.get("slot_polys_mm", []))


def _clip_to_bin_boundary(polys_mm: list, bin_width_mm: float,
                          bin_height_mm: float,
                          inset_mm: float = SLOT_INSET_MM) -> list:
    """Clip slot polygons to an inset rectangle inside the bin boundary.

    Finger slots can extend beyond the bin edges (by design — they overhang
    the tool outline). This clips them so the cutout stops inset_mm inside
    the bin wall, preserving structural wall material.
    Uses pyclipper polygon intersection.
    """
    SCALE = 1000  # pyclipper uses integer coordinates

    clip_rect = [
        (int(inset_mm * SCALE), int(inset_mm * SCALE)),
        (int((bin_width_mm - inset_mm) * SCALE), int(inset_mm * SCALE)),
        (int((bin_width_mm - inset_mm) * SCALE), int((bin_height_mm - inset_mm) * SCALE)),
        (int(inset_mm * SCALE), int((bin_height_mm - inset_mm) * SCALE)),
    ]

    result = []
    for poly in polys_mm:
        subject = [(int(pt[0] * SCALE), int(pt[1] * SCALE)) for pt in poly]

        pc = pyclipper.Pyclipper()
        pc.AddPath(subject, pyclipper.PT_SUBJECT, True)
        pc.AddPath(clip_rect, pyclipper.PT_CLIP, True)

        clipped = pc.Execute(pyclipper.CT_INTERSECTION,
                             pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        for path in clipped:
            result.append([[pt[0] / SCALE, pt[1] / SCALE] for pt in path])

    return result if result else polys_mm


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

def build_config(layout: dict, tool_heights: dict | float,
                 height_units: int = None,
                 stacking_lip: bool = True) -> dict:
    """Assemble the full JSON config for the Fusion 360 script.

    Args:
        layout: dict from load_layout_dxf()
        tool_heights: tool heights in mm — either a single float (uniform)
                      or a dict mapping tool index to height in mm.
        height_units: gridfinity height units (auto-calculated if None)
        stacking_lip: if False, omit the stacking lip (shorter overall bin
                      for shallow drawers). Pocket depth is unchanged.
    """
    grid_x = layout["grid_x"]
    grid_y = layout["grid_y"]
    tools = layout["tools"]

    # Resolve per-tool heights
    tool_height_values = []
    for i in range(len(tools)):
        if isinstance(tool_heights, (int, float)):
            tool_height_values.append(float(tool_heights))
        elif isinstance(tool_heights, dict):
            h = tool_heights.get(i, tool_heights.get("default", None))
            if h is None:
                raise ValueError(
                    f"No tool height for tool {i}. Provide a height for "
                    f"every tool or use a uniform value.")
            tool_height_values.append(float(h))
        else:
            raise ValueError(f"Invalid tool_heights type: {type(tool_heights)}")

    FLOOR_MIN_MM = 1.0  # solid floor below pocket — pocket sits this low

    # Auto-calculate height if not specified.
    # Pocket sits as low as possible (1mm above bin floor), so the bin only
    # needs to be tall enough to fit the tallest tool plus that 1mm floor,
    # rounded up to the nearest gridfinity height unit.
    if height_units is None:
        min_bin_height = max(tool_height_values) + FLOOR_MIN_MM
        height_units = max(1, math.ceil(min_bin_height / HEIGHT_UNIT_MM))

    bin_params = compute_bin_params(grid_x, grid_y, height_units)
    bin_d = height_units * HEIGHT_UNIT_MM

    # Compute pocket depths (distance from bin top down to pocket floor at 1mm)
    pocket_depths = [bin_d - FLOOR_MIN_MM for _ in tool_height_values]

    # Deck lowering: deck rises to half the tallest tool's height above the
    # pocket floor — so the upper half of the tool stands proud of the deck
    # for finger access while the lower half is buried in the pocket.
    deck_top_z = FLOOR_MIN_MM + max(tool_height_values) / 2.0
    deck_lowering = max(0.0, bin_d - deck_top_z)

    # Center the cutout (inner + tolerance + slot) in the bin before
    # thin-wall and slot-clip snapping use the final positions.
    _center_tools_in_bin(tools, bin_params["bin_width_mm"],
                         bin_params["bin_height_mm"])

    # Build tool entries
    tool_entries = []
    for i, tool in enumerate(tools):
        entry = {
            "name": f"tool_{i}",
            "inner_polys_mm": tool["inner_polys_mm"],
            "tolerance_polys_mm": _eliminate_thin_walls(
                tool["tolerance_polys_mm"],
                bin_params["bin_width_mm"],
                bin_params["bin_height_mm"]),
            "slot_polys_mm": _clip_to_bin_boundary(
                tool["slot_polys_mm"],
                bin_params["bin_width_mm"],
                bin_params["bin_height_mm"]),
            "tool_height_mm": tool_height_values[i],
            "pocket_depth_mm": pocket_depths[i],
        }
        tool_entries.append(entry)

    config = {
        **bin_params,
        "stacking_lip": bool(stacking_lip),
        "deck_lowering_mm": round(deck_lowering, 2),
        "deck_inset_mm": DECK_INSET_MM,
        "tools": tool_entries,
    }

    return config


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_bin(dxf_path: str, tool_heights: dict | float,
                height_units: int = None,
                stacking_lip: bool = True,
                output_path: str = None) -> str:
    """Main pipeline: load layout DXF, compute params, write JSON config.

    Returns the path to the written JSON file.
    """
    if output_path is None:
        output_path = str(Path("generated") / "bin_config.json")

    print(f"Loading layout DXF: {dxf_path}")
    layout = load_layout_dxf(dxf_path)

    print(f"  Grid: {layout['grid_x']}x{layout['grid_y']} units")
    print(f"  Tools found: {len(layout['tools'])}")

    config = build_config(
        layout, tool_heights,
        height_units=height_units,
        stacking_lip=stacking_lip,
    )

    print(f"  Height: {config['height_units']} units "
          f"({config['bin_depth_mm']:.0f}mm)")
    print(f"  Bin: {config['bin_width_mm']:.0f}x{config['bin_height_mm']:.0f}mm "
          f"x {config['bin_depth_mm']:.0f}mm")
    print(f"  Stacking lip: {'yes' if config['stacking_lip'] else 'no'}")

    for tool in config["tools"]:
        print(f"  {tool['name']}: height={tool['tool_height_mm']:.1f}mm, "
              f"pocket={tool['pocket_depth_mm']:.1f}mm")

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig written: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_tool_height_arg(height_strs: list[str]) -> dict | float:
    """Parse --tool-height arguments into a single float or index→float dict.

    Formats:
        --tool-height 17.0                        → 17.0 (uniform)
        --tool-height 0=17.0 --tool-height 1=20.0 → {0: 17.0, 1: 20.0}
    """
    if len(height_strs) == 1 and "=" not in height_strs[0]:
        return float(height_strs[0])

    heights = {}
    for s in height_strs:
        if "=" in s:
            key, val = s.split("=", 1)
            try:
                heights[int(key)] = float(val)
            except ValueError:
                heights[key] = float(val)
        else:
            heights["default"] = float(s)

    return heights


def main():
    parser = argparse.ArgumentParser(
        description="Prepare gridfinity bin config JSON from layout DXF"
    )
    parser.add_argument("dxf_file",
                        help="Combined layout DXF from layout_tools.py")
    parser.add_argument("--tool-height", action="append", dest="tool_heights",
                        required=True,
                        help="Tool height in mm (required). Use INDEX=HEIGHT "
                             "for per-tool (e.g. --tool-height 0=17.0 "
                             "--tool-height 1=20.0)")
    parser.add_argument("--height-units", type=int, default=None,
                        help="Bin height in gridfinity units (auto if omitted)")
    parser.add_argument("--stacking", type=_parse_bool_arg, default=True,
                        metavar="true|false",
                        help="Generate stacking lip (default: true). Set "
                             "false for shorter bins without the lip.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: generated/bin_config.json)")

    args = parser.parse_args()

    tool_heights = _parse_tool_height_arg(args.tool_heights)

    prepare_bin(
        dxf_path=args.dxf_file,
        tool_heights=tool_heights,
        height_units=args.height_units,
        stacking_lip=args.stacking,
        output_path=args.output,
    )


def _parse_bool_arg(value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(
        f"Expected true/false, got: {value!r}")


if __name__ == "__main__":
    main()
