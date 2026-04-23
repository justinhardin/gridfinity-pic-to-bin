"""
Agent 4 (part 2): Fusion 360 Script — Gridfinity Bin Generator

Reads a bin_config.json (from prepare_bin.py) and creates a parametric
gridfinity bin with tool pocket cutouts in Fusion 360.

Usage:
    1. In Fusion 360: File → Scripts and Add-Ins
    2. Click "+" next to "My Scripts", select this folder
    3. Click "Run"
    4. Select your bin_config.json in the file dialog

Dependencies: Only adsk.core, adsk.fusion, json, math, os (all built into
Fusion 360's Python environment — no pip installs needed).

All coordinates in the JSON are in mm. This script converts to cm internally
(Fusion 360's native unit).
"""

import json
import math
import os
import traceback

import adsk.core
import adsk.fusion

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Gridfinity stacking lip — matches FusionGridfinityGenerator reference.
# The lip is a solid block with a base-profile-shaped cutout inside,
# creating the mating surface for stacking bins.
LIP_HEIGHT_MM = 4.4              # total lip body height above bin wall
LIP_TOP_RECESS_MM = 0.6          # small recess cut from lip top
LIP_CUTOUT_CORNER_R_MM = 4.5     # cutout corner radius (4.0 spec + 0.5 clearance)

# Gridfinity base interface profile (mm, measured downward from bin floor):
#   0.0  — bin floor (pad shoulder, full width = 41.5mm = 42 - 2×0.25 clearance)
#   2.4  — bottom of cone (pad narrows to 36.7mm via 45° equal-dist chamfer)
#   4.2  — bottom of straight post (still 36.7mm)
#   5.0  — bottom face (35.1mm after 0.8mm bottom chamfer)
# Source: FusionGridfinityGenerator baseGenerator.py
BASE_PROFILE_HEIGHT_MM = 5.0      # total pad height (was incorrectly 2.6)
BASE_PAD_CONE_H_MM     = 2.4      # height of chamfered cone section
BASE_PAD_STRAIGHT_H_MM = 2.6      # straight + bottom-chamfer section (1.8 + 0.8)
BASE_CONE_CHAMFER_MM   = 2.4      # equal-distance chamfer → 45° mating cone
BASE_BOT_CHAMFER_MM    = 0.8      # bottom edge chamfer
BASE_CLEARANCE_MM      = 0.25     # XY clearance per side (gridfinity spec)
BASE_CORNER_RADIUS_MM  = 3.75     # = 4.0 mm spec − 0.25 mm clearance
SLOT_FLOOR_CLEARANCE_MM = 1.0


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def mm(v: float) -> float:
    """Convert millimeters to centimeters (Fusion 360 internal unit)."""
    return v / 10.0


# ---------------------------------------------------------------------------
# JSON config
# ---------------------------------------------------------------------------

def read_config(path: str) -> dict:
    """Load and validate the bin config JSON."""
    with open(path, "r") as f:
        config = json.load(f)

    required = ["grid_x", "grid_y", "height_units", "bin_width_mm",
                "bin_height_mm", "tools"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")

    return config


# ---------------------------------------------------------------------------
# Bin body
# ---------------------------------------------------------------------------

def create_bin_body(root_comp, config: dict):
    """Create the outer bin body as a solid box.

    Tool pockets cut into this solid block — no shell needed.
    Uses the full bin boundary dimensions. Clearance for the gridfinity
    grid fit is handled by the base interface (slightly smaller footprint).
    """
    bin_w = mm(config["bin_width_mm"])
    bin_h = mm(config["bin_height_mm"])
    bin_d = mm(config["height_units"] * 7.0)

    sketches = root_comp.sketches
    xy_plane = root_comp.xYConstructionPlane
    sketch = sketches.add(xy_plane)

    lines = sketch.sketchCurves.sketchLines
    p0 = adsk.core.Point3D.create(0, 0, 0)
    p1 = adsk.core.Point3D.create(bin_w, 0, 0)
    p2 = adsk.core.Point3D.create(bin_w, bin_h, 0)
    p3 = adsk.core.Point3D.create(0, bin_h, 0)

    lines.addByTwoPoints(p0, p1)
    lines.addByTwoPoints(p1, p2)
    lines.addByTwoPoints(p2, p3)
    lines.addByTwoPoints(p3, p0)

    profile = sketch.profiles.item(0)

    extrudes = root_comp.features.extrudeFeatures
    ext_input = extrudes.createInput(
        profile, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ext_input.setDistanceExtent(
        False, adsk.core.ValueInput.createByReal(bin_d))
    extrude = extrudes.add(ext_input)

    body = extrude.bodies.item(0)
    gx = config["grid_x"]
    gy = config["grid_y"]
    gz = config["height_units"]
    body.name = f"bin_body_{gx}x{gy}x{gz}"

    return body


# ---------------------------------------------------------------------------
# Stacking lip
# ---------------------------------------------------------------------------

def create_stacking_lip(root_comp, config: dict):
    """Create the gridfinity stacking lip with correct mating profile.

    Matches the FusionGridfinityGenerator reference implementation:
    1. Solid block extruded above bin body (4.4mm)
    2. Outer vertical corners filleted (4mm radius, full body height)
    3. Inner mating cutout (base profile inverse) with 45° chamfer cone
    4. Small top recess (0.6mm)

    The lip is the inverse of the base interface profile — when another
    bin stacks on top, its base pads slot into this lip cavity.
    """
    bin_w = mm(config["bin_width_mm"])
    bin_h = mm(config["bin_height_mm"])
    bin_d = mm(config["height_units"] * 7.0)
    lip_h = mm(LIP_HEIGHT_MM)
    lip_top_z = bin_d + lip_h

    xy_plane = root_comp.xYConstructionPlane
    planes = root_comp.constructionPlanes
    extrudes = root_comp.features.extrudeFeatures
    tol = 0.005  # edge detection tolerance (cm)

    # ------------------------------------------------------------------
    # Step 1: Solid lip block above bin body
    # ------------------------------------------------------------------
    plane_input = planes.createInput()
    plane_input.setByOffset(
        xy_plane, adsk.core.ValueInput.createByReal(bin_d))
    top_plane = planes.add(plane_input)

    sketch = root_comp.sketches.add(top_plane)
    lines = sketch.sketchCurves.sketchLines
    pts = [adsk.core.Point3D.create(x, y, 0)
           for x, y in [(0, 0), (bin_w, 0), (bin_w, bin_h), (0, bin_h)]]
    for i in range(4):
        lines.addByTwoPoints(pts[i], pts[(i + 1) % 4])

    ext_input = extrudes.createInput(
        sketch.profiles.item(0),
        adsk.fusion.FeatureOperations.JoinFeatureOperation)
    ext_input.setDistanceExtent(
        False, adsk.core.ValueInput.createByReal(lip_h))
    extrudes.add(ext_input)

    # ------------------------------------------------------------------
    # Step 2: Fillet outer vertical corners (4mm, full body height)
    # ------------------------------------------------------------------
    body = _find_bin_body(root_comp)
    fillet_r = mm(4.0)
    vert_edges = adsk.core.ObjectCollection.create()
    for i in range(body.edges.count):
        edge = body.edges.item(i)
        sp = edge.startVertex.geometry
        ep = edge.endVertex.geometry
        # Vertical edges at the 4 outer corners
        if abs(sp.x - ep.x) < tol and abs(sp.y - ep.y) < tol:
            edge_len = abs(sp.z - ep.z)
            if edge_len < 0.1:
                continue
            x, y = sp.x, sp.y
            at_corner = ((abs(x) < tol or abs(x - bin_w) < tol) and
                         (abs(y) < tol or abs(y - bin_h) < tol))
            if at_corner:
                vert_edges.add(edge)

    if vert_edges.count > 0:
        fi = root_comp.features.filletFeatures.createInput()
        fi.addConstantRadiusEdgeSet(
            vert_edges,
            adsk.core.ValueInput.createByReal(fillet_r),
            True)  # tangent chain catches connected segments
        root_comp.features.filletFeatures.add(fi)

    # ------------------------------------------------------------------
    # Step 3: Inner mating cutout (base profile inverse)
    #
    # The cutout is a two-part body matching the base interface shape:
    #   Top section: wide rounded rect, chamfered 45° (mating cone)
    #   Straight section: narrower rounded rect below the chamfer
    # ------------------------------------------------------------------
    cl = mm(BASE_CLEARANCE_MM)                  # 0.25mm → 0.025 cm
    cone_h = mm(BASE_PAD_CONE_H_MM)             # 2.4mm → 0.24 cm
    cutout_r = mm(LIP_CUTOUT_CORNER_R_MM)       # 4.5mm → 0.45 cm

    # Top section: wide rect at cutout origin (5mm above bin top)
    cutout_z = bin_d + mm(BASE_PROFILE_HEIGHT_MM)  # bin_d + 0.50 cm
    top_x0, top_y0 = -cl, -cl
    top_x1, top_y1 = bin_w + cl, bin_h + cl

    cutout_plane_input = planes.createInput()
    cutout_plane_input.setByOffset(
        xy_plane, adsk.core.ValueInput.createByReal(cutout_z))
    cutout_plane = planes.add(cutout_plane_input)

    top_sketch = root_comp.sketches.add(cutout_plane)
    _draw_rounded_rect(top_sketch, top_x0, top_y0, top_x1, top_y1, cutout_r)

    top_profile = _find_smallest_profile(top_sketch)
    ext_input = extrudes.createInput(
        top_profile,
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ext_input.setOneSideExtent(
        adsk.fusion.DistanceExtentDefinition.create(
            adsk.core.ValueInput.createByReal(cone_h)),
        adsk.fusion.ExtentDirections.NegativeExtentDirection)
    top_extrude = extrudes.add(ext_input)
    cutout_body = top_extrude.bodies.item(0)
    cutout_body.name = "Lip cutout"

    # Chamfer bottom edges of top section (2.4mm equal-distance → 45° cone)
    chamfer_z = cutout_z - cone_h
    chamfer_edges = adsk.core.ObjectCollection.create()
    for i in range(cutout_body.edges.count):
        edge = cutout_body.edges.item(i)
        sp = edge.startVertex.geometry
        ep = edge.endVertex.geometry
        if abs(sp.z - chamfer_z) < tol and abs(ep.z - chamfer_z) < tol:
            chamfer_edges.add(edge)

    if chamfer_edges.count > 0:
        ci = root_comp.features.chamferFeatures.createInput2()
        ci.chamferEdgeSets.addEqualDistanceChamferEdgeSet(
            chamfer_edges,
            adsk.core.ValueInput.createByReal(cone_h),
            True)
        root_comp.features.chamferFeatures.add(ci)

    # Straight section: narrower rect below the chamfer
    s_x0 = top_x0 + cone_h    # inset by chamfer distance
    s_y0 = top_y0 + cone_h
    s_x1 = top_x1 - cone_h
    s_y1 = top_y1 - cone_h
    s_r = max(cutout_r - cone_h, mm(0.1))   # 4.5 − 2.4 = 2.1mm
    straight_h = mm(BASE_PROFILE_HEIGHT_MM) - cone_h  # 5.0 − 2.4 = 2.6mm

    straight_plane_input = planes.createInput()
    straight_plane_input.setByOffset(
        xy_plane, adsk.core.ValueInput.createByReal(chamfer_z))
    straight_plane = planes.add(straight_plane_input)

    straight_sketch = root_comp.sketches.add(straight_plane)
    _draw_rounded_rect(straight_sketch, s_x0, s_y0, s_x1, s_y1, s_r)

    straight_profile = _find_smallest_profile(straight_sketch)
    ext_input = extrudes.createInput(
        straight_profile,
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ext_input.setOneSideExtent(
        adsk.fusion.DistanceExtentDefinition.create(
            adsk.core.ValueInput.createByReal(straight_h)),
        adsk.fusion.ExtentDirections.NegativeExtentDirection)
    straight_extrude = extrudes.add(ext_input)
    straight_body = straight_extrude.bodies.item(0)

    # Join the two cutout parts into one body
    combines = root_comp.features.combineFeatures
    join_coll = adsk.core.ObjectCollection.create()
    join_coll.add(straight_body)
    join_input = combines.createInput(cutout_body, join_coll)
    join_input.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
    combines.add(join_input)

    # Cut the combined cutout from the bin body
    body = _find_bin_body(root_comp)
    cut_coll = adsk.core.ObjectCollection.create()
    cut_coll.add(cutout_body)
    cut_input = combines.createInput(body, cut_coll)
    cut_input.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
    combines.add(cut_input)

    # ------------------------------------------------------------------
    # Step 4: Top recess (0.6mm cut from lip top)
    # ------------------------------------------------------------------
    recess_h = mm(LIP_TOP_RECESS_MM)
    lip_top_plane_input = planes.createInput()
    lip_top_plane_input.setByOffset(
        xy_plane, adsk.core.ValueInput.createByReal(lip_top_z))
    lip_top_plane = planes.add(lip_top_plane_input)

    recess_sketch = root_comp.sketches.add(lip_top_plane)
    margin = mm(1.0)  # oversized rect so it fully covers filleted corners
    rpts = [adsk.core.Point3D.create(x, y, 0)
            for x, y in [(-margin, -margin), (bin_w + margin, -margin),
                          (bin_w + margin, bin_h + margin),
                          (-margin, bin_h + margin)]]
    rlines = recess_sketch.sketchCurves.sketchLines
    for i in range(4):
        rlines.addByTwoPoints(rpts[i], rpts[(i + 1) % 4])

    ext_input = extrudes.createInput(
        recess_sketch.profiles.item(0),
        adsk.fusion.FeatureOperations.CutFeatureOperation)
    ext_input.setOneSideExtent(
        adsk.fusion.DistanceExtentDefinition.create(
            adsk.core.ValueInput.createByReal(recess_h)),
        adsk.fusion.ExtentDirections.NegativeExtentDirection)
    extrudes.add(ext_input)



# ---------------------------------------------------------------------------
# Deck lowering
# ---------------------------------------------------------------------------

def lower_deck(root_comp, config: dict):
    """Lower the inner deck surface to reduce visible pocket depth.

    Creates a step inside the stacking lip perimeter, preserving a solid
    border (wall_thickness + deck_inset) around the lip to avoid thin walls.
    """
    lowering = config.get("deck_lowering_mm", 0)
    if not lowering or lowering <= 0:
        return

    bin_w = mm(config["bin_width_mm"])
    bin_h = mm(config["bin_height_mm"])
    bin_d = mm(config["height_units"] * 7.0)
    wall_t = mm(config.get("wall_thickness_mm", 1.6))
    deck_inset = mm(config.get("deck_inset_mm", 2.0))
    inset = wall_t + deck_inset  # total inset from bin edge

    # Construction plane at top of bin body (below lip)
    xy_plane = root_comp.xYConstructionPlane
    planes = root_comp.constructionPlanes
    plane_input = planes.createInput()
    plane_input.setByOffset(
        xy_plane, adsk.core.ValueInput.createByReal(bin_d))
    deck_plane = planes.add(plane_input)

    sketch = root_comp.sketches.add(deck_plane)
    lines = sketch.sketchCurves.sketchLines
    p0 = adsk.core.Point3D.create(inset, inset, 0)
    p1 = adsk.core.Point3D.create(bin_w - inset, inset, 0)
    p2 = adsk.core.Point3D.create(bin_w - inset, bin_h - inset, 0)
    p3 = adsk.core.Point3D.create(inset, bin_h - inset, 0)
    lines.addByTwoPoints(p0, p1)
    lines.addByTwoPoints(p1, p2)
    lines.addByTwoPoints(p2, p3)
    lines.addByTwoPoints(p3, p0)

    if sketch.profiles.count > 0:
        profile = sketch.profiles.item(0)
        extrudes = root_comp.features.extrudeFeatures
        ext_input = extrudes.createInput(
            profile, adsk.fusion.FeatureOperations.CutFeatureOperation)
        ext_input.setOneSideExtent(
            adsk.fusion.DistanceExtentDefinition.create(
                adsk.core.ValueInput.createByReal(mm(lowering))),
            adsk.fusion.ExtentDirections.NegativeExtentDirection)
        extrudes.add(ext_input)



# ---------------------------------------------------------------------------
# Tool pockets
# ---------------------------------------------------------------------------

def draw_polygon_sketch(sketch, poly_cm: list[list[float]]):
    """Draw a closed polygon as connected sketch lines.

    Args:
        sketch: Fusion 360 sketch object
        poly_cm: list of [x, y] coordinates in cm
    """
    lines = sketch.sketchCurves.sketchLines
    n = len(poly_cm)
    for i in range(n):
        x1, y1 = poly_cm[i]
        x2, y2 = poly_cm[(i + 1) % n]
        p1 = adsk.core.Point3D.create(x1, y1, 0)
        p2 = adsk.core.Point3D.create(x2, y2, 0)
        lines.addByTwoPoints(p1, p2)


def cut_tool_pockets(root_comp, config: dict):
    """For each tool, draw tolerance polygons and extrude-cut downward.

    Uses tolerance polygons (which include fit clearance) for the pocket shape.
    Polygon coordinates from the layout are in mm, matching the bin boundary.
    Cuts start above the stacking lip so edge-reaching profiles cut the lip too.
    """
    bin_d = mm(config["height_units"] * 7.0)
    lip_h = mm(LIP_HEIGHT_MM)

    # Construction plane above the stacking lip so cuts go through lip + body
    xy_plane = root_comp.xYConstructionPlane
    planes = root_comp.constructionPlanes
    plane_input = planes.createInput()
    plane_input.setByOffset(
        xy_plane, adsk.core.ValueInput.createByReal(bin_d + lip_h))
    top_plane = planes.add(plane_input)

    for tool in config["tools"]:
        depth_cm = mm(tool["pocket_depth_mm"]) + lip_h

        # Use tolerance polys for the pocket cut (they include clearance)
        polys = tool.get("tolerance_polys_mm", [])
        if not polys:
            polys = tool.get("inner_polys_mm", [])

        for poly_mm in polys:
            sketch = root_comp.sketches.add(top_plane)

            # Convert mm to cm
            poly_cm = [[mm(pt[0]), mm(pt[1])] for pt in poly_mm]
            draw_polygon_sketch(sketch, poly_cm)

            if sketch.profiles.count > 0:
                profile = _find_smallest_profile(sketch)

                extrudes = root_comp.features.extrudeFeatures
                ext_input = extrudes.createInput(
                    profile,
                    adsk.fusion.FeatureOperations.CutFeatureOperation)
                # Cut downward (negative Z) into the body by pocket depth
                ext_input.setOneSideExtent(
                    adsk.fusion.DistanceExtentDefinition.create(
                        adsk.core.ValueInput.createByReal(depth_cm)),
                    adsk.fusion.ExtentDirections.NegativeExtentDirection)
                extrudes.add(ext_input)



def cut_slots(root_comp, config: dict):
    """Cut finger slots to the same depth as their tool's pocket.

    Cuts start above the stacking lip (like pockets) and use the same
    depth so the slot floor matches the pocket floor.
    """
    bin_d = mm(config["height_units"] * 7.0)
    lip_h = mm(LIP_HEIGHT_MM)

    # Construction plane above the stacking lip
    xy_plane = root_comp.xYConstructionPlane
    planes = root_comp.constructionPlanes
    plane_input = planes.createInput()
    plane_input.setByOffset(
        xy_plane, adsk.core.ValueInput.createByReal(bin_d + lip_h))
    top_plane = planes.add(plane_input)

    for tool in config["tools"]:
        slot_polys = tool.get("slot_polys_mm", [])
        # Same depth as the tool's pocket cut (through lip + pocket)
        slot_depth = mm(tool["pocket_depth_mm"]) + lip_h

        for poly_mm in slot_polys:
            sketch = root_comp.sketches.add(top_plane)

            poly_cm = [[mm(pt[0]), mm(pt[1])] for pt in poly_mm]
            draw_polygon_sketch(sketch, poly_cm)

            if sketch.profiles.count > 0:
                profile = _find_smallest_profile(sketch)

                try:
                    extrudes = root_comp.features.extrudeFeatures
                    ext_input = extrudes.createInput(
                        profile,
                        adsk.fusion.FeatureOperations.CutFeatureOperation)
                    # Cut to same floor as pocket
                    ext_input.setOneSideExtent(
                        adsk.fusion.DistanceExtentDefinition.create(
                            adsk.core.ValueInput.createByReal(slot_depth)),
                        adsk.fusion.ExtentDirections.NegativeExtentDirection)
                    extrudes.add(ext_input)
                except Exception:
                    # Slot may overlap with already-cut pocket — skip
                    pass



# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

def _find_bin_body(root_comp):
    """Return the main bin body (named bin_body_*), or the first body."""
    for i in range(root_comp.bRepBodies.count):
        b = root_comp.bRepBodies.item(i)
        if b.name.startswith("bin_body_"):
            return b
    return root_comp.bRepBodies.item(0)


def create_base_interface(root_comp, config: dict):
    """Create the gridfinity base mating geometry below the bin floor.

    Matches the FusionGridfinityGenerator reference implementation.
    Each 42 mm grid unit gets a 5 mm-tall pad that clicks into a gridfinity
    baseplate socket.  The pad cross-section (measured downward from z = 0):

        z =  0.0 mm  — bin floor / pad shoulder (full width 41.5 mm)
        z = -2.4 mm  — cone ends (36.7 mm wide) via 45° equal-dist chamfer
        z = -4.2 mm  — straight post ends (still 36.7 mm)
        z = -5.0 mm  — pad bottom (35.1 mm after 0.8 mm bottom chamfer)

    Build sequence:
      Pass 1 — extrude top sections (41.5 mm pads, 2.4 mm downward)
      Pass 2 — chamfer edges at z = −2.4 mm (creates 45° mating cone)
      Pass 3 — extrude narrow straight posts (36.7 mm, 2.6 mm downward)
      Pass 4 — chamfer edges at z = −5.0 mm (0.8 mm bottom bevel)
    """
    grid_x   = config["grid_x"]
    grid_y   = config["grid_y"]
    unit_mm  = 42.0

    pad_mm    = unit_mm - 2 * BASE_CLEARANCE_MM            # 41.5 mm
    corner_r  = BASE_CORNER_RADIUS_MM                      # 3.75 mm
    narrow_mm = pad_mm - 2 * BASE_CONE_CHAMFER_MM          # 36.7 mm
    narrow_r  = max(0.1, corner_r - BASE_CONE_CHAMFER_MM)  # 1.35 mm

    xy_plane = root_comp.xYConstructionPlane
    extrudes = root_comp.features.extrudeFeatures
    tol      = 0.005  # 0.05 mm — floating-point tolerance for edge z-check

    # ------------------------------------------------------------------
    # Pass 1: extrude pad top sections (full width, 2.4 mm downward)
    # ------------------------------------------------------------------
    for gx in range(grid_x):
        for gy in range(grid_y):
            x0 = mm(gx * unit_mm + BASE_CLEARANCE_MM)
            y0 = mm(gy * unit_mm + BASE_CLEARANCE_MM)
            x1 = mm((gx + 1) * unit_mm - BASE_CLEARANCE_MM)
            y1 = mm((gy + 1) * unit_mm - BASE_CLEARANCE_MM)

            sketch = root_comp.sketches.add(xy_plane)
            _draw_rounded_rect(sketch, x0, y0, x1, y1, mm(corner_r))

            if sketch.profiles.count > 0:
                profile = _find_smallest_profile(sketch)
                ext = extrudes.createInput(
                    profile, adsk.fusion.FeatureOperations.JoinFeatureOperation)
                ext.setOneSideExtent(
                    adsk.fusion.DistanceExtentDefinition.create(
                        adsk.core.ValueInput.createByReal(mm(BASE_PAD_CONE_H_MM))),
                    adsk.fusion.ExtentDirections.NegativeExtentDirection)
                extrudes.add(ext)

    # ------------------------------------------------------------------
    # Pass 2: chamfer edges at z = −2.4 mm → 45° mating cone
    # The chamfer consumes the entire side wall of the top section, leaving
    # the shoulder (at z = 0) untouched while tapering to 36.7 mm at z = −2.4.
    # ------------------------------------------------------------------
    try:
        body     = _find_bin_body(root_comp)
        target_z = -mm(BASE_PAD_CONE_H_MM)   # −0.24 cm

        cone_edges = adsk.core.ObjectCollection.create()
        for i in range(body.edges.count):
            edge = body.edges.item(i)
            sp = edge.startVertex.geometry
            ep = edge.endVertex.geometry
            if abs(sp.z - target_z) < tol and abs(ep.z - target_z) < tol:
                cone_edges.add(edge)

        if cone_edges.count > 0:
            chamfers = root_comp.features.chamferFeatures
            ci = chamfers.createInput2()
            ci.chamferEdgeSets.addEqualDistanceChamferEdgeSet(
                cone_edges,
                adsk.core.ValueInput.createByReal(mm(BASE_CONE_CHAMFER_MM)),
                True)   # tangent chain catches arc edges at rounded corners
            chamfers.add(ci)
    except Exception:
        pass  # non-fatal

    # ------------------------------------------------------------------
    # Pass 3: extrude narrow straight posts (36.7 mm, 2.6 mm downward)
    # Sketch on a construction plane at z = −2.4 mm.
    # ------------------------------------------------------------------
    planes      = root_comp.constructionPlanes
    plane_input = planes.createInput()
    plane_input.setByOffset(
        xy_plane,
        adsk.core.ValueInput.createByReal(-mm(BASE_PAD_CONE_H_MM)))
    mid_plane = planes.add(plane_input)

    for gx in range(grid_x):
        for gy in range(grid_y):
            cx = (gx + 0.5) * unit_mm
            cy = (gy + 0.5) * unit_mm
            x0 = mm(cx - narrow_mm / 2)
            y0 = mm(cy - narrow_mm / 2)
            x1 = mm(cx + narrow_mm / 2)
            y1 = mm(cy + narrow_mm / 2)

            sketch = root_comp.sketches.add(mid_plane)
            _draw_rounded_rect(sketch, x0, y0, x1, y1, mm(narrow_r))

            if sketch.profiles.count > 0:
                profile = _find_smallest_profile(sketch)
                ext = extrudes.createInput(
                    profile, adsk.fusion.FeatureOperations.JoinFeatureOperation)
                ext.setOneSideExtent(
                    adsk.fusion.DistanceExtentDefinition.create(
                        adsk.core.ValueInput.createByReal(mm(BASE_PAD_STRAIGHT_H_MM))),
                    adsk.fusion.ExtentDirections.NegativeExtentDirection)
                extrudes.add(ext)

    # ------------------------------------------------------------------
    # Pass 4: chamfer edges at z = −5.0 mm → 0.8 mm bottom bevel
    # ------------------------------------------------------------------
    try:
        body     = _find_bin_body(root_comp)
        target_z = -mm(BASE_PROFILE_HEIGHT_MM)   # −0.50 cm

        bot_edges = adsk.core.ObjectCollection.create()
        for i in range(body.edges.count):
            edge = body.edges.item(i)
            sp = edge.startVertex.geometry
            ep = edge.endVertex.geometry
            if abs(sp.z - target_z) < tol and abs(ep.z - target_z) < tol:
                bot_edges.add(edge)

        if bot_edges.count > 0:
            chamfers = root_comp.features.chamferFeatures
            ci = chamfers.createInput2()
            ci.chamferEdgeSets.addEqualDistanceChamferEdgeSet(
                bot_edges,
                adsk.core.ValueInput.createByReal(mm(BASE_BOT_CHAMFER_MM)),
                True)
            chamfers.add(ci)
    except Exception:
        pass  # non-fatal



# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_outputs(root_comp, config: dict, output_dir: str):
    """Export STL and optionally STEP files."""
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)

    # Export STL
    stl_path = os.path.join(output_dir, "gridfinity_bin.stl")
    export_mgr = design.exportManager
    stl_options = export_mgr.createSTLExportOptions(root_comp, stl_path)
    stl_options.meshRefinement = (
        adsk.fusion.MeshRefinementSettings.MeshRefinementMedium)
    export_mgr.execute(stl_options)

    # Export STEP
    step_path = os.path.join(output_dir, "gridfinity_bin.step")
    step_options = export_mgr.createSTEPExportOptions(step_path, root_comp)
    export_mgr.execute(step_options)

    return stl_path, step_path


def capture_screenshot(output_dir: str, width: int = 1920, height: int = 1080):
    """Capture a viewport screenshot of the finished bin.

    Sets an isometric-style camera looking down at the bin from above-front,
    fits the model in view, and saves as PNG.
    """
    app = adsk.core.Application.get()
    viewport = app.activeViewport
    camera = viewport.camera
    camera.isSmoothTransition = False

    # Set visual style to shaded with visible edges
    viewport.visualStyle = adsk.fusion.VisualStyles.ShadedWithVisibleEdgesOnlyVisualStyle

    # Fit model to fill the viewport
    viewport.fit()

    # Save screenshot
    preview_path = os.path.join(output_dir, "gridfinity_bin_preview.png")
    viewport.saveAsImageFile(preview_path, width, height)

    return preview_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_rounded_rect(sketch, x0, y0, x1, y1, r):
    """Draw a rounded rectangle (rectangle with filleted corners) in a sketch.

    Args:
        sketch: Fusion 360 sketch object
        x0, y0: bottom-left corner (cm)
        x1, y1: top-right corner (cm)
        r: corner fillet radius (cm)
    """
    lines = sketch.sketchCurves.sketchLines
    arcs = sketch.sketchCurves.sketchArcs

    p_bl = adsk.core.Point3D.create(x0, y0, 0)
    p_br = adsk.core.Point3D.create(x1, y0, 0)
    p_tr = adsk.core.Point3D.create(x1, y1, 0)
    p_tl = adsk.core.Point3D.create(x0, y1, 0)

    l_bot = lines.addByTwoPoints(p_bl, p_br)
    l_right = lines.addByTwoPoints(p_br, p_tr)
    l_top = lines.addByTwoPoints(p_tr, p_tl)
    l_left = lines.addByTwoPoints(p_tl, p_bl)

    # Fillet each corner to create rounded rectangle
    arcs.addFillet(l_bot, p_br, l_right, p_br, r)
    arcs.addFillet(l_right, p_tr, l_top, p_tr, r)
    arcs.addFillet(l_top, p_tl, l_left, p_tl, r)
    arcs.addFillet(l_left, p_bl, l_bot, p_bl, r)


def _find_smallest_profile(sketch):
    """Find the smallest profile in a sketch (by area).

    When drawing a polygon on a face, Fusion creates two profiles: the
    polygon interior and the surrounding face region. We want the smaller one.
    """
    smallest = None
    smallest_area = float("inf")
    for i in range(sketch.profiles.count):
        prof = sketch.profiles.item(i)
        area = prof.areaProperties().area
        if area < smallest_area:
            smallest_area = area
            smallest = prof
    return smallest


# ---------------------------------------------------------------------------
# Main entry point (Fusion 360 script)
# ---------------------------------------------------------------------------

def run(context):
    """Fusion 360 script entry point."""
    app = adsk.core.Application.get()
    ui = app.userInterface

    try:
        # Try default path first (bin_config.json in generated/ at project root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(script_dir)
        project_dir = os.path.dirname(src_dir)
        default_path = os.path.join(project_dir, "generated", "bin_config.json")

        if os.path.exists(default_path):
            config_path = default_path
        else:
            # File dialog fallback
            dialog = ui.createFileDialog()
            dialog.title = "Select Gridfinity Bin Config (bin_config.json)"
            dialog.filter = "JSON files (*.json);;All files (*.*)"
            dialog.initialDirectory = project_dir
            dialog.isMultiSelectEnabled = False

            result = dialog.showOpen()
            if result != adsk.core.DialogResults.DialogOK:
                return

            config_path = dialog.filename

        ui.messageBox(f"Loading config:\n{config_path}", "Gridfinity Bin Generator")
        config = read_config(config_path)

        # Create new document
        doc = app.documents.add(
            adsk.core.DocumentTypes.FusionDesignDocumentType)
        design = adsk.fusion.Design.cast(app.activeProduct)
        design.designType = adsk.fusion.DesignTypes.ParametricDesignType
        root_comp = design.rootComponent

        ui.messageBox(
            f"Creating {config['grid_x']}x{config['grid_y']} gridfinity bin\n"
            f"Height: {config['height_units']} units\n"
            f"Tools: {len(config['tools'])}",
            "Gridfinity Bin Generator")

        # Build the bin
        create_bin_body(root_comp, config)
        create_stacking_lip(root_comp, config)
        lower_deck(root_comp, config)
        cut_tool_pockets(root_comp, config)
        cut_slots(root_comp, config)
        create_base_interface(root_comp, config)

        # Export
        output_dir = os.path.dirname(config_path)
        stl_path, step_path = export_outputs(root_comp, config, output_dir)

        # Capture viewport screenshot
        try:
            preview_path = capture_screenshot(output_dir)
        except Exception:
            preview_path = "(screenshot failed)"

        ui.messageBox(
            f"Bin created successfully!\n\n"
            f"STL: {stl_path}\n"
            f"STEP: {step_path}\n"
            f"Preview: {preview_path}",
            "Gridfinity Bin Generator")

    except Exception:
        ui.messageBox(
            f"Error:\n{traceback.format_exc()}",
            "Gridfinity Bin Generator — Error")
