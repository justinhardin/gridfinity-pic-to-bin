"""Shared bin-building logic used by both the Fusion script and add-in.

Reads bin_config.json (from prepare_bin.py) and creates a parametric
gridfinity bin with tool pocket cutouts in a Fusion 360 document.

Coordinates in the JSON are in mm; converted to cm here (Fusion's native unit).
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

LIP_HEIGHT_MM = 4.4
LIP_TOP_RECESS_MM = 0.6
LIP_CUTOUT_CORNER_R_MM = 4.5

BASE_PROFILE_HEIGHT_MM = 5.0
BASE_PAD_CONE_H_MM = 2.4
BASE_PAD_STRAIGHT_H_MM = 2.6
BASE_CONE_CHAMFER_MM = 2.4
BASE_BOT_CHAMFER_MM = 0.8
BASE_CLEARANCE_MM = 0.25
BASE_CORNER_RADIUS_MM = 3.75
SLOT_FLOOR_CLEARANCE_MM = 1.0


def mm(v: float) -> float:
    """Convert millimeters to centimeters (Fusion 360 internal unit)."""
    return v / 10.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def read_config(path: str) -> dict:
    with open(path, "r") as f:
        config = json.load(f)
    required = ["grid_x", "grid_y", "height_units", "bin_width_mm",
                "bin_height_mm", "tools"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")
    return config


# ---------------------------------------------------------------------------
# Sketch helpers
# ---------------------------------------------------------------------------

def _draw_rounded_rect(sketch, x0, y0, x1, y1, r):
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

    arcs.addFillet(l_bot, p_br, l_right, p_br, r)
    arcs.addFillet(l_right, p_tr, l_top, p_tr, r)
    arcs.addFillet(l_top, p_tl, l_left, p_tl, r)
    arcs.addFillet(l_left, p_bl, l_bot, p_bl, r)


def draw_polygon_sketch(sketch, poly_cm):
    lines = sketch.sketchCurves.sketchLines
    n = len(poly_cm)
    for i in range(n):
        x1, y1 = poly_cm[i]
        x2, y2 = poly_cm[(i + 1) % n]
        p1 = adsk.core.Point3D.create(x1, y1, 0)
        p2 = adsk.core.Point3D.create(x2, y2, 0)
        lines.addByTwoPoints(p1, p2)


def _all_profiles(sketch):
    """Collect every profile in a sketch into an ObjectCollection.

    Use for sketches drawn on construction planes that have only the closed
    shapes we drew — there is no ambient "face remainder" profile to filter.
    """
    coll = adsk.core.ObjectCollection.create()
    for i in range(sketch.profiles.count):
        coll.add(sketch.profiles.item(i))
    return coll


def _find_smallest_profile(sketch):
    """Return the smallest-area profile in a sketch.

    For sketches drawn on a body face, Fusion creates an extra "face
    remainder" profile around the drawn shape; this picks the inner one.
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


def _find_bin_body(root_comp):
    for i in range(root_comp.bRepBodies.count):
        b = root_comp.bRepBodies.item(i)
        if b.name.startswith("bin_body_"):
            return b
    return root_comp.bRepBodies.item(0)


# Library + appearance name candidates for white ABS. Fusion version drift
# moves these around; we try each in order and stop on the first hit.
_ABS_WHITE_LIBRARIES = (
    "Fusion Appearance Library",
    "Fusion 360 Appearance Library",
)
_ABS_WHITE_NAMES = (
    "ABS (White)",
    "ABS - White",
    "ABS Plastic - White",
    "Plastic - ABS - White",
    "ABS White",
)


def _apply_abs_white(body) -> bool:
    """Set body appearance to the Plastic > ABS (White) library appearance.

    Returns True on success, False if the appearance can't be found. Never
    raises — appearance is cosmetic and must not fail the build.
    """
    try:
        app = adsk.core.Application.get()
        design = adsk.fusion.Design.cast(app.activeProduct)
        libs = app.materialLibraries
        for lib_name in _ABS_WHITE_LIBRARIES:
            lib = libs.itemByName(lib_name)
            if not lib:
                continue
            for ap_name in _ABS_WHITE_NAMES:
                lib_ap = lib.appearances.itemByName(ap_name)
                if not lib_ap:
                    continue
                # Copy into the design so the assignment persists with the doc.
                local = design.appearances.itemByName(ap_name)
                if not local:
                    local = design.appearances.addByCopy(lib_ap, ap_name)
                body.appearance = local
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Bin body
# ---------------------------------------------------------------------------

def _bin_body_height_mm(config):
    """Body extrusion height (mm) above the bin floor (z=0).

    Per the gridfinity spec, U×7 mm is the bin's labelled height
    (excluding the lip) and includes the base region. The base
    profile pads sit below z=0 (z = −BASE_PROFILE_HEIGHT_MM up to
    z = 0), so the body proper extrudes from z=0 to U×7 − base_h —
    landing the lip's bottom at the standard rim height and the
    pads' bottom at z = −5 mm. Older builds extruded the body to
    U×7 above z=0, which made bins 5 mm too tall.
    """
    return max(
        1.0,
        config["height_units"] * 7.0
        - config.get("base_profile_height_mm", 5.0),
    )


def _effective_bin_top_z_cm(config):
    """Z height (cm) of the bin body's top face, accounting for the
    deck-lowering shortcut applied when stacking lips are disabled.

    With stacking lip ON, the body is built to the full gridfinity unit
    height; the deck cut later removes a shallow recess inside the walls
    so the lip can sit above and provide finger access around the tool.

    With stacking lip OFF, the deck IS the bin top — there's no reason
    to leave empty walls extending above the recess. We bake that into
    the body height so the resulting print has a flat top at the deck
    level rather than a wall sticking up around the pocket.
    """
    bin_d = mm(_bin_body_height_mm(config))
    if not config.get("stacking_lip", True):
        lowering = mm(config.get("deck_lowering_mm", 0) or 0)
        if lowering > 0:
            return max(bin_d - lowering, mm(1.0))
    return bin_d


def create_bin_body(root_comp, config):
    bin_w = mm(config["bin_width_mm"])
    bin_h = mm(config["bin_height_mm"])
    bin_d = _effective_bin_top_z_cm(config)

    sketch = root_comp.sketches.add(root_comp.xYConstructionPlane)
    lines = sketch.sketchCurves.sketchLines
    pts = [adsk.core.Point3D.create(x, y, 0)
           for x, y in [(0, 0), (bin_w, 0), (bin_w, bin_h), (0, bin_h)]]
    for i in range(4):
        lines.addByTwoPoints(pts[i], pts[(i + 1) % 4])

    extrudes = root_comp.features.extrudeFeatures
    ext_input = extrudes.createInput(
        sketch.profiles.item(0),
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ext_input.setDistanceExtent(
        False, adsk.core.ValueInput.createByReal(bin_d))
    extrude = extrudes.add(ext_input)

    body = extrude.bodies.item(0)
    body.name = (f"bin_body_{config['grid_x']}x{config['grid_y']}"
                 f"x{config['height_units']}")
    _apply_abs_white(body)
    return body


def fillet_outer_corners(root_comp, config):
    """Round the four vertical outer corners of the bin body.

    Standard gridfinity bins have 4 mm corner rounding on the outside.
    Used to be inside ``create_stacking_lip``, factored out so it runs
    even when the stacking lip is disabled — the bin's outer corners
    should still be rounded either way.
    """
    bin_w = mm(config["bin_width_mm"])
    bin_h = mm(config["bin_height_mm"])
    fillet_r = mm(4.0)
    tol = 0.005

    body = _find_bin_body(root_comp)
    if body is None:
        return
    vert_edges = adsk.core.ObjectCollection.create()
    for i in range(body.edges.count):
        edge = body.edges.item(i)
        sp = edge.startVertex.geometry
        ep = edge.endVertex.geometry
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
            True)
        root_comp.features.filletFeatures.add(fi)


# ---------------------------------------------------------------------------
# Stacking lip
# ---------------------------------------------------------------------------

def create_stacking_lip(root_comp, config):
    bin_w = mm(config["bin_width_mm"])
    bin_h = mm(config["bin_height_mm"])
    bin_d = mm(_bin_body_height_mm(config))
    lip_h = mm(LIP_HEIGHT_MM)
    lip_top_z = bin_d + lip_h

    xy_plane = root_comp.xYConstructionPlane
    planes = root_comp.constructionPlanes
    extrudes = root_comp.features.extrudeFeatures
    tol = 0.005

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

    fillet_outer_corners(root_comp, config)

    cl = mm(BASE_CLEARANCE_MM)
    cone_h = mm(BASE_PAD_CONE_H_MM)
    cutout_r = mm(LIP_CUTOUT_CORNER_R_MM)

    cutout_z = bin_d + mm(BASE_PROFILE_HEIGHT_MM)
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

    s_x0 = top_x0 + cone_h
    s_y0 = top_y0 + cone_h
    s_x1 = top_x1 - cone_h
    s_y1 = top_y1 - cone_h
    s_r = max(cutout_r - cone_h, mm(0.1))
    straight_h = mm(BASE_PROFILE_HEIGHT_MM) - cone_h

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

    combines = root_comp.features.combineFeatures
    join_coll = adsk.core.ObjectCollection.create()
    join_coll.add(straight_body)
    join_input = combines.createInput(cutout_body, join_coll)
    join_input.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
    combines.add(join_input)

    body = _find_bin_body(root_comp)
    cut_coll = adsk.core.ObjectCollection.create()
    cut_coll.add(cutout_body)
    cut_input = combines.createInput(body, cut_coll)
    cut_input.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
    combines.add(cut_input)

    recess_h = mm(LIP_TOP_RECESS_MM)
    lip_top_plane_input = planes.createInput()
    lip_top_plane_input.setByOffset(
        xy_plane, adsk.core.ValueInput.createByReal(lip_top_z))
    lip_top_plane = planes.add(lip_top_plane_input)

    recess_sketch = root_comp.sketches.add(lip_top_plane)
    margin = mm(1.0)
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
# Deck
# ---------------------------------------------------------------------------

def lower_deck(root_comp, config):
    lowering = config.get("deck_lowering_mm", 0)
    if not lowering or lowering <= 0:
        return
    # When stacking lips are disabled the bin body has already been
    # built to the deck level by ``create_bin_body`` — there's no extra
    # material to recess. Skipping here also means the script never
    # cuts a sketch outside the body's top face, which would leave a
    # dangling sketch with no profile to extrude through.
    if not config.get("stacking_lip", True):
        return

    bin_w = mm(config["bin_width_mm"])
    bin_h = mm(config["bin_height_mm"])
    bin_d = mm(_bin_body_height_mm(config))
    wall_t = mm(config.get("wall_thickness_mm", 1.6))
    deck_inset = mm(config.get("deck_inset_mm", 2.0))
    inset = wall_t + deck_inset

    xy_plane = root_comp.xYConstructionPlane
    planes = root_comp.constructionPlanes
    plane_input = planes.createInput()
    plane_input.setByOffset(
        xy_plane, adsk.core.ValueInput.createByReal(bin_d))
    deck_plane = planes.add(plane_input)

    sketch = root_comp.sketches.add(deck_plane)
    lines = sketch.sketchCurves.sketchLines
    pts = [adsk.core.Point3D.create(x, y, 0) for x, y in [
        (inset, inset), (bin_w - inset, inset),
        (bin_w - inset, bin_h - inset), (inset, bin_h - inset)]]
    for i in range(4):
        lines.addByTwoPoints(pts[i], pts[(i + 1) % 4])

    if sketch.profiles.count == 0:
        return

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
# Tool pockets and slots
# ---------------------------------------------------------------------------

def _top_cut_plane(root_comp, config):
    """Construction plane just above the stacking lip (or bin top if no lip).

    Pocket and slot cuts start here and go negative-Z so edge-reaching
    profiles cut through the lip too. With stacking lip OFF, the body
    has already been shortened to the deck level (see
    ``_effective_bin_top_z_cm``) so this plane sits on the body's
    actual top face rather than in empty space above where the wall
    used to be.
    """
    bin_top = _effective_bin_top_z_cm(config)
    lip_h = mm(LIP_HEIGHT_MM) if config.get("stacking_lip", True) else 0.0
    planes = root_comp.constructionPlanes
    plane_input = planes.createInput()
    plane_input.setByOffset(
        root_comp.xYConstructionPlane,
        adsk.core.ValueInput.createByReal(bin_top + lip_h))
    return planes.add(plane_input), lip_h


def _effective_pocket_depth_mm(tool, config):
    """Pocket cut depth (mm) from the body's top face down to the pocket
    floor, accounting for the deck-lowering shortcut applied when
    stacking lips are disabled.

    The config's ``pocket_depth_mm`` was computed assuming the bin runs
    to the full gridfinity unit height (so a pocket cut from there to
    1 mm above the bin floor). When we instead build the body to deck
    level (stacking lip OFF), the cut starts that much lower and needs
    a correspondingly shorter depth — otherwise the cut punches through
    the bin floor.
    """
    requested = float(tool.get("pocket_depth_mm", 0.0))
    if not config.get("stacking_lip", True):
        lowering = float(config.get("deck_lowering_mm", 0) or 0)
        requested = max(requested - lowering, 0.5)
    return requested


def cut_tool_pockets(root_comp, config):
    """Cut each tool's pocket as a single sketch + single cut per tool."""
    top_plane, lip_h = _top_cut_plane(root_comp, config)
    extrudes = root_comp.features.extrudeFeatures

    for tool in config["tools"]:
        polys = tool.get("tolerance_polys_mm", []) \
                or tool.get("inner_polys_mm", [])
        if not polys:
            continue
        depth_cm = mm(_effective_pocket_depth_mm(tool, config)) + lip_h

        sketch = root_comp.sketches.add(top_plane)
        sketch.name = f"pocket_{tool.get('name', 'tool')}"
        for poly_mm in polys:
            poly_cm = [[mm(pt[0]), mm(pt[1])] for pt in poly_mm]
            draw_polygon_sketch(sketch, poly_cm)

        profiles = _all_profiles(sketch)
        if profiles.count == 0:
            continue

        ext_input = extrudes.createInput(
            profiles, adsk.fusion.FeatureOperations.CutFeatureOperation)
        ext_input.setOneSideExtent(
            adsk.fusion.DistanceExtentDefinition.create(
                adsk.core.ValueInput.createByReal(depth_cm)),
            adsk.fusion.ExtentDirections.NegativeExtentDirection)
        extrudes.add(ext_input)


def cut_slots(root_comp, config):
    """Cut each tool's finger slot(s) as a single sketch + single cut per tool."""
    top_plane, lip_h = _top_cut_plane(root_comp, config)
    extrudes = root_comp.features.extrudeFeatures

    for tool in config["tools"]:
        slot_polys = tool.get("slot_polys_mm", [])
        if not slot_polys:
            continue
        slot_depth = mm(_effective_pocket_depth_mm(tool, config)) + lip_h

        sketch = root_comp.sketches.add(top_plane)
        sketch.name = f"slot_{tool.get('name', 'tool')}"
        for poly_mm in slot_polys:
            poly_cm = [[mm(pt[0]), mm(pt[1])] for pt in poly_mm]
            draw_polygon_sketch(sketch, poly_cm)

        profiles = _all_profiles(sketch)
        if profiles.count == 0:
            continue

        try:
            ext_input = extrudes.createInput(
                profiles, adsk.fusion.FeatureOperations.CutFeatureOperation)
            ext_input.setOneSideExtent(
                adsk.fusion.DistanceExtentDefinition.create(
                    adsk.core.ValueInput.createByReal(slot_depth)),
                adsk.fusion.ExtentDirections.NegativeExtentDirection)
            extrudes.add(ext_input)
        except Exception:
            pass  # slot may overlap an already-cut pocket — non-fatal


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

def create_base_interface(root_comp, config):
    """Create the gridfinity base mating geometry below the bin floor.

    Pad cross-section (downward from z=0):
        z =  0.0 mm — pad shoulder, full 41.5 mm
        z = -2.4 mm — cone end, 36.7 mm (45° equal-distance chamfer)
        z = -4.2 mm — straight post end, still 36.7 mm
        z = -5.0 mm — pad bottom, 35.1 mm (after 0.8 mm bottom chamfer)

    All N pads share a single sketch + single Join extrude per pass.
    """
    grid_x = config["grid_x"]
    grid_y = config["grid_y"]
    unit_mm = 42.0

    pad_mm = unit_mm - 2 * BASE_CLEARANCE_MM
    corner_r = BASE_CORNER_RADIUS_MM
    narrow_mm = pad_mm - 2 * BASE_CONE_CHAMFER_MM
    narrow_r = max(0.1, corner_r - BASE_CONE_CHAMFER_MM)

    xy_plane = root_comp.xYConstructionPlane
    extrudes = root_comp.features.extrudeFeatures
    tol = 0.005

    # Pass 1 — wide pads, single sketch + single extrude
    wide_sketch = root_comp.sketches.add(xy_plane)
    wide_sketch.name = "base_pad_wide"
    for gx in range(grid_x):
        for gy in range(grid_y):
            x0 = mm(gx * unit_mm + BASE_CLEARANCE_MM)
            y0 = mm(gy * unit_mm + BASE_CLEARANCE_MM)
            x1 = mm((gx + 1) * unit_mm - BASE_CLEARANCE_MM)
            y1 = mm((gy + 1) * unit_mm - BASE_CLEARANCE_MM)
            _draw_rounded_rect(wide_sketch, x0, y0, x1, y1, mm(corner_r))

    wide_profiles = _all_profiles(wide_sketch)
    if wide_profiles.count > 0:
        ext = extrudes.createInput(
            wide_profiles,
            adsk.fusion.FeatureOperations.JoinFeatureOperation)
        ext.setOneSideExtent(
            adsk.fusion.DistanceExtentDefinition.create(
                adsk.core.ValueInput.createByReal(mm(BASE_PAD_CONE_H_MM))),
            adsk.fusion.ExtentDirections.NegativeExtentDirection)
        extrudes.add(ext)

    # Pass 2 — chamfer all edges at z = -cone_h
    try:
        body = _find_bin_body(root_comp)
        target_z = -mm(BASE_PAD_CONE_H_MM)
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
                True)
            chamfers.add(ci)
    except Exception:
        pass

    # Pass 3 — narrow posts on a plane at z=-cone_h, single sketch + single extrude
    planes = root_comp.constructionPlanes
    plane_input = planes.createInput()
    plane_input.setByOffset(
        xy_plane,
        adsk.core.ValueInput.createByReal(-mm(BASE_PAD_CONE_H_MM)))
    mid_plane = planes.add(plane_input)

    narrow_sketch = root_comp.sketches.add(mid_plane)
    narrow_sketch.name = "base_pad_narrow"
    for gx in range(grid_x):
        for gy in range(grid_y):
            cx = (gx + 0.5) * unit_mm
            cy = (gy + 0.5) * unit_mm
            x0 = mm(cx - narrow_mm / 2)
            y0 = mm(cy - narrow_mm / 2)
            x1 = mm(cx + narrow_mm / 2)
            y1 = mm(cy + narrow_mm / 2)
            _draw_rounded_rect(narrow_sketch, x0, y0, x1, y1, mm(narrow_r))

    narrow_profiles = _all_profiles(narrow_sketch)
    if narrow_profiles.count > 0:
        ext = extrudes.createInput(
            narrow_profiles,
            adsk.fusion.FeatureOperations.JoinFeatureOperation)
        ext.setOneSideExtent(
            adsk.fusion.DistanceExtentDefinition.create(
                adsk.core.ValueInput.createByReal(mm(BASE_PAD_STRAIGHT_H_MM))),
            adsk.fusion.ExtentDirections.NegativeExtentDirection)
        extrudes.add(ext)

    # Pass 4 — bottom chamfer
    try:
        body = _find_bin_body(root_comp)
        target_z = -mm(BASE_PROFILE_HEIGHT_MM)
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
        pass


# ---------------------------------------------------------------------------
# Export and screenshot
# ---------------------------------------------------------------------------

def export_outputs(root_comp, config, output_dir):
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)

    stl_path = os.path.join(output_dir, "gridfinity_bin.stl")
    export_mgr = design.exportManager
    stl_options = export_mgr.createSTLExportOptions(root_comp, stl_path)
    stl_options.meshRefinement = (
        adsk.fusion.MeshRefinementSettings.MeshRefinementMedium)
    export_mgr.execute(stl_options)

    step_path = os.path.join(output_dir, "gridfinity_bin.step")
    step_options = export_mgr.createSTEPExportOptions(step_path, root_comp)
    export_mgr.execute(step_options)

    return stl_path, step_path


def capture_screenshot(output_dir, width=1920, height=1080):
    app = adsk.core.Application.get()
    viewport = app.activeViewport

    camera = viewport.camera
    camera.isSmoothTransition = False
    viewport.camera = camera

    viewport.visualStyle = adsk.core.VisualStyles.ShadedWithVisibleEdgesOnlyVisualStyle
    viewport.fit()
    adsk.doEvents()

    preview_path = os.path.join(output_dir, "gridfinity_bin_preview.png")
    if not viewport.saveAsImageFile(preview_path, width, height):
        raise RuntimeError(
            f"viewport.saveAsImageFile returned False for {preview_path}")
    return preview_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _group_phase(timeline, name, fn, *args, **kwargs):
    """Run a build phase and wrap its timeline entries in a named group."""
    start = timeline.count
    fn(*args, **kwargs)
    end = timeline.count - 1
    if end >= start:
        try:
            grp = timeline.timelineGroups.add(start, end)
            grp.name = name
        except Exception:
            # Some phases may produce no groupable items in odd configs.
            pass


def build_bin(config_path: str, ui=None) -> dict:
    """Build the bin in a new document from the given config JSON.

    Returns a dict with stl_path, step_path, preview_path (preview_path may be
    a "(screenshot failed)..." string if capture failed but the model exported).
    """
    app = adsk.core.Application.get()

    config = read_config(config_path)

    doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    design = adsk.fusion.Design.cast(app.activeProduct)
    design.designType = adsk.fusion.DesignTypes.ParametricDesignType
    root_comp = design.rootComponent
    timeline = design.timeline

    _group_phase(timeline, "Bin Body", create_bin_body, root_comp, config)
    if config.get("stacking_lip", True):
        _group_phase(timeline, "Stacking Lip",
                     create_stacking_lip, root_comp, config)
    else:
        # Stacking lip itself filets the outer corners; when it's
        # disabled we still want them rounded (gridfinity convention),
        # so call the fillet phase directly.
        _group_phase(timeline, "Outer Corners",
                     fillet_outer_corners, root_comp, config)
    _group_phase(timeline, "Deck", lower_deck, root_comp, config)
    _group_phase(timeline, "Tool Pockets",
                 cut_tool_pockets, root_comp, config)
    _group_phase(timeline, "Finger Slots", cut_slots, root_comp, config)
    _group_phase(timeline, "Base Pads",
                 create_base_interface, root_comp, config)

    output_dir = os.path.dirname(config_path)
    stl_path, step_path = export_outputs(root_comp, config, output_dir)

    try:
        preview_path = capture_screenshot(output_dir)
    except Exception:
        preview_path = f"(screenshot failed)\n{traceback.format_exc()}"

    return {
        "config": config,
        "stl_path": stl_path,
        "step_path": step_path,
        "preview_path": preview_path,
    }
