"""Render the trace polygons on top of the rectified photo.

The LLM was being asked to mentally align the rectified photo (tool in
template-zone coords) with the layout preview (tool in bin coords,
possibly rotated/mirrored). It often "approved" layouts whose
tolerance perimeter clearly didn't match the physical tool because
the alignment work was harder than it looked. This module produces
a single overlay image where the trace's inner outline and tolerance
perimeter are drawn directly on the rectified photo at the same
millimeter scale, so the discrepancy is impossible to miss.

The module is intentionally side-effect-free: no global state, no
implicit file paths. Callers (e.g. ``JobManager.run_llm_evaluate``)
pass the rectified PNG path, the trace DXF path, the effective DPI,
and an output path; the module reads, renders, and writes a PNG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import ezdxf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# Anthropic accepts up to ~5 MB per image, but smaller is faster and uses
# fewer cache-miss tokens on auto-loop iterations. 1 MB is a comfortable
# upper bound that still preserves enough detail for the model to read
# tolerance-vs-edge gaps off the overlay.
DEFAULT_LLM_IMAGE_MAX_BYTES = 1_000_000


def cap_image_size_to_jpeg(
    src: Path,
    dst: Path,
    max_bytes: int = DEFAULT_LLM_IMAGE_MAX_BYTES,
) -> Path:
    """Re-save ``src`` as a JPEG at ``dst`` whose file size is ≤ ``max_bytes``.

    Iteratively lowers JPEG quality, then linearly shrinks pixel dimensions
    when quality alone isn't enough. Returns ``dst``. The destination is
    overwritten on each retry so we never leave a half-written file behind
    on success.

    Used to cap the overlay images sent to the LLM at a predictable upload
    size. Photos round-trip through this comfortably; a 5 MB rectified PNG
    typically lands in 200–600 kB of JPEG with no visible loss for the LLM's
    coarse fit-check task.
    """
    from PIL import Image

    with Image.open(src) as im:
        im.load()
        if im.mode != "RGB":
            im = im.convert("RGB")

        scale = 1.0
        quality = 90
        while True:
            if scale < 1.0:
                w = max(64, int(im.width * scale))
                h = max(64, int(im.height * scale))
                target = im.resize((w, h), Image.LANCZOS)
            else:
                target = im
            target.save(dst, "JPEG", quality=quality, optimize=True)
            size = dst.stat().st_size
            if size <= max_bytes:
                return dst
            # Cheap before expensive: drop quality first (fast, preserves
            # pixel detail), then shrink dimensions when even q=60 isn't
            # tight enough.
            if quality > 60:
                quality -= 10
                continue
            scale *= 0.85
            quality = 85
            if scale < 0.1:
                # Floor — accept whatever we produced rather than infinite-
                # loop on a pathological input.
                return dst


def _read_dpi_metadata(rectified_path: Path) -> Optional[float]:
    """Return the effective DPI saved alongside the rectified image, or None
    if the sidecar isn't present (older jobs predating the sidecar)."""
    sidecar = rectified_path.with_suffix("").parent / (
        rectified_path.stem + ".json"
    )
    if not sidecar.exists():
        return None
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        dpi = meta.get("effective_dpi")
        return float(dpi) if dpi else None
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def _read_trace_polygons(
    dxf_path: Path,
) -> dict[str, list[list[tuple[float, float]]]]:
    """Pull inner / tolerance / slot polygons from the trace DXF.

    Layer mapping mirrors what ``trace_export.potrace_to_dxf`` writes:
    layer "0" → inner display, "TOLERANCE" → tolerance perimeter,
    "SLOT" → finger-access stadium. Coordinates are in mm in the CAD
    Y-up convention (already Y-flipped from the rectified image's
    pixel coordinates by ``_potrace_curves_to_polygons``).
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    out: dict[str, list[list[tuple[float, float]]]] = {
        "inner": [], "tolerance": [], "slot": [],
    }
    for ent in msp:
        if ent.dxftype() != "LWPOLYLINE":
            continue
        layer = ent.dxf.layer
        pts = [(float(x), float(y)) for x, y, *_ in ent.get_points("xy")]
        if layer == "0":
            out["inner"].append(pts)
        elif layer == "TOLERANCE":
            out["tolerance"].append(pts)
        elif layer == "SLOT":
            out["slot"].append(pts)
    return out


def _read_straighten_metadata(trace_dxf_path: Path) -> Optional[dict]:
    """Return the straightening transform applied by ``trace_tool``, or
    None if the sidecar is missing (older jobs predating the sidecar)."""
    sidecar = trace_dxf_path.with_name(
        trace_dxf_path.stem.removesuffix("_trace") + "_trace_straighten.json"
    )
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _mm_to_image_pixels(
    polygon_mm: list[tuple[float, float]],
    image_height_px: int,
    dpi: float,
    straighten_info: Optional[dict] = None,
) -> np.ndarray:
    """Convert a polygon from CAD-up mm to rectified-image pixel coordinates.

    The trace pipeline can apply two transforms before exporting polygon
    mm coords. Both must be undone here so the polygon lands on the
    photo, not in the abstracted real-world frame:

    1. **Straightening** — ``trace_tool.straighten_mask`` rotates the
       mask by a small angle to align the principal axis with a cardinal
       direction. Polygon mm coords reference the post-rotation canvas;
       we apply the inverse affine to project back into the original
       rectified-photo pixel grid.

    2. **Parallax compensation** — when ``phone_height`` and
       ``tool_height`` are set, the trace ``scale`` is multiplied by
       ``parallax_factor = (phone_height − z) / phone_height`` so the
       exported polygon mm coords reflect the *real-world* tool size
       (smaller than the parallax-inflated photo silhouette). To overlay
       on the photo we need photo-frame mm — i.e. we divide the polygon
       coords by ``parallax_factor`` to undo the shrink.

    Args:
        polygon_mm: Polygon point list in CAD-up mm (post-straighten frame).
        image_height_px: Height of the rectified photo in pixels (the
            *pre-straighten* shape, i.e. the photo we draw on).
        dpi: Effective DPI of the rectified image.
        straighten_info: Sidecar dict from ``<stem>_trace_straighten.json``.
            When present and ``applied=True``, applies the inverse of the
            mask rotation. May also include ``parallax_factor`` (default
            1.0 if absent).
    """
    arr = np.array(polygon_mm, dtype=np.float64)
    px_per_mm = dpi / 25.4

    # Undo parallax compensation FIRST: in mm, polygon points get scaled
    # outward from the centroid of the polygon's coordinate frame so the
    # photo-frame mm size matches the inflated photo. Equivalent to using
    # an effective px_per_mm of dpi/(25.4·parallax_factor); we apply it as
    # a coordinate scale here so the formula composes cleanly with the
    # straightening invert below.
    parallax_factor = 1.0
    if straighten_info and isinstance(straighten_info.get("parallax_factor"), (int, float)):
        pf = float(straighten_info["parallax_factor"])
        if 0.001 < pf <= 1.0:
            parallax_factor = pf
    photo_mm = arr / parallax_factor  # 1/factor inflates back to photo scale

    if straighten_info and straighten_info.get("applied"):
        h_orig, w_orig = straighten_info["original_shape"]
        h_new, w_new = straighten_info["new_shape"]
        correction = float(straighten_info["correction_deg"])

        # Convert mm → POST-straighten pixel coords (in photo scale).
        post_x = photo_mm[:, 0] * px_per_mm
        post_y = h_new - photo_mm[:, 1] * px_per_mm

        # Reconstruct the forward affine straighten_mask used and invert.
        M = cv2.getRotationMatrix2D(
            (w_orig / 2.0, h_orig / 2.0), correction, 1.0
        )
        M[0, 2] += (w_new - w_orig) / 2.0
        M[1, 2] += (h_new - h_orig) / 2.0
        M_inv = cv2.invertAffineTransform(M)

        homog = np.column_stack([post_x, post_y, np.ones_like(post_x)])
        orig_xy = homog @ M_inv.T  # pre-straighten pixel coords
        out = np.empty_like(arr)
        out[:, 0] = orig_xy[:, 0]
        out[:, 1] = orig_xy[:, 1]
        return out

    # No straightening — direct mm → pixel mapping with Y-flip.
    out = np.empty_like(arr)
    out[:, 0] = photo_mm[:, 0] * px_per_mm
    out[:, 1] = image_height_px - photo_mm[:, 1] * px_per_mm
    return out


def generate_overlay_image(
    rectified_path: Path,
    trace_dxf_path: Path,
    output_path: Path,
    dpi: Optional[float] = None,
) -> Path:
    """Render the rectified photo with trace polygons overlaid.

    The output PNG shows:
    - The rectified photo as background, dimmed slightly so the
      polygon strokes read clearly.
    - The TOLERANCE perimeter as a thick dashed orange line — this
      is the polygon the bin will actually cut against.
    - The inner trace as a thin solid red line — the SAM2 segmentation
      result, included so the LLM can see whether the segmentation
      itself matches the tool.
    - The finger-slot cutout as a thin dotted blue line.

    Args:
        rectified_path: Per-tool ``<stem>_rectified.png``.
        trace_dxf_path: Per-tool ``<stem>_rectified_trace.dxf``.
        output_path: Where to write the overlay PNG.
        dpi: Effective DPI of the rectified image. If None, read from
            the sidecar metadata next to ``rectified_path``. Raises
            ``ValueError`` if neither source provides one.

    Returns:
        ``output_path`` as a Path on success.
    """
    rectified_path = Path(rectified_path)
    trace_dxf_path = Path(trace_dxf_path)
    output_path = Path(output_path)

    if dpi is None:
        dpi = _read_dpi_metadata(rectified_path)
    if dpi is None:
        raise ValueError(
            f"No effective_dpi metadata next to {rectified_path}; "
            f"can't compute mm↔pixel scale for the overlay. Re-run the "
            f"phone preprocessing step or pass dpi= explicitly."
        )

    img = mpimg.imread(str(rectified_path))
    h_px, w_px = img.shape[:2]
    polygons = _read_trace_polygons(trace_dxf_path)
    straighten_info = _read_straighten_metadata(trace_dxf_path)

    # Compose the figure at the rectified image's native resolution. The
    # output PNG matches the source's pixel dimensions so the LLM (and
    # any human) sees the polygons drawn at exactly tool-scale.
    fig_w_in = max(2.0, w_px / 100.0)
    fig_h_in = max(2.0, h_px / 100.0)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)  # image-coord origin: top-left
    ax.set_aspect("equal")
    ax.axis("off")

    ax.imshow(img, alpha=0.55, extent=(0, w_px, h_px, 0))

    def _plot(
        group: str, color: str, ls: str, lw: float, label: str,
        fill_alpha: float = 0.0,
    ):
        for poly in polygons[group]:
            if len(poly) < 2:
                continue
            pts = _mm_to_image_pixels(poly, h_px, dpi, straighten_info)
            xs = np.append(pts[:, 0], pts[0, 0])
            ys = np.append(pts[:, 1], pts[0, 1])
            if fill_alpha > 0:
                ax.fill(xs, ys, color=color, alpha=fill_alpha,
                        label=label)
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw)
            else:
                ax.plot(xs, ys, color=color, linestyle=ls,
                        linewidth=lw, label=label)
            label = "_nolegend_"  # only first poly per group gets a legend entry

    # Fill the inner trace as well as outlining it. Without the fill, the LLM
    # has misread silhouette concavities (e.g. the gap between open pruning-
    # shear blades) as topological discontinuities — the unfilled background
    # showing through a concavity looked to it like two separate polygons.
    # A translucent red fill makes the tool region unambiguous: one connected
    # filled blob = one tool; multiple separate blobs = a real merge/split.
    _plot("inner",     "#e63946", "-",  2.0, "Inner trace (SAM2)",
          fill_alpha=0.22)
    _plot("tolerance", "#ffa600", "--", 2.4, "Tolerance perimeter (cut)")
    _plot("slot",      "#1d70b8", ":",  1.5, "Finger slot")

    # Legend in a translucent box; placed top-left so it doesn't overlap
    # the centered tool in most photos.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left",
                  framealpha=0.7, fontsize=8)

    fig.savefig(str(output_path), dpi=100, bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)
    return output_path
