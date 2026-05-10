"""
Generate a printable PDF template with ArUco markers for the phone-camera
workflow.  The template is placed on a flat surface, the tool is placed on
top, and a phone photo is taken.  The markers provide perspective correction
and automatic scale calibration.

Template layout:
    8 ArUco markers from DICT_4X4_50 (IDs 0-7)
    4 corner markers + 4 edge midpoint markers for redundancy
    20mm marker size, ~20mm margins from paper edge
    Supports A4, US Letter, and US Legal paper sizes

CLI:
    generate-phone-template [--paper-size a4|letter|legal] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 20.0

# Paper dimensions in mm
PAPER_SIZES = {
    "a4":     (210.0, 297.0),
    "a5":     (148.0, 210.0),
    "letter": (215.9, 279.4),
    "legal":  (215.9, 355.6),
}

# Margin from paper edge to marker outer boundary (mm)
MARGIN_MM = 20.0

# White "quiet zone" padding placed behind each marker on non-white
# backgrounds. ArUco detection wants a clear light border around the marker;
# without this the chroma-key-green background bleeds up to the marker's
# black perimeter and detection becomes unreliable.
MARKER_QUIET_PAD_MM = 4.0


# ---------------------------------------------------------------------------
# Marker positions
# ---------------------------------------------------------------------------

def _compute_marker_positions(paper_w: float, paper_h: float,
                              ) -> list[tuple[int, float, float]]:
    """Compute marker center positions for a given paper size.

    Places 4 corner markers (IDs 0-3) and 4 edge midpoint markers (IDs 4-7).
    All coordinates are in mm from the paper top-left corner.

    Args:
        paper_w: Paper width in mm
        paper_h: Paper height in mm

    Returns:
        List of (marker_id, center_x_mm, center_y_mm)
    """
    half = MARKER_SIZE_MM / 2.0
    # Marker center is MARGIN + half-marker from the paper edge
    left   = MARGIN_MM + half
    right  = paper_w - MARGIN_MM - half
    top    = MARGIN_MM + half
    bottom = paper_h - MARGIN_MM - half
    cx     = (left + right) / 2.0
    cy     = (top + bottom) / 2.0

    return [
        (0, left,  top),        # top-left
        (1, right, top),        # top-right
        (2, right, bottom),     # bottom-right
        (3, left,  bottom),     # bottom-left
        (4, cx,    top),        # top-mid
        (5, right, cy),         # right-mid
        (6, cx,    bottom),     # bottom-mid
        (7, left,  cy),         # left-mid
    ]


def get_marker_positions(paper_size: str = "legal",
                         ) -> list[tuple[int, float, float]]:
    """Return marker (id, x_mm, y_mm) positions for the given paper size.

    Args:
        paper_size: One of "a4", "letter", "legal"

    Returns:
        List of (marker_id, center_x_mm, center_y_mm)

    Raises:
        ValueError: If paper_size is not recognized
    """
    key = paper_size.lower()
    if key not in PAPER_SIZES:
        raise ValueError(
            f"Unknown paper size '{paper_size}'. "
            f"Choose from: {', '.join(PAPER_SIZES)}")
    pw, ph = PAPER_SIZES[key]
    return _compute_marker_positions(pw, ph)


def get_placement_zone(paper_size: str = "legal",
                       ) -> tuple[float, float, float, float]:
    """Return the tool placement zone (x0, y0, x1, y1) in mm.

    The zone is the rectangle inside the markers where tools can be placed.
    It is inset by MARGIN + MARKER_SIZE from each paper edge.

    Returns:
        (x0, y0, x1, y1) in mm from paper top-left
    """
    key = paper_size.lower()
    if key not in PAPER_SIZES:
        raise ValueError(f"Unknown paper size '{paper_size}'")
    pw, ph = PAPER_SIZES[key]
    inset = MARGIN_MM + MARKER_SIZE_MM
    return (inset, inset, pw - inset, ph - inset)


# ---------------------------------------------------------------------------
# ArUco marker image generation
# ---------------------------------------------------------------------------

def _generate_marker_image(marker_id: int, size_px: int = 200) -> np.ndarray:
    """Generate an ArUco marker image.

    Args:
        marker_id: Marker ID (0-7)
        size_px: Output image size in pixels

    Returns:
        Grayscale marker image (uint8, size_px x size_px)
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    return marker_img


# ---------------------------------------------------------------------------
# Template PDF generation
# ---------------------------------------------------------------------------

def _darken_color(color: str, factor: float) -> str:
    """Multiply each RGB channel of a hex color by factor in [0, 1].

    Used to produce a "near-bg" shade for the placement-zone border so
    it's faintly visible to the user but not bright enough to confuse
    SAM2 segmentation.
    """
    s = color.lstrip("#")
    if len(s) != 6:
        return color
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02X}{g:02X}{b:02X}"


def _is_light_color(color: str) -> bool:
    """True if the hex color is light enough that white text on it is unreadable.

    Used to flip text/zone-border colors on dark backgrounds so they stay
    legible.
    """
    s = color.lstrip("#")
    if len(s) != 6:
        return True
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    # Rec. 601 luma
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return luma > 160


def generate_template(output_path: str, paper_size: str = "legal",
                      bg_color: str = "#FFFFFF") -> str:
    """Generate a printable PDF template with ArUco markers.

    Args:
        output_path: Path to save the PDF
        paper_size: One of "a4", "a5", "letter", "legal"
        bg_color: Page background as hex (e.g. "#FFFFFF" or "#00B140").
            Non-white backgrounds get a small white quiet-zone pad behind
            each marker so ArUco detection stays robust.

    Returns:
        Path to saved PDF
    """
    key = paper_size.lower()
    pw_mm, ph_mm = PAPER_SIZES[key]
    markers = get_marker_positions(key)
    zone = get_placement_zone(key)

    # Convert mm to inches for matplotlib
    pw_in = pw_mm / 25.4
    ph_in = ph_mm / 25.4

    fig, ax = plt.subplots(1, 1, figsize=(pw_in, ph_in))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, pw_mm)
    ax.set_ylim(ph_mm, 0)  # y-axis down (matches image coords)
    ax.set_aspect("equal")
    ax.axis("off")

    # Page background
    ax.add_patch(plt.Rectangle((0, 0), pw_mm, ph_mm,
                                facecolor=bg_color, edgecolor="none"))

    light_bg = _is_light_color(bg_color)
    # On a colored background, draw a white quiet-zone pad behind each
    # marker — ArUco needs a clear light border around the marker square.
    needs_quiet_pad = not light_bg or bg_color.lower() not in ("#ffffff", "#fff", "white")
    pad_half = MARKER_SIZE_MM / 2.0 + MARKER_QUIET_PAD_MM

    # Color choices that adapt to the background. Anything inside the
    # placement zone on a non-white background must NOT be white — SAM2
    # picks up white-on-green text/borders as foreground. We darken the
    # zone border to a near-bg shade and skip per-marker labels entirely
    # for non-white templates.
    title_color = "#999999" if light_bg else "#FFFFFF"
    subtitle_color = "#BBBBBB" if light_bg else "#FFFFFF"
    if light_bg:
        zone_color = "#CCCCCC"
        label_color = "#999999"
        draw_marker_labels = True
    else:
        # Slightly darkened version of the bg = invisible-ish to the user
        # but invisible to SAM2 (no bright contrast against the bg).
        zone_color = _darken_color(bg_color, 0.7)
        label_color = None
        draw_marker_labels = False

    # Place each marker (with optional white quiet-zone pad behind it)
    half = MARKER_SIZE_MM / 2.0
    marker_px = 200  # resolution of generated marker image
    for mid, cx, cy in markers:
        if needs_quiet_pad:
            ax.add_patch(plt.Rectangle(
                (cx - pad_half, cy - pad_half),
                2 * pad_half, 2 * pad_half,
                facecolor="white", edgecolor="none", zorder=1.5))
        img = _generate_marker_image(mid, marker_px)
        # extent: [left, right, bottom, top] in data coords
        ax.imshow(img, cmap="gray", interpolation="nearest",
                  extent=[cx - half, cx + half, cy + half, cy - half],
                  zorder=2)
        if draw_marker_labels:
            # Small gray "ID N" label below each marker. Only on white
            # backgrounds — on coloured backgrounds the label colour
            # would have to be white (for legibility) and would then
            # confuse SAM2 since the labels for the top markers fall
            # inside the placement zone.
            ax.text(cx, cy + half + 3, f"ID {mid}",
                    ha="center", va="top", fontsize=5, color=label_color)

    # Placement zone dotted border
    zx0, zy0, zx1, zy1 = zone
    ax.add_patch(plt.Rectangle(
        (zx0, zy0), zx1 - zx0, zy1 - zy0,
        linewidth=0.5, edgecolor=zone_color, facecolor="none",
        linestyle="dashed", zorder=1))

    # Title text
    ax.text(pw_mm / 2, MARGIN_MM / 2, "Gridfinity Pic-to-Bin Template",
            ha="center", va="center", fontsize=8, color=title_color,
            fontweight="bold")

    # Instructions
    ax.text(pw_mm / 2, ph_mm - MARGIN_MM / 2,
            f"Print at 100% scale (no fit-to-page)  |  "
            f"Paper: {paper_size.upper()} ({pw_mm:.0f} x {ph_mm:.0f} mm)",
            ha="center", va="center", fontsize=6, color=subtitle_color)

    # Save as PDF
    output_path = str(Path(output_path))
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)

    print(f"Template saved: {output_path}")
    print(f"  Paper: {paper_size.upper()} ({pw_mm:.0f} x {ph_mm:.0f} mm)")
    print(f"  Markers: 8 ArUco (DICT_4X4_50, IDs 0-7, {MARKER_SIZE_MM:.0f}mm)")
    print(f"  Placement zone: {zx1 - zx0:.0f} x {zy1 - zy0:.0f} mm")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a printable ArUco marker template for "
                    "phone-camera scanning")
    parser.add_argument(
        "--paper-size", choices=list(PAPER_SIZES), default="legal",
        help="Paper size (default: legal)")
    parser.add_argument(
        "--bg-color", type=str, default="#FFFFFF",
        help="Page background color in hex (default: #FFFFFF). "
             "Use #00B140 for chroma-key green.")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PDF path (default: phone_template_<size>.pdf)")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"phone_template_{args.paper_size}.pdf"

    generate_template(args.output, paper_size=args.paper_size,
                      bg_color=args.bg_color)


if __name__ == "__main__":
    main()
