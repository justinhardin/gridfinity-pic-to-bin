"""Ask Claude whether a pic-to-bin layout fits the physical tool.

The web app sends the rectified phone photo(s) and the generated layout
preview to Claude, which judges whether the planned pocket is correct
size + shape and, if not, suggests numeric parameter adjustments.

The model is forced to return its verdict via tool use, so the response
shape is always parseable. The system prompt + tool schema are marked
for prompt-cache reuse — repeat evaluations on the same job (the
auto-loop case) get most input tokens cached.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import anthropic


DEFAULT_MODEL = "claude-sonnet-4-6"

# Subset of pipeline params the LLM is allowed to suggest. Keep this small
# and well-documented — the schema doubles as the model's vocabulary.
ADJUSTABLE_PARAMS = {
    "tolerance": {
        "type": "number",
        "description": (
            "Uniform extra clearance applied to the pocket on top of a "
            "built-in 2 mm baseline, in mm. Default 0. Positive = looser "
            "fit (use when the layout looks too tight everywhere). "
            "Negative = tighter fit (use only if the user wants an "
            "interference grip). Range typically -1.0 to +2.0."
        ),
    },
    "axial_tolerance": {
        "type": "number",
        "description": (
            "Extra clearance pushed onto each end of the tool along its "
            "long (principal) axis only, in mm. Default 1.0. Increase "
            "(to e.g. 2.0) when only the tool's tips are tight while "
            "the wider sections fit fine — this is the SAM2 "
            "tip-under-detection compensation knob."
        ),
    },
    "display_smooth_sigma": {
        "type": "number",
        "description": (
            "Gaussian smoothing strength on the trace polygon, in mm. "
            "Default 2.5. Decrease (or set to 0) when the layout shows "
            "intentional sharp features being rounded off — e.g. "
            "brackets/braces with 90° corners. Increase (3-5) when "
            "reflective tools come out wavy. Range 0 to 5."
        ),
    },
    "mask_erode": {
        "type": "number",
        "description": (
            "Pre-trace mask erosion, in mm. Default 0. Increase to "
            "0.3-0.5 when the trace is visibly fatter than the tool "
            "(typical of strong shadow halos in the photo). Don't "
            "increase past 0.5 — it disproportionately shrinks tapered "
            "tips."
        ),
    },
    "bin_margin": {
        "type": "number",
        "description": (
            "Extra clearance between the tool extent and the bin "
            "boundary, in mm, applied before snap-to-grid. Default 0. "
            "Increase when the tool sits hard against the bin wall and "
            "you want the grid bumped up by one unit."
        ),
    },
    "gap": {
        "type": "number",
        "description": (
            "Minimum spacing between tools in a multi-tool layout, in "
            "mm. Default 3.0. Only relevant when more than one tool is "
            "in the same job."
        ),
    },
}


_VERDICT_TOOL = {
    "name": "evaluate_layout_fit",
    "description": (
        "Record your verdict on whether the planned pocket layout fits "
        "the physical tool shown in the photo(s). Always call this tool "
        "exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ok": {
                "type": "boolean",
                "description": (
                    "True if the dashed tolerance perimeter clears the "
                    "actual tool everywhere with reasonable (~2 mm) "
                    "uniform gap and no feature is clipped or "
                    "misshapen. False otherwise."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "One short paragraph explaining the verdict. If "
                    "ok=False, point at the specific problem region "
                    "(e.g. 'top-right corner is rounded off, but the "
                    "physical tool has a sharp 90° protrusion there'). "
                    "Shown verbatim to the user."
                ),
            },
            "suggested_params": {
                "type": "object",
                "description": (
                    "Suggested numeric adjustments. Only include "
                    "fields you want to change; leave the rest out. "
                    "Empty / omitted entirely when ok=True."
                ),
                "properties": ADJUSTABLE_PARAMS,
                "additionalProperties": False,
            },
            "corrective_points": {
                "type": "array",
                "description": (
                    "Per-overlay corrective clicks for the SAM2 "
                    "segmenter. Use ONLY when the RED inner-trace line "
                    "itself is wrong — i.e. it fills a region that's "
                    "clearly NOT part of the tool (white background "
                    "merged in, e.g. a gap between handles), or it "
                    "misses a region that clearly IS part of the tool. "
                    "Do NOT use these for clearance/tolerance issues; "
                    "use suggested_params for those. Coordinates are "
                    "in millimeters within the OVERLAY image's frame: "
                    "origin at top-left, +x right, +y down. Each "
                    "overlay's mm dimensions are stated in the user "
                    "message. Empty / omitted when the inner trace "
                    "looks correct."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "overlay_index": {
                            "type": "integer",
                            "description": (
                                "1-based index matching the 'Overlay N:' "
                                "label in the user message."
                            ),
                        },
                        "x_mm": {"type": "number"},
                        "y_mm": {"type": "number"},
                        "label": {
                            "type": "integer",
                            "enum": [0, 1],
                            "description": (
                                "1 = positive click (this point IS tool, "
                                "include it). 0 = negative click (this "
                                "point is NOT tool, exclude it)."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Brief justification, e.g. 'white gap "
                                "between the two open blades'."
                            ),
                        },
                    },
                    "required": ["overlay_index", "x_mm", "y_mm", "label"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["ok", "reasoning"],
        "additionalProperties": False,
    },
}


_SYSTEM_PROMPT = """\
You are reviewing a planned 3D-printed gridfinity pocket layout against a \
physical tool the user wants to store in it.

You will receive:

- One or more OVERLAY images. Each overlay is the actual tool photo \
(perspective-corrected, mm-scale, lightly dimmed) with the trace polygons \
drawn on top in their detected position, at the same scale as the photo. \
Three colored markings appear:
    • Red — the inner trace (SAM2 segmentation result). The TOOL REGION \
is shown as a translucent red FILL bounded by a solid red outline. ONE \
connected red-filled blob = one tool, traced as a single closed polygon. \
Concavities in the silhouette (e.g. the V-shaped gap between two open \
blades of pruning shears, the slot in a wrench head, the opening of a \
clamp) appear as UNFILLED background showing through INSIDE the bounding \
box of the trace — these are NORMAL and CORRECT: they mean the trace \
properly excluded background that intrudes into the tool's silhouette. \
A real disconnected segmentation would show as TWO SEPARATE red-filled \
blobs with a clean gap of fully unfilled space between them. Do not \
confuse a concavity (one filled blob with an indentation) with a \
disconnection (two filled blobs).
    • Orange dashed — the TOLERANCE perimeter. This is the polygon the bin \
will actually cut against. Your judgment is about THIS line.
    • Blue dotted — the finger-access slot cutout. Ignore for fit purposes.
- A separate layout preview PNG showing how the tool gets packed into the \
gridfinity bin. Use this only for context (overall packing); fit decisions \
come from the overlays.

Your job: decide whether the orange-dashed tolerance perimeter on each \
overlay will accommodate the physical tool visible underneath.

Good = the orange-dashed line clears the tool outline everywhere with a \
roughly uniform ~2 mm gap, and no feature of the tool is clipped or notably \
misshapen by the perimeter. The red-filled inner trace should also visibly \
cover the tool's silhouette (one connected blob per tool, with concavities \
correctly left unfilled); if the fill clearly covers non-tool or misses \
tool, the segmentation itself is off.

Bad = the orange-dashed line touches or cuts into the tool outline \
anywhere, visibly rounds off or clips an intentional feature (a corner, \
a notch, a tip), or has obviously asymmetric clearance (one side ~0 mm, \
the other ~5 mm). Also bad: the red-filled inner trace clearly extends \
past the tool into the background, or clearly fails to cover an obvious \
part of the tool (means the segmentation drifted). A concavity in the \
red fill that mirrors a real concavity in the tool's silhouette is NOT a \
problem.

Compare carefully — the overlay is at exact tool scale, so you can read \
the gap directly off the image. A 2 mm gap between the orange-dashed line \
and the tool's actual edge is the target. If the dashed line sits ON the \
tool's edge, the pocket will be too tight and the tool won't fit.

When you find a problem, decide first which KIND it is:

A) The RED inner-trace fill is wrong — it covers a region that's not \
tool (e.g. white background between two open blades that SAM2 merged \
in), or omits a region that obviously is tool. No numeric tolerance \
knob can fix this; the segmentation itself needs corrective input. \
\
Before flagging this, sanity-check: count the red-filled blobs. The \
overlay should have ONE filled blob per physical tool. If you see \
one filled blob with a concavity (unfilled background visible inside \
the bounding box, e.g. between open pruning-shear blades), that is \
NOT a topology error — the trace correctly went around a background \
intrusion. Only flag a problem when the filled coverage clearly \
includes obvious non-tool pixels OR omits obvious tool pixels. \
\
Emit `corrective_points` with one click per problem region: label=0 \
("not tool") for incorrectly-filled regions, label=1 ("is tool") for \
incorrectly-missed regions. Place the click in the visual center of \
the wrong region. Coordinates are in mm within the overlay's frame \
(origin top-left, +x right, +y down). The user message states each \
overlay's mm dimensions. One or two well-placed points is plenty — \
don't spam clicks along an edge. The pipeline will re-run SAM2 with \
your clicks and produce a new mask.

B) The RED inner-trace looks right, but the ORANGE tolerance perimeter \
is wrong — too tight, asymmetric, rounded off, etc. This is a numeric \
tolerance/geometry issue. Emit `suggested_params` from the input \
schema; do NOT invent new parameter names. Common fixes:
- Tip too short / tool tip extends past the dashed line at one end:
    increase axial_tolerance by 0.5–1.0.
- Whole pocket too tight uniformly (dashed line too close to tool \
everywhere):
    increase tolerance by 0.3–0.5 (max +1.0 in one step).
- Sharp corners on the tool are rounded off in the dashed line:
    decrease display_smooth_sigma to 1.0 or 0.
- Trace looks fatter than the actual tool (visible shadow halo in the \
photo):
    increase mask_erode to 0.3.

If both kinds of problem are present, emit BOTH `corrective_points` \
and `suggested_params`. Either field may be empty/omitted when not \
applicable. Always call the evaluate_layout_fit tool exactly once \
with your verdict.
"""


@dataclass(frozen=True)
class LLMVerdict:
    """Structured outcome of an LLM fit-check."""

    ok: bool
    reasoning: str
    suggested_params: dict = field(default_factory=dict)
    corrective_points: list = field(default_factory=list)
    model: str = ""

    def to_jsonable(self) -> dict:
        return {
            "ok": self.ok,
            "reasoning": self.reasoning,
            "suggested_params": dict(self.suggested_params),
            "corrective_points": list(self.corrective_points),
            "model": self.model,
        }


def _encode_image(path: Path) -> dict:
    """Return an Anthropic image content block for a PNG or JPEG file."""
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif suffix == ".png":
        media_type = "image/png"
    else:
        # Fall back to PNG; Claude's image classifier is forgiving.
        media_type = "image/png"
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(path.read_bytes()).decode("ascii"),
        },
    }


def _format_params_summary(current_params: dict) -> str:
    """Compact textual summary of the current pipeline params for the model."""
    keys = (
        "tolerance",
        "axial_tolerance",
        "display_smooth_sigma",
        "mask_erode",
        "bin_margin",
        "gap",
    )
    lines = []
    for k in keys:
        if k in current_params:
            lines.append(f"  {k}: {current_params[k]}")
    if not lines:
        return "(no overrides — all defaults in effect)"
    return "Current parameter values:\n" + "\n".join(lines)


def _parse_verdict(response, model: str) -> LLMVerdict:
    """Pull the tool-use call out of an Anthropic response and validate it."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "evaluate_layout_fit":
            args = block.input or {}
            ok = bool(args.get("ok", False))
            reasoning = str(args.get("reasoning", "")).strip()
            suggested = args.get("suggested_params") or {}
            # Drop any keys the model invented outside the schema.
            suggested = {
                k: v for k, v in suggested.items() if k in ADJUSTABLE_PARAMS
            }
            raw_points = args.get("corrective_points") or []
            corrective: list[dict] = []
            for pt in raw_points:
                if not isinstance(pt, dict):
                    continue
                try:
                    idx = int(pt["overlay_index"])
                    x = float(pt["x_mm"])
                    y = float(pt["y_mm"])
                    label = int(pt["label"])
                except (KeyError, TypeError, ValueError):
                    continue
                if label not in (0, 1):
                    continue
                entry = {
                    "overlay_index": idx,
                    "x_mm": x,
                    "y_mm": y,
                    "label": label,
                }
                if isinstance(pt.get("reason"), str):
                    entry["reason"] = pt["reason"]
                corrective.append(entry)
            return LLMVerdict(
                ok=ok,
                reasoning=reasoning,
                suggested_params=suggested,
                corrective_points=corrective,
                model=model,
            )
    raise RuntimeError(
        "Anthropic response did not contain an evaluate_layout_fit "
        "tool call (got: "
        + ", ".join(
            f"{getattr(b, 'type', '?')}" for b in response.content
        )
        + ")"
    )


def evaluate_layout(
    rectified_paths: Iterable[Path],
    layout_preview_path: Path,
    current_params: dict,
    api_key: str,
    *,
    overlay_dimensions_mm: Optional[Iterable[tuple[float, float]]] = None,
    model: str = DEFAULT_MODEL,
    client: Optional[anthropic.Anthropic] = None,
    max_tokens: int = 1024,
) -> LLMVerdict:
    """Ask Claude whether the layout fits the tool, returning a structured verdict.

    Args:
        rectified_paths: One or more rectified phone photos of the tool.
        layout_preview_path: The current `layout_preview.png` that the user
            sees on the preview screen.
        current_params: Dict of the pipeline params currently in effect.
            Only a handful of keys are surfaced to the model (see
            `_format_params_summary`); extra keys are ignored.
        api_key: Anthropic API key.
        model: Model ID. Defaults to `claude-sonnet-4-6` (vision-capable).
        client: Optional pre-built `anthropic.Anthropic` client (used by
            tests to inject a mock).
        max_tokens: Cap on response tokens. The verdict is short, so
            1024 is generous.

    Returns:
        An `LLMVerdict` with `ok`, `reasoning`, and (when `ok=False`) a
        `suggested_params` subset.
    """
    rectified_paths = [Path(p) for p in rectified_paths]
    layout_preview_path = Path(layout_preview_path)
    if not rectified_paths:
        raise ValueError("evaluate_layout: at least one rectified image required")
    if not layout_preview_path.exists():
        raise FileNotFoundError(f"layout preview not found: {layout_preview_path}")

    api = client or anthropic.Anthropic(api_key=api_key)

    # Build the user-message content blocks. Overlays first (each labelled),
    # then the bin-coordinate layout preview, then a text summary of current
    # params. The overlay images are the primary visual the model judges
    # against — they show the tool photo with the trace polygons drawn on
    # top at exact mm scale, so the model doesn't have to mentally align
    # two different coordinate systems.
    dims = list(overlay_dimensions_mm) if overlay_dimensions_mm else []
    user_content: list[dict] = []
    for i, p in enumerate(rectified_paths, start=1):
        if not p.exists():
            raise FileNotFoundError(f"rectified image not found: {p}")
        if i - 1 < len(dims):
            w_mm, h_mm = dims[i - 1]
            dim_text = (
                f" Overlay frame is {w_mm:.1f} mm wide × {h_mm:.1f} mm "
                f"tall (origin at top-left, +x right, +y down) — use "
                f"these dimensions if you need to emit corrective_points."
            )
        else:
            dim_text = ""
        user_content.append({
            "type": "text",
            "text": (
                f"Overlay {i}: physical tool (dimmed background) with "
                f"trace polygons drawn on top at mm scale.{dim_text}"
            ),
        })
        user_content.append(_encode_image(p))
    user_content.append({
        "type": "text",
        "text": (
            "Layout preview (bin-coordinate view, for context only — "
            "pocket-fit decisions come from the overlays above):"
        ),
    })
    user_content.append(_encode_image(layout_preview_path))
    user_content.append({
        "type": "text",
        "text": _format_params_summary(current_params),
    })

    # Cache the system prompt + tool schema. Photos and the per-call params
    # summary are uncached. On the auto-loop's repeat call within the
    # cache TTL, ~80% of input tokens hit the cache.
    response = api.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": _SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[{
            **_VERDICT_TOOL,
            "cache_control": {"type": "ephemeral"},
        }],
        tool_choice={"type": "tool", "name": "evaluate_layout_fit"},
        messages=[{"role": "user", "content": user_content}],
    )

    return _parse_verdict(response, model=model)
