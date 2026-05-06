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
Three colored line styles appear:
    • Red solid — the inner trace (SAM2 segmentation result). Where the \
software thinks the tool's edge is.
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
misshapen by the perimeter. The red-solid inner trace should also visibly \
follow the tool's outline; if it doesn't, the segmentation itself is off.

Bad = the orange-dashed line touches or cuts into the tool outline \
anywhere, visibly rounds off or clips an intentional feature (a corner, \
a notch, a tip), or has obviously asymmetric clearance (one side ~0 mm, \
the other ~5 mm). Also bad: the red-solid inner trace is noticeably \
smaller or larger than the tool (means the segmentation drifted).

Compare carefully — the overlay is at exact tool scale, so you can read \
the gap directly off the image. A 2 mm gap between the orange-dashed line \
and the tool's actual edge is the target. If the dashed line sits ON the \
tool's edge, the pocket will be too tight and the tool won't fit.

When you find a problem, suggest the smallest sensible parameter \
adjustment that would fix it. Choose from the parameters in the tool \
input schema; do NOT invent new parameter names. Common fixes:
- Tip too short / tool tip extends past the dashed line at one end:
    increase axial_tolerance by 0.5–1.0.
- Whole pocket too tight uniformly (dashed line too close to tool everywhere):
    increase tolerance by 0.3–0.5 (max +1.0 in one step).
- Sharp corners on the tool are rounded off in the dashed line:
    decrease display_smooth_sigma to 1.0 or 0.
- Trace looks fatter than the actual tool (visible shadow halo in the \
photo):
    increase mask_erode to 0.3.

Always call the evaluate_layout_fit tool exactly once with your verdict.
"""


@dataclass(frozen=True)
class LLMVerdict:
    """Structured outcome of an LLM fit-check."""

    ok: bool
    reasoning: str
    suggested_params: dict = field(default_factory=dict)
    model: str = ""

    def to_jsonable(self) -> dict:
        return {
            "ok": self.ok,
            "reasoning": self.reasoning,
            "suggested_params": dict(self.suggested_params),
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
            return LLMVerdict(
                ok=ok,
                reasoning=reasoning,
                suggested_params=suggested,
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
    user_content: list[dict] = []
    for i, p in enumerate(rectified_paths, start=1):
        if not p.exists():
            raise FileNotFoundError(f"rectified image not found: {p}")
        user_content.append({
            "type": "text",
            "text": (
                f"Overlay {i}: physical tool (dimmed background) with "
                f"trace polygons drawn on top at mm scale."
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
