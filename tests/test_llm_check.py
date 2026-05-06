"""Tests for the LLM fit-check module.

The Anthropic client is fully mocked; no network calls are made. The
focus is on verifying that:
  - The request body shape is correct (system prompt cached, tool
    forced, images encoded inline).
  - Various tool-use response shapes parse cleanly into LLMVerdict.
  - Out-of-schema suggested keys are dropped silently.
  - Missing tool calls raise a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pic_to_bin.web import llm_check
from pic_to_bin.web.llm_check import LLMVerdict, evaluate_layout


# ---------------------------------------------------------------------------
# Fake Anthropic response shape
# ---------------------------------------------------------------------------


@dataclass
class _FakeBlock:
    type: str
    name: str = ""
    input: dict = None
    text: str = ""


class _FakeResponse:
    def __init__(self, content_blocks):
        self.content = content_blocks


def _png_path(tmp_path: Path, name: str) -> Path:
    """Write a 1x1 PNG so `_encode_image` doesn't crash on file-not-found."""
    # Smallest valid PNG: 1x1 white pixel, 8-bit RGB.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa8L\xae\x9b"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    p = tmp_path / name
    p.write_bytes(png_bytes)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_evaluate_layout_parses_ok_verdict(tmp_path):
    """Tool call with ok=True and no suggestions → LLMVerdict(ok=True)."""
    rectified = _png_path(tmp_path, "tool_rectified.png")
    preview = _png_path(tmp_path, "layout_preview.png")

    fake_response = _FakeResponse([
        _FakeBlock(
            type="tool_use",
            name="evaluate_layout_fit",
            input={
                "ok": True,
                "reasoning": "The dashed perimeter clears the tool with ~2 mm "
                             "uniform gap. Looks good.",
            },
        ),
    ])
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    verdict = evaluate_layout(
        rectified_paths=[rectified],
        layout_preview_path=preview,
        current_params={"tolerance": 0.0, "axial_tolerance": 1.0},
        api_key="test-key",
        client=fake_client,
    )
    assert isinstance(verdict, LLMVerdict)
    assert verdict.ok is True
    assert verdict.suggested_params == {}
    assert "uniform gap" in verdict.reasoning


def test_evaluate_layout_parses_needs_fix_verdict(tmp_path):
    """Tool call with ok=False and suggestions → fields land in
    `suggested_params`, in-schema only."""
    rectified = _png_path(tmp_path, "tool_rectified.png")
    preview = _png_path(tmp_path, "layout_preview.png")

    fake_response = _FakeResponse([
        _FakeBlock(
            type="tool_use",
            name="evaluate_layout_fit",
            input={
                "ok": False,
                "reasoning": "Top-right corner of the bracket is rounded off "
                             "in the dashed perimeter.",
                "suggested_params": {
                    "display_smooth_sigma": 0.5,
                    "tolerance": 0.3,
                },
            },
        ),
    ])
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    verdict = evaluate_layout(
        rectified_paths=[rectified],
        layout_preview_path=preview,
        current_params={},
        api_key="test-key",
        client=fake_client,
    )
    assert verdict.ok is False
    assert verdict.suggested_params == {
        "display_smooth_sigma": 0.5,
        "tolerance": 0.3,
    }


def test_evaluate_layout_drops_out_of_schema_suggestions(tmp_path):
    """Anything outside the published `ADJUSTABLE_PARAMS` schema is silently
    dropped — the model can't trick us into editing arbitrary fields."""
    rectified = _png_path(tmp_path, "tool_rectified.png")
    preview = _png_path(tmp_path, "layout_preview.png")

    fake_response = _FakeResponse([
        _FakeBlock(
            type="tool_use",
            name="evaluate_layout_fit",
            input={
                "ok": False,
                "reasoning": "Pocket too tight.",
                "suggested_params": {
                    "tolerance": 0.5,
                    "spectacularly_made_up_param": 42,
                    "max_units": 99,  # not in ADJUSTABLE_PARAMS
                },
            },
        ),
    ])
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    verdict = evaluate_layout(
        rectified_paths=[rectified],
        layout_preview_path=preview,
        current_params={},
        api_key="test-key",
        client=fake_client,
    )
    assert verdict.suggested_params == {"tolerance": 0.5}


def test_evaluate_layout_raises_on_missing_tool_call(tmp_path):
    """If the model refuses to call the tool (e.g. text-only response),
    we surface a clear error rather than silently returning a default."""
    rectified = _png_path(tmp_path, "tool_rectified.png")
    preview = _png_path(tmp_path, "layout_preview.png")

    fake_response = _FakeResponse([
        _FakeBlock(type="text", text="I don't know."),
    ])
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    with pytest.raises(RuntimeError, match="evaluate_layout_fit"):
        evaluate_layout(
            rectified_paths=[rectified],
            layout_preview_path=preview,
            current_params={},
            api_key="test-key",
            client=fake_client,
        )


def test_evaluate_layout_request_shape(tmp_path):
    """Inspect the create() call: system prompt cached, tool forced,
    photos encoded as image content blocks, params summary text trailing."""
    rectified = _png_path(tmp_path, "first.png")
    rectified2 = _png_path(tmp_path, "second.png")
    preview = _png_path(tmp_path, "layout_preview.png")

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _FakeResponse([
        _FakeBlock(
            type="tool_use",
            name="evaluate_layout_fit",
            input={"ok": True, "reasoning": "fine"},
        ),
    ])

    evaluate_layout(
        rectified_paths=[rectified, rectified2],
        layout_preview_path=preview,
        current_params={"tolerance": 0.0},
        api_key="abc",
        client=fake_client,
    )

    fake_client.messages.create.assert_called_once()
    call_kwargs = fake_client.messages.create.call_args.kwargs

    # System prompt is cached.
    system = call_kwargs["system"]
    assert isinstance(system, list) and len(system) == 1
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    # The prompt now describes overlay images (tool photo + trace polygons
    # drawn on top); make sure the prompt actually mentions overlays.
    assert "overlay" in system[0]["text"].lower()

    # Tool is forced.
    assert call_kwargs["tool_choice"] == {
        "type": "tool", "name": "evaluate_layout_fit"
    }
    assert len(call_kwargs["tools"]) == 1
    assert call_kwargs["tools"][0]["name"] == "evaluate_layout_fit"
    assert call_kwargs["tools"][0]["cache_control"] == {"type": "ephemeral"}

    # User content: 2 photos + 1 preview = 3 image blocks plus interleaved text.
    user_msg = call_kwargs["messages"][0]
    assert user_msg["role"] == "user"
    image_blocks = [b for b in user_msg["content"] if b.get("type") == "image"]
    assert len(image_blocks) == 3, (
        f"Expected 3 image blocks (2 photos + 1 preview), got {len(image_blocks)}"
    )
    # Each image block has a media_type and base64 data.
    for blk in image_blocks:
        assert blk["source"]["type"] == "base64"
        assert blk["source"]["media_type"] == "image/png"
        assert blk["source"]["data"]


def test_evaluate_layout_rejects_empty_input(tmp_path):
    preview = _png_path(tmp_path, "preview.png")
    with pytest.raises(ValueError):
        evaluate_layout(
            rectified_paths=[],
            layout_preview_path=preview,
            current_params={},
            api_key="x",
            client=MagicMock(),
        )


def test_evaluate_layout_rejects_missing_preview(tmp_path):
    rectified = _png_path(tmp_path, "tool.png")
    missing = tmp_path / "no_such_preview.png"
    with pytest.raises(FileNotFoundError):
        evaluate_layout(
            rectified_paths=[rectified],
            layout_preview_path=missing,
            current_params={},
            api_key="x",
            client=MagicMock(),
        )


def test_llmverdict_to_jsonable_roundtrip():
    v = LLMVerdict(
        ok=False,
        reasoning="too tight",
        suggested_params={"tolerance": 0.4},
        model="claude-sonnet-4-6",
    )
    j = v.to_jsonable()
    assert j == {
        "ok": False,
        "reasoning": "too tight",
        "suggested_params": {"tolerance": 0.4},
        "model": "claude-sonnet-4-6",
    }
    # Ensure mutating the returned dict doesn't reach the dataclass
    # (it's frozen and we copy on the way out).
    j["suggested_params"]["tolerance"] = 999
    assert v.suggested_params["tolerance"] == 0.4
