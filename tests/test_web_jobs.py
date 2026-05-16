"""Tests for the web JobManager.

Mocks ``run_pipeline`` so the tests don't need SAM2, OpenCV, or a GPU.
Verifies: job creation writes files into the per-UUID dir, the worker pool
calls run_pipeline with our params, progress events flow through to
subscribers, and TTL sweep deletes terminal jobs.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from pic_to_bin.pipeline import ProgressEvent
from pic_to_bin.web.jobs import JobManager, JobStatus


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02):
    """Spin-wait so the worker thread has time to update job status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("predicate never became true")


@pytest.fixture
def mgr(tmp_path):
    m = JobManager(jobs_root=tmp_path / "jobs", gpu_concurrency=1, max_workers=2)
    yield m
    m.shutdown()


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace run_pipeline with a stub that records calls and emits events."""
    calls = []

    def fake(image_paths, *, output_dir, progress_cb=None, **kwargs):
        calls.append({
            "image_paths": list(image_paths),
            "output_dir": Path(output_dir),
            "stop_after": kwargs.get("stop_after", "all"),
            "skip_trace": kwargs.get("skip_trace", False),
            "kwargs": kwargs,
        })
        if progress_cb is not None:
            progress_cb(ProgressEvent(step="preprocess", message="fake preprocess"))
            progress_cb(ProgressEvent(step="layout", message="fake layout", fraction=1.0))

        # Stub the artifact files so artifact endpoints would resolve.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "layout_preview.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (Path(output_dir) / "combined_layout.dxf").write_text("0\nSECTION\n")
        result = {
            "dxf_paths": ["fake.dxf"],
            "combined_dxf": str(Path(output_dir) / "combined_layout.dxf"),
            "layout_preview": str(Path(output_dir) / "layout_preview.png"),
            "layout_result": {"grid_units_x": 3, "grid_units_y": 4},
            "grid_units_x": 3,
            "grid_units_y": 4,
            "bin_config": None,
        }
        if kwargs.get("stop_after") == "all":
            (Path(output_dir) / "bin_config.json").write_text("{}")
            result["bin_config"] = str(Path(output_dir) / "bin_config.json")
        return result

    monkeypatch.setattr("pic_to_bin.web.jobs.run_pipeline", fake)
    return calls


def test_create_job_materializes_inputs(mgr):
    job = mgr.create_job(
        params={"tool_heights": {0: 17.0}, "phone_height": 482.0},
        input_files=[("photo.png", b"PNGDATA")],
    )
    assert job.id
    assert (mgr.jobs_root / job.id / "inputs" / "photo.png").read_bytes() == b"PNGDATA"
    assert job.status == JobStatus.PENDING
    assert mgr.get(job.id) is job


def test_create_job_strips_path_components(mgr):
    job = mgr.create_job(
        params={"tool_heights": 17.0},
        input_files=[("../../etc/passwd", b"x")],
    )
    # Filename should be sanitized to its basename — no traversal.
    saved = list((mgr.jobs_root / job.id / "inputs").iterdir())
    assert len(saved) == 1
    assert saved[0].name == "passwd"


def test_phase_a_runs_pipeline_and_advances_status(mgr, fake_pipeline):
    job = mgr.create_job(
        params={"tool_heights": 17.0, "tolerance": 0.5},
        input_files=[("a.png", b"X"), ("b.png", b"Y")],
    )
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION)

    assert len(fake_pipeline) == 1
    call = fake_pipeline[0]
    assert call["stop_after"] == "layout"
    assert call["skip_trace"] is False
    assert call["kwargs"]["tolerance"] == 0.5
    assert {p.name for p in call["image_paths"]} == {"a.png", "b.png"}
    assert job.layout_result is not None
    assert job.layout_result["grid_units_x"] == 3


def test_phase_a_writes_per_job_log_file(mgr, monkeypatch):
    """Pipeline ``print()`` output during Phase A is captured into
    ``<job_dir>/job.log`` instead of falling out to the server console."""
    sentinel = "PIPELINE-LOG-SENTINEL-12345"

    def fake(image_paths, *, output_dir, progress_cb=None, **kwargs):
        # The fake stands in for run_pipeline; whatever it prints should
        # land in the per-job log because JobManager's log capture wraps
        # the call.
        print(sentinel)
        if progress_cb is not None:
            progress_cb(ProgressEvent(step="layout_ready", fraction=1.0))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {
            "dxf_paths": [], "combined_dxf": None, "layout_preview": None,
            "layout_result": {"grid_units_x": 1, "grid_units_y": 1},
            "grid_units_x": 1, "grid_units_y": 1, "bin_config": None,
        }

    monkeypatch.setattr("pic_to_bin.web.jobs.run_pipeline", fake)
    job = mgr.create_job(
        params={"tool_heights": 17.0},
        input_files=[("x.png", b"X")],
    )
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION)

    log_path = job.output_dir / "job.log"
    assert log_path.exists(), "JobManager should write per-job log file"
    contents = log_path.read_text(encoding="utf-8")
    assert sentinel in contents, (
        f"pipeline stdout not captured into job.log; got: {contents!r}"
    )


def test_phase_b_produces_bin_config(mgr, fake_pipeline):
    job = mgr.create_job(
        params={"tool_heights": 17.0},
        input_files=[("a.png", b"X")],
    )
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION)
    mgr.submit_phase_b(job)
    _wait_until(lambda: job.status == JobStatus.COMPLETE)

    assert job.final_result is not None
    assert job.final_result["bin_config"]
    assert (mgr.jobs_root / job.id / "bin_config.json").exists()
    # Phase B must have re-used cached traces.
    assert fake_pipeline[1]["skip_trace"] is True
    assert fake_pipeline[1]["stop_after"] == "all"


def test_phase_b_rejects_wrong_status(mgr, fake_pipeline):
    job = mgr.create_job(params={"tool_heights": 17.0},
                         input_files=[("a.png", b"X")])
    # Status is PENDING — proceed should not be allowed.
    with pytest.raises(RuntimeError, match="cannot proceed"):
        mgr.submit_phase_b(job)


def test_redo_layout_only_uses_skip_trace(mgr, fake_pipeline):
    job = mgr.create_job(params={"tool_heights": 17.0, "gap": 3.0},
                         input_files=[("a.png", b"X")])
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION)

    mgr.submit_redo(job, new_params={"gap": 5.0}, layout_only=True)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION and len(fake_pipeline) == 2)

    assert fake_pipeline[1]["skip_trace"] is True
    assert fake_pipeline[1]["kwargs"]["gap"] == 5.0
    assert job.params["gap"] == 5.0


def test_redo_with_sam_corrective_points_forwards_to_pipeline(mgr, fake_pipeline):
    """A redo carrying sam_corrective_points must forward them to
    run_pipeline so segment_tool can prompt SAM2 with the clicks."""
    job = mgr.create_job(params={"tool_heights": 17.0},
                         input_files=[("a.png", b"X")])
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION)

    points = {"a": [{"x_mm": 62.0, "y_mm": 35.0, "label": 0}]}
    # The server endpoint force-overrides layout_only=False when
    # corrective points are present; the JobManager itself doesn't, so
    # we pass layout_only=False directly here to mirror what the
    # server's redo handler would.
    mgr.submit_redo(job, new_params={"sam_corrective_points": points},
                    layout_only=False)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION and len(fake_pipeline) == 2)

    assert fake_pipeline[1]["skip_trace"] is False
    assert fake_pipeline[1]["kwargs"]["sam_corrective_points"] == points
    assert job.params["sam_corrective_points"] == points


def test_run_llm_evaluate_returns_overlay_stems(mgr, fake_pipeline, monkeypatch, tmp_path):
    """End-to-end through run_llm_evaluate: stub the Anthropic call and
    overlay generator, verify the returned ``overlay_stems`` list lines
    up with the input-image stems for which an overlay was written."""

    # Bring the manager into the LLM-enabled state.
    mgr.anthropic_api_key = "sk-fake"

    job = mgr.create_job(
        params={"tool_heights": 17.0},
        input_files=[("img_a.png", b"X"), ("img_b.png", b"Y")],
    )
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION)

    # The fake pipeline doesn't create per-tool subdirs or rectified images.
    # Materialize them so _rectified_paths_for() finds them, plus a stub
    # trace DXF (overlay generator skips when missing). Real PNG bytes —
    # ``cap_image_size_to_jpeg`` re-saves the overlay through PIL, which
    # rejects header-only stubs.
    from PIL import Image as _Image
    for stem in ("img_a", "img_b"):
        sub = job.output_dir / stem
        sub.mkdir(parents=True, exist_ok=True)
        _Image.new("RGB", (10, 10), color="white").save(
            sub / f"{stem}_rectified.png"
        )
        (sub / f"{stem}_rectified_trace.dxf").write_text("0\nSECTION\n")

    # Mock the LLM evaluator and overlay generator inside the import paths
    # used by run_llm_evaluate.
    from pic_to_bin.web import llm_check as _llm_check
    from pic_to_bin.web import overlay as _overlay

    def fake_evaluate_layout(rectified_paths, layout_preview_path,
                              current_params, api_key, **kwargs):
        return _llm_check.LLMVerdict(
            ok=False,
            reasoning="Tip looks short.",
            suggested_params={"axial_tolerance": 1.5},
            model="stub",
        )

    def fake_generate_overlay_image(rectified_path, trace_dxf_path,
                                      output_path, dpi=None):
        # Real PNG so cap_image_size_to_jpeg can read it.
        _Image.new("RGB", (10, 10), color="red").save(output_path)
        return Path(output_path)

    monkeypatch.setattr(_llm_check, "evaluate_layout", fake_evaluate_layout)
    monkeypatch.setattr(_overlay, "generate_overlay_image",
                        fake_generate_overlay_image)

    verdict, iterations, overlay_stems = mgr.run_llm_evaluate(
        job, auto_loop=False, max_iterations=1
    )

    assert verdict.ok is False
    assert iterations == 1
    assert sorted(overlay_stems) == ["img_a", "img_b"]
    # The capped JPEG (≤1 MB; what gets sent to the LLM) is what we expose
    # through the /overlays/{stem} endpoint.
    assert (job.output_dir / "img_a" / "img_a_rectified_overlay.jpg").exists()
    assert (job.output_dir / "img_b" / "img_b_rectified_overlay.jpg").exists()


def test_run_llm_evaluate_overlay_failure_falls_back(mgr, fake_pipeline,
                                                      monkeypatch):
    """When the overlay generator raises for one tool, the LLM eval still
    completes and the failed stem is omitted from overlay_stems."""
    mgr.anthropic_api_key = "sk-fake"
    job = mgr.create_job(
        params={"tool_heights": 17.0},
        input_files=[("good.png", b"X"), ("bad.png", b"Y")],
    )
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION)

    from PIL import Image as _Image
    for stem in ("good", "bad"):
        sub = job.output_dir / stem
        sub.mkdir(parents=True, exist_ok=True)
        _Image.new("RGB", (10, 10), color="white").save(
            sub / f"{stem}_rectified.png"
        )
        (sub / f"{stem}_rectified_trace.dxf").write_text("0\nSECTION\n")

    from pic_to_bin.web import llm_check as _llm_check
    from pic_to_bin.web import overlay as _overlay

    def fake_evaluate_layout(**kwargs):
        return _llm_check.LLMVerdict(ok=True, reasoning="fine",
                                      suggested_params={}, model="stub")

    def flaky_overlay(rectified_path, trace_dxf_path, output_path, dpi=None):
        if "bad" in str(rectified_path):
            raise RuntimeError("simulated overlay failure")
        _Image.new("RGB", (10, 10), color="green").save(output_path)
        return Path(output_path)

    monkeypatch.setattr(_llm_check, "evaluate_layout", fake_evaluate_layout)
    monkeypatch.setattr(_overlay, "generate_overlay_image", flaky_overlay)

    verdict, iterations, overlay_stems = mgr.run_llm_evaluate(
        job, auto_loop=False, max_iterations=1
    )
    assert verdict.ok is True
    assert overlay_stems == ["good"], (
        f"expected only successfully-overlaid 'good' stem, got {overlay_stems}"
    )
    # SSE event log should record the overlay failure for "bad".
    assert any(
        ev["step"] == "llm_overlay_failed" and "bad" in ev["message"]
        for ev in job.event_log
    )


def test_pipeline_error_marks_job_error(mgr, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr("pic_to_bin.web.jobs.run_pipeline", boom)

    job = mgr.create_job(params={"tool_heights": 17.0},
                         input_files=[("a.png", b"X")])
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.ERROR)
    assert "simulated failure" in (job.error or "")
    assert any(ev["step"] == "error" for ev in job.event_log)


def test_subscribe_replays_then_streams(mgr, fake_pipeline):
    """A subscriber that joins after Phase A finished should still see all events."""
    async def go():
        loop = asyncio.get_running_loop()
        mgr.bind_loop(loop)
        job = mgr.create_job(params={"tool_heights": 17.0},
                             input_files=[("a.png", b"X")])
        mgr.submit_phase_a(job)
        # Wait for Phase A to finish (status advances + log populated).
        for _ in range(250):
            if job.status == JobStatus.AWAITING_DECISION:
                break
            await asyncio.sleep(0.02)
        # Now subscribe — the existing log should replay.
        seen = []
        async for ev in mgr.subscribe(job):
            seen.append(ev["step"])
            if ev["step"] == "layout_ready":
                break
        return seen

    seen = asyncio.run(go())
    assert "preprocess" in seen
    assert "layout_ready" in seen


def test_signal_subscribers_shutdown_unblocks_subscribers(mgr, fake_pipeline):
    """Active subscribers should exit cleanly when shutdown is signaled."""
    async def go():
        loop = asyncio.get_running_loop()
        mgr.bind_loop(loop)
        job = mgr.create_job(
            params={"tool_heights": 17.0},
            input_files=[("a.png", b"X")],
        )
        # Don't submit phase A — keep the job idle so the subscriber blocks
        # on the empty queue, exactly like the real Ctrl-C scenario where
        # users are sitting on the preview screen.
        seen = []

        async def consume():
            async for ev in mgr.subscribe(job):
                seen.append(ev)

        consumer = asyncio.create_task(consume())
        # Let the consumer start awaiting on the queue.
        await asyncio.sleep(0.05)
        assert not consumer.done(), "subscriber should be blocked, not done"

        mgr.signal_subscribers_shutdown()
        # The consumer should exit on its own once it sees the sentinel.
        await asyncio.wait_for(consumer, timeout=1.0)
        assert seen == [], "sentinel should not be yielded as a regular event"

    asyncio.run(go())


def test_tool_heights_string_keys_are_coerced(mgr, fake_pipeline):
    """Frontend sends JSON object keys as strings; pipeline needs ints."""
    job = mgr.create_job(
        params={"tool_heights": {"0": 17.0, "1": 14.0}},
        input_files=[("a.png", b"X"), ("b.png", b"Y")],
    )
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION)
    th = fake_pipeline[0]["kwargs"]["tool_heights"]
    assert th == {0: 17.0, 1: 14.0}


def test_tool_heights_uniform_float(mgr, fake_pipeline):
    job = mgr.create_job(
        params={"tool_heights": 17.5},
        input_files=[("a.png", b"X")],
    )
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION)
    assert fake_pipeline[0]["kwargs"]["tool_heights"] == 17.5


def test_sweep_removes_old_terminal_jobs(mgr, fake_pipeline):
    job = mgr.create_job(params={"tool_heights": 17.0},
                         input_files=[("a.png", b"X")])
    mgr.submit_phase_a(job)
    _wait_until(lambda: job.status == JobStatus.AWAITING_DECISION)
    # AWAITING_DECISION is not terminal — sweep should leave it alone even if old.
    job.last_activity = 0.0
    assert mgr.sweep_expired() == 0
    assert (mgr.jobs_root / job.id).exists()

    # Mark complete + ancient → swept.
    job.status = JobStatus.COMPLETE
    job.last_activity = 0.0
    assert mgr.sweep_expired() == 1
    assert not (mgr.jobs_root / job.id).exists()
    assert mgr.get(job.id) is None


# ---------------------------------------------------------------------------
# LLM corrective-points helpers
# ---------------------------------------------------------------------------


def test_rectified_dimensions_mm_reads_sidecar(tmp_path):
    from pic_to_bin.web.jobs import _rectified_dimensions_mm
    import json as _json

    rect = tmp_path / "tool_rectified.png"
    rect.write_bytes(b"")  # contents irrelevant; helper reads sidecar
    (tmp_path / "tool_rectified.json").write_text(_json.dumps({
        "effective_dpi": 200.0,
        "image_width_px": 1000,
        "image_height_px": 2000,
    }), encoding="utf-8")
    w_mm, h_mm = _rectified_dimensions_mm(rect)
    # 1000 px / 200 dpi * 25.4 = 127 mm; 2000 px → 254 mm
    assert abs(w_mm - 127.0) < 0.01
    assert abs(h_mm - 254.0) < 0.01


def test_rectified_dimensions_mm_missing_sidecar(tmp_path):
    from pic_to_bin.web.jobs import _rectified_dimensions_mm
    rect = tmp_path / "tool_rectified.png"
    rect.write_bytes(b"")
    assert _rectified_dimensions_mm(rect) is None


def test_merge_corrective_points_groups_by_stem(tmp_path):
    from pic_to_bin.web.jobs import _merge_corrective_points

    # Two rectified paths in stem subdirs (mirrors run_pipeline output).
    rect_a = tmp_path / "imgA" / "imgA_rectified.png"
    rect_b = tmp_path / "imgB" / "imgB_rectified.png"
    rect_a.parent.mkdir()
    rect_b.parent.mkdir()
    rectified = [rect_a, rect_b]

    new_points = [
        {"overlay_index": 1, "x_mm": 12.0, "y_mm": 34.0, "label": 0,
         "reason": "white gap"},
        {"overlay_index": 2, "x_mm": 56.0, "y_mm": 78.0, "label": 1},
        # Out-of-range index — must be silently dropped.
        {"overlay_index": 99, "x_mm": 1.0, "y_mm": 2.0, "label": 0},
    ]
    out = _merge_corrective_points(None, new_points, rectified)
    assert set(out.keys()) == {"imgA", "imgB"}
    assert out["imgA"] == [{"x_mm": 12.0, "y_mm": 34.0, "label": 0}]
    assert out["imgB"] == [{"x_mm": 56.0, "y_mm": 78.0, "label": 1}]


def test_merge_corrective_points_accumulates_across_iterations(tmp_path):
    from pic_to_bin.web.jobs import _merge_corrective_points

    rect_a = tmp_path / "imgA" / "imgA_rectified.png"
    rect_a.parent.mkdir()
    rectified = [rect_a]

    iter1 = [{"overlay_index": 1, "x_mm": 10.0, "y_mm": 20.0, "label": 0}]
    after1 = _merge_corrective_points(None, iter1, rectified)

    iter2 = [{"overlay_index": 1, "x_mm": 30.0, "y_mm": 40.0, "label": 0}]
    after2 = _merge_corrective_points(after1, iter2, rectified)

    # Iteration 2's clicks accumulate onto iteration 1's — the earlier
    # click stays active so SAM2 doesn't regress on what it fixed.
    assert after2["imgA"] == [
        {"x_mm": 10.0, "y_mm": 20.0, "label": 0},
        {"x_mm": 30.0, "y_mm": 40.0, "label": 0},
    ]


def test_pipeline_kwargs_forwards_sam_corrective_points():
    """Auto-loop merges points into job.params['sam_corrective_points'];
    the kwargs filter must pass them through to run_pipeline."""
    from pic_to_bin.web.jobs import _pipeline_kwargs

    params = {
        "tool_heights": 17.0,
        "sam_corrective_points": {
            "imgA": [{"x_mm": 10.0, "y_mm": 20.0, "label": 0}],
        },
        # Unknown key — must be filtered out.
        "ignored_field": "drop me",
    }
    out = _pipeline_kwargs(params)
    assert out["sam_corrective_points"] == {
        "imgA": [{"x_mm": 10.0, "y_mm": 20.0, "label": 0}],
    }
    assert "ignored_field" not in out


# ---------------------------------------------------------------------------
# New security / DoS hardening tests (added 2026-05-15)
# ---------------------------------------------------------------------------

def test_create_job_rejects_oversized_file(mgr):
    """JobManager (library entry point) rejects files > MAX_IMAGE_BYTES."""
    from pic_to_bin.web.jobs import MAX_IMAGE_BYTES
    big = b"X" * (MAX_IMAGE_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds per-image limit"):
        mgr.create_job(
            params={"tool_heights": 17.0},
            input_files=[("huge.jpg", big)],
        )


def test_create_job_rejects_too_many_images(mgr):
    """JobManager rejects > MAX_IMAGES_PER_JOB submissions."""
    from pic_to_bin.web.jobs import MAX_IMAGES_PER_JOB
    files = [(f"{i}.jpg", b"ok") for i in range(MAX_IMAGES_PER_JOB + 1)]
    with pytest.raises(ValueError, match="too many images"):
        mgr.create_job(params={"tool_heights": 17.0}, input_files=files)


def test_llm_disabled_by_default(mgr):
    """When enable_llm=False (the public default), llm_available stays False
    even if a key is accidentally present in the environment."""
    # The fixture mgr was created without enable_llm=True.
    assert mgr.llm_available is False
    # Even if we had passed a key, the flag controls it.
