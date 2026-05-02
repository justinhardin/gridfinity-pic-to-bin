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
