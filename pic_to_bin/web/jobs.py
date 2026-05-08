"""JobManager: per-request UUID directories, thread pool, GPU semaphore, SSE event fan-out.

Each browser submission becomes a JobState. Phase A (preprocess + trace +
layout) runs on submit. The user reviews the layout preview and then either
"proceeds" (Phase B: prepare_bin → bin_config.json) or "re-does" with adjusted
parameters. The GPU semaphore serializes SAM2 inference across concurrent jobs;
without it the second concurrent submission would OOM the GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from pic_to_bin.pipeline import ProgressEvent, run_pipeline


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_DECISION = "awaiting_decision"  # layout preview ready; user must proceed/redo
    FINALIZING = "finalizing"                # Phase B in flight
    COMPLETE = "complete"
    ERROR = "error"


# Sentinel pushed to every subscriber queue during app shutdown so the SSE
# generator exits its await loop and EventSourceResponse can close its stream
# before uvicorn cancels the task. Without it, uvicorn logs "ASGI callable
# returned without completing response" for every active connection on Ctrl-C.
_SHUTDOWN_SENTINEL = object()


# ---------------------------------------------------------------------------
# Per-job stdout/stderr capture
# ---------------------------------------------------------------------------
#
# The pipeline emits a lot of useful diagnostics via ``print()`` (preprocess
# step, marker counts, parallax factor, refine iterations, etc.). On the CLI
# this lands on the terminal; in the web app it would land on the server's
# console where end users can't see it. We want each job's prints to land in
# ``<job_dir>/job.log`` instead so the user can inspect what their pipeline
# saw on a per-job basis.
#
# ``contextlib.redirect_stdout`` is global — fine for a single-threaded
# pipeline run, but the JobManager runs jobs in a ThreadPoolExecutor and
# multiple jobs (Phase A and Phase B for different jobs) can be in flight
# simultaneously. The thread-local router below wraps the original
# ``sys.stdout`` / ``sys.stderr`` and dispatches writes to a per-thread
# file handle when one is set; threads with no override (uvicorn workers,
# the main thread) keep writing to the original stream.

class _ThreadLocalStream:
    """Tee-like wrapper: dispatches writes to a per-thread file when set,
    falls through to the wrapped underlying stream otherwise.

    Installed once over ``sys.stdout`` / ``sys.stderr`` at JobManager init,
    leaving the rest of the process unchanged. Multiple JobManagers in the
    same process is unsupported (and not a real use case)."""

    def __init__(self, fallback):
        self._fallback = fallback
        self._tls = threading.local()

    def push_target(self, f):
        """Set this thread's redirect target, returning the previous one
        so callers can restore it. Callers must use the matching restore."""
        prev = getattr(self._tls, "target", None)
        self._tls.target = f
        return prev

    def restore_target(self, prev):
        self._tls.target = prev

    def _current(self):
        return getattr(self._tls, "target", None) or self._fallback

    def write(self, s):
        return self._current().write(s)

    def flush(self):
        return self._current().flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        # Delegate everything else (encoding, buffer, fileno on best-effort,
        # …) to whichever stream is current. Tools like tqdm probe these.
        return getattr(self._current(), name)


# Single shared instances so we install ``sys.stdout``/``sys.stderr``
# wrappers exactly once even if multiple JobManagers are created (only the
# first wins). The hot path stays identity-fast for non-job threads.
_stdout_router: Optional[_ThreadLocalStream] = None
_stderr_router: Optional[_ThreadLocalStream] = None


def _install_thread_local_streams() -> None:
    """Install the router over ``sys.stdout``/``sys.stderr``.

    Idempotent in normal use; re-attaches if something else replaced
    ``sys.stdout`` since the last install (e.g. pytest swaps it per test,
    so the wrapper we set in the first test is no longer effective in
    later tests). The router instance is reused; only its fallback target
    is refreshed."""
    global _stdout_router, _stderr_router
    if _stdout_router is None:
        _stdout_router = _ThreadLocalStream(sys.stdout)
    elif sys.stdout is not _stdout_router:
        _stdout_router._fallback = sys.stdout
    sys.stdout = _stdout_router

    if _stderr_router is None:
        _stderr_router = _ThreadLocalStream(sys.stderr)
    elif sys.stderr is not _stderr_router:
        _stderr_router._fallback = sys.stderr
    sys.stderr = _stderr_router


@contextlib.contextmanager
def _job_log_capture(job: "JobState"):
    """Append-redirect ``stdout``/``stderr`` from the current thread to
    ``<job.output_dir>/job.log`` for the duration of the ``with`` block.

    The log file is opened line-buffered so partial pipeline output is
    flushed promptly — handy when debugging a stuck job through the file
    while it's still running. A header timestamp marks each capture window
    so multiple Phase A / Phase B / redo runs on the same job stay
    distinguishable."""
    # Re-attach in case sys.stdout/stderr was swapped after JobManager init
    # (e.g. pytest replaces them per test phase, so the install we did during
    # the fixture setup is no longer effective in the call phase).
    _install_thread_local_streams()

    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.output_dir / "job.log"
    with open(log_path, "a", encoding="utf-8", buffering=1) as f:
        f.write(
            f"\n=== {datetime.datetime.now().isoformat(timespec='seconds')} "
            f"job={job.id} ===\n"
        )
        prev_out = _stdout_router.push_target(f) if _stdout_router else None
        prev_err = _stderr_router.push_target(f) if _stderr_router else None
        try:
            yield f
        finally:
            # Restore the previous target rather than blanket-clearing,
            # so nested captures (e.g. run_llm_evaluate wraps the outer
            # context and calls _run_phase_a which wraps its own) don't
            # accidentally drop output back to the server console when
            # the inner capture exits.
            if _stdout_router is not None:
                _stdout_router.restore_target(prev_out)
            if _stderr_router is not None:
                _stderr_router.restore_target(prev_err)


@dataclass
class JobState:
    id: str
    output_dir: Path
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    params: dict[str, Any] = field(default_factory=dict)
    input_image_paths: list[Path] = field(default_factory=list)
    event_log: list[dict] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    layout_result: Optional[dict] = None
    final_result: Optional[dict] = None
    error: Optional[str] = None
    # Optional user-provided label, sanitized to filesystem-safe chars.
    # When set, drives the download filename for each artifact.
    part_name: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def to_summary(self) -> dict:
        """User-facing snapshot. Excludes internal lock + subscriber queues."""
        return {
            "id": self.id,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "error": self.error,
            "grid_units_x": (self.layout_result or {}).get("grid_units_x"),
            "grid_units_y": (self.layout_result or {}).get("grid_units_y"),
            "artifacts": self._artifact_urls(),
        }

    def _artifact_urls(self) -> dict[str, str]:
        urls: dict[str, str] = {}
        if self.layout_result:
            urls["layout_preview"] = f"/jobs/{self.id}/artifacts/layout_preview.png"
            urls["fit_test_pdf"] = f"/jobs/{self.id}/artifacts/layout_actual_size.pdf"
            urls["fit_test_svg"] = f"/jobs/{self.id}/artifacts/layout_actual_size.svg"
            urls["combined_dxf"] = f"/jobs/{self.id}/artifacts/combined_layout.dxf"
        if self.final_result and self.final_result.get("bin_config"):
            urls["bin_config"] = f"/jobs/{self.id}/artifacts/bin_config.json"
        return urls


class JobManager:
    """Holds the in-memory job registry, dispatches pipeline runs, fans out events.

    A single JobManager is created at app startup and shared across requests.
    The asyncio loop reference is captured lazily on first event delivery.
    """

    def __init__(
        self,
        jobs_root: Path,
        gpu_concurrency: int = 1,
        max_workers: int = 4,
        ttl_hours: float = 24.0,
        anthropic_api_key: Optional[str] = None,
    ):
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobState] = {}
        self._jobs_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="pic-to-bin-job"
        )
        self._gpu_sem = threading.Semaphore(gpu_concurrency)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.ttl_hours = ttl_hours
        self.anthropic_api_key = anthropic_api_key
        # Wrap sys.stdout/stderr so worker-thread prints during a pipeline
        # run land in <job_dir>/job.log instead of the server console.
        # No-op for non-worker threads (uvicorn handlers, main).
        _install_thread_local_streams()
        # Rebuild the in-memory registry from on-disk state so jobs survive
        # a server restart. Dirs without job_state.json (created before this
        # persistence was added) are skipped — without persisted params we
        # can't safely redo them.
        restored = self._restore_jobs()
        if restored:
            print(f"[jobs] restored {restored} job(s) from {self.jobs_root}")

    @property
    def llm_available(self) -> bool:
        return bool(self.anthropic_api_key)

    # -- loop binding --------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Server calls this once on startup so worker threads can schedule
        callbacks on the asyncio loop."""
        self._loop = loop

    # -- job lifecycle -------------------------------------------------------

    def create_job(self, params: dict, input_files: list[tuple[str, bytes]]) -> JobState:
        """Allocate a job UUID, materialize uploaded files to disk, return state.

        ``input_files`` is a list of ``(filename, file_bytes)``. Names are
        sanitized (basename only) to prevent path traversal.
        """
        job_id = uuid.uuid4().hex
        job_dir = self.jobs_root / job_id
        inputs_dir = job_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=False)

        saved_paths: list[Path] = []
        for filename, data in input_files:
            safe_name = Path(filename).name  # strips any path components
            target = inputs_dir / safe_name
            target.write_bytes(data)
            saved_paths.append(target)

        # part_name is a user-facing label only — never used as a filesystem
        # path on the server. Sanitize aggressively to keep download
        # filenames clean and avoid Content-Disposition surprises.
        part_name = sanitize_part_name(params.pop("part_name", ""))

        state = JobState(
            id=job_id,
            output_dir=job_dir,
            params=dict(params),
            input_image_paths=saved_paths,
            part_name=part_name,
        )
        with self._jobs_lock:
            self._jobs[job_id] = state
        self._persist_state(state)
        return state

    def get(self, job_id: str) -> Optional[JobState]:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def list_ids(self) -> list[str]:
        with self._jobs_lock:
            return list(self._jobs.keys())

    # -- phase dispatch ------------------------------------------------------

    def submit_phase_a(self, job: JobState) -> None:
        """Run preprocess + trace + layout. Stops at the layout preview."""
        job.status = JobStatus.RUNNING
        self._executor.submit(self._run_phase_a, job)

    def submit_phase_b(self, job: JobState) -> None:
        """Run prepare_bin to produce bin_config.json. Fast, no GPU needed."""
        if job.status != JobStatus.AWAITING_DECISION or job.layout_result is None:
            raise RuntimeError(
                f"Job {job.id} cannot proceed from status {job.status}"
            )
        job.status = JobStatus.FINALIZING
        self._executor.submit(self._run_phase_b, job)

    def submit_redo(self, job: JobState, new_params: dict, layout_only: bool) -> None:
        """Re-run with adjusted parameters.

        ``layout_only=True`` re-uses cached per-tool DXFs (cheap; ~seconds).
        ``layout_only=False`` re-traces from the original photos (expensive).
        """
        merged = dict(job.params)
        merged.update(new_params)
        job.params = merged
        job.layout_result = None
        job.final_result = None
        job.error = None
        job.status = JobStatus.RUNNING
        self._persist_state(job)
        self._executor.submit(self._run_phase_a, job, layout_only)

    def overlay_dims_for_stem(self, job: JobState, stem: str) -> Optional[tuple[float, float]]:
        """Return (width_mm, height_mm) for a tool's rectified image, or None.

        Used by the server to embed mm dimensions alongside overlay URLs in
        the /llm_evaluate response so the frontend can convert click
        coordinates into the same mm frame the LLM uses.
        """
        rect = job.output_dir / stem / f"{stem}_rectified.png"
        if not rect.exists():
            return None
        return _rectified_dimensions_mm(rect)

    def generate_overlays(self, job: JobState) -> list[str]:
        """Render per-tool overlay images without calling any LLM.

        Same overlay output as ``run_llm_evaluate`` produces, but without
        the Anthropic round-trip — for the manual corrective-click flow
        where the user picks coordinates themselves. Returns the list of
        input-image stems for which an overlay was successfully written.
        """
        from pic_to_bin.web.overlay import (
            cap_image_size_to_jpeg,
            DEFAULT_LLM_IMAGE_MAX_BYTES,
            generate_overlay_image,
        )

        if job.status not in (JobStatus.AWAITING_DECISION, JobStatus.COMPLETE):
            raise RuntimeError(
                f"Job {job.id} not in awaiting_decision or complete "
                f"(status={job.status.value})"
            )

        rectified_paths = self._rectified_paths_for(job)
        if not rectified_paths:
            raise RuntimeError(
                "No rectified images found; run Phase A first."
            )

        stems: list[str] = []
        with _job_log_capture(job):
            for rect in rectified_paths:
                stem = rect.parent.name
                trace_dxf = rect.with_name(rect.stem + "_trace.dxf")
                overlay_full = rect.with_name(rect.stem + "_overlay_full.png")
                overlay_small = rect.with_name(rect.stem + "_overlay.jpg")
                if not trace_dxf.exists():
                    continue
                try:
                    generate_overlay_image(
                        rectified_path=rect,
                        trace_dxf_path=trace_dxf,
                        output_path=overlay_full,
                    )
                    cap_image_size_to_jpeg(
                        overlay_full, overlay_small,
                        max_bytes=DEFAULT_LLM_IMAGE_MAX_BYTES,
                    )
                    stems.append(stem)
                except Exception as e:  # noqa: BLE001
                    self._dispatch_event(job, ProgressEvent(
                        step="overlay_failed",
                        message=f"Overlay generation failed for {rect.name}: {e}",
                    ))
        return stems

    def run_llm_evaluate(
        self,
        job: JobState,
        auto_loop: bool,
        max_iterations: int,
    ) -> tuple["LLMVerdict", int, list[str]]:
        """Synchronously evaluate the layout via the Anthropic API.

        Runs on the calling thread (FastAPI handler dispatches to the
        executor pool via run_in_executor so the event loop isn't
        blocked). Emits SSE progress events so the UI can show a
        spinner with reasoning.

        In auto-loop mode, applies suggested params and re-runs Phase A
        up to `max_iterations` times. Phase A is invoked synchronously
        (`self._run_phase_a` directly), reusing the job's existing
        executor context — no nested submit() needed.

        Returns ``(verdict, iterations, overlay_stems)`` where
        ``overlay_stems`` is the list of input-image stems for which an
        overlay PNG was successfully written on the *final* iteration —
        the frontend uses these to build URLs into ``GET
        /jobs/{id}/overlays/{stem}`` so the verdict card can display the
        same overlay images the model judged.
        """
        # Local import keeps `anthropic` out of the module-level deps,
        # so non-LLM tests still load pic_to_bin.web.jobs cleanly.
        from pic_to_bin.web.llm_check import LLMVerdict, evaluate_layout
        from pic_to_bin.web.overlay import generate_overlay_image

        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        if job.status not in (JobStatus.AWAITING_DECISION, JobStatus.COMPLETE):
            raise RuntimeError(
                f"Job {job.id} not in awaiting_decision or complete "
                f"(status={job.status.value})"
            )
        if job.layout_result is None:
            raise RuntimeError(f"Job {job.id} has no layout to evaluate")

        max_iter = max(1, int(max_iterations)) if auto_loop else 1
        verdict: Optional["LLMVerdict"] = None
        overlay_stems: list[str] = []

        # Capture pipeline+LLM diagnostics into <job_dir>/job.log. Nested
        # _run_phase_a calls below add their own capture; the context manager
        # save/restores the previous target so output keeps flowing into
        # this same file while they're running.
        with _job_log_capture(job):
            verdict, iteration, overlay_stems = self._llm_evaluate_loop(
                job, evaluate_layout, generate_overlay_image,
                max_iter=max_iter, auto_loop=auto_loop,
            )
        return verdict, iteration, overlay_stems

    def _llm_evaluate_loop(
        self,
        job: "JobState",
        evaluate_layout,
        generate_overlay_image,
        *,
        max_iter: int,
        auto_loop: bool,
    ) -> tuple["LLMVerdict", int, list[str]]:
        """Inner loop body for run_llm_evaluate; factored out so the
        outer method can wrap the whole thing in a single log-capture
        context. See run_llm_evaluate for behavior."""
        verdict: Optional["LLMVerdict"] = None
        overlay_stems: list[str] = []

        for iteration in range(1, max_iter + 1):
            rectified_paths = self._rectified_paths_for(job)
            preview_path = job.output_dir / "layout_preview.png"
            if not rectified_paths:
                raise RuntimeError(
                    "No rectified images found for LLM evaluation; "
                    "run Phase A first."
                )
            if not preview_path.exists():
                raise RuntimeError(
                    f"Layout preview missing at {preview_path}; "
                    "run Phase A first."
                )

            # Build a per-tool overlay PNG that pastes the trace polygons
            # directly on top of the rectified photo at the same mm scale.
            # Without this, the LLM has to mentally align two different
            # coordinate systems (template-zone photo vs. bin-coordinate
            # layout preview), and it tends to wave through layouts whose
            # tolerance perimeter clearly doesn't match the physical tool.
            #
            # The full-resolution overlay PNG can be 5–10 MB on a high-res
            # iPhone photo; we cap each one to ≤ 1 MB by re-saving as JPEG
            # (with progressive quality + dimension reduction) before
            # forwarding to Anthropic. The capped JPEG is what gets served
            # by the /jobs/{id}/overlays/{stem} endpoint too — the LLM
            # sees the same image the human reviewer sees.
            #
            # `overlay_paths` is what gets sent to Anthropic; `overlay_stems`
            # tracks the per-tool input-image stems whose overlay was
            # successfully written, for the frontend to display alongside
            # the verdict.
            from pic_to_bin.web.overlay import (
                cap_image_size_to_jpeg,
                DEFAULT_LLM_IMAGE_MAX_BYTES,
            )
            overlay_paths: list[Path] = []
            overlay_stems = []  # reset each iteration; only the final list is returned
            for rect in rectified_paths:
                # rect is `<job_dir>/<input_stem>/<input_stem>_rectified.png`,
                # so `input_stem` is the parent directory's name — the URL
                # used by the new GET /jobs/{id}/overlays/{stem} endpoint.
                input_stem = rect.parent.name
                trace_dxf = rect.with_name(rect.stem + "_trace.dxf")
                overlay_full = rect.with_name(rect.stem + "_overlay_full.png")
                overlay_small = rect.with_name(rect.stem + "_overlay.jpg")
                if not trace_dxf.exists():
                    # Defensive: skip overlay generation if the trace
                    # didn't get to the export stage (e.g. partial
                    # failures). Fall back to the bare rectified, also
                    # capped to keep the upload bounded.
                    try:
                        cap_image_size_to_jpeg(
                            rect, overlay_small,
                            max_bytes=DEFAULT_LLM_IMAGE_MAX_BYTES,
                        )
                        overlay_paths.append(overlay_small)
                    except Exception:  # noqa: BLE001
                        overlay_paths.append(rect)
                    continue
                try:
                    generate_overlay_image(
                        rectified_path=rect,
                        trace_dxf_path=trace_dxf,
                        output_path=overlay_full,
                    )
                    cap_image_size_to_jpeg(
                        overlay_full, overlay_small,
                        max_bytes=DEFAULT_LLM_IMAGE_MAX_BYTES,
                    )
                    overlay_paths.append(overlay_small)
                    overlay_stems.append(input_stem)
                except Exception as e:  # noqa: BLE001
                    # Overlay rendering shouldn't tank the whole LLM
                    # check. Log via the SSE channel and fall back to
                    # the unannotated rectified image (capped).
                    self._dispatch_event(job, ProgressEvent(
                        step="llm_overlay_failed",
                        message=(
                            f"Overlay generation failed for "
                            f"{rect.name}: {e}. Falling back to bare "
                            f"rectified image."
                        ),
                    ))
                    try:
                        cap_image_size_to_jpeg(
                            rect, overlay_small,
                            max_bytes=DEFAULT_LLM_IMAGE_MAX_BYTES,
                        )
                        overlay_paths.append(overlay_small)
                    except Exception:  # noqa: BLE001
                        overlay_paths.append(rect)

            self._dispatch_event(job, ProgressEvent(
                step="llm_calling",
                message=(
                    f"Asking LLM (iteration {iteration}/{max_iter})…"
                    if auto_loop
                    else "Asking LLM…"
                ),
                fraction=0.0,
            ))

            # Read each rectified image's mm dimensions so the LLM can
            # emit corrective_points in the overlay's coordinate frame.
            # The overlay PNG renders the rectified at exact mm scale
            # (see pic_to_bin.web.overlay.generate_overlay_image), so
            # the rectified's frame IS the overlay's frame.
            overlay_dims: list[tuple[float, float]] = []
            for rect in rectified_paths:
                d = _rectified_dimensions_mm(rect)
                if d is not None:
                    overlay_dims.append(d)
                else:
                    # Fall back to (0, 0) so the index alignment with
                    # rectified_paths stays correct; evaluate_layout
                    # only uses entries with a positive mm size.
                    overlay_dims.append((0.0, 0.0))

            verdict = evaluate_layout(
                rectified_paths=overlay_paths,
                layout_preview_path=preview_path,
                current_params=dict(job.params),
                api_key=self.anthropic_api_key,
                overlay_dimensions_mm=overlay_dims,
            )

            self._dispatch_event(job, ProgressEvent(
                step="llm_verdict",
                message=verdict.reasoning,
                fraction=1.0,
                extra={
                    "verdict": verdict.to_jsonable(),
                    "iteration": iteration,
                    "auto_loop": auto_loop,
                    "max_iterations": max_iter,
                },
            ))

            # Stop conditions: model says ok, or we're not auto-looping,
            # or there's nothing actionable to apply (neither numeric
            # tweaks nor SAM2 corrective clicks).
            has_actionable = bool(
                verdict.suggested_params or verdict.corrective_points
            )
            if verdict.ok or not auto_loop or not has_actionable:
                return verdict, iteration, overlay_stems

            # Apply suggested params and re-run Phase A in-thread.
            self._dispatch_event(job, ProgressEvent(
                step="llm_applying",
                message=f"Applying LLM suggestion (iteration {iteration})…",
                fraction=0.0,
            ))
            merged = dict(job.params)
            merged.update(verdict.suggested_params)
            if verdict.corrective_points:
                merged["sam_corrective_points"] = _merge_corrective_points(
                    merged.get("sam_corrective_points"),
                    verdict.corrective_points,
                    rectified_paths,
                )
            job.params = merged
            job.layout_result = None
            job.error = None
            # Corrective points always require a SAM2 re-trace — no
            # numeric knob can replay them on a cached mask. Numeric
            # params follow the existing trace-affecting rule.
            layout_only = (
                not verdict.corrective_points
                and not _suggested_params_require_retrace(
                    verdict.suggested_params
                )
            )
            job.status = JobStatus.RUNNING
            self._run_phase_a(job, skip_trace=layout_only)
            if job.status != JobStatus.AWAITING_DECISION:
                # Pipeline failure during the redo — bail with the last
                # verdict we have.
                return verdict, iteration, overlay_stems

        # max_iter exhausted without a `ok` verdict; return the last one.
        assert verdict is not None
        return verdict, max_iter, overlay_stems

    def _rectified_paths_for(self, job: JobState) -> list[Path]:
        """Find each input image's rectified PNG produced by phone preprocessing.

        Phone preprocessing emits `<output_dir>/<stem>/<stem>_rectified.png`
        for each uploaded image. Missing files are skipped silently —
        an empty list signals "no Phase A run yet".
        """
        out: list[Path] = []
        for img in job.input_image_paths:
            stem = img.stem
            rect = job.output_dir / stem / f"{stem}_rectified.png"
            if rect.exists():
                out.append(rect)
        return out

    # -- worker bodies (run in thread pool) ----------------------------------

    def _run_phase_a(self, job: JobState, skip_trace: bool = False) -> None:
        try:
            with _job_log_capture(job), self._gpu_sem:
                result = run_pipeline(
                    image_paths=job.input_image_paths,
                    output_dir=job.output_dir,
                    stop_after="layout",
                    skip_trace=skip_trace,
                    progress_cb=lambda ev: self._dispatch_event(job, ev),
                    **_pipeline_kwargs(job.params),
                )
            job.layout_result = result
            job.status = JobStatus.AWAITING_DECISION
            self._dispatch_event(job, ProgressEvent(
                step="layout_ready",
                message="Layout preview ready",
                fraction=1.0,
            ))
        except Exception as e:  # noqa: BLE001 — surface every failure to user
            job.status = JobStatus.ERROR
            job.error = str(e)
            self._dispatch_event(job, ProgressEvent(
                step="error", message=str(e), fraction=1.0,
            ))
        finally:
            self._persist_state(job)

    def _run_phase_b(self, job: JobState) -> None:
        try:
            with _job_log_capture(job):
                # Phase B doesn't need the GPU. Run prepare_bin via run_pipeline with
                # skip_trace=True so the cached DXFs and combined layout are reused
                # and only prepare_bin executes (Phase A's outputs are still on disk).
                result = run_pipeline(
                    image_paths=job.input_image_paths,
                    output_dir=job.output_dir,
                    stop_after="all",
                    skip_trace=True,
                    progress_cb=lambda ev: self._dispatch_event(job, ev),
                    **_pipeline_kwargs(job.params),
                )
            job.final_result = result
            job.status = JobStatus.COMPLETE
            self._dispatch_event(job, ProgressEvent(
                step="complete", message="Bin config ready", fraction=1.0,
            ))
        except Exception as e:  # noqa: BLE001
            job.status = JobStatus.ERROR
            job.error = str(e)
            self._dispatch_event(job, ProgressEvent(
                step="error", message=str(e), fraction=1.0,
            ))
        finally:
            self._persist_state(job)

    # -- event fan-out (called from worker threads) --------------------------

    def _dispatch_event(self, job: JobState, event: ProgressEvent) -> None:
        ev_dict = event.to_dict()
        with job._lock:
            job.event_log.append(ev_dict)
            job.last_activity = time.time()
            subscribers = list(job.subscribers)
        loop = self._loop
        if loop is None:
            return  # no subscribers yet; the event_log replay covers this
        for q in subscribers:
            try:
                loop.call_soon_threadsafe(q.put_nowait, ev_dict)
            except RuntimeError:
                # Loop closed (server shutting down) — drop the event silently.
                pass

    # -- SSE subscription ----------------------------------------------------

    async def subscribe(self, job: JobState):
        """Async generator of event dicts. Replays log first, then streams.

        Closes when a terminal status event (``complete`` or ``error``) is
        delivered, or when ``signal_subscribers_shutdown`` pushes a sentinel
        during app shutdown.
        """
        queue: asyncio.Queue = asyncio.Queue()
        with job._lock:
            for ev in list(job.event_log):
                queue.put_nowait(ev)
            job.subscribers.append(queue)
        try:
            while True:
                ev = await queue.get()
                if ev is _SHUTDOWN_SENTINEL:
                    return
                yield ev
                if ev.get("step") in ("complete", "error"):
                    return
        finally:
            with job._lock:
                if queue in job.subscribers:
                    job.subscribers.remove(queue)

    def has_active_subscribers(self) -> bool:
        """True iff any job still has at least one queue connected to a live
        SSE subscriber. Used by the lifespan handler to short-circuit its
        shutdown drain loop the moment all SSE responses have flushed."""
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            with job._lock:
                if job.subscribers:
                    return True
        return False

    def signal_subscribers_shutdown(self) -> None:
        """Push the shutdown sentinel to every active subscriber queue.

        Called from the asyncio loop during app shutdown so SSE generators
        exit cleanly and EventSourceResponse can flush its closing frame
        before uvicorn cancels the task.
        """
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            with job._lock:
                queues = list(job.subscribers)
            for q in queues:
                try:
                    q.put_nowait(_SHUTDOWN_SENTINEL)
                except asyncio.QueueFull:
                    # Queues are unbounded so this shouldn't happen, but if
                    # it ever does, the awaiter will be cancelled by uvicorn
                    # in the next phase anyway.
                    pass

    # -- TTL cleanup ---------------------------------------------------------

    def sweep_expired(self, now: Optional[float] = None) -> int:
        """Delete jobs whose last_activity is older than ttl_hours. Returns
        the number of jobs removed."""
        cutoff = (now or time.time()) - self.ttl_hours * 3600.0
        removed = 0
        expired_ids: list[str] = []
        with self._jobs_lock:
            for jid, state in self._jobs.items():
                if state.last_activity < cutoff and state.status in (
                    JobStatus.COMPLETE, JobStatus.ERROR
                ):
                    expired_ids.append(jid)
            for jid in expired_ids:
                self._jobs.pop(jid, None)
        for jid in expired_ids:
            job_dir = self.jobs_root / jid
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            removed += 1
        return removed

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- on-disk persistence (for surviving server restarts) -----------------

    def _persist_state(self, job: JobState) -> None:
        """Snapshot the persistable fields of ``job`` to ``job_state.json``.

        Best-effort: persistence failures are logged but never propagate, so
        a transient disk hiccup can't break a running pipeline. The worst
        case is that a restart misses one update and the next save catches
        up. The on-disk artifacts (layout_preview.png, bin_config.json) are
        the source of truth for status; this file just carries the params,
        part_name, and grid summary that aren't otherwise recoverable."""
        with job._lock:
            ls = None
            if job.layout_result:
                ls = {
                    "grid_units_x": job.layout_result.get("grid_units_x"),
                    "grid_units_y": job.layout_result.get("grid_units_y"),
                }
            snapshot = {
                "version": 1,
                "id": job.id,
                "status": job.status.value,
                "params": job.params,
                "part_name": job.part_name,
                "created_at": job.created_at,
                "last_activity": job.last_activity,
                "input_filenames": [p.name for p in job.input_image_paths],
                "layout_summary": ls,
                "error": job.error,
            }
        try:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            target = job.output_dir / "job_state.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            tmp.replace(target)
        except Exception as e:  # noqa: BLE001
            print(f"[jobs] failed to persist state for {job.id}: {e}")

    def _restore_jobs(self) -> int:
        """Rebuild ``self._jobs`` from on-disk ``job_state.json`` files.

        Status is inferred from the artifacts that actually exist on disk
        (so a saved status that's out of sync with the file system loses to
        what's really there). Dirs without a ``job_state.json`` are skipped
        — they predate this persistence and we can't safely redo them
        without knowing the original params."""
        if not self.jobs_root.exists():
            return 0
        restored = 0
        for sub in self.jobs_root.iterdir():
            if not sub.is_dir():
                continue
            state_path = sub / "job_state.json"
            if not state_path.exists():
                continue
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"[jobs] skipping {sub.name}: bad job_state.json ({e})")
                continue

            job_id = saved.get("id") or sub.name
            inputs_dir = sub / "inputs"
            input_paths = [
                inputs_dir / fn
                for fn in (saved.get("input_filenames") or [])
                if (inputs_dir / fn).exists()
            ]

            layout_preview = sub / "layout_preview.png"
            combined_dxf = sub / "combined_layout.dxf"
            bin_config = sub / "bin_config.json"

            layout_result: Optional[dict] = None
            final_result: Optional[dict] = None
            error = saved.get("error")

            ls = saved.get("layout_summary") or {}
            grid_x = ls.get("grid_units_x")
            grid_y = ls.get("grid_units_y")

            if bin_config.exists() and layout_preview.exists():
                # Phase B finished successfully.
                if grid_x is None or grid_y is None:
                    # Older saves may not have the summary; recover from
                    # bin_config.json which has the same numbers.
                    try:
                        cfg = json.loads(
                            bin_config.read_text(encoding="utf-8")
                        )
                        grid_x = grid_x or cfg.get("grid_x")
                        grid_y = grid_y or cfg.get("grid_y")
                    except (json.JSONDecodeError, OSError):
                        pass
                status = JobStatus.COMPLETE
                layout_result = {
                    "grid_units_x": grid_x,
                    "grid_units_y": grid_y,
                    "preview_path": str(layout_preview),
                    "combined_dxf_path": str(combined_dxf),
                }
                final_result = {"bin_config": str(bin_config)}
            elif layout_preview.exists():
                # Phase A finished, awaiting user proceed/redo.
                status = JobStatus.AWAITING_DECISION
                layout_result = {
                    "grid_units_x": grid_x,
                    "grid_units_y": grid_y,
                    "preview_path": str(layout_preview),
                    "combined_dxf_path": str(combined_dxf),
                }
            else:
                # No layout artifacts → server died mid-Phase A. Surface
                # as an error so the user knows to resubmit; downloads
                # would 404 anyway.
                status = JobStatus.ERROR
                error = error or (
                    "Server restarted while this job was running. "
                    "Submit again to retry."
                )

            now = time.time()
            state = JobState(
                id=job_id,
                output_dir=sub,
                status=status,
                created_at=saved.get("created_at") or now,
                last_activity=saved.get("last_activity") or now,
                params=saved.get("params") or {},
                input_image_paths=input_paths,
                layout_result=layout_result,
                final_result=final_result,
                error=error,
                part_name=saved.get("part_name") or "",
            )
            self._jobs[job_id] = state
            restored += 1
        return restored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Suggested-param keys that affect tracing (and therefore force a full
# re-trace, not a layout-only redo). Must match the trace-required set
# the frontend uses for manual redos.
_TRACE_AFFECTING_PARAMS = {
    "tolerance",
    "axial_tolerance",
    "display_smooth_sigma",
    "mask_erode",
}


def _suggested_params_require_retrace(suggested: dict) -> bool:
    """True iff any of the suggested params would invalidate the cached trace."""
    return any(k in _TRACE_AFFECTING_PARAMS for k in suggested)


def _rectified_dimensions_mm(rectified_path: Path) -> Optional[tuple[float, float]]:
    """Return (width_mm, height_mm) of a rectified PNG via its sidecar JSON.

    Returns None if the sidecar is missing or unparseable. The web overlay
    image is rendered at the same mm scale as the rectified, so we can
    use these dimensions verbatim as the overlay's coordinate frame for
    the LLM.
    """
    sidecar = rectified_path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        dpi = float(meta.get("effective_dpi") or 0)
        w_px = float(meta.get("image_width_px") or 0)
        h_px = float(meta.get("image_height_px") or 0)
    except (json.JSONDecodeError, ValueError, OSError):
        return None
    if dpi <= 0 or w_px <= 0 or h_px <= 0:
        return None
    return (w_px / dpi * 25.4, h_px / dpi * 25.4)


def _merge_corrective_points(
    existing: Optional[dict],
    new_points: list,
    rectified_paths: list,
) -> dict:
    """Merge a verdict's corrective_points into the per-stem persistence dict.

    ``new_points`` is a list of ``{overlay_index, x_mm, y_mm, label, ...}``
    dicts as parsed by ``llm_check._parse_verdict``. ``overlay_index`` is
    1-based and indexes into ``rectified_paths`` in the order they were
    sent to the LLM. The rectified path's parent dir name is the input
    image's stem — that's the key ``run_pipeline`` expects. Out-of-range
    indices are dropped silently.

    Existing points for a stem are preserved and the new points are
    appended (accumulate-across-iterations semantics): a later
    iteration's clicks add to the SAM2 prompt set rather than replacing
    earlier ones, so a click that fixed iteration 1's issue keeps
    working in iteration 2.
    """
    out: dict = {k: list(v) for k, v in (existing or {}).items()}
    for pt in new_points:
        idx = pt.get("overlay_index")
        if not isinstance(idx, int) or idx < 1 or idx > len(rectified_paths):
            continue
        stem = Path(rectified_paths[idx - 1]).parent.name
        entry = {
            "x_mm": float(pt["x_mm"]),
            "y_mm": float(pt["y_mm"]),
            "label": int(pt["label"]),
        }
        out.setdefault(stem, []).append(entry)
    return out


# The keys that map straight from form params → run_pipeline kwargs.
_PIPELINE_PARAM_KEYS = {
    "tool_heights",
    "paper_size",
    "tolerance",
    "axial_tolerance",
    "phone_height",
    "tool_taper",
    "gap",
    "bin_margin",
    "max_units",
    "min_units",
    "height_units",
    "stacking",
    "slots",
    "straighten_threshold",
    "max_refine_iterations",
    "max_concavity_depth",
    "mask_erode",
    "display_smooth_sigma",
    "sam_model",
    "sam_corrective_points",
}


def _pipeline_kwargs(params: dict) -> dict:
    """Filter params to only the kwargs run_pipeline() accepts."""
    out = {k: v for k, v in params.items() if k in _PIPELINE_PARAM_KEYS}
    if "tool_heights" in out:
        out["tool_heights"] = _normalize_tool_heights(out["tool_heights"])
    return out


# Original artifact filename → friendly suffix used when part_name is set.
# Result: <part_name>_<suffix>. Original filenames stay on disk; this only
# affects what the browser saves.
_DOWNLOAD_SUFFIX = {
    "layout_preview.png": "preview.png",
    "layout_actual_size.pdf": "layout.pdf",
    "layout_actual_size.svg": "layout.svg",
    "combined_layout.dxf": "layout.dxf",
    "bin_config.json": "bin_config.json",
}


def sanitize_part_name(value) -> str:
    """Reduce a user-supplied label to filesystem-safe characters.

    Keeps letters, digits, underscores, and hyphens. Whitespace runs collapse
    to a single underscore. Leading/trailing punctuation is trimmed and the
    result is capped at 64 chars. Returns an empty string for falsy input
    or input that has nothing safe left after stripping.
    """
    if not value:
        return ""
    s = re.sub(r"\s+", "_", str(value).strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    s = s.strip("_-")
    return s[:64]


def download_filename(part_name: str, artifact_name: str) -> str:
    """Compute the download filename shown to the user.

    With ``part_name`` set, returns ``<part_name>_<friendly suffix>`` where
    the friendly suffix drops the verbose internal name (e.g.
    ``layout_actual_size.pdf`` → ``layout.pdf``). Without a part_name, falls
    back to the original artifact filename.
    """
    if part_name and artifact_name in _DOWNLOAD_SUFFIX:
        return f"{part_name}_{_DOWNLOAD_SUFFIX[artifact_name]}"
    return artifact_name


def _normalize_tool_heights(value):
    """Coerce tool_heights into the shape ``prepare_bin`` expects.

    The frontend sends ``{"0": 17, "1": 14}`` because JSON object keys are
    always strings, but ``prepare_bin.build_config`` indexes with integer
    tool indices and only treats ``"default"`` as a string key.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        normalized = {}
        for k, v in value.items():
            if k == "default":
                normalized["default"] = float(v)
                continue
            try:
                normalized[int(k)] = float(v)
            except (TypeError, ValueError):
                normalized[k] = float(v)
        return normalized
    return value
