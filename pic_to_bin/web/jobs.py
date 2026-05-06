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
        self._executor.submit(self._run_phase_a, job, layout_only)

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
        if job.status != JobStatus.AWAITING_DECISION:
            raise RuntimeError(
                f"Job {job.id} not in awaiting_decision (status={job.status.value})"
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

            verdict = evaluate_layout(
                rectified_paths=overlay_paths,
                layout_preview_path=preview_path,
                current_params=dict(job.params),
                api_key=self.anthropic_api_key,
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
            # or there's nothing actionable to apply.
            if verdict.ok or not auto_loop or not verdict.suggested_params:
                return verdict, iteration, overlay_stems

            # Apply suggested params and re-run Phase A in-thread.
            self._dispatch_event(job, ProgressEvent(
                step="llm_applying",
                message=f"Applying LLM suggestion (iteration {iteration})…",
                fraction=0.0,
            ))
            merged = dict(job.params)
            merged.update(verdict.suggested_params)
            job.params = merged
            job.layout_result = None
            job.error = None
            layout_only = not _suggested_params_require_retrace(
                verdict.suggested_params
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
