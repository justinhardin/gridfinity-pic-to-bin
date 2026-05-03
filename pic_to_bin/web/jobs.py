"""JobManager: per-request UUID directories, thread pool, GPU semaphore, SSE event fan-out.

Each browser submission becomes a JobState. Phase A (preprocess + trace +
layout) runs on submit. The user reviews the layout preview and then either
"proceeds" (Phase B: prepare_bin → bin_config.json) or "re-does" with adjusted
parameters. The GPU semaphore serializes SAM2 inference across concurrent jobs;
without it the second concurrent submission would OOM the GPU.
"""

from __future__ import annotations

import asyncio
import re
import shutil
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

    # -- worker bodies (run in thread pool) ----------------------------------

    def _run_phase_a(self, job: JobState, skip_trace: bool = False) -> None:
        try:
            with self._gpu_sem:
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
