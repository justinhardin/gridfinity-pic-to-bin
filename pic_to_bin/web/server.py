"""FastAPI server for the pic-to-bin web app.

Endpoints
---------
GET  /                          → static index.html
GET  /static/{path}             → static assets (JS, CSS, vendored Lit)
POST /jobs                      → create job (multipart: images + JSON params)
GET  /jobs/{id}                 → job summary (status, artifact URLs)
GET  /jobs/{id}/events          → SSE stream of progress events
POST /jobs/{id}/proceed         → run Phase B (bin_config.json)
POST /jobs/{id}/redo            → re-run with adjusted params
GET  /jobs/{id}/artifacts/{fn}  → download layout_preview.png / combined_layout.dxf / bin_config.json
POST /preview                   → convert a HEIC/HEIF upload to a small JPEG thumbnail
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from pic_to_bin.web.jobs import JobManager, JobStatus, download_filename

logger = logging.getLogger("pic_to_bin.web")

STATIC_DIR = Path(__file__).parent / "static"
ARTIFACT_WHITELIST = {
    "layout_preview.png": ("image/png", "layout_preview.png"),
    "layout_actual_size.pdf": ("application/pdf", "layout_actual_size.pdf"),
    "layout_actual_size.svg": ("image/svg+xml", "layout_actual_size.svg"),
    "combined_layout.dxf": ("application/dxf", "combined_layout.dxf"),
    "bin_config.json": ("application/json", "bin_config.json"),
}
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".heif"}


def create_app(
    jobs_root: Path,
    ttl_hours: float = 24.0,
    anthropic_api_key: Optional[str] = None,
) -> FastAPI:
    job_manager = JobManager(
        jobs_root=jobs_root,
        ttl_hours=ttl_hours,
        anthropic_api_key=anthropic_api_key,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        job_manager.bind_loop(asyncio.get_running_loop())
        cleanup_task = asyncio.create_task(_periodic_sweep(job_manager))
        try:
            yield
        finally:
            cleanup_task.cancel()
            # Tell active SSE subscribers to exit so EventSourceResponse can
            # close cleanly before uvicorn cancels their tasks. The flush
            # chain — subscribe() exits → event_generator() ends →
            # EventSourceResponse writes its closing frame → network flush —
            # takes several event-loop ticks per active connection, so a
            # single sleep(0) isn't enough on busy or slow machines. Yield
            # in short bursts up to ~250 ms total, exiting early once every
            # subscriber queue has drained.
            job_manager.signal_subscribers_shutdown()
            for _ in range(25):
                await asyncio.sleep(0.01)
                if not job_manager.has_active_subscribers():
                    break
            job_manager.shutdown()

    app = FastAPI(title="pic-to-bin", lifespan=lifespan)
    app.state.job_manager = job_manager

    # ---- Static UI ---------------------------------------------------------

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/favicon.ico")
    async def favicon() -> FileResponse:
        # Browsers auto-request /favicon.ico even when <link rel="icon">
        # points elsewhere. Serve the 32×32 PNG here so the request
        # succeeds without us shipping a separate .ico file.
        return FileResponse(
            STATIC_DIR / "favicon-32.png", media_type="image/png"
        )

    @app.get("/.well-known/appspecific/com.chrome.devtools.json")
    async def chrome_devtools_workspace() -> dict:
        # Chrome DevTools' "Automatic Workspace Folders" feature probes
        # this URL when DevTools is open against a localhost origin. A
        # proper response auto-attaches the project source for in-browser
        # editing. Spec:
        # https://chromium.googlesource.com/devtools/devtools-frontend/+/main/docs/ecosystem/automatic_workspace_folders.md
        import uuid as _uuid
        project_root = STATIC_DIR.parent.parent.parent
        return {
            "workspace": {
                "root": str(project_root),
                "uuid": str(_uuid.uuid5(_uuid.NAMESPACE_URL, str(project_root))),
            }
        }

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        # Cache-bust the JS/CSS by appending each file's mtime as a query
        # string. The browser caches them happily when unchanged but
        # re-fetches as soon as we edit the source — no manual hard-refresh.
        index_path = STATIC_DIR / "index.html"
        html_text = index_path.read_text(encoding="utf-8")
        for asset in ("app.js", "styles.css"):
            asset_path = STATIC_DIR / asset
            if asset_path.exists():
                v = int(asset_path.stat().st_mtime)
                html_text = html_text.replace(
                    f"/static/{asset}", f"/static/{asset}?v={v}"
                )
        return HTMLResponse(
            html_text, headers={"cache-control": "no-cache"}
        )

    # ---- Job endpoints -----------------------------------------------------

    @app.post("/jobs")
    async def create_job(
        params: str = Form(...),
        images: list[UploadFile] = File(...),
    ) -> dict:
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"params must be valid JSON: {e}")

        if not isinstance(params_dict, dict):
            raise HTTPException(400, "params must be a JSON object")

        if not images:
            raise HTTPException(400, "at least one image is required")

        # Validate tool_heights present (required) and image extensions.
        tool_heights = params_dict.get("tool_heights")
        if tool_heights in (None, "", {}, []):
            raise HTTPException(400, "tool_heights is required")

        files: list[tuple[str, bytes]] = []
        for upload in images:
            ext = Path(upload.filename or "").suffix.lower()
            if ext not in ALLOWED_IMAGE_EXTS:
                raise HTTPException(
                    400,
                    f"unsupported image extension: {ext!r}. "
                    f"Allowed: {sorted(ALLOWED_IMAGE_EXTS)}",
                )
            data = await upload.read()
            if not data:
                raise HTTPException(400, f"{upload.filename} is empty")
            files.append((upload.filename or f"image{ext}", data))

        job = job_manager.create_job(params_dict, files)
        job_manager.submit_phase_a(job)
        return {"job_id": job.id, "status": job.status.value}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        summary = job.to_summary()
        # Surface server-level capabilities alongside per-job state so the
        # frontend can hide the LLM button when the API key isn't set,
        # without an extra round-trip on every load.
        summary["llm_available"] = job_manager.llm_available
        return summary

    @app.get("/config")
    async def get_config() -> dict:
        """Server capabilities the frontend needs at startup time."""
        return {"llm_available": job_manager.llm_available}

    @app.get("/jobs/{job_id}/events")
    async def events(job_id: str, request: Request):
        job = job_manager.get(job_id)
        if job is None:
            # The classic 404 here works for the initial connection but is
            # awful when a browser tab outlives a server restart: jobs are
            # in-memory only, so an EventSource pointing at the old job ID
            # keeps reconnecting and hammering the new server's logs with
            # 404s every ~3 s. Return a 200 SSE stream instead, with a
            # `session_lost` event the frontend can recognize, plus
            # `retry: 0` so the browser stops auto-reconnecting.
            async def gone_stream():
                yield {
                    "event": "session_lost",
                    "retry": 0,
                    "data": json.dumps({
                        "step": "session_lost",
                        "message": (
                            "Job not found on this server (it may have "
                            "expired or the server was restarted)."
                        ),
                    }),
                }
            return EventSourceResponse(gone_stream())

        async def event_generator():
            async for ev in job_manager.subscribe(job):
                if await request.is_disconnected():
                    return
                # Single default event type so the browser only needs onmessage.
                # The "step" field inside the JSON tells the UI what kind it is.
                yield {"data": json.dumps(ev)}

        return EventSourceResponse(event_generator())

    @app.post("/jobs/{job_id}/proceed")
    async def proceed(job_id: str) -> dict:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status != JobStatus.AWAITING_DECISION:
            raise HTTPException(
                409,
                f"job is in status {job.status.value}; "
                f"can only proceed from {JobStatus.AWAITING_DECISION.value}",
            )
        job_manager.submit_phase_b(job)
        return {"job_id": job.id, "status": job.status.value}

    @app.post("/jobs/{job_id}/redo")
    async def redo(job_id: str, payload: dict) -> dict:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status not in (
            JobStatus.AWAITING_DECISION,
            JobStatus.ERROR,
            JobStatus.COMPLETE,
        ):
            raise HTTPException(
                409,
                f"can only redo from awaiting_decision, error, or complete "
                f"(current: {job.status.value})",
            )
        new_params = payload.get("params") or {}
        layout_only = bool(payload.get("layout_only", True))
        if not isinstance(new_params, dict):
            raise HTTPException(400, "params must be a JSON object")
        # Corrective SAM2 clicks invalidate any cached trace — they only
        # take effect on a fresh segmentation. Override the flag here so a
        # client can't accidentally request layout_only=True alongside
        # corrective points and silently get the cached (uncorrected) DXF.
        if new_params.get("sam_corrective_points"):
            layout_only = False
        job_manager.submit_redo(job, new_params, layout_only=layout_only)
        return {"job_id": job.id, "status": job.status.value}

    @app.post("/jobs/{job_id}/llm_evaluate")
    async def llm_evaluate(job_id: str, payload: dict) -> dict:
        """Send the rectified photo + layout preview to Claude for a fit verdict.

        Body: ``{"auto_loop": bool=False, "max_iterations": int=3}``.
        Returns ``{"verdict": {...}, "iterations": int}`` once the
        synchronous Anthropic call (or auto-loop sequence) finishes.
        Progress events stream through the existing SSE channel.
        """
        if not job_manager.llm_available:
            raise HTTPException(
                503,
                "LLM evaluation unavailable: ANTHROPIC_API_KEY is not set "
                "on the server. Add it to a .env file at the project root "
                "or export it before launching pic-to-bin-web.",
            )
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status not in (JobStatus.AWAITING_DECISION, JobStatus.COMPLETE):
            raise HTTPException(
                409,
                f"can only evaluate from awaiting_decision or complete "
                f"(current: {job.status.value})",
            )
        auto_loop = bool(payload.get("auto_loop", False))
        max_iterations = int(payload.get("max_iterations", 3))
        if max_iterations < 1 or max_iterations > 10:
            raise HTTPException(
                400, "max_iterations must be between 1 and 10"
            )

        loop = asyncio.get_running_loop()
        try:
            verdict, iterations, overlay_stems = await loop.run_in_executor(
                job_manager._executor,
                job_manager.run_llm_evaluate,
                job,
                auto_loop,
                max_iterations,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("LLM evaluation failed for job %s", job_id)
            raise HTTPException(500, f"LLM evaluation failed: {e}") from e
        overlays_payload: list[dict] = []
        for s in overlay_stems:
            entry = {
                "stem": s,
                "url": f"/jobs/{job.id}/overlays/{s}",
            }
            dims = job_manager.overlay_dims_for_stem(job, s)
            if dims is not None:
                entry["width_mm"] = round(dims[0], 3)
                entry["height_mm"] = round(dims[1], 3)
            overlays_payload.append(entry)
        return {
            "verdict": verdict.to_jsonable(),
            "iterations": iterations,
            "overlays": overlays_payload,
        }

    @app.post("/jobs/{job_id}/overlays")
    async def generate_overlays(job_id: str) -> dict:
        """Render per-tool overlays without calling the LLM.

        Used by the manual corrective-click flow: the frontend renders
        a click UI on top of these overlays so the user can pick
        SAM2 negative/positive points themselves.
        """
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status not in (JobStatus.AWAITING_DECISION, JobStatus.COMPLETE):
            raise HTTPException(
                409,
                f"can only generate overlays from awaiting_decision or "
                f"complete (current: {job.status.value})",
            )
        loop = asyncio.get_running_loop()
        try:
            stems = await loop.run_in_executor(
                job_manager._executor, job_manager.generate_overlays, job,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Overlay generation failed for job %s", job_id)
            raise HTTPException(500, f"overlay generation failed: {e}") from e
        overlays_payload: list[dict] = []
        for s in stems:
            entry = {
                "stem": s,
                "url": f"/jobs/{job.id}/overlays/{s}",
            }
            dims = job_manager.overlay_dims_for_stem(job, s)
            if dims is not None:
                entry["width_mm"] = round(dims[0], 3)
                entry["height_mm"] = round(dims[1], 3)
            overlays_payload.append(entry)
        return {"overlays": overlays_payload}

    @app.get("/jobs/{job_id}/artifacts/{name}")
    async def artifact(job_id: str, name: str):
        if name not in ARTIFACT_WHITELIST:
            raise HTTPException(404, "artifact not found")
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        media_type, filename = ARTIFACT_WHITELIST[name]
        path = job.output_dir / filename
        if not path.exists():
            raise HTTPException(404, f"{filename} not yet produced")
        return FileResponse(
            path,
            media_type=media_type,
            filename=download_filename(job.part_name, filename),
        )

    @app.get("/jobs/{job_id}/inputs/{name}")
    async def input_image(job_id: str, name: str):
        """Serve an originally-uploaded input image.

        Used by the resubmit flow when the user has restored a job: the
        frontend re-fetches the original bytes from this endpoint and
        re-uploads them under a fresh job. ``name`` must match one of
        the input filenames on this job — we never let the URL select
        an arbitrary file path.
        """
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        valid_names = {p.name for p in job.input_image_paths}
        if name not in valid_names:
            raise HTTPException(404, "input not found")
        path = job.output_dir / "inputs" / name
        if not path.exists():
            raise HTTPException(404, "input file missing on disk")
        ext = Path(name).suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".heic": "image/heic",
            ".heif": "image/heif",
        }.get(ext, "application/octet-stream")
        return FileResponse(path, media_type=media_type, filename=name)

    @app.get("/jobs/{job_id}/inputs/{name}/preview")
    async def input_preview(job_id: str, name: str):
        """Browser-renderable thumbnail for an input image.

        For jpg/png this is just the original file. For HEIC, the
        browser can't render it natively, so we lazily decode via
        pillow_heif on the first request and cache a small JPEG
        thumbnail at ``inputs/<name>.preview.jpg`` for subsequent
        loads. Used by the form-restore flow so step-2 thumbnails
        come back the same way they were uploaded.
        """
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        valid_names = {p.name for p in job.input_image_paths}
        if name not in valid_names:
            raise HTTPException(404, "input not found")
        src = job.output_dir / "inputs" / name
        if not src.exists():
            raise HTTPException(404, "input file missing on disk")

        ext = Path(name).suffix.lower()
        if ext not in (".heic", ".heif"):
            media_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
            }.get(ext, "application/octet-stream")
            return FileResponse(src, media_type=media_type)

        cached = src.parent / f"{name}.preview.jpg"
        if not cached.exists():
            try:
                from PIL import Image
                import pillow_heif
                pillow_heif.register_heif_opener()
                with Image.open(src) as img:
                    img.thumbnail((400, 400))
                    img.convert("RGB").save(
                        str(cached), format="JPEG", quality=70
                    )
            except Exception as e:  # noqa: BLE001
                raise HTTPException(
                    500, f"could not generate HEIC preview: {e}"
                )
        return FileResponse(cached, media_type="image/jpeg")

    @app.get("/jobs/{job_id}/overlays/{stem}")
    async def overlay(job_id: str, stem: str):
        """Serve a per-tool overlay image (rectified photo + trace polygons).

        Generated by the LLM evaluate flow; used by the frontend verdict
        card so the human reviewer sees exactly what the model judged.
        ``stem`` must match one of the input image stems on this job —
        we never let the URL select an arbitrary file path.

        Prefers the size-capped JPEG (``<stem>_rectified_overlay.jpg``,
        ≤ 1 MB) the LLM was actually shown; falls back to the older
        full-resolution PNG for jobs created before that pipeline change.
        """
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        valid_stems = {p.stem for p in job.input_image_paths}
        if stem not in valid_stems:
            raise HTTPException(404, "overlay not found")
        stem_dir = job.output_dir / stem
        candidates = [
            (stem_dir / f"{stem}_rectified_overlay.jpg", "image/jpeg"),
            (stem_dir / f"{stem}_rectified_overlay.png", "image/png"),
        ]
        for path, media in candidates:
            if path.exists():
                return FileResponse(path, media_type=media)
        raise HTTPException(
            404, "overlay not yet generated for this tool"
        )

    @app.post("/preview")
    async def preview(image: UploadFile = File(...)) -> Response:
        """Convert a HEIC/HEIF upload to a small JPEG thumbnail.

        Used by the form's drop-zone preview. Browsers (Chrome, Firefox,
        Edge) can't render HEIC natively, and the heic2any wasm shim
        bundles an old libheif that fails on modern iPhone HEIC variants.
        Routing through pillow-heif (already a core dep, used by the
        ingest pipeline) gives us one HEIC decoder that handles every
        format the actual pipeline can.
        """
        ext = Path(image.filename or "").suffix.lower()
        if ext not in {".heic", ".heif"}:
            raise HTTPException(400, "only HEIC/HEIF accepted")
        data = await image.read()
        if not data:
            raise HTTPException(400, "empty file")
        # Lazy import — avoids loading PIL on the cold start path; the
        # pipeline's segment_tool already pulls these in for HEIC inputs.
        from io import BytesIO
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        try:
            img = Image.open(BytesIO(data))
            img.thumbnail((400, 400))  # small thumbnail; preview only
            out = BytesIO()
            img.convert("RGB").save(out, format="JPEG", quality=70)
        except Exception as e:  # noqa: BLE001 — surface to browser
            raise HTTPException(400, f"could not decode HEIC: {e}")
        return Response(content=out.getvalue(), media_type="image/jpeg")

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return app


async def _periodic_sweep(job_manager: JobManager, interval_seconds: float = 1800.0):
    """Background task: every interval, delete expired job dirs."""
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            removed = job_manager.sweep_expired()
            if removed:
                logger.info("TTL sweep: removed %d expired job(s)", removed)
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    """Console-script entry point: ``pic-to-bin-web``."""
    import os

    import uvicorn

    # Pull a `.env` from the CWD before reading env vars. python-dotenv is
    # an optional dep (only installed with the `[web]` extras) so the
    # import is local.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Run the pic-to-bin web app (FastAPI + uvicorn)."
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port (default: 8000)")
    parser.add_argument("--jobs-dir", default="web_jobs",
                        help="Directory for per-job outputs (default: web_jobs/)")
    parser.add_argument("--job-ttl-hours", type=float, default=24.0,
                        help="Delete completed/errored jobs older than N hours (default: 24)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable hot reload (development only)")
    parser.add_argument("--log-level", default="info",
                        help="uvicorn log level (default: info)")
    args = parser.parse_args()

    jobs_root = Path(args.jobs_dir).resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)
    logger.info("Jobs directory: %s", jobs_root)

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or None
    if anthropic_api_key:
        logger.info("LLM fit-check enabled (ANTHROPIC_API_KEY present)")
    else:
        logger.info(
            "LLM fit-check disabled (set ANTHROPIC_API_KEY in env or .env "
            "to enable)"
        )

    # The factory closure captures jobs_root so the app instance is rebuilt
    # cleanly on reload.
    def factory():
        return create_app(
            jobs_root,
            ttl_hours=args.job_ttl_hours,
            anthropic_api_key=anthropic_api_key,
        )

    uvicorn.run(
        factory,
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    cli()
