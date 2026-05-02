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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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


def create_app(jobs_root: Path, ttl_hours: float = 24.0) -> FastAPI:
    job_manager = JobManager(jobs_root=jobs_root, ttl_hours=ttl_hours)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        job_manager.bind_loop(asyncio.get_running_loop())
        cleanup_task = asyncio.create_task(_periodic_sweep(job_manager))
        try:
            yield
        finally:
            cleanup_task.cancel()
            job_manager.shutdown()

    app = FastAPI(title="pic-to-bin", lifespan=lifespan)
    app.state.job_manager = job_manager

    # ---- Static UI ---------------------------------------------------------

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
        return job.to_summary()

    @app.get("/jobs/{job_id}/events")
    async def events(job_id: str, request: Request):
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")

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
        if job.status not in (JobStatus.AWAITING_DECISION, JobStatus.ERROR):
            raise HTTPException(
                409,
                f"can only redo from awaiting_decision or error "
                f"(current: {job.status.value})",
            )
        new_params = payload.get("params") or {}
        layout_only = bool(payload.get("layout_only", True))
        if not isinstance(new_params, dict):
            raise HTTPException(400, "params must be a JSON object")
        job_manager.submit_redo(job, new_params, layout_only=layout_only)
        return {"job_id": job.id, "status": job.status.value}

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
    import uvicorn

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

    # The factory closure captures jobs_root so the app instance is rebuilt
    # cleanly on reload.
    def factory():
        return create_app(jobs_root, ttl_hours=args.job_ttl_hours)

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
