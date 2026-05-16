"""FastAPI server for the pic-to-bin web app.

Endpoints
---------
GET  /                          → static home.html (marketing / instructions)
GET  /app                       → static index.html (the Lit app)
GET  /download/fusion-addin.zip → zipped Fusion 360 script+add-in for manual install
GET  /static/{path}             → static assets (JS, CSS, vendored Lit)
POST /jobs                      → create job (multipart: images + JSON params; 30 MiB/photo, 8 photos, 120 MiB total)
GET  /jobs/{id}                 → job summary (status, artifact URLs)
GET  /jobs/{id}/events          → SSE stream of progress events
POST /jobs/{id}/proceed         → run Phase B (bin_config.json)
POST /jobs/{id}/redo            → re-run with adjusted params
GET  /jobs/{id}/artifacts/{fn}  → download layout_preview.png / combined_layout.dxf / bin_config.json
POST /preview                   → convert a HEIC/HEIF upload to a small JPEG thumbnail (same 30 MiB limit)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from pic_to_bin.web.jobs import (
    JobManager,
    JobStatus,
    download_filename,
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_JOB,
    MAX_TOTAL_UPLOAD_BYTES,
)

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


# Sane ranges for numeric pipeline parameters (used by _validate_params).
# These are intentionally wider than the "typical" UI hints so power users
# are not blocked, but still prevent 9999-unit bins or 999 mm tolerances
# that would OOM or produce useless output.
_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "tolerance": (-5.0, 5.0),
    "axial_tolerance": (0.0, 10.0),
    "phone_height": (50.0, 2000.0),
    "gap": (0.0, 20.0),
    "bin_margin": (0.0, 20.0),
    "mask_erode": (0.0, 2.0),
    "display_smooth_sigma": (0.0, 10.0),
    "straighten_threshold": (0.0, 90.0),
    "max_refine_iterations": (0, 20),
    "max_concavity_depth": (0.0, 20.0),
    "max_units": (1, 12),
    "min_units_x": (1, 12),
    "min_units_y": (1, 12),
    "min_units_z": (1, 12),
    "height_units": (1, 12),
}


def _validate_params(p: dict) -> None:
    """Raise HTTPException(400) on obviously bad parameter values."""
    # paper_size
    if "paper_size" in p:
        ps = str(p["paper_size"]).lower()
        if ps not in {"a4", "a5", "letter", "legal"}:
            raise HTTPException(400, f"invalid paper_size: {ps}")

    # tool_heights: accept list[float] or dict[str,int]=height (per-image mapping)
    th = p.get("tool_heights")
    if th is not None and th not in ("", {}):
        heights: list[float] = []
        try:
            if isinstance(th, (list, tuple)):
                heights = [float(x) for x in th if x not in (None, "")]
            elif isinstance(th, dict):
                heights = [float(v) for v in th.values() if v not in (None, "")]
        except (TypeError, ValueError):
            raise HTTPException(400, "tool_heights contains non-numeric values")
        for h in heights:
            if not (1.0 <= h <= 200.0):
                raise HTTPException(400, f"tool_height {h} mm out of range [1,200]")

    # numeric ranges — tolerate empty string / null (means "auto" or "use default")
    # for fields like axial_tolerance, height_units, phone_height, etc.
    for key, (lo, hi) in _PARAM_RANGES.items():
        if key in p:
            raw = p[key]
            if raw is None:
                continue
            s = str(raw).strip()
            if s == "":
                continue  # "auto" / not specified by user
            try:
                val = float(s)
            except (TypeError, ValueError):
                raise HTTPException(400, f"{key} must be a number")
            if not (lo <= val <= hi):
                raise HTTPException(400, f"{key}={val} out of range [{lo},{hi}]")

    # sam_corrective_points is a dict of lists (one per input stem).
    # We only do a shallow type check here; deeper validation of the
    # point objects happens later in _resolve_corrective_points.
    scp = p.get("sam_corrective_points")
    if scp is not None and not isinstance(scp, dict):
        raise HTTPException(400, "sam_corrective_points must be an object")


def create_app(
    jobs_root: Path,
    ttl_hours: float = 24.0,
    anthropic_api_key: Optional[str] = None,
    enable_llm: bool = False,
) -> FastAPI:
    job_manager = JobManager(
        jobs_root=jobs_root,
        ttl_hours=ttl_hours,
        anthropic_api_key=anthropic_api_key,
        enable_llm=enable_llm,
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

    # --- Security headers (defence in depth; NGINX/Apache should also set them) ---
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        # Clickjacking / MIME / referrer / permissions
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )
        # CSP tuned for Lit + importmap + SSE + data: previews.
        # When you run `python -m pic_to_bin.web.vendor_lit` the only external
        # script is gone; you can then drop 'unsafe-inline' for scripts if you
        # also externalise the importmap into a tiny static file.
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' ws: wss:; "
            "font-src 'self' data:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        response.headers.setdefault("Content-Security-Policy", csp)
        # HSTS is best set by the terminating TLS proxy (Apache/NGINX) with
        # a long max-age + preload. We set a short one here only as a hint.
        if request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=86400; includeSubDomains",
            )
        return response

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

    def _serve_html(filename: str, assets: tuple[str, ...]) -> HTMLResponse:
        # Cache-bust the JS/CSS by appending each file's mtime as a query
        # string. The browser caches them happily when unchanged but
        # re-fetches as soon as we edit the source — no manual hard-refresh.
        path = STATIC_DIR / filename
        html_text = path.read_text(encoding="utf-8")
        for asset in assets:
            asset_path = STATIC_DIR / asset
            if asset_path.exists():
                v = int(asset_path.stat().st_mtime)
                html_text = html_text.replace(
                    f"/static/{asset}", f"/static/{asset}?v={v}"
                )
        return HTMLResponse(
            html_text, headers={"cache-control": "no-cache"}
        )

    @app.get("/", response_class=HTMLResponse)
    async def home() -> HTMLResponse:
        # Public landing page — explains the workflow, links to /app for
        # the actual tool. No Lit dependency, so this stays fast on first
        # load and renders before any third-party CDN has a chance to fail.
        return _serve_html("home.html", ("styles.css",))

    @app.get("/app", response_class=HTMLResponse)
    async def app_page() -> HTMLResponse:
        return _serve_html("index.html", ("app.js", "styles.css"))

    @app.get("/download/fusion-addin.zip")
    async def download_fusion_addin() -> Response:
        # Bundle the Fusion 360 script + add-in into a ZIP so hosted users
        # (who don't have the Python package installed) can drop it into
        # their Fusion API folders by hand. Built in-memory each call so
        # it always reflects the currently-deployed code — no stale
        # committed ZIP to remember to refresh.
        return Response(
            content=_build_fusion_addin_zip(),
            media_type="application/zip",
            headers={
                "content-disposition":
                    'attachment; filename="pic-to-bin-fusion.zip"',
                "cache-control": "no-cache",
            },
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

        # Basic server-side range / type validation (authoritative even if
        # the frontend is bypassed). Rejects obviously malicious or typo'd
        # values before they reach the heavy pipeline or get persisted.
        _validate_params(params_dict)

        if len(images) > MAX_IMAGES_PER_JOB:
            raise HTTPException(
                413, f"too many images: {len(images)} (max {MAX_IMAGES_PER_JOB})"
            )

        files: list[tuple[str, bytes]] = []
        total_bytes = 0
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
            if len(data) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    413,
                    f"{upload.filename} is {len(data) // (1024*1024)} MiB "
                    f"(max {MAX_IMAGE_BYTES // (1024*1024)} MiB). "
                    "Modern phone photos at default resolution are fine; "
                    "avoid RAW or ProRAW modes.",
                )
            total_bytes += len(data)
            if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
                raise HTTPException(
                    413,
                    f"total upload exceeds {MAX_TOTAL_UPLOAD_BYTES // (1024*1024)} MiB limit",
                )
            files.append((upload.filename or f"image{ext}", data))

        job = job_manager.create_job(
            params_dict,
            files,
            max_image_bytes=MAX_IMAGE_BYTES,
            max_images=MAX_IMAGES_PER_JOB,
            max_total_bytes=MAX_TOTAL_UPLOAD_BYTES,
        )
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
        _validate_params(new_params)
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
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(
                413,
                f"HEIC preview file is {len(data) // (1024*1024)} MiB "
                f"(max {MAX_IMAGE_BYTES // (1024*1024)} MiB)",
            )
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
    async def unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s: %s", request.url.path, exc)
        # Never leak Python tracebacks or local paths to the public.
        # Support staff can correlate via the timestamp + job UUID in logs.
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error. Please try again later."},
        )

    return app


# ---------------------------------------------------------------------------
# Fusion add-in ZIP bundle
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).parent.parent  # …/pic_to_bin
_FUSION_SCRIPT_DIR = _PACKAGE_ROOT / "pic_to_bin_script"
_FUSION_ADDIN_DIR = _PACKAGE_ROOT / "pic_to_bin_addin"

_FUSION_INSTALL_TXT = """\
Pic-to-Bin — Fusion 360 add-in install
=======================================

This ZIP contains the script + add-in form of the Pic-to-Bin Fusion 360
plugin. After the one-time install you can build a Gridfinity bin from
the web app's bin_config.json in a single click.

Inside the ZIP
--------------
  Scripts/pic_to_bin/   ← put inside <Fusion API>/Scripts/
  AddIns/pic_to_bin/    ← put inside <Fusion API>/AddIns/

Where is the Fusion API directory?
-----------------------------------
  Windows : %APPDATA%\\Autodesk\\Autodesk Fusion 360\\API
            (paste that into the File Explorer address bar)
  macOS   : ~/Library/Application Support/Autodesk/Autodesk Fusion 360/API

Install (one-time)
------------------
  1. Open the Fusion API folder above. You should see Scripts/ and
     AddIns/ folders inside it (Fusion creates them on first run; if
     they don't exist, create them).
  2. Copy this ZIP's Scripts/pic_to_bin folder into <API>/Scripts/
     (so the path ends with .../Scripts/pic_to_bin/pic_to_bin.py).
  3. Copy this ZIP's AddIns/pic_to_bin folder into <API>/AddIns/
     (so the path ends with .../AddIns/pic_to_bin/pic_to_bin.py).
  4. Launch Fusion 360. Press Shift+S → Add-Ins tab → select
     'pic_to_bin' → click Run. Tick "Run on Startup" so the button
     shows up every session.
  5. In a Design workspace, the new "Gridfinity Pic-to-Bin" button
     appears under Solid → Create.

Use it
------
  1. In the Pic-to-Bin web app, finish a job and download the
     bin_config.json file.
  2. In Fusion, click Solid → Create → Gridfinity Pic-to-Bin.
  3. Select the bin_config.json. Fusion builds the parametric bin
     in seconds — every phase lives in its own named timeline group
     so you can keep editing afterward.
  4. File → 3D Print (or Export → STL) to feed your slicer.

Upgrades
--------
  This ZIP always reflects the currently-deployed server. To upgrade,
  re-download from the web app's Step 0 link and overwrite the two
  pic_to_bin folders.

Source: https://github.com/justinhardin/gridfinity-pic-to-bin
"""


def _build_fusion_addin_zip() -> bytes:
    """Bundle pic_to_bin_script + pic_to_bin_addin into a self-installable ZIP.

    Matches the layout that ``fusion_install.py`` would write into the
    Fusion API folder, but boxed up for users who don't have the Python
    package installed locally. The shared ``_bin_builder.py`` is copied
    into both subfolders so each one is self-contained — same shape
    ``fusion_install.install()`` produces.
    """
    if not _FUSION_SCRIPT_DIR.is_dir() or not _FUSION_ADDIN_DIR.is_dir():
        # Should only happen in odd dev installs; surfacing as 500 is fine.
        raise RuntimeError(
            f"Fusion bundle dirs not found at {_FUSION_SCRIPT_DIR} / "
            f"{_FUSION_ADDIN_DIR}"
        )

    buf = io.BytesIO()
    skip_dirs = {"__pycache__", ".vscode"}

    def add_tree(src_dir: Path, arc_prefix: str) -> None:
        for entry in sorted(src_dir.rglob("*")):
            if any(part in skip_dirs or part.startswith(".")
                   for part in entry.relative_to(src_dir).parts):
                continue
            if entry.is_file():
                rel = entry.relative_to(src_dir).as_posix()
                zf.writestr(f"{arc_prefix}/{rel}", entry.read_bytes())

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("INSTALL.txt", _FUSION_INSTALL_TXT)
        add_tree(_FUSION_SCRIPT_DIR, "Scripts/pic_to_bin")
        # The add-in needs its own copy of _bin_builder.py (mirrors what
        # fusion_install.py does on local installs).
        add_tree(_FUSION_ADDIN_DIR, "AddIns/pic_to_bin")
        builder = _FUSION_SCRIPT_DIR / "_bin_builder.py"
        if builder.exists():
            zf.writestr(
                "AddIns/pic_to_bin/_bin_builder.py",
                builder.read_bytes(),
            )

    return buf.getvalue()


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
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        default=os.environ.get("PIC_TO_BIN_ENABLE_LLM", "").lower() in ("1", "true", "yes"),
        help="Enable the Anthropic-powered 'check with LLM' feature "
             "(costs real money per call; default OFF for public sites). "
             "Only honoured when ANTHROPIC_API_KEY is also set.",
    )
    args = parser.parse_args()

    jobs_root = Path(args.jobs_dir).resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)
    logger.info("Jobs directory: %s", jobs_root)

    # LLM is *opt-in only* for public deployments. Never enable on a
    # server reachable from the internet unless you are willing to pay
    # per-transaction Anthropic bills and accept the extra outbound
    # dependency.
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or None
    enable_llm = bool(args.enable_llm)
    if enable_llm and anthropic_api_key:
        logger.info("LLM fit-check ENABLED (ANTHROPIC_API_KEY present + --enable-llm)")
    elif anthropic_api_key and not enable_llm:
        logger.warning(
            "ANTHROPIC_API_KEY is set but --enable-llm was not passed "
            "(or PIC_TO_BIN_ENABLE_LLM != 1). LLM features are DISABLED. "
            "This is the correct default for any public-facing instance."
        )
    else:
        logger.info("LLM fit-check disabled (no key or --enable-llm not used)")

    # The factory closure captures jobs_root so the app instance is rebuilt
    # cleanly on reload.
    def factory():
        return create_app(
            jobs_root,
            ttl_hours=args.job_ttl_hours,
            anthropic_api_key=anthropic_api_key,
            enable_llm=enable_llm,
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
