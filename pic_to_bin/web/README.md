# `pic_to_bin/web/` — FastAPI + Lit web wrapper

Browser-based UI around `pipeline.run_pipeline()`. Designed for
multi-user from day one: per-job UUID directories, a GPU semaphore
serializing SAM2 across concurrent submissions, SSE-streamed progress,
and a two-phase preview-then-proceed flow that lets users iterate on
parameters cheaply (re-doing the layout reuses cached per-tool DXFs).

| File | Purpose |
|------|---------|
| `server.py` | FastAPI routes + uvicorn launcher (`pic-to-bin-web`). Endpoints: `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/events` (SSE), `POST /jobs/{id}/proceed`, `POST /jobs/{id}/redo`, `GET /jobs/{id}/artifacts/{name}`, `POST /preview` (HEIC thumbnail conversion). |
| `jobs.py` | `JobManager` — UUID registry, ThreadPoolExecutor, GPU semaphore, async SSE event fan-out, TTL sweep, and the `sanitize_part_name`/`download_filename` helpers used to rename downloads. |
| `vendor_lit.py` | `python -m pic_to_bin.web.vendor_lit` — downloads `lit-all.min.js` into `static/` and rewrites the import map for an offline-deployable build. |
| `__init__.py` | Re-exports `create_app` and the cli entry point. |
| `static/` | Frontend (Lit components + CSS). See `static/README.md`. |

Per-job filesystem layout under `web_jobs/<uuid>/`:

```
inputs/                       <- uploaded photos
<photo-stem>/                 <- per-photo trace + mask outputs
combined_layout.dxf           <- packed layout
layout_actual_size.pdf|.svg   <- 1:1 fit-test
layout_preview.png            <- low-res preview
bin_config.json               <- written after Phase B
```

To run: `pip install -e ".[web]"` then `pic-to-bin-web --port 8000`.
