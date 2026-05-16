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

## Public Hosting & Security Hardening

The web app is safe for internet exposure **only** when fronted by a reverse proxy (NGINX or Apache) and launched with the secure defaults below. The uvicorn process must **never** be directly reachable from the public internet.

### Recommended public launch (localhost only)

```bash
# Never set ANTHROPIC_API_KEY or use --enable-llm on a public box.
pic-to-bin-web \
  --host 127.0.0.1 \
  --port 8000 \
  --job-ttl-hours 6 \
  --log-level warning
```

- `--enable-llm` / `PIC_TO_BIN_ENABLE_LLM=1` is **off by default**. The "Check with LLM" button and `/llm_evaluate` endpoint are completely disabled (returns 503). This eliminates per-transaction Anthropic costs and any outbound network calls from the server process.
- Short TTL (6 h) + 30 MiB / 8 photo / 120 MiB total hard limits prevent disk-filling attacks.
- All numeric parameters are range-checked on the server before the pipeline runs.

### NGINX recommended snippet (TLS termination + rate limiting)

```nginx
# /etc/nginx/sites-available/pic-to-bin
server {
    listen 443 ssl http2;
    server_name your-domain.example.com;

    # TLS certs (certbot --nginx or manual)
    ssl_certificate     /etc/letsencrypt/live/your-domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain/privkey.pem;
    ssl_trusted_certificate /etc/letsencrypt/live/your-domain/chain.pem;

    # Security headers (NGINX wins; app also sets a subset)
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;
    add_header Permissions-Policy "camera=(), microphone=(), geolocation=()" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' ws: wss:; frame-ancestors 'none';" always;

    # Size & rate limits (protects before the app sees the request)
    client_max_body_size 150m;          # 8×30 MiB + overhead
    limit_req_zone $binary_remote_addr zone=job:10m rate=8r/m;   # 8 jobs per minute per IP
    limit_req_zone $binary_remote_addr zone=preview:10m rate=30r/m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;   # long-running SAM2 jobs
    }

    location = /jobs {
        limit_req zone=job burst=5 nodelay;
        proxy_pass http://127.0.0.1:8000;
        # ... same headers as above
    }

    location /preview {
        limit_req zone=preview burst=10 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }

    # Static assets & downloads can be cached aggressively
    location /static/ {
        proxy_pass http://127.0.0.1:8000;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
```

### Apache notes

Use `mod_proxy`, `mod_proxy_http`, `mod_ratelimit`, `LimitRequestBody 157286400` (150 MiB), and the same security headers via `Header always set ...`.

### What the hardening protects against

- **Resource exhaustion / DoS**: 30 MiB per photo, 8 photos max, 120 MiB total per job. NGINX `client_max_body_size` + app-level 413 before any disk write. GPU/CPU work only happens for valid small jobs.
- **LLM cost abuse**: Completely disabled on public instances. Even an env var leak does nothing unless `--enable-llm` is also passed.
- **Path traversal / file disclosure**: UUID job dirs + strict whitelists for artifacts, inputs, and overlays. No user-controlled paths reach the filesystem.
- **Information leaks**: Exception handler returns only a generic message; full tracebacks stay in logs.
- **Clickjacking / MIME / XSS / CSP bypass**: Strong headers + CSP (tuned for Lit). Vendoring Lit (`python -m pic_to_bin.web.vendor_lit`) removes the esm.sh dependency entirely.
- **CSRF**: No cookie-based authentication or session state. All actions are job-UUID based. Rate limiting + size limits make "drive-by job creation" harmless.

### Monitoring after go-live

- Watch disk usage of `web_jobs/` (should stay < a few GB even with traffic).
- `journalctl -u pic-to-bin` for "LLM fit-check disabled" at startup.
- NGINX access logs for 413/429 rates.
- GPU utilisation (the built-in semaphore already serialises SAM2).

With these settings you can host the service publicly and sleep soundly.

## Vendoring Lit (supply-chain & CSP win)

```bash
python -m pic_to_bin.web.vendor_lit
```

This downloads `lit-all.min.js` into `static/` and rewrites the import map. After this step the only HTTP requests the browser makes are to your own origin (plus the SSE and artifact downloads). Perfect for strict CSP or air-gapped deploys.
