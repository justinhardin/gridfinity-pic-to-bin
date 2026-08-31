# `pic_to_bin/web/static/` — frontend assets

Lit components + plain CSS, served as static files by the FastAPI
backend. The whole UI is one file (`app.js`) so there's no bundler in
the loop.

| File | Purpose |
|------|---------|
| `index.html` | Page shell. Holds the import map, which points at the vendored `/static/lit-all.min.js`. The `<script src="/static/app.js">` is rewritten by the server's `index()` route to include an mtime cache-buster. |
| `app.js` | Lit components: `pic-app` (root, owns modal + history), `pic-form`, `pic-progress`, `pic-preview`, `pic-downloads`. Also the `FIELD_INFO` map driving the (i) info modals and the heic2any-replacement that POSTs HEIC files to `/preview` for thumbnailing. |
| `styles.css` | All styles. Variables at the top (`--bg`, `--accent`, etc.). |

| `lit-all.min.js` | Vendored Lit 3.2.1 (29 kB, the official pre-bundled `lit/dist` build, BSD-3-Clause). Committed and shipped in the wheel so the browser never contacts a CDN — the server's CSP allows same-origin scripts only. Refresh it with `python -m pic_to_bin.web.vendor_lit`. It exports Lit's core and directives but **not** `lit/decorators.js`; `app.js` uses `static properties` instead. |
