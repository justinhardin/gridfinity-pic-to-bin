# `pic_to_bin/web/static/` — frontend assets

Lit components + plain CSS, served as static files by the FastAPI
backend. The whole UI is one file (`app.js`) so there's no bundler in
the loop.

| File | Purpose |
|------|---------|
| `index.html` | Page shell. Holds the import map (Lit from esm.sh by default; rewritten by `vendor_lit.py` to point locally). The `<script src="/static/app.js">` is rewritten by the server's `index()` route to include an mtime cache-buster. |
| `app.js` | Lit components: `pic-app` (root, owns modal + history), `pic-form`, `pic-progress`, `pic-preview`, `pic-downloads`. Also the `FIELD_INFO` map driving the (i) info modals and the heic2any-replacement that POSTs HEIC files to `/preview` for thumbnailing. |
| `styles.css` | All styles. Variables at the top (`--bg`, `--accent`, etc.). |

After `python -m pic_to_bin.web.vendor_lit`, this directory also
contains `lit-all.min.js` and `lit-decorators.js` (gitignored, both
~80 kB).
