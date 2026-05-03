# `pic_to_bin/pic_to_bin_addin/` — Fusion 360 Add-In entry

Fusion 360 add-ins register themselves into the Fusion UI on load. This
directory is what `pic-to-bin-fusion install` copies into Fusion's
user-add-ins dir.

| File | Purpose |
|------|---------|
| `pic_to_bin.py` | Add-in entry. Registers a "Gridfinity Pic-to-Bin" button in Solid → Create. Click → file dialog → calls `_bin_builder.build_bin()`. `importlib.reload(_bin_builder)` on each click so edits land without restarting Fusion. |
| `pic_to_bin.manifest` | Fusion manifest — `"type": "addin"`. |
| `resources/` | Toolbar icons (16/32/64 px). |

Note: there's no copy of `_bin_builder.py` in this directory in the
source tree — `fusion_install.py` copies it in alongside this file when
you run the install command, so the add-in dir on disk has its own
self-contained copy after installation.
