# `pic_to_bin/pic_to_bin_addin/` — Fusion 360 Add-In

The only Fusion 360 entry point. Add-ins register themselves into the
Fusion UI on load; this directory is what `pic-to-bin-fusion install`
copies into Fusion's user add-ins folder, and what the web app ships in
its `pic-to-bin-fusion.zip` download.

| File | Purpose |
|------|---------|
| `pic_to_bin.py` | Add-in entry. Registers a "Gridfinity Pic-to-Bin" button in Solid → Create. Click → file dialog → calls `_bin_builder.build_bin()`. `importlib.reload(_bin_builder)` on each click so edits land without restarting Fusion. |
| `_bin_builder.py` | The build logic — sketches, named timeline groups, ABS-white appearance, STL/STEP/PNG export. |
| `pic_to_bin.manifest` | Fusion manifest — `"type": "addin"`. |
| `resources/` | Toolbar icons (16/32/64 px). |
