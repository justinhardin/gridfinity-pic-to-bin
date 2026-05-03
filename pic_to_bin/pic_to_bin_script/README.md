# `pic_to_bin/pic_to_bin_script/` — Fusion 360 Script entry

Fusion 360 calls Python scripts via Tools → Scripts and Add-Ins →
Scripts. This directory is what `pic-to-bin-fusion install` copies into
Fusion's user-scripts dir.

| File | Purpose |
|------|---------|
| `pic_to_bin.py` | Script entry. Pops a file dialog (defaults to `<project>/generated/bin_config.json`, then `~/Desktop`) and calls `_bin_builder.build_bin()` on the chosen JSON. `importlib.reload(_bin_builder)` runs on every invocation so edits land without restarting Fusion. |
| `pic_to_bin.manifest` | Fusion manifest — `"type": "script"`, plus the script name. |
| `_bin_builder.py` | Shared build code (sketches, named timeline groups, ABS-white appearance, STL/STEP/PNG export). The add-in copy of this file is kept in sync by `fusion_install.py`. |

The add-in (`pic_to_bin_addin/`) wraps the same `_bin_builder.py` in a
toolbar button instead of a script-list entry.
