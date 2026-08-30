# Gridfinity Pic-to-Bin

Generate 3D-printable gridfinity bins with custom tool cutouts from phone camera photos.

A printed ArUco marker template handles perspective correction and automatic scale calibration. The rest of the pipeline — SAM2 segmentation, vectorization, layout packing, and Fusion 360 bin generation — runs automatically.

## Contents

- [Quick start](#quick-start)
- [Installation](#installation)
  - [Installing pipx](#installing-pipx)
  - [1. The core code](#1-the-core-code)
  - [2. The web app](#2-the-web-app)
  - [3. The Fusion 360 add-in](#3-the-fusion-360-add-in)
  - [Where to run these commands](#where-to-run-these-commands)
  - [pipx, venv, or plain pip](#pipx-venv-or-plain-pip)
  - [Upgrading and uninstalling](#upgrading-and-uninstalling)
  - [From source](#from-source)
- [Step 1: Print the template](#step-1-print-the-template)
- [Step 2: Take the photo](#step-2-take-the-photo)
- [Step 3: Run the pipeline](#step-3-run-the-pipeline)
  - [All `pic-to-bin` options](#all-pic-to-bin-options)
  - [Bin sizing logic](#bin-sizing-logic)
  - [Output files](#output-files)
- [Web app (browser frontend)](#web-app-browser-frontend)
- [Running individual steps](#running-individual-steps)
- [Fusion 360 integration](#fusion-360-integration)
- [Troubleshooting](#troubleshooting)
- [Running tests](#running-tests)
- [License](#license)

---

## Quick start

1. **[Install the pipeline](#installation)** —
   `pipx install "gridfinity-pic-to-bin[web]"`. One command, and it covers all
   three pieces: the CLI, the [web app](#2-the-web-app), and the bundled
   [Fusion 360 add-in](#3-the-fusion-360-add-in).
2. **[Print the ArUco template](#step-1-print-the-template)** —
   `generate-phone-template --paper-size letter`. One-time setup; print at
   exactly 100% scale, no fit-to-page.
3. **[Photograph the tool on the template](#step-2-take-the-photo)** — lay the
   tool in the dotted zone, shoot from above with all 8 markers in frame, and
   measure the tool's depth with calipers.
4. **[Run the pipeline](#step-3-run-the-pipeline)** —
   `pic-to-bin photo.jpg --tool-height 17`. Produces `bin_config.json`, a
   combined DXF, and [1:1 fit-test printouts](#output-files) you can lay the
   real tool on before printing.
5. **[Build the bin in Fusion 360](#fusion-360-integration)** — click
   **Solid → Create → Gridfinity Pic-to-Bin**, then export STL or STEP and
   slice it.

Prefer a browser to a terminal? `pic-to-bin-web` replaces step 4 with
drag-and-drop upload, live progress, and a layout preview — see
[Web app](#web-app-browser-frontend).

---

## Installation

**Recommended — one command:**

```bash
pipx install "gridfinity-pic-to-bin[web]"
```

[pipx](https://pipx.pypa.io/) is the standard installer for Python
*applications* — as opposed to libraries you `import`. It builds a private
virtual environment for this package and puts its commands on your `PATH`, so
there is no venv to create, activate, or remember, and nothing this package
drags in can disturb the rest of your system. Don't have pipx yet? See
[Installing pipx](#installing-pipx). If you would rather use a venv and plain
`pip`, that still works — see
[pipx, venv, or plain pip](#pipx-venv-or-plain-pip).

That single command covers all three pieces of the project:

| # | What | How you get it |
|---|------|----------------|
| 1 | **The core code** — the pipeline that turns photos into a bin config | Always installed |
| 2 | **The web app** — browser frontend over the same pipeline | Included, via the `[web]` extra above |
| 3 | **The Fusion 360 add-in** — turns the bin config into a solid model | Ships inside the package; run `pic-to-bin-fusion install` once to copy it into Fusion 360 |

There is no second install to run. The `[web]` extra is a superset of the
core install, and the Fusion add-in is bundled in the package itself —
`pic-to-bin-fusion install` is a file copy into Fusion 360's own add-ins
folder, not a download.

**The `[web]` part of that command does matter, though.** The web app's
*code* ships in the base package either way, so `pic-to-bin-web` appears on
your `PATH` even without the extra — but its server dependencies (`fastapi`,
`uvicorn`, …) arrive only with `[web]`, and without them the command exits
immediately with `ModuleNotFoundError: No module named 'fastapi'`. If you
already installed without the extra, add it in place:

```bash
pipx install --force "gridfinity-pic-to-bin[web]"
```

Want the CLI only, on a machine that will never serve a browser UI? Use
`pipx install gridfinity-pic-to-bin`. The extra adds only a handful of small
pure-Python packages next to the multi-GB PyTorch download that dominates the
install either way, which is why `[web]` is the recommended default.

Install from wherever you like — but **where you later run the commands does
matter**; see [Where to run these commands](#where-to-run-these-commands).

### Installing pipx

**Windows — PowerShell, *not* "Run as Administrator":**

```powershell
py -m pip install --user pipx
py -m pipx ensurepath
```

**macOS — Terminal:**

```bash
brew install pipx        # or: python3 -m pip install --user pipx
pipx ensurepath
```

`ensurepath` edits your shell profile, so **close the terminal and open a new
one** before continuing — that is what puts `pipx`, and the commands it
installs, on your `PATH`.

pipx builds each app's environment with your default Python. This package
needs **Python 3.10 or newer**; if your default is older, point pipx at a
newer one: `pipx install --python 3.12 "gridfinity-pic-to-bin[web]"`.

### 1. The core code

```bash
pipx install "gridfinity-pic-to-bin[web]"   # recommended: core + web app
pipx install gridfinity-pic-to-bin          # core only
```

Everything that actually does the work: ArUco template generation, photo
preprocessing (marker detection, perspective correction, scale calibration),
SAM2 segmentation and vectorization, layout packing, the Fusion 360 config
writer, and the add-in installer.

Install it if you are happy driving the pipeline from a terminal. It is also
the base for both other pieces, so there is no way to skip it.

It gives you these commands:

| Command | Purpose |
|---------|---------|
| `pic-to-bin` | Run the whole pipeline: photos → `bin_config.json` |
| `generate-phone-template` | Make the printable ArUco template PDF |
| `preprocess-phone` | Photo → rectified, scale-calibrated image |
| `trace-tool` | Rectified image → tool outline DXF/SVG |
| `layout-tools` | Pack tool DXFs into one bin footprint |
| `prepare-bin` | Combined layout → Fusion 360 config JSON |
| `pic-to-bin-fusion` | Install/uninstall the Fusion 360 add-in |

Dependencies (installed automatically): `ultralytics` (SAM2),
`opencv-python`, `numpy`, `ezdxf`, `potracer`, `pyclipper`, `matplotlib`,
`Pillow`, `pillow-heif`.

### 2. The web app

```bash
pipx install "gridfinity-pic-to-bin[web]"   # the same single install as above
pic-to-bin-web --port 8000                  # http://localhost:8000
```

This is not a second installation — it is the one package with its `[web]`
extra. Already installed without the extra? `pipx install --force
"gridfinity-pic-to-bin[web]"` adds it without disturbing anything else.

A FastAPI server plus a Lit frontend layered on top of the core code. It
does **not** reimplement the pipeline — it calls the same `run_pipeline()`
the CLI calls. What it adds is the interaction layer: drag-and-drop photo
upload, live step-by-step progress streamed over SSE, an in-browser layout
preview with printable fit-test downloads, a cheap Re-do loop that re-packs
the layout without re-tracing, and one-click downloads of the DXF/PDF/JSON.

Install it if:

- You want a GUI instead of memorizing CLI flags.
- You want to review the layout preview and tweak parameters interactively
  before committing to a 3D print.
- You want to serve other people. The web app is multi-user by design:
  per-job UUID directories, a GPU semaphore so concurrent jobs queue instead
  of fighting over SAM2, and a TTL sweep for old jobs.

Extra dependencies pulled in by `[web]`: `fastapi`, `uvicorn[standard]`,
`python-multipart`, `sse-starlette`, `anthropic`, `python-dotenv`. Those
dependencies are the *only* thing the extra adds — the web app's own code and
static assets are already in the base package. That is why `pic-to-bin-web`
exists on your `PATH` after a plain install but fails with
`ModuleNotFoundError: No module named 'fastapi'` until you add `[web]`.

See [Web app](#web-app-browser-frontend) below for usage, and
`pic_to_bin/web/README.md` for hosting it behind a public reverse proxy.

### 3. The Fusion 360 add-in

```bash
pic-to-bin-fusion install
```

No download and no second install — the add-in files ship inside the Python
package. This command copies them into Fusion 360's own add-ins directory
(`%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\pic_to_bin\` on Windows,
`~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns/pic_to_bin/`
on macOS). Nothing is installed into Fusion 360 itself — you need Fusion
already installed, and it only runs on Windows and macOS.

Once enabled inside Fusion, a **Gridfinity Pic-to-Bin** button appears under
**Solid → Create**. Clicking it reads a `bin_config.json` and builds the
whole parametric bin — body, stacking lip, deck, tool pockets, finger slots,
and gridfinity base pads — in a fresh document, with optional STL/STEP/PNG
export.

Install it if you want the finished 3D model. Steps #1 and #2 stop at
`bin_config.json` plus DXF/PDF files; this is the piece that turns those
into printable geometry. Skip it if you only want the 1:1 fit-test printouts,
or if you plan to import the DXF into some other CAD package yourself.

The command is idempotent — re-run it after upgrading the package to refresh
the installed copy. `pic-to-bin-fusion uninstall` removes it.

See [Fusion 360 integration](#fusion-360-integration) below for enabling the
add-in inside Fusion and what gets built.

### Where to run these commands

`pipx install` does not care which directory you are in, and because pipx
puts the commands on your `PATH` there is no environment to activate later
either. But three things that come *after* the install do care about your
current directory, so it is worth setting up one working folder now and
running everything from there:

- `pic-to-bin` writes its results to `generated/` **relative to the current
  directory**, and with no image arguments it processes every PNG/JPG in the
  current directory.
- `pic-to-bin-web` creates its `web_jobs/` directory in the current
  directory.
- Ultralytics downloads the SAM2 weights (`sam2.1_l.pt`, several hundred MB)
  on the first trace, into the directory you ran from.

So: make a dedicated folder in your home directory and run every command
from there.

**Windows — PowerShell, *not* "Run as Administrator":**

```powershell
pipx install "gridfinity-pic-to-bin[web]"   # once, from anywhere
mkdir $HOME\pic-to-bin
cd $HOME\pic-to-bin                         # then work from here
```

`$HOME\pic-to-bin` resolves to `C:\Users\<you>\pic-to-bin`.

**macOS — Terminal:**

```bash
pipx install "gridfinity-pic-to-bin[web]"   # once, from anywhere
mkdir -p ~/pic-to-bin
cd ~/pic-to-bin                             # then work from here
```

Every later session, `cd` back to that folder before running `pic-to-bin`.
With pipx there is nothing to activate — the commands are already on your
`PATH`. (If you installed into a venv instead, re-activate it first:
`.\.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on
macOS.)

**Locations to avoid on both platforms:**

| Avoid | Why |
|-------|-----|
| Windows `Desktop`/`Documents` while OneDrive backup is on; macOS `Desktop`/`Documents` while iCloud "Desktop & Documents Folders" is on | Sync fights the multi-hundred-MB SAM2 weights and the per-job output folders, and cloud-only eviction can make files disappear mid-run |
| `C:\Program Files`, `/Applications`, `/usr/local/lib` | Require admin rights; `pip` and `pipx` should never be writing there |
| `sudo pip install …` / `sudo pipx install …` | Never needed, and it can break OS tooling that depends on the system Python |
| Deeply nested Windows paths | Some tooling still trips over the 260-character `MAX_PATH` limit |

`pic-to-bin-fusion install` is the one exception to the working-folder rule:
it copies files into Fusion 360's own add-ins directory, so it can be run
from anywhere.

### pipx, venv, or plain pip

pipx is the recommended installer here because this package is an
*application* — you run `pic-to-bin` from a terminal, you don't `import
pic_to_bin` from your own code. pipx is built for exactly that case: it
creates a dedicated virtual environment behind the scenes and exposes only
the commands.

**Why isolation matters for this package in particular:**

- **It is heavy.** `ultralytics` pulls in PyTorch and torchvision — a couple
  of GB — and `ultralytics`, `opencv-python` and `matplotlib` each constrain
  the `numpy` version. Installed machine-wide with `pip`, that can break an
  unrelated project that wanted a different `numpy` or torch build.
- **Some Pythons refuse a machine-wide install outright.** Homebrew Python
  on macOS and most Linux distro Pythons mark themselves "externally
  managed" ([PEP 668](https://peps.python.org/pep-0668/)) and reject
  `pip install` outside a virtual environment with
  `error: externally-managed-environment`. pipx sidesteps this entirely;
  plain `pip` does not.
- **Clean uninstall.** `pipx uninstall gridfinity-pic-to-bin` takes all ~100
  transitive dependencies with it. Undoing a machine-wide `pip install`
  means chasing them individually.
- **Nothing to activate.** A venv only works once you remember to activate
  it, in every new terminal, which is easy to get wrong when you also have
  to `cd` to a working folder. pipx commands are simply on your `PATH`.

**When plain `pip` in a venv is still the right call:**

- **You are working on the source.** Editable installs (`pip install -e .`)
  and the `[dev]` test dependencies belong in a venv — see
  [From source](#from-source).
- **You want to `import pic_to_bin` from your own scripts.** pipx's
  environment is deliberately not on your import path; a venv is.
- **You need a specific PyTorch build**, e.g. a CUDA wheel from
  `download.pytorch.org`. That is easier to control in a venv. It is still
  possible under pipx via
  `pipx runpip gridfinity-pic-to-bin install torch --index-url …`, just
  fiddlier.

For a venv install, replace the `pipx install` line with the usual three:

```bash
python3 -m venv .venv          # Windows: py -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install "gridfinity-pic-to-bin[web]"
```

On Windows, if `Activate.ps1` is blocked by execution policy, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry.

Further reading:

- [pipx documentation](https://pipx.pypa.io/)
- [Python docs — `venv`](https://docs.python.org/3/library/venv.html)
- [Python Packaging User Guide — installing packages with pip and virtual
  environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

### Upgrading and uninstalling

```bash
pipx upgrade gridfinity-pic-to-bin      # keeps the extras you installed with
pipx uninstall gridfinity-pic-to-bin    # removes the package and every dependency
```

pipx records the exact spec you installed — including `[web]` — so
`pipx upgrade` keeps the web app. To change your mind about the extra, use
`pipx install --force` with the spec you want.

Two things live outside the package and are not touched by either command:

- The Fusion 360 add-in copy. Re-run `pic-to-bin-fusion install` after an
  upgrade to refresh it, and `pic-to-bin-fusion uninstall` before removing
  the package.
- The SAM2 weights (`sam2.1_l.pt`) and the `generated/` and `web_jobs/`
  folders in your working directory. Delete those by hand if you want the
  disk space back.

### From source

pipx installs the tool for *using*; to work on the code itself, use a venv
and an editable install. Clone wherever you normally keep code —
`C:\Users\<you>\source\` on Windows, `~/src` on macOS — again avoiding
cloud-synced folders:

```bash
git clone https://github.com/justinhardin/gridfinity-pic-to-bin.git
cd gridfinity-pic-to-bin
python3 -m venv .venv          # optional; Windows: py -m venv .venv
source .venv/bin/activate      # optional; Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[web]"        # or ".[dev]" / "." for core only
```

An editable install still runs from anywhere, so you can keep the checkout
here and run `pic-to-bin` from your working folder — or run it from the
checkout and let `generated/` land next to the source (it is gitignored).

---

## Step 1: Print the template

Generate and print a template for your paper size. **Print at exactly 100% scale — no fit-to-page.**

```bash
generate-phone-template --paper-size letter --output template.pdf
```

```
Options:
  --paper-size {a4,letter,legal}   Paper size (default: a4)
  --output PATH                    Output PDF path (default: phone_template_<size>.pdf)
```

The template places 8 ArUco markers (IDs 0–7) — 4 corners and 4 edge midpoints — around a dotted placement zone. The markers are 20 mm squares with ~20 mm margins from the paper edge.

**Placement zone sizes by paper:**

| Paper   | Placement zone (W × H) |
|---------|------------------------|
| A4      | 130 × 217 mm           |
| Letter  | 136 × 199 mm           |
| Legal   | 136 × 275 mm           |

---

## Step 2: Take the photo

1. Lay the printed template on a flat surface.
2. Place the tool inside the dotted placement zone.
3. Photograph from above. Moderate angles are fine — perspective is corrected automatically.
4. All 8 markers should be visible (3 minimum, 8 ideal).
5. Measure the tool's **depth** with calipers — you'll need it for `--tool-height`.

**Lighting tips:** Use diffuse overhead light. Avoid harsh shadows across the markers. Do not cover markers with the tool.

**Format:** JPEG, PNG, or HEIC/HEIF (iPhone) are all supported.

---

## Step 3: Run the pipeline

### Single tool

```bash
pic-to-bin photo.jpg --tool-height 17
```

### Multiple tools (one photo each)

```bash
pic-to-bin a.jpg b.jpg --tool-height 0=17 --tool-height 1=14
```

Tool indices correspond to image order. Each tool's DXF is traced separately and packed into one bin.

### Specifying paper size

```bash
pic-to-bin photo.jpg --tool-height 17 --paper-size a4
```

The paper size must match what you printed. Default is `letter`.

### Shallow drawer (no stacking lip)

```bash
pic-to-bin photo.jpg --tool-height 17 --stacking false
```

Drops the 4.4 mm stacking lip for shorter bins in shallow drawers. Pocket depth is unchanged.

### All `pic-to-bin` options

```
positional:
  images                        Photo files to process (default: all PNG/JPG in cwd)

required:
  --tool-height VALUE           Tool depth in mm. Use INDEX=VALUE per tool
                                (e.g. --tool-height 0=17 --tool-height 1=14)

optional:
  --paper-size {a4,letter,legal}  Template paper size (default: legal)
  --tolerance MM                  Extra clearance on top of a 2 mm baseline
                                  (default: 0 = 2 mm physical clearance).
                                  Positive = looser fit, negative = tighter,
                                  -2 = exact-trace match.
  --axial-tolerance MM            Extra clearance only along the tool's
                                  principal axis (default: 'auto'; 2 mm
                                  floor + taper-proportional bonus).
                                  Compensates for SAM2 length under-
                                  detection.
  --phone-height MM               Camera height above template, mm (default: 480).
                                  Drives the parallax-compensation scale-down.
  --gap MM                        Minimum gap between tools in layout, mm (default: 3.0)
  --bin-margin MM                 Extra clearance from tool extent to bin wall (default: 0)
  --min-units-x N                 Minimum X grid size in units (default: 1)
  --min-units-y N                 Minimum Y grid size in units (default: 1)
  --min-units-z N                 Minimum Z grid size in height units (default: 1).
                                  Floor on the auto height; ignored when
                                  --height-units is set.
  --max-units N                   Max gridfinity grid size per axis (default: 7)
  --height-units N                Force bin height in gridfinity units (default: auto)
  --stacking BOOL                 Generate stacking lip (default: true). Set
                                  false for a shorter bin without the lip.
  --slots BOOL                    Generate finger-access slots (default: true)
  --output-dir DIR                Output directory (default: generated/)
  --straighten-threshold DEG      Max degrees to auto-straighten trace (default: 45, 0=off)
  --max-refine-iterations N       SAM2 cleanup iterations (default: 5)
  --max-concavity-depth MM        Max acceptable concavity loss, mm (default: 3.0)
  --mask-erode MM                 Post-SAM mask erosion (default: 0). Use 0.3-0.5
                                  only if your photos have a clear shadow halo.
  --sam-model WEIGHTS             SAM2 model file (default: sam2.1_l.pt)
  --skip-trace                    Skip tracing, reuse existing DXFs in generated/
```

### Bin sizing logic

- The bin auto-sizes to the smallest gridfinity unit count that fits the tool: `ceil((tool_height + 1mm) / 7mm)` height units.
- The pocket floor sits 1 mm above the bin floor.
- The deck rises to half the tool's height — the upper half of the tool stands proud for finger access; the lower half is buried in the pocket.
- The combined cutout (pocket + finger slot) is centered in the bin floor; slack from rounding up to whole gridfinity units is distributed evenly on all four sides.

### Output files

```
generated/
  <stem>/
    <stem>_rectified.png       Perspective-corrected image (scanner equivalent)
    <stem>_trace.dxf           Tool outline DXF (inner + tolerance + finger slot)
    <stem>_trace.svg           SVG preview of the trace
  combined_layout.dxf          All tools packed into a bin footprint
  layout_preview.png           Screen-viewable layout preview (matplotlib, 150 DPI)
  layout_actual_size.pdf       1:1 scale fit-test drawing (PDF page = bin footprint)
  layout_actual_size.svg       1:1 scale fit-test drawing (SVG width/height in mm)
  bin_config.json              Fusion 360 input config
```

The `layout_actual_size.pdf` and `.svg` files are sized to the bin's exact
mm dimensions — print at "Actual size" / 100% scale (NOT "Fit to page") and
lay your real tool on top to verify the fit before committing to a 3D print.

After Fusion runs:

```
generated/
  gridfinity_bin.stl
  gridfinity_bin.step
  gridfinity_bin_preview.png  Viewport screenshot of the finished bin
```

---

## Web app (browser frontend)

A FastAPI + Lit web wrapper exposes the same pipeline through a browser. Multi-user
ready: per-job UUID directories, GPU semaphore around SAM2 so concurrent submissions
queue rather than fight over the GPU, SSE-streamed progress.

```bash
pipx install "gridfinity-pic-to-bin[web]"     # from a checkout: pip install -e ".[web]"
pic-to-bin-web --port 8000
```

Open http://localhost:8000, drag in a photo, fill in the tool height, watch
the step tracker, review the layout preview (with print-at-actual-size PDF /
SVG downloads to test fit), then click Proceed to generate the bin config.

The browser back button navigates between screens (form → progress → preview
→ downloads). The form fields all have an `(i)` info button next to their
label that opens a modal with a multi-paragraph explanation.

To replace the default `esm.sh` Lit import with a vendored local copy:

```bash
python -m pic_to_bin.web.vendor_lit
```

---

## Running individual steps

You can run each stage of the pipeline independently.

### Preprocess a photo

Detect markers, correct perspective, and save the rectified image:

```bash
preprocess-phone photo.jpg --paper-size letter --output-dir generated/photo
```

```
positional:
  image                         Phone photo file

optional:
  --paper-size {a4,letter,legal}  Template paper size (default: a4)
  --output-dir DIR                Output directory (default: generated/<stem>)
```

Outputs `<stem>_rectified.png` at the computed effective DPI (typically 100–250).

### Trace a rectified image

Run SAM2 segmentation + vectorization on a rectified image:

```bash
trace-tool rectified.png --dpi 150
```

### Pack tool DXFs into a layout

```bash
layout-tools tool1.dxf tool2.dxf --gap 3 --max-units 5
```

### Generate the Fusion 360 config

```bash
prepare-bin generated/combined_layout.dxf --tool-height 17
```

---

## Fusion 360 integration

`pic-to-bin` ships in two flavors for Fusion 360 — a **toolbar add-in** (recommended) and a classic **script**. One install command sets up both:

```bash
pic-to-bin-fusion install
```

This copies the add-in to `…/API/AddIns/pic_to_bin/` and the script to `…/API/Scripts/pic_to_bin/`, sharing the build code (`_bin_builder.py`) between them.

### Add-in (recommended) — toolbar button

1. Open Fusion 360.
2. Press **Shift+S → Add-Ins tab**.
3. Select **pic_to_bin → Run** (toggle **Run on Startup** to keep the button available every session).
4. In a Design workspace, the **Solid > Create** panel now contains a **Gridfinity Pic-to-Bin** button.
5. Click the button. The script auto-loads `<project>/generated/bin_config.json` if it exists; otherwise a file dialog opens defaulting to your Desktop.
6. Bin gets built, exported as STL + STEP, and a viewport screenshot is saved alongside `bin_config.json`.

### Script form (alternate)

If you prefer the classic Scripts dialog:

1. Press **Shift+S → Scripts tab**.
2. Select **pic_to_bin → Run**.

The behavior is identical to the add-in button.

### What gets built

The bin is generated in a fresh document with these timeline groups for easy navigation:

- **Bin Body** — outer rectangular block. Appearance: ABS (White).
- **Stacking Lip** (if enabled) — solid block + corner fillets + base-profile mating cutout + 0.6 mm top recess.
- **Deck** — recessed surface around the pocket.
- **Tool Pockets** — one cut per tool.
- **Finger Slots** — one cut per tool, same floor as the pocket.
- **Base Pads** — gridfinity baseplate-mating geometry, one Join extrude for all wide pads, one for the narrow posts, plus the two chamfers.

### Reinstalling and reload

Re-running `pic-to-bin-fusion install` overwrites both folders. The script and add-in entry points reload `_bin_builder.py` from disk on every invocation, so most code changes land on the next button click without restarting Fusion. Only changes to the entry-point files themselves (`pic_to_bin_script/pic_to_bin.py` or `pic_to_bin_addin/pic_to_bin.py`) require a Stop/Run on the add-in or a Fusion restart.

### Uninstall

```bash
pic-to-bin-fusion uninstall
```

Removes both the script and the add-in.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `MarkerDetectionError: No markers detected` | Template not visible or too blurry | Ensure all markers are in frame; hold camera steadier |
| `MarkerDetectionError: Only N markers (need ≥3)` | Markers obscured or overexposed | Improve lighting; don't cover markers with tool |
| `ScaleInconsistencyError: H/V scales differ >5%` | Template not printed at 100% | Reprint with fit-to-page disabled |
| `WARNING: Low effective DPI (<100)` | Camera too far away | Hold phone closer; use higher resolution mode |
| Tools don't fit in grid | Tools too large for `--max-units` | Increase `--max-units` |
| Fusion freezes building pockets | Stale cached `_bin_builder` after editing | The reload is already wired in — just click the button again. If still stuck, restart Fusion. |
| Pocket fits too loose | Default `--tolerance 0` produces 2 mm physical clearance + ≥2 mm at each tip | Lower with `--tolerance -0.5` and/or `--axial-tolerance 1.0` |
| Pocket fits too tight at the tool's tips only | SAM2 under-detected the tapered ends | Increase `--axial-tolerance` (default 'auto', 2 mm floor) |
| Pocket fits too tight everywhere | Trace itself is short (shadow halo, parallax, mask erosion) | First try `--tolerance 1` (= 3 mm physical). If still tight, check `--phone-height` matches your shooting distance |

### Common photo issues

- **Markers partially cut off**: Keep all 8 markers visible. At 3 minimum the pipeline will run but accuracy drops.
- **Blurry markers**: Tap to focus on the template before shooting; avoid camera shake.
- **Shadows on markers**: Use diffuse or overhead lighting, not a side lamp.
- **HEIC images (iPhone)**: Supported natively — the pipeline converts automatically via `pillow-heif`.

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## License

MIT — see [LICENSE.md](LICENSE.md).
