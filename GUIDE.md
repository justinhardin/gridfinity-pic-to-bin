# Gridfinity Pic-to-Bin — Full Guide

Everything the [README](README.md) leaves out: how the pipeline works, every
install path, every CLI flag, the geometry rules the bin is built from, and a
troubleshooting table for when a photo doesn't cooperate.

## Contents

- [How it works](#how-it-works)
- [Installation](#installation)
  - [Installing pipx](#installing-pipx)
  - [The three pieces](#the-three-pieces)
  - [Where to run these commands](#where-to-run-these-commands)
  - [pipx, venv, or plain pip](#pipx-venv-or-plain-pip)
  - [Upgrading and uninstalling](#upgrading-and-uninstalling)
  - [From source](#from-source)
- [Step 1: Print the template](#step-1-print-the-template)
- [Step 2: Take the photo](#step-2-take-the-photo)
- [Step 3a: The web app](#step-3a-the-web-app)
- [Step 3b: The CLI](#step-3b-the-cli)
  - [All `pic-to-bin` options](#all-pic-to-bin-options)
  - [Tolerance model](#tolerance-model)
  - [Bin sizing logic](#bin-sizing-logic)
  - [Output files](#output-files)
- [Running individual steps](#running-individual-steps)
- [Step 4: Fusion 360](#step-4-fusion-360)
  - [Installing the add-in](#installing-the-add-in)
  - [Manual install from the web app ZIP](#manual-install-from-the-web-app-zip)
  - [macOS: "cannot verify the developer"](#macos-cannot-verify-the-developer)
  - [What gets built](#what-gets-built)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## How it works

You photograph a tool lying on a printed sheet of ArUco markers. The markers
are at known millimetre positions, so detecting them tells the pipeline both
where the camera was and how big a millimetre is in the image. Everything
downstream works in real-world mm.

```
1. Print the ArUco template (one-time)
2. Place the tool on it, take a phone photo
         |
Phone preprocessing  (phone_preprocess.py)
    ArUco detection -> homography -> perspective warp -> scale
    calibration -> crop to the placement zone
         |  rectified image + effective DPI
Trace generation  (trace_tool.py, refine_trace.py, trace_export.py)
    SAM2 segmentation -> iterative mask cleanup -> potrace ->
    uniform offset -> axial PCA stretch -> Douglas-Peucker -> SVG/DXF
         |  SVG + DXF (inner trace, tolerance outline, finger slot)
Layout packing  (layout_tools.py)
    DXF reading -> rotation + mirror -> polygon collision packing ->
    combined DXF -> layout_preview.png + 1:1 layout_actual_size.pdf/.svg
         |
Bin config  (prepare_bin.py)
    center the cutout in the bin -> bin_config.json
         |
Fusion 360 add-in  (pic_to_bin_addin/)
    parametric bin body + pockets + finger slots + stacking lip +
    gridfinity base pads -> optional STL / STEP / PNG export
```

The web app is a thin FastAPI + Lit layer over the exact same
`run_pipeline()` the CLI calls — it does not reimplement any of the above.

**Why ArUco markers.** They solve two problems at once. Eight markers give up
to 32 point correspondences for a homography, which `cv2.findHomography()`
fits with RANSAC so a wrinkled sheet or a partly occluded marker gets
rejected as an outlier rather than skewing the result. And because the
markers are a known 20 mm square at known spacing, the same homography yields
the effective DPI — no reference coin, no manual scale entry.

Some detail worth knowing:

- **Detection** uses `cv2.aruco.ArucoDetector` with `DICT_4X4_50` and
  sub-pixel corner refinement (`CORNER_REFINE_SUBPIX`), with adaptive
  thresholding tuned for uneven phone lighting. Only IDs 0–7 are considered.
- **Scale cross-check**: horizontal and vertical scale are compared. A
  mismatch over 2% warns; over 5% raises `ScaleInconsistencyError`, which
  almost always means the template was printed with fit-to-page on.
- **Warping** uses `cv2.warpPerspective` with a white border fill
  (`borderValue=255,255,255`), so anything outside the template reads as
  clean white background — which is what SAM2 handles best.
- **Effective DPI** is derived from the homography inverse (pixels per mm at
  the template center) and is typically 100–250 for a phone photo. Below 100
  you get a warning; hold the camera closer.
- **Accuracy**: sub-pixel corners land within roughly 0.1 px, which over a
  160 mm marker span works out to well under 0.5 mm of scale error.

---

## Installation

**One command covers everything:**

```bash
pipx install gridfinity-pic-to-bin
```

No extras, no second step: the web app's dependencies are part of the base
package. (Earlier releases put them behind a `[web]` extra. That still
resolves, so old instructions keep working — it just no longer adds anything
the plain install lacks.)

[pipx](https://pipx.pypa.io/) is the standard installer for Python
*applications* — as opposed to libraries you `import`. It builds a private
virtual environment for this package and puts its commands on your `PATH`, so
there is no venv to create, activate, or remember, and nothing this package
drags in can disturb the rest of your system.

One feature is optional: the **"check with LLM"** fit review, which calls the
Anthropic API and bills your account per call. It is off unless you pass
`--enable-llm`, and its SDK installs separately:

```bash
pipx install --force "gridfinity-pic-to-bin[llm]"
```

Keep those quotes — zsh, the default shell on macOS, reads `[llm]` as a glob
pattern and answers `zsh: no matches found` before pipx ever runs. The quotes
are harmless in bash, so quote it everywhere.

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
newer one: `pipx install --python 3.12 gridfinity-pic-to-bin`.

### The three pieces

| # | What | How you get it |
|---|------|----------------|
| 1 | **The core code** — the pipeline that turns photos into a bin config | Always installed |
| 2 | **The web app** — browser frontend over the same pipeline | Always installed |
| 3 | **The Fusion 360 add-in** — turns the bin config into a solid model | Ships inside the package; run `pic-to-bin-fusion install` once to copy it into Fusion 360 |

There is no second install to run. `pic-to-bin` and `pic-to-bin-web` both
come from that one command, and the Fusion add-in is bundled in the package
itself — `pic-to-bin-fusion install` is a file copy into Fusion 360's own
add-ins folder, not a download.

The server dependencies (`fastapi`, `uvicorn`, …) are a handful of small
pure-Python packages next to the multi-GB PyTorch download that dominates the
install either way, so there is nothing to save by leaving them out.

**Commands the install gives you:**

| Command | Purpose |
|---------|---------|
| `pic-to-bin` | Run the whole pipeline: photos → `bin_config.json` |
| `pic-to-bin-web` | Serve the browser frontend |
| `generate-phone-template` | Make the printable ArUco template PDF |
| `preprocess-phone` | Photo → rectified, scale-calibrated image |
| `trace-tool` | Rectified image → tool outline DXF/SVG |
| `layout-tools` | Pack tool DXFs into one bin footprint |
| `prepare-bin` | Combined layout → Fusion 360 config JSON |
| `pic-to-bin-fusion` | Install/uninstall the Fusion 360 add-in |

**Dependencies** (installed automatically): `ultralytics` (SAM2),
`opencv-python`, `numpy`, `ezdxf`, `potracer`, `svgpathtools`, `pyclipper`,
`matplotlib`, `Pillow`, `pillow-heif`, plus the web app's `fastapi`,
`uvicorn[standard]`, `python-multipart`, `sse-starlette`, and
`python-dotenv`. Note that `cv2.aruco` is part of standard `opencv-python`
(4.13+) — `opencv-contrib-python` is not needed.

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
pipx install gridfinity-pic-to-bin   # once, from anywhere
mkdir $HOME\pic-to-bin
cd $HOME\pic-to-bin                  # then work from here
```

`$HOME\pic-to-bin` resolves to `C:\Users\<you>\pic-to-bin`.

**macOS — Terminal:**

```bash
pipx install gridfinity-pic-to-bin   # once, from anywhere
mkdir -p ~/pic-to-bin
cd ~/pic-to-bin                      # then work from here
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
*application* — you run `pic-to-bin` from a terminal, you don't
`import pic_to_bin` from your own code. pipx is built for exactly that case:
it creates a dedicated virtual environment behind the scenes and exposes only
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
pip install gridfinity-pic-to-bin
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
pipx upgrade gridfinity-pic-to-bin      # keeps the spec you installed with
pipx uninstall gridfinity-pic-to-bin    # removes the package and every dependency
```

pipx records the exact spec you installed — including any extras — so
`pipx upgrade` preserves it. To change the spec later (adding `[llm]`, say),
use `pipx install --force` with the one you want.

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
pip install -e .               # add ".[dev]" for the test deps
```

An editable install still runs from anywhere, so you can keep the checkout
here and run `pic-to-bin` from your working folder — or run it from the
checkout and let `generated/` land next to the source (it is gitignored).

---

## Step 1: Print the template

Generate and print a template for your paper size. **Print at exactly 100%
scale — no fit-to-page.** This is the single most common cause of a bad
result: fit-to-page silently shrinks the sheet a few percent, and every
dimension downstream inherits the error.

```bash
generate-phone-template --paper-size letter --output template.pdf
```

```
Options:
  --paper-size {a4,a5,letter,legal}   Paper size (default: legal)
  --output PATH                       Output PDF path
                                      (default: phone_template_<size>.pdf)
```

The template places 8 ArUco markers (IDs 0–7) — 4 corners and 4 edge
midpoints — around a dotted placement zone. The markers are 20 mm squares
with 20 mm margins from the paper edge, each on a 4 mm white quiet-zone pad
so detection stays reliable even on a non-white background.

**Placement zone sizes by paper** (the sheet inset 40 mm on every side):

| Paper   | Sheet (mm)      | Placement zone (W × H) |
|---------|-----------------|------------------------|
| A4      | 210 × 297       | 130 × 217 mm           |
| Letter  | 215.9 × 279.4   | 136 × 199 mm           |
| Legal   | 215.9 × 355.6   | 136 × 276 mm           |

Legal gives the most room for long tools. A5 templates can be generated, but
the downstream `--paper-size` flags accept only `a4`, `letter`, and `legal`.

Print once and keep the sheet — it is reusable indefinitely.

---

## Step 2: Take the photo

1. Lay the printed template on a flat surface.
2. Place the tool inside the dotted placement zone.
3. Photograph from above. Moderate angles are fine — perspective is
   corrected automatically.
4. All 8 markers should be visible (3 minimum, 8 ideal).
5. Measure the tool's **depth** with calipers — you'll need it for
   `--tool-height`. This is the one number the photo cannot give you.

**Lighting:** diffuse overhead light. Avoid harsh shadows across the markers,
and don't let the tool cover any of them. A side lamp casting a hard shadow
edge next to the tool is the usual reason a trace comes out fat.

**Format:** JPEG, PNG, and HEIC/HEIF (iPhone) are all supported —
`pillow-heif` converts HEIC automatically.

**One tool per photo.** For a multi-tool bin, shoot each tool separately and
pass all the photos at once; the pipeline traces each and packs them into a
single bin.

---

## Step 3a: The web app

```bash
pic-to-bin-web --port 8000        # http://localhost:8000
```

Drag in a photo, fill in the tool height, watch the step tracker, review the
layout preview (with print-at-actual-size PDF/SVG downloads to test fit),
then click **Proceed** to generate the bin config.

**What the web app adds over the CLI:**

- **Two-phase flow with a cheap re-do.** Submitting runs Phase A —
  preprocess, trace, layout — and stops. You review `layout_preview.png` and
  either Proceed (Phase B = `prepare_bin`) or Re-do with different
  parameters. A re-do that only changes layout parameters re-uses the cached
  per-tool DXFs and finishes in seconds; one that changes trace-affecting
  parameters re-traces from the photos.
- **Live progress over SSE.** Worker threads push `ProgressEvent`s to the
  browser; the event log is replayed if you connect late or reload.
- **Field info modals.** Every form field has an `(i)` button next to its
  label opening a multi-paragraph explanation of what it does.
- **Browser back navigates between screens** (form → progress → preview →
  downloads) instead of leaving the site.
- **Multi-user by design.** Per-job UUID directories under `web_jobs/`, a GPU
  semaphore so concurrent SAM2 runs queue rather than fight over the GPU, and
  a background sweep that deletes terminal jobs older than `--job-ttl-hours`
  (default 24). Job files survive a server restart, so the preview/re-do flow
  still works after a reboot.

**Downloads offered:** `layout_preview.png`, `layout_actual_size.pdf`,
`layout_actual_size.svg`, `combined_layout.dxf`, `bin_config.json`, plus a
ZIP of the Fusion 360 add-in.

**Offline-safe frontend.** Lit is vendored inside the package
(`pic_to_bin/web/static/lit-all.min.js`), so the page makes no CDN request.
To re-download it after deleting it, or to bump the pinned version:

```bash
python -m pic_to_bin.web.vendor_lit
```

**The LLM fit review is opt-in.** It is disabled unless you pass
`--enable-llm` (or set `PIC_TO_BIN_ENABLE_LLM`), because it calls the
Anthropic API and bills your account per request. Without the `[llm]` extra
installed, the flag exits with install instructions rather than a traceback.

**Hosting it publicly:** see
[`pic_to_bin/web/README.md`](pic_to_bin/web/README.md) for the NGINX/Apache
reverse-proxy setup. The app ships with upload limits (30 MiB per file, 8
photos, 120 MiB total), security headers, and server-side parameter
validation, and is intended to sit behind a proxy on `127.0.0.1:8000`.

---

## Step 3b: The CLI

### Single tool

```bash
pic-to-bin photo.jpg --tool-height 17
```

### Multiple tools (one photo each)

```bash
pic-to-bin a.jpg b.jpg --tool-height 0=17 --tool-height 1=14
```

Tool indices correspond to image order. Each tool's DXF is traced separately
and packed into one bin.

### Specifying paper size

```bash
pic-to-bin photo.jpg --tool-height 17 --paper-size a4
```

The paper size must match what you printed. Default is `legal`.

### Shallow drawer (no stacking lip)

```bash
pic-to-bin photo.jpg --tool-height 17 --stacking false
```

Drops the 4.4 mm stacking lip for shorter bins in shallow drawers. Pocket
depth is unchanged.

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

### Tolerance model

Fit is the parameter most worth understanding, because it is the difference
between a tool that drops in and one that has to be forced.

**Uniform tolerance.** The pipeline silently adds a **2 mm baseline** to
whatever you pass via `--tolerance` (the constant
`pipeline.TOLERANCE_BASELINE_MM`). So the default `--tolerance 0` produces
2 mm of physical clearance, calibrated for typical FDM printing. Pass
`--tolerance -2` for an exact-trace match, and more negative than that for an
interference fit.

**Axial tolerance.** After the uniform offset, the tolerance polygon is
stretched along the tool's PCA principal axis so each *end* gets
`--axial-tolerance` mm of additional clearance, with the perpendicular extent
unchanged. This exists because SAM2 systematically under-detects tool length
— present even on square or rounded ends, worse on tapered tips like a
screwdriver blade. The default `'auto'` computes
`2.0 + 0.014 × axial_length × taper`, where taper comes from a per-bin width
analysis along the principal axis: square-ended tools get the 2 mm floor
(matching the perpendicular baseline), sharply tapered ones get more. Set it
to `0` for a fully uniform tolerance. Implemented in
`trace_export._axial_stretch_polygons` via SVD on the polygon point cloud
plus a linear-ramp scale in the rotated frame.

**Mask erosion defaults to 0.** Uniform erosion disproportionately shrinks
tapered tips — 0.3 mm of erosion can cost meaningful coverage at a
screwdriver tip while barely touching a wide handle. Turn it on
(`--mask-erode 0.3`) only when a photo has a clearly fat shadow halo.

**Simplification.** The TOLERANCE layer is always Douglas-Peucker simplified
at ε = 0.3 mm and corner-rounded, for any offset value including 0. This
keeps Fusion sketches to roughly 50 points per tool instead of hundreds,
which is the difference between a build that takes seconds and one that
crawls.

### Bin sizing logic

- The bin auto-sizes to the smallest gridfinity unit count that fits the
  tool: `ceil((tool_height + 1mm) / 7mm)` height units.
- The pocket floor sits 1 mm above the bin floor, leaving solid material
  underneath.
- The deck rises to half the tool's height — the upper half of the tool
  stands proud for finger access, the lower half is buried in the pocket.
- The finger slot is centered along the tool's PCA principal axis, using the
  global axis center rather than a local midpoint, so asymmetric tools
  (screwdrivers, wrenches) don't end up with the slot pushed toward one end.
- The combined cutout (pocket + finger slot) is centered in the bin floor,
  so the slack from rounding up to whole gridfinity units is distributed
  evenly on all four sides rather than piling up on two.

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

`layout_actual_size.pdf` and `.svg` are sized to the bin's exact mm
dimensions. Print at "Actual size" / 100% (NOT "Fit to page") and lay your
real tool on the printout to verify the fit before committing to hours of 3D
printing. This is the cheapest possible check and it catches nearly every
scale or tolerance mistake.

After Fusion runs, if you tick the export checkboxes:

```
gridfinity_bin.stl
gridfinity_bin.step
gridfinity_bin_preview.png   Viewport screenshot of the finished bin
```

---

## Running individual steps

Each stage of the pipeline can run on its own — useful for debugging a bad
trace without re-running everything.

### Preprocess a photo

Detect markers, correct perspective, and save the rectified image:

```bash
preprocess-phone photo.jpg --paper-size letter --output-dir generated/photo
```

```
positional:
  image                         Phone photo file

optional:
  --paper-size {a4,letter,legal}  Template paper size (default: legal)
  --output-dir DIR                Output directory (default: generated/<stem>)
```

Outputs `<stem>_rectified.png` at the computed effective DPI (typically
100–250). Open it: if the markers look square and the tool looks
undistorted, calibration worked.

### Trace a rectified image

Run SAM2 segmentation + vectorization:

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

## Step 4: Fusion 360

### Installing the add-in

```bash
pic-to-bin-fusion install
```

The add-in files ship inside the Python package; this copies them into
Fusion 360's own add-ins directory:

- Windows: `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\pic_to_bin\`
- macOS: `~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns/pic_to_bin/`

Fusion must already be installed, and it only runs on Windows and macOS.

Then, inside Fusion:

1. Press **Shift+S** → **Add-Ins** tab.
2. Select **pic_to_bin** → **Run**. Tick **Run on Startup** so the button is
   there every session.
3. In a Design workspace, **Solid → Create** now has a **Gridfinity
   Pic-to-Bin** button.
4. Click it. A small dialog offers three checkboxes — save STL / STEP /
   preview PNG to your Desktop — all **off** by default, so an unchecked
   build leaves nothing on disk but the in-Fusion design.
5. It loads `<project>/generated/bin_config.json` if it finds one; otherwise
   a file dialog opens, defaulting to your Desktop.

The command is idempotent — re-run it after upgrading the package to refresh
the installed copy. `pic-to-bin-fusion uninstall` removes it (and cleans up
the legacy `API/Scripts/pic_to_bin/` folder left behind by versions that
installed a separate script entry point).

The add-in reloads `_bin_builder.py` from disk on every button click, so most
code changes land on the next press without restarting Fusion. Only changes
to the entry point itself (`pic_to_bin_addin/pic_to_bin.py`) need a Stop/Run
or a Fusion restart.

### Manual install from the web app ZIP

If you are using a hosted instance and never installed the Python package,
the web app's Step 0 link downloads `pic-to-bin-fusion.zip` containing the
add-in plus one-click installers. Unzip it and double-click
`install_windows.bat` or `install_macos.command`.

Or install by hand — the installers only do a folder copy. Drag the ZIP's
`AddIns/pic_to_bin` folder into Fusion's user API directory so the final path
ends with `.../AddIns/pic_to_bin/pic_to_bin.py`, then follow the Fusion steps
above.

> On Windows, paste `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns` into
> the File Explorer address bar — File Explorer expands `%APPDATA%`,
> PowerShell does not. And note that
> `C:\Users\<you>\AppData\Local\Autodesk\webdeploy\...` is Fusion's
> *bundled* add-ins folder, which gets wiped on every Fusion update. User
> add-ins do not go there.

### macOS: "cannot verify the developer"

Double-clicking `install_macos.command` may produce a Gatekeeper dialog
saying macOS cannot verify the developer, offering only **Move to Trash** or
**Cancel**. That is expected: the ZIP came from a browser, so it carries the
`com.apple.quarantine` attribute, which Finder propagates to everything
extracted from it — and macOS 15 (Sequoia) removed the "Open Anyway" button
from that dialog. Nothing is wrong with the script.

Three ways through, easiest first:

1. **Run it from Terminal.** Quarantine gates double-click launches, not an
   explicitly invoked interpreter:

   ```bash
   bash ~/Downloads/pic-to-bin-fusion/install_macos.command
   ```

   (Type `bash ` then drag the file into Terminal to fill in the path.)

2. **Strip the quarantine flag**, then double-click normally:

   ```bash
   xattr -cr ~/Downloads/pic-to-bin-fusion
   ```

3. **System Settings → Privacy & Security**, scroll to the Security section,
   and click **Open Anyway** next to the blocked item — it appears there for
   about an hour after the dialog. Then double-click the script again and
   confirm.

To avoid it entirely next time, unzip from Terminal
(`unzip pic-to-bin-fusion.zip -d pic-to-bin-fusion`) — the command-line
`unzip` does not propagate quarantine to extracted files. Or skip the
installer and do the manual folder copy above.

### What gets built

The bin is generated in a fresh document, each phase in its own named
timeline group so you can keep editing afterward:

- **Bin Body** — one rectangular extrude, with the Fusion appearance
  "ABS (White)" applied immediately.
- **Stacking Lip** (if enabled) — solid block + 4 mm fillet + base-profile
  inverse cutout + 0.6 mm top recess.
- **Deck** — a single rectangular cut inside the wall+inset perimeter,
  lowered to expose the upper half of the tallest tool.
- **Tool Pockets** — one sketch per tool holding all that tool's tolerance
  polygons, one Cut extrude per tool. The cut starts above the lip so
  edge-reaching profiles cut the lip too.
- **Finger Slots** — one sketch and one cut per tool, at the same depth as
  the pocket.
- **Base Pads** — gridfinity baseplate-mating geometry: one sketch and Join
  extrude for all wide pads, the same for the narrow posts, plus two
  chamfers. That is 2 sketches + 2 extrudes + 2 chamfers regardless of how
  many grid units the bin spans.

Then **File → 3D Print** (or Export → STL) to feed your slicer.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `MarkerDetectionError: No markers detected` | Template not visible or too blurry | Ensure all markers are in frame; hold the camera steadier |
| `MarkerDetectionError: Only N markers (need ≥3)` | Markers obscured or overexposed | Improve lighting; don't cover markers with the tool |
| `ScaleInconsistencyError: H/V scales differ >5%` | Template not printed at 100% | Reprint with fit-to-page disabled |
| `WARNING: Low effective DPI (<100)` | Camera too far away | Hold the phone closer; use a higher resolution mode |
| `zsh: no matches found: gridfinity-pic-to-bin[llm]` | zsh globbed the extras bracket | Quote the spec: `pipx install --force "gridfinity-pic-to-bin[llm]"` |
| `pic-to-bin-web` says it could not import its web dependencies | Install predates the release that made them core, or was pruned | `pipx install --force gridfinity-pic-to-bin` |
| `--enable-llm` exits saying the anthropic SDK is missing | The LLM review is the one optional extra | `pipx install --force "gridfinity-pic-to-bin[llm]"`, or drop the flag |
| macOS won't run `install_macos.command` ("unknown developer") | Gatekeeper quarantine from the browser download | See [macOS: "cannot verify the developer"](#macos-cannot-verify-the-developer) |
| Tools don't fit in the grid | Tools too large for `--max-units` | Increase `--max-units` |
| Fusion freezes building pockets | Stale cached `_bin_builder` after editing | The reload is already wired in — click the button again. If still stuck, restart Fusion |
| Pocket fits too loose | Default `--tolerance 0` gives 2 mm physical clearance + ≥2 mm at each tip | Lower with `--tolerance -0.5` and/or `--axial-tolerance 1.0` |
| Pocket fits too tight at the tips only | SAM2 under-detected the tapered ends | Increase `--axial-tolerance` (default `auto`, 2 mm floor) |
| Pocket fits too tight everywhere | Trace itself is short (shadow halo, parallax, mask erosion) | Try `--tolerance 1` (= 3 mm physical). If still tight, check `--phone-height` matches your shooting distance |

### Common photo issues

- **Markers partially cut off** — keep all 8 in frame. Three is the minimum
  the pipeline will run with, but accuracy drops.
- **Blurry markers** — tap to focus on the template before shooting.
- **Shadows on markers** — use diffuse or overhead light, not a side lamp.
- **HEIC images (iPhone)** — supported natively; `pillow-heif` converts
  automatically.

---

## Development

```bash
git clone https://github.com/justinhardin/gridfinity-pic-to-bin.git
cd gridfinity-pic-to-bin
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Package internals are documented in [`CLAUDE.md`](CLAUDE.md) and the
per-directory `README.md` files under `pic_to_bin/`.

---

## License

MIT — see [LICENSE.md](LICENSE.md).
