# Gridfinity Pic-to-Bin

Photograph a tool lying on a printed marker sheet. Get a 3D-printable
gridfinity bin with a pocket cut to the tool's exact shape.

📖 **[Full guide →](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md)**

---

## Get it running

### 1. Install

```bash
pipx install gridfinity-pic-to-bin
pic-to-bin-fusion install
```

The second command adds a **Gridfinity Pic-to-Bin** button under **Solid →
Create** in Fusion 360 — that's what turns the result into a solid model
([enable it with Shift+S →](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md#installing-the-add-in)).

No pipx yet? `py -m pip install --user pipx; py -m pipx ensurepath` (Windows)
or `brew install pipx && pipx ensurepath` (macOS) — then open a new terminal.
Needs Python 3.10+.
[More install options →](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md#installation)

### 2. Print the template

```bash
generate-phone-template --paper-size letter
```

Print the PDF at **exactly 100% scale — not fit-to-page.** One-time setup;
the sheet is reusable.

### 3. Photograph the tool

Lay it inside the dotted zone, shoot from above with all 8 markers in frame.
Measure the tool's depth with calipers — that's the one number the photo
can't give you.
[Photo tips →](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md#step-2-take-the-photo)

### 4. Start the web app

```bash
mkdir ~/pic-to-bin && cd ~/pic-to-bin   # Windows: mkdir $HOME\pic-to-bin; cd $HOME\pic-to-bin
pic-to-bin-web --port 8000
```

Run it from a dedicated folder — output and the SAM2 model weights (several
hundred MB, downloaded on the first trace) land in the current directory.

### 5. Use it

Open **http://localhost:8000**, drag in the photo, enter the tool depth, and
submit. You get a layout preview, a **1:1 fit-test PDF** to print and lay the
real tool on to check the fit, and a `bin_config.json`.

Then in Fusion 360, click **Solid → Create → Gridfinity Pic-to-Bin** and
point it at that `bin_config.json`. It builds the whole parametric bin —
body, pockets, finger slots, stacking lip, gridfinity base pads. Export STL
and print.

---

Prefer the terminal? The web app is a wrapper around one command:

```bash
pic-to-bin photo.jpg --tool-height 17
```

---

## Documentation

| | |
|---|---|
| [How it works](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md#how-it-works) | ArUco calibration, SAM2 tracing, layout packing, bin generation |
| [Installation](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md#installation) | pipx vs. venv, where to run commands, upgrading, from source |
| [Web app](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md#step-3a-the-web-app) | Two-phase re-do flow, multi-user setup, the opt-in LLM review |
| [CLI reference](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md#all-pic-to-bin-options) | Every `pic-to-bin` flag, plus the individual pipeline steps |
| [Tolerance & fit](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md#tolerance-model) | Why the pocket fits how it does, and how to tighten or loosen it |
| [Fusion 360](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md#step-4-fusion-360) | Add-in install, manual ZIP install, macOS Gatekeeper, what gets built |
| [Troubleshooting](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/GUIDE.md#troubleshooting) | Marker detection errors, bad fits, common photo problems |
| [Hosting the web app](https://github.com/justinhardin/gridfinity-pic-to-bin/blob/master/pic_to_bin/web/README.md) | Reverse proxy setup and security notes |

---

MIT licensed — see [LICENSE.md](LICENSE.md).
