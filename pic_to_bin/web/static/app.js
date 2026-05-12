import { LitElement, html, css, nothing } from "lit";

// Pipeline parameter metadata. Mirrors pipeline.run_pipeline kwargs.
// "Soft default" = fields with a default that we highlight until the user
// touches them, so they consciously confirm or change the value.
// (Currently empty — phone_height moved to backend EXIF auto-detect.)
const SOFT_DEFAULT_FIELDS = new Set();

// Must match TOLERANCE_BASELINE_MM in pipeline.py — used in modal copy only.
const TOLERANCE_BASELINE_MM = 2.0;

const FORM_DEFAULTS = {
  part_name: "",
  paper_size: "legal",
  tolerance: 0.0,
  axial_tolerance: "",  // empty = "auto" (taper-based heuristic on the backend)
  tool_taper: "top",
  gap: 3.0,
  bin_margin: 0.0,
  max_units: 7,
  min_units: 1,
  height_units: "",
  stacking: true,
  slots: true,
  straighten_threshold: 45.0,
  max_refine_iterations: 5,
  max_concavity_depth: 3.0,
  mask_erode: 0.0,
  display_smooth_sigma: 2.5,
};

// Which fields, when changed on a redo, force a re-trace (expensive).
// Everything else is layout-only (cheap; cached DXFs reused).
const TRACE_REQUIRED_FIELDS = new Set([
  "tool_taper",
  "tolerance",
  "axial_tolerance",
  "slots",
  "straighten_threshold",
  "max_refine_iterations",
  "max_concavity_depth",
  "mask_erode",
  "display_smooth_sigma",
]);

// Detailed per-field documentation. The key matches the form-field name; the
// title is shown in the modal heading and the body is rendered as paragraphs
// (split on \n\n). A short 1-line `hint` is shown inline under the input.
const FIELD_INFO = {
  part_name: {
    title: "Part name (optional)",
    hint: "If set, downloaded files will use this name (e.g. zircon_layout.pdf).",
    body: [
      "Optional label for this job. When set, the files you download are renamed with this prefix — for example, layout_actual_size.pdf becomes <part_name>_layout.pdf, layout_preview.png becomes <part_name>_preview.png, and so on.",
      "Special characters are stripped automatically: only letters, numbers, underscores, and hyphens are kept; spaces become underscores. Leave it blank to keep the default filenames.",
    ],
  },
  photo_tips: {
    title: "Photo tips",
    hint: "How to take a photo that traces cleanly.",
    images: [
      { src: "/static/photo_tips/pipe_cutter.jpg",
        caption: "Large hand tool, white template, centered, soft overhead light." },
      { src: "/static/photo_tips/blade_case.jpg",
        caption: "Boxy item with high-contrast labels, no hard shadows around the edges." },
      { src: "/static/photo_tips/screwdriver_green.jpg",
        caption: "Dark slim tool on the green chroma-key template — helps SAM2 with small light-on-dark tools." },
    ],
    body: [
      "Shoot from directly above. Hold the phone parallel to the template and centered over it — the closer to a true overhead angle, the better the ArUco markers calibrate the scale. Moderate tilts are corrected by the homography, but extreme angles lose accuracy at the tool tips.",
      "Center the tool within the dotted placement zone on the template. Keep some white space around it so SAM2 sees a clean tool-vs-background edge on every side.",
      "Use even, diffuse lighting. Soft overhead light (a bright room, a window with a sheer curtain, or a ring/softbox) produces the cleanest trace. Try to keep shadows few and soft — hard shadows from a single point source can be picked up as part of the tool outline.",
      "Make sure all eight ArUco markers are visible and unobstructed. Three is the minimum the pipeline accepts, but eight gives the best perspective correction.",
      "Print the template at 100% scale on white paper (or chroma-key green for light/reflective tools). Never \"fit to page\" — that rescales the markers and breaks calibration.",
    ],
  },
  tool_height: {
    title: "Tool height (mm)",
    hint: "Required. Measure with calipers — depth of the tool when it lies flat.",
    body: [
      "The thickest dimension of the tool when laid flat on the template — what you'd measure with calipers across its depth.",
      "The bin auto-sizes so the upper half of the tool stands above the deck (for finger access) and the lower half is buried in the pocket. The pocket floor sits 1 mm above the bin floor.",
      "If you're tracing several photos in one job, give each one its own height — the form lets you set them independently.",
    ],
  },
  paper_size: {
    title: "Paper size",
    hint: "Match the size you printed the ArUco template on.",
    body: [
      "The marker template is provided in three sizes: A4 (210×297 mm), US Letter (215.9×279.4 mm), and US Legal (215.9×355.6 mm).",
      "Pick the size you actually printed — the homography uses the marker positions for that size to compute the millimeter scale.",
    ],
  },
  tool_taper: {
    title: "Tool side profile",
    hint: "Where the tool is widest, viewed from the side. Affects parallax.",
    body: [
      "Pick the side-profile shape closest to your tool. The pipeline uses this to decide which height to use when compensating for parallax — tools that taper inward (wider at the bottom) need no compensation, while tools that flare outward at the top, have vertical sides, or bulge in the middle need the full correction.",
      "Widest at top, uniform, or widest in the middle — flares outward at the top, has vertical sides where the top and bottom outlines match, or bulges in the middle (cylinders, rounded handles, lozenge shapes). Most hand tools fall here (screwdrivers, pliers, hammers, wrenches with flared handles), as do boxy items (multimeters, batteries, USB drives) and rounded items (flashlights, batteries on their side). All three share the same parallax math because the silhouette peak sits near the middle of the tool's height.",
      "Widest at bottom — tapers inward going up. Zircon stud finders, computer mice, phone cases, tape-measure cases.",
      "Picking the wrong option makes the trace a few percent too small or too big, with the error growing with tool height. For a 30 mm-tall tool at typical hand-held distance the swing between options is around 3 %; for a 100 mm-tall tool it's about 10 %.",
    ],
  },
  tolerance: {
    title: "Tolerance (mm)",
    hint: "Extra clearance on top of the standard fit. Default 0 is recommended.",
    body: [
      `Extra clearance applied uniformly to the pocket on top of a built-in ${TOLERANCE_BASELINE_MM} mm baseline. The baseline is calibrated for typical FDM 3D printer tolerances — at the default 0, your tool should slide into the printed pocket comfortably.`,
      "Positive — looser fit. Use for soft or rubber-handled tools, or if your printer over-extrudes.",
      "Negative — tighter fit. The first −0.3 to −0.5 mm gives a snug, hand-fit feel. Going more negative produces an interference fit (the tool wedges in).",
      `−${TOLERANCE_BASELINE_MM} — pocket matches the trace exactly (no clearance at all). Below this value the pocket is smaller than the trace.`,
      `Typical range: −${TOLERANCE_BASELINE_MM} to +2 mm. Default: 0.`,
      "Higher → looser pocket, easier to drop the tool in, but it can rattle. Lower → tighter pocket, premium feel, harder to insert; below −0.5 mm you may need to sand or force the tool in.",
      "The tolerance polygon is always Douglas-Peucker simplified at 0.3 mm regardless of value, so Fusion gets a clean low-point-count cut.",
    ],
  },
  axial_tolerance: {
    title: "Axial tolerance (mm)",
    hint: "Leave blank for auto. Extra clearance at the tool's tips, not its sides.",
    body: [
      "Extra clearance pushed onto each end of the tool along its long (principal) axis only. The perpendicular extent is unchanged.",
      "Why this exists: SAM2 segmentation under-detects tapered or thin tool tips, so the raw trace is shorter than the real tool. Adding *uniform* tolerance to compensate makes the wider sections too loose; this setting fixes only the ends.",
      "Default 'auto' (blank field) computes a value from the trace: a square-ended tool (e.g. ruler) gets ~0.5 mm; a sharply tapered tool (e.g. shears) gets up to ~3 mm. Formula: 0.5 + 0.014 × axial_length × taper, where taper = 1 − tip_width / body_width.",
      "Override by typing a number — e.g. 1.0 to push each end outward by 1 mm regardless of shape. Set to 0 for fully uniform tolerance.",
      "Typical range: 0 to 5 mm (or blank = auto). Default: auto.",
      "Higher → longer pocket, the tool tip never bottoms out, but the pocket sticks out beyond the tool. Lower → tighter at the tips; below auto, sharply tapered tools may not seat all the way in.",
      "Caveat: the stretch is linear in the rotated frame, so any features along the axis stretch slightly too. Fine for typical hand tools, less ideal for tools with internal axis-parallel features (rare).",
    ],
  },
  gap: {
    title: "Gap between tools (mm)",
    hint: "Minimum space between adjacent pockets in the layout.",
    body: [
      "When packing multiple tools into one bin, leave at least this much wall between any two pockets.",
      "Note: this is the MINIMUM. When the bin has slack after snapping to a whole gridfinity unit, the extra space is redistributed evenly between tools, so actual gaps can be larger.",
      "Typical range: 1 to 10 mm. Default: 3 mm.",
      "Higher → roomier separation, easier to grab a single tool, but the bin grows when the extra width crosses a 42 mm unit boundary. Lower → more tools fit in a smaller bin, but pockets can feel cramped and the inter-pocket wall gets fragile to print.",
    ],
  },
  bin_margin: {
    title: "Bin margin (mm)",
    hint: "Extra clearance from tool extents to the bin wall.",
    body: [
      "Extra padding between the outermost tool extent and the bin boundary, applied before snapping to a whole gridfinity unit.",
      "Usually you don't need this — the natural slack from rounding the bin size up to a whole 42 mm unit, plus the tolerance, is enough.",
      "Set this >0 to force the bin one unit larger when a tool would otherwise sit right against the wall.",
      "Typical range: 0 to 6 mm. Default: 0.",
      "Higher → bin bumps up to the next 42 mm unit sooner, giving more breathing room between tool and wall — useful if your prints distort near the edges. Lower (or 0) → smallest possible bin; tools may sit right against the wall when the snap slack is small.",
    ],
  },
  min_units: {
    title: "Minimum grid units",
    hint: "Smallest bin footprint per axis (gridfinity units, 42 mm each).",
    body: [
      "Force the bin to be at least this many units wide and tall, even if the tool would fit in something smaller.",
      "Useful when you want the bin to match an existing drawer slot or to look uniform alongside other bins in a set.",
      "Typical range: 1 to 7 (must be ≤ max_units). Default: 1.",
      "Higher → bin is at least N×N units even for a small tool, so it slots into a fixed drawer layout — at the cost of using more filament and bench space. Lower → bin shrinks to the tightest gridfinity unit count that fits the tool.",
    ],
  },
  max_units: {
    title: "Maximum grid units",
    hint: "Largest bin footprint allowed before the pipeline gives up.",
    body: [
      "Cap on how big the bin can grow per axis. If the tools don't fit in this size the pipeline raises an error rather than silently producing a giant bin.",
      "Default 7×7 is plenty for most hand tools.",
      "Typical range: 2 to 12 (must be ≥ min_units). Default: 7.",
      "Higher → larger tools or multi-tool layouts are allowed to spawn larger bins. Lower → packing fails fast with an actionable error instead of producing a bin that won't fit your gridfinity baseplate.",
    ],
  },
  height_units: {
    title: "Bin height (units, blank = auto)",
    hint: "Force a specific bin height, or leave blank to auto-size.",
    body: [
      "Each gridfinity height unit is 7 mm.",
      "When blank (default), the pipeline picks the smallest height that fits the tallest tool plus 1 mm of floor below the pocket.",
      "Set explicitly if you need to match an existing stack or want a deeper pocket than the tool requires.",
      "Typical range: 2 to 12 units (each = 7 mm). Default: auto.",
      "Higher → taller bin, deeper pocket — the tool sits lower and exposes less for finger access, and the print takes more filament. Lower → shorter bin; if you set this below the auto value the tool will stick out the top.",
    ],
  },
  stacking: {
    title: "Stacking lip",
    hint: "Toggle the lip that lets bins stack on each other.",
    body: [
      "When on, the bin gets the standard gridfinity stacking lip on its top edge so other bins can stack on it.",
      "Turn off for shallow drawers where the lip would push the bin too tall.",
    ],
  },
  slots: {
    title: "Finger-access slots",
    hint: "Toggle the cutout that lets you slide a finger under the tool.",
    body: [
      "Adds a slot in the pocket centered along the tool's principal axis so you can slide a finger underneath and lift it out.",
      "Turn off for very small tools where the slot would intrude awkwardly, or for tools you'd rather pinch from the top.",
    ],
  },
  straighten_threshold: {
    title: "Straighten threshold (degrees)",
    hint: "Auto-rotate the trace if it's within this many degrees of axis-aligned.",
    body: [
      "After tracing, if the tool's principal axis is within this many degrees of horizontal or vertical, the pipeline rotates the trace to align with the bin axes.",
      "Set to 0 to disable auto-straightening (use the exact rotation captured in the photo).",
      "Typical range: 0 to 45 degrees. Default: 45.",
      "Higher → almost any rotation gets snapped to horizontal/vertical, giving a cleaner-looking pocket in the bin even from a sloppy photo. Lower → the pocket preserves the exact tilt of the tool in the photo; set to 0 if your tool genuinely sits at an off-axis angle in real use.",
    ],
  },
  max_refine_iterations: {
    title: "Max refine iterations",
    hint: "How many cleanup passes the trace is allowed to take.",
    body: [
      "After SAM2 produces an initial mask, the pipeline iteratively cleans it up — closing notches, smoothing contours — and re-checks the result.",
      "More iterations let it produce smoother outlines for noisy photos but take longer. Default 5 is a good balance.",
      "Typical range: 1 to 10. Default: 5.",
      "Higher → smoother contours on noisy photos at the cost of a few extra seconds per tool, with diminishing returns past ~6 passes. Lower → faster trace; set to 1 to disable iterative refinement entirely and use the raw SAM2 mask, which can leave small notches or jagged edges.",
    ],
  },
  max_concavity_depth: {
    title: "Max concavity depth (mm)",
    hint: "How aggressively cleanup is allowed to fill in concavities.",
    body: [
      "Cleanup may fill shallow concavities to smooth the outline. This is the deepest concavity (in mm) the cleanup is allowed to lose before stopping.",
      "Increase if your tool has deliberate notches that are getting filled in. Decrease for cleaner outlines on simple shapes.",
      "Typical range: 0.5 to 10 mm. Default: 3 mm.",
      "Higher → cleanup keeps going until even deep notches are preserved, which is right for tools with intentional cutouts (scissor finger holes, wrench openings) but may leave jagged outlines on noisy traces. Lower → aggressive smoothing fills in real features; useful for blob-shaped tools (handles, grips) where you don't care about minor surface detail.",
    ],
  },
  mask_erode: {
    title: "Mask erosion (mm)",
    hint: "Pixel-shrink the SAM mask. Default 0; only enable if the trace looks fat.",
    body: [
      "Phone photos can have a soft shadow halo around the tool that SAM2 picks up as part of the mask, making the trace slightly fatter than reality.",
      "This setting erodes the mask by N millimeters before vectorization. The default is 0 because a uniform erosion disproportionately shrinks thin or tapered tool tips, which often makes those pockets too tight even when the wider sections fit.",
      "Increase to 0.3–0.5 mm only if your photo has a visible shadow halo and the trace clearly extends beyond the tool's actual outline.",
      "Typical range: 0 to 1 mm. Default: 0.",
      "Higher → tighter trace, useful when shadows have inflated it; but tapered tips shrink faster than the body so the pocket can pinch the tip before the body seats. Lower (0) → keep the raw SAM2 outline; prefer this and use negative Tolerance to tighten the fit uniformly instead.",
    ],
  },
  display_smooth_sigma: {
    title: "Smoothing strength (mm)",
    hint: "Smooths SAM2 noise along the trace. Default 2.5 mm.",
    body: [
      "Gaussian smoothing applied to the trace polygon perpendicular to the tool's principal axis. It removes wave-noise that SAM2 can produce along an otherwise straight edge.",
      "Increase (3–5 mm) if the trace comes out with a visibly wavy outline along sections that should be straight or gently curved. Decrease (or set to 0) if the trace is over-smoothing intentional features — e.g. a bracket whose 90° corners are showing up rounded.",
      "Typical range: 0 to 5 mm. Default: 2.5 mm.",
      "Higher → glassy-smooth outline; but moderately curved features (radii, fillets) soften past their real shape. Lower → preserves curvy detail and sharp transitions; set to 0 to keep the trace exactly as Douglas-Peucker emitted it.",
      "Sharp corners are preserved automatically by curvature-aware blending regardless of this value, but moderately curved features still soften at higher settings. Default 2.5 mm balances both cases.",
    ],
  },
  template_setup: {
    title: "Print the ArUco template",
    hint: "Print one of the templates below at 100% scale before taking photos.",
    body: [
      "Every photo must include the printed ArUco template — the eight markers around the edge tell the pipeline how to remove perspective and how many millimeters every pixel represents.",
      "Pick the paper size that matches what's loaded in your printer. A5, US Letter, and US Legal are all supported. Print at 100% scale (or \"Actual size\") — never \"Fit to page,\" since that rescales the markers and breaks the calibration.",
      "Background color: pick White if you want the lowest ink usage, or Green (#00B140 chroma-key) if your tool is light-colored or has reflective metal that SAM2 sometimes confuses with a white background. The green variant prints small white pads behind each marker so detection still works.",
      "Each template is reusable — print once, photograph as many tools as you like.",
    ],
  },
  fit_test: {
    title: "Print at actual size to test fit",
    hint: "Print, lay the tool on top, and verify before 3D printing.",
    body: [
      "Before you commit to a multi-hour 3D print, print the layout on a normal printer and lay your real tool on top of the outlines to verify the fit.",
      "Use the PDF or SVG download — both encode the bin's exact mm dimensions, so they print at 1:1 scale on any printer regardless of its DPI. The PNG version is for screen viewing only and will print at the wrong size.",
      "When you print, choose \"Actual size\" or \"100% scale\" (NOT \"Fit to page\"). The page will be exactly the bin footprint, so on a Letter/A4 sheet you'll see the bin shape with whitespace around it.",
      "If your tool is bigger than 8.5×11\", the PDF page may be larger than your paper — most printers will then offer a \"poster / tile across multiple pages\" mode. The SVG opens in any browser and you can print from there as well.",
      "What to check: the dashed outline is the actual pocket shape. Your tool should fit inside it with the tolerance you set (1 mm clearance by default).",
    ],
  },
};

// ---------------------------------------------------------------------------
// Root: single scrolling page. Each pipeline phase reveals a new section
// below the form rather than navigating away from it.
// ---------------------------------------------------------------------------

class PicApp extends LitElement {
  static properties = {
    jobId: { state: true },
    files: { state: true },         // [{file: File, toolHeight: number|null, dataUrl: string}]
    formValues: { state: true },    // dict of form field values
    touched: { state: true },       // Set of field names the user has touched
    running: { state: true },       // a job phase is in flight
    layoutInfo: { state: true },
    artifacts: { state: true },
    errorMessage: { state: true },
    eventLog: { state: true },
    modalField: { state: true },
  };

  // Don't isolate inside shadow DOM — we want the global stylesheet to apply.
  createRenderRoot() { return this; }

  constructor() {
    super();
    this.jobId = null;
    this.files = [];
    this.formValues = { ...FORM_DEFAULTS };
    this.touched = new Set();
    this.running = false;
    this.layoutInfo = null;
    this.artifacts = {};
    this.errorMessage = null;
    this.eventLog = [];
    this.modalField = null;
    this._artifactKey = 0;
    this._eventSource = null;
    // Track which transition events have already been acted on. The server
    // replays the full event_log on every SSE reconnect, so layout_ready /
    // complete arrive again after every idle disconnect; without these flags
    // the preview img refetches each time and the page flickers.
    this._seenLayoutReady = false;
    this._seenComplete = false;
    // Sections that were visible last render — used to scroll the newly-
    // revealed one into view when state changes.
    this._prevVisible = new Set();
  }

  connectedCallback() {
    super.connectedCallback();
    // Children dispatch CustomEvent("show-info", {detail: {field}, bubbles: true})
    // and we render the modal here so it works for any section.
    this._showInfoHandler = (e) => { this.modalField = e.detail.field; };
    this.addEventListener("show-info", this._showInfoHandler);
    this._escHandler = (e) => {
      if (e.key === "Escape" && this.modalField) this.modalField = null;
    };
    window.addEventListener("keydown", this._escHandler);
    // ?job=<uuid> in the URL means "resume that job". Fire-and-forget;
    // the async restore updates state once the server summary comes back.
    this._restoreFromUrl();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.removeEventListener("show-info", this._showInfoHandler);
    window.removeEventListener("keydown", this._escHandler);
  }

  // ---- URL <-> jobId sync -------------------------------------------------

  _setJobInUrl(jobId) {
    const url = new URL(window.location.href);
    url.searchParams.set("job", jobId);
    history.replaceState(history.state, "", url.toString());
  }

  _clearJobFromUrl() {
    const url = new URL(window.location.href);
    url.searchParams.delete("job");
    history.replaceState(history.state, "", url.toString());
  }

  async _restoreFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const jobId = params.get("job");
    if (!jobId) return;

    let res;
    try {
      res = await fetch(`/jobs/${encodeURIComponent(jobId)}`);
    } catch (e) {
      this.errorMessage = `Failed to load job: ${e.message}`;
      this._clearJobFromUrl();
      return;
    }
    if (res.status === 404) {
      this.errorMessage =
        `Job ${jobId.slice(0, 8)}… not found on this server. ` +
        `It may have expired or the server was restarted.`;
      this._clearJobFromUrl();
      return;
    }
    if (!res.ok) {
      this.errorMessage = `Failed to load job: HTTP ${res.status}`;
      this._clearJobFromUrl();
      return;
    }

    const summary = await res.json();
    this.jobId = jobId;
    const hasLayout = !!summary.artifacts?.layout_preview;
    const hasFinal = !!summary.artifacts?.bin_config;

    // Repopulate the form so the user sees the photos + tool heights they
    // submitted — without this, reloading on a job ID drops them onto an
    // empty form and they think their work is gone.
    if (Array.isArray(summary.input_filenames) && summary.input_filenames.length > 0) {
      const toolHeights = summary.params?.tool_heights || {};
      this.files = summary.input_filenames.map((name, i) => {
        const h = toolHeights[i] ?? toolHeights[String(i)] ?? null;
        return {
          // Synthetic stand-in for a File: only `.name` is read by the
          // renderer + HEIC sniff. _submitJob detects `restored: true`
          // and re-fetches the bytes from the server when the user
          // submits a fresh job from this prepopulated form.
          file: { name },
          toolHeight: h !== null && h !== undefined ? Number(h) : null,
          // The /preview endpoint serves the original for jpg/png and a
          // lazily-cached JPEG thumbnail for HEIC, so the renderer's
          // <img src=${dataUrl}> works for both. Re-upload still uses
          // the no-suffix URL so the new job gets the original bytes.
          dataUrl: `/jobs/${encodeURIComponent(jobId)}/inputs/${encodeURIComponent(name)}/preview`,
          restored: true,
        };
      });
    }
    if (summary.params || summary.part_name) {
      const fv = { ...FORM_DEFAULTS };
      for (const [k, v] of Object.entries(summary.params || {})) {
        // tool_heights is per-file (lives in `this.files[i].toolHeight`),
        // not on the form itself. sam_corrective_points stays so
        // pic-preview's willUpdate can re-seed its click state from it
        // on the artifactKey bump that's about to fire.
        if (k === "tool_heights") continue;
        fv[k] = v;
      }
      if (summary.part_name) fv.part_name = summary.part_name;
      this.formValues = fv;
      // Mark every restored field as touched so SOFT_DEFAULT_FIELDS get
      // sent back verbatim on resubmit instead of being dropped in favor
      // of the auto-detect fallback.
      this.touched = new Set(Object.keys(summary.params || {}));
    }

    if (hasLayout) {
      this.layoutInfo = summary;
      this._artifactKey = Date.now();
      this._seenLayoutReady = true;
    }
    if (hasFinal) {
      this.artifacts = summary.artifacts || {};
    }
    if (summary.status === "complete") {
      this._seenComplete = true;
      // No SSE connection — terminal state. Progress section stays hidden
      // since eventLog is empty; the user picks up at preview + downloads.
    } else if (summary.status === "error") {
      this.errorMessage = summary.error || "Pipeline error.";
    } else {
      // pending / running / awaiting_decision / finalizing — live job.
      // Connect SSE: the server replays event_log on subscribe, so the
      // progress card populates with everything that has happened so far,
      // then keeps streaming live updates. `running` reflects whether a
      // phase is in flight (anything but awaiting_decision).
      this.running = summary.status !== "awaiting_decision";
      this._connectEvents(jobId);
    }
  }

  updated(_changed) {
    const visible = this._visibleSections();
    for (const id of visible) {
      if (!this._prevVisible.has(id)) {
        const el = this.querySelector(`#section-${id}`);
        if (el) {
          requestAnimationFrame(() => {
            el.scrollIntoView({ behavior: "smooth", block: "start" });
          });
        }
      }
    }
    this._prevVisible = visible;
  }

  _visibleSections() {
    const v = new Set();
    if (this.running || this.eventLog.length > 0) v.add("progress");
    if (this.errorMessage) v.add("error");
    if (this.layoutInfo) v.add("preview");
    if (this.artifacts && Object.keys(this.artifacts).length > 0) v.add("downloads");
    return v;
  }

  render() {
    return html`
      <section id="section-form">
        ${this._renderForm()}
      </section>
      ${this.running || this.eventLog.length > 0 ? html`
        <section id="section-progress">
          ${this._renderProgress()}
        </section>
      ` : nothing}
      ${this.errorMessage ? html`
        <section id="section-error">
          ${this._renderError()}
        </section>
      ` : nothing}
      ${this.layoutInfo ? html`
        <section id="section-preview">
          ${this._renderPreview()}
        </section>
      ` : nothing}
      ${this.artifacts && Object.keys(this.artifacts).length > 0 ? html`
        <section id="section-downloads">
          ${this._renderDownloads()}
        </section>
      ` : nothing}
      ${this._renderModal()}
    `;
  }

  _renderModal() {
    if (!this.modalField) return nothing;
    const info = FIELD_INFO[this.modalField];
    if (!info) return nothing;
    const close = () => this.modalField = null;
    return html`
      <div class="modal-backdrop" @click=${close}>
        <div class="modal" @click=${(e) => e.stopPropagation()} role="dialog">
          <div class="modal-header">
            <h3>${info.title}</h3>
            <button class="modal-close" type="button" @click=${close} aria-label="Close">×</button>
          </div>
          <div class="modal-body">
            ${info.images && info.images.length ? html`
              <div class="modal-image-grid">
                ${info.images.map(im => html`
                  <figure class="modal-example">
                    <img src=${im.src} alt=${im.caption || ""} loading="lazy">
                    ${im.caption ? html`<figcaption>${im.caption}</figcaption>` : nothing}
                  </figure>
                `)}
              </div>
            ` : nothing}
            ${info.body.map(p => html`<p>${p}</p>`)}
          </div>
          <div class="modal-footer">
            <button class="primary" type="button" @click=${close}>Got it</button>
          </div>
        </div>
      </div>
    `;
  }

  // ---- Sections -----------------------------------------------------------

  _renderForm() {
    const hasResult = this.layoutInfo != null ||
      (this.artifacts && Object.keys(this.artifacts).length > 0);
    return html`
      <pic-form
        .files=${this.files}
        .formValues=${this.formValues}
        .touched=${this.touched}
        .running=${this.running}
        .submitLabel=${hasResult ? "Re-run with current values" : "Generate bin"}
        @files-changed=${(e) => { this.files = e.detail; }}
        @field-changed=${this._onFieldChange}
        @submit-job=${this._submitJob}
      ></pic-form>
    `;
  }

  _renderProgress() {
    return html`
      <pic-progress
        .events=${this.eventLog}
      ></pic-progress>
    `;
  }

  _renderPreview() {
    return html`
      <pic-preview
        .jobId=${this.jobId}
        .layoutInfo=${this.layoutInfo}
        .artifactKey=${this._artifactKey}
        .llmAvailable=${!!this.layoutInfo?.llm_available}
        .currentParams=${this.formValues}
        .running=${this.running}
        @proceed=${this._onProceed}
        @redo=${this._onRedo}
      ></pic-preview>
    `;
  }

  _renderDownloads() {
    return html`
      <pic-downloads
        .artifacts=${this.artifacts}
        .artifactKey=${this._artifactKey}
        @start-over=${this._reset}
      ></pic-downloads>
    `;
  }

  _renderError() {
    return html`
      <div class="card error-card">
        <div class="error-banner">${this.errorMessage}</div>
        <div class="actions">
          <button class="secondary" @click=${() => { this.errorMessage = null; }}>
            Dismiss
          </button>
          <button class="secondary" @click=${this._reset}>Start over</button>
        </div>
      </div>
    `;
  }

  // ---- Event handlers -----------------------------------------------------

  _onFieldChange = (e) => {
    const { name, value } = e.detail;
    this.formValues = { ...this.formValues, [name]: value };
    const t = new Set(this.touched);
    t.add(name);
    this.touched = t;
  };

  _submitJob = async () => {
    const params = this._buildParams();
    if (params == null) return;  // validation already showed an error

    const fd = new FormData();
    fd.append("params", JSON.stringify(params));
    for (const f of this.files) {
      if (f.restored) {
        // No real File object — bytes live on the server under the
        // restored job's inputs/ dir. Fetch them so the new job gets
        // the same images the original used. We use this.jobId because
        // restoration always sets it before populating this.files.
        try {
          const url = `/jobs/${encodeURIComponent(this.jobId)}/inputs/${encodeURIComponent(f.file.name)}`;
          const blob = await fetch(url).then(r => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.blob();
          });
          fd.append("images", blob, f.file.name);
        } catch (e) {
          this._fail(
            `Could not re-fetch ${f.file.name} from the prior job: ${e.message}. ` +
            `Remove and re-add it before submitting.`
          );
          return;
        }
      } else {
        fd.append("images", f.file, f.file.name);
      }
    }

    // Tear down any prior SSE so a re-submit doesn't get duplicate events
    // from the previous job's stream.
    if (this._eventSource) {
      this._eventSource.close();
      this._eventSource = null;
    }
    this.running = true;
    this.eventLog = [];
    this.layoutInfo = null;
    this.artifacts = {};
    this.errorMessage = null;
    this._seenLayoutReady = false;
    this._seenComplete = false;

    let res;
    try {
      res = await fetch("/jobs", { method: "POST", body: fd });
    } catch (e) {
      this._fail(`Network error: ${e.message}`);
      return;
    }
    if (!res.ok) {
      const body = await res.text();
      this._fail(`Server rejected job (${res.status}): ${body}`);
      return;
    }
    const { job_id } = await res.json();
    this.jobId = job_id;
    this._setJobInUrl(job_id);
    this._connectEvents(job_id);
  };

  _onProceed = async () => {
    this.running = true;
    this.errorMessage = null;
    this.eventLog = [...this.eventLog, {
      step: "bin_config", message: "Generating bin config...", fraction: 0.5,
    }];
    const res = await fetch(`/jobs/${this.jobId}/proceed`, { method: "POST" });
    if (!res.ok) {
      this._fail(`Proceed failed: ${await res.text()}`);
    }
    // Wait for SSE "complete" event; existing EventSource is still listening.
  };

  _onRedo = async (e) => {
    const { params, layoutOnly } = e.detail;
    this.formValues = { ...this.formValues, ...params };
    this.running = true;
    this.eventLog = [];
    // Drop the stale layout/downloads so the user doesn't see an outdated
    // preview while the redo runs; they'll re-appear on layout_ready/complete.
    this.layoutInfo = null;
    this.artifacts = {};
    this.errorMessage = null;
    this._seenLayoutReady = false;
    this._seenComplete = false;
    // _onComplete closes the EventSource on terminal status. When the user
    // redoes from a `complete` job, the channel is gone and layout_ready/
    // complete events from the new run wouldn't reach the UI. Reopen here;
    // the server replays event_log on subscribe so we don't miss anything.
    if (!this._eventSource) {
      this._connectEvents(this.jobId);
    }
    const res = await fetch(`/jobs/${this.jobId}/redo`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ params, layout_only: layoutOnly }),
    });
    if (!res.ok) {
      this._fail(`Redo failed: ${await res.text()}`);
    }
  };

  _reset = () => {
    if (this._eventSource) {
      this._eventSource.close();
      this._eventSource = null;
    }
    this.jobId = null;
    this.files = [];
    this.formValues = { ...FORM_DEFAULTS };
    this.touched = new Set();
    this.running = false;
    this.layoutInfo = null;
    this.artifacts = {};
    this.errorMessage = null;
    this.eventLog = [];
    this._seenLayoutReady = false;
    this._seenComplete = false;
    this._clearJobFromUrl();
    // Take the user back to the top of the page so they're looking at the
    // freshly-cleared form, not wherever the prior run left them scrolled.
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  // ---- SSE plumbing -------------------------------------------------------

  _connectEvents(jobId) {
    if (this._eventSource) this._eventSource.close();
    const es = new EventSource(`/jobs/${jobId}/events`);
    this._eventSource = es;
    es.onmessage = (msg) => {
      const ev = JSON.parse(msg.data);
      this.eventLog = [...this.eventLog, ev];
      if (ev.step === "layout_ready") {
        this._onLayoutReady();
      } else if (ev.step === "complete") {
        this._onComplete();
      } else if (ev.step === "error") {
        this._fail(ev.message || "pipeline error");
      }
    };
    // The server emits a "session_lost" event (with retry: 0) when the job
    // ID isn't in the in-memory registry — typical after a server restart.
    // Treat it as a soft failure: stop the browser's auto-reconnect, show
    // a friendly error, and let the user start over instead of leaving the
    // tab silently retrying.
    es.addEventListener("session_lost", (msg) => {
      try {
        const ev = JSON.parse(msg.data);
        this._fail(
          ev.message ||
          "Session lost — the server has restarted. Start over to continue."
        );
      } catch {
        this._fail("Session lost — the server has restarted. Start over to continue.");
      }
    });
    es.onerror = () => {
      // EventSource auto-reconnects on transport-level errors; we only
      // surface a hard failure when the server has explicitly told us via
      // the session_lost event handler above. Transient network blips are
      // left to the browser's default retry behavior.
    };
  }

  async _onLayoutReady() {
    if (this._seenLayoutReady) return;
    this._seenLayoutReady = true;
    const summary = await (await fetch(`/jobs/${this.jobId}`)).json();
    this.layoutInfo = summary;
    this._artifactKey = Date.now();
    // Phase A is done; the user is now looking at the preview deciding
    // whether to proceed. Re-enable the form's submit button.
    this.running = false;
  }

  async _onComplete() {
    if (this._seenComplete) return;
    this._seenComplete = true;
    // Job is done — close the SSE so the browser stops auto-reconnecting
    // and re-replaying the event log once the downloads section is up.
    if (this._eventSource) {
      this._eventSource.close();
      this._eventSource = null;
    }
    const summary = await (await fetch(`/jobs/${this.jobId}`)).json();
    this.artifacts = summary.artifacts || {};
    this._artifactKey = Date.now();
    this.running = false;
  }

  _fail(message) {
    this.errorMessage = message;
    this.running = false;
    if (this._eventSource) {
      this._eventSource.close();
      this._eventSource = null;
    }
  }

  // ---- Param assembly -----------------------------------------------------

  _buildParams() {
    if (this.files.length === 0) {
      alert("Drop at least one photo first.");
      return null;
    }
    const tool_heights = {};
    for (let i = 0; i < this.files.length; i++) {
      const v = this.files[i].toolHeight;
      if (v == null || v === "" || isNaN(v)) {
        alert(`Tool height required for image ${i + 1} (${this.files[i].file.name}).`);
        return null;
      }
      tool_heights[i] = Number(v);
    }
    const params = { tool_heights };
    for (const [k, v] of Object.entries(this.formValues)) {
      if (v === "" || v == null) continue;
      // SOFT_DEFAULT_FIELDS are displayed with a default value in the form
      // for context (so users see what the pipeline will use), but the
      // backend has a smarter per-photo auto-detect that should win unless
      // the user explicitly typed something. Drop these from the params
      // payload while they're still untouched so the backend takes over.
      if (SOFT_DEFAULT_FIELDS.has(k) && !this.touched.has(k)) continue;
      if (typeof v === "boolean") {
        params[k] = v;
      } else if (typeof v === "number") {
        params[k] = v;
      } else {
        // strings → coerce numerics where the field is numeric
        const n = Number(v);
        params[k] = !isNaN(n) && /^[\d.\-+eE]+$/.test(String(v)) ? n : v;
      }
    }
    return params;
  }
}

customElements.define("pic-app", PicApp);

// ---------------------------------------------------------------------------
// pic-form — the input form
// ---------------------------------------------------------------------------

// Helper: dispatch a show-info event that bubbles to <pic-app>.
function showInfo(el, field) {
  el.dispatchEvent(new CustomEvent("show-info", {
    detail: { field }, bubbles: true, composed: true,
  }));
}

const _isHeic = (filename) => /\.(heic|heif)$/i.test(filename || "");

class PicForm extends LitElement {
  static properties = {
    files: { type: Array },
    formValues: { type: Object },
    touched: { type: Object },
    running: { type: Boolean },
    submitLabel: { type: String },
    dragOver: { state: true },
  };
  createRenderRoot() { return this; }

  constructor() {
    super();
    this.dragOver = false;
    this.running = false;
    this.submitLabel = "Generate bin";
  }

  render() {
    const allHeightsSet = this.files.length > 0 &&
      this.files.every(f => f.toolHeight != null && f.toolHeight !== "" && !isNaN(f.toolHeight));
    const submitDisabled = !allHeightsSet || this.running;
    return html`
      ${this._renderTemplateSetup()}

      <div class="card">
        <div class="card-header">
          <h2>1. Part name <span class="card-h2-sub">(optional)</span></h2>
          ${this._renderInfoLink("part_name", "What is this?")}
        </div>
        <input type="text" class="part-name-input"
               placeholder="Your Tool Name"
               .value=${this.formValues.part_name ?? ""}
               @input=${(e) => this._emit("part_name", e.target.value)}>
        <p class="hint">Used as a prefix for downloaded filenames. Leave blank to keep defaults.</p>
      </div>

      <div class="card">
        <div class="card-header">
          <h2>2. Photos</h2>
          ${this._renderInfoLink("photo_tips", "Photo tips")}
        </div>
        <div class="dropzone ${this.dragOver ? "over" : ""}"
             @click=${() => this.querySelector("input[type=file]").click()}
             @dragover=${this._onDragOver}
             @dragleave=${() => this.dragOver = false}
             @drop=${this._onDrop}>
          <input type="file" multiple accept=".png,.jpg,.jpeg,.heic,.heif"
                 @change=${this._onPick}>
          <p><strong>Drag photos here</strong> or click to choose.</p>
          <p class="hint">JPG / PNG / HEIC. Each photo must contain the printed ArUco template.</p>
        </div>
        ${this._renderThumbs()}
      </div>

      <div class="card">
        <h2>3. Setup</h2>
        <div class="field-row">
          ${this._renderField("paper_size", "select", {
            options: ["a4", "a5", "letter", "legal"],
          })}
        </div>
        ${this._renderTaperField()}
        <div class="field-row toggles">
          ${this._renderField("stacking", "checkbox")}
          ${this._renderField("slots", "checkbox")}
        </div>
      </div>

      <details class="card advanced-card">
        <summary class="advanced-summary">
          <h2>4. Advanced <span class="card-h2-sub">(optional)</span></h2>
          <span class="advanced-toggle-hint">Click to expand</span>
        </summary>

        <h3 class="advanced-subhead">Tool fitting</h3>
        <div class="field-row">
          ${this._renderField("tolerance", "number", { step: 0.1 })}
          ${this._renderField("axial_tolerance", "number", { step: 0.1, placeholder: "auto" })}
        </div>
        <div class="field-row">
          ${this._renderField("gap", "number", { step: 0.1 })}
          ${this._renderField("bin_margin", "number", { step: 0.1 })}
        </div>

        <h3 class="advanced-subhead">Bin sizing</h3>
        <div class="field-row">
          ${this._renderField("min_units", "number", { step: 1 })}
          ${this._renderField("max_units", "number", { step: 1 })}
          ${this._renderField("height_units", "number", { step: 1, placeholder: "auto" })}
        </div>

        <h3 class="advanced-subhead">Tracing tuning</h3>
        <p class="hint section-hint">
          These tune the SAM2 cleanup and vectorization. Defaults are good
          for most photos — only change if your traces look noisy or wrong.
        </p>
        <div class="field-row">
          ${this._renderField("straighten_threshold", "number", { step: 1 })}
          ${this._renderField("max_refine_iterations", "number", { step: 1 })}
          ${this._renderField("max_concavity_depth", "number", { step: 0.1 })}
          ${this._renderField("mask_erode", "number", { step: 0.05 })}
          ${this._renderField("display_smooth_sigma", "number", { step: 0.1 })}
        </div>
      </details>

      <div class="card submit-card">
        <button class="primary" ?disabled=${submitDisabled}
                @click=${() => this.dispatchEvent(new CustomEvent("submit-job"))}>
          ${this.running ? "Working…" : this.submitLabel}
        </button>
        ${!allHeightsSet ? html`
          <p class="hint">Drop at least one photo and fill in tool height to enable.</p>
        ` : nothing}
      </div>

    `;
  }

  _renderThumbImage(f) {
    if (f.dataUrl) {
      return html`<img src=${f.dataUrl} alt=${f.file.name}>`;
    }
    if (f.converting) {
      return html`
        <div class="thumb-placeholder converting">
          <span class="thumb-spinner" aria-hidden="true"></span>
          <span>Converting HEIC…</span>
        </div>
      `;
    }
    if (_isHeic(f.file.name)) {
      // heic2any failed — keep the form usable but show a clear placeholder.
      return html`
        <div class="thumb-placeholder">
          <strong>HEIC</strong>
          <span>Preview unavailable</span>
        </div>
      `;
    }
    return html`<img src=${f.dataUrl} alt=${f.file.name}>`;
  }

  _renderThumbs() {
    if (this.files.length === 0) return nothing;
    return html`
      <div class="thumbs">
        ${this.files.map((f, i) => html`
          <div class="thumb">
            ${this._renderThumbImage(f)}
            <div class="name" title=${f.file.name}>${f.file.name}</div>
            <label class="thumb-field">
              <span class="thumb-label">
                Tool height (mm) *
                <button class="info-btn" type="button"
                        @click=${(e) => { e.stopPropagation(); showInfo(this, "tool_height"); }}
                        aria-label="About tool height">i</button>
              </span>
              <input type="number" step="0.1" .value=${f.toolHeight ?? ""}
                     @input=${(e) => this._setHeight(i, e.target.value)}>
            </label>
            <button class="remove" type="button" @click=${() => this._remove(i)}>Remove</button>
          </div>
        `)}
      </div>
    `;
  }

  _renderTemplateSetup() {
    const size = this._templateSize ?? "letter";
    const bg = this._templateBg ?? "white";
    const href = `/static/templates/template_${size}_${bg}.pdf`;
    const filename = `pic_to_bin_template_${size}_${bg}.pdf`;
    const sizeOptions = [
      { value: "letter", label: "US Letter (8.5 × 11 in)" },
      { value: "legal",  label: "US Legal (8.5 × 14 in)" },
      { value: "a5",     label: "A5 (148 × 210 mm)" },
    ];
    const bgOptions = [
      { value: "white", label: "White" },
      { value: "green", label: "Chroma-key green (#00B140)" },
    ];
    return html`
      <details class="card advanced-card setup-card">
        <summary class="advanced-summary">
          <h2>0. Setup <span class="card-h2-sub">— print template</span></h2>
          <span class="advanced-toggle-hint">Click to expand</span>
        </summary>
        <div class="card-header" style="margin: 0 0 0.5rem;">
          <span class="field-label" style="font-weight: 600;">Download a printable ArUco template</span>
          ${this._renderInfoLink("template_setup", "About this step")}
        </div>
        <p class="hint section-hint">
          Print at 100% scale (no fit-to-page). Each photo must include this
          template — its markers calibrate scale and correct perspective. The
          green variant helps SAM2 segment light or reflective tools.
        </p>
        <div class="field-row">
          <div class="field">
            <label><span class="field-label">Paper size</span></label>
            <select @change=${(e) => { this._templateSize = e.target.value; this.requestUpdate(); }}>
              ${sizeOptions.map(o => html`
                <option value=${o.value} ?selected=${o.value === size}>${o.label}</option>
              `)}
            </select>
          </div>
          <div class="field">
            <label><span class="field-label">Background</span></label>
            <select @change=${(e) => { this._templateBg = e.target.value; this.requestUpdate(); }}>
              ${bgOptions.map(o => html`
                <option value=${o.value} ?selected=${o.value === bg}>${o.label}</option>
              `)}
            </select>
          </div>
        </div>
        <div class="actions" style="margin-top: 0.75rem;">
          <a class="primary button-link" href=${href} download=${filename}>
            Download template (PDF)
          </a>
        </div>
      </details>
    `;
  }

  _renderTaperField() {
    const info = FIELD_INFO.tool_taper;
    const value = this.formValues.tool_taper;
    // Side-profile silhouettes on a ground line. Trapezoid points use
    // bottom edge at y=44, top at y=8 so the top/bottom width difference
    // is visually obvious.
    //
    // "top", "uniform", and "widest in the middle" use identical parallax
    // math (z ≈ tool_height/2), so they're presented as a single radio that
    // emits "top". All three silhouettes are shown side by side inside the
    // option so users can recognize their tool's shape.
    const options = [
      {
        value: "top",
        label: "Widest at top, uniform, or widest in the middle",
        icon: html`
          <svg viewBox="0 0 200 56" width="220" height="64" aria-hidden="true">
            <line x1="2" y1="48" x2="62" y2="48"
                  stroke="currentColor" stroke-width="1.5" opacity="0.35"/>
            <line x1="70" y1="48" x2="130" y2="48"
                  stroke="currentColor" stroke-width="1.5" opacity="0.35"/>
            <line x1="138" y1="48" x2="198" y2="48"
                  stroke="currentColor" stroke-width="1.5" opacity="0.35"/>
            <polygon points="22,44 42,44 60,8 4,8"
                     fill="currentColor" fill-opacity="0.18"
                     stroke="currentColor" stroke-width="2"
                     stroke-linejoin="round"/>
            <rect x="82" y="8" width="36" height="36"
                  fill="currentColor" fill-opacity="0.18"
                  stroke="currentColor" stroke-width="2"
                  stroke-linejoin="round"/>
            <ellipse cx="168" cy="26" rx="26" ry="18"
                     fill="currentColor" fill-opacity="0.18"
                     stroke="currentColor" stroke-width="2"/>
          </svg>`,
      },
      {
        value: "bottom",
        label: "Widest at bottom",
        icon: html`
          <svg viewBox="0 0 64 56" width="80" height="64" aria-hidden="true">
            <line x1="2" y1="48" x2="62" y2="48"
                  stroke="currentColor" stroke-width="1.5" opacity="0.35"/>
            <polygon points="4,44 60,44 42,8 22,8"
                     fill="currentColor" fill-opacity="0.18"
                     stroke="currentColor" stroke-width="2"
                     stroke-linejoin="round"/>
          </svg>`,
      },
    ];
    // "uniform" is still a valid backend value (CLI/legacy form data); map
    // it onto the combined "top" radio for display purposes only.
    const selectedValue = value === "uniform" ? "top" : value;
    return html`
      <div class="field taper-field">
        <label class="field-label">
          ${info.title}
          <button class="info-btn" type="button"
                  @click=${() => showInfo(this, "tool_taper")}
                  aria-label="Explain ${info.title}">i</button>
        </label>
        <div class="taper-options" role="radiogroup">
          ${options.map(o => html`
            <label class="taper-option ${selectedValue === o.value ? "selected" : ""}">
              <input type="radio" name="tool_taper" .value=${o.value}
                     .checked=${selectedValue === o.value}
                     @change=${() => this._emit("tool_taper", o.value)}>
              ${o.icon}
              <span class="taper-label">${o.label}</span>
            </label>
          `)}
        </div>
        <span class="hint">${info.hint}</span>
      </div>
    `;
  }

  _renderField(name, kind, opts = {}) {
    const info = FIELD_INFO[name] || {};
    const label = info.title || name;
    const hint = info.hint;
    const value = this.formValues[name];
    const isDefault = SOFT_DEFAULT_FIELDS.has(name) && !this.touched.has(name);
    const cls = ["field", isDefault ? "default-applied" : ""].join(" ");

    const labelEl = html`
      <span class="field-label">
        ${label}
        ${info.body ? html`
          <button class="info-btn" type="button"
                  @click=${() => showInfo(this, name)}
                  aria-label="Explain ${label}">i</button>
        ` : nothing}
      </span>
    `;

    if (kind === "checkbox") {
      return html`
        <div class=${cls}>
          <label class="checkbox-label">
            <input type="checkbox" .checked=${!!value}
                   @change=${(e) => this._emit(name, e.target.checked)}>
            ${labelEl}
          </label>
          ${hint ? html`<span class="hint">${hint}</span>` : nothing}
        </div>
      `;
    }
    if (kind === "select") {
      return html`
        <div class=${cls}>
          <label>${labelEl}</label>
          <select @change=${(e) => this._emit(name, e.target.value)}>
            ${opts.options.map(o => html`
              <option value=${o} ?selected=${o === value}>${o}</option>
            `)}
          </select>
          ${hint ? html`<span class="hint">${hint}</span>` : nothing}
        </div>
      `;
    }
    return html`
      <div class=${cls}>
        <label>${labelEl}</label>
        <input type=${kind} .value=${value === "" ? "" : String(value)}
               step=${opts.step ?? "any"}
               placeholder=${opts.placeholder ?? ""}
               @input=${(e) => this._emit(name, e.target.value === "" ? "" : Number(e.target.value))}>
        ${isDefault ? html`<span class="hint default-hint">Default applied — confirm or change.</span>` : nothing}
        ${hint ? html`<span class="hint">${hint}</span>` : nothing}
      </div>
    `;
  }

  _renderInfoLink(fieldName, label) {
    if (!FIELD_INFO[fieldName]) return nothing;
    return html`
      <button class="info-link" type="button"
              @click=${() => showInfo(this, fieldName)}>
        <span class="info-btn">i</span> ${label}
      </button>
    `;
  }


  _emit(name, value) {
    this.dispatchEvent(new CustomEvent("field-changed", {
      detail: { name, value },
    }));
  }

  _onDragOver = (e) => {
    e.preventDefault();
    this.dragOver = true;
  };

  _onDrop = (e) => {
    e.preventDefault();
    this.dragOver = false;
    this._addFiles(e.dataTransfer.files);
  };

  _onPick = (e) => {
    this._addFiles(e.target.files);
    e.target.value = "";
  };

  async _addFiles(fileList) {
    // Add files to state immediately with placeholder previews so the user
    // sees thumbnails right away. HEIC takes a couple of seconds to convert
    // via heic2any (libheif WASM), so we don't want to block the form.
    const incoming = Array.from(fileList).map(file => ({
      file,
      toolHeight: null,
      dataUrl: "",
      converting: _isHeic(file.name),
    }));
    this.dispatchEvent(new CustomEvent("files-changed", {
      detail: [...this.files, ...incoming],
    }));

    // Resolve each preview asynchronously, dispatching updates as they
    // complete. Re-read this.files each iteration so a removal or another
    // add operation while a conversion is in flight isn't clobbered.
    for (const entry of incoming) {
      const dataUrl = await this._readDataUrl(entry.file);
      const updated = this.files.map(e => e.file === entry.file
        ? { ...e, dataUrl, converting: false }
        : e
      );
      this.dispatchEvent(new CustomEvent("files-changed", { detail: updated }));
    }
  }

  async _readDataUrl(file) {
    // Browsers other than Safari can't render HEIC. The bundled-WASM
    // shims (heic2any) ship an old libheif that fails on modern iPhone
    // HEIC variants — POST to the server's /preview endpoint instead,
    // which decodes via pillow-heif (the same lib the ingest pipeline
    // uses, so it handles whatever the pipeline supports).
    if (_isHeic(file.name)) {
      try {
        const fd = new FormData();
        fd.append("image", file, file.name);
        const res = await fetch("/preview", { method: "POST", body: fd });
        if (!res.ok) {
          throw new Error(`/preview HTTP ${res.status}: ${await res.text()}`);
        }
        const blob = await res.blob();
        return URL.createObjectURL(blob);
      } catch (err) {
        console.warn(`HEIC preview failed for ${file.name}:`, err);
        return "";
      }
    }
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => resolve("");
      reader.readAsDataURL(file);
    });
  }

  _setHeight(i, value) {
    const next = this.files.map((f, idx) => idx === i ? { ...f, toolHeight: value } : f);
    this.dispatchEvent(new CustomEvent("files-changed", { detail: next }));
  }

  _remove(i) {
    const next = this.files.filter((_, idx) => idx !== i);
    this.dispatchEvent(new CustomEvent("files-changed", { detail: next }));
  }
}

customElements.define("pic-form", PicForm);

// ---------------------------------------------------------------------------
// pic-progress — step tracker driven by SSE event log
// ---------------------------------------------------------------------------

const STEP_DEFINITIONS = [
  { key: "preprocess_trace", label: "Preprocess + trace photos",
    matchSteps: new Set(["preprocess", "trace"]) },
  { key: "layout",           label: "Pack layout",
    matchSteps: new Set(["layout", "layout_ready"]) },
  { key: "bin_config",       label: "Generate bin config",
    matchSteps: new Set(["bin_config", "done", "complete"]) },
];

class PicProgress extends LitElement {
  static properties = {
    events: { type: Array },
  };
  createRenderRoot() { return this; }

  render() {
    const stepStates = this._computeStepStates();
    const subSteps = this._computeImageSubsteps();
    return html`
      <div class="card">
        <h2>5. Working...</h2>
        <ul class="steps">
          ${stepStates.map((s, i) => html`
            <li class=${s.cls}>
              <span class="step-icon">${s.cls === "done" ? "✓" : (i + 1)}</span>
              <div>
                <div>${s.label}</div>
                ${s.message ? html`<div class="hint">${s.message}</div>` : nothing}
                ${i === 0 && subSteps.length > 0 ? html`
                  <ul class="substeps">
                    ${subSteps.map(ss => html`<li>${ss.icon} ${ss.text}</li>`)}
                  </ul>
                ` : nothing}
              </div>
            </li>
          `)}
        </ul>
      </div>
    `;
  }

  _computeStepStates() {
    const lastByStep = new Map();
    for (const ev of this.events) lastByStep.set(ev.step, ev);

    const seenSteps = new Set(this.events.map(e => e.step));
    const hasError = seenSteps.has("error") &&
                     !this.events.some(e => e.step === "complete" || e.step === "layout_ready");

    return STEP_DEFINITIONS.map((def, idx) => {
      const seen = [...def.matchSteps].some(s => seenSteps.has(s));
      const finished = this._isStepFinished(def, seenSteps);
      let cls = "";
      if (finished) cls = "done";
      else if (seen) cls = "active";
      if (hasError && !finished && seen) cls = "error";
      const lastEv = [...def.matchSteps]
        .map(s => lastByStep.get(s))
        .filter(Boolean)
        .pop();
      return { ...def, cls, message: lastEv?.message || "" };
    });
  }

  _isStepFinished(def, seenSteps) {
    if (def.key === "preprocess_trace") {
      return seenSteps.has("layout") || seenSteps.has("layout_ready") ||
             seenSteps.has("bin_config") || seenSteps.has("complete");
    }
    if (def.key === "layout") {
      return seenSteps.has("layout_ready") || seenSteps.has("bin_config") ||
             seenSteps.has("complete");
    }
    if (def.key === "bin_config") {
      return seenSteps.has("complete");
    }
    return false;
  }

  _computeImageSubsteps() {
    // Last event per (step, image_index) gives current state for each image.
    const byImage = new Map();
    for (const ev of this.events) {
      if (ev.image_name == null) continue;
      const key = ev.image_name;
      byImage.set(key, ev);
    }
    return [...byImage.values()].map(ev => {
      let icon = "•";
      if (ev.step === "error") icon = "✗";
      else if (ev.step === "trace") icon = "…";
      else if (ev.step === "preprocess") icon = "…";
      return { icon, text: `${ev.message}` };
    });
  }
}

customElements.define("pic-progress", PicProgress);

// ---------------------------------------------------------------------------
// pic-preview — layout preview with proceed/redo
// ---------------------------------------------------------------------------

// Pretty labels for the diff list shown to the user when the LLM suggests
// parameter changes. Keys must match the suggested_params schema in
// pic_to_bin/web/llm_check.py.
const SUGGESTED_PARAM_LABELS = {
  tolerance: "Tolerance (mm)",
  axial_tolerance: "Axial tolerance (mm)",
  display_smooth_sigma: "Smoothing strength (mm)",
  mask_erode: "Mask erosion (mm)",
  bin_margin: "Bin margin (mm)",
  gap: "Gap (mm)",
};

class PicPreview extends LitElement {
  static properties = {
    jobId: { type: String },
    layoutInfo: { type: Object },
    artifactKey: { type: Number },
    llmAvailable: { type: Boolean },
    currentParams: { type: Object },
    running: { type: Boolean },
    llmBusy: { state: true },
    llmVerdict: { state: true },
    llmIterations: { state: true },
    llmOverlays: { state: true },   // [{stem, url, width_mm, height_mm}] returned by /llm_evaluate
    llmError: { state: true },
    llmAutoLoop: { state: true },
    llmMaxIterations: { state: true },
    showAdvanced: { state: true },
    correctivePoints: { state: true }, // { stem: [{x_mm, y_mm, label}], ... }
    correctiveMode: { state: true },   // "negative" | "positive"
    overlaysBusy: { state: true },     // true while POST /overlays is in flight
    correctiveError: { state: true },  // human-readable error to surface near the controls
  };
  createRenderRoot() { return this; }

  constructor() {
    super();
    this.artifactKey = 0;
    this.llmAvailable = false;
    this.currentParams = {};
    this.running = false;
    this.llmBusy = false;
    this.llmVerdict = null;
    this.llmIterations = 0;
    this.llmOverlays = [];
    this.llmError = null;
    this.llmAutoLoop = false;
    this.llmMaxIterations = 3;
    this.showAdvanced = false;
    this.correctivePoints = {};
    this.correctiveMode = "negative";
    this.overlaysBusy = false;
    this.correctiveError = null;
  }

  // pic-app sets `currentParams` and `layoutInfo` before each render. When the
  // backend regenerates the layout (after auto-loop or manual redo), we want
  // any stale verdict that referenced the OLD layout to clear so the user
  // sees the fresh preview unbiased by yesterday's reasoning.
  willUpdate(changed) {
    if (changed.has("artifactKey") && this.llmVerdict !== null) {
      this.llmVerdict = null;
      this.llmIterations = 0;
      this.llmOverlays = [];
      this.llmError = null;
    }
    // Re-seed corrective points from the canonical applied state ONLY
    // when a fresh layout lands (artifactKey bumps) AND that fresh state
    // explicitly includes a sam_corrective_points dict. We never clear
    // local clicks on a redo whose params didn't touch corrective points
    // (e.g. a "Apply suggested tolerance" redo) — the user's pending
    // clicks should survive that round-trip.
    if (changed.has("artifactKey")) {
      const fromServer = this.currentParams?.sam_corrective_points;
      if (fromServer && typeof fromServer === "object") {
        this.correctivePoints = JSON.parse(JSON.stringify(fromServer));
      }
    }
    // Auto-fetch the per-tool overlays as soon as a fresh layout exists.
    // Generating them is just matplotlib drawing the trace DXF onto the
    // rectified photo — no LLM round-trip — so there's no reason to gate
    // it behind a button click. Dedupe by artifactKey so we don't spam
    // the endpoint on every unrelated re-render.
    if (
      changed.has("artifactKey") &&
      this.artifactKey &&
      this.jobId &&
      this._lastOverlayFetchKey !== this.artifactKey
    ) {
      this._lastOverlayFetchKey = this.artifactKey;
      // Defer to a microtask so we don't trigger a state mutation inside
      // willUpdate. _onShowOverlaysForClicks is gated by overlaysBusy /
      // llmBusy / running, so it self-suppresses if it's not the right
      // moment to fetch.
      queueMicrotask(() => this._onShowOverlaysForClicks(true));
    }
  }

  render() {
    const k = this.artifactKey;
    const url = this.layoutInfo?.artifacts?.layout_preview;
    const pdfUrl = this.layoutInfo?.artifacts?.fit_test_pdf;
    const svgUrl = this.layoutInfo?.artifacts?.fit_test_svg;
    return html`
      <div class="card">
        <h2>6. Layout preview</h2>
        ${this.layoutInfo ? html`
          <p class="hint">Bin: ${this.layoutInfo.grid_units_x} L × ${this.layoutInfo.grid_units_y} W${this.layoutInfo.grid_units_z != null ? html` × ${this.layoutInfo.grid_units_z} H` : nothing} gridfinity units</p>
        ` : nothing}
        ${url ? html`<img class="preview-img" src=${`${url}?v=${k}`}>` : nothing}
      </div>

      ${this.llmError ? html`
        <div class="card llm-verdict-card error">
          <p class="hint"><strong>LLM check failed.</strong> ${this.llmError}</p>
        </div>
      ` : nothing}

      ${this.llmVerdict ? this._renderVerdictCard() : nothing}
      ${this._renderStandaloneCorrectiveCard()}

      ${pdfUrl || svgUrl ? html`
        <div class="card fit-test-card">
          <div class="card-header">
            <h2>8. Test the fit before printing</h2>
            <button class="info-link" type="button"
                    @click=${() => showInfo(this, "fit_test")}>
              <span class="info-btn">i</span> What is this?
            </button>
          </div>
          <p class="hint">
            Print this layout at <strong>actual size / 100% scale</strong>
            (not "fit to page") and lay your real tool on top to verify
            it fits before you commit to a multi-hour 3D print. Both files
            encode the bin's exact mm dimensions so they print at 1:1 on
            any printer.
          </p>
          <div class="downloads">
            ${pdfUrl ? html`
              <a href=${`${pdfUrl}?v=${k}`} download>
                Layout — actual size (PDF)
              </a>` : nothing}
            ${svgUrl ? html`
              <a href=${`${svgUrl}?v=${k}`} download>
                Layout — actual size (SVG)
              </a>` : nothing}
          </div>
        </div>
      ` : nothing}

      <div class="card">
        <h2>9. Looks good?</h2>
        <div class="actions">
          <button class="primary"
                  ?disabled=${this.llmBusy || this.running}
                  @click=${() => this.dispatchEvent(new CustomEvent("proceed"))}>
            ${this.running ? "Working…" : "Proceed → generate bin config"}
          </button>
          <button class="primary"
                  ?disabled=${this.llmBusy || this.running || this.overlaysBusy}
                  @click=${this._onShowOverlaysForClicks}>
            ${this.overlaysBusy
              ? html`Loading overlays…`
              : html`Add corrective clicks`}
          </button>
          ${this.llmAvailable ? html`
            <button class="secondary"
                    ?disabled=${this.llmBusy || this.running}
                    @click=${this._onCheckWithLlm}>
              ${this.llmBusy
                ? (this.llmAutoLoop
                    ? html`Asking LLM (auto-loop)…`
                    : html`Asking LLM…`)
                : html`Check or refine with LLM`}
            </button>
          ` : nothing}
        </div>
        ${this._renderAdvancedToggle()}
      </div>
    `;
  }

  _renderAdvancedToggle() {
    if (!this.llmAvailable) return nothing;
    return html`
      <div class="advanced-llm">
        <button class="advanced-toggle" type="button"
                @click=${() => this.showAdvanced = !this.showAdvanced}
                aria-expanded=${this.showAdvanced ? "true" : "false"}>
          ${this.showAdvanced ? "▾" : "▸"} Advanced
        </button>
        ${this.showAdvanced ? html`
          <div class="advanced-llm-body">
            <label class="checkbox">
              <input type="checkbox"
                     .checked=${this.llmAutoLoop}
                     @change=${(e) => this.llmAutoLoop = e.target.checked}>
              Auto-loop until LLM approves (or max iterations reached)
            </label>
            <label class="field">
              Max iterations
              <input type="number" min="1" max="10" step="1"
                     .value=${String(this.llmMaxIterations)}
                     @input=${(e) => {
                       const v = parseInt(e.target.value, 10);
                       if (!isNaN(v)) this.llmMaxIterations = Math.max(1, Math.min(10, v));
                     }}
                     ?disabled=${!this.llmAutoLoop}>
            </label>
            <p class="hint">
              When enabled, the LLM's suggested parameter changes are applied
              automatically and the layout is re-evaluated, up to the iteration
              cap. When off (default), the LLM proposes a single change and
              you approve or cancel it.
            </p>
          </div>
        ` : nothing}
      </div>
    `;
  }

  // -------- Corrective-points click UI --------------------------------------

  _correctiveTotal() {
    return Object.values(this.correctivePoints || {})
      .reduce((acc, list) => acc + (Array.isArray(list) ? list.length : 0), 0);
  }

  _renderClickableOverlay(o, cacheKey) {
    const stem = o.stem;
    const wMm = Number(o.width_mm) || 0;
    const hMm = Number(o.height_mm) || 0;
    const clickable = wMm > 0 && hMm > 0;
    const points = (this.correctivePoints[stem] || []);
    return html`
      <figure class="overlay-figure">
        <div class="overlay-clicker ${clickable ? '' : 'no-coord-frame'}"
             title=${clickable
               ? `Click to add a ${this.correctiveMode} corrective point. Frame: ${wMm.toFixed(1)} × ${hMm.toFixed(1)} mm`
               : "Overlay mm dimensions unavailable; clicking disabled."}
             @click=${clickable ? (e) => this._onOverlayClick(e, stem, wMm, hMm) : null}>
          <img src=${`${o.url}?v=${cacheKey}`} alt="Overlay for ${stem}">
          ${points.map((p, i) => {
            const xPct = (p.x_mm / wMm) * 100;
            const yPct = (p.y_mm / hMm) * 100;
            return html`
              <span class="click-dot ${p.label === 0 ? 'negative' : 'positive'}"
                    style="left:${xPct}%; top:${yPct}%"
                    title="${p.label === 0 ? 'Negative' : 'Positive'} click at (${p.x_mm.toFixed(1)}, ${p.y_mm.toFixed(1)}) mm — click to remove"
                    @click=${(e) => { e.stopPropagation(); this._onRemovePoint(stem, i); }}>
                ${p.label === 0 ? '−' : '+'}
              </span>
            `;
          })}
        </div>
        <figcaption>
          ${stem}
          ${clickable ? html`<span class="overlay-dims-hint">(${wMm.toFixed(0)} × ${hMm.toFixed(0)} mm)</span>` : nothing}
        </figcaption>
      </figure>
    `;
  }

  _onOverlayClick = (e, stem, wMm, hMm) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const xRatio = (e.clientX - rect.left) / rect.width;
    const yRatio = (e.clientY - rect.top) / rect.height;
    if (xRatio < 0 || xRatio > 1 || yRatio < 0 || yRatio > 1) return;
    const label = this.correctiveMode === "positive" ? 1 : 0;
    const newPoint = {
      x_mm: +(xRatio * wMm).toFixed(2),
      y_mm: +(yRatio * hMm).toFixed(2),
      label,
    };
    const list = this.correctivePoints[stem] ? [...this.correctivePoints[stem]] : [];
    list.push(newPoint);
    this.correctivePoints = { ...this.correctivePoints, [stem]: list };
  };

  _onRemovePoint = (stem, index) => {
    const list = (this.correctivePoints[stem] || []).filter((_, i) => i !== index);
    const next = { ...this.correctivePoints };
    if (list.length === 0) delete next[stem]; else next[stem] = list;
    this.correctivePoints = next;
  };

  _renderCorrectiveControls(overlays) {
    const total = this._correctiveTotal();
    const haveCoordFrame = overlays.some(o => Number(o.width_mm) > 0 && Number(o.height_mm) > 0);
    if (!haveCoordFrame) return nothing;
    const submitting = this.running;
    return html`
      <div class="corrective-controls">
        <p class="hint">
          Click on an overlay to add a corrective click for SAM2. Use
          <strong>negative</strong> on a region wrongly included in the inner
          trace (e.g. background between handles), <strong>positive</strong>
          on a tool region the trace missed. Click an existing dot to remove
          it.
        </p>
        <div class="corrective-row">
          <div class="corrective-mode" role="group" aria-label="Click mode">
            <button class="mode-btn negative ${this.correctiveMode === 'negative' ? 'active' : ''}"
                    type="button"
                    @click=${() => this.correctiveMode = 'negative'}>
              − Negative (not tool)
            </button>
            <button class="mode-btn positive ${this.correctiveMode === 'positive' ? 'active' : ''}"
                    type="button"
                    @click=${() => this.correctiveMode = 'positive'}>
              + Positive (is tool)
            </button>
          </div>
          <div class="corrective-actions">
            <span class="corrective-count">
              ${total === 0 ? "no clicks yet" : `${total} click${total === 1 ? "" : "s"} pending`}
            </span>
            <button class="secondary"
                    type="button"
                    ?disabled=${total === 0 || this.llmBusy || this.running}
                    @click=${() => this.correctivePoints = {}}>
              Clear
            </button>
            <button class="primary"
                    type="button"
                    ?disabled=${total === 0 || this.llmBusy || submitting}
                    @click=${this._onApplyCorrective}>
              ${submitting
                ? "Submitting…"
                : `Apply ${total} corrective click${total === 1 ? "" : "s"} & re-run`}
            </button>
          </div>
        </div>
        ${this.correctiveError ? html`
          <p class="hint error-hint">${this.correctiveError}</p>
        ` : nothing}
      </div>
    `;
  }

  _onApplyCorrective = () => {
    if (this._correctiveTotal() === 0) return;
    this.correctiveError = null;
    // Send the FULL current set (not a delta) so the backend's
    // sam_corrective_points dict is the canonical applied state. The
    // server-side redo handler force-overrides layout_only=False when
    // sam_corrective_points is present, but we send false here too so
    // the round-trip is explicit.
    const detail = {
      params: { sam_corrective_points: this.correctivePoints },
      layoutOnly: false,
    };
    console.log("[corrective] dispatching redo", detail);
    this.dispatchEvent(new CustomEvent("redo", {
      detail,
      bubbles: true,
      composed: true,
    }));
    // dispatchEvent is synchronous: if pic-app's @redo listener fired,
    // its handler already ran the synchronous body of _onRedo (which
    // sets pic-app.running = true) before dispatchEvent returned. If
    // pic-app.running is still false here, the listener didn't catch
    // the event — fall back to invoking _onRedo directly so the user
    // never sees a silent no-op.
    const picApp = document.querySelector("pic-app");
    if (picApp && !picApp.running) {
      console.warn("[corrective] redo event not caught; invoking pic-app._onRedo directly");
      if (typeof picApp._onRedo === "function") {
        picApp._onRedo({ detail });
      } else {
        this.correctiveError =
          "Could not trigger re-run — pic-app handler missing. Try a hard reload (Ctrl-Shift-R).";
      }
    }
  };

  _onShowOverlaysForClicks = async (auto = false) => {
    if (this.overlaysBusy || this.llmBusy || this.running) return;
    // If the user clicks the button after overlays already loaded
    // (auto-fetch on layout-ready), just scroll to them rather than
    // refetching. The auto path skips this — it's the one that loads
    // them in the first place.
    if (!auto && this.llmOverlays && this.llmOverlays.length > 0) {
      const grid = this.querySelector(".overlay-grid");
      if (grid) grid.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    this.overlaysBusy = true;
    this.correctiveError = null;
    try {
      const res = await fetch(`/jobs/${this.jobId}/overlays`, {
        method: "POST",
        headers: { "content-type": "application/json" },
      });
      if (!res.ok) {
        throw new Error((await res.text()) || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const list = Array.isArray(data.overlays) ? data.overlays : [];
      if (list.length === 0 && !auto) {
        // Only surface this as an error on a manual click. The auto
        // path runs unsolicited; an empty list there shouldn't pop a
        // scary message in front of the user.
        throw new Error("No overlays were generated for this job.");
      }
      this.llmOverlays = list;
    } catch (err) {
      // Same rationale: auto-fetch failures are silent. The user can
      // click "Add corrective clicks" manually to see the real error.
      if (!auto) {
        this.correctiveError = err.message || String(err);
      }
    } finally {
      this.overlaysBusy = false;
    }
  };

  _renderOverlayLegend() {
    return html`
      <p class="hint overlay-legend">
        <span class="overlay-legend-swatch" style="border-color:#e63946;background:rgba(230,57,70,0.22)">red</span>
        = inner trace (the tool region SAM2 segmented),
        <span class="overlay-legend-swatch" style="border-color:#ffa600;border-style:dashed">orange dashed</span>
        = tolerance perimeter (the line the bin will cut),
        <span class="overlay-legend-swatch" style="border-color:#1d70b8;border-style:dotted">blue dotted</span>
        = finger slot.
      </p>
    `;
  }

  _renderStandaloneCorrectiveCard() {
    // Only show when overlays exist but no verdict — otherwise the
    // controls live inside the verdict card and we'd be rendering
    // them twice.
    if (this.llmVerdict) return nothing;
    if (this.correctiveError && (!this.llmOverlays || this.llmOverlays.length === 0)) {
      return html`
        <div class="card error-card">
          <p class="hint"><strong>Couldn't load overlays.</strong> ${this.correctiveError}</p>
        </div>
      `;
    }
    if (!this.llmOverlays || this.llmOverlays.length === 0) return nothing;
    const k = this.artifactKey;
    return html`
      <div class="card corrective-card">
        <h2>7. Corrective clicks <span class="card-h2-sub">(optional)</span></h2>
        <p class="hint">
          If the inner trace below missed part of your tool or grabbed
          background you can re-segment by adding clicks on the overlays.
          Positive clicks mark regions that <em>are</em> the tool; negative
          clicks mark regions that aren't. The clicks are sent to the
          local SAM2 model as point prompts and re-run the trace — they
          don't change tolerance, smoothing, or layout.
        </p>
        ${this._renderOverlayLegend()}
        <div class="overlay-grid">
          ${this.llmOverlays.map(o => this._renderClickableOverlay(o, k))}
        </div>
        ${this._renderCorrectiveControls(this.llmOverlays)}
        ${this.correctiveError ? html`
          <p class="hint error-hint">${this.correctiveError}</p>
        ` : nothing}
      </div>
    `;
  }

  // -------- Verdict card --------------------------------------------------

  _renderVerdictCard() {
    const verdict = this.llmVerdict;
    const ok = !!verdict.ok;
    const suggested = verdict.suggested_params || {};
    const hasSuggestions = Object.keys(suggested).length > 0;
    const iterations = this.llmIterations || 1;
    const overlays = this.llmOverlays || [];
    const k = this.artifactKey;
    return html`
      <div class="card llm-verdict-card ${ok ? "ok" : "needs-fix"}">
        <h3>
          ${ok
            ? html`<span class="verdict-icon">✓</span> Layout looks good`
            : html`<span class="verdict-icon">⚠</span> Needs adjustment`}
          ${iterations > 1 ? html`<span class="iteration-tag">after ${iterations} iterations</span>` : nothing}
        </h3>
        ${overlays.length ? html`
          <div class="verdict-overlays">
            <p class="hint">
              These are the overlays the LLM evaluated — your tool photo
              with the trace polygons drawn on top at the same mm scale.
            </p>
            ${this._renderOverlayLegend()}
            <div class="overlay-grid">
              ${overlays.map(o => this._renderClickableOverlay(o, k))}
            </div>
            ${this._renderCorrectiveControls(overlays)}
          </div>
        ` : nothing}
        <p class="verdict-reasoning">${verdict.reasoning || "(no reasoning provided)"}</p>
        ${!ok && hasSuggestions ? html`
          <div class="verdict-suggestions">
            <h4>Suggested adjustments</h4>
            <table class="diff-table">
              ${Object.entries(suggested).map(([key, newValue]) => {
                const oldValue = this.currentParams[key];
                const oldDisplay = oldValue == null || oldValue === ""
                  ? "(default)"
                  : String(oldValue);
                return html`
                  <tr>
                    <td class="diff-label">${SUGGESTED_PARAM_LABELS[key] || key}</td>
                    <td class="diff-old">${oldDisplay}</td>
                    <td class="diff-arrow">→</td>
                    <td class="diff-new">${String(newValue)}</td>
                  </tr>
                `;
              })}
            </table>
            <div class="actions">
              <button class="primary"
                      ?disabled=${this.llmBusy}
                      @click=${this._onApplySuggestion}>
                Apply &amp; re-run
              </button>
              <button class="secondary"
                      @click=${() => {
                        this.llmVerdict = null;
                        this.llmIterations = 0;
                        this.llmOverlays = [];
                      }}>
                Cancel
              </button>
            </div>
          </div>
        ` : nothing}
      </div>
    `;
  }

  _onCheckWithLlm = async () => {
    if (this.llmBusy) return;
    this.llmBusy = true;
    this.llmError = null;
    this.llmVerdict = null;
    this.llmIterations = 0;
    this.llmOverlays = [];
    // Auto-loop fires internal Phase A redos that emit layout_ready /
    // progress events. From a `complete` job the SSE channel was closed
    // by _onComplete; reopen so those events reach the UI and the
    // layout-preview cache-busts after each iteration.
    const picApp = document.querySelector("pic-app");
    if (this.llmAutoLoop && picApp && !picApp._eventSource) {
      picApp._connectEvents(this.jobId);
      picApp._seenLayoutReady = false;
      picApp._seenComplete = false;
    }
    try {
      const res = await fetch(`/jobs/${this.jobId}/llm_evaluate`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          auto_loop: !!this.llmAutoLoop,
          max_iterations: this.llmMaxIterations,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const data = await res.json();
      this.llmVerdict = data.verdict;
      this.llmIterations = data.iterations || 1;
      // Each entry: { stem: str, url: "/jobs/<id>/overlays/<stem>" }.
      // Empty list when overlay generation failed for every tool —
      // verdict still renders, just without the visual.
      this.llmOverlays = Array.isArray(data.overlays) ? data.overlays : [];
    } catch (err) {
      this.llmError = err.message || String(err);
    } finally {
      this.llmBusy = false;
    }
  };

  _onApplySuggestion = () => {
    if (!this.llmVerdict || !this.llmVerdict.suggested_params) return;
    const params = { ...this.llmVerdict.suggested_params };
    if (Object.keys(params).length === 0) return;
    const layoutOnly = !Object.keys(params).some(k => TRACE_REQUIRED_FIELDS.has(k));
    this.dispatchEvent(new CustomEvent("redo", {
      detail: { params, layoutOnly },
      bubbles: true,
      composed: true,
    }));
  };
}

customElements.define("pic-preview", PicPreview);

// ---------------------------------------------------------------------------
// pic-downloads — final download links
// ---------------------------------------------------------------------------

class PicDownloads extends LitElement {
  static properties = {
    artifacts: { type: Object },
    artifactKey: { type: Number },
    copyState: { state: true },   // "idle" | "copied" | "error"
  };
  createRenderRoot() { return this; }

  constructor() {
    super();
    this.artifactKey = 0;
    this.copyState = "idle";
  }

  render() {
    const k = this.artifactKey;
    const a = this.artifacts;
    const withKey = (u) => `${u}?v=${k}`;
    return html`
      <div class="card">
        <h2>10. Proceed in Fusion</h2>
        <p>
          Your bin geometry is ready. To turn it into a printable 3D model,
          open it in Fusion 360 with the Pic-to-Bin add-in:
        </p>
        <ol class="fusion-steps">
          <li>
            <strong>Install the add-in once</strong> (if you haven't already).
            Clone or download the project, then from a terminal run
            <code>pic-to-bin-fusion install</code>. This copies the add-in
            and script into your Fusion <code>API/AddIns</code> and
            <code>API/Scripts</code> folders.
          </li>
          <li>
            <strong>Launch Fusion 360.</strong> The add-in registers a
            "Gridfinity Pic-to-Bin" button under the
            <em>Solid → Create</em> menu. (On first launch you may need to
            open <em>Utilities → Add-Ins</em>, find Pic-to-Bin in the list,
            and enable "Run on Startup".)
          </li>
          <li>
            <strong>Click the Pic-to-Bin button.</strong> Fusion opens a
            file picker.
          </li>
          <li>
            <strong>Select the bin_config.json you downloaded below.</strong>
            The add-in builds the parametric bin body, stacking lip, pocket,
            finger slots, and gridfinity base pads in a new document — a
            timeline group per phase so you can tweak features after the
            fact.
          </li>
          <li>
            <strong>Export &amp; print.</strong> Use Fusion's
            <em>File → 3D Print</em> or <em>Export</em> to produce an STL
            for your slicer. The body already has the
            "ABS (White)" appearance applied so renders look like the final
            print.
          </li>
        </ol>
        <p class="hint">
          Re-running the add-in on an edited <code>bin_config.json</code>
          (e.g. after a redo) builds a fresh document — your prior tweaks
          stay in the original file.
        </p>
        <h3 class="downloads-subhead">Downloads</h3>
        <div class="downloads">
          ${a.bin_config ? html`
            <a href=${withKey(a.bin_config)} download>
              Bin config (JSON) — open this with the Fusion add-in
            </a>` : nothing}
          ${a.layout_preview ? html`
            <a href=${withKey(a.layout_preview)} download>
              Layout preview (PNG) — for screen viewing
            </a>` : nothing}
          ${a.fit_test_pdf ? html`
            <a href=${withKey(a.fit_test_pdf)} download>
              Layout — actual size (PDF) — print at 100% to test fit
            </a>` : nothing}
          ${a.fit_test_svg ? html`
            <a href=${withKey(a.fit_test_svg)} download>
              Layout — actual size (SVG) — print at 100% to test fit
            </a>` : nothing}
          ${a.combined_dxf ? html`
            <a href=${withKey(a.combined_dxf)} download>
              Combined layout (DXF) — for CAD inspection
            </a>` : nothing}
        </div>
      </div>

      <div class="card">
        <h2>11. Save or start over</h2>
        <p class="hint">
          Bookmark or share this job's link to come back later — the URL
          contains the job ID, and the server keeps job files for 24 hours
          by default.
        </p>
        <div class="actions">
          <button class="primary" type="button" @click=${this._onCopyLink}>
            ${this.copyState === "copied"
              ? "Link copied ✓"
              : this.copyState === "error"
                ? "Copy failed — copy from address bar"
                : "Copy link to this job"}
          </button>
          <button class="secondary" @click=${() => this.dispatchEvent(new CustomEvent("start-over"))}>
            Start a new bin
          </button>
        </div>
      </div>
    `;
  }

  _onCopyLink = async () => {
    const url = window.location.href;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(url);
      } else {
        // Fallback for older browsers / non-secure contexts.
        const ta = document.createElement("textarea");
        ta.value = url;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        if (!ok) throw new Error("execCommand copy failed");
      }
      this.copyState = "copied";
    } catch {
      this.copyState = "error";
    }
    setTimeout(() => { this.copyState = "idle"; }, 2500);
  };
}

customElements.define("pic-downloads", PicDownloads);
