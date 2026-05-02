import { LitElement, html, svg, css, nothing } from "lit";

// Pipeline parameter metadata. Mirrors pipeline.run_pipeline kwargs.
// "Hard required" = fields the user must touch (no default).
// "Soft default" = fields with a default that we highlight until the user
// touches them, so they consciously confirm or change the value.
const SOFT_DEFAULT_FIELDS = new Set(["phone_height"]);

// Must match TOLERANCE_BASELINE_MM in pipeline.py — used in modal copy only.
const TOLERANCE_BASELINE_MM = 2.0;

const FORM_DEFAULTS = {
  part_name: "",
  paper_size: "legal",
  tolerance: 0.0,
  axial_tolerance: 1.0,
  phone_height: 482.0,
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
};

// Which fields, when changed on a redo, force a re-trace (expensive).
// Everything else is layout-only (cheap; cached DXFs reused).
const TRACE_REQUIRED_FIELDS = new Set([
  "phone_height",
  "tool_taper",
  "tolerance",
  "axial_tolerance",
  "slots",
  "straighten_threshold",
  "max_refine_iterations",
  "max_concavity_depth",
  "mask_erode",
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
  phone_height: {
    title: "Phone height (mm)",
    hint: "Camera height above the paper. Compensates parallax.",
    body: [
      "How far the phone camera was from the paper when you took the photo.",
      "A tool that sits above the paper appears slightly larger in the photo than it really is. The pipeline compensates by scaling the trace down by phone_height / (phone_height − tool_height/2).",
      "482 mm is a reasonable default for hand-held overhead shots. Lower values apply more aggressive compensation; set to 0 to disable parallax correction entirely.",
    ],
  },
  tool_taper: {
    title: "Tool side profile",
    hint: "Where the tool is widest, viewed from the side. Affects parallax.",
    body: [
      "Pick the side-profile shape closest to your tool. The pipeline uses this to decide which height to use when compensating for parallax — tools that taper inward (wider at the bottom) need no compensation, while tools that flare outward at the top need the full correction.",
      "Widest at top — handles, grips, anything that flares out at the top. Most hand tools fall here: screwdrivers, pliers, hammers, wrenches.",
      "Uniform — vertical sides, top outline matches bottom outline. Boxy multimeters, batteries, USB drives, blocks of metal.",
      "Widest at bottom — tapers inward going up. Zircon stud finders, computer mice, phone cases, tape-measure cases.",
      "Picking the wrong option makes the trace a few percent too small or too big, with the error growing with tool height. For a 30 mm-tall tool at the default 482 mm phone height, the swing between options is around 6%.",
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
      "The tolerance polygon is always Douglas-Peucker simplified at 0.3 mm regardless of value, so Fusion gets a clean low-point-count cut.",
    ],
  },
  axial_tolerance: {
    title: "Axial tolerance (mm)",
    hint: "Extra clearance only at the tool's tips, not its sides.",
    body: [
      "Extra clearance pushed onto each end of the tool along its long (principal) axis only. The perpendicular extent is unchanged.",
      "The SAM2 segmentation tends to under-detect tapered or thin tool tips, so the trace is shorter than the actual tool. Adding uniform tolerance to fix this would make the wider sections too loose. This setting fixes only the ends.",
      "Default 1 mm pushes each end outward by 1 mm (so the pocket is 2 mm longer overall along the axis). Increase if tool tips still don't fit; set to 0 for fully uniform tolerance.",
      "Caveat: this is a linear stretch in the rotated frame, so any features along the axis stretch slightly too. Fine for typical hand tools, less ideal for tools with internal axis-parallel features (rare).",
    ],
  },
  gap: {
    title: "Gap between tools (mm)",
    hint: "Minimum space between adjacent pockets in the layout.",
    body: [
      "When packing multiple tools into one bin, leave at least this much wall between any two pockets.",
      "Smaller values pack more tools into a smaller bin but make individual tools harder to grab. Larger values give roomier separation but grow the bin.",
    ],
  },
  bin_margin: {
    title: "Bin margin (mm)",
    hint: "Extra clearance from tool extents to the bin wall.",
    body: [
      "Extra padding between the outermost tool extent and the bin boundary, applied before snapping to a whole gridfinity unit.",
      "Usually you don't need this — the natural slack from rounding the bin size up to a whole 42 mm unit, plus the tolerance, is enough.",
      "Set this >0 to force the bin one unit larger when a tool would otherwise sit right against the wall.",
    ],
  },
  min_units: {
    title: "Minimum grid units",
    hint: "Smallest bin footprint per axis (gridfinity units, 42 mm each).",
    body: [
      "Force the bin to be at least this many units wide and tall, even if the tool would fit in something smaller.",
      "Useful when you want the bin to match an existing drawer slot or to look uniform alongside other bins in a set.",
    ],
  },
  max_units: {
    title: "Maximum grid units",
    hint: "Largest bin footprint allowed before the pipeline gives up.",
    body: [
      "Cap on how big the bin can grow per axis. If the tools don't fit in this size the pipeline raises an error rather than silently producing a giant bin.",
      "Default 7×7 is plenty for most hand tools.",
    ],
  },
  height_units: {
    title: "Bin height (units, blank = auto)",
    hint: "Force a specific bin height, or leave blank to auto-size.",
    body: [
      "Each gridfinity height unit is 7 mm.",
      "When blank (default), the pipeline picks the smallest height that fits the tallest tool plus 1 mm of floor below the pocket.",
      "Set explicitly if you need to match an existing stack or want a deeper pocket than the tool requires.",
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
    ],
  },
  max_refine_iterations: {
    title: "Max refine iterations",
    hint: "How many cleanup passes the trace is allowed to take.",
    body: [
      "After SAM2 produces an initial mask, the pipeline iteratively cleans it up — closing notches, smoothing contours — and re-checks the result.",
      "More iterations let it produce smoother outlines for noisy photos but take longer. Default 5 is a good balance.",
    ],
  },
  max_concavity_depth: {
    title: "Max concavity depth (mm)",
    hint: "How aggressively cleanup is allowed to fill in concavities.",
    body: [
      "Cleanup may fill shallow concavities to smooth the outline. This is the deepest concavity (in mm) the cleanup is allowed to lose before stopping.",
      "Increase if your tool has deliberate notches that are getting filled in. Decrease for cleaner outlines on simple shapes.",
    ],
  },
  mask_erode: {
    title: "Mask erosion (mm)",
    hint: "Pixel-shrink the SAM mask. Default 0; only enable if the trace looks fat.",
    body: [
      "Phone photos can have a soft shadow halo around the tool that SAM2 picks up as part of the mask, making the trace slightly fatter than reality.",
      "This setting erodes the mask by N millimeters before vectorization. The default is 0 because a uniform erosion disproportionately shrinks thin or tapered tool tips, which often makes those pockets too tight even when the wider sections fit.",
      "Increase to 0.3–0.5 mm only if your photo has a visible shadow halo and the trace clearly extends beyond the tool's actual outline.",
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
// Root: state machine across screens
// ---------------------------------------------------------------------------

class PicApp extends LitElement {
  static properties = {
    screen: { state: true },        // "form" | "progress" | "preview" | "downloads" | "error"
    jobId: { state: true },
    files: { state: true },         // [{file: File, toolHeight: number|null, dataUrl: string}]
    formValues: { state: true },    // dict of form field values
    touched: { state: true },       // Set of field names the user has touched
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
    this.screen = "form";
    this.jobId = null;
    this.files = [];
    this.formValues = { ...FORM_DEFAULTS };
    this.touched = new Set();
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
  }

  connectedCallback() {
    super.connectedCallback();
    // Children dispatch CustomEvent("show-info", {detail: {field}, bubbles: true})
    // and we render the modal here so it works on every screen.
    this._showInfoHandler = (e) => { this.modalField = e.detail.field; };
    this.addEventListener("show-info", this._showInfoHandler);
    this._escHandler = (e) => {
      if (e.key === "Escape" && this.modalField) this.modalField = null;
    };
    window.addEventListener("keydown", this._escHandler);

    // Browser back/forward navigates between screens instead of leaving the
    // app. The initial state is replaceState (not push) so back from the
    // form still leaves the site as the user expects.
    if (history.state == null || history.state.screen == null) {
      history.replaceState({ screen: this.screen }, "");
    }
    this._popstateHandler = (e) => {
      const target = e.state?.screen ?? "form";
      this._fromHistory = true;
      this.screen = target;
    };
    window.addEventListener("popstate", this._popstateHandler);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.removeEventListener("show-info", this._showInfoHandler);
    window.removeEventListener("keydown", this._escHandler);
    window.removeEventListener("popstate", this._popstateHandler);
  }

  updated(changed) {
    if (changed.has("screen")) {
      if (!this._fromHistory) {
        history.pushState({ screen: this.screen }, "");
      }
      this._fromHistory = false;
    }
  }

  render() {
    let screen;
    if (this.screen === "form")           screen = this._renderForm();
    else if (this.screen === "progress")  screen = this._renderProgress();
    else if (this.screen === "preview")   screen = this._renderPreview();
    else if (this.screen === "downloads") screen = this._renderDownloads();
    else if (this.screen === "error")     screen = this._renderError();
    else screen = nothing;
    return html`${screen}${this._renderModal()}`;
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
            ${info.body.map(p => html`<p>${p}</p>`)}
          </div>
          <div class="modal-footer">
            <button class="primary" type="button" @click=${close}>Got it</button>
          </div>
        </div>
      </div>
    `;
  }

  // ---- Screens ------------------------------------------------------------

  _renderForm() {
    return html`
      <pic-form
        .files=${this.files}
        .formValues=${this.formValues}
        .touched=${this.touched}
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
      <div class="card">
        <div class="error-banner">${this.errorMessage}</div>
        <button class="secondary" @click=${this._reset}>Start over</button>
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
      fd.append("images", f.file, f.file.name);
    }

    this.screen = "progress";
    this.eventLog = [];

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
    this._connectEvents(job_id);
  };

  _onProceed = async () => {
    this.screen = "progress";
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
    this.screen = "progress";
    this.eventLog = [];
    this._seenLayoutReady = false;
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
    this.screen = "form";
    this.jobId = null;
    this.files = [];
    this.formValues = { ...FORM_DEFAULTS };
    this.touched = new Set();
    this.layoutInfo = null;
    this.artifacts = {};
    this.errorMessage = null;
    this.eventLog = [];
    this._seenLayoutReady = false;
    this._seenComplete = false;
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
    es.onerror = () => {
      // EventSource auto-reconnects; only fail the UI if we're not already
      // in a terminal state.
    };
  }

  async _onLayoutReady() {
    if (this._seenLayoutReady) return;
    this._seenLayoutReady = true;
    const summary = await (await fetch(`/jobs/${this.jobId}`)).json();
    this.layoutInfo = summary;
    this._artifactKey = Date.now();
    this.screen = "preview";
  }

  async _onComplete() {
    if (this._seenComplete) return;
    this._seenComplete = true;
    // Job is done — close the SSE so the browser stops auto-reconnecting
    // and re-replaying the event log on the downloads screen.
    if (this._eventSource) {
      this._eventSource.close();
      this._eventSource = null;
    }
    const summary = await (await fetch(`/jobs/${this.jobId}`)).json();
    this.artifacts = summary.artifacts || {};
    this._artifactKey = Date.now();
    this.screen = "downloads";
  }

  _fail(message) {
    this.errorMessage = message;
    this.screen = "error";
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

class PicForm extends LitElement {
  static properties = {
    files: { type: Array },
    formValues: { type: Object },
    touched: { type: Object },
    dragOver: { state: true },
  };
  createRenderRoot() { return this; }

  constructor() {
    super();
    this.dragOver = false;
  }

  render() {
    const allHeightsSet = this.files.length > 0 &&
      this.files.every(f => f.toolHeight != null && f.toolHeight !== "" && !isNaN(f.toolHeight));
    return html`
      <div class="card">
        <div class="card-header">
          <h2>1. Part name <span class="card-h2-sub">(optional)</span></h2>
          ${this._renderInfoLink("part_name", "What is this?")}
        </div>
        <input type="text" class="part-name-input"
               placeholder="e.g. zircon_stud_finder"
               .value=${this.formValues.part_name ?? ""}
               @input=${(e) => this._emit("part_name", e.target.value)}>
        <p class="hint">Used as a prefix for downloaded filenames. Leave blank to keep defaults.</p>
      </div>

      <div class="card">
        <div class="card-header">
          <h2>2. Photos</h2>
          ${this._renderInfoLink("tool_height", "About tool height")}
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
            options: ["a4", "letter", "legal"],
          })}
          ${this._renderField("phone_height", "number", { step: 1 })}
        </div>
        ${this._renderTaperField()}
      </div>

      <div class="card">
        <h2>4. Tool fitting</h2>
        <div class="field-row">
          ${this._renderField("tolerance", "number", { step: 0.1 })}
          ${this._renderField("axial_tolerance", "number", { step: 0.1 })}
        </div>
        <div class="field-row">
          ${this._renderField("gap", "number", { step: 0.1 })}
          ${this._renderField("bin_margin", "number", { step: 0.1 })}
        </div>
      </div>

      <div class="card">
        <h2>5. Bin sizing</h2>
        <div class="field-row">
          ${this._renderField("min_units", "number", { step: 1 })}
          ${this._renderField("max_units", "number", { step: 1 })}
          ${this._renderField("height_units", "number", { step: 1, placeholder: "auto" })}
        </div>
        <div class="field-row toggles">
          ${this._renderField("stacking", "checkbox")}
          ${this._renderField("slots", "checkbox")}
        </div>

        <details>
          <summary>Advanced (tracing tuning)</summary>
          <p class="hint section-hint">
            These tune the SAM2 cleanup and vectorization. Defaults are good
            for most photos — only change if your traces look noisy or wrong.
          </p>
          <div class="field-row">
            ${this._renderField("straighten_threshold", "number", { step: 1 })}
            ${this._renderField("max_refine_iterations", "number", { step: 1 })}
            ${this._renderField("max_concavity_depth", "number", { step: 0.1 })}
            ${this._renderField("mask_erode", "number", { step: 0.05 })}
          </div>
        </details>
      </div>

      <div class="card submit-card">
        <button class="primary" ?disabled=${!allHeightsSet}
                @click=${() => this.dispatchEvent(new CustomEvent("submit-job"))}>
          Generate bin
        </button>
        ${!allHeightsSet ? html`
          <p class="hint">Drop at least one photo and fill in tool height to enable.</p>
        ` : nothing}
      </div>

    `;
  }

  _renderThumbs() {
    if (this.files.length === 0) return nothing;
    return html`
      <div class="thumbs">
        ${this.files.map((f, i) => html`
          <div class="thumb">
            <img src=${f.dataUrl} alt=${f.file.name}>
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

  _renderTaperField() {
    const info = FIELD_INFO.tool_taper;
    const value = this.formValues.tool_taper;
    // Side-profile silhouettes on a ground line. Trapezoid points use
    // viewBox 0 0 64 56 — bottom edge at y=44, top at y=8, sides flared
    // so the difference between top/bottom widths is visually obvious.
    const ground = svg`<line x1="2" y1="48" x2="62" y2="48"
                             stroke="currentColor" stroke-width="1.5"
                             opacity="0.35"/>`;
    const shapeAttrs = `fill="currentColor" fill-opacity="0.18"
                        stroke="currentColor" stroke-width="2"
                        stroke-linejoin="round"`;
    const options = [
      {
        value: "top",
        label: "Widest at top",
        // Bottom 22→42 (20 wide), top 4→60 (56 wide) — flares outward.
        shape: svg`<polygon points="22,44 42,44 60,8 4,8"
                            fill="currentColor" fill-opacity="0.18"
                            stroke="currentColor" stroke-width="2"
                            stroke-linejoin="round"/>`,
      },
      {
        value: "uniform",
        label: "Uniform",
        // Plain rectangle, vertical sides.
        shape: svg`<rect x="14" y="8" width="36" height="36"
                         fill="currentColor" fill-opacity="0.18"
                         stroke="currentColor" stroke-width="2"
                         stroke-linejoin="round"/>`,
      },
      {
        value: "bottom",
        label: "Widest at bottom",
        // Bottom 4→60 (56 wide), top 22→42 (20 wide) — tapers inward.
        shape: svg`<polygon points="4,44 60,44 42,8 22,8"
                            fill="currentColor" fill-opacity="0.18"
                            stroke="currentColor" stroke-width="2"
                            stroke-linejoin="round"/>`,
      },
    ];
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
            <label class="taper-option ${value === o.value ? "selected" : ""}">
              <input type="radio" name="tool_taper" .value=${o.value}
                     .checked=${value === o.value}
                     @change=${() => this._emit("tool_taper", o.value)}>
              <svg viewBox="0 0 64 56" width="80" height="64" aria-hidden="true">
                ${ground}
                ${o.shape}
              </svg>
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
    const next = [...this.files];
    for (const file of Array.from(fileList)) {
      const dataUrl = await this._readDataUrl(file);
      next.push({ file, toolHeight: null, dataUrl });
    }
    this.dispatchEvent(new CustomEvent("files-changed", { detail: next }));
  }

  _readDataUrl(file) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      // HEIC won't render as a thumbnail in most browsers; show a placeholder.
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
        <h2>Working...</h2>
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
      <div class="card">
        <h3>Log</h3>
        <div class="log">${this.events.map(e => `[${e.step}] ${e.message}\n`).join("")}</div>
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

class PicPreview extends LitElement {
  static properties = {
    jobId: { type: String },
    layoutInfo: { type: Object },
    artifactKey: { type: Number },
    showRedo: { state: true },
    redoParams: { state: true },
  };
  createRenderRoot() { return this; }

  constructor() {
    super();
    this.showRedo = false;
    this.redoParams = {};
    this.artifactKey = 0;
  }

  render() {
    const k = this.artifactKey;
    const url = this.layoutInfo?.artifacts?.layout_preview;
    const pdfUrl = this.layoutInfo?.artifacts?.fit_test_pdf;
    const svgUrl = this.layoutInfo?.artifacts?.fit_test_svg;
    return html`
      <div class="card">
        <h2>Layout preview</h2>
        ${this.layoutInfo ? html`
          <p class="hint">Bin: ${this.layoutInfo.grid_units_x} × ${this.layoutInfo.grid_units_y} gridfinity units</p>
        ` : nothing}
        ${url ? html`<img class="preview-img" src=${`${url}?v=${k}`}>` : nothing}
      </div>

      ${pdfUrl || svgUrl ? html`
        <div class="card fit-test-card">
          <div class="card-header">
            <h2>Test the fit before printing</h2>
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
        <h2>Looks good?</h2>
        <div class="actions">
          <button class="primary" @click=${() => this.dispatchEvent(new CustomEvent("proceed"))}>
            Proceed → generate bin config
          </button>
          <button class="secondary" @click=${() => this.showRedo = !this.showRedo}>
            ${this.showRedo ? "Hide re-do options" : "Re-do with adjustments"}
          </button>
        </div>
      </div>
      ${this.showRedo ? this._renderRedoPanel() : nothing}
    `;
  }

  _renderRedoPanel() {
    const traceFields = TRACE_REQUIRED_FIELDS;
    const willRetrace = Object.keys(this.redoParams).some(k => traceFields.has(k));
    return html`
      <div class="card">
        <h3>Re-do parameters</h3>
        <p class="hint">Only the fields you change here will be applied.
          ${willRetrace
            ? html`<strong>Re-tracing required</strong> (slower) — your changes affect the trace step.`
            : html`Layout-only re-run (fast) — using cached traces.`}
        </p>
        <div class="field-row">
          ${this._redoNum("tolerance", "Tolerance (mm)", 0.1)}
          ${this._redoNum("axial_tolerance", "Axial tolerance (mm)", 0.1)}
          ${this._redoNum("gap", "Gap (mm)", 0.1)}
          ${this._redoNum("bin_margin", "Bin margin (mm)", 0.1)}
          ${this._redoNum("min_units", "Min units", 1)}
          ${this._redoNum("max_units", "Max units", 1)}
          ${this._redoNum("height_units", "Height units", 1)}
          ${this._redoNum("phone_height", "Phone height (mm)", 1)}
        </div>
        <div class="actions">
          <button class="primary"
                  @click=${() => this.dispatchEvent(new CustomEvent("redo", {
                    detail: { params: this.redoParams, layoutOnly: !willRetrace },
                  }))}>
            Re-run pipeline
          </button>
        </div>
      </div>
    `;
  }

  _redoNum(name, label, step) {
    return html`
      <div class="field">
        <label>${label}</label>
        <input type="number" step=${step}
               @input=${(e) => {
                 const v = e.target.value;
                 const next = { ...this.redoParams };
                 if (v === "") delete next[name];
                 else next[name] = Number(v);
                 this.redoParams = next;
               }}>
      </div>
    `;
  }
}

customElements.define("pic-preview", PicPreview);

// ---------------------------------------------------------------------------
// pic-downloads — final download links
// ---------------------------------------------------------------------------

class PicDownloads extends LitElement {
  static properties = {
    artifacts: { type: Object },
    artifactKey: { type: Number },
  };
  createRenderRoot() { return this; }

  constructor() {
    super();
    this.artifactKey = 0;
  }

  render() {
    const k = this.artifactKey;
    const a = this.artifacts;
    const withKey = (u) => `${u}?v=${k}`;
    return html`
      <div class="card">
        <h2>Downloads</h2>
        <div class="downloads">
          ${a.bin_config ? html`
            <a href=${withKey(a.bin_config)} download>
              Bin config (JSON) — for the Fusion 360 add-in
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
        <p class="hint" style="margin-top:1rem">
          Next: open Fusion 360 → Solid → Create → Gridfinity Pic-to-Bin →
          load the bin_config.json.
          <button class="info-link" type="button" style="margin-left:0.5rem"
                  @click=${() => showInfo(this, "fit_test")}>
            <span class="info-btn">i</span> About fit testing
          </button>
        </p>
        <div class="actions" style="margin-top:1rem">
          <button class="secondary" @click=${() => this.dispatchEvent(new CustomEvent("start-over"))}>
            Start a new bin
          </button>
        </div>
      </div>
    `;
  }
}

customElements.define("pic-downloads", PicDownloads);
