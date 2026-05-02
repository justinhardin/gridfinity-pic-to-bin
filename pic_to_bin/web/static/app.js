import { LitElement, html, css, nothing } from "lit";

// Pipeline parameter metadata. Mirrors pipeline.run_pipeline kwargs.
// "Hard required" = fields the user must touch (no default).
// "Soft default" = fields with a default that we highlight until the user
// touches them, so they consciously confirm or change the value.
const SOFT_DEFAULT_FIELDS = new Set(["phone_height"]);

const FORM_DEFAULTS = {
  paper_size: "legal",
  tolerance: 1.0,
  phone_height: 482.0,
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
  mask_erode: 0.3,
};

// Which fields, when changed on a redo, force a re-trace (expensive).
// Everything else is layout-only (cheap; cached DXFs reused).
const TRACE_REQUIRED_FIELDS = new Set([
  "phone_height",
  "tolerance",
  "slots",
  "straighten_threshold",
  "max_refine_iterations",
  "max_concavity_depth",
  "mask_erode",
]);

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
    this._eventSource = null;
  }

  render() {
    if (this.screen === "form")      return this._renderForm();
    if (this.screen === "progress")  return this._renderProgress();
    if (this.screen === "preview")   return this._renderPreview();
    if (this.screen === "downloads") return this._renderDownloads();
    if (this.screen === "error")     return this._renderError();
    return nothing;
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
        @proceed=${this._onProceed}
        @redo=${this._onRedo}
      ></pic-preview>
    `;
  }

  _renderDownloads() {
    return html`
      <pic-downloads
        .artifacts=${this.artifacts}
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
    const summary = await (await fetch(`/jobs/${this.jobId}`)).json();
    this.layoutInfo = summary;
    this.screen = "preview";
  }

  async _onComplete() {
    const summary = await (await fetch(`/jobs/${this.jobId}`)).json();
    this.artifacts = summary.artifacts || {};
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
        <h2>Photos</h2>
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
        <h2>Parameters</h2>
        ${this._renderField("phone_height", "Phone height (mm)", "number", {
          hint: "Camera height above paper. Compensates parallax.",
        })}
        ${this._renderField("paper_size", "Paper size", "select", {
          options: ["a4", "letter", "legal"],
        })}
        <div class="field-row">
          ${this._renderField("tolerance", "Tolerance (mm)", "number", {
            step: 0.1, hint: "Pocket clearance. + looser, − tighter."
          })}
          ${this._renderField("gap", "Gap (mm)", "number", { step: 0.1 })}
          ${this._renderField("bin_margin", "Bin margin (mm)", "number", { step: 0.1 })}
        </div>
        <div class="field-row">
          ${this._renderField("min_units", "Min grid units", "number", { step: 1 })}
          ${this._renderField("max_units", "Max grid units", "number", { step: 1 })}
          ${this._renderField("height_units", "Height units (auto if blank)", "number", { step: 1 })}
        </div>
        <div class="field-row">
          ${this._renderField("stacking", "Stacking lip", "checkbox")}
          ${this._renderField("slots", "Finger slots", "checkbox")}
        </div>

        <details>
          <summary>Advanced (tracing tuning)</summary>
          <div class="field-row">
            ${this._renderField("straighten_threshold", "Straighten threshold (°)", "number", { step: 1 })}
            ${this._renderField("max_refine_iterations", "Max refine iterations", "number", { step: 1 })}
            ${this._renderField("max_concavity_depth", "Max concavity depth (mm)", "number", { step: 0.1 })}
            ${this._renderField("mask_erode", "Mask erode (mm)", "number", { step: 0.05 })}
          </div>
        </details>
      </div>

      <div class="card">
        <button class="primary" ?disabled=${!allHeightsSet}
                @click=${() => this.dispatchEvent(new CustomEvent("submit-job"))}>
          Generate bin
        </button>
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
            <label>
              Tool height (mm) *
              <input type="number" step="0.1" .value=${f.toolHeight ?? ""}
                     @input=${(e) => this._setHeight(i, e.target.value)}>
            </label>
            <button class="remove" @click=${() => this._remove(i)}>Remove</button>
          </div>
        `)}
      </div>
    `;
  }

  _renderField(name, label, kind, opts = {}) {
    const value = this.formValues[name];
    const isDefault = SOFT_DEFAULT_FIELDS.has(name) && !this.touched.has(name);
    const cls = ["field", isDefault ? "default-applied" : ""].join(" ");

    if (kind === "checkbox") {
      return html`
        <div class=${cls}>
          <label>
            <input type="checkbox" .checked=${!!value}
                   @change=${(e) => this._emit(name, e.target.checked)}>
            ${label}
          </label>
        </div>
      `;
    }
    if (kind === "select") {
      return html`
        <div class=${cls}>
          <label>${label}</label>
          <select @change=${(e) => this._emit(name, e.target.value)}>
            ${opts.options.map(o => html`
              <option value=${o} ?selected=${o === value}>${o}</option>
            `)}
          </select>
        </div>
      `;
    }
    return html`
      <div class=${cls}>
        <label>${label}</label>
        <input type=${kind} .value=${value === "" ? "" : String(value)}
               step=${opts.step ?? "any"}
               @input=${(e) => this._emit(name, e.target.value === "" ? "" : Number(e.target.value))}>
        ${isDefault ? html`<span class="hint">Default applied — confirm or change.</span>` : nothing}
        ${opts.hint ? html`<span class="hint">${opts.hint}</span>` : nothing}
      </div>
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
    showRedo: { state: true },
    redoParams: { state: true },
  };
  createRenderRoot() { return this; }

  constructor() {
    super();
    this.showRedo = false;
    this.redoParams = {};
  }

  render() {
    const url = this.layoutInfo?.artifacts?.layout_preview;
    return html`
      <div class="card">
        <h2>Layout preview</h2>
        ${this.layoutInfo ? html`
          <p class="hint">Bin: ${this.layoutInfo.grid_units_x} × ${this.layoutInfo.grid_units_y} gridfinity units</p>
        ` : nothing}
        ${url ? html`<img class="preview-img" src=${url + "?" + Date.now()}>` : nothing}
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
  };
  createRenderRoot() { return this; }

  render() {
    return html`
      <div class="card">
        <h2>Downloads</h2>
        <div class="downloads">
          ${this.artifacts.layout_preview ? html`
            <a href=${this.artifacts.layout_preview} download>Layout preview (PNG)</a>` : nothing}
          ${this.artifacts.combined_dxf ? html`
            <a href=${this.artifacts.combined_dxf} download>Combined layout (DXF)</a>` : nothing}
          ${this.artifacts.bin_config ? html`
            <a href=${this.artifacts.bin_config} download>Bin config (JSON for Fusion 360)</a>` : nothing}
        </div>
        <p class="hint" style="margin-top:1rem">
          Next: open Fusion 360 → Solid → Create → Gridfinity Pic-to-Bin →
          load the bin_config.json.
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
