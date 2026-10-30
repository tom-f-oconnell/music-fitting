"use strict";
// musicfit web UI: jobs, progress, visualisation, live patch tweaking.

const QUALITY = {
  quick: { roll_steps: 150, note_steps: 100 },
  standard: { roll_steps: 400, note_steps: 300 },
  thorough: { roll_steps: 1000, note_steps: 600 },
};
const VOICE_COLORS = ["#ffb547", "#c792ea", "#67d391", "#ff7a90"];
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const $ = (sel) => document.querySelector(sel);

const S = {
  specs: null,
  jobId: null,
  job: null,
  fitted: null,    // fitted patches (from the job result)
  patches: null,   // current (tweaked) patches
  notes: null,     // [{pitch, onset, duration, velocity, voice, muted}]
  mutedVoices: new Set(),
  voice: 0,
  buffers: {},     // target / fit / tweaked AudioBuffers
  specs2d: {},     // cached spectrograms
  tweakedBlob: null,
  renderSeq: 0,
  renderAbort: null,
  play: null,      // {src, which, start, offset}
  pollTimer: null,
};

let actx = null;
function audioCtx() {
  if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
  return actx;
}

// ---------------------------------------------------------------- helpers

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* not json */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r;
}
const noteName = (p) => NOTE_NAMES[((Math.round(p) % 12) + 12) % 12] + (Math.floor(Math.round(p) / 12) - 1);
function fmt(v, unit) {
  if (unit === "Hz") return v >= 1000 ? (v / 1000).toFixed(2) + "k" : v.toFixed(0);
  if (unit === "s") return v < 1 ? (v * 1000).toFixed(v < 0.01 ? 1 : 0) + "ms" : v.toFixed(2) + "s";
  return v.toFixed(Math.abs(v) < 1 ? 3 : 2);
}
function parseVal(text, unit) {
  const t = text.trim().toLowerCase();
  let v = parseFloat(t);
  if (isNaN(v)) return NaN;
  if (t.endsWith("ms")) v /= 1000;
  else if (t.endsWith("k")) v *= 1000;
  return v;
}
function toUnit(spec, v) {
  v = Math.min(Math.max(v, spec.low), spec.high);
  return spec.log ? Math.log(v / spec.low) / Math.log(spec.high / spec.low) : (v - spec.low) / (spec.high - spec.low);
}
function fromUnit(spec, u) {
  return spec.log ? spec.low * Math.pow(spec.high / spec.low, u) : spec.low + (spec.high - spec.low) * u;
}
function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
function store(key, val) { try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) { /* ignore */ } }
function load(key) { try { return JSON.parse(localStorage.getItem(key)); } catch (e) { return null; } }
function sizeCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return [w, h, dpr];
}
const baseName = () => (S.job ? S.job.name.replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "_") : "musicfit");

// ---------------------------------------------------------------- jobs list

async function refreshJobs() {
  let jobs;
  try { jobs = await (await api("/api/jobs")).json(); } catch (e) { return; }
  const ul = $("#jobs");
  ul.innerHTML = "";
  if (!jobs.length) { ul.innerHTML = '<li class="hint">No jobs yet.</li>'; return; }
  for (const j of jobs) {
    const li = document.createElement("li");
    li.className = "job" + (j.id === S.jobId ? " active" : "");
    const pct = j.status === "running" ? ` ${Math.round(j.progress * 100)}%` : "";
    li.innerHTML = `<span class="jname"></span><span class="badge ${j.status}">${j.status}${pct}</span>`;
    li.querySelector(".jname").textContent = j.name;
    li.title = new Date(j.created * 1000).toLocaleString();
    li.onclick = () => selectJob(j.id);
    ul.appendChild(li);
  }
}

async function selectJob(id) {
  if (S.jobId !== id) stop();
  S.jobId = id;
  location.hash = "job=" + id;
  S.job = null;
  S.buffers = {};
  S.specs2d = {};
  $("#empty").hidden = true;
  $("#job-view").hidden = false;
  $("#result-view").hidden = true;
  refreshJobs();
  await poll();
}

async function poll() {
  clearTimeout(S.pollTimer);
  const id = S.jobId;
  if (!id) return;
  let job;
  try { job = await (await api(`/api/jobs/${id}`)).json(); } catch (e) {
    $("#job-error").hidden = false; $("#job-error").textContent = e.message; return;
  }
  if (id !== S.jobId) return;
  const wasDone = S.job && S.job.status === "done";
  S.job = job;
  showJobHead(job);
  if (job.status === "done" && !wasDone) await showResult(job);
  if (job.status === "queued" || job.status === "running") S.pollTimer = setTimeout(poll, 800);
  else refreshJobs();
}

function showJobHead(job) {
  $("#job-name").textContent = job.name;
  const st = $("#job-status");
  st.className = "badge " + job.status;
  st.textContent = job.status;
  const active = job.status === "queued" || job.status === "running";
  $("#cancel-btn").hidden = !active;
  $("#delete-btn").hidden = active;
  $("#progress-wrap").hidden = !active;
  $("#progress-bar").style.width = (job.progress * 100).toFixed(1) + "%";
  const stageName = job.stage === "roll" ? "Stage 1/2: discovering notes (piano roll)" : job.stage === "notes" ? "Stage 2/2: refining notes and patch" : "Waiting for the worker";
  const c = job.config;
  $("#progress-text").textContent = job.status === "queued" ? "Queued behind another fit." :
    `${stageName} — step ${job.step}/${job.total} · ${job.elapsed_s.toFixed(0)} s elapsed` +
    (job.progress > 0.03 ? ` · ~${Math.max(0, job.elapsed_s * (1 / job.progress - 1)).toFixed(0)} s left` : "");
  const err = $("#job-error");
  err.hidden = !job.error;
  err.textContent = job.error || "";
  drawLossChart(job.loss_curve || [], c);
}

function drawLossChart(curve, cfg) {
  const c = $("#loss-chart");
  c.hidden = curve.length < 2;
  if (c.hidden) return;
  const [w, h, dpr] = sizeCanvas(c);
  const g = c.getContext("2d");
  g.clearRect(0, 0, w, h);
  const ys = curve.map((p) => Math.log(Math.max(p[1], 1e-6)));
  const total = (cfg.roll_steps || 400) + (cfg.note_steps || 300) + 2;
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad = 6 * dpr;
  const X = (s) => pad + (w - 2 * pad) * s / total;
  const Y = (v) => h - pad - (h - 2 * pad) * (v - ymin) / Math.max(ymax - ymin, 1e-6);
  g.strokeStyle = "#2e333d"; g.lineWidth = dpr;
  g.beginPath(); g.moveTo(X(cfg.roll_steps + 1), 0); g.lineTo(X(cfg.roll_steps + 1), h); g.stroke();
  g.fillStyle = "#8d94a1"; g.font = `${11 * dpr}px system-ui`;
  g.fillText("loss (log) · stage 1 | stage 2", pad, 12 * dpr);
  g.fillText(curve[curve.length - 1][1].toFixed(3), w - 60 * dpr, 12 * dpr);
  g.strokeStyle = "#ffb547"; g.lineWidth = 1.5 * dpr;
  g.beginPath();
  curve.forEach((p, i) => { const x = X(p[0]), y = Y(ys[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.stroke();
}

// ---------------------------------------------------------------- result view

async function showResult(job) {
  const r = job.result;
  S.fitted = r.patches.map((p) => JSON.parse(JSON.stringify(p)));
  const saved = load("musicfit:" + job.id);
  S.patches = saved && saved.patches && saved.patches.length === r.patches.length ? saved.patches : JSON.parse(JSON.stringify(r.patches));
  S.notes = (saved && saved.notes) || r.notes.map((n) => ({ ...n, muted: false }));
  S.mutedVoices = new Set((saved && saved.mutedVoices) || []);
  S.voice = 0;
  $("#result-view").hidden = false;
  $("#ref-legend").hidden = !r.reference;
  $("#true-legend").hidden = !r.reference;
  buildVoiceTabs();
  buildSliders();
  showMetrics();
  const ctx = audioCtx();
  const [t, f] = await Promise.all(["target", "fit"].map(async (w) => {
    const buf = await (await api(`/api/jobs/${job.id}/audio/${w}.wav`)).arrayBuffer();
    return ctx.decodeAudioData(buf);
  }));
  if (S.jobId !== job.id) return;
  S.buffers.target = t;
  S.buffers.fit = f;
  S.specs2d.target = spectrogram(t.getChannelData(0), t.sampleRate);
  $("#loss-fit").textContent = r.final_loss.toFixed(4);
  drawAll();
  scheduleRender(0);
}

function showMetrics() {
  const r = S.job.result;
  const el = $("#metrics");
  const nAct = S.notes.filter((n) => !n.muted).length;
  let html = `${S.notes.length} fitted notes (${nAct} active), ${r.patches.length} voice${r.patches.length > 1 ? "s" : ""}, clip ${(r.n_samples / r.sr).toFixed(2)} s.`;
  if (r.reference) {
    const m = r.reference.metrics_final;
    html += ` Against the ground truth: precision ${m.precision.toFixed(2)}, recall ${m.recall.toFixed(2)}, F1 ${m.f1.toFixed(2)} (${m.tp}/${m.n_ref} notes found).`;
  }
  el.innerHTML = html;
}

function persist() {
  if (S.job) store("musicfit:" + S.job.id, { patches: S.patches, notes: S.notes, mutedVoices: [...S.mutedVoices] });
}

// ---------------------------------------------------------------- sliders

function buildVoiceTabs() {
  const tabs = $("#voice-tabs");
  tabs.innerHTML = "";
  if (S.patches.length < 2) return;
  S.patches.forEach((_, v) => {
    const b = document.createElement("button");
    b.textContent = "Voice " + (v + 1);
    b.style.color = v === S.voice ? VOICE_COLORS[v % 4] : "";
    b.className = v === S.voice ? "active" : "";
    b.onclick = () => { S.voice = v; buildVoiceTabs(); buildSliders(); };
    tabs.appendChild(b);
  });
}

function sliderRow(label, spec, get, set, fittedVal, trueVal) {
  const row = document.createElement("div");
  row.className = "slider";
  row.innerHTML = `<span class="pname"></span><div class="track"><input type="range" min="0" max="1000" step="1"></div><input type="text">`;
  row.querySelector(".pname").textContent = label;
  row.querySelector(".pname").title = label;
  const range = row.querySelector("input[type=range]");
  const text = row.querySelector("input[type=text]");
  const track = row.querySelector(".track");
  const mark = (v, cls) => {
    if (v === undefined || v === null) return;
    const m = document.createElement("div");
    m.className = "mark " + cls;
    // range thumbs are ~16px wide; keep marks within the thumb's travel
    m.style.left = `calc(8px + (100% - 16px) * ${toUnit(spec, v)} - 1px)`;
    track.appendChild(m);
  };
  mark(fittedVal, "fit");
  mark(trueVal, "true");
  const sync = () => { const v = get(); range.value = Math.round(toUnit(spec, v) * 1000); text.value = fmt(v, spec.unit); };
  range.oninput = () => { set(fromUnit(spec, range.value / 1000)); text.value = fmt(get(), spec.unit); changed(); };
  text.onchange = () => {
    const v = parseVal(text.value, spec.unit);
    if (!isNaN(v)) { set(Math.min(Math.max(v, spec.low), spec.high)); changed(); }
    sync();
  };
  range.ondblclick = () => { set(fittedVal); sync(); changed(); };
  range.title = "double-click to reset to the fitted value";
  sync();
  row.sync = sync;
  return row;
}

function buildSliders() {
  const grid = $("#sliders");
  grid.innerHTML = "";
  const v = S.voice;
  const p = S.patches[v], f = S.fitted[v];
  const ref = S.job.result.reference && v === 0 ? S.job.result.reference.patch : null;
  $("#voice-mute").checked = S.mutedVoices.has(v);

  const osc = document.createElement("div");
  osc.className = "group";
  osc.innerHTML = "<h3>Oscillator mix</h3>";
  const wspec = { low: 0, high: 1, log: false, unit: "" };
  for (const w of S.specs.waveforms) {
    osc.appendChild(sliderRow(w, wspec, () => p.waveform[w] ?? 0, (x) => { p.waveform[w] = x; }, f.waveform[w], ref ? ref.waveform[w] : null));
  }
  const note = document.createElement("p");
  note.className = "hint";
  note.textContent = "Weights are normalised to sum to 1.";
  osc.appendChild(note);
  grid.appendChild(osc);

  for (const g of S.specs.groups) {
    const box = document.createElement("div");
    box.className = "group";
    box.innerHTML = "<h3></h3>";
    box.querySelector("h3").textContent = g.title;
    for (const name of g.params) {
      const spec = S.specs.params[name];
      const label = name.replace(/^(amp|filt)_/, "").replace(/_/g, " ");
      box.appendChild(sliderRow(label, spec, () => p[name], (x) => { p[name] = x; }, f[name], ref ? ref[name] : null));
    }
    grid.appendChild(box);
  }
}

function changed() {
  persist();
  scheduleRender(120);
}

// ---------------------------------------------------------------- rendering

let renderTimer = null;
function scheduleRender(delay) {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(renderTweaked, delay);
}

async function renderTweaked() {
  if (!S.job || S.job.status !== "done") return;
  const seq = ++S.renderSeq;
  if (S.renderAbort) S.renderAbort.abort();
  S.renderAbort = new AbortController();
  $("#render-state").textContent = "rendering…";
  const notes = S.notes.filter((n) => !n.muted).map(({ muted, ...n }) => n);
  try {
    const r = await api(`/api/jobs/${S.job.id}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patches: S.patches, notes, muted_voices: [...S.mutedVoices] }),
      signal: S.renderAbort.signal,
    });
    const loss = r.headers.get("X-Spectral-Loss");
    const blob = await r.blob();
    const buf = await audioCtx().decodeAudioData(await blob.arrayBuffer());
    if (seq !== S.renderSeq) return;
    S.tweakedBlob = blob;
    S.buffers.tweaked = buf;
    S.specs2d.tweaked = spectrogram(buf.getChannelData(0), buf.sampleRate);
    $("#loss-tweak").textContent = loss ? parseFloat(loss).toFixed(4) : "–";
    $("#render-state").textContent = "";
    drawSpec($("#spec-tweak"), S.specs2d.tweaked);
    if (S.play && S.play.which === "tweaked") restartAtCurrent();
  } catch (e) {
    if (e.name !== "AbortError" && seq === S.renderSeq) $("#render-state").textContent = "render failed: " + e.message;
  }
}

// ---------------------------------------------------------------- playback

function playheadTime() {
  if (!S.play) return null;
  const ctx = audioCtx();
  const dur = S.play.src.buffer.duration;
  let t = S.play.offset + (ctx.currentTime - S.play.start);
  if (S.play.src.loop) t = t % dur;
  return t <= dur ? t : null;
}

function play(which, offset = 0) {
  const buf = S.buffers[which];
  if (!buf) return;
  stop();
  const ctx = audioCtx();
  ctx.resume();
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.loop = $("#loop").checked;
  src.connect(ctx.destination);
  src.start(0, offset % buf.duration);
  const me = { src, which, start: ctx.currentTime, offset: offset % buf.duration };
  src.onended = () => { if (S.play === me) { S.play = null; markPlaying(); } };
  S.play = me;
  markPlaying();
  requestAnimationFrame(tick);
}

function restartAtCurrent() {
  const t = playheadTime();
  if (S.play) play(S.play.which, t || 0);
}

function stop() {
  if (S.play) { const p = S.play; S.play = null; try { p.src.stop(); } catch (e) { /* already stopped */ } }
  markPlaying();
}

function markPlaying() {
  document.querySelectorAll("[data-play]").forEach((b) => {
    b.style.outline = S.play && S.play.which === b.dataset.play ? "2px solid #fff" : "";
  });
  if (!S.play) document.querySelectorAll(".playhead").forEach((p) => (p.style.display = "none"));
}

function tick() {
  if (!S.play) return;
  const t = playheadTime();
  const dur = S.job.result.n_samples / S.job.result.sr;
  document.querySelectorAll(".playhead").forEach((p) => {
    if (t === null) { p.style.display = "none"; return; }
    p.style.display = "block";
    p.style.left = (100 * t / dur) + "%";
  });
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- spectrogram

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

// Log-frequency spectrogram (rows from FMIN to Nyquist), in dB.
const N_FFT = 2048, HOP = 128, ROWS = 180, FMIN = 40;
function spectrogram(x, sr) {
  const frames = Math.max(1, Math.floor(x.length / HOP));
  const win = new Float64Array(N_FFT).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N_FFT));
  const out = new Float32Array(frames * ROWS);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
  const fmax = sr / 2;
  const rowBin = new Float64Array(ROWS).map((_, r) => FMIN * Math.pow(fmax / FMIN, r / (ROWS - 1)) / sr * N_FFT);
  for (let f = 0; f < frames; f++) {
    const c = f * HOP - N_FFT / 2;
    for (let i = 0; i < N_FFT; i++) { const k = c + i; re[i] = k >= 0 && k < x.length ? x[k] * win[i] : 0; im[i] = 0; }
    fft(re, im);
    for (let r = 0; r < ROWS; r++) {
      const b = Math.min(rowBin[r], N_FFT / 2 - 1), b0 = Math.floor(b), t = b - b0;
      const m0 = Math.hypot(re[b0], im[b0]), m1 = Math.hypot(re[b0 + 1], im[b0 + 1]);
      out[f * ROWS + r] = 20 * Math.log10(m0 * (1 - t) + m1 * t + 1e-9);
    }
  }
  return { frames, data: out, sr };
}

// approx. "magma"
const CMAP = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function cmap(t) {
  t = Math.min(Math.max(t, 0), 1) * (CMAP.length - 1);
  const i = Math.min(Math.floor(t), CMAP.length - 2), f = t - i;
  return CMAP[i].map((c, k) => c + (CMAP[i + 1][k] - c) * f);
}

function drawSpec(canvas, spec) {
  if (!spec) return;
  const ref = S.specs2d.target || spec;   // shared dB scale so the two are comparable
  let top = -Infinity;
  for (const v of ref.data) if (v > top) top = v;
  const floor = top - 80;
  const img = new ImageData(spec.frames, ROWS);
  for (let f = 0; f < spec.frames; f++) {
    for (let r = 0; r < ROWS; r++) {
      const [R, G, B] = cmap((spec.data[f * ROWS + r] - floor) / 80);
      const o = ((ROWS - 1 - r) * spec.frames + f) * 4;
      img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = spec.frames; off.height = ROWS;
  off.getContext("2d").putImageData(img, 0, 0);
  const [w, h] = sizeCanvas(canvas);
  const g = canvas.getContext("2d");
  g.imageSmoothingEnabled = true;
  g.drawImage(off, 0, 0, w, h);
}

// ---------------------------------------------------------------- piano roll

function rollLayout() {
  const r = S.job.result;
  const all = S.notes.concat(r.reference ? r.reference.notes : []);
  let lo = Math.min(...all.map((n) => n.pitch), 60), hi = Math.max(...all.map((n) => n.pitch), 60);
  lo = Math.floor(lo) - 2; hi = Math.ceil(hi) + 2;
  if (hi - lo < 14) { const m = (hi + lo) / 2; lo = Math.floor(m - 7); hi = Math.ceil(m + 7); }
  return { lo, hi, dur: r.n_samples / r.sr };
}

function drawRoll() {
  const c = $("#roll");
  const [w, h, dpr] = sizeCanvas(c);
  const g = c.getContext("2d");
  const { lo, hi, dur } = rollLayout();
  const rowH = h / (hi - lo + 1);
  const X = (t) => w * t / dur, Y = (p) => h - (p - lo + 1) * rowH;
  g.clearRect(0, 0, w, h);
  for (let p = lo; p <= hi; p++) {
    const black = [1, 3, 6, 8, 10].includes(((p % 12) + 12) % 12);
    g.fillStyle = black ? "#1d2027" : "#252a33";
    g.fillRect(0, Y(p), w, rowH);
    if (p % 12 === 0) {
      g.fillStyle = "#8d94a1"; g.font = `${10 * dpr}px system-ui`;
      g.fillText(noteName(p), 3 * dpr, Y(p) + rowH - 2 * dpr);
    }
  }
  g.strokeStyle = "#ffffff14"; g.lineWidth = dpr;
  for (let t = 0; t <= dur; t += 0.5) { g.beginPath(); g.moveTo(X(t), 0); g.lineTo(X(t), h); g.stroke(); }
  const ref = S.job.result.reference;
  if (ref) {
    g.strokeStyle = "#b8bfcc"; g.lineWidth = 1.5 * dpr;
    for (const n of ref.notes) g.strokeRect(X(n.onset), Y(n.pitch) + 1, Math.max(X(n.duration), 2), rowH - 2);
  }
  for (const n of S.notes) {
    const col = VOICE_COLORS[n.voice % 4];
    const muted = n.muted || S.mutedVoices.has(n.voice);
    const x = X(n.onset), y = Y(n.pitch) + rowH * 0.18, ww = Math.max(X(n.duration), 3), hh = rowH * 0.64;
    g.globalAlpha = muted ? 1 : 0.35 + 0.65 * n.velocity;
    if (muted) { g.strokeStyle = col; g.setLineDash([3 * dpr, 3 * dpr]); g.strokeRect(x, y, ww, hh); g.setLineDash([]); }
    else { g.fillStyle = col; g.fillRect(x, y, ww, hh); }
    g.globalAlpha = 1;
  }
}

function noteAt(ev) {
  const c = $("#roll");
  const rect = c.getBoundingClientRect();
  const { lo, hi, dur } = rollLayout();
  const t = (ev.clientX - rect.left) / rect.width * dur;
  const p = hi - Math.floor((ev.clientY - rect.top) / rect.height * (hi - lo + 1));
  let best = null;
  for (const n of S.notes) {
    if (Math.round(n.pitch) === p && t >= n.onset - 0.02 && t <= n.onset + Math.max(n.duration, 0.05)) best = n;
  }
  return best;
}

function drawAll() {
  if (!S.job || !S.job.result) return;
  drawSpec($("#spec-target"), S.specs2d.target);
  drawSpec($("#spec-tweak"), S.specs2d.tweaked);
  drawRoll();
}

// ---------------------------------------------------------------- wiring

function wire() {
  const drop = $("#drop"), file = $("#file");
  const showFile = () => { $("#drop-text").textContent = file.files[0] ? file.files[0].name : "Drop an audio file here or click to choose"; };
  file.onchange = showFile;
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); file.files = e.dataTransfer.files; showFile(); };

  $("#upload-form").onsubmit = async (e) => {
    e.preventDefault();
    if (!file.files[0]) { alert("Choose an audio file first."); return; }
    const fd = new FormData(e.target);
    const q = QUALITY[fd.get("quality")];
    fd.delete("quality");
    fd.set("roll_steps", q.roll_steps);
    fd.set("note_steps", q.note_steps);
    const btn = $("#upload-btn");
    btn.disabled = true; btn.textContent = "Uploading…";
    try {
      const job = await (await api("/api/jobs", { method: "POST", body: fd })).json();
      await selectJob(job.id);
    } catch (err) { alert("Upload failed: " + err.message); }
    btn.disabled = false; btn.textContent = "Upload and fit";
  };

  $("#demo-btn").onclick = async () => {
    const q = QUALITY[$("#demo-quality").value];
    try {
      const job = await (await api("/api/jobs/demo", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed: parseInt($("#demo-seed").value, 10) || 1, ...q }),
      })).json();
      await selectJob(job.id);
    } catch (err) { alert(err.message); }
  };

  $("#cancel-btn").onclick = async () => { await api(`/api/jobs/${S.jobId}/cancel`, { method: "POST" }); poll(); };
  $("#delete-btn").onclick = async () => {
    if (!confirm("Delete this job and its results?")) return;
    await api(`/api/jobs/${S.jobId}`, { method: "DELETE" });
    stop();
    S.jobId = null; S.job = null; location.hash = "";
    $("#job-view").hidden = true; $("#empty").hidden = false;
    refreshJobs();
  };

  document.querySelectorAll("[data-play]").forEach((b) => (b.onclick = () => {
    const t = S.play ? playheadTime() : 0;   // A/B: switching keeps the position
    play(b.dataset.play, t || 0);
  }));
  $("#stop-btn").onclick = stop;
  $("#loop").onchange = () => { if (S.play) S.play.src.loop = $("#loop").checked; };
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(document.activeElement.tagName)) return;
    e.preventDefault();
    if (S.play) stop(); else play("tweaked");
  });

  const seek = (ev, canvas) => {
    if (!S.job || !S.job.result) return;
    const rect = canvas.getBoundingClientRect();
    const t = (ev.clientX - rect.left) / rect.width * S.job.result.n_samples / S.job.result.sr;
    play(S.play ? S.play.which : "tweaked", t);
  };
  $("#spec-target").onclick = (e) => seek(e, e.target);
  $("#spec-tweak").onclick = (e) => seek(e, e.target);

  const roll = $("#roll"), tip = $("#tooltip");
  roll.onclick = (e) => {
    const n = noteAt(e);
    if (!n) return;
    n.muted = !n.muted;
    drawRoll(); showMetrics(); changed();
  };
  roll.onmousemove = (e) => {
    const n = noteAt(e);
    if (!n) { tip.hidden = true; return; }
    const rect = roll.getBoundingClientRect();
    tip.hidden = false;
    tip.textContent = `${noteName(n.pitch)} (${n.pitch.toFixed(2)}) · ${n.onset.toFixed(3)} s · ${(n.duration * 1000).toFixed(0)} ms · vel ${n.velocity.toFixed(2)} · voice ${n.voice + 1}${n.muted ? " · muted" : ""}`;
    tip.style.left = Math.min(e.clientX - rect.left + 12, rect.width - 260) + "px";
    tip.style.top = (e.clientY - rect.top + 12) + "px";
  };
  roll.onmouseleave = () => (tip.hidden = true);

  $("#voice-mute").onchange = (e) => {
    if (e.target.checked) S.mutedVoices.add(S.voice); else S.mutedVoices.delete(S.voice);
    drawRoll(); changed();
  };
  $("#reset-btn").onclick = () => {
    S.patches[S.voice] = JSON.parse(JSON.stringify(S.fitted[S.voice]));
    S.notes = S.job.result.notes.map((n) => ({ ...n, muted: false }));
    S.mutedVoices.clear();
    buildSliders(); drawRoll(); showMetrics(); changed();
  };

  $("#exp-patch").onclick = () => {
    const out = { format: "musicfit-patch", version: 1, sr: S.job.result.sr, voices: S.patches };
    download(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" }), baseName() + "_patch.json");
  };
  $("#exp-notes").onclick = () => {
    const notes = S.notes.filter((n) => !n.muted && !S.mutedVoices.has(n.voice)).map(({ muted, ...n }) => n);
    download(new Blob([JSON.stringify(notes, null, 2)], { type: "application/json" }), baseName() + "_notes.json");
  };
  $("#exp-midi").onclick = async () => {
    const notes = S.notes.filter((n) => !n.muted && !S.mutedVoices.has(n.voice)).map(({ muted, ...n }) => n);
    const r = await api("/api/midi", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ notes }) });
    download(await r.blob(), baseName() + ".mid");
  };
  $("#exp-wav").onclick = () => { if (S.tweakedBlob) download(S.tweakedBlob, baseName() + "_tweaked.wav"); };
  $("#imp-patch").onchange = async (e) => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    try {
      const d = JSON.parse(await f.text());
      const voices = Array.isArray(d.voices) ? d.voices : [d];
      voices.slice(0, S.patches.length).forEach((p, i) => {
        const merged = { ...S.patches[i], ...p, waveform: { ...S.patches[i].waveform, ...(p.waveform || {}) } };
        for (const [k, spec] of Object.entries(S.specs.params)) merged[k] = Math.min(Math.max(+merged[k], spec.low), spec.high);
        S.patches[i] = merged;
      });
      buildSliders(); changed();
    } catch (err) { alert("Could not read patch: " + err.message); }
  };

  let rt = null;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { drawAll(); if (S.job) drawLossChart(S.job.loss_curve || [], S.job.config); }, 100); });
}

async function main() {
  S.specs = await (await api("/api/specs")).json();
  wire();
  await refreshJobs();
  setInterval(refreshJobs, 4000);
  const m = location.hash.match(/job=(\w+)/);
  if (m) selectJob(m[1]);
}
main();
