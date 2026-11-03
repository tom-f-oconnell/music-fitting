"use strict";
// musicfit web UI: jobs, progress, visualisation, live patch tweaking.

const QUALITY = {
  quick: { roll_steps: 150, note_steps: 100 },
  standard: { roll_steps: 400, note_steps: 300 },
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
  muteNoise: false,
  voice: 0,
  buffers: {},     // target / fit / tweaked AudioBuffers
  specs2d: {},     // cached spectrograms
  tweakedBlob: null,
  renderSeq: 0,
  renderAbort: null,
  play: null,      // {src, which, start, offset}
  pollTimer: null,
  library: null,   // datasets, presets and downloads from /api/library
  libraryTimer: null,
  keyboard: null,
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

function formatWhen(seconds, withSeconds = false) {
  const d = new Date(seconds * 1000);
  const parts = { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" };
  if (withSeconds) parts.second = "2-digit";
  if (d.getFullYear() !== new Date().getFullYear()) parts.year = "numeric";
  return d.toLocaleString([], parts);
}
function formatSpan(seconds) {
  const s = Math.round(seconds);
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}
// One line for the jobs list: when it ended, or if it hasn't, where it has got to.
function jobWhen(j) {
  if (j.status === "queued") return `Queued ${formatWhen(j.created)}`;
  if (j.status === "running") return `Started ${formatWhen(j.started ?? j.created)}`;
  const end = j.finished ?? j.created + j.elapsed_s;
  const verb = { done: "Finished", failed: "Failed", cancelled: "Cancelled" }[j.status] || j.status;
  if (!j.elapsed_s) return `${verb} ${formatWhen(end)}`;
  return `${verb} ${formatWhen(end)}, ${j.status === "done" ? "took" : "after"} ${formatSpan(j.elapsed_s)}`;
}
// The whole story, for the job's header and the list's tooltip.
function jobTimes(j) {
  const parts = [j.started ? `Started ${formatWhen(j.started, true)}` : `Submitted ${formatWhen(j.created, true)}`];
  if (j.finished) parts.push(`${{ done: "Finished", failed: "Failed", cancelled: "Cancelled" }[j.status] || "Ended"} ${formatWhen(j.finished, true)}`);
  if (j.finished && j.elapsed_s) parts.push(`took ${formatSpan(j.elapsed_s)}`);
  return parts.join(" · ");
}

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
    li.innerHTML = `<div class="jmain"><span class="jname"></span><span class="badge ${j.status}">${j.status}${pct}</span>` +
                   `<button class="jdel" type="button">×</button></div><div class="jwhen"></div>`;
    li.querySelector(".jname").textContent = j.name;
    li.querySelector(".jwhen").textContent = jobWhen(j);
    li.title = jobTimes(j);
    li.onclick = () => selectJob(j.id);
    const del = li.querySelector(".jdel");
    del.title = jobActive(j) ? "Stop and delete this job" : "Delete this job";
    del.setAttribute("aria-label", del.title);
    del.onclick = (ev) => { ev.stopPropagation(); deleteJob(j); };
    ul.appendChild(li);
  }
}

const jobActive = (j) => j.status === "queued" || j.status === "running";

async function deleteJob(j) {
  const what = jobActive(j) ? `Stop and delete “${j.name}”?` : `Delete “${j.name}” and its results?`;
  if (!confirm(what)) return;
  try { await api(`/api/jobs/${j.id}`, { method: "DELETE" }); } catch (e) { alert(e.message); }
  if (S.jobId === j.id) {
    stop();
    clearTimeout(S.pollTimer);
    S.jobId = null; S.job = null; location.hash = "";
    $("#job-view").hidden = true; $("#empty").hidden = false;
  }
  refreshJobs();
}

async function selectJob(id) {
  if (S.jobId !== id) stop();
  S.keyboard.allOff();
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
  $("#job-times").textContent = jobTimes(job);
  const active = jobActive(job);
  $("#cancel-btn").hidden = !active;
  $("#delete-btn").hidden = active;
  $("#progress-wrap").hidden = !active;
  $("#progress-bar").style.width = (job.progress * 100).toFixed(1) + "%";
  const stageName = job.stage === "roll" ? "Stage 1/2: discovering notes (piano roll)" : job.stage === "notes" ? "Stage 2/2: refining notes and patch" :
    job.stage === "kits" ? `Comparing drum kits, ${job.detail}` : job.stage === "drums" ? "Fitting the drums" : "Waiting for the worker";
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
  const total = cfg.total_steps || (cfg.roll_steps || 400) + (cfg.note_steps || 300) + 2;
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
  S.drumHits = (saved && saved.drumHits) || (r.drum_hits || []).map(([kind, onset, velocity]) => ({ kind, onset, velocity, muted: false }));
  S.drumSettings = (saved && saved.drumSettings) || fittedDrumSettings();
  // drums-only fits have no voices: nothing to tweak or play
  document.querySelector(".card.synth").hidden = document.querySelector(".card.play").hidden = !r.patches.length;
  S.mutedVoices = new Set((saved && saved.mutedVoices) || []);
  S.muteNoise = !!(saved && saved.muteNoise);
  document.querySelector(".card.drums").hidden = !r.has_noise;
  $("#noise-mute").checked = S.muteNoise;
  S.voice = 0;
  $("#result-view").hidden = false;
  S.keyboard.jobChanged();
  $("#ref-legend").hidden = !r.reference;
  $("#true-legend").hidden = !r.reference;
  buildVoiceTabs();
  buildSliders();
  buildDrumSliders();
  showKitComparison();
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
  let html = `${S.notes.length} fitted notes (${nAct} active), ${r.patches.length} voice${r.patches.length !== 1 ? "s" : ""}, clip ${(r.n_samples / r.sr).toFixed(2)} s.`;
  if (S.drumHits.length) {
    const counts = {};
    for (const hit of S.drumHits) counts[hit.kind] = (counts[hit.kind] || 0) + 1;
    const list = Object.entries(counts).map(([kind, n]) => `${drumLabel(Number(kind))} ${n}`).join(", ");
    html += ` ${S.drumHits.length} drum hits (${S.drumHits.filter((x) => !x.muted).length} active): ${list}.`;
  }
  if (r.reference) {
    const m = r.reference.metrics_final;
    html += ` Against the ground truth: precision ${m.precision.toFixed(2)}, recall ${m.recall.toFixed(2)}, F1 ${m.f1.toFixed(2)} (${m.tp}/${m.n_ref} notes found).`;
  }
  el.innerHTML = html;
}

function persist() {
  if (S.job) store("musicfit:" + S.job.id, { patches: S.patches, notes: S.notes, drumHits: S.drumHits, drumSettings: S.drumSettings,
                                             mutedVoices: [...S.mutedVoices], muteNoise: S.muteNoise });
}

// ---------------------------------------------------------------- sliders

function buildVoiceTabs() {
  const tabs = $("#voice-tabs");
  tabs.innerHTML = "";
  if (!S.patches.length) return;
  if (S.patches.length < 2) return;
  S.patches.forEach((_, v) => {
    const b = document.createElement("button");
    b.textContent = "Voice " + (v + 1);
    b.style.color = v === S.voice ? VOICE_COLORS[v % 4] : "";
    b.className = v === S.voice ? "active" : "";
    b.onclick = () => { S.keyboard.allOff(); S.voice = v; buildVoiceTabs(); buildSliders(); S.keyboard.patchChanged(); };
    tabs.appendChild(b);
  });
}

function sliderRow(label, spec, help, get, set, fittedVal, trueVal) {
  const row = document.createElement("div");
  row.className = "slider";
  row.innerHTML = `<span class="pname"></span><div class="track"><input type="range" min="0" max="1000" step="1"></div><input type="text">`;
  row.querySelector(".pname").textContent = label;
  row.title = `${help ? `${label}: ${help}` : label}\nDouble-click the slider to reset it to the fitted value.`;
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
  text.onkeydown = (e) => { if (e.key === "Enter") text.blur(); };  // done typing: the keys can play again
  range.ondblclick = () => { set(fittedVal); sync(); changed(); };
  sync();
  row.sync = sync;
  return row;
}

function buildSliders() {
  const grid = $("#sliders");
  grid.innerHTML = "";
  if (!S.patches.length) return;
  const v = S.voice;
  const p = S.patches[v], f = S.fitted[v];
  const ref = S.job.result.reference && v === 0 ? S.job.result.reference.patch : null;
  $("#voice-mute").checked = S.mutedVoices.has(v);

  const osc = document.createElement("div");
  osc.className = "group";
  osc.innerHTML = "<h3>Oscillator mix</h3>";
  const wspec = { low: 0, high: 1, log: false, unit: "" };
  for (const w of S.specs.waveforms) {
    osc.appendChild(sliderRow(w, wspec, S.specs.waveform_help?.[w], () => p.waveform[w] ?? 0, (x) => { p.waveform[w] = x; }, f.waveform[w], ref ? ref.waveform[w] : null));
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
      box.appendChild(sliderRow(label, spec, spec.help, () => p[name], (x) => { p[name] = x; }, f[name], ref ? ref[name] : null));
    }
    grid.appendChild(box);
  }
}

// ---------------------------------------------------------------- drum sliders

// Which sliders a job's drums have: a kit's drums, or the generic voice's sounds (null for jobs from before either)
function drumVoiceKind() {
  const d = (S.job.result.drums || [])[0];
  if (!d || !("level" in d)) return null;
  return "tune" in d ? "kit" : "noise";
}

// The kits a "compare" fit tried, best first, each with its drums to listen to.
function showKitComparison() {
  const r = S.job.result, rows = $("#kit-compare-rows");
  $("#kit-compare").hidden = !(r.drum_kits || []).length;
  rows.innerHTML = "";
  if (!(r.drum_kits || []).length) return;
  $("#kit-compare-note").textContent = `Each was fitted to ${r.drum_kits_on === "drum stem" ? "the drums on their own (Demucs's drum stem)" : "the whole clip"}; the fit kept the lowest loss.`;
  r.drum_kits.forEach((k, i) => {
    const tr = document.createElement("tr");
    tr.className = i === 0 ? "used" : "";
    tr.innerHTML = "<td></td><td></td><td></td><td></td>";
    const [name, loss, hits, cell] = tr.children;
    name.textContent = k.kit.split(",").map(kitName).join(" + ");
    loss.textContent = k.loss.toFixed(3);
    hits.textContent = k.n_hits;
    const b = document.createElement("button");
    b.className = "small";
    b.textContent = "▶ Listen";
    b.title = "This kit's fitted drums on their own";
    b.dataset.play = k.audio;
    b.onclick = async () => {
      if (S.play && S.play.which === k.audio) { stop(); return; }
      if (!S.buffers[k.audio]) {
        const job = S.job.id, buf = await (await api(`/api/jobs/${job}/audio/${k.audio}.wav`)).arrayBuffer();
        if (S.job.id !== job) return;
        S.buffers[k.audio] = await audioCtx().decodeAudioData(buf);
      }
      play(k.audio, 0);
    };
    cell.appendChild(b);
    rows.appendChild(tr);
  });
}

function fittedDrumSettings() {
  const kind = drumVoiceKind();
  if (!kind) return null;
  const names = Object.keys(S.specs.drum_params[kind]);
  return S.job.result.drums.map((d) => {
    const s = Object.fromEntries(names.map((n) => [n, d[n]]));
    if (kind === "kit") s.kit = d.kit;
    return s;
  });
}

function buildDrumSliders() {
  const grid = $("#drum-sliders");
  grid.innerHTML = "";
  const kind = drumVoiceKind();
  const kinds = [...new Set(S.drumHits.map((h) => h.kind))].sort((a, b) => a - b);
  if (!kind || !S.drumSettings || !kinds.length) {
    grid.innerHTML = `<p class="hint">${kind ? "No drum hits were found." : "This job is from before drums could be adjusted; fit it again to get sliders."}</p>`;
    return;
  }
  const drums = S.job.result.drums, specs = S.specs.drum_params[kind];
  for (const k of kinds) {
    const box = document.createElement("div");
    box.className = "group";
    box.innerHTML = "<h3></h3>";
    const title = box.querySelector("h3");
    title.textContent = drumLabel(k);
    const s = S.drumSettings[k], f = drums[k];
    if (kind === "kit" && (f.kits || []).length > 1) {
      const row = document.createElement("label");
      row.className = "slider kit";
      row.title = S.specs.kit_help;
      row.innerHTML = '<span class="pname">kit</span><select></select>';
      const sel = row.querySelector("select");
      for (const name of f.kits) sel.add(new Option(kitName(name) + (name === f.kit ? " (fitted)" : ""), name));
      sel.value = s.kit;
      sel.onchange = () => { s.kit = sel.value; title.textContent = drumLabel(k); drawRoll(); changed(); };
      box.appendChild(row);
    }
    for (const [name, spec] of Object.entries(specs)) {
      box.appendChild(sliderRow(name.replace(/_/g, " "), spec, spec.help, () => s[name], (x) => { s[name] = x; }, f[name], null));
    }
    grid.appendChild(box);
  }
}

function changed() {
  persist();
  scheduleRender(120);
  S.keyboard.patchChanged();
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
      body: JSON.stringify({ patches: S.patches, notes, muted_voices: [...S.mutedVoices], mute_noise: S.muteNoise,
                             muted_drum_hits: S.drumHits.filter((x) => x.muted).map((x) => [x.kind, x.onset]),
                             drum_settings: S.drumSettings || null }),
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

// One lane per voice, over the pitches it uses, and one for the drums with a row per drum that was hit.
const LANE_TITLE = 14;  // CSS pixels

function rollLanes() {
  const r = S.job.result, lanes = [];
  for (let v = 0; v < r.patches.length; v++) {
    const own = S.notes.filter((n) => n.voice === v).concat(r.reference ? r.reference.notes.filter((n) => (n.voice || 0) === v) : []);
    let lo = own.length ? Math.min(...own.map((n) => n.pitch)) : 60, hi = own.length ? Math.max(...own.map((n) => n.pitch)) : 60;
    lo = Math.floor(lo) - 2; hi = Math.ceil(hi) + 2;
    if (hi - lo < 12) { const m = (hi + lo) / 2; lo = Math.floor(m - 6); hi = Math.ceil(m + 6); }
    lanes.push({ kind: "voice", voice: v, lo, hi, rows: hi - lo + 1, title: r.patches.length > 1 ? `Voice ${v + 1}` : "Notes" });
  }
  const kinds = [...new Set(S.drumHits.map((h) => h.kind))].sort((a, b) => a - b);
  if (kinds.length) lanes.push({ kind: "drums", kinds, rows: kinds.length, title: "Drums" });
  return lanes;
}

function layoutRoll(c) {
  const lanes = rollLanes();
  const weight = (l) => (l.kind === "drums" ? 2.2 : 1) * l.rows;  // drum rows are taller: there are few of them
  const units = lanes.reduce((a, l) => a + weight(l), 0) || 1;
  const cssHeight = Math.round(Math.min(640, Math.max(180, units * 8 + lanes.length * LANE_TITLE)));
  if (c.style.height !== cssHeight + "px") c.style.height = cssHeight + "px";
  const [w, h, dpr] = sizeCanvas(c);
  const title = LANE_TITLE * dpr, unit = (h - title * lanes.length) / units;
  let y = 0;
  for (const l of lanes) {
    l.top = y;
    l.body = y + title;
    l.rowH = unit * (l.kind === "drums" ? 2.2 : 1);
    l.bottom = l.body + l.rowH * l.rows;
    y = l.bottom;
  }
  const r = S.job.result;
  return { lanes, w, h, dpr, dur: r.n_samples / r.sr };
}

function drumLabel(kind) {
  const d = (S.job.result.drums || [])[kind];
  const kit = (S.drumSettings && S.drumSettings[kind] && S.drumSettings[kind].kit) || (d && d.kit);
  return d ? d.name + (kit ? ` (${kitName(kit)})` : "") : `drum ${kind + 1}`;
}

function drawRoll() {
  const c = $("#roll");
  const { lanes, w, h, dpr, dur } = layoutRoll(c);
  const g = c.getContext("2d");
  const X = (t) => w * t / dur;
  g.clearRect(0, 0, w, h);
  const ref = S.job.result.reference;
  for (const l of lanes) {
    g.fillStyle = "#16181d";
    g.fillRect(0, l.top, w, l.body - l.top);
    g.fillStyle = l.kind === "voice" ? VOICE_COLORS[l.voice % 4] : "#c8ccd4";
    g.font = `600 ${10 * dpr}px system-ui`;
    g.fillText(l.title, 4 * dpr, l.top + 10.5 * dpr);
    if (l.kind === "voice") {
      const Y = (p) => l.body + (l.hi - p) * l.rowH;
      for (let p = l.lo; p <= l.hi; p++) {
        g.fillStyle = [1, 3, 6, 8, 10].includes(((p % 12) + 12) % 12) ? "#1d2027" : "#252a33";
        g.fillRect(0, Y(p), w, l.rowH);
        if (p % 12 === 0) { g.fillStyle = "#8d94a1"; g.font = `${9 * dpr}px system-ui`; g.fillText(noteName(p), 3 * dpr, Y(p) + l.rowH - 1.5 * dpr); }
      }
      if (ref) {
        g.strokeStyle = "#b8bfcc"; g.lineWidth = 1.5 * dpr;
        for (const n of ref.notes) if ((n.voice || 0) === l.voice) g.strokeRect(X(n.onset), Y(n.pitch) + 1, Math.max(X(n.duration), 2), l.rowH - 2);
      }
      const col = VOICE_COLORS[l.voice % 4];
      for (const n of S.notes) {
        if (n.voice !== l.voice) continue;
        const muted = n.muted || S.mutedVoices.has(n.voice);
        const x = X(n.onset), y = Y(n.pitch) + l.rowH * 0.18, ww = Math.max(X(n.duration), 3), hh = l.rowH * 0.64;
        g.globalAlpha = muted ? 1 : 0.35 + 0.65 * n.velocity;
        if (muted) { g.strokeStyle = col; g.setLineDash([3 * dpr, 3 * dpr]); g.strokeRect(x, y, ww, hh); g.setLineDash([]); }
        else { g.fillStyle = col; g.fillRect(x, y, ww, hh); }
        g.globalAlpha = 1;
      }
    } else {
      l.kinds.forEach((kind, i) => {
        const y = l.body + i * l.rowH;
        g.fillStyle = i % 2 ? "#1d2027" : "#252a33";
        g.fillRect(0, y, w, l.rowH);
        g.fillStyle = "#8d94a1"; g.font = `${9 * dpr}px system-ui`;
        g.fillText(drumLabel(kind), 3 * dpr, y + l.rowH - 3 * dpr);
      });
      for (const hit of S.drumHits) {
        const i = l.kinds.indexOf(hit.kind);
        const muted = hit.muted || S.muteNoise;
        const x = X(hit.onset), y = l.body + i * l.rowH + l.rowH * 0.18, ww = Math.max(5 * dpr, X(0.025)), hh = l.rowH * 0.64;
        g.globalAlpha = muted ? 1 : 0.35 + 0.65 * Math.min(hit.velocity, 1);
        if (muted) { g.strokeStyle = "#e6e8ec"; g.setLineDash([3 * dpr, 3 * dpr]); g.strokeRect(x, y, ww, hh); g.setLineDash([]); }
        else { g.fillStyle = "#e6e8ec"; g.fillRect(x, y, ww, hh); }
        g.globalAlpha = 1;
      }
    }
  }
  g.strokeStyle = "#ffffff14"; g.lineWidth = dpr;
  for (let t = 0; t <= dur; t += 0.5) { g.beginPath(); g.moveTo(X(t), 0); g.lineTo(X(t), h); g.stroke(); }
}

// The note or drum hit under the mouse: {note} or {hit}, or null.
function itemAt(ev) {
  const c = $("#roll");
  const rect = c.getBoundingClientRect();
  const { lanes, h, dur } = layoutRoll(c);
  const t = (ev.clientX - rect.left) / rect.width * dur;
  const y = (ev.clientY - rect.top) / rect.height * h;
  const l = lanes.find((x) => y >= x.body && y < x.bottom);
  if (!l) return null;
  const row = Math.floor((y - l.body) / l.rowH);
  if (l.kind === "voice") {
    const p = l.hi - row;
    let best = null;
    for (const n of S.notes) {
      if (n.voice === l.voice && Math.round(n.pitch) === p && t >= n.onset - 0.02 && t <= n.onset + Math.max(n.duration, 0.05)) best = n;
    }
    return best && { note: best };
  }
  const kind = l.kinds[row], slack = Math.max(0.03, 6 / rect.width * dur);
  let best = null;
  for (const hit of S.drumHits) {
    if (hit.kind === kind && Math.abs(hit.onset + 0.0125 - t) <= slack && (!best || Math.abs(hit.onset - t) < Math.abs(best.onset - t))) best = hit;
  }
  return best && { hit: best };
}

function drawAll() {
  if (!S.job || !S.job.result) return;
  drawSpec($("#spec-target"), S.specs2d.target);
  drawSpec($("#spec-tweak"), S.specs2d.tweaked);
  drawRoll();
}

// ---------------------------------------------------------------- library

const formatSize = (bytes) => (bytes >= 1e9 ? (bytes / 1e9).toFixed(1) + " GB" : Math.max(bytes / 1e6, 0.1).toFixed(1) + " MB");

// ---------------------------------------------------------------- drum kits

const GENERIC = { id: "noise", name: "Generic sounds", kind: "generic", drums: [], installed: true };
const kitName = (id) => (S.kitNames && S.kitNames[id]) || id;

async function refreshKits() {
  let r;
  try { r = await (await api("/api/drumkits")).json(); } catch (e) { return; }
  const installed = r.kits.filter((k) => k.installed).map((k) => k.id).join();
  if (S.kits && installed === S.kitsInstalled) return;  // nothing new: keep the boxes as they are
  const ticked = S.kits ? new Set(chosenKits()) : new Set(r.default);
  for (const k of r.kits) if (k.installed && S.kits && !S.kits.some((x) => x.id === k.id && x.installed)) ticked.add(k.id);  // just downloaded
  S.kits = r.kits;
  S.kitsInstalled = installed;
  S.kitDefault = r.default;
  S.kitNames = Object.fromEntries([...r.kits, GENERIC].map((k) => [k.id, k.name.replace(" (imitation)", "")]));
  const box = $("#kit-chips");
  box.innerHTML = "";
  for (const k of [...r.kits.filter((k) => k.installed), GENERIC]) {
    const chip = document.createElement("label");
    chip.className = "chip";
    const what = { machine: "a drum machine, made from its circuits", imitation: "an imitation: that machine played recordings, which can't be shipped",
                   recorded: "a recorded acoustic kit", folder: "your own samples", generic: "shapes its own sounds, so it finds hits but doesn't say which drum they are" }[k.kind];
    chip.title = `${k.name}: ${what}.` + (k.drums.length ? ` ${k.drums.length} drums: ${k.drums.join(", ")}.` : "") + (k.credit ? ` ${k.credit}.` : "");
    chip.innerHTML = `<input type="checkbox"> <span></span>`;
    chip.querySelector("span").textContent = kitName(k.id);
    const cb = chip.querySelector("input");
    cb.value = k.id;
    cb.checked = ticked.has(k.id);
    cb.onchange = syncKits;
    box.appendChild(chip);
  }
  const missing = r.kits.filter((k) => !k.installed);
  if (missing.length) {
    const more = document.createElement("button");
    more.type = "button";
    more.className = "link";
    more.textContent = `+ ${missing.map((k) => k.name).join(", ")} (download)`;
    more.title = "Recorded acoustic kits, from the Data menu";
    more.onclick = openDataDialog;
    box.appendChild(more);
  }
  syncKits();
}

function chosenKits() {
  return [...document.querySelectorAll("#kit-chips input:checked")].map((cb) => cb.value);
}

function syncKits() {
  const kits = chosenKits(), compare = $("#compare-kits").checked;
  $("#drum-kit").value = kits.join(",");
  document.querySelectorAll("#kit-chips .chip").forEach((c) => c.classList.toggle("on", c.querySelector("input").checked));
  $("#kit-hint").textContent = !kits.length ? "Choose at least one kit." :
    compare ? (kits.length < 2 ? "Choose at least two kits to compare." :
      `Each of the ${kits.length} kits is fitted to the drums on their own (Demucs's drum stem when it's installed), and the fit keeps the one with the lowest loss.`) :
    kits.includes("noise") && kits.length > 1 ? "The generic sounds can't be mixed with kits: compare them instead." :
    kits.length > 1 ? "Each drum plays whichever of these kits' versions fits it best." : "";
}

function kitProblem() {
  if ($("#fit-mode").value === "notes") return "";
  const kits = chosenKits(), compare = $("#compare-kits").checked;
  if (!kits.length) return "Choose at least one drum kit.";
  if (compare && kits.length < 2) return "Choose at least two drum kits to compare.";
  if (!compare && kits.includes("noise") && kits.length > 1) return "The generic sounds can't be mixed with kits: compare them instead.";
  return "";
}

async function refreshLibrary() {
  clearTimeout(S.libraryTimer);
  const problem = $("#data-error");
  try {
    S.library = await (await api("/api/library")).json();
    problem.hidden = true;
  } catch (e) {
    problem.hidden = false;
    problem.textContent = `Could not load the data library (${e.message}). If the app was updated recently, restart the server.`;
    return;
  }
  showTracks();
  showDataDialog();
  refreshKits();  // a kit that finished downloading can be chosen at once
  const active = S.library.downloads.some((d) => d.status === "queued" || d.status === "running");
  if (active || $("#data-dialog").open) S.libraryTimer = setTimeout(refreshLibrary, active ? 1500 : 5000);
}

// Search that ignores case and accents; every word has to appear in the track's title or its dataset's name.
const searchable = (text) => text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

function formatLength(seconds) {
  if (!seconds) return "";
  const s = Math.round(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function showTracks() {
  const list = $("#track-list"), words = searchable($("#track-search").value).split(/\s+/).filter(Boolean);
  const datasets = S.library ? S.library.datasets : [];
  const key = JSON.stringify([words, S.chosenKey || "", datasets.map((d) => [d.id, d.name, d.tracks.map((t) => t.id)])]);
  if (key === S.tracksShown) return;  // rebuilding would lose the scroll position for nothing
  S.tracksShown = key;
  const scroll = list.scrollTop;
  list.innerHTML = "";
  S.trackRows = [];

  const addRow = (title, length, rowKey, track) => {
    const row = document.createElement("div");
    row.className = "track-row" + (rowKey === (S.chosenKey || "") ? " chosen" : "");
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", rowKey === (S.chosenKey || ""));
    row.title = title;
    row.innerHTML = '<span class="track-title"></span><span class="track-length"></span>';
    row.querySelector(".track-title").textContent = title;
    row.querySelector(".track-length").textContent = length;
    row.onclick = () => chooseTrack(rowKey, track);
    list.appendChild(row);
    S.trackRows.push(row);
  };

  if (!words.length) addRow("Upload a file", "", "", null);
  let shown = 0, total = 0;
  for (const d of datasets) {
    const name = searchable(d.name);
    const matching = d.tracks.filter((t) => { const text = `${name} ${searchable(t.title)}`; return words.every((w) => text.includes(w)); });
    total += d.tracks.length;
    shown += matching.length;
    if (!matching.length) continue;
    const head = document.createElement("div");
    head.className = "track-group";
    head.textContent = `${d.name} (${matching.length}${matching.length === d.tracks.length ? "" : ` of ${d.tracks.length}`})`;
    list.appendChild(head);
    for (const t of matching) addRow(t.title, formatLength(t.duration_s), `${d.id}\n${t.id}`, t);
  }
  if (!shown && words.length) {
    const none = document.createElement("div");
    none.className = "hint track-none";
    none.textContent = "No tracks match.";
    list.appendChild(none);
  }
  list.scrollTop = scroll;
  $("#track-count").textContent = words.length ? `${shown} of ${total} tracks` : `${total} tracks`;
  S.activeRow = -1;
}

// Choose a row of the list, or none (key "") to upload a file.
function chooseTrack(key, track, fill = true) {
  S.chosenKey = key;
  S.chosenTrack = track;
  $("#track").value = track ? track.id : "";
  showTracks();
  showChosenTrack(fill);
}

function moveActiveRow(by) {
  const rows = S.trackRows;
  if (!rows.length) return;
  if (S.activeRow >= 0 && rows[S.activeRow]) rows[S.activeRow].classList.remove("active");
  S.activeRow = Math.min(Math.max(S.activeRow + by, 0), rows.length - 1);
  rows[S.activeRow].classList.add("active");
  rows[S.activeRow].scrollIntoView({ block: "nearest" });
}

function showChosenTrack(fill) {
  const form = $("#upload-form"), t = S.chosenTrack;
  $("#drop").hidden = !!t;
  $("#track-info").hidden = !t;
  $("#upload-btn").textContent = t ? "Fit" : "Upload and fit";
  const src = t ? `/api/library/audio?track=${encodeURIComponent(t.id)}` : "";
  const preview = $("#track-preview");
  if (preview.getAttribute("src") !== src) { if (src) preview.src = src; else preview.removeAttribute("src"); }
  if (!t) return;
  $("#track-note").textContent = `${t.title}${t.duration_s ? `, ${formatLength(t.duration_s)} long` : ""}. ${t.note}`.trim();
  if (fill) {
    form.elements.offset.value = t.offset;
    form.elements.duration.value = t.length;
    form.elements.voices.value = t.voices;
  }
}

function dataRow(ul, title, detail) {
  const li = document.createElement("li");
  li.innerHTML = '<div class="data-text"><b></b><div class="hint"></div></div>';
  li.querySelector("b").textContent = title;
  li.querySelector(".hint").textContent = detail;
  ul.appendChild(li);
  return li;
}
function actionButton(li, label, action) {
  const b = document.createElement("button");
  b.className = "small";
  b.textContent = label;
  b.onclick = async () => {
    b.disabled = true;
    try { await action(); } catch (e) { alert(e.message); }
    refreshLibrary();
  };
  li.appendChild(b);
}
const postJson = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

function showDataDialog() {
  const L = S.library;
  const busy = new Set(L.downloads.filter((d) => d.status === "queued" || d.status === "running").map((d) => d.dataset));
  $("#data-root").textContent = L.root;

  const select = $("#dl-source");
  if (!select.options.length) {
    for (const s of L.sources) {
      if (s.id === "zenodo" || L.can_use_youtube) select.add(new Option(s.label, s.id));
    }
    showSourceHelp();
  }

  const presets = $("#presets");
  presets.innerHTML = "";
  for (const p of L.presets) {
    const cached = L.datasets.find((d) => d.id === p.id);
    const li = dataRow(presets, p.name, `${p.description}${cached ? ` You have ${cached.tracks.length} ${cached.tracks.length === 1 ? "track" : "tracks"} (${formatSize(cached.size_bytes)}).` : ""}`);
    if (!busy.has(p.id) && !p.complete) actionButton(li, cached ? "Get the rest" : "Download", () => postJson("/api/library/downloads", { preset: p.id }));
  }

  const downloads = $("#downloads");
  $("#downloads-wrap").hidden = !L.downloads.length;
  downloads.innerHTML = "";
  for (const d of L.downloads.slice(0, 8)) {
    const running = d.status === "running" || d.status === "queued";
    const count = d.unit === "bytes" ? `${formatSize(d.done)} of ${formatSize(d.total)}` : `${d.done} of ${d.total}`;
    const li = dataRow(downloads, d.name, `${d.status}${d.total ? ", " + count : ""}${d.message ? ": " + d.message : ""}`);
    if (d.error) {
      const err = document.createElement("div");
      err.className = "error";
      err.textContent = d.error;
      li.querySelector(".data-text").appendChild(err);
    }
    if (running && d.total) {
      const bar = document.createElement("div");
      bar.className = "progress";
      bar.innerHTML = `<div style="width:${(100 * d.done) / d.total}%"></div>`;
      li.querySelector(".data-text").appendChild(bar);
    }
    if (running) actionButton(li, "Stop", () => api(`/api/library/downloads/${d.id}/cancel`, { method: "POST" }));
  }

  const datasets = $("#datasets");
  datasets.innerHTML = "";
  const cached = L.datasets.filter((d) => d.id !== "examples");
  if (!cached.length) datasets.innerHTML = '<li class="hint">Nothing yet.</li>';
  for (const d of cached) {
    const label = L.sources.find((s) => s.id === d.source);
    const li = dataRow(datasets, d.name, `${d.tracks.length} ${d.tracks.length === 1 ? "track" : "tracks"}, ${formatSize(d.size_bytes)}${label ? `. From ${label.label.toLowerCase()}: ${d.query}` : ""}`);
    actionButton(li, "Show in list", async () => {
      $("#track-search").value = d.name;
      showTracks();
      $("#data-dialog").close();
      $("#track-list").scrollTop = 0;
      $("#track-search").focus();
    });
    if (label && d.source !== "zenodo" && !busy.has(d.id)) {
      actionButton(li, "Get more", () => postJson("/api/library/downloads", { source: d.source, query: d.query, name: d.name, limit: d.limit || 25 }));
    }
  }
}

function showSourceHelp() {
  const s = S.library.sources.find((x) => x.id === $("#dl-source").value);
  if (!s) return;
  $("#dl-query").placeholder = s.placeholder;
  $("#dl-limit-wrap").hidden = s.id === "zenodo";
}

function openDataDialog() {
  $("#data-dialog").showModal();
  refreshLibrary();
}

// ---------------------------------------------------------------- wiring

function wire() {
  $("#data-btn").onclick = openDataDialog;
  $("#more-data").onclick = openDataDialog;
  $("#data-close").onclick = () => $("#data-dialog").close();
  $("#track-search").oninput = () => { $("#track-list").scrollTop = 0; showTracks(); };
  $("#track-search").onkeydown = (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      moveActiveRow(e.key === "ArrowDown" ? 1 : -1);
    } else if (e.key === "Enter") {
      e.preventDefault();  // choosing a track isn't submitting the form
      const row = S.trackRows[S.activeRow >= 0 ? S.activeRow : 0];
      if (row) row.click();
    } else if (e.key === "Escape" && e.target.value) {
      e.target.value = "";
      e.stopPropagation();
      showTracks();
    }
  };
  $("#dl-source").onchange = showSourceHelp;
  $("#download-form").onsubmit = async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const body = { source: fd.get("source"), query: fd.get("query"), name: fd.get("name") };
    if (fd.get("limit")) body.limit = parseInt(fd.get("limit"), 10);
    $("#dl-error").hidden = true;
    try {
      await postJson("/api/library/downloads", body);
      e.target.elements.query.value = "";
      e.target.elements.name.value = "";
    } catch (err) {
      $("#dl-error").hidden = false;
      $("#dl-error").textContent = err.message;
    }
    refreshLibrary();
  };

  const drop = $("#drop"), file = $("#file");
  const showFile = () => { $("#drop-text").textContent = file.files[0] ? file.files[0].name : "Drop an audio file here or click to choose"; };
  file.onchange = showFile;
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); file.files = e.dataTransfer.files; showFile(); };

  // drums-only fits have no voices to count
  // Drums alone are found far better in Demucs's drum stem (design.md), so it's ticked for them unless chosen by hand
  const separate = $("#upload-form").elements.separate;
  separate.addEventListener("change", () => { separate.dataset.chosen = "1"; });
  $("#fit-mode").onchange = (e) => {
    const mode = e.target.value, form = $("#upload-form").elements;
    form.voices.disabled = mode === "drums";
    $("#kit-wrap").hidden = $("#compare-wrap").hidden = mode === "notes";
    if (S.specs && S.specs.tools.demucs && !separate.dataset.chosen) separate.checked = mode === "drums";
  };
  $("#compare-kits").onchange = syncKits;
  $("#upload-form").onsubmit = async (e) => {
    e.preventDefault();
    const chosen = !!$("#track").value;
    if (!chosen && !file.files[0]) { alert("Choose an audio file first."); return; }
    const problem = kitProblem();
    if (problem) { alert(problem); return; }
    const fd = new FormData(e.target);
    if (chosen) fd.delete("file");
    else fd.delete("track");
    const q = QUALITY[fd.get("quality")];
    fd.delete("quality");
    fd.set("roll_steps", q.roll_steps);
    fd.set("note_steps", q.note_steps);
    const btn = $("#upload-btn");
    btn.disabled = true; btn.textContent = chosen ? "Starting…" : "Uploading…";
    try {
      const job = await (await api("/api/jobs", { method: "POST", body: fd })).json();
      await selectJob(job.id);
    } catch (err) { alert((chosen ? "Could not start the fit: " : "Upload failed: ") + err.message); }
    btn.disabled = false; btn.textContent = chosen ? "Fit" : "Upload and fit";
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
  $("#delete-btn").onclick = () => S.job && deleteJob(S.job);

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
    const item = itemAt(e);
    if (!item) return;
    const x = item.note || item.hit;
    x.muted = !x.muted;
    drawRoll(); showMetrics(); changed();
  };
  roll.onmousemove = (e) => {
    const item = itemAt(e);
    if (!item) { tip.hidden = true; return; }
    const rect = roll.getBoundingClientRect();
    tip.hidden = false;
    const n = item.note, x = item.hit;
    tip.textContent = n
      ? `${noteName(n.pitch)} (${n.pitch.toFixed(2)}) · ${n.onset.toFixed(3)} s · ${(n.duration * 1000).toFixed(0)} ms · vel ${n.velocity.toFixed(2)} · voice ${n.voice + 1}${n.muted ? " · muted" : ""}`
      : `${drumLabel(x.kind)} · ${x.onset.toFixed(3)} s · strength ${x.velocity.toFixed(2)}${x.muted ? " · muted" : ""}`;
    tip.style.left = Math.min(e.clientX - rect.left + 12, rect.width - 260) + "px";
    tip.style.top = (e.clientY - rect.top + 12) + "px";
  };
  roll.onmouseleave = () => (tip.hidden = true);

  $("#voice-mute").onchange = (e) => {
    if (e.target.checked) S.mutedVoices.add(S.voice); else S.mutedVoices.delete(S.voice);
    drawRoll(); changed();
  };
  $("#noise-mute").onchange = (e) => { S.muteNoise = e.target.checked; changed(); };
  $("#reset-btn").onclick = () => {
    S.patches[S.voice] = JSON.parse(JSON.stringify(S.fitted[S.voice]));
    S.notes = S.job.result.notes.map((n) => ({ ...n, muted: false }));
    S.mutedVoices.clear();
    buildSliders(); drawRoll(); showMetrics(); changed();
  };
  $("#drum-reset-btn").onclick = () => {
    S.drumSettings = fittedDrumSettings();
    S.drumHits.forEach((x) => { x.muted = false; });
    S.muteNoise = false;
    $("#noise-mute").checked = false;
    buildDrumSliders(); drawRoll(); showMetrics(); changed();
  };

  $("#exp-patch").onclick = () => {
    const out = { format: "musicfit-patch", version: 1, sr: S.job.result.sr, voices: S.patches };
    if (S.drumSettings) out.drums = S.drumSettings.map((s, k) => ({ name: S.job.result.drums[k].name, ...s }));
    download(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" }), baseName() + "_patch.json");
  };
  $("#exp-notes").onclick = () => {
    const notes = S.notes.filter((n) => !n.muted && !S.mutedVoices.has(n.voice)).map(({ muted, ...n }) => n);
    download(new Blob([JSON.stringify(notes, null, 2)], { type: "application/json" }), baseName() + "_notes.json");
  };
  $("#exp-midi").onclick = async () => {
    const notes = S.notes.filter((n) => !n.muted && !S.mutedVoices.has(n.voice)).map(({ muted, ...n }) => n);
    const drums = S.job.result.drums || [];
    const drum_hits = S.muteNoise ? [] : S.drumHits.filter((x) => !x.muted && drums[x.kind]).map((x) => [drums[x.kind].gm, x.onset, x.velocity]);
    const r = await api("/api/midi", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ notes, drum_hits }) });
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
      // drums by name, so a patch from another fit sets the drums both have
      const kind = drumVoiceKind();
      for (const drum of (kind && S.drumSettings && Array.isArray(d.drums)) ? d.drums : []) {
        const k = S.job.result.drums.findIndex((x) => x.name === drum.name);
        if (k < 0) continue;
        for (const [name, spec] of Object.entries(S.specs.drum_params[kind])) {
          if (drum[name] != null && Number.isFinite(+drum[name])) S.drumSettings[k][name] = Math.min(Math.max(+drum[name], spec.low), spec.high);
        }
        if (kind === "kit" && (S.job.result.drums[k].kits || []).includes(drum.kit)) S.drumSettings[k].kit = drum.kit;
      }
      buildSliders(); buildDrumSliders(); drawRoll(); changed();
    } catch (err) { alert("Could not read patch: " + err.message); }
  };

  let rt = null;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { drawAll(); if (S.job) drawLossChart(S.job.loss_curve || [], S.job.config); }, 100); });
}

async function main() {
  S.keyboard = setupKeyboard({
    getPatch: () => S.patches[S.voice],
    getHarmonics: () => S.job.result.n_harmonics,
    getSampleRate: () => S.job.result.sr,
    isActive: () => !$("#result-view").hidden && !!S.patches && S.patches.length > 0 && !!S.job && !!S.job.result,
    noteName,
    defaultSlide: () => !!(S.job && S.job.config.lead),
  });
  S.specs = await (await api("/api/specs")).json();
  $("#separate-wrap").hidden = !S.specs.tools.demucs;
  $("#basic-pitch-wrap").hidden = !S.specs.tools.basic_pitch;
  wire();
  refreshKits();
  refreshLibrary();
  await refreshJobs();
  setInterval(refreshJobs, 4000);
  const m = location.hash.match(/job=(\w+)/);
  if (m) selectJob(m[1]);
}
main();
