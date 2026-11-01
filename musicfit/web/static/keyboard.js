"use strict";
// Playing the current patch live. LiveSynth is musicfit/synth.py as a Web Audio graph, so a note
// starts at once instead of after a round trip to the server (and without an AudioWorklet, which
// browsers only allow over HTTPS or on localhost). tests/test_keyboard.py compares it with the
// Python synth, so a change to one needs the same change to the other.

const DRIVE_RANGE = 8;     // the waveshaper takes -1..1, so the mix is scaled down by this going in
const REVERB_SECONDS = 2;  // length of the impulse response, as NoiseReverb
const MAX_LIVE_NOTES = 32; // beyond this the oldest is cut, in case long releases pile up

function clamp(x, lo, hi) { return Math.min(Math.max(x, lo), hi); }

function gaussian() { return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()); }

// Fourier-series amplitudes of the waveform mix: waveform_harmonics in synth.py.
function waveformHarmonics(waveform, nHarmonics) {
  const raw = ["sine", "saw", "square", "triangle"].map((w) => Math.max(waveform[w] ?? 0, 1e-3));
  const sum = raw.reduce((a, b) => a + b, 0);
  const [sine, saw, square, triangle] = raw.map((w) => w / sum);
  const imag = new Float32Array(nHarmonics + 1);
  for (let k = 1; k <= nHarmonics; k++) {
    const odd = k % 2 === 1;
    let a = (saw * (2 / Math.PI) * (odd ? 1 : -1)) / k;
    if (k === 1) a += sine;
    if (odd) {
      a += (square * (4 / Math.PI)) / k;
      a += (triangle * (8 / Math.PI ** 2) * (((k - 1) / 2) % 2 === 0 ? 1 : -1)) / k ** 2;
    }
    imag[k] = a;
  }
  return imag;
}

// The ADSR of synth.py: a linear attack, then exponential decay towards the sustain level.
function startEnvelope(param, t, peak, attack, decay, sustain) {
  param.setValueAtTime(0, t);
  param.linearRampToValueAtTime(peak, t + attack);
  param.setTargetAtTime(peak * sustain, t + attack, decay);
}
function envelopeLevel(elapsed, attack, decay, sustain) {
  return elapsed < attack ? elapsed / attack : sustain + (1 - sustain) * Math.exp(-(elapsed - attack) / decay);
}
function releaseEnvelope(param, t, level, release) {
  param.cancelScheduledValues(t);
  param.setValueAtTime(level, t);
  param.setTargetAtTime(0, t, release);
}

class LiveSynth {
  constructor(ctx) {
    this.ctx = ctx;
    this.notes = new Set();
    this.p = null;
    this.bus = ctx.createGain();
    this.bus.gain.value = 1 / DRIVE_RANGE;
    this.shaper = ctx.createWaveShaper();
    this.delay = ctx.createDelay(1.5);
    this.echoIn = ctx.createGain();
    this.feedback = ctx.createGain();
    this.echoMix = ctx.createGain();
    this.afterDelay = ctx.createGain();
    this.reverb = ctx.createConvolver();
    this.reverb.normalize = false;
    this.reverbMix = ctx.createGain();
    this.master = ctx.createGain();
    // drive -> feedback delay -> reverb -> gain, each effect adding to the dry signal as in Synth.effects
    this.bus.connect(this.shaper);
    this.shaper.connect(this.afterDelay);
    this.shaper.connect(this.echoIn);
    this.echoIn.connect(this.delay);
    this.delay.connect(this.feedback);
    this.feedback.connect(this.echoIn);
    this.delay.connect(this.echoMix);
    this.echoMix.connect(this.afterDelay);
    this.afterDelay.connect(this.master);
    this.afterDelay.connect(this.reverb);
    this.reverb.connect(this.reverbMix);
    this.reverbMix.connect(this.master);
    this.master.connect(ctx.destination);
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.master.connect(this.analyser);
    this.levelBuffer = new Float32Array(this.analyser.fftSize);

    const noise = ctx.createBuffer(1, 2 * ctx.sampleRate, ctx.sampleRate);
    const samples = noise.getChannelData(0);
    for (let i = 0; i < samples.length; i++) samples[i] = 0.3 * gaussian();
    this.noise = ctx.createBufferSource();
    this.noise.buffer = noise;
    this.noise.loop = true;
    this.noise.start();
  }

  // Apply a patch: the effects and the filters of held notes change at once, envelopes from the next note.
  setPatch(patch, nHarmonics) {
    const ctx = this.ctx, p = patch, t = ctx.currentTime;
    const first = !this.p;
    this.p = p;
    const waveKey = JSON.stringify([p.waveform, nHarmonics]);
    if (waveKey !== this.waveKey) {
      this.waveKey = waveKey;
      const imag = waveformHarmonics(p.waveform, nHarmonics);
      this.wave = ctx.createPeriodicWave(new Float32Array(imag.length), imag, { disableNormalization: true });
      for (const n of this.notes) n.oscillators.forEach((o) => o.setPeriodicWave(this.wave));
    }
    if (p.drive !== this.drive) {
      this.drive = p.drive;
      const curve = new Float32Array(8193);
      for (let i = 0; i < curve.length; i++) {
        curve[i] = Math.tanh(p.drive * DRIVE_RANGE * (2 * i / (curve.length - 1) - 1)) / Math.tanh(p.drive);
      }
      this.shaper.curve = curve;
    }
    if (p.reverb_rt60 !== this.rt60) {
      this.rt60 = p.reverb_rt60;
      this.reverb.buffer = this.impulseResponse(p.reverb_rt60);
    }
    const set = (param, v) => { if (first) param.value = v; else param.setTargetAtTime(v, t, 0.01); };
    set(this.delay.delayTime, p.delay_time);
    set(this.feedback.gain, p.delay_feedback);
    set(this.echoMix.gain, p.delay_mix);
    set(this.reverbMix.gain, p.reverb_mix);
    set(this.master.gain, p.gain);
    for (const n of this.notes) {
      for (const f of n.filters) { set(f.frequency, this.cutoff(p)); set(f.Q, this.q(p)); }
      n.noiseGain.gain.setTargetAtTime(p.noise_level, t, 0.01);
    }
  }

  // The peak of the output just now, for the level meter.
  level() {
    this.analyser.getFloatTimeDomainData(this.levelBuffer);
    return this.levelBuffer.reduce((peak, x) => Math.max(peak, Math.abs(x)), 0);
  }

  cutoff(p) { return clamp(p.cutoff, 20, this.ctx.sampleRate / 2); }
  q(p) { return 20 * Math.log10(p.resonance); }  // Web Audio's low-pass Q is in dB

  impulseResponse(rt60) {
    const n = REVERB_SECONDS * this.ctx.sampleRate;
    const buffer = this.ctx.createBuffer(1, n, this.ctx.sampleRate);
    const ir = buffer.getChannelData(0);
    let energy = 0;
    for (let i = 0; i < n; i++) {
      ir[i] = gaussian() * Math.exp((-6.9078 * i) / this.ctx.sampleRate / rt60);
      energy += ir[i] * ir[i];
    }
    const norm = Math.sqrt(energy) + 1e-8;
    for (let i = 0; i < n; i++) ir[i] /= norm;
    return buffer;
  }

  // Start a note (pitch in MIDI numbers, velocity 0..1) and return a handle for noteOff.
  // With slideFrom, it glides in from that pitch as a note with a full slide does in synth.py.
  noteOn(pitch, velocity, when, slideFrom = null) {
    const ctx = this.ctx, p = this.p;
    const t = when ?? ctx.currentTime;
    if (this.notes.size >= MAX_LIVE_NOTES) this.stopNote(this.notes.values().next().value, t);
    const amp = ctx.createGain();
    amp.gain.value = 0;
    amp.connect(this.bus);
    const filters = [ctx.createBiquadFilter(), ctx.createBiquadFilter()];  // 24 dB/oct: two 2-pole sections
    for (const f of filters) {
      f.type = "lowpass";
      f.frequency.value = this.cutoff(p);
      f.Q.value = this.q(p);
    }
    filters[0].connect(filters[1]);
    filters[1].connect(amp);
    const noiseGain = ctx.createGain();
    noiseGain.gain.value = p.noise_level;
    this.noise.connect(noiseGain);
    noiseGain.connect(amp);

    const spread = p.unison_cents >= 0.5 ? p.unison_cents : 0;
    const copies = spread ? [[-spread, 0.0021], [0, 0], [spread, 0.0037]] : [[0, 0]];  // phases unrelated
    const sum = ctx.createGain();
    sum.gain.value = spread ? 1 / Math.sqrt(3) : 1;
    sum.connect(filters[0]);
    const f0 = 440 * 2 ** ((pitch - 69) / 12);
    const oscillators = copies.map(([cents, shift]) => {
      const o = ctx.createOscillator();
      o.setPeriodicWave(this.wave);
      o.frequency.value = f0;
      if (slideFrom === null) o.detune.value = cents;
      else {
        o.detune.setValueAtTime(cents + 100 * (slideFrom - pitch), t);
        o.detune.setTargetAtTime(cents, t, p.glide_time);
      }
      o.connect(sum);
      o.start(t + shift);
      return o;
    });

    // the filter envelope is in octaves above the cutoff, which detune (in cents) expresses exactly
    const octaves = 1200 * p.filt_env_amount * velocity;
    startEnvelope(amp.gain, t, velocity, p.amp_attack, p.amp_decay, p.amp_sustain);
    for (const f of filters) startEnvelope(f.detune, t, octaves, p.filt_attack, p.filt_decay, p.filt_sustain);
    const note = { amp, filters, noiseGain, sum, oscillators, start: t, velocity, octaves, patch: p, released: false };
    oscillators[oscillators.length - 1].onended = () => {
      amp.disconnect();
      filters.forEach((f) => f.disconnect());
      sum.disconnect();
      noiseGain.disconnect();
      this.noise.disconnect(noiseGain);
      this.notes.delete(note);
    };
    this.notes.add(note);
    return note;
  }

  noteOff(note, when) {
    if (note.released) return;
    note.released = true;
    const t = Math.max(when ?? this.ctx.currentTime, note.start);
    const p = note.patch, elapsed = t - note.start;
    releaseEnvelope(note.amp.gain, t, note.velocity * envelopeLevel(elapsed, p.amp_attack, p.amp_decay, p.amp_sustain), p.amp_release);
    const fenv = envelopeLevel(elapsed, p.filt_attack, p.filt_decay, p.filt_sustain);
    for (const f of note.filters) releaseEnvelope(f.detune, t, note.octaves * fenv, p.filt_release);
    note.oscillators.forEach((o) => o.stop(t + 10 * p.amp_release));  // -87 dB
  }

  stopNote(note, t) {
    note.released = true;
    note.amp.gain.cancelScheduledValues(t);
    note.amp.gain.setTargetAtTime(0, t, 0.005);
    note.oscillators.forEach((o) => o.stop(t + 0.05));
  }
}

// ---------------------------------------------------------------- the keyboard

// Ableton Live's computer MIDI keyboard: the middle row is the white keys from note 60 (Live's C3),
// the row above the black keys, Z/X the octave, C/V the velocity, M switches it on and off.
const KEY_MAP = [
  ["KeyA", "A", 0], ["KeyW", "W", 1], ["KeyS", "S", 2], ["KeyE", "E", 3], ["KeyD", "D", 4], ["KeyF", "F", 5],
  ["KeyT", "T", 6], ["KeyG", "G", 7], ["KeyY", "Y", 8], ["KeyH", "H", 9], ["KeyU", "U", 10], ["KeyJ", "J", 11],
  ["KeyK", "K", 12], ["KeyO", "O", 13], ["KeyL", "L", 14], ["KeyP", "P", 15], ["Semicolon", ";", 16], ["Quote", "'", 17],
];
const BLACK = new Set([1, 3, 6, 8, 10]);
const VELOCITIES = [1, 20, 40, 60, 80, 100, 120, 127];
const KEYS_SPAN = 17;

// Boxes that take letters keep the keyboard. A number box or a dropdown has no use for A to ', so it
// doesn't: focus stays in those easily, after choosing an option or (in some browsers) clicking a button.
const TEXT_INPUTS = new Set(["text", "search", "url", "email", "password", "tel"]);
function textEntry() {
  const el = document.activeElement;
  if (!el) return null;
  return el.isContentEditable || el.tagName === "TEXTAREA" || (el.tagName === "INPUT" && TEXT_INPUTS.has(el.type)) ? el : null;
}

// Build the keyboard in the page. `host` supplies getPatch(), getHarmonics(), getSampleRate(), isActive(),
// noteName(pitch) and defaultSlide(); returns patchChanged(), jobChanged(), allOff() and state().
function setupKeyboard(host) {
  const $k = (sel) => document.querySelector(sel);
  const piano = $k("#piano");
  const held = new Map();  // key or pointer -> {note, pitch, el, label}
  let ctx = null, synth = null, ctxRate = null, soundProblem = "", meterRunning = false, toastTimer = null;
  let base = 60, velocity = 100, message = "Press a key to play.", warning = false;
  let enabled = localStorage.getItem("musicfit:keyboard") !== "off";
  const els = new Map();
  const labelOf = new Map([["KeyZ", "Z"], ["KeyX", "X"], ["KeyC", "C"], ["KeyV", "V"], ["KeyM", "M"]]);
  const codeByKey = new Map([...labelOf].map(([code, label]) => [label.toLowerCase(), code]));

  const whites = KEY_MAP.filter(([, , o]) => !BLACK.has(o % 12)).length;
  let white = 0;
  for (const [code, label, offset] of KEY_MAP) {
    const el = document.createElement("div");
    const black = BLACK.has(offset % 12);
    el.className = "pkey " + (black ? "black" : "white");
    el.dataset.offset = offset;
    el.dataset.label = label;
    el.style.left = `${(100 * (black ? white : white++)) / whites}%`;
    el.style.width = `${100 / whites}%`;
    el.innerHTML = `<span class="pname"></span><span class="pletter">${label === "'" ? "&#39;" : label}</span>`;
    piano.appendChild(el);
    els.set(code, el);
    labelOf.set(code, label);
    codeByKey.set(label.toLowerCase(), code);
  }
  // some remote desktops and input methods don't report physical keys
  const codeOf = (e) => (e.code && e.code !== "Unidentified" ? e.code : codeByKey.get((e.key || "").toLowerCase()) || "");

  // ---- what the user sees: keys held, the last thing that happened, and the sound

  function say(text, isWarning = false) {
    message = text;
    warning = isWarning;
    showStatus();
  }
  function soundText() {
    if (soundProblem) return `No sound: ${soundProblem}`;
    if (!ctx) return "Sound starts with the first note";
    if (ctx.state === "running") return "Sound on";
    return "Sound is waiting for the browser: click the page";
  }
  function showStatus(fromKey = false) {
    const pressed = [...held.values()];
    const chips = pressed.map((h) => `<kbd>${h.label === "'" ? "&#39;" : h.label}</kbd> ${host.noteName(h.pitch)}`).join(" ");
    const text = $k("#kbd-message");
    $k("#kbd-pressed").innerHTML = chips;
    text.textContent = message;
    text.classList.toggle("warn", warning);
    const sound = $k("#kbd-sound");
    sound.textContent = soundText();
    sound.classList.toggle("warn", !!soundProblem || (!!ctx && ctx.state !== "running"));
    if (fromKey) showToast(chips, message);
  }
  // the Play box is far down the page: keep key presses visible from anywhere
  function showToast(chips, text) {
    const box = $k(".card.play").getBoundingClientRect();
    const toast = $k("#kbd-toast");
    clearTimeout(toastTimer);
    if (box.top >= 0 && box.bottom <= window.innerHeight) { toast.hidden = true; return; }
    toast.innerHTML = chips || "";
    toast.appendChild(document.createTextNode((chips ? "  " : "") + text));
    toast.classList.toggle("warn", warning);
    toast.hidden = false;
    toastTimer = setTimeout(() => { if (!held.size) toast.hidden = true; }, 2000);
  }
  function runMeter() {
    if (meterRunning || !synth) return;
    meterRunning = true;
    const bar = $k("#kbd-meter");
    const frame = () => {
      const level = synth ? synth.level() : 0;
      bar.style.width = `${Math.min(100, 100 * Math.sqrt(level))}%`;
      if (synth && (synth.notes.size || level > 0.001)) requestAnimationFrame(frame);
      else { meterRunning = false; bar.style.width = "0"; }
    };
    requestAnimationFrame(frame);
  }

  // ---- notes

  function context() {
    const sr = host.getSampleRate();
    if (!ctx || ctxRate !== sr) {
      if (ctx) ctx.close();
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      try {
        ctx = new AudioContextClass({ sampleRate: sr, latencyHint: "interactive" });
      } catch (e) {  // this browser won't run at the fit's rate: the notes still sound right at its own
        ctx = new AudioContextClass();
      }
      ctxRate = sr;
      ctx.onstatechange = () => showStatus();
      synth = new LiveSynth(ctx);
    }
    if (ctx.state !== "running") ctx.resume().catch((e) => { soundProblem = e.message; showStatus(); });
    return ctx;
  }

  function press(id, offset, el) {
    if (held.has(id)) return;
    const pitch = base + offset;
    const h = { note: null, pitch, el, label: el.dataset.label };
    const previous = [...held.values()].pop();
    held.set(id, h);
    el.classList.add("down");
    message = `${host.noteName(pitch)}, velocity ${velocity}`;
    warning = false;
    try {  // the key has already lit up: a problem with the sound shouldn't hide that it was pressed
      const c = context();
      synth.setPatch(host.getPatch(), host.getHarmonics());
      const slide = $k("#kbd-slide").checked && previous ? previous.pitch : null;
      h.note = synth.noteOn(pitch, velocity / 127, c.currentTime, slide);
      soundProblem = "";
      runMeter();
    } catch (e) {
      soundProblem = e.message;
      console.error(e);
    }
    showStatus(true);
  }
  function release(id) {
    const h = held.get(id);
    if (!h) return;
    held.delete(id);
    if (h.note) synth.noteOff(h.note, ctx.currentTime);
    if (![...held.values()].some((x) => x.el === h.el)) h.el.classList.remove("down");
    showStatus(true);
  }
  function allOff() { [...held.keys()].forEach(release); }

  function show() {
    $k("#kbd-range").textContent = `${host.noteName(base)}–${host.noteName(base + KEYS_SPAN)}`;
    $k("#kbd-velocity").textContent = velocity;
    $k("#kbd-on").checked = enabled;
    piano.classList.toggle("off", !enabled);
    for (const el of els.values()) {
      const pitch = base + Number(el.dataset.offset);
      el.querySelector(".pname").textContent = pitch % 12 === 0 ? host.noteName(pitch) : "";
    }
  }
  function shiftOctave(by) { base = clamp(base + 12 * by, 0, 108); show(); say(`Keys ${host.noteName(base)}–${host.noteName(base + KEYS_SPAN)}`); showStatus(true); }
  function stepVelocity(by) {
    const i = VELOCITIES.findIndex((v) => v >= velocity);
    velocity = VELOCITIES[clamp(i + by, 0, VELOCITIES.length - 1)];
    show();
    say(`Velocity ${velocity}`);
    showStatus(true);
  }
  function setEnabled(on) {
    enabled = on;
    localStorage.setItem("musicfit:keyboard", on ? "on" : "off");
    if (!on) [...held.keys()].filter((id) => !id.startsWith("pointer")).forEach(release);
    show();
    say(on ? "Computer keyboard on." : "Computer keyboard off. Press M or tick the box to turn it back on.", !on);
    showStatus(true);
  }

  // ---- keys: listened for first of all, so nothing in the page can swallow them

  const controls = { KeyZ: () => shiftOctave(-1), KeyX: () => shiftOctave(1), KeyC: () => stepVelocity(-1), KeyV: () => stepVelocity(1) };
  const inTextBox = "Typing goes to the text box. Click the piano or press Esc to play.";
  window.addEventListener("keydown", (e) => {
    if (!host.isActive() || $k("dialog[open]")) return;
    const code = codeOf(e), box = textEntry();
    if (e.key === "Escape" && box) { box.blur(); return; }
    const ours = els.has(code) || code in controls || code === "KeyM";
    if (!ours) {
      if (!box && !e.ctrlKey && !e.metaKey && !e.altKey && e.key && e.key.length === 1 && e.key !== " ") say(`${e.key.toUpperCase()} isn't a key on this keyboard.`);
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;  // browser shortcuts
    if (box) { say(`${labelOf.get(code)} didn't play. ${inTextBox}`, true); showStatus(true); return; }
    e.preventDefault();  // Firefox's quick find is on ' and /, and these aren't for a dropdown or number box
    if (e.repeat) return;
    if (code === "KeyM") { setEnabled(!enabled); return; }
    if (!enabled) { say(`${labelOf.get(code)} didn't play: the computer keyboard is off. Press M or tick the box.`, true); showStatus(true); return; }
    if (code in controls) controls[code]();
    else press(code, Number(els.get(code).dataset.offset), els.get(code));
  }, true);
  window.addEventListener("keyup", (e) => release(codeOf(e)), true);
  window.addEventListener("blur", () => {
    allOff();
    say("This window doesn't have the keyboard. Click the page to play.", true);
  });
  window.addEventListener("focus", () => say(textEntry() ? inTextBox : "Press a key to play.", !!textEntry()));
  document.addEventListener("focusin", () => { if (textEntry()) say(inTextBox, true); });
  document.addEventListener("focusout", () => { if (warning && message === inTextBox) say("Press a key to play."); });
  document.addEventListener("visibilitychange", () => { if (document.hidden) allOff(); });

  // ---- mouse and touch; dragging across the keys plays each in turn

  const keyAt = (e) => { const el = document.elementFromPoint(e.clientX, e.clientY); return el && el.closest(".pkey"); };
  $k(".card.play").addEventListener("pointerdown", () => { const box = textEntry(); if (box) box.blur(); });
  piano.addEventListener("pointerdown", (e) => {
    const el = keyAt(e);
    if (!el) return;
    e.preventDefault();
    press(`pointer${e.pointerId}`, Number(el.dataset.offset), el);
  });
  piano.addEventListener("pointermove", (e) => {
    const h = held.get(`pointer${e.pointerId}`);
    const el = h && keyAt(e);
    if (el && el !== h.el) {
      release(`pointer${e.pointerId}`);
      press(`pointer${e.pointerId}`, Number(el.dataset.offset), el);
    }
  });
  for (const type of ["pointerup", "pointercancel"]) window.addEventListener(type, (e) => release(`pointer${e.pointerId}`));

  $k("#kbd-octave-down").onclick = () => shiftOctave(-1);
  $k("#kbd-octave-up").onclick = () => shiftOctave(1);
  $k("#kbd-velocity-down").onclick = () => stepVelocity(-1);
  $k("#kbd-velocity-up").onclick = () => stepVelocity(1);
  $k("#kbd-on").onchange = (e) => setEnabled(e.target.checked);
  show();
  showStatus();

  return {
    patchChanged() { if (synth && host.isActive()) synth.setPatch(host.getPatch(), host.getHarmonics()); },
    jobChanged() { allOff(); $k("#kbd-slide").checked = host.defaultSlide(); show(); say(textEntry() ? inTextBox : "Press a key to play.", !!textEntry()); },
    allOff,
    state: () => ({ base, velocity, enabled, message, warning, soundProblem, held: held.size, sounding: synth ? synth.notes.size : 0, audio: ctx ? ctx.state : null, level: synth ? synth.level() : 0 }),
  };
}
