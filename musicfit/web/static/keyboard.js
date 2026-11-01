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

// Build the keyboard in the page. `host` supplies getPatch(), getHarmonics(), getSampleRate(), isActive(),
// noteName(pitch) and defaultSlide(); returns patchChanged(), jobChanged() and allOff().
function setupKeyboard(host) {
  const piano = document.querySelector("#piano");
  const held = new Map();  // key or pointer -> {note, pitch, el}
  let ctx = null, synth = null;
  let base = 60, velocity = 100;
  let enabled = localStorage.getItem("musicfit:keyboard") !== "off";
  const els = new Map();

  const whites = KEY_MAP.filter(([, , o]) => !BLACK.has(o % 12)).length;
  let white = 0;
  for (const [code, label, offset] of KEY_MAP) {
    const el = document.createElement("div");
    const black = BLACK.has(offset % 12);
    el.className = "pkey " + (black ? "black" : "white");
    el.dataset.offset = offset;
    el.style.left = `${(100 * (black ? white : white++)) / whites}%`;
    el.style.width = `${100 / whites}%`;
    el.innerHTML = `<span class="pname"></span><span class="pletter">${label === "'" ? "&#39;" : label}</span>`;
    piano.appendChild(el);
    els.set(code, el);
  }

  function context() {
    const sr = host.getSampleRate();
    if (!ctx || ctx.sampleRate !== sr) {
      if (ctx) ctx.close();
      ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sr, latencyHint: "interactive" });
      synth = new LiveSynth(ctx);
    }
    if (ctx.state !== "running") ctx.resume();
    return ctx;
  }

  function press(id, offset, el) {
    if (held.has(id)) return;
    const pitch = base + offset;
    const c = context();
    synth.setPatch(host.getPatch(), host.getHarmonics());
    const previous = [...held.values()].pop();
    const slide = document.querySelector("#kbd-slide").checked && previous ? previous.pitch : null;
    held.set(id, { note: synth.noteOn(pitch, velocity / 127, c.currentTime, slide), pitch, el });
    el.classList.add("down");
  }
  function release(id) {
    const h = held.get(id);
    if (!h) return;
    held.delete(id);
    synth.noteOff(h.note, ctx.currentTime);
    if (![...held.values()].some((x) => x.el === h.el)) h.el.classList.remove("down");
  }
  function allOff() { [...held.keys()].forEach(release); }

  function show() {
    document.querySelector("#kbd-range").textContent = `${host.noteName(base)}–${host.noteName(base + KEYS_SPAN)}`;
    document.querySelector("#kbd-velocity").textContent = velocity;
    document.querySelector("#kbd-on").checked = enabled;
    piano.classList.toggle("off", !enabled);
    for (const el of els.values()) {
      const pitch = base + Number(el.dataset.offset);
      el.querySelector(".pname").textContent = pitch % 12 === 0 ? host.noteName(pitch) : "";
    }
  }
  function shiftOctave(by) { base = clamp(base + 12 * by, 0, 108); show(); }
  function stepVelocity(by) {
    const i = VELOCITIES.findIndex((v) => v >= velocity);
    velocity = VELOCITIES[clamp(i + by, 0, VELOCITIES.length - 1)];
    show();
  }
  function setEnabled(on) {
    enabled = on;
    localStorage.setItem("musicfit:keyboard", on ? "on" : "off");
    if (!on) [...held.keys()].filter((id) => !id.startsWith("pointer")).forEach(release);
    show();
  }

  const typing = () => {
    const el = document.activeElement;
    if (!el) return false;
    if (el.isContentEditable || el.tagName === "TEXTAREA" || el.tagName === "SELECT") return true;
    return el.tagName === "INPUT" && !["range", "checkbox", "radio", "button", "file"].includes(el.type);
  };
  const ignoreKey = (e) => e.ctrlKey || e.metaKey || e.altKey || typing() || !host.isActive() || document.querySelector("dialog[open]");
  document.addEventListener("keydown", (e) => {
    if (ignoreKey(e)) return;
    if (e.code === "KeyM") { if (!e.repeat) setEnabled(!enabled); return; }
    if (!enabled) return;
    const control = { KeyZ: () => shiftOctave(-1), KeyX: () => shiftOctave(1), KeyC: () => stepVelocity(-1), KeyV: () => stepVelocity(1) }[e.code];
    const el = els.get(e.code);
    if (!control && !el) return;
    e.preventDefault();  // Firefox's quick find is on ' and /
    if (e.repeat) return;
    if (control) control();
    else press(e.code, Number(el.dataset.offset), el);
  });
  document.addEventListener("keyup", (e) => release(e.code));
  window.addEventListener("blur", allOff);
  document.addEventListener("visibilitychange", () => { if (document.hidden) allOff(); });

  // mouse and touch; dragging across the keys plays each in turn
  const keyAt = (e) => { const el = document.elementFromPoint(e.clientX, e.clientY); return el && el.closest(".pkey"); };
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

  document.querySelector("#kbd-octave-down").onclick = () => shiftOctave(-1);
  document.querySelector("#kbd-octave-up").onclick = () => shiftOctave(1);
  document.querySelector("#kbd-velocity-down").onclick = () => stepVelocity(-1);
  document.querySelector("#kbd-velocity-up").onclick = () => stepVelocity(1);
  document.querySelector("#kbd-on").onchange = (e) => setEnabled(e.target.checked);
  show();

  return {
    patchChanged() { if (synth && host.isActive()) synth.setPatch(host.getPatch(), host.getHarmonics()); },
    jobChanged() { allOff(); document.querySelector("#kbd-slide").checked = host.defaultSlide(); show(); },
    allOff,
    state: () => ({ base, velocity, enabled, held: held.size, sounding: synth ? synth.notes.size : 0, audio: ctx ? ctx.state : null }),
  };
}
