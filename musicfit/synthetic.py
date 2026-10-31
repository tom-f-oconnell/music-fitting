"""Synthetic targets: a random melody rendered with a random patch (ground truth known)."""

from __future__ import annotations

import math
import subprocess

import numpy as np
import torch

from .notes import Note
from .synth import Synth, SynthParams


def random_melody(gen, n_notes, length_s, pitch_lo=45, pitch_hi=76, chord_prob=0.2):
    """A sequence on a 16th-note grid at 120 bpm, minor pentatonic, with occasional dyads."""
    scale = [0, 3, 5, 7, 10]
    pitches = [p for p in range(pitch_lo, pitch_hi + 1) if (p - pitch_lo) % 12 in scale]
    step = 0.125
    slots = torch.randperm(int((length_s - 0.5) / step), generator=gen)[:n_notes].sort().values
    notes = []
    for s in slots.tolist():
        on = 0.05 + s * step
        dur = step * int(torch.randint(1, 4, (), generator=gen))
        vel = 0.5 + 0.5 * float(torch.rand((), generator=gen))
        k = int(torch.randint(len(pitches), (), generator=gen))
        notes.append(Note(float(pitches[k]), on, dur, vel))
        if float(torch.rand((), generator=gen)) < chord_prob and k + 2 < len(pitches):
            notes.append(Note(float(pitches[k + 2]), on, dur, vel))
    return notes


MISMATCHES = ("tuning", "unison", "noise", "drums", "codec")


def synthetic_target(seed: int = 1, length_s: float = 3.0, n_notes: int = 10, sr: int = 16000,
                     mismatch: dict | None = None):
    """Returns (target audio, reference notes, true SynthParams).

    mismatch adds things real recordings have and the fitter's synth cannot
    produce exactly, to measure robustness with known ground truth (the melody
    and patch stay those of the seed; reference pitches stay integers):

      tuning: cents; every note shifted off the A440 semitone grid
      unison: cents; three copies of each note at -x, 0, +x cents (supersaw style), the
              outer two shifted by a random 0-5 ms so their phases are not aligned
      noise:  dB re the peak; constant white noise floor (RMS)
      drums:  dB re the peak; kick on the beats, hi-hat on the off-beats (120 bpm)
      codec:  kbit/s; Opus encode/decode round trip through ffmpeg
    """
    m = mismatch or {}
    unknown = set(m) - set(MISMATCHES)
    if unknown:
        raise KeyError(f"unknown mismatch kinds: {sorted(unknown)}")
    gen = torch.Generator().manual_seed(seed)
    true_params = SynthParams.random(gen)
    synth = Synth(sr, params=true_params)
    ref = random_melody(gen, n_notes, length_s)
    T = int(length_s * sr)
    pitch = torch.tensor([n.pitch for n in ref]) + m.get("tuning", 0.0) / 100
    onset, duration, velocity = (torch.tensor([getattr(n, a) for n in ref]) for a in ("onset", "duration", "velocity"))
    extra = torch.Generator().manual_seed(10_000 + seed)  # keeps the melody/patch stream untouched
    with torch.no_grad():
        if "unison" in m:
            shifts = [0.005 * float(torch.rand((), generator=extra)) for _ in range(2)]
            target = (synth.render_notes(pitch - m["unison"] / 100, onset + shifts[0], duration, velocity, T)
                      + synth.render_notes(pitch, onset, duration, velocity, T)
                      + synth.render_notes(pitch + m["unison"] / 100, onset + shifts[1], duration, velocity, T)) / 3
        else:
            target = synth.render_notes(pitch, onset, duration, velocity, T)
    peak = float(target.abs().max())
    if "drums" in m:
        drums = _drums(T, sr, extra)
        target = target + drums * peak * 10 ** (m["drums"] / 20) / float(drums.abs().max())
    if "noise" in m:
        target = target + torch.randn(T, generator=extra) * peak * 10 ** (m["noise"] / 20)
    if "codec" in m:
        target = _opus_round_trip(target, sr, m["codec"])
    return target, ref, true_params


def acid_target(seed: int = 1, length_s: float = 3.0, sr: int = 16000, drums_db: float | None = None):
    """A TB-303-style line with slides and accents, from a separate time-domain synth.

    Rendered sample by sample with a saturating 4-pole ladder filter at 4x the sample rate,
    so the fitter's synth (harmonic-domain filter, per-note envelopes) cannot reproduce it
    exactly. A note with a slide holds its gate and glides into the next note, which then
    doesn't retrigger the envelopes; an accent means a louder, brighter, faster filter sweep.

    Args:
        seed: selects the pattern and sound.
        length_s: length in seconds.
        sr: sample rate.
        drums_db: if given, add kick and hi-hat at this level in dB re the peak.

    Returns:
        (target audio, reference notes, dict of the line's settings).
    """
    rng = np.random.default_rng(seed)
    oversample = 4
    fs = sr * oversample
    n = int(length_s * fs)
    bpm = rng.uniform(115, 135)
    step = 15 / bpm
    root = int(rng.integers(36, 44))
    scale = [0, 0, 3, 5, 7, 10, 12]
    wave = "saw" if rng.random() < 0.7 else "square"
    res = rng.uniform(1.5, 3.6)
    base_cutoff = np.exp(rng.uniform(np.log(150), np.log(500)))
    env_octaves = rng.uniform(2.0, 5.0)
    filter_decay = rng.uniform(0.15, 0.5)
    glide_tau = rng.uniform(0.02, 0.06)

    ref, events = [], []
    for s in range(int((length_s - 0.25) / step)):
        if rng.random() < 0.15:
            continue
        on = 0.05 + s * step
        pitch = root + int(rng.choice(scale)) + (12 if rng.random() < 0.15 else 0)
        accent, slide = rng.random() < 0.25, rng.random() < 0.25
        gate = step if slide else step * rng.uniform(0.5, 0.8)
        events.append((on, pitch, gate, accent, slide))
        ref.append(Note(float(pitch), on, gate, 1.0 if accent else 0.6))

    freq_midi, amp_t, cut_t = np.zeros(n), np.zeros(n), np.zeros(n)
    pitch_now, gate_end = float(events[0][1]) if events else 48.0, 0.0
    idx, prev_slid = 0, False
    fenv, aenv, boost, f_tau = 0.0, 0.0, 1.0, filter_decay
    attack, release = 1 / (0.003 * fs), np.exp(-1 / (0.008 * fs))
    for i in range(n):
        t = i / fs
        while idx < len(events) and t >= events[idx][0]:
            on, pitch_target, gate, accent, slide = events[idx]
            if not (prev_slid and on - gate_end < 0.01):
                fenv, boost, f_tau = 1.0, 1.5 if accent else 1.0, 0.08 if accent else filter_decay
                pitch_now = float(pitch_target)
                aenv = min(aenv, 0.1)
            gate_end = on + gate
            prev_slid = slide
            idx += 1
            target_pitch, level = pitch_target, 1.0 if accent else 0.6
        if idx:
            pitch_now += (target_pitch - pitch_now) * (1 / (glide_tau * fs))
            aenv = min(1.0, aenv + attack) if t < gate_end else aenv * release
            fenv *= np.exp(-1 / (f_tau * fs))
            amp_t[i] = aenv * level
        freq_midi[i] = pitch_now
        cut_t[i] = base_cutoff * 2 ** (env_octaves * boost * fenv)

    phase = np.cumsum(440.0 * 2 ** ((freq_midi - 69) / 12)) / fs % 1.0
    osc = 2 * phase - 1 if wave == "saw" else np.where(phase < 0.5, 1.0, -1.0)
    g = 1 - np.exp(-2 * math.pi * np.minimum(cut_t, 0.45 * sr) / fs)
    y1 = y2 = y3 = y4 = 0.0
    out = np.zeros(n)
    for i, (x, gi) in enumerate(zip(osc.tolist(), g.tolist())):
        u = math.tanh(1.5 * (x - res * y4))
        y1 += gi * (u - y1)
        y2 += gi * (y1 - y2)
        y3 += gi * (y2 - y3)
        y4 += gi * (y3 - y4)
        out[i] = y4
    out = np.tanh(1.5 * out * amp_t)
    spec = np.fft.rfft(out)[: int(length_s * sr) // 2 + 1]
    audio = torch.from_numpy(np.fft.irfft(spec, int(length_s * sr)) / oversample).float()
    audio = audio / (audio.abs().max() + 1e-8) * 0.5
    if drums_db is not None:
        drums = _drums(audio.numel(), sr, torch.Generator().manual_seed(10_000 + seed))
        audio = audio + drums * 0.5 * 10 ** (drums_db / 20) / float(drums.abs().max())
    info = {"wave": wave, "bpm": bpm, "resonance_k": res, "cutoff": base_cutoff, "env_octaves": env_octaves,
            "filter_decay": filter_decay, "glide_tau": glide_tau, "drums_db": drums_db,
            "accents": sum(e[3] for e in events), "slides": sum(e[4] for e in events)}
    return audio, ref, info


def _drums(T: int, sr: int, gen: torch.Generator, beat_s: float = 0.5) -> torch.Tensor:
    """Kick (pitch-swept sine) on every beat, short high-passed noise hi-hat half a beat later."""
    out = torch.zeros(T)
    t = torch.arange(T) / sr
    for k in range(int(T / sr / beat_s) + 1):
        for start, kind in ((k * beat_s, "kick"), ((k + 0.5) * beat_s, "hat")):
            on = t >= start
            tt = t[on] - start
            if kind == "kick":
                freq = 45.0 + 75.0 * torch.exp(-tt / 0.03)
                hit = torch.sin(2 * math.pi * torch.cumsum(freq, 0) / sr) * torch.exp(-tt / 0.15)
            else:
                noise = torch.randn(len(tt) + 1, generator=gen)
                hit = 0.4 * (noise[1:] - noise[:-1]) * torch.exp(-tt / 0.03)
            out[on] += hit
    return out


def _opus_round_trip(x: torch.Tensor, sr: int, kbps: float) -> torch.Tensor:
    common = ["ffmpeg", "-nostdin", "-v", "error"]
    raw = ["-f", "f32le", "-ac", "1", "-ar", str(sr)]
    enc = subprocess.run(common + raw + ["-i", "-", "-c:a", "libopus", "-b:a", f"{kbps}k", "-f", "ogg", "-"],
                         input=x.numpy().astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    dec = subprocess.run(common + ["-i", "-"] + raw + ["-"], input=enc, capture_output=True, check=True).stdout
    y = torch.from_numpy(np.frombuffer(dec, dtype=np.float32).copy())
    return torch.nn.functional.pad(y, (0, max(0, x.numel() - y.numel())))[: x.numel()]
