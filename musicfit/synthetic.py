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
