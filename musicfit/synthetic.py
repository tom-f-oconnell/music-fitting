"""Synthetic targets: a random melody rendered with a random patch (ground truth known)."""

from __future__ import annotations

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


def synthetic_target(seed: int = 1, length_s: float = 3.0, n_notes: int = 10, sr: int = 16000):
    """Returns (target audio, reference notes, true SynthParams)."""
    gen = torch.Generator().manual_seed(seed)
    true_params = SynthParams.random(gen)
    synth = Synth(sr, params=true_params)
    ref = random_melody(gen, n_notes, length_s)
    T = int(length_s * sr)
    with torch.no_grad():
        target = synth.render_notes(
            torch.tensor([n.pitch for n in ref]), torch.tensor([n.onset for n in ref]),
            torch.tensor([n.duration for n in ref]), torch.tensor([n.velocity for n in ref]), T)
    return target, ref, true_params
