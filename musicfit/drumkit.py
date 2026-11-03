"""Drum machine kits to fit drums against: TR-808 and TR-909 style kits, or your own samples.

The 808 and 909 sounds are made here from what is known of the machines' circuits (bridged-T
resonators for the drums, six square waves at fixed inharmonic frequencies for the 808's metal,
filtered noise for snares and claps), so nothing has to be downloaded or licensed. They aren't
recordings of the machines, and nothing at 16 kHz can hold the top octave of a hi-hat anyway.

A kit of your own is a folder of audio files whose names say which drum each is ("BD kick.wav",
"snare 2.wav", "CH.wav", ...); see :data:`ALIASES`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

# The drums a kit can have, with their General MIDI note numbers (channel 10).
DRUMS = [("kick", 36), ("rim", 37), ("snare", 38), ("clap", 39), ("closed hat", 42), ("low tom", 45),
         ("open hat", 46), ("high tom", 50), ("crash", 49), ("cowbell", 56)]
KIT_SECONDS = 1.5

# words in a sample's file name that say which drum it is, checked in this order
ALIASES = [
    ("open hat", r"open|\boh\b|ohh"), ("closed hat", r"closed|\bch\b|chh|hi.?hat|hihat|\bhh\b"),
    ("kick", r"kick|\bbd\b|bass.?drum"), ("snare", r"snare|\bsd\b"), ("clap", r"clap|\bcp\b"),
    ("rim", r"rim|\brs\b|side.?stick"), ("low tom", r"low.?tom|\blt\b|floor"), ("high tom", r"hi(gh)?.?tom|\bht\b|\btom"),
    ("crash", r"crash|cymbal|\bcy\b|ride"), ("cowbell", r"cowbell|\bcb\b|bell"),
]
AUDIO_EXTS = (".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3")


@dataclass
class Kit:
    """Several kits' samples of the drums any of them has."""

    names: list[str]  # drum names, from DRUMS
    kit_names: list[str]
    samples: torch.Tensor  # [drums, kits, samples], each peak-normalised; zeros where a kit lacks the drum
    available: torch.Tensor  # [drums, kits] bool


# ---- sound making --------------------------------------------------------------


def _band(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr) + 1e-3
    return np.fft.irfft(spec / (1 + (lo / f) ** 4) / (1 + (f / hi) ** 4), len(x))


def _sweep(t: np.ndarray, sr: int, start: float, end: float, tau: float) -> np.ndarray:
    f = end + (start - end) * np.exp(-t / tau)
    return np.sin(2 * np.pi * np.cumsum(f) / sr)


def _square(t: np.ndarray, sr: int, f: float) -> np.ndarray:
    k = np.arange(1, int(0.45 * sr / f) + 1, 2)  # odd harmonics below Nyquist: no aliasing
    return 4 / np.pi * (np.sin(2 * np.pi * f * k[:, None] * t) / k[:, None]).sum(0)


def _metal(t: np.ndarray, sr: int, freqs) -> np.ndarray:
    return sum(_square(t, sr, f) for f in freqs)


def _bursts(t: np.ndarray, gaps: float, n: int, tau: float, tail_tau: float) -> np.ndarray:
    env = sum((t >= i * gaps) * np.exp(-(t - i * gaps).clip(0) / tau) for i in range(n))
    return env + 0.7 * (t >= (n - 1) * gaps) * np.exp(-(t - (n - 1) * gaps).clip(0) / tail_tau)


METAL_808 = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)
METAL_909 = (263.0, 400.0, 421.0, 474.0, 587.0, 845.0)


def _kit_808(sr: int) -> dict[str, np.ndarray]:
    t = np.arange(int(KIT_SECONDS * sr)) / sr
    rng = np.random.default_rng(808)
    noise = rng.standard_normal(len(t))
    e = lambda tau: np.exp(-t / tau)  # noqa: E731
    return {
        "kick": np.tanh(1.2 * (_sweep(t, sr, 80, 50, 0.02) * e(0.5) + 0.3 * _band(noise, sr, 1000, 5000) * e(0.001))),
        "rim": np.tanh(2 * (0.8 * np.sin(2 * np.pi * 1667 * t) * e(0.006) + 0.5 * np.sin(2 * np.pi * 455 * t) * e(0.01))),
        "snare": 0.7 * np.sin(2 * np.pi * 185 * t) * e(0.09) + 0.4 * np.sin(2 * np.pi * 330 * t) * e(0.06)
                 + 0.6 * _band(noise, sr, 1500, 8000) * e(0.13),
        "clap": _band(noise, sr, 800, 2400) * _bursts(t, 0.009, 4, 0.003, 0.12),
        "closed hat": _band(_metal(t, sr, METAL_808), sr, 5000, 9000) * e(0.045),
        "open hat": _band(_metal(t, sr, METAL_808), sr, 5000, 9000) * e(0.35),
        "low tom": _sweep(t, sr, 110, 90, 0.06) * e(0.3) + 0.05 * _band(noise, sr, 200, 2000) * e(0.03),
        "high tom": _sweep(t, sr, 220, 180, 0.05) * e(0.2) + 0.05 * _band(noise, sr, 300, 3000) * e(0.03),
        "crash": _band(_metal(t, sr, METAL_808), sr, 3000, 9000) * e(0.9) + 0.3 * _band(noise, sr, 5000, 9000) * e(0.6),
        "cowbell": _band(_square(t, sr, 540) + _square(t, sr, 800), sr, 400, 3000) * (0.6 * e(0.01) + 0.4 * e(0.15)),
    }


def _kit_909(sr: int) -> dict[str, np.ndarray]:
    t = np.arange(int(KIT_SECONDS * sr)) / sr
    rng = np.random.default_rng(909)
    noise = rng.standard_normal(len(t))
    e = lambda tau: np.exp(-t / tau)  # noqa: E731
    hat = 0.6 * _band(noise, sr, 6000, 9000) + 0.4 * _band(_metal(t, sr, METAL_909), sr, 6000, 9000)
    return {
        "kick": np.tanh(1.5 * (_sweep(t, sr, 400, 55, 0.012) * e(0.3) + 0.4 * _band(noise, sr, 2000, 8000) * e(0.003))),
        "rim": np.tanh(2 * (0.8 * np.sin(2 * np.pi * 2200 * t) * e(0.005) + 0.5 * np.sin(2 * np.pi * 800 * t) * e(0.008)
                            + 0.3 * _band(noise, sr, 2000, 8000) * e(0.002))),
        "snare": 0.6 * np.sin(2 * np.pi * 190 * t) * e(0.07) + 0.3 * np.sin(2 * np.pi * 300 * t) * e(0.05)
                 + 0.9 * _band(noise, sr, 500, 9000) * e(0.18),
        "clap": _band(noise, sr, 1000, 3500) * _bursts(t, 0.007, 4, 0.002, 0.18),
        "closed hat": hat * e(0.05),
        "open hat": hat * e(0.45),
        "low tom": _sweep(t, sr, 160, 90, 0.06) * e(0.35) + 0.15 * _band(noise, sr, 300, 3000) * e(0.05),
        "high tom": _sweep(t, sr, 280, 170, 0.05) * e(0.25) + 0.15 * _band(noise, sr, 400, 4000) * e(0.05),
        "crash": _band(noise, sr, 3000, 9000) * e(1.2) + 0.2 * _band(_metal(t, sr, METAL_909), sr, 3000, 9000) * e(0.8),
        # the 909 has no cowbell
    }


BUILT_IN = {"808": _kit_808, "909": _kit_909}


def _drum_of(file_name: str) -> str | None:
    stem = os.path.splitext(file_name)[0].lower().replace("_", " ").replace("-", " ")
    for drum, pattern in ALIASES:
        if re.search(pattern, stem):
            return drum
    return None


def load_kit_folder(path: str, sr: int) -> dict[str, np.ndarray]:
    """Samples from a folder, one per drum that a file's name identifies (the first, alphabetically)."""
    from .audio_io import load_audio

    sounds = {}
    for f in sorted(os.listdir(path)):
        drum = _drum_of(f)
        if drum and drum not in sounds and f.lower().endswith(AUDIO_EXTS):
            sounds[drum] = load_audio(os.path.join(path, f), sr, 0.0, KIT_SECONDS).numpy().astype(np.float64)
    if not sounds:
        raise ValueError(f"{path} has no samples named after a drum (kick, snare, hat, ...)")
    return sounds


@lru_cache(maxsize=8)
def make_kit(spec: str, sr: int) -> Kit:
    """The kits in ``spec``, comma-separated: "808", "909", or a folder of samples.

    Returns:
        A Kit of every drum at least one of them has, in the order of DRUMS.
    """
    kits, kit_names = [], []
    for item in (s.strip() for s in spec.split(",") if s.strip()):
        kits.append(BUILT_IN[item](sr) if item in BUILT_IN else load_kit_folder(item, sr))
        kit_names.append(item if item in BUILT_IN else os.path.basename(os.path.normpath(item)))
    if not kits:
        raise ValueError("no drum kit given")
    L = int(KIT_SECONDS * sr)
    names = [name for name, _ in DRUMS if any(name in k for k in kits)]
    samples = torch.zeros(len(names), len(kits), L)
    for d, name in enumerate(names):
        for k, kit in enumerate(kits):
            if name in kit:
                x = kit[name][:L]
                samples[d, k, :len(x)] = torch.from_numpy(x / (np.abs(x).max() + 1e-9)).float()
    return Kit(names, kit_names, samples, samples.abs().amax(dim=2) > 0)


GM_NOTE = dict(DRUMS)
# General MIDI drum notes onto the drums here, for scoring fits against MIDI and for reading in kits
GM_TO_DRUM = {35: "kick", 36: "kick", 37: "rim", 38: "snare", 40: "snare", 39: "clap", 42: "closed hat", 44: "closed hat",
              46: "open hat", 41: "low tom", 43: "low tom", 45: "low tom", 47: "high tom", 48: "high tom", 50: "high tom",
              49: "crash", 51: "crash", 52: "crash", 53: "crash", 55: "crash", 57: "crash", 59: "crash", 56: "cowbell"}
# coarser groups, as drum transcription is usually scored
GROUP = {"kick": "kick", "rim": "snare", "snare": "snare", "clap": "snare", "closed hat": "hat", "open hat": "hat",
         "low tom": "tom", "high tom": "tom", "crash": "cymbal", "cowbell": "other"}
