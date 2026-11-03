"""Drum kits to fit drums against: drum machines made from scratch, recorded kits, or your own samples.

The analogue machines (TR-808, TR-909, TR-606, CR-78) are made here from what is known of their
circuits (bridged-T resonators for the drums, banks of square waves at fixed inharmonic
frequencies for metal, filtered noise for snares, claps and shakers), so nothing has to be
downloaded or licensed. They aren't recordings of the machines, and nothing at 16 kHz can hold
the top octave of a hi-hat anyway.

The LinnDrum, Oberheim DMX and TR-707 played recordings of real drums, which can't be made from a
circuit or shipped here. Their kits are imitations: acoustic-style drums made from scratch,
stored as those machines stored theirs (8-bit companded, or 6-bit), with the drums each had.

A kit of your own is a folder of audio files whose names say which drum each is ("BD kick.wav",
"snare 2.wav", "CH.wav", ...); see :data:`ALIASES`. Recorded kits that the web app downloads
(:mod:`musicfit.dataset.drumkits`) are read the same way, through their own mappings.
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
         ("open hat", 46), ("mid tom", 47), ("high tom", 50), ("crash", 49), ("ride", 51), ("cowbell", 56),
         ("tambourine", 54), ("shaker", 70), ("conga", 63), ("bongo", 60), ("claves", 75)]
KIT_SECONDS = 1.5

# words in a sample's file name that say which drum it is, checked in this order
ALIASES = [
    ("open hat", r"open|\boh\b|ohh"), ("closed hat", r"closed|\bch\b|chh|hi.?hat|hihat|\bhh\b"),
    ("kick", r"kick|\bbd\b|bass.?drum"), ("snare", r"snare|\bsd\b"), ("clap", r"clap|\bcp\b"),
    ("rim", r"rim|\brs\b|side.?stick"), ("low tom", r"low.?tom|\blt\b|floor"), ("mid tom", r"mid.?tom|\bmt\b"),
    ("high tom", r"hi(gh)?.?tom|\bht\b|\btom"), ("ride", r"ride"), ("crash", r"crash|cymbal|\bcy\b"),
    ("tambourine", r"tamb"), ("shaker", r"shak|maraca|cabasa|\bma\b"), ("conga", r"conga"), ("bongo", r"bongo"),
    ("claves", r"clave|\bcl\b"), ("cowbell", r"cowbell|\bcb\b|bell"),
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


def _partials(t: np.ndarray, sr: int, rng, n: int, lo: float, hi: float, tau: float) -> np.ndarray:
    """n sines at random frequencies between lo and hi (below Nyquist), each decaying around tau."""
    if lo >= 0.45 * sr:
        return np.zeros_like(t)
    freqs = rng.uniform(lo, min(hi, 0.45 * sr), n)
    taus = tau * rng.uniform(0.5, 1.5, n)
    return (np.sin(2 * np.pi * freqs[:, None] * t + rng.uniform(0, 2 * np.pi, n)[:, None]) * np.exp(-t / taus[:, None])).sum(0) / np.sqrt(n)


def _stored(x: np.ndarray, bits: int, mu: float = 0.0) -> np.ndarray:
    """x as a sample-based drum machine stored it: peak-normalised, companded if mu, then quantised."""
    x = x / (np.abs(x).max() + 1e-12)
    if mu:
        x = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = 2 ** (bits - 1) - 1
    x = np.round(x * q) / q
    return np.sign(x) * np.expm1(np.abs(x) * np.log1p(mu)) / mu if mu else x


METAL_808 = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)
METAL_909 = (263.0, 400.0, 421.0, 474.0, 587.0, 845.0)
METAL_606 = (245.0, 351.0, 437.0, 543.0, 612.0, 871.0)


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
        "mid tom": _sweep(t, sr, 165, 135, 0.05) * e(0.25) + 0.05 * _band(noise, sr, 250, 2500) * e(0.03),
        "conga": _sweep(t, sr, 400, 370, 0.02) * e(0.12),  # the toms' circuits, tuned up
        "shaker": _band(noise, sr, 5000, 9000) * (1 - e(0.002)) * e(0.02),  # maracas
        "claves": np.sin(2 * np.pi * 2500 * t) * e(0.02),
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
        "mid tom": _sweep(t, sr, 220, 130, 0.05) * e(0.3) + 0.15 * _band(noise, sr, 350, 3500) * e(0.05),
        # the 909's cymbals were 6-bit recordings
        "ride": _stored(_partials(t, sr, rng, 40, 2500, 9000, 1.2) + 0.3 * _band(noise, sr, 4000, 9000) * e(0.8), 6) * e(1.5),
        # the 909 has no cowbell
    }


def _kit_606(sr: int) -> dict[str, np.ndarray]:
    t = np.arange(int(KIT_SECONDS * sr)) / sr
    rng = np.random.default_rng(606)
    noise = rng.standard_normal(len(t))
    e = lambda tau: np.exp(-t / tau)  # noqa: E731
    metal = _band(0.6 * _metal(t, sr, METAL_606) + 0.4 * noise, sr, 6000, 9000)
    return {  # the 606 has these seven, higher and shorter than the 808's
        "kick": np.tanh(1.3 * _sweep(t, sr, 110, 62, 0.012) * e(0.17)),
        "snare": 0.6 * np.sin(2 * np.pi * 235 * t) * e(0.05) + 0.3 * np.sin(2 * np.pi * 365 * t) * e(0.04)
                 + 0.7 * _band(noise, sr, 1500, 8000) * e(0.09),
        "low tom": _sweep(t, sr, 190, 150, 0.04) * e(0.16),
        "high tom": _sweep(t, sr, 280, 220, 0.04) * e(0.12),
        "closed hat": metal * e(0.035),
        "open hat": metal * e(0.25),
        "crash": _band(_metal(t, sr, METAL_606) + 0.5 * noise, sr, 3000, 9000) * e(0.6),
    }


def _kit_cr78(sr: int) -> dict[str, np.ndarray]:
    t = np.arange(int(KIT_SECONDS * sr)) / sr
    rng = np.random.default_rng(78)
    noise = rng.standard_normal(len(t))
    e = lambda tau: np.exp(-t / tau)  # noqa: E731
    return {  # soft, short and noise-based, with the Latin percussion of a home rhythm machine
        "kick": _sweep(t, sr, 72, 56, 0.01) * e(0.22),
        "snare": 0.5 * np.sin(2 * np.pi * 260 * t) * e(0.05) + 0.8 * _band(noise, sr, 2000, 8000) * e(0.08),
        "rim": np.sin(2 * np.pi * 2000 * t) * e(0.008) + 0.3 * _band(noise, sr, 2000, 6000) * e(0.003),
        "closed hat": _band(noise, sr, 7000, 9000) * e(0.03),
        "open hat": _band(noise, sr, 7000, 9000) * e(0.2),
        "crash": _band(noise, sr, 4000, 9000) * e(0.7),
        "cowbell": _band(_square(t, sr, 587) + _square(t, sr, 845), sr, 400, 3000) * e(0.12),
        "tambourine": _band(noise, sr, 6000, 9000) * _bursts(t, 0.012, 3, 0.006, 0.18),
        "shaker": _band(noise, sr, 6000, 9000) * (1 - e(0.003)) * e(0.03),
        "conga": _sweep(t, sr, 245, 225, 0.02) * e(0.14),
        "bongo": _sweep(t, sr, 430, 400, 0.015) * e(0.09),
        "claves": np.sin(2 * np.pi * 2400 * t) * e(0.025),
    }


def _acoustic(sr: int, seed: int, v: dict) -> dict[str, np.ndarray]:
    """Acoustic-style drums made from scratch: the stand-in for a sample-based machine's recordings.

    ``v`` sets each drum's character: pitches in Hz and decays in s (see the machines below).
    """
    t = np.arange(int(KIT_SECONDS * sr)) / sr
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(t))
    e = lambda tau: np.exp(-t / tau)  # noqa: E731
    f, tau = v["kick"]
    tom = lambda f, tau: (_sweep(t, sr, 1.15 * f, f, 0.04) * e(tau) + 0.4 * np.sin(2 * np.pi * 1.5 * f * t) * e(0.5 * tau)  # noqa: E731
                          + 0.1 * _band(noise, sr, 300, 3000) * e(0.02))
    hat = _partials(t, sr, rng, 30, 3000, 12000, 1.0) + 0.7 * _band(noise, sr, 6000, 10000)
    out = {
        "kick": np.tanh(1.3 * (_sweep(t, sr, 2.2 * f, f, 0.015) * e(tau) + v["click"] * _band(noise, sr, 1500, 6000) * e(0.003)
                               + 0.3 * _band(noise, sr, 40, 200) * e(0.05))),
        "snare": (np.sin(2 * np.pi * v["snare"][0] * t) * e(v["snare"][1]) + 0.5 * np.sin(2 * np.pi * 1.58 * v["snare"][0] * t) * e(0.6 * v["snare"][1])
                  + v["snare"][3] * _band(noise, sr, 1800, 9000) * e(v["snare"][2]) + 0.5 * _band(noise, sr, 800, 4000) * e(0.004)),
        "rim": np.tanh(np.sin(2 * np.pi * 1700 * t) * e(0.01) + 0.5 * np.sin(2 * np.pi * 680 * t) * e(0.015)
                       + 0.4 * _band(noise, sr, 2000, 8000) * e(0.002)),
        "clap": _band(noise, sr, v["clap"][0], v["clap"][1]) * _bursts(t, v["clap"][2], 4, 0.002, v["clap"][3]),
        "closed hat": hat * e(v["hats"][0]),
        "open hat": hat * e(v["hats"][1]),
        "crash": (_partials(t, sr, rng, 40, 2000, 9000, 0.7 * v["crash"]) + 0.8 * _band(noise, sr, 3000, 9000)) * e(v["crash"]),
    }
    for name, f_tom in zip(("low tom", "mid tom", "high tom"), v["toms"]):
        out[name] = tom(f_tom, v["tom_decay"])
    if "ride" in v:
        out["ride"] = (_partials(t, sr, rng, 40, 2500, 9000, v["ride"]) + 0.6 * (np.sin(2 * np.pi * 2350 * t) + np.sin(2 * np.pi * 3550 * t)) * e(0.6)
                       + 0.3 * _band(noise, sr, 5000, 9000) * e(0.3)) * e(v["ride"])
    if "cowbell" in v:
        out["cowbell"] = _band(_square(t, sr, v["cowbell"][0]) + _square(t, sr, v["cowbell"][1]), sr, 400, 4000) * (0.6 * e(0.008) + 0.4 * e(0.2))
    if "tambourine" in v:
        out["tambourine"] = (_band(noise, sr, 5000, 9500) * _bursts(t, 0.006, 3, 0.004, 0.12) + 0.5 * _partials(t, sr, rng, 12, 6000, 9000, 0.1))
    if "shaker" in v:
        out["shaker"] = _band(noise, sr, 4000, 9000) * (1 - e(0.004)) * e(v["shaker"])
    if "conga" in v:
        out["conga"] = _sweep(t, sr, 1.05 * v["conga"], v["conga"], 0.02) * e(0.2) + 0.3 * _band(noise, sr, 1000, 4000) * e(0.004)
    return {name: _stored(x, v["bits"], v.get("mu", 0.0)) * (np.abs(x).max() > 0) for name, x in out.items()}


def _kit_linndrum(sr: int) -> dict[str, np.ndarray]:
    return _acoustic(sr, 1982, {  # dry and tight; 8-bit companded, with cabasa, tambourine and congas
        "bits": 8, "mu": 255.0, "kick": (55, 0.25), "click": 0.6, "snare": (190, 0.08, 0.2, 1.0), "clap": (900, 2800, 0.008, 0.15),
        "hats": (0.05, 0.45), "toms": (95, 130, 175), "tom_decay": 0.4, "crash": 1.3, "ride": 1.6, "cowbell": (560, 845),
        "tambourine": True, "shaker": 0.05, "conga": 200})


def _kit_dmx(sr: int) -> dict[str, np.ndarray]:
    return _acoustic(sr, 1981, {  # a boomy, punchy kick and a cracking snare; 8-bit companded
        "bits": 8, "mu": 255.0, "kick": (50, 0.35), "click": 0.9, "snare": (200, 0.1, 0.15, 1.3), "clap": (1000, 3200, 0.01, 0.2),
        "hats": (0.045, 0.35), "toms": (90, 120, 160), "tom_decay": 0.45, "crash": 1.1, "tambourine": True, "shaker": 0.04})


def _kit_707(sr: int) -> dict[str, np.ndarray]:
    return _acoustic(sr, 1985, {  # short and bright, from low-resolution samples
        "bits": 6, "kick": (60, 0.18), "click": 0.5, "snare": (210, 0.06, 0.13, 1.0), "clap": (1100, 3500, 0.007, 0.12),
        "hats": (0.04, 0.3), "toms": (110, 150, 200), "tom_decay": 0.25, "crash": 0.9, "ride": 1.2, "cowbell": (830, 550),
        "tambourine": True})


BUILT_IN = {"808": _kit_808, "909": _kit_909, "606": _kit_606, "cr78": _kit_cr78, "linndrum": _kit_linndrum,
            "dmx": _kit_dmx, "707": _kit_707}
# what the web app calls them
BUILT_IN_NAMES = {"808": "TR-808", "909": "TR-909", "606": "TR-606", "cr78": "CR-78", "linndrum": "LinnDrum (imitation)",
                  "dmx": "Oberheim DMX (imitation)", "707": "TR-707 (imitation)"}


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
    """The kits in ``spec``, comma-separated: built-in ones (:data:`BUILT_IN`), downloaded ones
    (:data:`musicfit.dataset.drumkits.RECORDED`), folders under ``<data>/drumkits/`` by name,
    or any folder of samples by path.

    Returns:
        A Kit of every drum at least one of them has, in the order of DRUMS.
    """
    from .dataset.drumkits import RECORDED, kit_path, read_recorded

    kits, kit_names = [], []
    for item in (s.strip() for s in spec.split(",") if s.strip()):
        if item in BUILT_IN:
            kits.append(BUILT_IN[item](sr))
        elif item in RECORDED:
            kits.append(read_recorded(item, sr, KIT_SECONDS))
        elif kit_path(item) is not None:
            kits.append(load_kit_folder(kit_path(item), sr))
        elif os.path.isdir(item):
            kits.append(load_kit_folder(item, sr))
        else:
            raise ValueError(f"no drum kit called {item}: use {', '.join(BUILT_IN)}, a downloaded kit or a folder of samples")
        kit_names.append(os.path.basename(os.path.normpath(item)))
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
              46: "open hat", 41: "low tom", 43: "low tom", 45: "low tom", 47: "mid tom", 48: "mid tom", 50: "high tom",
              49: "crash", 52: "crash", 55: "crash", 57: "crash", 51: "ride", 53: "ride", 59: "ride", 56: "cowbell",
              54: "tambourine", 69: "shaker", 70: "shaker", 82: "shaker", 60: "bongo", 61: "bongo", 62: "conga", 63: "conga",
              64: "conga", 75: "claves"}
# coarser groups, as drum transcription is usually scored
GROUP = {"kick": "kick", "rim": "snare", "snare": "snare", "clap": "snare", "closed hat": "hat", "open hat": "hat",
         "low tom": "tom", "mid tom": "tom", "high tom": "tom", "crash": "cymbal", "ride": "cymbal", "cowbell": "other",
         "tambourine": "other", "shaker": "other", "conga": "other", "bongo": "other", "claves": "other"}
