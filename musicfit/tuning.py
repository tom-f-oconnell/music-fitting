"""Estimate how far a recording's tuning is from the A440 equal-tempered grid."""

from __future__ import annotations

import math

import numpy as np
import torch


def estimate_tuning(audio: torch.Tensor, sr: int, fmin: float = 60.0, fmax: float = 2000.0,
                    n_fft: int = 8192) -> tuple[float, float]:
    """Global tuning offset in cents, in [-50, 50), and how consistent the evidence is.

    Spectral peaks of the frame-averaged magnitude spectrum (parabolically
    interpolated) are compared to the nearest A440 semitone; the offset is the
    magnitude-weighted circular mean of their deviations. The second value is
    the length of that mean vector, from 0 (peaks all over the place: no
    clear tuning, e.g. drums or glides) to 1 (every peak agrees).
    """
    x = audio.detach().cpu().double().numpy()
    n = min(n_fft, 1 << (max(len(x), 2).bit_length() - 1))
    window = np.hanning(n)
    hops = range(0, len(x) - n + 1, n // 4)
    S = np.mean([np.abs(np.fft.rfft(x[i:i + n] * window)) for i in hops], axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = np.nonzero((freqs > fmin) & (freqs < min(fmax, sr / 2)))[0]
    band = band[(band > 0) & (band < len(S) - 1)]
    if len(band) == 0 or S[band].max() <= 0:
        return 0.0, 0.0
    floor = 0.05 * S[band].max()
    peaks = [i for i in band if S[i] > S[i - 1] and S[i] > S[i + 1] and S[i] > floor]
    if not peaks:
        return 0.0, 0.0
    angles, weights = [], []
    for i in peaks:
        a, b, c = S[i - 1], S[i], S[i + 1]
        f = freqs[i] + 0.5 * (a - c) / (a - 2 * b + c) * (freqs[1] - freqs[0])
        angles.append(2 * math.pi * ((12 * math.log2(f / 440.0)) % 1.0))
        weights.append(b)
    w, ang = np.array(weights), np.array(angles)
    s, c = float((w * np.sin(ang)).sum()), float((w * np.cos(ang)).sum())
    cents = (math.atan2(s, c) / (2 * math.pi) * 100 + 50) % 100 - 50
    return cents, math.hypot(s, c) / float(w.sum())
