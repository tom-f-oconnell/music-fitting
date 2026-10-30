"""Unpitched sounds: drum hits triggered by an onset roll, and a stationary noise floor.

Real recordings have drums, noise and wash that the pitched synth voices cannot
make; without somewhere else to go, the fitter explains them with spurious
notes. A :class:`NoiseVoice` gives them that place:

* ``n_kinds`` drum kinds, each a fixed kernel replayed at every hit (as a drum
  machine or sample does): white noise shaped by ``n_bands`` log-spaced band
  gains with an exponential decay, plus a pitch-swept decaying sine (kick).
  Hits come from a soft onset roll ``[kinds, frames]``, so the render is one FFT
  convolution per kind, linear in the roll like :class:`~musicfit.notes.PianoRoll`.
* A stationary noise floor: white noise through its own band gains.

The bands are deliberately coarse (16 over the whole spectrum), so harmonic
peaks stay cheaper to explain with notes than with noise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .synth import ParamSpec, _logit, _next_pow2

KERNEL_SPECS = {
    "noise_decay": ParamSpec(0.005, 1.0, 0.08, log=True),
    "sweep_start": ParamSpec(60.0, 400.0, 150.0, log=True),
    "sweep_end": ParamSpec(30.0, 150.0, 50.0, log=True),
    "sweep_time": ParamSpec(0.003, 0.1, 0.03, log=True),
    "tone_decay": ParamSpec(0.02, 0.8, 0.15, log=True),
    "tone_level": ParamSpec(0.0, 1.0, 0.0),
}

# initial kinds: (band-gain peak frequency in Hz, noise decay, tone level)
_KIND_PRESETS = [(80.0, 0.05, 0.8), (1500.0, 0.12, 0.1), (8000.0, 0.03, 0.0)]


def band_interp(gains: torch.Tensor, freqs: torch.Tensor, f_lo: float, f_hi: float) -> torch.Tensor:
    """Piecewise-linear interpolation (in log frequency) of per-band gains [..., B] at freqs [F] -> [..., F]."""
    B = gains.shape[-1]
    u = (torch.log(freqs.clamp(min=f_lo)) - math.log(f_lo)) / (math.log(f_hi) - math.log(f_lo)) * (B - 1)
    u = u.clamp(0, B - 1)
    i0 = u.floor().long().clamp(max=B - 2)
    w = u - i0
    return gains[..., i0] * (1 - w) + gains[..., i0 + 1] * w


class NoiseVoice(nn.Module):
    def __init__(self, n_frames: int, frame_hop_s: float, sr: int, n_kinds: int = 3, n_bands: int = 16,
                 kernel_s: float = 0.5, noise_floor: bool = True, init_logit: float = -5.0, init_noise: float = 0.5,
                 generator: torch.Generator | None = None, seed: int = 0):
        super().__init__()
        self.sr, self.frame_hop_s, self.noise_floor = sr, frame_hop_s, noise_floor
        self.f_lo, self.f_hi = 40.0, 0.5 * sr
        K = n_kinds
        noise = torch.randn(K, n_frames, generator=generator) * init_noise
        self.logits = nn.Parameter(torch.full((K, n_frames), init_logit) + noise)
        centers = torch.exp(torch.linspace(math.log(self.f_lo), math.log(self.f_hi), n_bands))
        gains, raw = [], {k: [] for k in KERNEL_SPECS}
        for k in range(K):
            peak, decay, tone = _KIND_PRESETS[k % len(_KIND_PRESETS)]
            # log band gains: a broad bump around the preset's frequency
            gains.append(-1.0 - 1.5 * (torch.log(centers / peak) / math.log(4.0)) ** 2)
            init = {"noise_decay": decay, "tone_level": tone}
            for name, spec in KERNEL_SPECS.items():
                raw[name].append(_logit(spec.to_unit(init.get(name, spec.default))))
        self.log_gains = nn.Parameter(torch.stack(gains))  # [K, B]
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.tensor(v)) for k, v in raw.items()})  # each [K]
        self.floor_log_gains = nn.Parameter(torch.full((n_bands,), -9.0))
        g = torch.Generator().manual_seed(seed)
        L = int(kernel_s * sr)
        self.register_buffer("kernel_noise", torch.randn(K, L, generator=g))
        self._floor_noise_seed = seed + 1
        self._floor_noise: torch.Tensor | None = None

    def strengths(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def values(self) -> dict[str, torch.Tensor]:
        return {k: spec.from_unit(torch.sigmoid(self.raw[k])) for k, spec in KERNEL_SPECS.items()}

    def kernels(self) -> torch.Tensor:
        """One waveform per kind, [K, L]."""
        p = self.values()
        K, L = self.kernel_noise.shape
        t = torch.arange(L, device=self.kernel_noise.device) / self.sr
        freqs = torch.fft.rfftfreq(L, 1.0 / self.sr, device=t.device)
        shape = band_interp(torch.exp(self.log_gains), freqs, self.f_lo, self.f_hi)  # [K, F]
        noise = torch.fft.irfft(torch.fft.rfft(self.kernel_noise) * shape, L)
        noise = noise * torch.exp(-t / p["noise_decay"][:, None])
        f = p["sweep_end"][:, None] + (p["sweep_start"] - p["sweep_end"])[:, None] * torch.exp(-t / p["sweep_time"][:, None])
        tone = torch.sin(2 * math.pi * torch.cumsum(f, dim=1) / self.sr) * torch.exp(-t / p["tone_decay"][:, None])
        return noise + p["tone_level"][:, None] * tone

    def _floor(self, n_samples: int, device) -> torch.Tensor:
        if self._floor_noise is None or self._floor_noise.shape[0] < n_samples or self._floor_noise.device != device:
            g = torch.Generator().manual_seed(self._floor_noise_seed)
            self._floor_noise = torch.randn(n_samples, generator=g).to(device)
        n_fft = _next_pow2(n_samples)
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / self.sr, device=device)
        shape = band_interp(torch.exp(self.floor_log_gains), freqs, self.f_lo, self.f_hi)
        return torch.fft.irfft(torch.fft.rfft(self._floor_noise[:n_samples], n_fft) * shape, n_fft)[:n_samples]

    def render(self, n_samples: int) -> torch.Tensor:
        v = self.strengths()
        K, M = v.shape
        hop = int(round(self.frame_hop_s * self.sr))
        impulses = torch.zeros(K, n_samples, device=v.device)
        idx = torch.arange(M, device=v.device) * hop
        keep = idx < n_samples
        impulses[:, idx[keep]] = v[:, keep]
        h = self.kernels()
        n_fft = _next_pow2(n_samples + h.shape[1])
        out = torch.fft.irfft(torch.fft.rfft(impulses, n_fft) * torch.fft.rfft(h, n_fft), n_fft)[:, :n_samples].sum(0)
        if self.noise_floor:
            out = out + self._floor(n_samples, v.device)
        return out

    @torch.no_grad()
    def hits(self, threshold: float = 0.3, merge_frames: int = 3) -> list[tuple[int, float, float]]:
        """Peak-picked hits as (kind, time in s, strength), like PianoRoll.decode."""
        v = self.strengths()
        pad = torch.nn.functional.pad(v[None], (merge_frames, merge_frames))[0]
        local_max = torch.nn.functional.max_pool1d(pad[None], 2 * merge_frames + 1, stride=1)[0]
        out = []
        for k in range(v.shape[0]):
            for m in torch.nonzero((v[k] >= local_max[k]) & (v[k] > 1e-3)).flatten().tolist():
                lo, hi = max(0, m - merge_frames), min(v.shape[1], m + merge_frames + 1)
                mass = float(v[k, lo:hi].sum())
                if mass >= threshold:
                    out.append((k, m * self.frame_hop_s, min(mass, 1.0)))
        return sorted(out, key=lambda h: h[1])
