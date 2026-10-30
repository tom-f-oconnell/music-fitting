"""A small differentiable subtractive synthesizer written in PyTorch.

Signal path for one voice (one "instrument"):

    band-limited oscillator (sine/saw/square/triangle mix, additive harmonics)
      -> resonant low-pass filter whose cutoff follows a filter ADSR
      -> amplitude ADSR * note velocity
      (+ enveloped white noise)
    sum over notes
      -> drive (tanh saturation) -> feedback delay -> noise reverb -> master gain

Every stage is a smooth (or piecewise smooth) function of its parameters, so the
whole thing can be fitted to audio by gradient descent.  The filter is applied
in the harmonic domain: each partial is scaled by the filter's magnitude
response at that partial's frequency, evaluated at control rate.  That is exact
for a static filter and a good approximation for a slowly moving cutoff, and it
avoids back-propagating through a sample-by-sample IIR recursion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

WAVEFORMS = ("sine", "saw", "square", "triangle")


@dataclass(frozen=True)
class ParamSpec:
    low: float
    high: float
    default: float
    log: bool = False

    def to_unit(self, value: float) -> float:
        if self.log:
            return (math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (value - self.low) / (self.high - self.low)

    def from_unit(self, u: torch.Tensor) -> torch.Tensor:
        if self.log:
            return torch.exp(math.log(self.low) + (math.log(self.high) - math.log(self.low)) * u)
        return self.low + (self.high - self.low) * u


# Global (per-voice) synth parameters.  Times are in seconds, cutoff in Hz,
# filter envelope amount in octaves.
PARAM_SPECS: dict[str, ParamSpec] = {
    "amp_attack": ParamSpec(0.001, 1.0, 0.005, log=True),
    "amp_decay": ParamSpec(0.01, 3.0, 0.3, log=True),
    "amp_sustain": ParamSpec(0.0, 1.0, 0.6),
    "amp_release": ParamSpec(0.005, 2.0, 0.08, log=True),
    "filt_attack": ParamSpec(0.001, 1.0, 0.005, log=True),
    "filt_decay": ParamSpec(0.01, 3.0, 0.3, log=True),
    "filt_sustain": ParamSpec(0.0, 1.0, 0.3),
    "filt_release": ParamSpec(0.005, 2.0, 0.1, log=True),
    "cutoff": ParamSpec(40.0, 12000.0, 1500.0, log=True),
    "resonance": ParamSpec(0.5, 10.0, 1.0, log=True),
    "filt_env_amount": ParamSpec(0.0, 6.0, 1.5),
    "noise_level": ParamSpec(0.0, 0.5, 0.01),
    "drive": ParamSpec(0.1, 8.0, 0.5, log=True),
    "delay_time": ParamSpec(0.05, 1.0, 0.25, log=True),
    "delay_feedback": ParamSpec(0.0, 0.9, 0.2),
    "delay_mix": ParamSpec(0.0, 1.0, 0.05),
    "reverb_rt60": ParamSpec(0.1, 4.0, 0.8, log=True),
    "reverb_mix": ParamSpec(0.0, 1.0, 0.05),
    "gain": ParamSpec(0.01, 10.0, 0.5, log=True),
}


def _logit(u: float, eps: float = 1e-4) -> float:
    u = min(max(u, eps), 1 - eps)
    return math.log(u / (1 - u))


class SynthParams(nn.Module):
    """Unconstrained trainable parameters, mapped into physical ranges by sigmoids."""

    def __init__(self, values: dict[str, float] | None = None, waveform: dict[str, float] | None = None):
        super().__init__()
        values = values or {}
        unknown = set(values) - set(PARAM_SPECS)
        if unknown:
            raise KeyError(f"unknown synth parameters: {sorted(unknown)}")
        self.raw = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(_logit(spec.to_unit(values.get(name, spec.default)))))
                for name, spec in PARAM_SPECS.items()
            }
        )
        # Waveform mixture weights (softmax over WAVEFORMS).  Default: mostly saw.
        weights = waveform or {"sine": 0.1, "saw": 0.6, "square": 0.2, "triangle": 0.1}
        w = torch.tensor([max(weights.get(n, 0.0), 1e-3) for n in WAVEFORMS])
        self.wave_logits = nn.Parameter(torch.log(w / w.sum()))

    def values(self) -> dict[str, torch.Tensor]:
        out = {name: spec.from_unit(torch.sigmoid(self.raw[name])) for name, spec in PARAM_SPECS.items()}
        out["waveform"] = torch.softmax(self.wave_logits, dim=0)
        return out

    def describe(self) -> dict[str, object]:
        with torch.no_grad():
            v = self.values()
            d: dict[str, object] = {k: round(float(x), 4) for k, x in v.items() if k != "waveform"}
            d["waveform"] = {n: round(float(w), 3) for n, w in zip(WAVEFORMS, v["waveform"])}
        return d

    @classmethod
    def random(cls, generator: torch.Generator | None = None) -> "SynthParams":
        """Sample a random (but musically plausible) patch, e.g. for synthetic targets."""
        g = generator
        u = lambda lo, hi: lo + (hi - lo) * float(torch.rand((), generator=g))  # noqa: E731
        values = {
            "amp_attack": math.exp(u(math.log(0.002), math.log(0.05))),
            "amp_decay": math.exp(u(math.log(0.05), math.log(1.0))),
            "amp_sustain": u(0.2, 0.9),
            "amp_release": math.exp(u(math.log(0.02), math.log(0.4))),
            "filt_attack": math.exp(u(math.log(0.002), math.log(0.05))),
            "filt_decay": math.exp(u(math.log(0.05), math.log(0.8))),
            "filt_sustain": u(0.0, 0.6),
            "filt_release": math.exp(u(math.log(0.02), math.log(0.4))),
            "cutoff": math.exp(u(math.log(200), math.log(3000))),
            "resonance": math.exp(u(math.log(0.7), math.log(6.0))),
            "filt_env_amount": u(0.5, 4.0),
            "noise_level": u(0.0, 0.05),
            "drive": math.exp(u(math.log(0.2), math.log(3.0))),
            "delay_time": math.exp(u(math.log(0.1), math.log(0.5))),
            "delay_feedback": u(0.0, 0.6),
            "delay_mix": u(0.0, 0.4),
            "reverb_rt60": math.exp(u(math.log(0.2), math.log(2.0))),
            "reverb_mix": u(0.0, 0.3),
            "gain": 0.5,
        }
        w = torch.rand(len(WAVEFORMS), generator=g) ** 3
        return cls(values, waveform={n: float(x) for n, x in zip(WAVEFORMS, w)})


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def waveform_harmonics(weights: torch.Tensor, n_harmonics: int) -> torch.Tensor:
    """Fourier-series amplitudes (signed) of the waveform mix, shape [K]."""
    k = torch.arange(1, n_harmonics + 1, dtype=weights.dtype, device=weights.device)
    odd = (k % 2 == 1).to(weights.dtype)
    sine = (k == 1).to(weights.dtype)
    saw = (2 / math.pi) * torch.where(k % 2 == 1, 1.0, -1.0) / k
    square = (4 / math.pi) * odd / k
    tri_sign = torch.where(((k - 1) / 2) % 2 == 0, 1.0, -1.0)
    triangle = (8 / math.pi**2) * odd * tri_sign / k**2
    basis = torch.stack([sine, saw, square, triangle])  # [4, K]
    return weights @ basis


def lowpass_magnitude(freq: torch.Tensor, cutoff: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Magnitude response of two cascaded resonant 2-pole low-pass sections (24 dB/oct)."""
    x = freq / cutoff
    h2 = 1.0 / torch.sqrt((1 - x**2) ** 2 + (x / q) ** 2 + 1e-12)
    return h2**2


def adsr(t: torch.Tensor, duration: torch.Tensor, a, d, s, r) -> torch.Tensor:
    """Closed-form ADSR envelope.

    t: time since note-on (any shape, broadcastable with duration), seconds.
    duration: gate length (time from note-on to note-off).
    Linear attack, exponential decay towards sustain, exponential release from
    whatever level the envelope had at note-off.  Zero before the onset.
    Piecewise smooth in t, onset and duration, so all three get gradients.
    """
    tau = torch.minimum(torch.clamp(t, min=0.0), duration)  # time inside the gate
    attack = tau / a
    decayed = s + (1 - s) * torch.exp(-torch.clamp(tau - a, min=0.0) / d)
    gate_level = torch.where(tau < a, attack, decayed)
    release = torch.exp(-torch.clamp(t - duration, min=0.0) / r)
    return torch.where(t >= 0, gate_level * release, torch.zeros_like(t))


def soft_drive(x: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    return torch.tanh(drive * x) / torch.tanh(drive)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def feedback_delay(x: torch.Tensor, sr: int, time: torch.Tensor, feedback: torch.Tensor, mix: torch.Tensor):
    """Feedback delay line, evaluated exactly in the frequency domain.

    H(w) = 1 + mix * g e^{-iwD} / (1 - g e^{-iwD}); D is continuous (fractional
    delay), so the delay time itself is differentiable.
    """
    n = x.shape[-1]
    n_fft = _next_pow2(2 * n)
    X = torch.fft.rfft(x, n_fft)
    w = 2 * math.pi * torch.fft.rfftfreq(n_fft, d=1.0, device=x.device)  # rad / sample
    z = feedback * torch.exp(-1j * w * (time * sr))
    # first echo at full level (g^0), then decaying by g each repeat
    H = 1 + mix * torch.exp(-1j * w * (time * sr)) / (1 - z)
    return torch.fft.irfft(X * H, n_fft)[..., :n]


class NoiseReverb(nn.Module):
    """Exponentially decaying noise impulse response with learnable RT60 and mix."""

    def __init__(self, sr: int, length_s: float = 2.0, seed: int = 1234):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.sr = sr
        self.register_buffer("noise", torch.randn(int(length_s * sr), generator=g))

    def forward(self, x: torch.Tensor, rt60: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        t = torch.arange(self.noise.shape[0], device=x.device) / self.sr
        ir = self.noise * torch.exp(-6.9078 * t / rt60)
        ir = ir / (ir.norm() + 1e-8)
        n_fft = _next_pow2(n + ir.shape[0])
        wet = torch.fft.irfft(torch.fft.rfft(x, n_fft) * torch.fft.rfft(ir, n_fft), n_fft)[..., :n]
        return x + mix * wet


# ---------------------------------------------------------------------------
# Synth voice
# ---------------------------------------------------------------------------


def midi_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 440.0 * 2 ** ((m - 69.0) / 12.0)


class Synth(nn.Module):
    """One synth voice: trainable patch + fixed rendering settings."""

    def __init__(
        self,
        sr: int = 16000,
        n_harmonics: int = 32,
        control_hop: int = 32,
        params: SynthParams | None = None,
        noise_seed: int = 0,
    ):
        super().__init__()
        self.sr = sr
        self.n_harmonics = n_harmonics
        self.control_hop = control_hop
        self.params = params or SynthParams()
        self.reverb = NoiseReverb(sr)
        self.noise_seed = noise_seed
        self._noise: torch.Tensor | None = None
        self._sin_cache: dict = {}

    def _white_noise(self, n: int, device) -> torch.Tensor:
        if self._noise is None or self._noise.shape[0] < n:
            g = torch.Generator().manual_seed(self.noise_seed)
            self._noise = torch.randn(n, generator=g) * 0.3
        return self._noise[:n].to(device)

    def render_partials(
        self,
        p: dict[str, torch.Tensor],
        f0: torch.Tensor,
        phase_ref: torch.Tensor | None,
        amp_env: torch.Tensor,
        filt_env: torch.Tensor,
        n_samples: int,
        cache_key=None,
    ) -> torch.Tensor:
        """Sum of filtered, enveloped harmonic oscillators.

        f0: [N] fundamental frequencies (Hz).
        phase_ref: [N] times (s) at which each oscillator starts at phase 0, or
            None for phase reference 0 (lets the sinusoid table be cached).
        amp_env: [N, T] amplitude envelope (already includes velocity).
        filt_env: [N, T] filter envelope in [0, 1].
        """
        sr, K, hop = self.sr, self.n_harmonics, self.control_hop
        device = amp_env.device
        k = torch.arange(1, K + 1, device=device, dtype=torch.float32)
        freqs = f0[:, None] * k  # [N, K]
        alias_mask = (freqs < 0.5 * sr * 0.95).to(freqs.dtype)
        wave = waveform_harmonics(p["waveform"], K)  # [K]

        # filter gains at control rate
        fenv_ctrl = filt_env[:, ::hop]  # [N, F]
        cutoff = p["cutoff"] * torch.pow(2.0, p["filt_env_amount"] * fenv_ctrl)
        cutoff = torch.clamp(cutoff, 20.0, 0.5 * sr)
        H = lowpass_magnitude(freqs[:, :, None], cutoff[:, None, :], p["resonance"])  # [N, K, F]
        G = wave[None, :, None] * alias_mask[:, :, None] * H
        N = G.shape[0]
        G = F.interpolate(G.reshape(1, N * K, -1), size=n_samples, mode="linear", align_corners=False)
        G = G.reshape(N, K, n_samples)

        sines = self._sinusoids(freqs, phase_ref, n_samples, cache_key)
        osc = (G * sines).sum(dim=1)  # [N, T]
        return (amp_env * osc).sum(dim=0)

    def _sinusoids(self, freqs, phase_ref, n_samples, cache_key):
        if phase_ref is None and cache_key is not None:
            key = (cache_key, n_samples, freqs.shape)
            if key not in self._sin_cache:
                self._sin_cache.clear()
                self._sin_cache[key] = self._compute_sin(freqs.detach(), None, n_samples)
            return self._sin_cache[key]
        return self._compute_sin(freqs, phase_ref, n_samples)

    def _compute_sin(self, freqs, phase_ref, n_samples):
        # float64 phase: 2*pi*f*t gets large for long clips.
        t = torch.arange(n_samples, device=freqs.device, dtype=torch.float64) / self.sr
        f = freqs.to(torch.float64)[:, :, None]
        if phase_ref is None:
            phase = 2 * math.pi * f * t
        else:
            phase = 2 * math.pi * f * (t - phase_ref.to(torch.float64)[:, None, None])
        phase = torch.remainder(phase, 2 * math.pi)
        return torch.sin(phase).to(torch.float32)

    def effects(self, p: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        x = soft_drive(x, p["drive"])
        x = feedback_delay(x, self.sr, p["delay_time"], p["delay_feedback"], p["delay_mix"])
        x = self.reverb(x, p["reverb_rt60"], p["reverb_mix"])
        return p["gain"] * x

    # -- note-list rendering ------------------------------------------------

    def render_notes(
        self,
        pitch: torch.Tensor,
        onset: torch.Tensor,
        duration: torch.Tensor,
        velocity: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Render a list of notes. pitch in (fractional) MIDI numbers, times in seconds."""
        p = self.params.values()
        if pitch.numel() == 0:
            return self.effects(p, torch.zeros(n_samples, device=p["gain"].device))
        t = torch.arange(n_samples, device=pitch.device, dtype=torch.float32) / self.sr
        t_rel = t[None, :] - onset[:, None]
        dur = duration[:, None]
        amp = adsr(t_rel, dur, p["amp_attack"], p["amp_decay"], p["amp_sustain"], p["amp_release"])
        amp = amp * velocity[:, None]
        fenv = adsr(t_rel, dur, p["filt_attack"], p["filt_decay"], p["filt_sustain"], p["filt_release"])
        fenv = fenv * velocity[:, None]
        x = self.render_partials(p, midi_to_hz(pitch), onset, amp, fenv, n_samples)
        x = x + p["noise_level"] * amp.sum(0) * self._white_noise(n_samples, x.device)
        return self.effects(p, x)

    # -- piano-roll rendering ----------------------------------------------

    def render_roll(
        self,
        onset_roll: torch.Tensor,
        pitches: torch.Tensor,
        frame_hop_s: float,
        duration: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Render a soft onset piano roll.

        onset_roll: [P, M] non-negative note-on strengths (velocity) per pitch and frame.
        Every cell triggers a note of the given (global) duration; the envelopes are a
        linear function of the roll, so they are computed as one FFT convolution of
        the roll (as an impulse train) with the ADSR kernel.
        """
        p = self.params.values()
        sr = self.sr
        P, M = onset_roll.shape
        hop = int(round(frame_hop_s * sr))
        impulses = torch.zeros(P, n_samples, device=onset_roll.device)
        idx = torch.arange(M, device=onset_roll.device) * hop
        keep = idx < n_samples
        impulses[:, idx[keep]] = onset_roll[:, keep]

        t = torch.arange(n_samples, device=onset_roll.device, dtype=torch.float32) / sr
        k_amp = adsr(t, duration, p["amp_attack"], p["amp_decay"], p["amp_sustain"], p["amp_release"])
        k_filt = adsr(t, duration, p["filt_attack"], p["filt_decay"], p["filt_sustain"], p["filt_release"])
        n_fft = _next_pow2(2 * n_samples)
        I = torch.fft.rfft(impulses, n_fft)
        amp = torch.fft.irfft(I * torch.fft.rfft(k_amp, n_fft), n_fft)[:, :n_samples]
        fenv = torch.fft.irfft(I * torch.fft.rfft(k_filt, n_fft), n_fft)[:, :n_samples]
        fenv = torch.clamp(fenv, 0.0, 1.0)

        key = ("roll", tuple(pitches.tolist()))
        x = self.render_partials(p, midi_to_hz(pitches), None, amp, fenv, n_samples, cache_key=key)
        x = x + p["noise_level"] * amp.sum(0) * self._white_noise(n_samples, x.device)
        return self.effects(p, x)
