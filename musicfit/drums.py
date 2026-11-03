"""Unpitched sounds: drum hits triggered by an onset roll, and a stationary noise floor.

Two drum voices share the same interface. :class:`KitVoice` (the default) plays drum machine
kits' drums, each a named General MIDI drum; :class:`NoiseVoice` makes its own anonymous kinds.

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

from .drumkit import GM_NOTE, GROUP, Kit
from .synth import ParamSpec, _logit, _next_pow2

KERNEL_SPECS = {
    # kept short and low: longer or higher tones and noise bursts could stand in for notes
    "noise_decay": ParamSpec(0.005, 0.5, 0.08, log=True),
    "sweep_start": ParamSpec(60.0, 400.0, 150.0, log=True),
    "sweep_end": ParamSpec(30.0, 100.0, 50.0, log=True),
    "sweep_time": ParamSpec(0.003, 0.1, 0.03, log=True),
    "tone_decay": ParamSpec(0.02, 0.3, 0.15, log=True),
    "tone_level": ParamSpec(0.0, 1.0, 0.0),
    "level": ParamSpec(0.01, 2.0, 0.3, log=True),  # peak of a full-strength hit
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


def _render_roll(v: torch.Tensor, kernels: torch.Tensor, frame_hop_s: float, sr: int, n_samples: int) -> torch.Tensor:
    """Each row of an onset roll [K, frames] played through its kernel [K, L], summed."""
    K, M = v.shape
    hop = int(round(frame_hop_s * sr))
    impulses = torch.zeros(K, n_samples, device=v.device)
    idx = torch.arange(M, device=v.device) * hop
    keep = idx < n_samples
    impulses[:, idx[keep]] = v[:, keep]
    n_fft = _next_pow2(n_samples + kernels.shape[1])
    return torch.fft.irfft(torch.fft.rfft(impulses, n_fft) * torch.fft.rfft(kernels, n_fft), n_fft)[:, :n_samples].sum(0)


def _decode_hits(v: torch.Tensor, frame_hop_s: float, threshold: float, merge_frames: int) -> list[tuple[int, float, float]]:
    pad = torch.nn.functional.pad(v[None], (merge_frames, merge_frames))[0]
    local_max = torch.nn.functional.max_pool1d(pad[None], 2 * merge_frames + 1, stride=1)[0]
    out = []
    for k in range(v.shape[0]):
        for m in torch.nonzero((v[k] >= local_max[k]) & (v[k] > 1e-3)).flatten().tolist():
            lo, hi = max(0, m - merge_frames), min(v.shape[1], m + merge_frames + 1)
            mass = float(v[k, lo:hi].sum())
            if mass >= threshold:
                out.append((k, m * frame_hop_s, min(mass, 1.0)))
    return sorted(out, key=lambda h: h[1])


class _DrumVoice(nn.Module):
    """What both drum voices share: the roll, the noise floor, playing hits, and saving."""

    names: list[str]  # one per kind
    gm_notes: list[int]  # General MIDI drum note of each kind, for MIDI files
    hit_threshold: float = 0.3  # what hits() counts as a hit, unless told otherwise

    def _init_shared(self, n_kinds: int, n_frames: int, frame_hop_s: float, sr: int, noise_floor: bool, init_logit: float,
                     init_noise: float, generator, seed: int, n_bands: int = 16) -> None:
        self.sr, self.frame_hop_s, self.noise_floor = sr, frame_hop_s, noise_floor
        self.f_lo, self.f_hi = 40.0, 0.5 * sr
        noise = torch.randn(n_kinds, n_frames, generator=generator) * init_noise
        self.logits = nn.Parameter(torch.full((n_kinds, n_frames), init_logit) + noise)
        self.floor_log_gains = nn.Parameter(torch.full((n_bands,), -9.0))
        self._floor_noise_seed = seed + 1
        self._floor_noise: torch.Tensor | None = None

    def strengths(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

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
        out = _render_roll(v, self.kernels(), self.frame_hop_s, self.sr, n_samples)
        if self.noise_floor:
            out = out + self._floor(n_samples, v.device)
        return out

    @torch.no_grad()
    def hits(self, threshold: float | None = None, merge_frames: int = 3) -> list[tuple[int, float, float]]:
        """Peak-picked hits as (kind, time in s, strength), like PianoRoll.decode."""
        threshold = self.hit_threshold if threshold is None else threshold
        return _decode_hits(self.strengths(), self.frame_hop_s, threshold, merge_frames)

    @torch.no_grad()
    def render_hits(self, hits: list[tuple[int, float, float]], n_samples: int, floor: bool = True) -> torch.Tensor:
        """Play a list of hits, e.g. an edited copy of hits(); the noise floor too unless floor is False."""
        h = self.kernels()
        impulses = torch.zeros(h.shape[0], n_samples, device=h.device)
        for kind, t, strength in hits:
            i = int(round(t * self.sr))
            if 0 <= i < n_samples and 0 <= kind < h.shape[0]:
                impulses[kind, i] += strength
        n_fft = _next_pow2(n_samples + h.shape[1])
        out = torch.fft.irfft(torch.fft.rfft(impulses, n_fft) * torch.fft.rfft(h, n_fft), n_fft)[:, :n_samples].sum(0)
        if floor and self.noise_floor:
            out = out + self._floor(n_samples, h.device)
        return out

    def save(self, path: str) -> None:
        """Everything needed to play hits again later, without the code that fitted it."""
        torch.save({"kind": type(self).__name__, "args": self.init_args, "state": self.state_dict(),
                    "hit_threshold": self.hit_threshold}, path)


def load_drum_voice(path: str) -> "_DrumVoice":
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cls = {"KitVoice": KitVoice, "NoiseVoice": NoiseVoice}[saved["kind"]]
    voice = cls.from_args(saved["args"])
    voice.load_state_dict(saved["state"])
    voice.hit_threshold = saved.get("hit_threshold", 0.3)
    return voice


class NoiseVoice(_DrumVoice):
    """Drum kinds of its own making: band-shaped noise and a swept sine (the drum voice before kits)."""

    def __init__(self, n_frames: int, frame_hop_s: float, sr: int, n_kinds: int = 3, n_bands: int = 16,
                 kernel_s: float = 0.5, noise_floor: bool = True, init_logit: float = -5.0, init_noise: float = 0.5,
                 generator: torch.Generator | None = None, seed: int = 0):
        super().__init__()
        self.init_args = {"n_frames": n_frames, "frame_hop_s": frame_hop_s, "sr": sr, "n_kinds": n_kinds, "n_bands": n_bands,
                          "kernel_s": kernel_s, "noise_floor": noise_floor}
        self._init_shared(n_kinds, n_frames, frame_hop_s, sr, noise_floor, init_logit, init_noise, generator, seed, n_bands)
        K = n_kinds
        # anonymous kinds: named after where they start, low to high, which is only a guess at what they become
        self.names = [f"drum {k + 1}" for k in range(K)]
        self.gm_notes = [(GM_NOTE["kick"], GM_NOTE["snare"], GM_NOTE["closed hat"])[k % 3] for k in range(K)]
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
        g = torch.Generator().manual_seed(seed)
        L = int(kernel_s * sr)
        self.register_buffer("kernel_noise", torch.randn(K, L, generator=g))

    @classmethod
    def from_args(cls, args: dict) -> "NoiseVoice":
        return cls(**args)

    def describe(self) -> list[dict]:
        return [{"name": n, "gm": g, "kit": ""} for n, g in zip(self.names, self.gm_notes)]

    def values(self) -> dict[str, torch.Tensor]:
        return {k: spec.from_unit(torch.sigmoid(self.raw[k])) for k, spec in KERNEL_SPECS.items()}

    def kernels(self) -> torch.Tensor:
        """One waveform per kind, [K, L], with peak `level`."""
        p = self.values()
        K, L = self.kernel_noise.shape
        t = torch.arange(L, device=self.kernel_noise.device) / self.sr
        freqs = torch.fft.rfftfreq(L, 1.0 / self.sr, device=t.device)
        shape = band_interp(torch.exp(self.log_gains), freqs, self.f_lo, self.f_hi)  # [K, F]
        noise = torch.fft.irfft(torch.fft.rfft(self.kernel_noise) * shape, L)
        noise = noise * torch.exp(-t / p["noise_decay"][:, None])
        f = p["sweep_end"][:, None] + (p["sweep_start"] - p["sweep_end"])[:, None] * torch.exp(-t / p["sweep_time"][:, None])
        tone = torch.sin(2 * math.pi * torch.cumsum(f, dim=1) / self.sr) * torch.exp(-t / p["tone_decay"][:, None])
        h = noise + p["tone_level"][:, None] * tone
        # peak-normalised, so the band gains only shape the spectrum: loudness is hit strength x level,
        # and the L1 penalty on hit strength cannot be dodged by turning the gains up
        return h / (h.abs().amax(dim=1, keepdim=True) + 1e-8) * p["level"][:, None]


KIT_SPECS = {
    "level": ParamSpec(0.01, 2.0, 0.3, log=True),  # peak of a full-strength hit
    "tune": ParamSpec(-12.0, 12.0, 0.0),  # semitones, by playing the sample faster or slower
    "decay": ParamSpec(-40.0, 2.0, 0.0),  # added to the tail's decay rate, 1/s: below 0 shortens it
}


class KitVoice(_DrumVoice):
    """Drum machine kits' drums (:mod:`musicfit.drumkit`), one roll row per drum.

    While the roll is fitted, each drum is a soft mix of the kits' versions of it, so that gradients
    can choose; :meth:`commit` then keeps the kit that fits best, for the rest of the fit. Each drum
    also has a level, a tuning and a change to its decay, like a drum machine's knobs.
    """

    def __init__(self, n_frames: int, frame_hop_s: float, sr: int, kit: Kit, noise_floor: bool = False,
                 init_logit: float = -5.0, init_noise: float = 0.5, generator: torch.Generator | None = None, seed: int = 0):
        super().__init__()
        self.init_args = {"n_frames": n_frames, "frame_hop_s": frame_hop_s, "sr": sr, "noise_floor": noise_floor,
                          "names": list(kit.names), "kit_names": list(kit.kit_names), "shape": list(kit.samples.shape)}
        self._init_shared(len(kit.names), n_frames, frame_hop_s, sr, noise_floor, init_logit, init_noise, generator, seed)
        self.names, self.kit_names = list(kit.names), list(kit.kit_names)
        self.gm_notes = [GM_NOTE[n] for n in self.names]
        self.register_buffer("samples", kit.samples.clone())  # [drums, kits, L]
        self.register_buffer("available", kit.available.clone())  # [drums, kits]
        self.kit_logits = nn.Parameter(torch.zeros(kit.available.shape))
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.full((len(kit.names),), _logit(spec.to_unit(spec.default))))
                                     for k, spec in KIT_SPECS.items()})

    @classmethod
    def from_args(cls, args: dict) -> "KitVoice":
        C, K, L = args["shape"]
        kit = Kit(args["names"], args["kit_names"], torch.zeros(C, K, L), torch.ones(C, K, dtype=torch.bool))
        return cls(args["n_frames"], args["frame_hop_s"], args["sr"], kit, args["noise_floor"])

    def values(self) -> dict[str, torch.Tensor]:
        return {k: spec.from_unit(torch.sigmoid(self.raw[k])) for k, spec in KIT_SPECS.items()}

    def kit_weights(self) -> torch.Tensor:
        return torch.softmax(self.kit_logits.masked_fill(~self.available, -1e4), dim=1)

    @torch.no_grad()
    def commit(self) -> None:
        """Keep each drum's best-fitting kit and stop choosing."""
        best = self.kit_weights().argmax(dim=1, keepdim=True)
        self.kit_logits.copy_(torch.full_like(self.kit_logits, -1e4).scatter(1, best, 0.0))
        self.kit_logits.requires_grad_(False)

    def kernels(self) -> torch.Tensor:
        p = self.values()
        C, K, L = self.samples.shape
        n = torch.arange(L, device=self.samples.device, dtype=torch.float32)
        # retuning by reading the sample faster or slower; positions stay continuous, so tune gets a gradient
        pos = (n[None, :] * torch.pow(2.0, p["tune"] / 12.0)[:, None]).clamp(max=L - 1.001)  # [C, L]
        i0 = pos.floor().long()
        frac = (pos - i0)[:, None, :]
        idx = i0[:, None, :].expand(C, K, L)
        warped = self.samples.gather(2, idx) * (1 - frac) + self.samples.gather(2, (idx + 1).clamp(max=L - 1)) * frac
        warped = warped * (n[None, None, :] < (L - 1) / torch.pow(2.0, p["tune"] / 12.0)[:, None, None])
        h = (self.kit_weights()[:, :, None] * warped).sum(1)  # [C, L]
        t = n / self.sr
        return h * torch.exp(p["decay"][:, None] * t[None, :]) * p["level"][:, None]

    def hits(self, threshold: float | None = None, merge_frames: int = 3, group_s: float = 0.05) -> list[tuple[int, float, float]]:
        """Peak-picked hits, keeping only the strongest of a group's drums within ``group_s``.

        Synthetic drums only roughly match real ones, so the fit often builds one real snare out of
        its snare, clap and rim at once; as hits, that's one snare (design.md).
        """
        kept = []
        groups = [GROUP[n] for n in self.names]
        for k, t, v in sorted(super().hits(threshold, merge_frames), key=lambda h: -h[2]):
            if not any(groups[j] == groups[k] and abs(t - u) < group_s for j, u, _ in kept):
                kept.append((k, t, v))
        return sorted(kept, key=lambda h: h[1])

    @torch.no_grad()
    def describe(self) -> list[dict]:
        p, w = self.values(), self.kit_weights()
        return [{"name": name, "gm": gm, "kit": self.kit_names[int(w[d].argmax())], "kit_weight": round(float(w[d].max()), 3),
                 "level": round(float(p["level"][d]), 4), "tune": round(float(p["tune"][d]), 3), "decay": round(float(p["decay"][d]), 3)}
                for d, (name, gm) in enumerate(zip(self.names, self.gm_notes))]
