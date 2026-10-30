"""Fast (no-gradient) rendering of a patch + note list, used for live tweaking."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch

from ..loss import MultiResolutionSTFTLoss
from ..notes import Note
from ..synth import PARAM_SPECS, WAVEFORMS, Synth, SynthParams

_loss_fn = MultiResolutionSTFTLoss()


def patch_from_dict(d: dict) -> SynthParams:
    """Build SynthParams from a patch dict as produced by SynthParams.describe()."""
    values = {}
    for name, spec in PARAM_SPECS.items():
        if name in d:
            values[name] = float(min(max(float(d[name]), spec.low), spec.high))
    wave = d.get("waveform") or {}
    return SynthParams(values, {n: float(wave.get(n, 0.0)) for n in WAVEFORMS} if wave else None)


def notes_from_dicts(items: list[dict]) -> list[Note]:
    return [Note(float(n["pitch"]), float(n["onset"]), float(n["duration"]), float(n["velocity"]),
                 int(n.get("voice", 0))) for n in items]


@torch.no_grad()
def render(patches: list[dict], notes: list[Note], n_samples: int, sr: int = 16000, n_harmonics: int = 32,
           muted_voices: set[int] = frozenset()) -> torch.Tensor:
    out = torch.zeros(n_samples)
    for v, patch in enumerate(patches):
        if v in muted_voices:
            continue
        synth = Synth(sr, n_harmonics, params=patch_from_dict(patch), noise_seed=v)
        vn = [n for n in notes if n.voice == v]
        if not vn:
            continue
        out = out + synth.render_notes(
            torch.tensor([n.pitch for n in vn]), torch.tensor([n.onset for n in vn]),
            torch.tensor([n.duration for n in vn]), torch.tensor([n.velocity for n in vn]), n_samples)
    return out


@torch.no_grad()
def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(_loss_fn(pred, target))


def wav_bytes(audio: torch.Tensor, sr: int) -> bytes:
    buf = io.BytesIO()
    x = np.clip(audio.detach().cpu().float().numpy(), -1, 1)
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
