"""Optional pretrained helpers: Demucs source separation and basic-pitch transcription.

Install with ``pip install -r requirements-tools.txt`` and then
``pip install --no-deps basic-pitch``: basic-pitch runs on ONNX, but its package
metadata asks for an old TensorFlow that doesn't exist for current Pythons.
Neither is imported until it is used, so the rest of musicfit works without them.
Demucs downloads its weights on first use.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
from dataclasses import replace
from functools import lru_cache
from typing import Callable

import torch

from .fit import FitConfig, FitResult, fit, resolve_device
from .loss import MultiResolutionSTFTLoss
from .notes import Note

PITCHED_STEMS = ("bass", "other", "vocals")


def available_tools() -> dict[str, bool]:
    """Which of the optional tools can be imported, by name."""
    return {"demucs": importlib.util.find_spec("demucs") is not None,
            "basic_pitch": all(importlib.util.find_spec(m) is not None for m in ("basic_pitch", "onnxruntime"))}


def _missing(tool: str, error: ImportError) -> ImportError:
    return ImportError(f"{tool} is not installed ({error}); see requirements-tools.txt")


@lru_cache(maxsize=1)
def _demucs_model(name: str):
    try:
        from demucs.pretrained import get_model
    except ImportError as e:
        raise _missing("Demucs", e) from e
    return get_model(name).eval()


@torch.no_grad()
def separate(audio: torch.Tensor, sr: int, model: str = "htdemucs", device: str = "auto") -> dict[str, torch.Tensor]:
    """Split mono audio into stems with Demucs.

    Args:
        audio: [T] mono waveform at rate ``sr``.
        sr: sample rate of ``audio``; the stems come back at the same rate.
        model: Demucs model name.
        device: "auto", "cpu", "cuda", ...

    Returns:
        {stem name: [T] waveform on the CPU}: drums, bass, other and vocals for the
        default model. Their sum is approximately ``audio``.
    """
    from julius import resample_frac

    net = _demucs_model(model)
    x = resample_frac(audio.float().cpu(), sr, net.samplerate)
    mean, std = x.mean(), x.std() + 1e-8
    # Demucs is trained on normalised stereo; a clip shorter than its segment gets padded by apply_model
    stereo = ((x - mean) / std).expand(net.audio_channels, -1)
    from demucs.apply import apply_model

    out = apply_model(net, stereo[None], device=resolve_device(device), progress=False)[0].cpu()
    stems = {}
    for name, source in zip(net.sources, out):
        y = resample_frac(source.mean(0) * std + mean / len(net.sources), net.samplerate, sr)
        stems[name] = torch.nn.functional.pad(y, (0, max(0, audio.numel() - y.numel())))[: audio.numel()]
    return stems


def transcribe(
    audio: torch.Tensor,
    sr: int,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_ms: float = 58.0,
    voice: int = 0,
) -> list[Note]:
    """Transcribe audio into notes with basic-pitch (Bittner et al., ICASSP 2022).

    Args:
        audio: [T] mono waveform at rate ``sr``.
        sr: sample rate of ``audio``.
        onset_threshold: posterior needed to start a note; lower finds more notes.
        frame_threshold: posterior needed to continue one.
        min_note_ms: shortest note to keep.
        voice: voice index to give the notes.

    Returns:
        Notes sorted by onset, with integer pitches and velocity from the model's amplitude.
    """
    try:
        import soundfile as sf
        import basic_pitch
        from basic_pitch.inference import predict
    except ImportError as e:
        raise _missing("basic-pitch", e) from e
    logging.getLogger().setLevel(max(logging.getLogger().level, logging.ERROR))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.wav")
        sf.write(path, audio.detach().cpu().float().numpy(), sr, subtype="FLOAT")
        _, _, events = predict(path, basic_pitch.ICASSP_2022_MODEL_PATH, onset_threshold=onset_threshold,
                               frame_threshold=frame_threshold, minimum_note_length=min_note_ms)
    notes = [Note(float(p), float(on), float(off - on), min(max(float(a), 0.02), 1.0), voice)
             for on, off, p, a, _ in events]
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def fit_with_tools(
    target: torch.Tensor,
    config: FitConfig | None = None,
    use_demucs: bool = False,
    use_basic_pitch: bool = False,
    verbose: bool = True,
    progress: Callable[[dict], None] | None = None,
) -> FitResult:
    """Run :func:`musicfit.fit.fit` with Demucs and/or basic-pitch doing part of the work.

    With ``use_demucs`` the notes are fitted to the pitched stems only and the drum voice
    to the drum stem only, so neither can explain the other's sound. With
    ``use_basic_pitch`` the transcription replaces stage 1 and all its notes in the fit's
    pitch range are kept: on mixes the fit's own ways of dropping notes cost far more
    recall than they win in precision (design.md).

    Args:
        target: [T] mono target waveform.
        config: fit configuration; its drum settings apply to the drum stem under Demucs.
        use_demucs: separate first.
        use_basic_pitch: start from transcribed notes.
        verbose: print progress.
        progress: as for :func:`musicfit.fit.fit`, called during the note fit only.

    Returns:
        A FitResult whose audio is the whole fit, and whose final history entry has the
        loss against the full target.
    """
    cfg = config or FitConfig()
    note_cfg = replace(cfg, prune_tol=0.0, note_sparsity=0.0, prune_velocity=0.0) if use_basic_pitch else cfg

    def initial_notes(audio: torch.Tensor) -> list[Note] | None:
        if not use_basic_pitch:
            return None
        return [n for n in transcribe(audio, cfg.sr) if cfg.pitch_min <= n.pitch <= cfg.pitch_max]

    if not use_demucs:
        return fit(target, note_cfg, verbose, progress, init_notes=initial_notes(target))

    stems = separate(target, cfg.sr, device=cfg.device)
    pitched = sum(stems[s] for s in PITCHED_STEMS if s in stems)
    res = fit(pitched, replace(note_cfg, drum_kinds=0, noise_floor=False), verbose, progress,
              init_notes=initial_notes(pitched))
    if cfg.drum_kinds or cfg.noise_floor:
        drum_progress = None if progress is None else (lambda info: progress({**info, "stage": "drums"}))
        drums = fit(stems["drums"], replace(cfg, n_voices=0, drum_start=0.0), verbose, drum_progress)
        res.noise, res.audio = drums.noise, res.audio + drums.audio
    loss = float(MultiResolutionSTFTLoss(cfg.fft_sizes)(res.audio, target.float().cpu()))
    res.history.append({"stage": "final", "step": 0, "loss": loss, "n_notes": len(res.notes)})
    return res
