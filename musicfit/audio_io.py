"""Audio loading/saving.  Loading goes through ffmpeg so any format yt-dlp produces works."""

from __future__ import annotations

import subprocess

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str, sr: int = 16000, offset: float = 0.0, duration: float | None = None) -> torch.Tensor:
    """Decode any audio file to a mono float32 tensor at `sr` using ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(offset)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())


def save_audio(path: str, audio: torch.Tensor, sr: int, normalize: bool = False) -> None:
    x = audio.detach().cpu().float().numpy()
    if normalize:
        x = x / (np.abs(x).max() + 1e-9) * 0.9
    sf.write(path, np.clip(x, -1, 1), sr, subtype="PCM_16")
