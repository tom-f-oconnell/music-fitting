"""Trainable note representations: a soft onset piano roll and an explicit note list."""

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .synth import Synth


@dataclass
class Note:
    pitch: float  # MIDI number (may be fractional after fine-tuning)
    onset: float  # seconds
    duration: float  # seconds (gate length)
    velocity: float  # 0..1
    voice: int = 0

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


class PianoRoll(nn.Module):
    """Soft onset piano roll: one non-negative note-on strength per (pitch, frame).

    This is the relaxation used to discover *which* notes are played: the
    discrete note sequence becomes a continuous [P, M] array that gradient
    descent can move freely, with an L1 penalty pushing it to be sparse.
    Each pitch row has its own (trainable) gate length.
    """

    def __init__(
        self,
        n_frames: int,
        pitch_min: int = 36,
        pitch_max: int = 84,
        frame_hop_s: float = 0.01,
        init_logit: float = -5.0,
        init_noise: float = 0.5,
        note_duration: float = 0.25,
        generator: torch.Generator | None = None,
        tuning_cents: float = 0.0,
    ):
        super().__init__()
        self.frame_hop_s = frame_hop_s
        # one row per semitone, shifted with the recording's tuning offset
        pitches = torch.arange(pitch_min, pitch_max + 1, dtype=torch.float32) + tuning_cents / 100
        self.register_buffer("pitches", pitches)
        P = pitch_max - pitch_min + 1
        noise = torch.randn(P, n_frames, generator=generator) * init_noise
        self.logits = nn.Parameter(torch.full((P, n_frames), init_logit) + noise)
        # gate length per pitch (log-parameterised); one shared length made sustained
        # sounds come out as rapid re-triggers
        self.log_duration = nn.Parameter(torch.full((P,), math.log(note_duration)))

    def strengths(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def duration(self) -> torch.Tensor:
        return torch.exp(self.log_duration).clamp(0.02, 8.0)

    def render(self, synth: Synth, n_samples: int) -> torch.Tensor:
        return synth.render_roll(self.strengths(), self.pitches, self.frame_hop_s, self.duration(), n_samples)

    @torch.no_grad()
    def decode(
        self,
        threshold: float = 0.15,
        merge_frames: int = 3,
        voice: int = 0,
    ) -> list[Note]:
        """Peak-pick the roll into discrete notes.

        Mass spread over neighbouring frames (a common way for the optimiser to
        represent an onset between grid points) is merged: velocity is the sum
        within +/- merge_frames, onset is the strength-weighted mean time.
        """
        v = self.strengths()
        P, M = v.shape
        durs = self.duration().tolist()
        notes: list[Note] = []
        pad = torch.nn.functional.pad(v[None], (merge_frames, merge_frames), value=0.0)[0]
        local_max = torch.nn.functional.max_pool1d(pad[None], 2 * merge_frames + 1, stride=1)[0]
        for pi in range(P):
            for m in torch.nonzero((v[pi] >= local_max[pi]) & (v[pi] > 1e-3)).flatten().tolist():
                lo, hi = max(0, m - merge_frames), min(M, m + merge_frames + 1)
                w = v[pi, lo:hi]
                mass = float(w.sum())
                if mass < threshold:
                    continue
                frames = torch.arange(lo, hi, dtype=torch.float32, device=v.device)
                onset = float((w * frames).sum() / w.sum()) * self.frame_hop_s
                notes.append(Note(float(self.pitches[pi]), onset, durs[pi], min(mass, 1.0), voice))
        notes.sort(key=lambda n: (n.onset, n.pitch))
        return notes


class NoteSequence(nn.Module):
    """Explicit list of notes with continuous, trainable onset/duration/velocity/fine pitch.

    Coarse pitch is fixed (it comes from the piano-roll stage); a bounded
    detune of +/- max_detune semitones is learnable.
    """

    def __init__(self, notes: list[Note], max_detune: float = 0.5):
        super().__init__()
        self.max_detune = max_detune
        self.voice = notes[0].voice if notes else 0
        self.register_buffer("base_pitch", torch.tensor([n.pitch for n in notes], dtype=torch.float32))
        self.detune_raw = nn.Parameter(torch.zeros(len(notes)))
        self.onset = nn.Parameter(torch.tensor([n.onset for n in notes], dtype=torch.float32))
        self.log_duration = nn.Parameter(torch.tensor([math.log(max(n.duration, 0.02)) for n in notes]))
        vel = torch.tensor([min(max(n.velocity, 0.02), 0.98) for n in notes])
        self.vel_logit = nn.Parameter(torch.log(vel / (1 - vel)))

    def __len__(self) -> int:
        return self.base_pitch.shape[0]

    def pitch(self) -> torch.Tensor:
        return self.base_pitch + self.max_detune * torch.tanh(self.detune_raw)

    def duration(self) -> torch.Tensor:
        return torch.exp(self.log_duration).clamp(0.01, 16.0)

    def velocity(self) -> torch.Tensor:
        return torch.sigmoid(self.vel_logit)

    def render(self, synth: Synth, n_samples: int) -> torch.Tensor:
        return synth.render_notes(self.pitch(), self.onset, self.duration(), self.velocity(), n_samples)

    @torch.no_grad()
    def to_notes(self, min_velocity: float = 0.0) -> list[Note]:
        out = []
        for p, o, d, v in zip(self.pitch(), self.onset, self.duration(), self.velocity()):
            if float(v) >= min_velocity:
                out.append(Note(float(p), float(o), float(d), float(v), self.voice))
        out.sort(key=lambda n: (n.onset, n.pitch))
        return out


# ---------------------------------------------------------------------------
# Minimal standard MIDI file writer (no dependencies)
# ---------------------------------------------------------------------------


def _varlen(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def write_midi(notes: list[Note], path: str, tempo_bpm: float = 120.0, ticks_per_beat: int = 480) -> None:
    """Write notes to a type-1 MIDI file, one track (and channel) per voice.

    Fractional pitches are rounded; times are converted at a fixed tempo.
    """
    sec_per_tick = 60.0 / tempo_bpm / ticks_per_beat
    voices = sorted({n.voice for n in notes}) or [0]
    tracks = []
    tempo_us = int(60_000_000 / tempo_bpm)
    for vi, voice in enumerate(voices):
        events = []
        ch = vi % 16
        for n in notes:
            if n.voice != voice:
                continue
            key = int(min(max(round(n.pitch), 0), 127))
            vel = int(min(max(round(n.velocity * 127), 1), 127))
            on = max(0, int(round(n.onset / sec_per_tick)))
            off = max(on + 1, int(round((n.onset + n.duration) / sec_per_tick)))
            events.append((on, 1, bytes([0x90 | ch, key, vel])))
            events.append((off, 0, bytes([0x80 | ch, key, 0])))
        events.sort(key=lambda e: (e[0], e[1]))
        data = bytearray()
        if vi == 0:
            data += b"\x00\xff\x51\x03" + tempo_us.to_bytes(3, "big")
        last = 0
        for tick, _, msg in events:
            data += _varlen(tick - last) + msg
            last = tick
        data += b"\x00\xff\x2f\x00"
        tracks.append(b"MTrk" + struct.pack(">I", len(data)) + bytes(data))
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ticks_per_beat)
    with open(path, "wb") as f:
        f.write(header + b"".join(tracks))
