"""BabySlakh: songs rendered from MIDI with sampled instruments, so the true notes are known.

The first 20 tracks of Slakh2100 (Manilow et al., WASPAA 2019), CC BY 4.0, 16 kHz:
https://zenodo.org/records/4603870.  Each ``TrackNNNNN`` directory has ``mix.wav``,
``stems/Sxx.wav``, ``MIDI/Sxx.mid`` and ``metadata.yaml``.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, replace

import mido
import soundfile as sf
import torch
import yaml

from .notes import Note


# These sound an octave below their MIDI (measured with basic-pitch on every stem; see design.md):
# on Acoustic Bass notes are found at the written pitch 2 % of the time and an octave down 97 %.
OCTAVE_DOWN_CLASSES = {"Bass"}
OCTAVE_DOWN_PROGRAMS = {"Lead 2 (sawtooth)", "Brass Section", "French Horn", "Trombone"}


@dataclass
class Stem:
    name: str  # S00, S01, ...
    inst_class: str  # Bass, Piano, Drums, Synth Lead, ...
    program: str  # e.g. Electric Bass (finger)
    is_drum: bool
    audio: str  # path of the stem's WAV
    notes: list[Note]


@dataclass
class Clip:
    """A window of a track: its audio and the true notes starting inside it."""

    clip_id: int
    track: str
    start: float
    audio: torch.Tensor
    ref: list[Note]
    ref_class: list[str]  # the instrument class of each reference note
    n_out_of_range: int  # true notes left out of ref for being outside the fit's pitch range


def read_midi_notes(path: str) -> list[Note]:
    """All notes of a MIDI file, with times in seconds (following its tempo changes).

    Args:
        path: path of a standard MIDI file.

    Returns:
        Notes sorted by onset, with velocity scaled to 0..1.
    """
    notes, active, now = [], {}, 0.0
    for msg in mido.MidiFile(path):
        now += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue
        key = (msg.channel, msg.note)
        if msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(key, []).append((now, msg.velocity))
        elif active.get(key):
            on, velocity = active[key].pop(0)
            notes.append(Note(float(msg.note), on, now - on, velocity / 127))
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def read_track(track_dir: str) -> list[Stem]:
    """The rendered stems of a track, with their notes at the pitch they sound.

    Args:
        track_dir: a ``TrackNNNNN`` directory.
    """
    with open(os.path.join(track_dir, "metadata.yaml")) as f:
        meta = yaml.safe_load(f)
    stems = []
    for name, info in sorted(meta["stems"].items()):
        audio, midi = os.path.join(track_dir, "stems", name + ".wav"), os.path.join(track_dir, "MIDI", name + ".mid")
        # the metadata's audio_rendered is false even for stems that are there
        if os.path.exists(audio) and os.path.exists(midi):
            program = info.get("midi_program_name", "")
            notes = read_midi_notes(midi)
            if info["inst_class"] in OCTAVE_DOWN_CLASSES or program in OCTAVE_DOWN_PROGRAMS:
                notes = [replace(n, pitch=n.pitch - 12) for n in notes]
            stems.append(Stem(name, info["inst_class"], program, bool(info["is_drum"]), audio, notes))
    return stems


def source_stems(stems: list[Stem], source: str) -> list[Stem]:
    """The stems making up a source.

    Args:
        stems: all stems of a track.
        source: "mix" (all of them), "pitched" (all but drums; what perfect drum removal would
            leave), or an instrument class such as "Bass".
    """
    if source == "mix":
        return stems
    if source == "pitched":
        return [s for s in stems if not s.is_drum]
    return [s for s in stems if s.inst_class.lower() == source.lower()]


def _read_window(path: str, start: float, length: float, sr: int) -> torch.Tensor:
    y, file_sr = sf.read(path, start=int(start * sr), frames=int(length * sr), dtype="float32", always_2d=True)
    if file_sr != sr:
        raise ValueError(f"{path} is at {file_sr} Hz, expected {sr}")
    return torch.nn.functional.pad(torch.from_numpy(y.mean(1)), (0, max(0, int(length * sr) - y.shape[0])))


def make_clips(
    root: str,
    source: str = "mix",
    length: float = 4.0,
    clips_per_track: int = 1,
    min_notes: int = 4,
    max_notes: int = 60,
    pitch_range: tuple[int, int] = (36, 84),
    sr: int = 16000,
) -> list[Clip]:
    """Cut every track into windows and pick evenly spaced ones with a usable number of notes.

    The choice depends only on the notes of the *pitched stems of the mix* and the length,
    so different sources and settings are compared on the same windows wherever a source plays.

    Args:
        root: directory holding the ``Track*`` directories.
        source: see :func:`source_stems`.
        length: window length in seconds.
        clips_per_track: windows to take from each track.
        min_notes: skip windows with fewer true notes of the source than this.
        max_notes: skip windows with more true notes of the source than this.
        pitch_range: lowest and highest MIDI pitch the fit can find; notes outside it are not
            counted as missed.
        sr: expected sample rate.

    Returns:
        Clips, with audio normalised to a peak of 0.5 (as ``scripts/fit_audio.py`` does).
    """
    clips = []
    for track_dir in sorted(glob.glob(os.path.join(root, "Track*"))):
        track = os.path.basename(track_dir)
        stems = source_stems(read_track(track_dir), source)
        pitched = [s for s in stems if not s.is_drum]
        if not pitched:
            continue
        end = sf.info(stems[0].audio).duration
        starts = []
        for k in range(int(end // length)):
            n = sum(k * length <= q.onset < (k + 1) * length for s in pitched for q in s.notes)
            if min_notes <= n <= max_notes:
                starts.append(k * length)
        step = max(1, len(starts) // (clips_per_track + 1))
        for i, start in enumerate(starts[step - 1::step][:clips_per_track]):
            audio = sum(_read_window(s.audio, start, length, sr) for s in stems)
            ref, ref_class, dropped = [], [], 0
            for s in pitched:
                for q in s.notes:
                    if start <= q.onset < start + length:
                        if pitch_range[0] <= q.pitch <= pitch_range[1]:
                            ref.append(replace(q, onset=q.onset - start))
                            ref_class.append(s.inst_class)
                        else:
                            dropped += 1
            order = sorted(range(len(ref)), key=lambda j: (ref[j].onset, ref[j].pitch))
            clips.append(Clip(int(track[5:]) * 100 + i, track, start, audio / (audio.abs().max() + 1e-8) * 0.5,
                              [ref[j] for j in order], [ref_class[j] for j in order], dropped))
    return clips
