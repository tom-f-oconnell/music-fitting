"""BabySlakh loader, unison spectrum, glides, acid target and the optional tools."""

import importlib.util
import math
import os

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from musicfit.fit import FitConfig, fit, note_metrics
from musicfit.loss import MultiResolutionSTFTLoss, unison_spectrum
from musicfit.notes import Note, NoteSequence, write_midi
from musicfit.slakh import make_clips, read_midi_notes, read_track, source_stems
from musicfit.synth import Synth, SynthParams, glide_warp
from musicfit.synthetic import acid_target

SR = 16000


def _fake_track(root, name="Track00001"):
    """A tiny track: a bass an octave below its MIDI, a piano, drums, and one stem with no audio."""
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "stems"))
    os.makedirs(os.path.join(d, "MIDI"))
    parts = {
        "S00": ("Bass", "Electric Bass (finger)", False, [Note(52.0, 0.5, 0.5, 0.8), Note(55.0, 4.5, 0.5, 0.8)] * 1),
        "S01": ("Piano", "Bright Acoustic Piano", False, [Note(60.0 + i, 0.5 * i, 0.4, 0.8) for i in range(16)]),
        "S02": ("Drums", "Drums", True, [Note(36.0, 0.5 * i, 0.1, 0.8) for i in range(16)]),
        "S03": ("Guitar", "Overdriven Guitar", False, [Note(70.0, 1.0, 0.1, 0.8)]),
    }
    stems = {}
    for s, (cls, program, drum, notes) in parts.items():
        stems[s] = {"inst_class": cls, "midi_program_name": program, "is_drum": drum, "audio_rendered": False}
        write_midi(notes, os.path.join(d, "MIDI", s + ".mid"))
        if s != "S03":
            sf.write(os.path.join(d, "stems", s + ".wav"), np.full(8 * SR, 0.1, dtype=np.float32), SR)
    with open(os.path.join(d, "metadata.yaml"), "w") as f:
        yaml.safe_dump({"stems": stems}, f)
    return d


def test_read_midi_notes_round_trip(tmp_path):
    notes = [Note(60.0, 0.5, 0.25, 0.5), Note(64.0, 0.5, 1.0, 1.0), Note(60.0, 2.0, 0.5, 0.25)]
    path = str(tmp_path / "x.mid")
    write_midi(notes, path)
    got = read_midi_notes(path)
    assert [n.pitch for n in got] == [60.0, 64.0, 60.0]
    for g, n in zip(got, notes):
        assert g.onset == pytest.approx(n.onset, abs=1e-3)
        assert g.duration == pytest.approx(n.duration, abs=1e-3)
        assert g.velocity == pytest.approx(n.velocity, abs=0.01)


def test_read_track_transposes_what_sounds_an_octave_down(tmp_path):
    stems = read_track(_fake_track(str(tmp_path)))
    assert [s.name for s in stems] == ["S00", "S01", "S02"]  # the guitar has no audio
    assert stems[0].notes[0].pitch == 40.0  # bass, written 52
    assert stems[1].notes[0].pitch == 60.0  # piano sounds as written
    assert [s.name for s in source_stems(stems, "pitched")] == ["S00", "S01"]
    assert [s.name for s in source_stems(stems, "bass")] == ["S00"]


def test_make_clips_windows_and_reference(tmp_path):
    _fake_track(str(tmp_path))
    clips = make_clips(str(tmp_path), "mix", length=4.0, clips_per_track=1, pitch_range=(36, 84))
    assert len(clips) == 1
    c = clips[0]
    assert c.audio.numel() == 4 * SR and float(c.audio.abs().max()) == pytest.approx(0.5)
    assert all(0 <= n.onset < 4.0 for n in c.ref)
    assert "Drums" not in c.ref_class  # drums are never reference notes
    assert len(c.ref) == len(c.ref_class)
    narrow = make_clips(str(tmp_path), "mix", length=4.0, pitch_range=(60, 84))[0]
    assert narrow.n_out_of_range > 0 and all(n.pitch >= 60 for n in narrow.ref)


def test_unison_spectrum_keeps_power_and_widens_peaks():
    x = torch.sin(2 * math.pi * 2000.0 * torch.arange(SR) / SR)
    loss = MultiResolutionSTFTLoss((2048,))
    mag = loss._mag(x, 2048)
    wide = unison_spectrum(mag, 25.0)
    assert float((wide**2).sum() / (mag**2).sum()) == pytest.approx(1.0, abs=0.03)
    assert float(wide.amax()) < 0.9 * float(mag.amax())  # 2 kHz +- 25 cents is resolved at this size
    assert (wide > 0.05 * mag.amax()).sum() > (mag > 0.05 * mag.amax()).sum()


def test_unison_spectrum_matches_a_real_unison():
    """Against a real unison, the expected spectrum beats a single copy, and the search finds it."""
    synth = Synth(SR, n_harmonics=64, params=SynthParams({"reverb_mix": 0.0, "delay_mix": 0.0, "cutoff": 8000.0}))
    args = (torch.tensor([0.1]), torch.tensor([1.5]), torch.tensor([0.9]), 2 * SR)
    with torch.no_grad():
        one = synth.render_notes(torch.tensor([50.0]), *args)
        copies = [synth.render_notes(torch.tensor([50.0 + c]), torch.tensor([0.1 + d]), *args[1:])
                  for c, d in ((-0.25, 0.0021), (0.0, 0.0), (0.25, 0.0037))]
    target = sum(copies) / 3**0.5
    fn = MultiResolutionSTFTLoss()
    assert float(fn(one, target, 25.0)) < float(fn(one, target))
    assert fn.best_spread(one, target, (0.0, 5.0, 25.0, 60.0)) == 25.0
    assert fn.best_spread(one, one, (0.0, 5.0, 25.0, 60.0)) == 0.0


def test_glide_reaches_the_note_and_starts_off_it():
    t = (torch.arange(SR) / SR)[None]
    warp = glide_warp(t, torch.tensor([12.0]), torch.tensor(0.02), SR)
    rate = 1 + torch.diff(warp[0]) * SR  # relative frequency
    assert float(rate[0]) == pytest.approx(2.0, abs=0.01)  # an octave up at note-on
    assert float(rate[-1]) == pytest.approx(1.0, abs=1e-3)
    silent = glide_warp(t - 0.5, torch.tensor([12.0]), torch.tensor(0.02), SR)
    assert float(silent[0, SR // 2 - 1]) == 0.0  # nothing accumulates before the onset


def test_notes_with_glides_survive_a_round_trip():
    notes = [Note(50.0, 0.1, 0.2, 0.8), Note(57.0, 0.4, 0.2, 0.8, glide=-3.5)]
    seq = NoteSequence(notes, glides=True)
    assert seq.previous.tolist() == [0, 0]
    got = sorted(seq.to_notes(), key=lambda n: n.onset)
    assert got[1].glide == pytest.approx(-3.5, abs=1e-3)
    assert NoteSequence(notes).glide() is None


def test_many_harmonics_only_cost_partials_below_nyquist():
    synth = Synth(SR, n_harmonics=256, params=SynthParams({"reverb_mix": 0.0, "delay_mix": 0.0, "cutoff": 12000.0}))
    with torch.no_grad():
        y = synth.render_notes(torch.tensor([36.0, 96.0]), torch.tensor([0.1, 0.1]), torch.tensor([0.5, 0.5]),
                               torch.tensor([0.8, 0.8]), SR)
    assert torch.isfinite(y).all() and float(y.abs().max()) > 0.01


def test_acid_target_has_slides_accents_and_known_notes():
    audio, ref, info = acid_target(3)
    assert audio.numel() == 3 * SR and float(audio.abs().max()) == pytest.approx(0.5)
    assert len(ref) >= 8 and info["slides"] > 0 and info["accents"] > 0
    assert {n.velocity for n in ref} == {0.6, 1.0}
    with_drums, ref2, _ = acid_target(3, drums_db=-6.0)
    assert [n.pitch for n in ref2] == [n.pitch for n in ref]
    assert float((with_drums - audio).abs().max()) > 0.1


def test_drums_only_fit_has_no_notes():
    gen = torch.Generator().manual_seed(0)
    target = torch.randn(SR, generator=gen) * torch.exp(-torch.arange(SR) / (0.05 * SR)) * 0.5
    res = fit(target, FitConfig(n_voices=0, drum_start=0.0, roll_steps=20, note_steps=5, polish_steps=2, device="cpu",
                                log_every=100), verbose=False)
    assert res.notes == [] and res.synths == [] and res.noise is not None
    assert res.audio.numel() == SR and float(res.audio.abs().max()) > 0


def test_fit_from_given_notes_skips_stage_one():
    synth = Synth(SR, n_harmonics=16, params=SynthParams({"reverb_mix": 0.0, "delay_mix": 0.0}))
    ref = [Note(60.0, 0.2, 0.3, 0.8), Note(67.0, 0.6, 0.3, 0.8)]
    with torch.no_grad():
        target = synth.render_notes(*(torch.tensor([getattr(n, a) for n in ref]) for a in
                                      ("pitch", "onset", "duration", "velocity")), SR)
    cfg = FitConfig(n_harmonics=16, roll_steps=40, note_steps=10, polish_steps=2, drum_kinds=0, device="cpu",
                    prune_tol=0.0, tuning_cents=0.0, log_every=100)
    res = fit(target, cfg, verbose=False, init_notes=ref)
    assert res.roll_notes == ref
    assert note_metrics(res.notes, ref)["f1"] == 1.0
    assert max(h["step"] for h in res.history if h["stage"] == "roll") == cfg.roll_steps // 4  # just the warm-up


def test_fit_keeps_transcribed_notes_in_its_pitch_range(monkeypatch):
    from musicfit import tools

    t = torch.arange(SR) / SR
    target = 0.3 * torch.sin(2 * math.pi * 392.0 * t)
    quiet_and_wrong = Note(60.0, 0.2, 0.3, 0.02)  # the fit's own penalties would drop this one
    transcribed = [quiet_and_wrong, Note(67.0, 0.1, 0.8, 0.8), Note(100.0, 0.4, 0.2, 0.8)]
    monkeypatch.setattr(tools, "transcribe", lambda audio, sr: transcribed)
    cfg = FitConfig(n_harmonics=16, roll_steps=8, note_steps=6, drum_kinds=0, device="cpu", tuning_cents=0.0, log_every=100)
    res = tools.fit_with_tools(target, cfg, use_basic_pitch=True, verbose=False)
    assert sorted(round(n.pitch) for n in res.notes) == [60, 67]


@pytest.mark.skipif(importlib.util.find_spec("basic_pitch") is None, reason="basic-pitch not installed")
def test_transcribe_finds_a_clear_note():
    from musicfit.tools import transcribe

    t = torch.arange(2 * SR) / SR
    env = ((t > 0.5) & (t < 1.5)).float()
    y = 0.5 * env * sum(torch.sin(2 * math.pi * 220.0 * k * t) / k for k in range(1, 6))
    notes = transcribe(y, SR, voice=1)
    assert any(round(n.pitch) == 57 and abs(n.onset - 0.5) < 0.1 and n.voice == 1 for n in notes)


def _demucs_cached():
    return importlib.util.find_spec("demucs") is not None and os.environ.get("MUSICFIT_TEST_DEMUCS")


@pytest.mark.skipif(not _demucs_cached(), reason="needs Demucs and its weights: set MUSICFIT_TEST_DEMUCS=1")
def test_separate_returns_stems_that_sum_to_the_mix():
    from musicfit.tools import separate

    gen = torch.Generator().manual_seed(0)
    y = torch.randn(2 * SR, generator=gen) * 0.1
    stems = separate(y, SR, device="cpu")
    assert set(stems) >= {"drums", "bass", "other"}
    assert all(s.numel() == y.numel() for s in stems.values())
    assert float((sum(stems.values()) - y).pow(2).sum() / y.pow(2).sum()) < 0.1


def test_tools_explain_a_missing_install(monkeypatch):
    import builtins

    from musicfit import tools

    real_import = builtins.__import__

    def no_basic_pitch(name, *args, **kwargs):
        if name.startswith("basic_pitch"):
            raise ImportError("no " + name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_basic_pitch)
    with pytest.raises(ImportError, match="requirements-tools"):
        tools.transcribe(torch.zeros(SR), SR)
