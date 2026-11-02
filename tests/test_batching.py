"""Batched rendering must give what rendering everything at once gives, for less memory."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from musicfit.batching import backward_in_batches, render_mix  # noqa: E402
from musicfit.fit import (  # noqa: E402
    FitConfig,
    _note_batch,
    _prune_by_subtraction,
    _prune_redundant,
    _prune_tolerance,
    _roll_batch,
    fit,
)
from musicfit.loss import MultiResolutionSTFTLoss  # noqa: E402
from musicfit.notes import Note, NoteSequence, PianoRoll  # noqa: E402
from musicfit.synth import Synth, SynthParams  # noqa: E402

SR = 16000
T = 2 * SR
NOTES = [Note(48.0, 0.10, 0.3, 0.9), Note(60.0, 0.15, 0.2, 0.6), Note(67.0, 0.55, 0.3, 0.8), Note(55.0, 0.9, 0.2, 1.0),
         Note(72.0, 1.0, 0.25, 0.5), Note(64.0, 1.4, 0.3, 0.7)]
PATCH = {"amp_release": 0.05, "noise_level": 0.05, "delay_mix": 0.2, "reverb_mix": 0.2}


def _synth(**more):
    return Synth(SR, 32, params=SynthParams({**PATCH, **more}))


def _relative_error(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("more,glides", [({}, False), ({"cutoff": 300.0, "resonance": 5.0}, False), ({"unison_cents": 20.0}, False), ({}, True)])
def test_windowed_notes_sound_the_same(more, glides):
    synth, seq = _synth(**more), NoteSequence(NOTES, glides=glides)
    with torch.no_grad():
        full = seq.render(synth, T)
        # a glide's start can round to a neighbouring sample, which shifts its phase and nothing else
        assert _relative_error(seq.render_windowed(synth, T), full) < (2e-2 if glides else 1e-3)


def test_windows_cover_only_the_notes():
    synth = _synth()
    y, start = synth.render_notes_window(torch.tensor([60.0]), torch.tensor([1.0]), torch.tensor([0.1]), torch.tensor([0.8]), T)
    assert start == 1.0 * SR // 32 * 32 and start % 32 == 0
    assert y.numel() < 1.0 * SR  # 0.1 s of note and 0.7 s of release
    empty, _ = synth.render_notes_window(torch.tensor([60.0]), torch.tensor([5.0]), torch.tensor([0.1]), torch.tensor([0.8]), T)
    assert empty.numel() == 0


def test_no_notes_is_silence_through_the_effects():
    assert float(_synth().render_notes_windowed(*(torch.zeros(0) for _ in range(4)), T).abs().max()) == 0


def _target():
    t = torch.arange(T) / SR
    return 0.3 * torch.sin(2 * math.pi * 220 * t) * (t > 0.2)


def _smooth_loss(pred, target):
    # the real loss's log magnitudes make 32-bit gradients noisy near silence, whatever computes them
    return (pred - target).pow(2).mean()


def _gradients(*models):
    return {f"{i}.{n}": p.grad.clone() for i, m in enumerate(models) for n, p in m.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("kind", ["notes", "roll"])
def test_batched_gradients_match_ordinary_ones(kind):
    synth, target = _synth(), _target()
    if kind == "notes":
        model, size = NoteSequence(NOTES), 2
    else:
        model, size = PianoRoll(200, 48, 60, 0.01, generator=torch.Generator().manual_seed(0)), 4
    _smooth_loss(model.render(synth, T), target).backward()
    ordinary = _gradients(synth, model)
    synth.zero_grad()
    model.zero_grad()

    def loss_of_mixes(mixes):
        return _smooth_loss(synth.effects(synth.params.values(), mixes[0]), target)

    backward_in_batches([model.batches(synth, T, size)], T, "cpu", loss_of_mixes)
    batched = _gradients(synth, model)
    assert set(batched) == set(ordinary)
    va = torch.cat([batched[k].flatten() for k in ordinary])
    vb = torch.cat([ordinary[k].flatten() for k in ordinary])
    assert _relative_error(va, vb) < (1e-2 if kind == "notes" else 1e-3)


def test_a_mix_is_the_sum_of_its_batches():
    synth, roll = _synth(), PianoRoll(200, 48, 60, 0.01, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        mix = render_mix(roll.batches(synth, T, 4), T, "cpu")
        assert _relative_error(synth.effects(synth.params.values(), mix), roll.render(synth, T)) < 1e-4


def test_pruning_by_subtraction_decides_the_same():
    synth, loss_fn = _synth(), MultiResolutionSTFTLoss()
    with torch.no_grad():
        target = NoteSequence(NOTES[:3]).render(synth, T)
    candidates = sorted(NOTES, key=lambda n: n.onset)
    for tol in (0.002, 0.01, 0.05):
        old = _prune_redundant(candidates, [synth], target, loss_fn, tol)
        new = _prune_by_subtraction(candidates, [synth], target, loss_fn, tol)
        assert old == new
    assert len(_prune_by_subtraction(candidates, [synth], target, loss_fn, 0.01)) < len(candidates)


def test_the_sinusoid_cache_stays_within_its_limit():
    synth, roll = _synth(), PianoRoll(200, 48, 60, 0.01, generator=torch.Generator().manual_seed(0))
    synth.sin_cache_limit = 16 * 32 * T * 4  # room for about one batch of four rows
    with torch.no_grad():
        render_mix(roll.batches(synth, T, 4), T, "cpu")
    assert 0 < sum(t.numel() * 4 for t in synth._sin_cache.values()) <= synth.sin_cache_limit


def test_batch_sizes_follow_the_clip_length():
    cfg = FitConfig()
    assert [_roll_batch(cfg, int(s * SR)) for s in (4, 15, 30, 60, 120, 300)] == [49, 13, 6, 3, 1, 1]
    assert _roll_batch(FitConfig(roll_batch=3), 60 * SR) == 3
    assert _note_batch(cfg, 30, 4 * SR) == 30 and _note_batch(cfg, 0, 4 * SR) == 1  # short clips: all at once
    assert _note_batch(cfg, 120, 15 * SR) == 16 and _note_batch(FitConfig(note_batch=5), 120, 15 * SR) == 5


def test_the_pruning_bar_can_scale_with_the_clip():
    default = FitConfig()
    assert _prune_tolerance(default, 4 * SR) == 0.01 and _prune_tolerance(default, 15 * SR) == 0.01
    assert _prune_tolerance(default, 60 * SR) == pytest.approx(0.0025)
    assert _prune_tolerance(FitConfig(prune_reference_s=0), 60 * SR) == 0.01


TINY = dict(n_harmonics=8, roll_steps=4, note_steps=2, polish_steps=1, drum_kinds=0, device="cpu", tuning_cents=0.0, log_every=100,
            prune_tol=0.0, pitch_min=48, pitch_max=60)


def _tone():
    t = torch.arange(SR // 2) / SR
    return 0.3 * torch.sin(2 * math.pi * 220 * t)


def test_running_out_of_memory_shrinks_the_batches_and_carries_on(monkeypatch):
    seen = {"roll": set(), "notes": set()}
    roll_dry, window = Synth.render_roll_dry, Synth.render_notes_window

    def small_rolls_only(self, p, onset_roll, *args):
        seen["roll"].add(onset_roll.shape[0])
        if onset_roll.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("full")
        return roll_dry(self, p, onset_roll, *args)

    def small_notes_only(self, pitch, *args):
        seen["notes"].add(pitch.numel())
        return window(self, pitch, *args)

    monkeypatch.setattr(Synth, "render_roll_dry", small_rolls_only)
    monkeypatch.setattr(Synth, "render_notes_window", small_notes_only)
    res = fit(_tone(), FitConfig(**TINY), verbose=False)
    assert res.audio.numel() == SR // 2
    assert max(seen["roll"]) == 13 and min(seen["roll"]) == 1  # 13 rows, then 6, 3 and 1


def test_running_out_of_memory_at_one_row_a_time_is_an_error(monkeypatch):
    def always_full(self, *args):
        raise torch.cuda.OutOfMemoryError("full")

    monkeypatch.setattr(Synth, "render_roll_dry", always_full)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        fit(_tone(), FitConfig(**TINY), verbose=False)


def test_notes_shrink_too_when_memory_runs_out(monkeypatch):
    sizes = []
    window = Synth.render_notes_window

    def small_only(self, pitch, *args):
        sizes.append(pitch.numel())
        if pitch.numel() > 1:
            raise torch.cuda.OutOfMemoryError("full")
        return window(self, pitch, *args)

    monkeypatch.setattr(Synth, "render_notes_window", small_only)
    notes = [Note(57.0, 0.05, 0.1, 0.8), Note(60.0, 0.2, 0.1, 0.8), Note(64.0, 0.3, 0.1, 0.8)]
    res = fit(_tone(), FitConfig(**TINY), verbose=False, init_notes=notes)
    assert sizes[0] == 3 and 1 in sizes and len(res.notes) == 3


@pytest.mark.parametrize("mode", ["batched", "full"])
def test_both_memory_modes_fit(mode):
    res = fit(_tone(), FitConfig(**TINY, memory_mode=mode), verbose=False)
    assert res.audio.numel() == SR // 2 and res.history[-1]["stage"] == "final"


def test_an_unknown_memory_mode_is_refused():
    with pytest.raises(ValueError, match="memory_mode"):
        fit(_tone(), FitConfig(**TINY, memory_mode="neither"), verbose=False)
