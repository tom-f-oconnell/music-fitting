import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from musicfit import MultiResolutionSTFTLoss, Note, NoteSequence, PianoRoll, Synth, SynthParams, write_midi  # noqa: E402
from musicfit.fit import FitConfig, fit, note_metrics  # noqa: E402
from musicfit.synth import adsr, waveform_harmonics  # noqa: E402

SR = 8000


def test_adsr_shape():
    t = torch.linspace(-0.1, 1.0, 1101)
    env = adsr(t, torch.tensor(0.5), torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.5), torch.tensor(0.1))
    assert torch.all(env[t < 0] == 0)
    assert abs(float(env[(t - 0.1).abs().argmin()]) - 1.0) < 0.02  # peak at end of attack
    assert float(env[(t - 0.5).abs().argmin()]) > 0.5  # above sustain at note-off
    assert float(env[-1]) < 0.05  # released


def test_waveform_harmonics_pure_shapes():
    w = waveform_harmonics(torch.tensor([1.0, 0.0, 0.0, 0.0]), 8)
    assert torch.allclose(w, torch.tensor([1.0] + [0.0] * 7))
    sq = waveform_harmonics(torch.tensor([0.0, 0.0, 1.0, 0.0]), 8)
    assert sq[1] == 0 and sq[0] > sq[2] > sq[4] > 0


def test_render_notes_and_roll_agree():
    """A roll with a single onset must render the same as the explicit note."""
    synth = Synth(SR, n_harmonics=16)
    T = SR
    dur = torch.tensor(0.3)
    hop = 0.01
    roll = torch.zeros(1, 100)
    roll[0, 20] = 0.8  # onset at 0.2 s
    with torch.no_grad():
        a = synth.render_roll(roll, torch.tensor([60.0]), hop, dur, T)
        b = synth.render_notes(torch.tensor([60.0]), torch.tensor([0.2]), dur[None], torch.tensor([0.8]), T)
    assert a.shape == b.shape == (T,)
    assert torch.isfinite(a).all()
    # the two paths use different oscillator phase references, so compare spectrograms
    A = torch.stft(a, 512, 128, window=torch.hann_window(512), return_complex=True).abs()
    B = torch.stft(b, 512, 128, window=torch.hann_window(512), return_complex=True).abs()
    rel = float((A - B).norm() / B.norm())
    assert rel < 0.1, rel


def test_gradients_reach_everything():
    synth = Synth(SR, n_harmonics=8)
    notes = [Note(60.0, 0.1, 0.2, 0.7), Note(64.0, 0.4, 0.2, 0.5)]
    seq = NoteSequence(notes)
    target = torch.randn(SR) * 0.1
    loss = MultiResolutionSTFTLoss((512, 128))(seq.render(synth, SR), target)
    loss.backward()
    for name, p in list(synth.named_parameters()) + list(seq.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for name in ("onset", "log_duration", "vel_logit", "detune_raw"):
        assert getattr(seq, name).grad.abs().sum() > 0, name
    roll = PianoRoll(100, 55, 65, 0.01)
    loss = MultiResolutionSTFTLoss((512, 128))(roll.render(synth, SR), target)
    loss.backward()
    assert roll.logits.grad.abs().sum() > 0 and roll.log_duration.grad.abs().sum() > 0


def test_piano_roll_decode():
    roll = PianoRoll(50, 60, 62, 0.01, init_logit=-20.0, init_noise=0.0)
    with torch.no_grad():
        roll.logits[1, 10] = 2.0  # pitch 61 at 0.10 s
        roll.logits[1, 11] = 0.0  # neighbouring frame: merged into the same note
        roll.logits[0, 30] = -3.0  # too weak
    notes = roll.decode(threshold=0.3)
    assert len(notes) == 1
    assert notes[0].pitch == 61 and abs(notes[0].onset - 0.1036) < 0.002


def test_write_midi(tmp_path):
    p = tmp_path / "x.mid"
    write_midi([Note(60.0, 0.0, 0.5, 0.8), Note(64.2, 0.5, 0.25, 0.5, voice=1)], str(p))
    data = p.read_bytes()
    assert data[:4] == b"MThd" and data.count(b"MTrk") == 2


def test_note_metrics():
    ref = [Note(60.0, 0.0, 0.1, 1.0), Note(64.0, 0.5, 0.1, 1.0)]
    pred = [Note(60.1, 0.02, 0.1, 1.0), Note(67.0, 0.5, 0.1, 1.0)]
    m = note_metrics(pred, ref)
    assert m["tp"] == 1 and m["precision"] == 0.5 and m["recall"] == 0.5


def test_prune_redundant_keeps_only_needed_notes():
    from musicfit.fit import _prune_redundant

    synth = Synth(SR, n_harmonics=16, params=SynthParams({"reverb_mix": 0.0, "delay_mix": 0.0}))
    real = [Note(60.0, 0.1, 0.3, 0.8), Note(67.0, 0.5, 0.3, 0.7)]
    with torch.no_grad():
        target = synth.render_notes(*(torch.tensor([getattr(n, a) for n in real])
                                      for a in ("pitch", "onset", "duration", "velocity")), SR)
    notes = real + [Note(72.0, 0.8, 0.1, 0.1), Note(60.0, 0.1, 0.3, 0.1)]  # spurious, and a duplicate onset
    kept = _prune_redundant(notes, [synth], target, MultiResolutionSTFTLoss((512, 128)), 0.01)
    assert kept == real


def test_fit_smoke_recovers_single_note():
    """Tiny end-to-end fit: one note, few steps; loss must drop and the note must be found."""
    torch.manual_seed(0)
    true = Synth(SR, n_harmonics=16, params=SynthParams({"cutoff": 1200.0, "reverb_mix": 0.0, "delay_mix": 0.0}))
    T = SR
    with torch.no_grad():
        target = true.render_notes(torch.tensor([64.0]), torch.tensor([0.3]), torch.tensor([0.4]), torch.tensor([0.8]), T)
    cfg = FitConfig(sr=SR, n_harmonics=16, pitch_min=60, pitch_max=67, roll_steps=60, note_steps=20, log_every=30)
    res = fit(target, cfg, verbose=False)
    losses = [h["loss"] for h in res.history if h["stage"] == "roll"]
    assert losses[-1] < losses[0]
    m = note_metrics(res.notes, [Note(64.0, 0.3, 0.4, 0.8)], onset_tol=0.06)
    assert m["recall"] == 1.0


def test_synthetic_mismatch_keeps_melody_and_sets_levels():
    from musicfit.synthetic import synthetic_target

    clean, ref, _ = synthetic_target(9, length_s=2.0)
    same, ref2, _ = synthetic_target(9, length_s=2.0, mismatch={})
    assert torch.equal(clean, same) and ref == ref2
    peak = float(clean.abs().max())
    noisy, ref3, _ = synthetic_target(9, length_s=2.0, mismatch={"noise": -40})
    assert ref3 == ref
    assert abs(20 * torch.log10((noisy - clean).pow(2).mean().sqrt() / peak) + 40) < 0.5
    drums, _, _ = synthetic_target(9, length_s=2.0, mismatch={"drums": -6})
    assert abs(20 * torch.log10((drums - clean).abs().max() / peak) + 6) < 0.1
    coded, _, _ = synthetic_target(9, length_s=2.0, mismatch={"codec": 64})
    assert coded.shape == clean.shape
    assert float((coded - clean).pow(2).sum() / clean.pow(2).sum()) < 0.05  # aligned, small coding error
    tuned, _, _ = synthetic_target(9, length_s=2.0, mismatch={"tuning": 30})
    assert not torch.allclose(tuned, clean)


def test_estimate_tuning():
    from musicfit.tuning import estimate_tuning

    t = torch.arange(2 * SR) / SR
    tones = sum(torch.sin(2 * torch.pi * 440 * 2 ** ((p + 0.3) / 12) * k * t) / k for p in (-9, -5, 0) for k in (1, 2, 3))
    cents, confidence = estimate_tuning(tones, SR)
    assert abs(cents - 30) < 3 and confidence > 0.8
    _, noise_confidence = estimate_tuning(torch.randn(2 * SR, generator=torch.Generator().manual_seed(0)), SR)
    assert noise_confidence < 0.3


def test_noise_voice_render_is_linear_in_hits_and_differentiable():
    from musicfit.drums import NoiseVoice

    voice = NoiseVoice(100, 0.01, SR, n_kinds=2, noise_floor=True, init_logit=-30.0, init_noise=0.0)
    with torch.no_grad():
        floor = voice.render(SR)
        voice.logits[0, 20] = 30.0  # one kind-0 hit at 0.2 s
        one = voice.render(SR) - floor
        voice.logits[0, 50] = 30.0
        two = voice.render(SR) - floor
    hop = SR // 100
    assert one[: 20 * hop].abs().max() < 1e-6 and one.abs().max() > 0  # nothing before the hit
    assert torch.allclose(two[50 * hop:] - one[50 * hop:], one[20 * hop: 20 * hop + len(two) - 50 * hop], atol=1e-5)
    assert [k for k, _, _ in voice.hits()] == [0, 0]
    loss = MultiResolutionSTFTLoss((512, 128))(voice.render(SR), torch.randn(SR) * 0.1)
    loss.backward()
    for name, p in voice.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
