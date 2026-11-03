"""Drum kits, the drum voices, drums-only fits, drums in MIDI files, and scoring drum hits."""

import math
import os
import sys

import mido
import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from musicfit.drumkit import DRUMS, make_kit  # noqa: E402
from musicfit.drums import KitVoice, NoiseVoice, load_drum_voice  # noqa: E402
from musicfit.fit import FitConfig, fit  # noqa: E402
from musicfit.notes import Note, write_midi  # noqa: E402
from musicfit.slakh import score_drums  # noqa: E402

SR = 16000


def test_the_built_in_kits():
    kit = make_kit("808,909", SR)
    assert kit.names == [name for name, _ in DRUMS] and kit.kit_names == ["808", "909"]
    assert kit.available[:, 0].all()  # the 808 has every drum
    assert not kit.available[kit.names.index("cowbell"), 1]  # the 909 has no cowbell
    peaks = kit.samples.abs().amax(dim=2)
    assert torch.allclose(peaks[kit.available], torch.ones(int(kit.available.sum())), atol=1e-4)
    freqs = torch.fft.rfftfreq(kit.samples.shape[2], 1 / SR)

    def share(drum, kit_index, lo, hi):  # of the sound's energy, between lo and hi Hz
        power = torch.fft.rfft(kit.samples[kit.names.index(drum), kit_index]).abs() ** 2
        return float(power[(freqs >= lo) & (freqs < hi)].sum() / power.sum())

    for k in (0, 1):
        assert share("kick", k, 0, 200) > 0.8 and share("closed hat", k, 4000, 8000) > 0.8


def test_a_kit_of_your_own(tmp_path):
    for name in ("BD 808.wav", "Snare 2.wav", "hihat closed.wav", "open hh.wav", "pad.wav", "notes.txt"):
        path = tmp_path / name
        if name.endswith(".wav"):
            sf.write(path, 0.5 * np.sin(np.linspace(0, 200, SR // 2)), SR)
        else:
            path.write_text("x")
    kit = make_kit(str(tmp_path), SR)
    assert kit.names == ["kick", "snare", "closed hat", "open hat"] and kit.kit_names == [tmp_path.name]
    with pytest.raises(ValueError, match="named after a drum"):
        make_kit(str(tmp_path / "missing_folder_is_not_a_kit") if False else str(_empty(tmp_path)), SR)


def _empty(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "pad.wav").write_bytes(b"")
    return d


def _voice(n_frames=100):
    return KitVoice(n_frames, 0.01, SR, make_kit("808,909", SR), generator=torch.Generator().manual_seed(0))


def test_kit_voice_is_linear_in_hits_and_differentiable():
    v = _voice()
    with torch.no_grad():
        v.logits.fill_(-30.0)
        v.logits[0, 10] = 30.0
        one = v.render(SR)
        v.logits[0, 50] = 30.0
        two = v.render(SR)
    assert torch.allclose(two[:5000], one[:5000], atol=1e-5) and float((two - one).abs().max()) > 0.1
    v.render(SR).pow(2).sum().backward()
    assert v.kit_logits.grad.abs().sum() > 0 and all(torch.isfinite(p.grad).all() for p in v.parameters() if p.grad is not None)


def test_committing_keeps_one_kit_per_drum():
    v = _voice()
    with torch.no_grad():
        v.kit_logits[0] = torch.tensor([0.0, 2.0])  # kick leans 909; the rest are ties
    v.commit()
    w = v.kit_weights()
    assert torch.allclose(w.max(dim=1).values, torch.ones(len(v.names)))
    assert v.describe()[0]["kit"] == "909" and v.describe()[v.names.index("cowbell")]["kit"] == "808"
    assert not v.kit_logits.requires_grad


@pytest.mark.parametrize("make", [_voice, lambda: NoiseVoice(100, 0.01, SR, 3, generator=torch.Generator().manual_seed(0))])
def test_hits_play_the_same_after_saving(tmp_path, make):
    v = make()
    with torch.no_grad():
        v.logits.fill_(-30.0)
        v.logits[0, 20] = v.logits[-1, 60] = 30.0
    hits = v.hits()
    assert [(k, round(t, 2)) for k, t, _ in hits] == [(0, 0.2), (len(v.names) - 1, 0.6)]
    with torch.no_grad():
        assert torch.allclose(v.render_hits(hits, SR), v.render(SR), atol=1e-4)
    v.save(str(tmp_path / "drums.pt"))
    again = load_drum_voice(str(tmp_path / "drums.pt"))
    assert again.names == v.names and again.gm_notes == v.gm_notes
    assert torch.allclose(again.render_hits(hits, SR), v.render_hits(hits, SR), atol=1e-5)
    assert torch.allclose(again.render_hits(hits[:1], SR), v.render_hits(hits[:1], SR), atol=1e-5)


def test_a_drums_only_fit_finds_the_hits_and_the_kit():
    kit = make_kit("808,909", SR)
    kick, snare = kit.samples[kit.names.index("kick"), 0], kit.samples[kit.names.index("snare"), 1]
    target = torch.zeros(SR)
    for t, sample in ((0.2, kick), (0.7, kick), (0.45, snare)):
        i = int(t * SR)
        target[i:] += 0.4 * sample[:SR - i]
    # drum_start is left at its default: with no voices the drums have to start at once. The default
    # hit cost suits real drums, which no kit here matches exactly; with an exact match, a higher
    # one finds just the true hits (design.md).
    res = fit(target, FitConfig(n_voices=0, roll_steps=200, note_steps=60, polish_steps=0, device="cpu", tuning_cents=0.0,
                                log_every=1000, seed=1, drum_sparsity_alone=5.0), verbose=False)
    names = res.noise.names
    hits = {(names[k], round(t, 1)) for k, t, v in res.noise.hits() if v > 0.3}
    assert {("kick", 0.2), ("kick", 0.7), ("snare", 0.4) if ("snare", 0.4) in hits else ("snare", 0.5)} <= hits
    chosen = {d["name"]: d["kit"] for d in res.noise.describe()}
    assert chosen["kick"] == "808" and chosen["snare"] == "909"
    assert res.notes == [] and res.audio.numel() == SR


def test_drums_go_on_channel_ten(tmp_path):
    path = tmp_path / "x.mid"
    write_midi([Note(60.0, 0.0, 0.5, 0.8, 0), Note(48.0, 0.25, 0.5, 0.8, 1)], str(path), drums=[(36, 0.5, 1.0), (42, 0.75, 0.5)])
    tracks = mido.MidiFile(path).tracks
    on = [[(m.channel, m.note) for m in tr if m.type == "note_on"] for tr in tracks]
    assert on == [[(0, 60)], [(1, 48)], [(9, 36), (9, 42)]]


def _hits(*items):
    return [(name, t) for name, t in items]


def test_scoring_drum_hits():
    ref = [Note(36.0, 0.0, 0.1, 1.0), Note(42.0, 0.0, 0.1, 1.0), Note(38.0, 0.5, 0.1, 1.0), Note(46.0, 1.0, 0.1, 1.0)]
    s = score_drums(_hits(("kick", 0.01), ("closed hat", 0.02), ("snare", 0.53), ("kick", 1.5)), ref)
    assert (s["any"]["tp"], s["any"]["n_pred"], s["any"]["n_ref"]) == (3, 4, 4)
    assert s["grouped"]["tp"] == 3 and s["hat"]["recall"] == 0.5 and s["kick"]["precision"] == 0.5
    wrong = score_drums(_hits(("snare", 0.0),), ref)
    assert wrong["any"]["tp"] == 1 and wrong["grouped"]["tp"] == 0
    # a generic voice's kinds count for the group they mostly hit
    generic = score_drums(_hits(("drum 1", 0.0), ("drum 1", 1.5), ("drum 2", 0.5)), ref)
    assert generic["grouped"]["tp"] == 2 and generic["kick"]["tp"] == 1 and generic["snare"]["tp"] == 1
    assert score_drums([], ref)["any"]["f1"] == 0.0 and score_drums([], [])["any"]["f1"] == 0.0


def test_hits_can_be_played_with_other_settings():
    v = _voice()
    v.commit()
    fitted = v.describe()
    kick = v.names.index("kick")
    hits = [(kick, 0.1, 1.0)]
    base = v.render_hits(hits, SR)
    level = fitted[kick]["level"]
    louder = [{}] * len(v.names)
    louder[kick] = {"level": 2 * level}
    assert torch.allclose(v.render_hits(hits, SR, settings=louder), 2 * base, atol=1e-5)
    # the other kit's kick, as if the fit had chosen it
    other = "909" if fitted[kick]["kit"] == "808" else "808"
    switched = [{}] * len(v.names)
    switched[kick] = {"kit": other}
    with torch.no_grad():
        v2 = _voice()
        v2.kit_logits[kick] = torch.tensor([0.0, 10.0] if other == "909" else [10.0, 0.0])
        v2.commit()
        v2.raw.load_state_dict(v.raw.state_dict())
    assert torch.allclose(v.render_hits(hits, SR, settings=switched), v2.render_hits(hits, SR), atol=1e-5)
    # tuned up an octave, the sample plays twice as fast
    up = [{}] * len(v.names)
    up[kick] = {"tune": 12.0}
    tuned = v.render_hits(hits, SR, settings=up)
    assert float((tuned[int(0.6 * SR):] ** 2).sum()) < 0.5 * float((base[int(0.6 * SR):] ** 2).sum())
    # out of range is clamped, nonsense ignored, and the voice itself unchanged
    wild = [{}] * len(v.names)
    wild[kick] = {"level": 1e6, "decay": "loud", "kit": "TR-1000"}
    top = [{}] * len(v.names)
    top[kick] = {"level": 2.0}
    assert torch.allclose(v.render_hits(hits, SR, settings=wild), v.render_hits(hits, SR, settings=top))
    assert v.describe() == fitted and fitted[kick]["kits"] == ["808", "909"]


def test_the_generic_voice_can_be_played_with_other_settings():
    v = NoiseVoice(100, 0.01, SR, 3, generator=torch.Generator().manual_seed(0))
    d = v.describe()
    assert set(NoiseVoice.SETTINGS) <= set(d[0]) and d[0]["kit"] == ""
    hits = [(1, 0.1, 1.0)]
    quieter = [{}, {"level": d[1]["level"] / 4}, {}]
    assert torch.allclose(v.render_hits(hits, SR, False, quieter), v.render_hits(hits, SR, False) / 4, atol=1e-6)


def test_the_fitted_drums_play_back_whole_with_hits_muted():
    v = _voice()
    v.commit()
    snare, clap, kick = (v.names.index(n) for n in ("snare", "clap", "kick"))
    with torch.no_grad():
        v.logits.fill_(-30.0)
        v.logits[kick, 20] = v.logits[snare, 50] = 30.0
        v.logits[clap, 51] = 0.0  # a weaker layer of the same snare, which hits() merges away
        v.logits[kick, 70] = -1.0  # too weak to be a hit, but still part of the sound
        hits = v.hits()
        assert [(v.names[k], round(t, 2)) for k, t, _ in hits] == [("kick", 0.2), ("snare", 0.5)]
        assert torch.allclose(v.render_fitted(SR), v.render(SR), atol=1e-6)
        assert not torch.allclose(v.render_hits(hits, SR), v.render(SR), atol=1e-3)  # the hits alone miss some
        muted = v.render_fitted(SR, muted=[(snare, 0.5)])
        v.logits[snare, 50] = v.logits[clap, 51] = -30.0
        assert torch.allclose(muted, v.render(SR), atol=1e-4)  # the snare went, and its clap layer with it
