"""Recorded kits read from their own layouts, downloads of them, and comparing kits in a fit."""

import io
import os
import sys
import tarfile

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from musicfit.dataset import drumkits  # noqa: E402
from musicfit.drumkit import make_kit  # noqa: E402
from musicfit.fit import FitConfig  # noqa: E402

SR = 16000


def _tone(f, seconds=0.5, level=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return level * np.sin(2 * np.pi * f * t) * np.exp(-t / 0.1)


def _salamander(root):
    oh = root / "drumkits" / "salamander" / "salamanderDrumkit" / "OH"
    oh.mkdir(parents=True)
    for name, f in (("kick_OH_P_1", 60), ("kick_OH_F_2", 61), ("kick_OH_F_1", 62), ("snare_OH_FF_1", 200),
                    ("hihatClosed_OH_MP_3", 7000), ("cowbell_FF_2", 800)):
        sf.write(oh / f"{name}.wav", _tone(f), SR)
    return oh


def _drumgizmo(root):
    base = root / "drumkits" / "drskit" / "drskit_minimal-1.0" / "DRSKit"
    for inst, f in (("Kdrum_with_contact", 60), ("Snare", 200)):
        (base / inst / "samples").mkdir(parents=True)
        for n in range(1, 6):  # numbered softest to loudest; two mics each
            for mic in ("OHL", "Snare_top"):
                sf.write(base / inst / "samples" / f"{n}-{inst}-{mic}.wav", _tone(f + n, level=0.1 * n), SR)
    return base


def test_recorded_kits_are_read_from_their_own_layouts(tmp_path, monkeypatch):
    monkeypatch.setattr(drumkits, "DATA_ROOT", str(tmp_path))
    assert drumkits.installed() == [] and drumkits.kit_path("salamander") is None
    with pytest.raises(ValueError, match="Data menu"):
        drumkits.read_recorded("salamander", SR, 1.5)
    _salamander(tmp_path)
    _drumgizmo(tmp_path)
    assert [k["id"] for k in drumkits.installed()] == ["salamander", "drskit"]
    s = drumkits.read_recorded("salamander", SR, 1.5)
    assert sorted(s) == ["closed hat", "cowbell", "kick", "snare"]
    freq = lambda x: np.fft.rfftfreq(len(x), 1 / SR)[np.abs(np.fft.rfft(x)).argmax()]  # noqa: E731
    assert abs(freq(s["kick"]) - 62) < 2  # forte first, the first take of it
    d = drumkits.read_recorded("drskit", SR, 1.5)
    assert sorted(d) == ["kick", "snare"] and abs(freq(d["snare"]) - 204) < 2  # three quarters of the way up: hit 4 of 5
    assert np.abs(d["snare"]).max() == pytest.approx(0.8, rel=0.05)  # its two mics summed
    assert any(f.endswith(".npz") for f in os.listdir(tmp_path / "drumkits" / "drskit"))  # read once, then from the cache
    kit = make_kit("drskit,salamander,808", SR)
    assert kit.kit_names == ["drskit", "salamander", "808"] and kit.available[kit.names.index("cowbell")].tolist() == [False, True, True]


def test_folders_of_samples_are_kits_by_name_and_names_cant_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(drumkits, "DATA_ROOT", str(tmp_path))
    mine = tmp_path / "drumkits" / "my_kit"
    mine.mkdir(parents=True)
    sf.write(mine / "BD.wav", _tone(55), SR)
    (tmp_path / "drumkits" / "not_a_kit").mkdir()
    assert [k["id"] for k in drumkits.installed()] == ["my_kit"]
    assert make_kit("my_kit", SR).names == ["kick"]
    for bad in ("../drumkits", "my_kit/..", ".hidden", "/etc"):
        assert drumkits.kit_path(bad) is None
    with pytest.raises(ValueError, match="no drum kit called"):
        make_kit("TR-1000", SR)


def test_the_catalogue_offers_kits_not_yet_downloaded(tmp_path):
    kits = {k["id"]: k for k in drumkits.catalogue(str(tmp_path))}
    assert [k for k in kits if kits[k]["kind"] in ("machine", "imitation")] == ["808", "909", "606", "cr78", "linndrum", "dmx", "707"]
    assert not kits["salamander"]["installed"] and not kits["drskit"]["installed"] and kits["cr78"]["drums"][0] == "kick"


def test_a_kit_download_is_checked_and_unpacked(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:bz2") as t:
        for name in ("salamanderDrumkit/OH/kick_OH_F_1.wav", "salamanderDrumkit/README"):
            data = io.BytesIO()
            sf.write(data, _tone(60), SR, format="WAV") if name.endswith(".wav") else data.write(b"x")
            info = tarfile.TarInfo(name)
            info.size = len(data.getvalue())
            t.addfile(info, io.BytesIO(data.getvalue()))
    fetched = []

    def fake_download(url, dest, checksum, report):
        fetched.append((url, checksum))
        with open(dest, "wb") as f:
            f.write(archive.getvalue())
        report(len(archive.getvalue()))

    monkeypatch.setattr(drumkits, "_download_checked", fake_download)
    progress = []
    info = drumkits.fetch_drumkit("salamander", str(tmp_path), lambda done, total, name: progress.append(name))
    assert info == {"title": "Salamander Drumkit", "license": "CC BY-SA 3.0"}
    assert fetched == [(drumkits.RECORDED["salamander"]["url"], drumkits.RECORDED["salamander"]["checksum"])]
    assert progress == ["salamanderDrumkit.tar.bz2"] and drumkits.kit_path("salamander", str(tmp_path))
    assert sorted(drumkits.read_recorded("salamander", SR, 1.5, str(tmp_path))) == ["kick"]


def _target():
    kit = make_kit("808,909", SR)
    x = torch.zeros(SR)
    for t, drum, k in ((0.2, "kick", 0), (0.45, "snare", 1), (0.7, "kick", 0)):
        i = int(t * SR)
        x[i:] += 0.4 * kit.samples[kit.names.index(drum), k, : SR - i]
    return x


def test_comparing_kits_keeps_the_best_and_reports_them_all(monkeypatch):
    from musicfit import tools

    monkeypatch.setattr(tools, "available_tools", lambda: {"demucs": False, "basic_pitch": False})
    cfg = FitConfig(n_voices=0, roll_steps=150, note_steps=20, polish_steps=0, device="cpu", tuning_cents=0.0, log_every=1000, seed=1)
    seen = []
    res = tools.fit_with_tools(_target(), cfg, compare_kits=["cr78", "808,909"], verbose=False,
                               progress=lambda info: seen.append((info["stage"], info.get("kit"))))
    assert [k["kit"] for k in res.drum_kits] == ["808,909", "cr78"]  # the target is made of 808 and 909 drums
    assert res.drum_kits[0]["loss"] < res.drum_kits[1]["loss"] and res.drum_kits_on == "clip"
    assert {d["kit"] for d in res.noise.describe()} <= {"808", "909"}  # the final drums are the winner's
    assert torch.allclose(res.audio, res.drum_kits[0]["audio"])  # drums alone on the same audio: not fitted twice
    assert ("kits", "cr78") in seen and ("kits", "808,909") in seen and res.history[-1]["stage"] == "final"
