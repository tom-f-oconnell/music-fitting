import io
import os
import sys
import time

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from musicfit.notes import Note  # noqa: E402
from musicfit.web.app import create_app  # noqa: E402


def _wait(client, job_id, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.2)
    raise TimeoutError(job)


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "jobs")))


def test_demo_job_render_and_exports(client):
    job = client.post("/api/jobs/demo", json={"seed": 1, "length": 1.5, "notes": 3, "roll_steps": 8,
                                              "note_steps": 4}).json()
    job = _wait(client, job["id"])
    assert job["status"] == "done", job["error"]
    r = job["result"]
    assert r["reference"]["metrics_final"]["n_ref"] == 3
    assert job["loss_curve"] and job["progress"] == 1.0
    # the curve covers both stages incl. the post-pruning refinement, within the chart's x range
    assert max(p[0] for p in job["loss_curve"]) < job["config"]["total_steps"]
    for which in ("target", "fit"):
        wav = client.get(f"/api/jobs/{job['id']}/audio/{which}.wav")
        assert wav.status_code == 200 and wav.content[:4] == b"RIFF"

    notes = r["notes"] or [{"pitch": 60, "onset": 0.1, "duration": 0.3, "velocity": 0.8, "voice": 0}]
    resp = client.post(f"/api/jobs/{job['id']}/render", json={"patches": r["patches"], "notes": notes})
    assert resp.status_code == 200
    x, sr = sf.read(io.BytesIO(resp.content))
    assert sr == r["sr"] and len(x) == r["n_samples"] and np.abs(x).max() > 0
    base_loss = float(resp.headers["x-spectral-loss"])

    # a drastic patch change must change the audio and the loss
    patch = dict(r["patches"][0], cutoff=40.0, gain=0.05)
    resp2 = client.post(f"/api/jobs/{job['id']}/render", json={"patches": [patch], "notes": notes})
    y, _ = sf.read(io.BytesIO(resp2.content))
    assert not np.allclose(x, y) and float(resp2.headers["x-spectral-loss"]) != base_loss

    muted = client.post(f"/api/jobs/{job['id']}/render",
                        json={"patches": r["patches"], "notes": notes, "muted_voices": [0], "mute_noise": True})
    z, _ = sf.read(io.BytesIO(muted.content))
    assert np.abs(z).max() == 0
    # the fitted drums / noise floor stay in unless muted too
    assert r["has_noise"]
    only_noise = client.post(f"/api/jobs/{job['id']}/render",
                             json={"patches": r["patches"], "notes": notes, "muted_voices": [0]})
    w, _ = sf.read(io.BytesIO(only_noise.content))
    assert np.abs(w).max() > 0

    midi = client.post("/api/midi", json={"notes": notes})
    assert midi.status_code == 200 and midi.content[:4] == b"MThd"


def test_upload_job(client, tmp_path):
    sr = 16000
    t = np.arange(int(0.8 * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (t < 0.5)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    job = client.post("/api/jobs", files={"file": ("tone.wav", buf.getvalue(), "audio/wav")},
                      data={"duration": "0.8", "roll_steps": "5", "note_steps": "3", "pitch_min": "50",
                            "pitch_max": "62"}).json()
    job = _wait(client, job["id"])
    assert job["status"] == "done", job["error"]
    assert job["result"]["n_samples"] == len(x)
    assert client.get("/api/jobs").json()[0]["id"] == job["id"]
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_tools_are_offered_only_when_installed(client, monkeypatch):
    assert set(client.get("/api/specs").json()["tools"]) == {"demucs", "basic_pitch"}
    monkeypatch.setattr("musicfit.web.app.available_tools", lambda: {"demucs": False, "basic_pitch": False})
    for option in ("separate", "basic_pitch"):
        bad = client.post("/api/jobs", files={"file": ("x.wav", b"x", "audio/wav")}, data={"duration": "1", option: "on"})
        assert bad.status_code == 422 and "requirements-tools" in bad.json()["detail"]


def test_every_slider_has_hover_text(client):
    specs = client.get("/api/specs").json()
    assert all(p["help"] for p in specs["params"].values())
    assert {name for g in specs["groups"] for name in g["params"]} == set(specs["params"])
    assert all(specs["waveform_help"][w] for w in specs["waveforms"])


def test_upload_job_with_tools(client, monkeypatch):
    """The form's checkboxes reach fit_with_tools (with the tools themselves replaced)."""
    sr = 16000
    t = np.arange(int(0.8 * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    monkeypatch.setattr("musicfit.web.app.available_tools", lambda: {"demucs": True, "basic_pitch": True})
    monkeypatch.setattr("musicfit.tools.separate", lambda audio, sr, device="auto": {
        "drums": audio * 0.1, "bass": audio * 0.3, "other": audio * 0.6})
    monkeypatch.setattr("musicfit.tools.transcribe", lambda audio, sr: [Note(57.0, 0.1, 0.5, 0.8)])
    job = client.post("/api/jobs", files={"file": ("tone.wav", buf.getvalue(), "audio/wav")},
                      data={"duration": "0.8", "roll_steps": "8", "note_steps": "3", "pitch_min": "50", "pitch_max": "62",
                            "separate": "on", "basic_pitch": "on", "lead": "on"}).json()
    job = _wait(client, job["id"])
    assert job["status"] == "done", job["error"]
    assert [n["pitch"] for n in job["result"]["roll_notes"]] == [pytest.approx(57.0, abs=0.5)]
    assert job["result"]["has_noise"] and job["result"]["n_harmonics"] == 128
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200


def test_rejects_bad_input(client):
    bad = client.post("/api/jobs", files={"file": ("x.wav", b"not audio", "audio/wav")}, data={"duration": "60"})
    assert bad.status_code == 422
    job = client.post("/api/jobs", files={"file": ("x.wav", b"not audio", "audio/wav")}, data={"duration": "1"}).json()
    job = _wait(client, job["id"])
    assert job["status"] == "failed" and job["error"]
