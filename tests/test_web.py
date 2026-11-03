import io
import json
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
from musicfit.web.jobs import JobManager  # noqa: E402


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
    return TestClient(create_app(str(tmp_path / "jobs"), str(tmp_path / "data")))


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

    # the drums' sliders: a kick at the fitted settings, louder, and from the other kit
    drums = r["drums"]
    kick = next(i for i, d in enumerate(drums) if d["name"] == "kick")

    def drums_alone(settings):
        resp = client.post(f"/api/jobs/{job['id']}/render", json={
            "patches": r["patches"], "notes": notes, "muted_voices": [0], "drum_hits": [[kick, 0.3, 1.0]], "drum_settings": settings})
        assert resp.status_code == 200
        return sf.read(io.BytesIO(resp.content))[0]

    fitted = [{"level": d["level"], "tune": d["tune"], "decay": d["decay"], "kit": d["kit"]} for d in drums]
    as_fitted = drums_alone(None)
    assert np.abs(as_fitted).max() > 0 and np.allclose(drums_alone(fitted), as_fitted, atol=1e-3)
    louder = [dict(x) for x in fitted]
    louder[kick]["level"] = 2 * fitted[kick]["level"]
    assert np.allclose(drums_alone(louder), 2 * as_fitted, atol=2e-3)
    other = [dict(x) for x in fitted]
    other[kick]["kit"] = next(k for k in drums[kick]["kits"] if k != drums[kick]["kit"])
    assert not np.allclose(drums_alone(other), as_fitted, atol=1e-2)

    midi = client.post("/api/midi", json={"notes": notes})
    assert midi.status_code == 200 and midi.content[:4] == b"MThd"


def test_the_older_way_of_fitting_is_an_option(client):
    sr = 16000
    x = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(0.5 * sr)) / sr)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    for form, mode in (({}, "batched"), ({"full_memory": "on"}, "full")):
        job = client.post("/api/jobs", files={"file": ("tone.wav", buf.getvalue(), "audio/wav")},
                          data={"duration": "0.5", "roll_steps": "4", "note_steps": "2", "pitch_min": "50", "pitch_max": "62", **form}).json()
        job = _wait(client, job["id"])
        assert job["status"] == "done", job["error"]
        assert client.app.state.jobs._fit_config(job["config"]).memory_mode == mode


def test_gpu_memory_is_released_after_a_job(client, monkeypatch):
    released = []
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: released.append(True))
    manager = client.app.state.jobs
    monkeypatch.setattr(manager, "_run", lambda job: None)
    manager._run_and_release(manager._new("x", "demo", {}))
    assert released


def test_clips_can_be_longer_than_they_used_to(client):
    too_long = client.post("/api/jobs", files={"file": ("x.wav", b"x", "audio/wav")}, data={"duration": "121"})
    assert too_long.status_code == 422 and "120" in too_long.json()["detail"]
    assert client.get("/api/specs").json()["max_duration_s"] == 120


def test_jobs_record_when_they_ran(client):
    before = time.time()
    job = client.post("/api/jobs/demo", json={"seed": 1, "length": 1.0, "notes": 2, "roll_steps": 4, "note_steps": 2}).json()
    job = _wait(client, job["id"])
    assert job["status"] == "done", job["error"]
    assert before <= job["created"] <= job["started"] <= job["finished"] <= time.time()
    assert job["finished"] - job["started"] == pytest.approx(job["elapsed_s"], abs=0.1)
    listed = client.get("/api/jobs").json()[0]
    assert (listed["started"], listed["finished"]) == (job["started"], job["finished"])


def _old_record(root, job_id, status, elapsed, ended_at):
    os.makedirs(root / job_id)
    path = root / job_id / "job.json"
    path.write_text(json.dumps({"id": job_id, "name": "old", "kind": "demo", "config": {}, "created": ended_at - elapsed - 30,
                                "status": status, "elapsed_s": elapsed}))
    os.utime(path, (ended_at, ended_at))


def test_records_from_before_times_were_kept_get_them_from_their_files(tmp_path):
    root = tmp_path / "jobs"
    _old_record(root, "finished", "done", 57.0, 1_000_000.0)
    _old_record(root, "interrupted", "running", 12.0, 2_000_000.0)
    _old_record(root, "neverstarted", "queued", 0.0, 3_000_000.0)
    jobs = JobManager(str(root)).jobs
    assert (jobs["finished"].started, jobs["finished"].finished) == (1_000_000.0 - 57.0, 1_000_000.0)
    assert (jobs["interrupted"].status, jobs["interrupted"].finished) == ("failed", 2_000_000.0)
    assert jobs["interrupted"].started == 2_000_000.0 - 12.0
    assert (jobs["neverstarted"].status, jobs["neverstarted"].started, jobs["neverstarted"].finished) == ("failed", None, 3_000_000.0)


def test_times_survive_a_restart_and_a_cancelled_job_never_started(tmp_path):
    manager = JobManager(str(tmp_path / "jobs"))
    job = manager._new("waiting", "demo", {})
    manager.cancel(job.id)
    assert job.status == "cancelled" and job.started is None and job.finished is not None
    again = JobManager(str(tmp_path / "jobs")).jobs[job.id]
    assert (again.started, again.finished) == (None, job.finished)


def test_jobs_can_be_deleted_while_queued_or_running(client, tmp_path, monkeypatch):
    """Deleting stops a running fit and skips a queued one; their files go, and the next job still runs."""
    import threading

    from musicfit.web import jobs as jobs_module

    release, started = threading.Event(), threading.Event()
    real_fit = jobs_module.fit

    def slow_fit(target, cfg, verbose=False, progress=None):
        started.set()
        while not release.is_set():  # a fit that only stops when told to
            progress({"stage": "roll", "step": 0, "total": 1, "loss": 1.0})
            time.sleep(0.01)
        return real_fit(target, cfg, verbose, progress)

    monkeypatch.setattr("musicfit.web.jobs.fit", slow_fit)
    demo = {"seed": 1, "length": 1.0, "notes": 2, "roll_steps": 4, "note_steps": 2}
    running = client.post("/api/jobs/demo", json=demo).json()["id"]
    assert started.wait(60)
    queued = client.post("/api/jobs/demo", json=demo).json()["id"]
    jobs_dir = tmp_path / "jobs"
    assert client.delete(f"/api/jobs/{queued}").status_code == 200
    assert not (jobs_dir / queued).exists()
    assert client.delete(f"/api/jobs/{running}").status_code == 200
    assert client.get(f"/api/jobs/{running}").status_code == 404 and client.delete(f"/api/jobs/{running}").status_code == 404
    assert not (jobs_dir / running / "job.json").exists()  # a restart now wouldn't list it
    t0 = time.time()
    while (jobs_dir / running).exists() and time.time() - t0 < 30:
        time.sleep(0.05)
    assert not (jobs_dir / running).exists()  # the fit stopped and its files went
    assert client.get("/api/jobs").json() == []

    release.set()
    after = _wait(client, client.post("/api/jobs/demo", json=demo).json()["id"])
    assert after["status"] == "done", after["error"]


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


@pytest.mark.parametrize("path", ["/", "/app.js", "/keyboard.js", "/style.css"])
def test_browsers_must_check_static_files_for_changes(client, path):
    first = client.get(path)
    assert first.status_code == 200 and first.headers["cache-control"] == "no-cache"
    unchanged = client.get(path, headers={"If-None-Match": first.headers["etag"]})
    assert unchanged.status_code == 304 and unchanged.headers["cache-control"] == "no-cache"


def test_every_slider_has_hover_text(client):
    from musicfit.drums import KitVoice, NoiseVoice

    specs = client.get("/api/specs").json()
    assert all(p["help"] for p in specs["params"].values())
    for voice, cls in (("kit", KitVoice), ("noise", NoiseVoice)):
        assert set(specs["drum_params"][voice]) == set(cls.SETTINGS) and all(p["help"] for p in specs["drum_params"][voice].values())
    assert specs["kit_help"]
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
    bad = client.post("/api/jobs", files={"file": ("x.wav", b"not audio", "audio/wav")}, data={"duration": "600"})
    assert bad.status_code == 422
    job = client.post("/api/jobs", files={"file": ("x.wav", b"not audio", "audio/wav")}, data={"duration": "1"}).json()
    job = _wait(client, job["id"])
    assert job["status"] == "failed" and job["error"]
