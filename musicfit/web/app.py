"""FastAPI app: upload a clip, fit synth + notes in the background, tweak and re-render.

    uvicorn musicfit.web.app:app --port 8000          # then open http://localhost:8000
    MUSICFIT_JOBS=/path/to/jobs uvicorn ...            # where jobs and results are stored
"""

from __future__ import annotations

import functools
import os
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..notes import write_midi
from ..synth import PARAM_SPECS, WAVEFORMS
from .jobs import JobManager
from .render import notes_from_dicts, render, spectral_loss, wav_bytes

STATIC = os.path.join(os.path.dirname(__file__), "static")
MAX_DURATION_S = 15.0
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Slider groups and display units for the frontend.
PARAM_GROUPS = [
    ("Amp envelope", ["amp_attack", "amp_decay", "amp_sustain", "amp_release"]),
    ("Filter", ["cutoff", "resonance", "filt_env_amount"]),
    ("Filter envelope", ["filt_attack", "filt_decay", "filt_sustain", "filt_release"]),
    ("Noise & drive", ["noise_level", "drive"]),
    ("Delay", ["delay_time", "delay_feedback", "delay_mix"]),
    ("Reverb", ["reverb_rt60", "reverb_mix"]),
    ("Output", ["gain"]),
]
UNITS = {"cutoff": "Hz", "filt_env_amount": "oct", "reverb_rt60": "s"}
for _n in PARAM_SPECS:
    if _n.endswith(("_attack", "_decay", "_release")) or _n == "delay_time":
        UNITS[_n] = "s"


class DemoRequest(BaseModel):
    seed: int = 1
    length: float = Field(3.0, ge=1.0, le=8.0)
    notes: int = Field(10, ge=1, le=40)
    roll_steps: int = Field(400, ge=1, le=3000)
    note_steps: int = Field(300, ge=1, le=3000)
    sparsity: float = 1.0


class RenderRequest(BaseModel):
    patches: list[dict]
    notes: list[dict]
    muted_voices: list[int] = []
    mute_noise: bool = False  # leave out the fitted drums / noise floor


class MidiRequest(BaseModel):
    notes: list[dict]
    tempo_bpm: float = 120.0


def create_app(jobs_dir: str | None = None) -> FastAPI:
    jobs = JobManager(jobs_dir or os.environ.get("MUSICFIT_JOBS", os.path.join(os.getcwd(), "web_jobs")))
    app = FastAPI(title="musicfit")
    app.state.jobs = jobs

    @functools.lru_cache(maxsize=8)
    def target_of(job_id: str):
        return jobs.target_audio(job_id)

    @functools.lru_cache(maxsize=8)
    def noise_of(job_id: str):
        return jobs.noise_audio(job_id)

    def done_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job.status != "done":
            raise HTTPException(409, f"job is {job.status}")
        return job

    @app.get("/api/specs")
    def specs():
        return {
            "params": {n: {"low": s.low, "high": s.high, "default": s.default, "log": s.log, "unit": UNITS.get(n, "")}
                       for n, s in PARAM_SPECS.items()},
            "groups": [{"title": t, "params": p} for t, p in PARAM_GROUPS],
            "waveforms": list(WAVEFORMS),
            "max_duration_s": MAX_DURATION_S,
        }

    @app.get("/api/jobs")
    def list_jobs():
        return jobs.list()

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        offset: float = Form(0.0),
        duration: float = Form(4.0),
        voices: int = Form(1),
        roll_steps: int = Form(400),
        note_steps: int = Form(300),
        sparsity: float = Form(0.5),
        pitch_min: int = Form(36),
        pitch_max: int = Form(84),
        seed: int = Form(0),
    ):
        if not 0.25 <= duration <= MAX_DURATION_S:
            raise HTTPException(422, f"duration must be between 0.25 and {MAX_DURATION_S} s")
        if not 1 <= voices <= 4 or not 0 <= pitch_min < pitch_max <= 127:
            raise HTTPException(422, "1-4 voices and 0 <= pitch_min < pitch_max <= 127")
        fd, tmp = tempfile.mkstemp(dir=jobs.root)
        size = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    os.unlink(tmp)
                    raise HTTPException(413, "file too large")
                out.write(chunk)
        cfg = dict(offset=max(offset, 0.0), duration=duration, voices=voices, roll_steps=max(roll_steps, 1),
                   note_steps=max(note_steps, 1), sparsity=sparsity, pitch_min=pitch_min, pitch_max=pitch_max,
                   seed=seed)
        job = jobs.submit_upload(os.path.basename(file.filename or "upload"), tmp, cfg)
        return job.public()

    @app.post("/api/jobs/demo")
    def create_demo(req: DemoRequest):
        return jobs.submit_demo(req.model_dump()).public()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        if jobs.get(job_id) is None:
            raise HTTPException(404, "no such job")
        jobs.cancel(job_id)
        return {"ok": True}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job.status in ("queued", "running"):
            raise HTTPException(409, "cancel the job first")
        del jobs.jobs[job_id]
        shutil.rmtree(jobs.dir(job_id), ignore_errors=True)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/audio/{which}.wav")
    def job_audio(job_id: str, which: str):
        if which not in ("target", "fit") or jobs.get(job_id) is None:
            raise HTTPException(404)
        path = os.path.join(jobs.dir(job_id), f"{which}.wav")
        if not os.path.exists(path):
            raise HTTPException(404, "not rendered yet")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/jobs/{job_id}/render")
    def render_job(job_id: str, req: RenderRequest):
        """Re-render the clip with an edited patch / note list (no fitting; well under a second)."""
        job = done_job(job_id)
        r = job.result
        if len(req.patches) != len(r["patches"]):
            raise HTTPException(422, f"expected {len(r['patches'])} patches")
        audio = render(req.patches, notes_from_dicts(req.notes), r["n_samples"], r["sr"], r["n_harmonics"],
                       set(req.muted_voices))
        noise = None if req.mute_noise else noise_of(job_id)
        if noise is not None:
            audio = audio + noise[: audio.numel()]
        loss = spectral_loss(audio, target_of(job_id))
        return Response(wav_bytes(audio, r["sr"]), media_type="audio/wav",
                        headers={"X-Spectral-Loss": f"{loss:.5f}", "Access-Control-Expose-Headers": "X-Spectral-Loss"})

    @app.post("/api/midi")
    def midi(req: MidiRequest):
        fd, path = tempfile.mkstemp(suffix=".mid")
        os.close(fd)
        try:
            write_midi(notes_from_dicts(req.notes), path, tempo_bpm=req.tempo_bpm)
            with open(path, "rb") as f:
                data = f.read()
        finally:
            os.unlink(path)
        return Response(data, media_type="audio/midi")

    @app.exception_handler(ValueError)
    def value_error(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def __getattr__(name):
    # `uvicorn musicfit.web.app:app` -- created lazily so importing the module has no side effects
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
