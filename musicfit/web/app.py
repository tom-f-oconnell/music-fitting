"""FastAPI app: upload a clip, fit synth + notes in the background, tweak and re-render.

    uvicorn musicfit.web.app:app --port 43153          # then open http://localhost:43153
    MUSICFIT_JOBS=/path/to/jobs uvicorn ...            # where jobs and results are stored
    MUSICFIT_DATA=/path/to/data uvicorn ...            # where downloaded datasets are cached
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

from ..dataset.library import Library
from ..notes import write_midi
from ..synth import PARAM_SPECS, WAVEFORMS
from ..tools import available_tools
from .downloads import DownloadManager
from .jobs import JobManager
from .render import notes_from_dicts, render, spectral_loss, wav_bytes

STATIC = os.path.join(os.path.dirname(__file__), "static")
MAX_DURATION_S = 15.0
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Slider groups and display units for the frontend.
PARAM_GROUPS = [
    ("Amp envelope", ["amp_attack", "amp_decay", "amp_sustain", "amp_release"]),
    ("Filter", ["cutoff", "resonance", "filt_env_amount"]),
    ("Glide & unison", ["glide_time", "unison_cents"]),
    ("Filter envelope", ["filt_attack", "filt_decay", "filt_sustain", "filt_release"]),
    ("Noise & drive", ["noise_level", "drive"]),
    ("Delay", ["delay_time", "delay_feedback", "delay_mix"]),
    ("Reverb", ["reverb_rt60", "reverb_mix"]),
    ("Output", ["gain"]),
]
UNITS = {"cutoff": "Hz", "filt_env_amount": "oct", "reverb_rt60": "s", "unison_cents": "cents"}
for _n in PARAM_SPECS:
    if _n.endswith(("_attack", "_decay", "_release")) or _n in ("delay_time", "glide_time"):
        UNITS[_n] = "s"
# Hover text for the sliders.
WAVEFORM_HELP = {
    "sine": "A pure tone with no overtones: soft and round.",
    "saw": "Has every harmonic: bright and buzzy, the classic lead and bass sound.",
    "square": "Has only odd harmonics: hollow and reedy.",
    "triangle": "Odd harmonics that fade quickly: soft, between a sine and a square.",
}
PARAM_HELP = {
    "amp_attack": "How long a note takes to rise from silence to its peak volume. Short is percussive; long fades in.",
    "amp_decay": "How quickly the volume falls from its peak towards the sustain level.",
    "amp_sustain": "The volume a held note settles at, as a share of its peak. 0 fades out completely, like a pluck; "
                   "1 stays at full volume, like an organ.",
    "amp_release": "How quickly a note fades out after it ends. Long values let notes ring on.",
    "cutoff": "Where the low-pass filter starts cutting harmonics. Lower sounds darker, higher sounds brighter. "
              "The filter envelope opens the filter upwards from here.",
    "resonance": "Boosts the harmonics around the cutoff. Low values roll off smoothly; high values add a sharp, "
                 "whistling peak, as in acid basslines.",
    "filt_env_amount": "How far the filter envelope opens the filter on each note, in octaves above the cutoff, "
                       "and less on quieter notes. 0 keeps the filter still; high values sweep it.",
    "glide_time": "How quickly a note settles on its pitch when it slides in from the previous note. "
                  "Only affects notes with a slide, which the Mono lead option fits.",
    "unison_cents": "Plays three copies of each note this far apart in pitch, for a thick, chorused sound "
                    "(100 cents is a semitone). Below 0.5 there is just one copy.",
    "filt_attack": "How long the filter envelope takes to reach its peak, opening the filter. "
                    "Short gives a sharp start; long gives a slow sweep upwards.",
    "filt_decay": "How quickly the filter envelope falls from its peak towards its sustain level, closing the filter. "
                  "Short gives a snappy pluck; long gives a slow sweep downwards.",
    "filt_sustain": "How far the filter stays open while a note is held, as a share of the envelope's peak.",
    "filt_release": "How quickly the filter envelope falls after a note ends, closing the filter.",
    "noise_level": "Unfiltered white noise added to each note and following its volume, for a breathy or rough edge.",
    "drive": "Saturation, like an overdriven amp. Low values are almost clean; high values squash loud parts "
             "and add grit.",
    "delay_time": "The time between echoes.",
    "delay_feedback": "How much of each echo carries into the next. 0 gives a single echo; "
                      "higher values give more repeats that fade more slowly.",
    "delay_mix": "How loud the echoes are, added on top of the original sound. 0 turns the delay off.",
    "reverb_rt60": "How long the reverb tail lasts: the time it takes to fade by 60 dB. Longer sounds like a bigger room.",
    "reverb_mix": "How loud the reverb is, added on top of the original sound. 0 turns the reverb off.",
    "gain": "This voice's overall volume, applied after everything else.",
}


class RevalidatedStaticFiles(StaticFiles):
    """Static files that browsers must check for changes before reusing a saved copy.

    Without this, a reload after an update can pair the new index.html with the old app.js.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


class DemoRequest(BaseModel):
    seed: int = 1
    length: float = Field(3.0, ge=1.0, le=8.0)
    notes: int = Field(10, ge=1, le=40)
    roll_steps: int = Field(400, ge=1, le=3000)
    note_steps: int = Field(300, ge=1, le=3000)
    sparsity: float = 0.5


class RenderRequest(BaseModel):
    patches: list[dict]
    notes: list[dict]
    muted_voices: list[int] = []
    mute_noise: bool = False  # leave out the fitted drums / noise floor


class MidiRequest(BaseModel):
    notes: list[dict]
    tempo_bpm: float = 120.0


class DownloadRequest(BaseModel):
    preset: str = ""  # a ready-made dataset, or:
    source: str = ""  # a key of downloads.SOURCES
    query: str = Field("", max_length=500)
    name: str = Field("", max_length=100)
    limit: int = Field(25, ge=1, le=500)


def create_app(jobs_dir: str | None = None, data_dir: str | None = None) -> FastAPI:
    jobs = JobManager(jobs_dir or os.environ.get("MUSICFIT_JOBS", os.path.join(os.getcwd(), "web_jobs")))
    library = Library(data_dir or os.environ.get("MUSICFIT_DATA", os.path.join(os.getcwd(), "data")))
    downloads = DownloadManager(library)
    app = FastAPI(title="musicfit")
    app.state.jobs = jobs
    app.state.downloads = downloads

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
            "params": {n: {"low": s.low, "high": s.high, "default": s.default, "log": s.log, "unit": UNITS.get(n, ""),
                           "help": PARAM_HELP[n]}
                       for n, s in PARAM_SPECS.items()},
            "groups": [{"title": t, "params": p} for t, p in PARAM_GROUPS],
            "waveforms": list(WAVEFORMS),
            "waveform_help": WAVEFORM_HELP,
            "max_duration_s": MAX_DURATION_S,
            "tools": available_tools(),
        }

    @app.get("/api/jobs")
    def list_jobs():
        return jobs.list()

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile | None = File(None),
        track: str = Form(""),
        offset: float = Form(0.0),
        duration: float = Form(4.0),
        voices: int = Form(1),
        roll_steps: int = Form(400),
        note_steps: int = Form(300),
        sparsity: float = Form(0.5),
        pitch_min: int = Form(36),
        pitch_max: int = Form(84),
        seed: int = Form(0),
        lead: bool = Form(False),
        separate: bool = Form(False),
        basic_pitch: bool = Form(False),
    ):
        if not 0.25 <= duration <= MAX_DURATION_S:
            raise HTTPException(422, f"duration must be between 0.25 and {MAX_DURATION_S} s")
        if not 1 <= voices <= 4 or not 0 <= pitch_min < pitch_max <= 127:
            raise HTTPException(422, "1-4 voices and 0 <= pitch_min < pitch_max <= 127")
        tools = available_tools()
        if (separate and not tools["demucs"]) or (basic_pitch and not tools["basic_pitch"]):
            raise HTTPException(422, "that tool is not installed; see requirements-tools.txt")
        cfg = dict(offset=max(offset, 0.0), duration=duration, voices=voices, roll_steps=max(roll_steps, 1),
                   note_steps=max(note_steps, 1), sparsity=sparsity, pitch_min=pitch_min, pitch_max=pitch_max,
                   seed=seed, lead=lead, separate=separate, basic_pitch=basic_pitch)
        if track:
            return jobs.submit_library(library.title(track), library.resolve(track), cfg).public()
        if file is None:
            raise HTTPException(422, "upload a file or choose a track")
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

    @app.get("/api/library")
    def list_library():
        return {"root": library.root, "datasets": library.datasets(), **downloads.state()}

    @app.get("/api/library/audio")
    def library_audio(track: str):
        try:
            return FileResponse(library.resolve(track))
        except ValueError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/library/downloads")
    def start_download(req: DownloadRequest):
        if req.preset:
            return downloads.submit_preset(req.preset)
        return downloads.submit(req.source, req.query, req.name, req.limit)

    @app.post("/api/library/downloads/{download_id}/cancel")
    def cancel_download(download_id: str):
        if not downloads.cancel(download_id):
            raise HTTPException(404, "no such download")
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

    app.mount("/", RevalidatedStaticFiles(directory=STATIC, html=True), name="static")
    return app


def __getattr__(name):
    # `uvicorn musicfit.web.app:app` -- created lazily so importing the module has no side effects
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
