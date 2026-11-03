"""Background fitting jobs, persisted to disk so results survive a server restart."""

from __future__ import annotations

import gc
import contextlib
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import soundfile as sf
import torch

from ..audio_io import load_audio, save_audio
from ..drums import load_drum_voice
from ..fit import FitConfig, fit, lead_settings, note_metrics, stage2_steps
from ..synthetic import synthetic_target
from ..tools import fit_with_tools


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    name: str
    kind: str  # "upload" | "library" | "demo"
    config: dict
    created: float = field(default_factory=time.time)
    started: float | None = None  # when the worker began the fit; None if it never did
    finished: float | None = None  # when it ended, however it ended
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = ""
    detail: str = ""  # e.g. which kit is being tried
    step: int = 0
    total: int = 0
    progress: float = 0.0  # 0..1 over both stages
    loss_curve: list = field(default_factory=list)  # [[global_step, loss], ...]
    error: str | None = None
    elapsed_s: float = 0.0
    result: dict | None = None

    def public(self, with_result: bool = True) -> dict:
        d = asdict(self)
        if not with_result:
            d.pop("result")
            d.pop("loss_curve")
        return d


class JobManager:
    """Runs fits one at a time on a worker thread (a fit already uses all cores)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._cancel: set[str] = set()
        self._deleted: set[str] = set()  # deleted while running: remove the files when the fit stops
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._load_existing()

    # -- persistence -------------------------------------------------------

    def dir(self, job_id: str) -> str:
        return os.path.join(self.root, job_id)

    def _save(self, job: Job) -> None:
        if job.id in self._deleted:
            return
        tmp = os.path.join(self.dir(job.id), "job.json.tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(job), f)
        os.replace(tmp, os.path.join(self.dir(job.id), "job.json"))

    def _load_existing(self) -> None:
        for name in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, name, "job.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    job = Job(**json.load(f))
            except Exception:
                continue
            if job.status in ("queued", "running"):
                job.status, job.error = "failed", "server restarted while the job was running"
            if job.finished is None:
                # saved before these times were kept, or cut off by a restart: a job's file is last
                # written as it ends, and elapsed_s counts from when it started
                job.finished = os.path.getmtime(path)
                if job.started is None and job.elapsed_s:
                    job.started = job.finished - job.elapsed_s
            self.jobs[job.id] = job

    # -- API -------------------------------------------------------------

    def list(self) -> list[dict]:
        return [j.public(with_result=False) for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def submit_upload(self, name: str, src_path: str, cfg: dict) -> Job:
        job = self._new(name, "upload", cfg)
        os.replace(src_path, os.path.join(self.dir(job.id), "upload" + os.path.splitext(name)[1]))
        self._pool.submit(self._run_and_release, job)
        return job

    def submit_library(self, name: str, path: str, cfg: dict) -> Job:
        """Fit a cached file, which stays where it is."""
        job = self._new(name, "library", {**cfg, "source_path": path})
        self._pool.submit(self._run_and_release, job)
        return job

    def submit_demo(self, cfg: dict) -> Job:
        job = self._new(f"synthetic demo (seed {cfg.get('seed', 1)})", "demo", cfg)
        self._pool.submit(self._run_and_release, job)
        return job

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self.jobs[job_id]
            if job.status == "queued":
                job.status, job.finished = "cancelled", time.time()
                self._save(job)
            elif job.status == "running":
                self._cancel.add(job_id)

    def delete(self, job_id: str) -> None:
        """Forget a job and remove its files. A running job is stopped, and its files go once it has."""
        with self._lock:
            job = self.jobs.pop(job_id)
            if job.status == "running":
                self._cancel.add(job_id)
                self._deleted.add(job_id)
                with contextlib.suppress(FileNotFoundError):  # so that a restart can't bring it back
                    os.remove(os.path.join(self.dir(job_id), "job.json"))
                return
            job.status = "cancelled"  # if it's still queued, the worker skips it
        shutil.rmtree(self.dir(job_id), ignore_errors=True)

    def _new(self, name: str, kind: str, cfg: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name, kind=kind, config=cfg)
        os.makedirs(self.dir(job.id))
        with self._lock:
            self.jobs[job.id] = job
        self._save(job)
        return job

    # -- worker ------------------------------------------------------------

    def _fit_config(self, c: dict) -> FitConfig:
        return FitConfig(
            sr=int(c.get("sr", 16000)), n_voices=0 if c.get("fit_mode") == "drums" else int(c.get("voices", 1)),
            drum_kinds=0 if c.get("fit_mode") == "notes" else 3, drum_kit=c.get("drum_kit", "808,909"),
            pitch_min=int(c.get("pitch_min", 36)), pitch_max=int(c.get("pitch_max", 84)),
            roll_steps=int(c.get("roll_steps", 400)), note_steps=int(c.get("note_steps", 300)),
            sparsity=float(c.get("sparsity", 0.5)), seed=int(c.get("seed", 0)),
            memory_mode="full" if c.get("full_memory") else "batched",
            **(lead_settings() if c.get("lead") else {}),
        )

    def _run_and_release(self, job: Job) -> None:
        self._run(job)
        # everything the fit used is gone by now: an idle server shouldn't sit on a shared GPU's memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run(self, job: Job) -> None:
        d = self.dir(job.id)
        c = job.config
        cfg = self._fit_config(c)
        t0 = time.time()
        with self._lock:
            if job.status == "cancelled":
                return
            job.status, job.started = "running", t0
            self._save(job)
        total_steps = cfg.roll_steps + stage2_steps(cfg) + 2
        c["total_steps"] = total_steps  # for the loss chart's x axis
        last_save = [0.0]

        def end(status: str, error: str | None = None) -> None:
            # the times first: a client that sees the job has ended must also see when
            job.finished = time.time()
            job.elapsed_s = round(job.finished - t0, 1)
            job.error = error
            job.status = status

        def progress(info: dict) -> None:
            if job.id in self._cancel:
                raise Cancelled()
            if info["stage"] in ("drums", "kits"):  # separate short fits, not on the loss chart
                job.stage, job.step, job.total = info["stage"], info["step"], info["total"]
                if info["stage"] == "kits":
                    job.detail = f"kit {info['kit_index'] + 1} of {info['n_kits']}: {info['kit']}"
                return
            offset = 0 if info["stage"] == "roll" else cfg.roll_steps + 1
            g = offset + info["step"]
            job.stage, job.step, job.total = info["stage"], info["step"], info["total"]
            job.progress = g / total_steps
            job.elapsed_s = round(time.time() - t0, 1)
            job.loss_curve.append([g, round(info["loss"], 5), info["stage"]])
            if time.time() - last_save[0] > 2:
                last_save[0] = time.time()
                self._save(job)

        try:
            reference = None
            if job.kind == "demo":
                target, ref, true_params = synthetic_target(
                    int(c.get("seed", 1)), float(c.get("length", 3.0)), int(c.get("notes", 10)), cfg.sr)
            else:
                source = c.get("source_path") or os.path.join(d, next(f for f in os.listdir(d) if f.startswith("upload")))
                target = load_audio(source, cfg.sr, float(c.get("offset", 0.0)), float(c.get("duration", 4.0)))
                if target.numel() < cfg.sr * 0.25:
                    raise ValueError("decoded clip is shorter than 0.25 s; check offset/duration")
                target = target / (target.abs().max() + 1e-8) * 0.5
            save_audio(os.path.join(d, "target.wav"), target, cfg.sr)

            if c.get("separate") or c.get("basic_pitch") or c.get("compare_kits"):
                res = fit_with_tools(target, cfg, bool(c.get("separate")), bool(c.get("basic_pitch")), verbose=False,
                                     progress=progress, compare_kits=cfg.drum_kit.split(",") if c.get("compare_kits") else None)
            else:
                res = fit(target, cfg, verbose=False, progress=progress)

            save_audio(os.path.join(d, "fit.wav"), res.audio, cfg.sr)
            if res.noise is not None:  # drums + noise floor, added to every re-render of the notes
                with torch.no_grad():
                    noise = res.noise.render(int(target.numel())).cpu()
                sf.write(os.path.join(d, "noise.wav"), noise.numpy(), cfg.sr, subtype="FLOAT")
                res.noise.cpu().save(os.path.join(d, "drums.pt"))  # to play edited hits later
            for i, k in enumerate(res.drum_kits):  # each kit's drums, to listen to
                save_audio(os.path.join(d, f"kit_{i}.wav"), k["audio"], cfg.sr)
            if job.kind == "demo":
                reference = {
                    "patch": true_params.describe(),
                    "notes": [n.as_dict() for n in ref],
                    "metrics_roll": note_metrics(res.roll_notes, ref),
                    "metrics_final": note_metrics(res.notes, ref),
                }
            job.result = {
                "sr": cfg.sr,
                "n_samples": int(target.numel()),
                "n_harmonics": cfg.n_harmonics,
                "patches": [s.params.describe() for s in res.synths],
                "notes": [n.as_dict() for n in res.notes],
                "roll_notes": [n.as_dict() for n in res.roll_notes],
                "final_loss": res.history[-1]["loss"],
                "tuning_cents": res.tuning_cents,
                "has_noise": res.noise is not None,
                "drum_hits": [[k, round(t, 3), round(v, 3)] for k, t, v in res.noise.hits()] if res.noise else [],
                "drums": res.noise.describe() if res.noise else [],  # name, General MIDI note and kit of each kind
                # kits compared, best first, with their fitted drums as kit_<i>.wav, and what they were fitted to
                "drum_kits": [{"kit": k["kit"], "loss": round(k["loss"], 5), "n_hits": k["n_hits"], "audio": f"kit_{i}"}
                              for i, k in enumerate(res.drum_kits)],
                "drum_kits_on": res.drum_kits_on,
                "history": res.history,
                "reference": reference,
            }
            job.progress = 1.0
            end("done")
        except Cancelled:
            end("cancelled")
        except Exception as e:  # surfaced in the UI
            end("failed", f"{type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            with self._lock:
                self._cancel.discard(job.id)
                if job.id in self._deleted:
                    self._deleted.discard(job.id)
                    shutil.rmtree(d, ignore_errors=True)
                else:
                    self._save(job)

    def target_audio(self, job_id: str) -> torch.Tensor:
        x, _ = sf.read(os.path.join(self.dir(job_id), "target.wav"), dtype="float32")
        return torch.from_numpy(x)

    def drum_voice(self, job_id: str):
        """The fitted drum voice, for playing edited hits; None for jobs from before it was saved."""
        path = os.path.join(self.dir(job_id), "drums.pt")
        return load_drum_voice(path) if os.path.exists(path) else None

    def noise_audio(self, job_id: str) -> torch.Tensor | None:
        path = os.path.join(self.dir(job_id), "noise.wav")
        if not os.path.exists(path):
            return None
        x, _ = sf.read(path, dtype="float32")
        return torch.from_numpy(x)
