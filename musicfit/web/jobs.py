"""Background fitting jobs, persisted to disk so results survive a server restart."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import soundfile as sf
import torch

from ..audio_io import load_audio, save_audio
from ..fit import FitConfig, fit, note_metrics, stage2_steps
from ..synthetic import synthetic_target


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    name: str
    kind: str  # "upload" | "demo"
    config: dict
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = ""
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
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._load_existing()

    # -- persistence -------------------------------------------------------

    def dir(self, job_id: str) -> str:
        return os.path.join(self.root, job_id)

    def _save(self, job: Job) -> None:
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
            self.jobs[job.id] = job

    # -- API -------------------------------------------------------------

    def list(self) -> list[dict]:
        return [j.public(with_result=False) for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def submit_upload(self, name: str, src_path: str, cfg: dict) -> Job:
        job = self._new(name, "upload", cfg)
        os.replace(src_path, os.path.join(self.dir(job.id), "upload" + os.path.splitext(name)[1]))
        self._pool.submit(self._run, job)
        return job

    def submit_demo(self, cfg: dict) -> Job:
        job = self._new(f"synthetic demo (seed {cfg.get('seed', 1)})", "demo", cfg)
        self._pool.submit(self._run, job)
        return job

    def cancel(self, job_id: str) -> None:
        job = self.jobs[job_id]
        if job.status == "queued":
            job.status = "cancelled"
            self._save(job)
        elif job.status == "running":
            self._cancel.add(job_id)

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
            sr=int(c.get("sr", 16000)), n_voices=int(c.get("voices", 1)),
            pitch_min=int(c.get("pitch_min", 36)), pitch_max=int(c.get("pitch_max", 84)),
            roll_steps=int(c.get("roll_steps", 400)), note_steps=int(c.get("note_steps", 300)),
            sparsity=float(c.get("sparsity", 1.0)), seed=int(c.get("seed", 0)),
        )

    def _run(self, job: Job) -> None:
        if job.status == "cancelled":
            return
        d = self.dir(job.id)
        c = job.config
        cfg = self._fit_config(c)
        job.status = "running"
        self._save(job)
        t0 = time.time()
        total_steps = cfg.roll_steps + stage2_steps(cfg) + 2
        c["total_steps"] = total_steps  # for the loss chart's x axis
        last_save = [0.0]

        def progress(info: dict) -> None:
            if job.id in self._cancel:
                raise Cancelled()
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
                upload = next(f for f in os.listdir(d) if f.startswith("upload"))
                target = load_audio(os.path.join(d, upload), cfg.sr, float(c.get("offset", 0.0)),
                                    float(c.get("duration", 4.0)))
                if target.numel() < cfg.sr * 0.25:
                    raise ValueError("decoded clip is shorter than 0.25 s; check offset/duration")
                target = target / (target.abs().max() + 1e-8) * 0.5
            save_audio(os.path.join(d, "target.wav"), target, cfg.sr)

            res = fit(target, cfg, verbose=False, progress=progress)

            save_audio(os.path.join(d, "fit.wav"), res.audio, cfg.sr)
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
                "history": res.history,
                "reference": reference,
            }
            job.status, job.progress = "done", 1.0
        except Cancelled:
            job.status = "cancelled"
        except Exception as e:  # surfaced in the UI
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            self._cancel.discard(job.id)
            job.elapsed_s = round(time.time() - t0, 1)
            self._save(job)

    def target_audio(self, job_id: str) -> torch.Tensor:
        x, _ = sf.read(os.path.join(self.dir(job_id), "target.wav"), dtype="float32")
        return torch.from_numpy(x)
