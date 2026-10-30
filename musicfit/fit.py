"""Joint fitting of synth parameters and a note sequence to a target clip.

Stage 1 (discover notes): every voice gets a soft onset piano roll.  Roll and
synth patch are optimised together against a multi-resolution STFT loss plus
an L1 sparsity penalty on the roll.

Stage 2 (refine notes): each roll is peak-picked into a discrete note list
whose onsets, durations, velocities and fine pitch are continuous parameters.
Notes and patch are optimised together again, with an L1 penalty on note
velocities.  Then notes are pruned leave-one-out (each note has to lower the
loss by at least ``prune_tol`` to stay) and the survivors are refined briefly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from .loss import MultiResolutionSTFTLoss
from .notes import Note, NoteSequence, PianoRoll
from .synth import Synth, SynthParams


@dataclass
class FitConfig:
    sr: int = 16000
    n_voices: int = 1
    pitch_min: int = 36  # C2
    pitch_max: int = 84  # C6
    frame_hop_s: float = 0.01
    n_harmonics: int = 32
    # stage 1
    roll_steps: int = 400
    roll_lr: float = 0.1
    synth_lr: float = 0.03
    sparsity: float = 1.0  # L1 weight on roll mass per frame, reached after sparsity_warmup
    sparsity_warmup: float = 0.5  # fraction of roll_steps over which the L1 weight ramps up from 0
    decode_threshold: float = 0.3
    # stage 2
    note_steps: int = 300
    note_lrs: dict = field(default_factory=lambda: {"onset": 0.003, "duration": 0.03, "velocity": 0.05, "detune": 0.02})
    synth_lr_stage2: float = 0.01
    note_sparsity: float = 3.0  # L1 weight on note velocities (sum / n_frames, like the stage-1 penalty)
    prune_velocity: float = 0.05
    prune_tol: float = 0.01  # leave-one-out pruning: drop notes whose removal raises the loss by < this (0: off)
    polish_steps: int = 100  # refinement steps after leave-one-out pruning
    fft_sizes: tuple = (2048, 1024, 512, 256, 128)
    lr_decay: float = 0.1  # final/initial learning-rate ratio (linear decay within each stage)
    log_every: int = 50
    seed: int = 0
    device: str = "auto"  # "auto" (CUDA if available), "cpu", "cuda", "cuda:1", ...


@dataclass
class FitResult:
    synths: list[Synth]
    notes: list[Note]
    audio: torch.Tensor
    history: list[dict]
    roll_notes: list[Note]


def _initial_patch(voice: int) -> SynthParams:
    # different starting points break the symmetry between voices
    presets = [
        ({}, {"saw": 0.6, "square": 0.2, "sine": 0.1, "triangle": 0.1}),
        ({"cutoff": 600.0, "amp_decay": 0.8}, {"sine": 0.5, "triangle": 0.3, "saw": 0.1, "square": 0.1}),
        ({"cutoff": 3000.0, "amp_decay": 0.1, "amp_sustain": 0.2}, {"square": 0.6, "saw": 0.2, "sine": 0.1, "triangle": 0.1}),
    ]
    values, wave = presets[voice % len(presets)]
    return SynthParams(values, wave)


def stage2_steps(cfg: FitConfig) -> int:
    """Optimiser steps in stage 2, including the refinement after pruning."""
    return cfg.note_steps + (cfg.polish_steps + 1 if cfg.prune_tol > 0 else 0)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _linear_decay(opt, n_steps: int, final_ratio: float):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 - (1.0 - final_ratio) * min(i, n_steps) / max(n_steps, 1))


def _log(history, stage, step, loss, extra, t0, verbose):
    rec = {"stage": stage, "step": step, "loss": loss, **extra, "time_s": round(time.time() - t0, 1)}
    history.append(rec)
    if verbose:
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)


@torch.no_grad()
def _render_subset(notes: list[Note], synths: list[Synth], keep: list[bool], T: int, device) -> torch.Tensor:
    out = torch.zeros(T, device=device)
    for v, synth in enumerate(synths):
        vn = [n for n, k in zip(notes, keep) if k and n.voice == v]
        out = out + synth.render_notes(*(torch.tensor([getattr(n, a) for n in vn], device=device)
                                         for a in ("pitch", "onset", "duration", "velocity")), T)
    return out


@torch.no_grad()
def removal_deltas(notes: list[Note], synths: list[Synth], target: torch.Tensor, loss_fn) -> tuple[float, list[float]]:
    """Loss of all notes, and how much it rises when each note alone is removed."""
    T, dev = target.shape[-1], target.device
    keep = [True] * len(notes)
    base = float(loss_fn(_render_subset(notes, synths, keep, T, dev), target))
    deltas = []
    for i in range(len(notes)):
        keep[i] = False
        deltas.append(float(loss_fn(_render_subset(notes, synths, keep, T, dev), target)) - base)
        keep[i] = True
    return base, deltas


@torch.no_grad()
def _prune_redundant(notes: list[Note], synths: list[Synth], target: torch.Tensor, loss_fn, tol: float) -> list[Note]:
    """Greedily drop notes, quietest first, whose removal raises the loss by less than tol."""
    T, dev = target.shape[-1], target.device
    keep = [True] * len(notes)
    base = float(loss_fn(_render_subset(notes, synths, keep, T, dev), target))
    for i in sorted(range(len(notes)), key=lambda i: notes[i].velocity):
        keep[i] = False
        loss = float(loss_fn(_render_subset(notes, synths, keep, T, dev), target))
        if loss - base < tol:
            base = loss
        else:
            keep[i] = True
    return [n for n, k in zip(notes, keep) if k]


def fit(
    target: torch.Tensor,
    config: FitConfig | None = None,
    verbose: bool = True,
    progress: Callable[[dict], None] | None = None,
) -> FitResult:
    """Run both stages.  `progress`, if given, is called after every step with
    {"stage", "step", "total", "loss"}; raising from it aborts the fit.

    The fit runs on ``config.device``; the returned synths are left there, and
    the returned audio is on the CPU.
    """
    cfg = config or FitConfig()
    report = progress or (lambda info: None)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    target = target.float().to(device)
    T = target.shape[-1]
    n_frames = int(math.ceil(T / (cfg.frame_hop_s * cfg.sr)))
    loss_fn = MultiResolutionSTFTLoss(cfg.fft_sizes)
    history: list[dict] = []
    t0 = time.time()

    synths = [Synth(cfg.sr, cfg.n_harmonics, params=_initial_patch(v), noise_seed=v).to(device)
              for v in range(cfg.n_voices)]
    rolls = [
        PianoRoll(n_frames, cfg.pitch_min, cfg.pitch_max, cfg.frame_hop_s, generator=gen).to(device)
        for _ in range(cfg.n_voices)
    ]

    # ---- stage 1: soft piano roll -------------------------------------------
    opt = torch.optim.Adam(
        [{"params": list(r.parameters()), "lr": cfg.roll_lr} for r in rolls]
        + [{"params": list(s.params.parameters()), "lr": cfg.synth_lr} for s in synths]
    )
    sched = _linear_decay(opt, cfg.roll_steps, cfg.lr_decay)
    for step in range(cfg.roll_steps + 1):
        pred = sum(r.render(s, T) for r, s in zip(rolls, synths))
        spec = loss_fn(pred, target)
        sparse = sum(r.strengths().sum(0).mean() for r in rolls)  # mean active mass per frame
        ramp = min(1.0, step / max(1, cfg.sparsity_warmup * cfg.roll_steps))
        loss = spec + ramp * cfg.sparsity * sparse
        report({"stage": "roll", "step": step, "total": cfg.roll_steps, "loss": spec.item()})
        if step % cfg.log_every == 0 or step == cfg.roll_steps:
            _log(history, "roll", step, spec.item(), {"mass_per_frame": sparse.item()}, t0, verbose)
        if step == cfg.roll_steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    roll_notes = [n for v, r in enumerate(rolls) for n in r.decode(cfg.decode_threshold, voice=v)]
    if verbose:
        print(f"decoded {len(roll_notes)} notes from the piano roll", flush=True)

    # ---- stage 2: explicit notes ---------------------------------------------
    seqs = [NoteSequence([n for n in roll_notes if n.voice == v] or [Note(60.0, 0.0, 0.1, 0.02, v)]).to(device)
            for v in range(cfg.n_voices)]
    total = stage2_steps(cfg)

    def refine(seqs, n_steps, first_step, lr_scale=1.0):
        groups = [{"params": list(s.params.parameters()), "lr": cfg.synth_lr_stage2 * lr_scale} for s in synths]
        for q in seqs:
            groups += [
                {"params": [q.onset], "lr": cfg.note_lrs["onset"] * lr_scale},
                {"params": [q.log_duration], "lr": cfg.note_lrs["duration"] * lr_scale},
                {"params": [q.vel_logit], "lr": cfg.note_lrs["velocity"] * lr_scale},
                {"params": [q.detune_raw], "lr": cfg.note_lrs["detune"] * lr_scale},
            ]
        opt = torch.optim.Adam(groups)
        sched = _linear_decay(opt, n_steps, cfg.lr_decay)
        for i in range(n_steps + 1):
            step = first_step + i
            pred = sum(q.render(s, T) for q, s in zip(seqs, synths))
            loss = loss_fn(pred, target)
            report({"stage": "notes", "step": step, "total": total, "loss": loss.item()})
            if step % cfg.log_every == 0 or i == n_steps:
                _log(history, "notes", step, loss.item(), {"n_notes": sum(len(q) for q in seqs)}, t0, verbose)
            if i == n_steps:
                break
            if cfg.note_sparsity:
                loss = loss + cfg.note_sparsity * sum(q.velocity().sum() for q in seqs) / n_frames
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

    refine(seqs, cfg.note_steps, 0)
    notes = [n for q in seqs for n in q.to_notes(cfg.prune_velocity)]
    if cfg.prune_tol > 0:
        notes = _prune_redundant(notes, synths, target, loss_fn, cfg.prune_tol)
        if verbose:
            print(f"{len(notes)} notes left after leave-one-out pruning", flush=True)
        seqs = [NoteSequence([n for n in notes if n.voice == v]).to(device) for v in range(cfg.n_voices)]
        refine(seqs, cfg.polish_steps, cfg.note_steps + 1, lr_scale=0.3)
        notes = [n for q in seqs for n in q.to_notes(cfg.prune_velocity)]
    notes.sort(key=lambda n: (n.onset, n.pitch))
    with torch.no_grad():
        final_seqs = [NoteSequence([n for n in notes if n.voice == v]).to(device) for v in range(cfg.n_voices)]
        audio = sum(q.render(s, T) for q, s in zip(final_seqs, synths))
        final_loss = float(loss_fn(audio, target))
    _log(history, "final", 0, final_loss, {"n_notes": len(notes)}, t0, verbose)
    return FitResult(synths, notes, audio.detach().cpu(), history, roll_notes)


def match_notes(pred: list[Note], ref: list[Note], onset_tol: float = 0.05) -> dict[int, int]:
    """Greedy one-to-one matching on rounded pitch and onset within onset_tol seconds.

    Returns {reference index: predicted index} for the matched pairs.
    """
    matches: dict[int, int] = {}
    used = set()
    for j, r in enumerate(ref):
        best, best_dt = None, onset_tol
        for i, p in enumerate(pred):
            if i in used or round(p.pitch) != round(r.pitch):
                continue
            dt = abs(p.onset - r.onset)
            if dt <= best_dt:
                best, best_dt = i, dt
        if best is not None:
            used.add(best)
            matches[j] = best
    return matches


def note_metrics(pred: list[Note], ref: list[Note], onset_tol: float = 0.05) -> dict:
    """Precision/recall/F1 of predicted notes under :func:`match_notes`."""
    tp = len(match_notes(pred, ref, onset_tol))
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "n_pred": len(pred), "n_ref": len(ref), "precision": precision, "recall": recall, "f1": f1}
