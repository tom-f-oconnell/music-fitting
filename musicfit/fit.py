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

import gc
import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import torch

from .batching import backward_in_batches, render_mix
from .drumkit import DEFAULT_KITS, make_kit
from .drums import KitVoice, NoiseVoice
from .loss import MultiResolutionSTFTLoss
from .notes import Note, NoteSequence, PianoRoll
from .synth import Synth, SynthParams
from .tuning import estimate_tuning


@dataclass
class FitConfig:
    sr: int = 16000
    n_voices: int = 1
    pitch_min: int = 36  # C2
    pitch_max: int = 84  # C6
    frame_hop_s: float = 0.01
    n_harmonics: int = 32
    # offset of the piano-roll pitch grid from A440 in cents, or "auto" to estimate it from
    # the target (used if the estimate's consistency is at least tuning_min_confidence, else 0)
    tuning_cents: float | str = "auto"
    tuning_min_confidence: float = 0.2
    # detuned unison: candidate spreads in cents, searched during the fit (loss.unison_spectrum); empty: off
    spread_grid: tuple = (0, 5, 10, 15, 25, 40)
    # stage 1
    roll_steps: int = 400
    roll_lr: float = 0.1
    synth_lr: float = 0.03
    sparsity: float = 0.5  # L1 weight on roll mass per frame, reached after sparsity_warmup
    sparsity_warmup: float = 0.5  # fraction of roll_steps over which the L1 weight ramps up from 0
    decode_threshold: float = 0.3
    # unpitched sounds (drums.py): drum hits from an onset roll, and a noise floor
    drum_kinds: int = 3  # 0: no drums; with drum_kit "noise", how many kinds of its own to make
    # kits to fit the drums against, comma-separated: built-in, downloaded or folders of samples (drumkit.py);
    # each drum keeps whichever fits it best. "noise": generic noise bursts instead (the drum voice before kits)
    drum_kit: str = DEFAULT_KITS
    noise_floor: bool = False
    # multiplies `sparsity` for the drum roll (higher: fewer, stronger hits). Alongside notes, cheap drums take
    # over pitched sound (design.md); fitted alone (no voices, as in a Demucs drum stem) they can be cheaper
    drum_sparsity: float = 1.0
    drum_sparsity_alone: float = 0.2
    drum_threshold: float = 0.3  # how strong a peak of the drum roll has to be to count as a hit
    drum_start: float = 0.25  # fraction of roll_steps before the drums join (pitched content first; see design.md)
    # stage 2; a faster onset rate drifts notes off the onsets stage 1 found (design.md)
    note_steps: int = 300
    note_lrs: dict = field(default_factory=lambda: {"onset": 0.001, "duration": 0.03, "velocity": 0.05, "detune": 0.02,
                                                    "slide": 0.05})
    glides: bool = False  # fit a portamento from the previous note per note (for mono leads)
    synth_lr_stage2: float = 0.01
    note_sparsity: float = 3.0  # L1 weight on note velocities (sum / n_frames, like the stage-1 penalty)
    prune_velocity: float = 0.05
    prune_tol: float = 0.01  # leave-one-out pruning: drop notes whose removal raises the loss by < this (0: off)
    prune_reference_s: float = 15.0  # scale prune_tol down for clips longer than this (0: never), see _prune_tolerance
    polish_steps: int = 100  # refinement steps after leave-one-out pruning
    fft_sizes: tuple = (2048, 1024, 512, 256, 128)
    lr_decay: float = 0.1  # final/initial learning-rate ratio (linear decay within each stage)
    log_every: int = 50
    seed: int = 0
    device: str = "auto"  # "auto" (CUDA if available), "cpu", "cuda", "cuda:1", ...
    # "batched": render a few notes or pitch rows at a time, and notes only where they sound, which takes
    # a fraction of the memory for the same gradients (design.md); "full": render everything at once
    memory_mode: str = "batched"
    roll_batch: int = 0  # pitch rows per batch in stage 1; 0: by clip length, to hold memory steady
    note_batch: int = 16  # notes per batch in stage 2, once a clip is too long to render its notes in one
    sin_cache_gb: float = 2.0  # sinusoid tables kept between steps of stage 1, which saves recomputing them


@dataclass
class FitResult:
    synths: list[Synth]
    notes: list[Note]
    audio: torch.Tensor
    history: list[dict]
    roll_notes: list[Note]
    tuning_cents: float = 0.0  # pitch-grid offset used in stage 1
    noise: NoiseVoice | None = None  # drums / noise floor, if fitted (its render is part of audio)
    spread_cents: float = 0.0  # detuned-unison spread found by the search (cfg.spread_grid)
    # kits compared by tools.fit_with_tools, best first: {"kit", "loss", "n_hits", "audio"}, and what they were fitted to
    drum_kits: list[dict] = field(default_factory=list)
    drum_kits_on: str = ""


def lead_settings() -> dict:
    """FitConfig fields for a mono lead or acid line: a portamento per note, and enough
    harmonics that a bright low note isn't cut off at 2 kHz (design.md)."""
    return {"glides": True, "n_harmonics": 128}


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


# What fits in about 3 GB on the GPU. Short clips stay in one batch, which is ordinary backprop: on a GPU,
# many small batches cost far more time than their size suggests (at 4 s, 8 rows at a time takes 3.5 times as long).
ROW_SECONDS_PER_BATCH = 200
NOTE_SECONDS_PER_BATCH = 200


def _roll_batch(cfg: FitConfig, n_samples: int) -> int:
    rows = cfg.pitch_max - cfg.pitch_min + 1
    return min(rows, cfg.roll_batch or max(1, int(ROW_SECONDS_PER_BATCH * cfg.sr / n_samples)))


def _note_batch(cfg: FitConfig, n_notes: int, n_samples: int) -> int:
    return max(n_notes, 1) if n_notes * n_samples / cfg.sr <= NOTE_SECONDS_PER_BATCH else cfg.note_batch


def _prune_tolerance(cfg: FitConfig, n_samples: int) -> float:
    """How much removing a note must raise the loss for the note to stay.

    The loss is an average over the clip, so beyond prune_reference_s a note isn't held to a higher bar
    just because it is a smaller part of the whole (design.md).
    """
    seconds = n_samples / cfg.sr
    return cfg.prune_tol * min(1.0, cfg.prune_reference_s / seconds) if cfg.prune_reference_s else cfg.prune_tol


def _noise_groups(noise: NoiseVoice | None, roll_lr: float, param_lr: float) -> list[dict]:
    if noise is None:
        return []
    other = [p for name, p in noise.named_parameters() if name != "logits"]
    return [{"params": [noise.logits], "lr": roll_lr}, {"params": other, "lr": param_lr}]


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
def _render_subset(notes: list[Note], synths: list[Synth], keep: list[bool], T: int, device,
                   background: torch.Tensor | None = None) -> torch.Tensor:
    out = torch.zeros(T, device=device) if background is None else background.clone()
    for v, synth in enumerate(synths):
        vn = [n for n, k in zip(notes, keep) if k and n.voice == v]
        glide = torch.tensor([n.glide for n in vn], device=device) if any(n.glide for n in vn) else None
        out = out + synth.render_notes(*(torch.tensor([getattr(n, a) for n in vn], device=device)
                                         for a in ("pitch", "onset", "duration", "velocity")), T, glide)
    return out


@torch.no_grad()
def removal_deltas(notes: list[Note], synths: list[Synth], target: torch.Tensor, loss_fn,
                   background: torch.Tensor | None = None) -> tuple[float, list[float]]:
    """Loss of all notes (on top of background), and how much it rises when each note alone is removed."""
    T, dev = target.shape[-1], target.device
    keep = [True] * len(notes)
    base = float(loss_fn(_render_subset(notes, synths, keep, T, dev, background), target))
    deltas = []
    for i in range(len(notes)):
        keep[i] = False
        deltas.append(float(loss_fn(_render_subset(notes, synths, keep, T, dev, background), target)) - base)
        keep[i] = True
    return base, deltas


@torch.no_grad()
def _prune_redundant(notes: list[Note], synths: list[Synth], target: torch.Tensor, loss_fn, tol: float,
                     background: torch.Tensor | None = None) -> list[Note]:
    """Greedily drop notes, quietest first, whose removal raises the loss by less than tol."""
    T, dev = target.shape[-1], target.device
    keep = [True] * len(notes)
    base = float(loss_fn(_render_subset(notes, synths, keep, T, dev, background), target))
    for i in sorted(range(len(notes)), key=lambda i: notes[i].velocity):
        keep[i] = False
        loss = float(loss_fn(_render_subset(notes, synths, keep, T, dev, background), target))
        if loss - base < tol:
            base = loss
        else:
            keep[i] = True
    return [n for n, k in zip(notes, keep) if k]


@torch.no_grad()
def _prune_by_subtraction(notes: list[Note], synths: list[Synth], target: torch.Tensor, loss_fn, tol: float,
                          background: torch.Tensor | None = None) -> list[Note]:
    """As _prune_redundant, rendering each note once and trying the mix without it, not every note every time."""
    T, dev = target.shape[-1], target.device
    extra = 0.0 if background is None else background
    mixes = [torch.zeros(T, device=dev) for _ in synths]
    parts = []
    for n in notes:
        y, start = synths[n.voice].render_notes_window(
            *(torch.tensor([getattr(n, a)], device=dev) for a in ("pitch", "onset", "duration", "velocity")), T,
            torch.tensor([n.glide], device=dev) if n.glide else None)
        mixes[n.voice][start:start + y.numel()] += y
        parts.append((y, start))
    wet = [s.effects(s.params.values(), m) for s, m in zip(synths, mixes)]
    base = float(loss_fn(sum(wet) + extra, target))
    keep = [True] * len(notes)
    for i in sorted(range(len(notes)), key=lambda i: notes[i].velocity):
        v = notes[i].voice
        y, start = parts[i]
        mix = mixes[v].clone()
        mix[start:start + y.numel()] -= y
        trial = synths[v].effects(synths[v].params.values(), mix)
        loss = float(loss_fn(sum(trial if j == v else w for j, w in enumerate(wet)) + extra, target))
        if loss - base < tol:
            base, keep[i], mixes[v], wet[v] = loss, False, mix, trial
    return [n for n, k in zip(notes, keep) if k]


def fit(
    target: torch.Tensor,
    config: FitConfig | None = None,
    verbose: bool = True,
    progress: Callable[[dict], None] | None = None,
    init_notes: list[Note] | None = None,
) -> FitResult:
    """Run both stages.  `progress`, if given, is called after every step with
    {"stage", "step", "total", "loss"}; raising from it aborts the fit.

    The fit runs on ``config.device``; the returned synths are left there, and
    the returned audio is on the CPU.

    With ``init_notes`` (e.g. from a transcriber) there is no stage 1: the patch
    and drums are fitted to those notes for ``roll_steps // 4`` steps, then stage
    2 starts from them.  ``n_voices = 0`` fits only the drums.
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
    if cfg.tuning_cents == "auto":
        cents, confidence = estimate_tuning(target, cfg.sr)
        tuning = cents if confidence >= cfg.tuning_min_confidence else 0.0
        if verbose:
            print(f"tuning: {cents:+.1f} cents (consistency {confidence:.2f}), using {tuning:+.1f}", flush=True)
    else:
        tuning = float(cfg.tuning_cents)
    rolls = [
        PianoRoll(n_frames, cfg.pitch_min, cfg.pitch_max, cfg.frame_hop_s, generator=gen,
                  tuning_cents=tuning).to(device)
        for _ in range(cfg.n_voices)
    ]
    noise = None
    if cfg.drum_kinds and cfg.drum_kit != "noise":
        noise = KitVoice(n_frames, cfg.frame_hop_s, cfg.sr, make_kit(str(cfg.drum_kit), cfg.sr), noise_floor=cfg.noise_floor,
                         generator=gen, seed=cfg.seed).to(device)
    elif cfg.drum_kinds or cfg.noise_floor:
        noise = NoiseVoice(n_frames, cfg.frame_hop_s, cfg.sr, cfg.drum_kinds, noise_floor=cfg.noise_floor,
                           generator=gen, seed=cfg.seed).to(device)
    render_noise = (lambda: noise.render(T)) if noise is not None else (lambda: 0.0)
    drum_cost = cfg.drum_sparsity if cfg.n_voices else cfg.drum_sparsity_alone
    noise_mass = (lambda: drum_cost * noise.strengths().sum(0).mean()) if noise is not None else (lambda: 0.0)

    spread = 0.0
    spread_steps = {cfg.roll_steps // 4, cfg.roll_steps // 2, 3 * cfg.roll_steps // 4}

    batched = cfg.memory_mode == "batched"
    if not batched and cfg.memory_mode != "full":
        raise ValueError(f"memory_mode must be batched or full, not {cfg.memory_mode}")
    batch_size = {"roll": _roll_batch(cfg, T)}  # the notes' size is worked out from the notes, until memory runs out
    if batched:
        for s in synths:
            s.sin_cache_limit = cfg.sin_cache_gb * 1e9

    def evaluate(models, kind, render_extra, objective, opt=None) -> float:
        """The value of objective(pred) -> (loss to minimise, spectral loss), and with opt its gradients."""
        if not batched:
            loss, spec = objective(sum(m.render(s, T) for m, s in zip(models, synths)) + render_extra())
            if opt is not None:
                opt.zero_grad()
                loss.backward()
            return spec.item()

        spectral = []

        def loss_of_mixes(mixes):
            pred = sum(s.effects(s.params.values(), mix) for s, mix in zip(synths, mixes)) + render_extra()
            loss, spec = objective(pred)
            spectral.append(spec.item())
            return loss

        while True:
            spectral.clear()
            try:
                voices = [m.batches(s, T, batch_size.get(kind) or _note_batch(cfg, len(m), T)) for m, s in zip(models, synths)]
                if opt is None:
                    with torch.no_grad():
                        loss_of_mixes([render_mix(b, T, device) for b in voices])
                else:
                    opt.zero_grad()
                    backward_in_batches(voices, T, device, loss_of_mixes)
                return spectral[-1]
            except torch.cuda.OutOfMemoryError:
                current = batch_size.get(kind) or max(_note_batch(cfg, len(m), T) for m in models)
                if current == 1:
                    raise
            # outside the handler, so that the failed attempt's tensors are gone
            batch_size[kind] = current // 2
            for s in synths:
                s._sin_cache.clear()
                s.sin_cache_limit /= 2
            gc.collect()
            torch.cuda.empty_cache()
            if verbose:
                print(f"out of GPU memory: now rendering {batch_size[kind]} at a time in {kind}", flush=True)

    # ---- stage 1: soft piano roll -------------------------------------------
    if init_notes is None:
        opt = torch.optim.Adam(
            [{"params": list(r.parameters()), "lr": cfg.roll_lr} for r in rolls]
            + [{"params": list(s.params.parameters()), "lr": cfg.synth_lr} for s in synths]
            + _noise_groups(noise, cfg.roll_lr, cfg.synth_lr)
        )
        sched = _linear_decay(opt, cfg.roll_steps, cfg.lr_decay)
        for step in range(cfg.roll_steps + 1):
            # with no voices, the drums are all there is to fit from the start
            drums_on = noise is not None and (not rolls or step >= cfg.drum_start * cfg.roll_steps)
            ramp = min(1.0, step / max(1, cfg.sparsity_warmup * cfg.roll_steps))
            mass = []

            def objective(pred):
                nonlocal spread
                if cfg.spread_grid and step in spread_steps:
                    spread = loss_fn.best_spread(pred.detach(), target, cfg.spread_grid)
                spec = loss_fn(pred, target, spread)
                sparse = sum((r.strengths().sum(0).mean() for r in rolls), torch.zeros((), device=device))  # mass per frame
                loss = spec + ramp * cfg.sparsity * sparse
                if drums_on:
                    drum_mass = noise_mass()
                    sparse = sparse + drum_mass
                    loss = loss + ramp * cfg.sparsity * drum_mass
                mass.append(sparse.item())
                return loss, spec

            spec = evaluate(rolls, "roll", render_noise if drums_on else (lambda: 0.0), objective,
                            opt if step < cfg.roll_steps else None)
            report({"stage": "roll", "step": step, "total": cfg.roll_steps, "loss": spec})
            if step % cfg.log_every == 0 or step == cfg.roll_steps:
                _log(history, "roll", step, spec, {"mass_per_frame": mass[-1]}, t0, verbose)
            if step == cfg.roll_steps:
                break
            opt.step()
            sched.step()

        for s in synths:
            s._sin_cache.clear()
        roll_notes = [n for v, r in enumerate(rolls) for n in r.decode(cfg.decode_threshold, voice=v)]
        if verbose:
            print(f"decoded {len(roll_notes)} notes from the piano roll", flush=True)
    else:
        roll_notes = [replace(n, pitch=n.pitch + tuning / 100, voice=min(n.voice, cfg.n_voices - 1))
                      for n in init_notes]

    # ---- stage 2: explicit notes ---------------------------------------------
    make_seq = lambda notes: NoteSequence(notes, glides=cfg.glides).to(device)  # noqa: E731
    seqs = [make_seq([n for n in roll_notes if n.voice == v] or [Note(60.0, 0.0, 0.1, 0.02, v)])
            for v in range(cfg.n_voices)]
    total = stage2_steps(cfg)

    def refine(seqs, n_steps, first_step, lr_scale=1.0, notes_frozen=False):
        nonlocal spread
        groups = [{"params": list(s.params.parameters()), "lr": cfg.synth_lr_stage2 * lr_scale} for s in synths]
        groups += _noise_groups(noise, 0.1 * cfg.roll_lr * lr_scale, cfg.synth_lr_stage2 * lr_scale)
        for q in seqs if not notes_frozen else ():
            groups += [
                {"params": [q.onset], "lr": cfg.note_lrs["onset"] * lr_scale},
                {"params": [q.log_duration], "lr": cfg.note_lrs["duration"] * lr_scale},
                {"params": [q.vel_logit], "lr": cfg.note_lrs["velocity"] * lr_scale},
                {"params": [q.detune_raw], "lr": cfg.note_lrs["detune"] * lr_scale},
            ]
            if q.slide_logit is not None:
                groups.append({"params": [q.slide_logit], "lr": cfg.note_lrs["slide"] * lr_scale})
        opt = torch.optim.Adam(groups)
        sched = _linear_decay(opt, n_steps, cfg.lr_decay)
        for i in range(n_steps + 1):
            step = first_step + i

            def objective(pred):
                nonlocal spread
                if cfg.spread_grid and i == 0:
                    spread = loss_fn.best_spread(pred.detach(), target, cfg.spread_grid)
                spec = loss_fn(pred, target, spread)
                loss = spec
                if cfg.note_sparsity:
                    loss = loss + cfg.note_sparsity * sum(q.velocity().sum() for q in seqs) / n_frames
                return loss + cfg.sparsity * noise_mass(), spec

            spec = evaluate(seqs, "notes", render_noise, objective, opt if i < n_steps else None)
            # the warm-up with frozen notes stands in for stage 1, on the progress chart too
            report({"stage": "roll" if notes_frozen else "notes", "step": step, "total": n_steps if notes_frozen else total,
                    "loss": spec})
            if step % cfg.log_every == 0 or i == n_steps:
                _log(history, "roll" if notes_frozen else "notes", step, spec,
                     {"n_notes": sum(len(q) for q in seqs)}, t0, verbose)
            if i == n_steps:
                break
            opt.step()
            sched.step()

    if init_notes is not None:
        refine(seqs, cfg.roll_steps // 4, 0, lr_scale=cfg.synth_lr / cfg.synth_lr_stage2, notes_frozen=True)
    if isinstance(noise, KitVoice):
        noise.commit()  # each drum keeps one kit's sound from here on
    if noise is not None:
        noise.hit_threshold = cfg.drum_threshold
    refine(seqs, cfg.note_steps, 0)
    notes = [n for q in seqs for n in q.to_notes(cfg.prune_velocity)]
    if cfg.prune_tol > 0:
        with torch.no_grad():
            background = render_noise() if noise is not None else None
        prune = _prune_by_subtraction if batched else _prune_redundant
        notes = prune(notes, synths, target, lambda a, b: loss_fn(a, b, spread), _prune_tolerance(cfg, T), background)
        if verbose:
            print(f"{len(notes)} notes left after leave-one-out pruning", flush=True)
        seqs = [make_seq([n for n in notes if n.voice == v]) for v in range(cfg.n_voices)]
        refine(seqs, cfg.polish_steps, cfg.note_steps + 1, lr_scale=0.3)
        notes = [n for q in seqs for n in q.to_notes(cfg.prune_velocity)]
    notes.sort(key=lambda n: (n.onset, n.pitch))

    @torch.no_grad()
    def render_final():
        final_seqs = [make_seq([n for n in notes if n.voice == v]) for v in range(cfg.n_voices)]
        # after running out of memory, the batches stay small
        render_one = (lambda q, s: q.render_windowed(s, T, batch_size.get("notes", 16))) if batched else (lambda q, s: q.render(s, T))
        audio = sum(render_one(q, s) for q, s in zip(final_seqs, synths)) + render_noise()
        return audio, float(loss_fn(audio, target))

    audio, final_loss = render_final()
    extra = {"n_notes": len(notes)}
    if spread:
        # The search compared expected spectra. Real copies beat, so keep them only if they fit
        # better as they are; the wider spectrum may just have made finding notes easier.
        for s in synths:
            s.params.set_value("unison_cents", spread)
        unison_audio, unison_loss = render_final()
        extra["loss_one_copy"], extra["spread_searched"] = final_loss, spread
        if unison_loss < final_loss:
            audio, final_loss = unison_audio, unison_loss
        else:
            spread = 0.0
            for s in synths:
                s.params.set_value("unison_cents", 0.0)
    _log(history, "final", 0, final_loss, extra, t0, verbose)
    return FitResult(synths, notes, audio.detach().cpu(), history, roll_notes, tuning, noise, spread)


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
