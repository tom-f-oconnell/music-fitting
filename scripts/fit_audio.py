"""Fit the synth + note sequence to a clip of a real recording.

    python scripts/fit_audio.py data/aphex_twin/audio/Aphex_Twin_Avril_14th.opus \
        --offset 10 --duration 4 --voices 2 --out fits/avril

Writes target.wav, fit.wav (all of it), noise.wav (the fitted drums and noise floor
alone), notes.mid, notes.json and params.json (patches, tuning, drum hits).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from musicfit.audio_io import load_audio, save_audio  # noqa: E402
from musicfit.fit import FitConfig, fit  # noqa: E402
from musicfit.notes import write_midi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=float, default=0.0, help="start of the clip in seconds")
    ap.add_argument("--duration", type=float, default=4.0, help="clip length in seconds")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--voices", type=int, default=1, help="number of synth voices (independent patches)")
    ap.add_argument("--pitch-min", type=int, default=36)
    ap.add_argument("--pitch-max", type=int, default=84)
    ap.add_argument("--roll-steps", type=int, default=400)
    ap.add_argument("--note-steps", type=int, default=300)
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drum-kinds", type=int, default=3, help="unpitched drum kinds to fit (0: none)")
    ap.add_argument("--noise-floor", action="store_true", help="also fit a stationary noise floor")
    ap.add_argument("--device", default="auto", help="auto (CUDA if available), cpu, cuda, ...")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    target = load_audio(args.audio, args.sr, args.offset, args.duration)
    target = target / (target.abs().max() + 1e-8) * 0.5
    cfg = FitConfig(sr=args.sr, n_voices=args.voices, pitch_min=args.pitch_min, pitch_max=args.pitch_max,
                    roll_steps=args.roll_steps, note_steps=args.note_steps, sparsity=args.sparsity, seed=args.seed,
                    drum_kinds=args.drum_kinds, noise_floor=args.noise_floor, device=args.device)
    res = fit(target, cfg)

    save_audio(os.path.join(args.out, "target.wav"), target, args.sr)
    save_audio(os.path.join(args.out, "fit.wav"), res.audio, args.sr)
    if res.noise is not None:
        with torch.no_grad():
            save_audio(os.path.join(args.out, "noise.wav"), res.noise.render(target.numel()), args.sr)
    write_midi(res.notes, os.path.join(args.out, "notes.mid"))
    with open(os.path.join(args.out, "notes.json"), "w") as f:
        json.dump([n.as_dict() for n in res.notes], f, indent=2)
    with open(os.path.join(args.out, "params.json"), "w") as f:
        hits = res.noise.hits() if res.noise is not None else []
        json.dump({"voices": [s.params.describe() for s in res.synths], "tuning_cents": res.tuning_cents,
                   "drum_hits": [{"kind": k, "time": round(t, 3), "strength": round(v, 3)} for k, t, v in hits],
                   "history": res.history}, f, indent=2)
    print(f"wrote {len(res.notes)} notes and {len(res.synths)} patches to {args.out}")


if __name__ == "__main__":
    main()
