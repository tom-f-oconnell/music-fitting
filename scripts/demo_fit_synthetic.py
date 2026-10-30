"""Sanity check: render a random melody with a random patch, then fit it from scratch.

    python scripts/demo_fit_synthetic.py --out demo_out

Writes target.wav, fit.wav, target.mid, fit.mid and report.json, and prints
note precision/recall/F1 and the recovered vs. true synth parameters.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from musicfit.audio_io import save_audio  # noqa: E402
from musicfit.fit import FitConfig, fit, note_metrics  # noqa: E402
from musicfit.notes import write_midi  # noqa: E402
from musicfit.synthetic import synthetic_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--length", type=float, default=3.0)
    ap.add_argument("--notes", type=int, default=10)
    ap.add_argument("--roll-steps", type=int, default=400)
    ap.add_argument("--note-steps", type=int, default=300)
    ap.add_argument("--device", default="auto", help="auto (CUDA if available), cpu, cuda, ...")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sr = 16000
    target, ref, true_params = synthetic_target(args.seed, args.length, args.notes, sr)

    cfg = FitConfig(sr=sr, roll_steps=args.roll_steps, note_steps=args.note_steps, seed=args.seed,
                    device=args.device)
    res = fit(target, cfg)

    roll_m = note_metrics(res.roll_notes, ref)
    final_m = note_metrics(res.notes, ref)
    report = {
        "true_params": true_params.describe(),
        "fitted_params": res.synths[0].params.describe(),
        "reference_notes": [n.as_dict() for n in ref],
        "fitted_notes": [n.as_dict() for n in res.notes],
        "metrics_after_roll_stage": roll_m,
        "metrics_final": final_m,
        "history": res.history,
    }
    save_audio(os.path.join(args.out, "target.wav"), target, sr)
    save_audio(os.path.join(args.out, "fit.wav"), res.audio, sr)
    write_midi(ref, os.path.join(args.out, "target.mid"))
    write_midi(res.notes, os.path.join(args.out, "fit.mid"))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nnotes after roll stage:", {k: round(v, 3) if isinstance(v, float) else v for k, v in roll_m.items()})
    print("notes after refinement:", {k: round(v, 3) if isinstance(v, float) else v for k, v in final_m.items()})
    print("\n%-16s %10s %10s" % ("param", "true", "fitted"))
    for k, v in report["true_params"].items():
        if k != "waveform":
            print("%-16s %10.4f %10.4f" % (k, v, report["fitted_params"][k]))
    print("waveform true  ", report["true_params"]["waveform"])
    print("waveform fitted", report["fitted_params"]["waveform"])


if __name__ == "__main__":
    main()
