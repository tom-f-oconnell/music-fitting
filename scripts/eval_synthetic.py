"""Fit synthetic targets for many seeds and report note precision/recall/F1.

Single fits say little about whether a change helps (GPU fits are not
deterministic and the optimisation is chaotic; see design.md), so compare
settings on many targets, paired by target:

    python scripts/eval_synthetic.py --seeds 1-24 --out eval_out/defaults.json
    python scripts/eval_synthetic.py --seeds 1-24 --set prune_tol=0.005 --out eval_out/tol005.json \\
        --compare eval_out/defaults.json --kinds

--set overrides a FitConfig field; the value is parsed as JSON (--set sparsity=0.5,
--set roll_steps=150). --mismatch adds something the synth can't produce to every
target (tuning/unison in cents, noise/drums in dB re the peak, codec in kbit/s; see
musicfit.synthetic.synthetic_target), e.g. --mismatch tuning=30 --mismatch noise=-40.
Seed s renders synthetic target s (same melody and patch with or without
mismatch) and also seeds the fit.
Targets are fitted in parallel processes (--jobs: default 5 on a GPU, 1 on a CPU,
where one fit already uses every core). 24 targets at the default settings take
about 5 minutes on an RTX 3090.

Prints precision/recall/F1 per target and on average, for the notes decoded
from the piano roll (stage 1) and the final notes. --compare adds the paired
mean F1 difference against an earlier --out file, --kinds a breakdown of the
spurious notes, and --out saves everything incl. reference, decoded and final
notes and the true and fitted patches.
"""

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from musicfit.fit import FitConfig, fit, match_notes, note_metrics  # noqa: E402
from musicfit.notes import Note  # noqa: E402
from musicfit.synthetic import synthetic_target  # noqa: E402


def fit_target(args) -> dict:
    seed, overrides, mismatch, length, n_notes, threads = args
    torch.set_num_threads(threads)
    cfg = FitConfig(seed=seed, **overrides)
    target, ref, true_patch = synthetic_target(seed, length, n_notes, cfg.sr, mismatch)
    t0 = time.time()
    res = fit(target, cfg, verbose=False)
    return {
        "seed": seed,
        "time_s": round(time.time() - t0, 1),
        "final_loss": res.history[-1]["loss"],
        **note_metrics(res.notes, ref),
        "roll_metrics": note_metrics(res.roll_notes, ref),
        "ref_notes": [n.as_dict() for n in ref],
        "roll_notes": [n.as_dict() for n in res.roll_notes],
        "notes": [n.as_dict() for n in res.notes],
        "true_patch": true_patch.describe(),
        "fitted_patches": [s.params.describe() for s in res.synths],
    }


def classify_false_positives(ref: list[Note], pred: list[Note], true_patch: dict, onset_tol: float = 0.05) -> Counter:
    """Count unmatched predicted notes by the first description that fits.

    double: same pitch and onset as a reference note that already has a match.
    echo: same pitch as a reference note, 1-3 true delay times later (±30 ms).
    harmonic: at a reference onset, 5/7/12/19/24 semitones away.
    timing: same pitch as a reference note, onset 50-150 ms off.
    pitch+-2: at a reference onset, 1-2 semitones away.
    inside_note: same pitch, during a reference note or its release.
    other: none of these.
    """
    matched = set(match_notes(pred, ref, onset_tol).values())
    delay, release = true_patch["delay_time"], true_patch["amp_release"]
    kinds: Counter = Counter()
    for i, p in enumerate(pred):
        if i in matched:
            continue
        pitch, t = round(p.pitch), p.onset
        same = [q for q in ref if round(q.pitch) == pitch]
        near = [q for q in ref if abs(t - q.onset) <= onset_tol]
        if any(abs(t - q.onset) <= onset_tol for q in same):
            kind = "double"
        elif any(abs(t - q.onset - k * delay) < 0.03 for q in same for k in (1, 2, 3)):
            kind = "echo"
        elif any(abs(pitch - round(q.pitch)) in (5, 7, 12, 19, 24) for q in near):
            kind = "harmonic"
        elif any(abs(t - q.onset) <= 0.15 for q in same):
            kind = "timing"
        elif any(abs(pitch - round(q.pitch)) <= 2 for q in near):
            kind = "pitch+-2"
        elif any(q.onset < t < q.onset + q.duration + release for q in same):
            kind = "inside_note"
        else:
            kind = "other"
        kinds[kind] += 1
    return kinds


def _notes(items: list[dict]) -> list[Note]:
    return [Note(**d) for d in items]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="1-24", help="a range like 1-24, or one seed")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE", help="override a FitConfig field")
    ap.add_argument("--mismatch", action="append", default=[], metavar="KIND=VALUE",
                    help="add tuning/unison/noise/drums/codec to the targets")
    ap.add_argument("--length", type=float, default=3.0, help="target length in seconds")
    ap.add_argument("--notes", type=int, default=10, help="notes per target (plus dyads)")
    ap.add_argument("--jobs", type=int, default=None, help="parallel fits (default 5 on a GPU, else 1)")
    ap.add_argument("--out", help="save all results as JSON")
    ap.add_argument("--compare", help="earlier --out file to compare F1 against, paired by seed")
    ap.add_argument("--kinds", action="store_true", help="break down the spurious notes")
    args = ap.parse_args()

    lo, _, hi = args.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    mismatch = {k: float(v) for k, v in (s.split("=", 1) for s in args.mismatch)}
    on_gpu = torch.cuda.is_available() and overrides.get("device", "auto") != "cpu"
    jobs = args.jobs or (5 if on_gpu else 1)
    threads = max(1, (os.cpu_count() or 1) // jobs)
    tasks = [(s, overrides, mismatch, args.length, args.notes, threads) for s in seeds]

    rows = []
    with mp.get_context("spawn").Pool(jobs) as pool:
        for r in pool.imap(fit_target, tasks):
            rows.append(r)
            rm = r["roll_metrics"]
            print(f"seed {r['seed']:3d}: P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.2f} "
                  f"({r['tp']}/{r['n_ref']} found, {r['n_pred']} predicted; stage 1 F1 {rm['f1']:.2f}) "
                  f"loss {r['final_loss']:.3f}, {r['time_s']} s", flush=True)

    mean = {k: statistics.mean(r[k] for r in rows) for k in ("precision", "recall", "f1", "final_loss")}
    mean["roll_f1"] = statistics.mean(r["roll_metrics"]["f1"] for r in rows)
    print(f"\nmean over {len(rows)} targets: P {mean['precision']:.3f} R {mean['recall']:.3f} F1 {mean['f1']:.3f} "
          f"(stage 1 F1 {mean['roll_f1']:.3f}), loss {mean['final_loss']:.3f}; "
          f"{sum(r['n_pred'] for r in rows)} notes predicted, {sum(r['n_ref'] for r in rows)} real")

    if args.kinds:
        kinds = sum((classify_false_positives(_notes(r["ref_notes"]), _notes(r["notes"]), r["true_patch"])
                     for r in rows), Counter())
        print(f"spurious notes: {sum(kinds.values())}", dict(kinds.most_common()))

    if args.compare:
        before = {r["seed"]: r for r in json.load(open(args.compare))["rows"]}
        diff = [r["f1"] - before[r["seed"]]["f1"] for r in rows if r["seed"] in before]
        if len(diff) >= 2:
            se = statistics.stdev(diff) / len(diff) ** 0.5
            print(f"F1 vs {args.compare}: {statistics.mean(diff):+.3f} ± {se:.3f} (SE, {len(diff)} paired targets), "
                  f"better on {sum(d > 0 for d in diff)}, worse on {sum(d < 0 for d in diff)}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"overrides": overrides, "mismatch": mismatch, "length": args.length, "notes": args.notes,
                       "mean": mean, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
