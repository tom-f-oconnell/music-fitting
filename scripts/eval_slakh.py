"""Score the fitter against true notes on real-sounding audio (BabySlakh).

The synthetic benchmark can only contain what we put in it, and it has already
preferred settings that fail on real music (design.md).  BabySlakh is 20 songs
rendered from MIDI with sampled instruments and drums, so the notes are known:

    curl -L -o babyslakh_16k.tar.gz https://zenodo.org/records/4603870/files/babyslakh_16k.tar.gz
    tar xzf babyslakh_16k.tar.gz -C data/babyslakh

    python scripts/eval_slakh.py --source mix --out eval_out/slakh_mix.json
    python scripts/eval_slakh.py --source Bass --set n_voices=1 --set pitch_min=28 --out eval_out/slakh_bass.json
    python scripts/eval_slakh.py --source mix --method basic-pitch --separate --compare eval_out/slakh_mix.json

--source: "mix", "pitched" (all stems but the drums: perfect drum removal), or an
instrument class ("Bass", "Piano", "Synth Lead", ...). --method: "fit" (default),
"basic-pitch" (the transcription alone, as a reference for what a pretrained
model gets) or "fit+basic-pitch" (stage 2 from its notes). --separate runs Demucs
first (see musicfit.tools). Notes outside the fit's pitch range aren't counted.
Results are paired by clip, like eval_synthetic.py, whose options work the same.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
from eval_synthetic import compare_f1, run_targets, save_results, summarize  # noqa: E402

from musicfit.fit import FitConfig, match_notes, note_metrics  # noqa: E402
from musicfit.slakh import Clip, make_clips  # noqa: E402


def transcribe_only(clip: Clip, cfg: FitConfig, use_demucs: bool):
    from musicfit.tools import PITCHED_STEMS, separate, transcribe

    audio = clip.audio
    if use_demucs:
        stems = separate(audio, cfg.sr, device=cfg.device)
        audio = sum(stems[s] for s in PITCHED_STEMS if s in stems)
    return [n for n in transcribe(audio, cfg.sr) if cfg.pitch_min <= n.pitch <= cfg.pitch_max]


def fit_clip(args) -> dict:
    clip, overrides, method, use_demucs, threads = args
    torch.set_num_threads(threads)
    cfg = FitConfig(seed=clip.clip_id, **overrides)
    t0 = time.time()
    if method == "basic-pitch":
        notes = roll_notes = transcribe_only(clip, cfg, use_demucs)
        loss, patches = float("nan"), []
    else:
        from musicfit.fit import fit

        if method == "fit" and not use_demucs:
            res = fit(clip.audio, cfg, verbose=False)
        else:
            from musicfit.tools import fit_with_tools

            res = fit_with_tools(clip.audio, cfg, use_demucs, method == "fit+basic-pitch", verbose=False)
        notes, roll_notes = res.notes, res.roll_notes
        loss, patches = res.history[-1]["loss"], [s.params.describe() for s in res.synths]
    found = match_notes(notes, clip.ref)
    return {
        "seed": clip.clip_id,
        "track": clip.track,
        "start": clip.start,
        "time_s": round(time.time() - t0, 1),
        "final_loss": loss,
        **note_metrics(notes, clip.ref),
        "roll_metrics": note_metrics(roll_notes, clip.ref),
        "n_out_of_range": clip.n_out_of_range,
        "class_total": dict(Counter(clip.ref_class)),
        "class_found": dict(Counter(clip.ref_class[j] for j in found)),
        "ref_notes": [n.as_dict() for n in clip.ref],
        "roll_notes": [n.as_dict() for n in roll_notes],
        "notes": [n.as_dict() for n in notes],
        "fitted_patches": patches,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/babyslakh/babyslakh_16k", help="directory with the Track* directories")
    ap.add_argument("--source", default="mix", help="mix, pitched, or an instrument class such as Bass")
    ap.add_argument("--method", default="fit", choices=("fit", "basic-pitch", "fit+basic-pitch"))
    ap.add_argument("--separate", action="store_true", help="run Demucs first")
    ap.add_argument("--clips-per-track", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="use only the first LIMIT clips")
    ap.add_argument("--length", type=float, default=4.0, help="clip length in seconds")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE", help="override a FitConfig field")
    ap.add_argument("--jobs", type=int, default=None, help="parallel fits (default 3 on a GPU, else 1)")
    ap.add_argument("--out", help="save all results as JSON")
    ap.add_argument("--compare", help="earlier --out file to compare F1 against, paired by clip")
    args = ap.parse_args()

    overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    overrides.setdefault("n_voices", 2)
    cfg = FitConfig(**overrides)
    clips = make_clips(args.data, args.source, args.length, args.clips_per_track,
                       pitch_range=(cfg.pitch_min, cfg.pitch_max), sr=cfg.sr)[:args.limit]
    on_gpu = torch.cuda.is_available() and cfg.device != "cpu" and args.method != "basic-pitch"
    jobs = args.jobs or (3 if on_gpu else 1)
    threads = max(1, (os.cpu_count() or 1) // jobs)
    rows = run_targets(fit_clip, [(c, overrides, args.method, args.separate, threads) for c in clips], jobs)
    mean = summarize(rows)

    totals, found = Counter(), Counter()
    for r in rows:
        totals.update(r["class_total"])
        found.update(r["class_found"])
    recall = {k: f"{found[k]}/{n} = {found[k] / n:.2f}" for k, n in totals.most_common()}
    print("recall by instrument:", recall)
    print(f"true notes outside the pitch range (not counted): {sum(r['n_out_of_range'] for r in rows)}; "
          f"median F1 {statistics.median(r['f1'] for r in rows):.3f}")

    save_results(args.out, {"overrides": overrides, "source": args.source, "method": args.method,
                            "separate": args.separate, "mean": mean, "rows": rows})
    compare_f1(rows, args.compare)


if __name__ == "__main__":
    main()
