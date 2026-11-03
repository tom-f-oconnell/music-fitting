"""Cut the example clips shipped in musicfit/dataset/examples from BabySlakh (CC BY 4.0).

    python scripts/make_examples.py --data data/babyslakh/babyslakh_16k

The clips are 4 s windows from the evaluation (scripts/eval_slakh.py), chosen for how
well or how badly they show the fitter working. See examples/ATTRIBUTION.md.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf  # noqa: E402

from musicfit.dataset.library import EXAMPLES_DIR  # noqa: E402
from musicfit.slakh import make_clips  # noqa: E402

# file -> (source, as for make_clips; track; "drums" for that window's drum stems alone)
CLIPS = {
    "sparse_piano.flac": ("Piano", "Track00004", ""),
    "busier_piano.flac": ("Piano", "Track00013", ""),
    "bass_line.flac": ("Bass", "Track00013", ""),
    "full_band.flac": ("mix", "Track00002", ""),  # sounding throughout, with drums and five kinds of instrument
    "drums.flac": ("mix", "Track00016", "drums"),  # 69 hits: tambourine, kick, hats, snare, claves and shaker
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/babyslakh/babyslakh_16k", help="directory with the Track* directories")
    args = ap.parse_args()
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    for name, (source, track, part) in CLIPS.items():
        clip = next(c for c in make_clips(args.data, source) if c.track == track)
        audio = clip.drum_audio if part == "drums" else clip.audio
        sf.write(os.path.join(EXAMPLES_DIR, name), audio.numpy(), 16000, subtype="PCM_16")
        print(f"{name}: {track} from {clip.start:.0f} s, {source}{' drums' if part else ''}, {len(clip.ref)} notes, {len(clip.drum_ref)} drum hits")


if __name__ == "__main__":
    main()
