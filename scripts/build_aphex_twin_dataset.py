"""Download non-duplicate Aphex Twin works from YouTube into a training set.

    # catalogue from MusicBrainz (falls back to the curated list if unreachable)
    python scripts/build_aphex_twin_dataset.py --out data/aphex_twin

    # curated list only, albums only, search but don't download
    python scripts/build_aphex_twin_dataset.py --out data/aphex_twin --source curated --no-eps --dry-run

    # pass browser cookies if YouTube asks you to sign in / blocks the sandbox IP
    python scripts/build_aphex_twin_dataset.py --out data/aphex_twin --cookies cookies.txt

Output layout:

    <out>/catalog.json      the works we looked for (title, album, year, duration)
    <out>/manifest.jsonl    one line per work: status, chosen video, file, candidates
    <out>/audio/*.opus      downloaded audio (one file per distinct work / video)

Re-running is incremental: works with status "ok" are skipped and a video id
that is already in the manifest is never downloaded twice.

Please make sure you are allowed to download the material in your
jurisdiction and under YouTube's terms before running this for real.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from musicfit.dataset.aphex_twin_catalog import ALBUMS, EPS, curated_works  # noqa: E402
from musicfit.dataset.catalog import dedupe_works, musicbrainz_tracks, normalize_title, save_catalog  # noqa: E402
from musicfit.dataset.youtube import YouTubeClient, build_dataset  # noqa: E402

MAJOR_ALBUMS = list(ALBUMS)
MAJOR_EPS = list(EPS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/aphex_twin")
    ap.add_argument("--source", choices=["auto", "musicbrainz", "curated"], default="auto",
                    help="where the track list comes from (auto: MusicBrainz, curated on failure)")
    ap.add_argument("--all-releases", action="store_true",
                    help="with MusicBrainz: every official album/EP, not just the major ones")
    ap.add_argument("--no-eps", action="store_true", help="major albums only")
    ap.add_argument("--codec", default="opus", help="ffmpeg audio codec for yt-dlp (opus, mp3, m4a, wav, flac)")
    ap.add_argument("--cookies", default=None, help="Netscape cookies.txt for yt-dlp")
    ap.add_argument("--n-search", type=int, default=8)
    ap.add_argument("--duration-tol", type=float, default=0.08, help="relative duration tolerance for matching")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between downloads")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N works (testing)")
    ap.add_argument("--dry-run", action="store_true", help="search and pick videos but do not download")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    works = None
    if args.source in ("auto", "musicbrainz"):
        try:
            albums = None if args.all_releases else MAJOR_ALBUMS + ([] if args.no_eps else MAJOR_EPS)
            works = musicbrainz_tracks("Aphex Twin", albums=albums, include_eps=not args.no_eps)
            # attach search aliases (SAW II "#n" names etc.) from the curated list, matching
            # on curated aliases too (MusicBrainz "Mold" is the curated "Mould"); titles take
            # precedence, so the RDJ Album track "4" doesn't pick up SAW II's "#4"
            cur = curated_works(include_eps=True)
            curated = {c.key: c for c in cur}
            for c in cur:
                for name in c.aliases:
                    curated.setdefault(normalize_title(name), c)
            for w in works:
                c = curated.get(w.key)
                if c is not None:
                    extra = [a for a in [c.title] + c.aliases if normalize_title(a) != w.key]
                    w.aliases = list(dict.fromkeys(w.aliases + extra))
            print(f"MusicBrainz catalogue: {len(works)} distinct works")
        except Exception as e:  # noqa: BLE001
            if args.source == "musicbrainz":
                raise
            print(f"MusicBrainz unavailable ({type(e).__name__}: {e}); using curated catalogue")
    if works is None:
        works = curated_works(include_eps=not args.no_eps)
        print(f"Curated catalogue: {len(works)} distinct works")
    works = dedupe_works(works)
    if args.limit:
        works = works[: args.limit]
    save_catalog(works, os.path.join(args.out, "catalog.json"))

    client = YouTubeClient(cookies_file=args.cookies, sleep_between=args.sleep)
    manifest = build_dataset(works, args.out, client, codec=args.codec, n_search=args.n_search,
                             dry_run=args.dry_run, retry_failed=args.retry_failed, duration_tol=args.duration_tol)
    print("\nmanifest:", manifest.path)
    print("status counts:", manifest.summary())


if __name__ == "__main__":
    main()
