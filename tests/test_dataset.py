import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from musicfit.dataset.aphex_twin_catalog import ALBUMS, curated_works  # noqa: E402
from musicfit.dataset.catalog import Work, dedupe_works, normalize_title  # noqa: E402
from musicfit.dataset.youtube import Candidate, Manifest, build_dataset, choose_candidate, score_candidate  # noqa: E402


def test_normalize_title():
    assert normalize_title("Xtal (2019 Remaster)") == "xtal"
    assert normalize_title("Aphex Twin - Avril 14th [Official Audio]") == "avril 14th"
    assert normalize_title("Girl/Boy Song") == "girl boy song"
    assert normalize_title("Come On You Slags!") == "come on you slags"
    assert normalize_title("Xtal") == normalize_title("XTAL ")
    # content that distinguishes works is kept
    assert normalize_title("Come to Daddy (Pappy Mix)") != normalize_title("Come to Daddy (Mummy Mix)")


def test_dedupe_works_merges_rereleases():
    works = [
        Work("Xtal", "Aphex Twin", "SAW 85-92", 294.0),
        Work("Xtal (Remastered)", "Aphex Twin", "SAW 85-92 (2006 reissue)", None),
        Work("Tha", "Aphex Twin", "SAW 85-92", 542.0),
    ]
    out = dedupe_works(works)
    assert [w.title for w in out] == ["Xtal", "Tha"]
    assert out[0].aliases == ["Xtal (Remastered)"]


def test_curated_catalog_has_major_albums_and_no_duplicates():
    works = curated_works(include_eps=False)
    albums = {w.album for w in works}
    assert set(ALBUMS) == albums
    keys = [w.key for w in works]
    assert len(keys) == len(set(keys))
    assert len(works) >= 13 + 25 + 12 + 10 + 30 + 12
    # "To Cure a Weakling Child" (album) and its EP remix stay distinct works
    eps = curated_works(include_eps=True)
    titles = {w.title for w in eps}
    assert "To Cure a Weakling Child" in titles and "To Cure a Weakling Child (Contour Regard)" in titles


def _work():
    return Work("Avril 14th", "Aphex Twin", "Drukqs", 125.0)


def test_score_candidate_prefers_official_and_rejects_covers():
    w = _work()
    official = Candidate("a", "Avril 14th", 125, uploader="Aphex Twin - Topic")
    fan = Candidate("b", "Aphex Twin - Avril 14th", 126, uploader="someone")
    cover = Candidate("c", "Avril 14th (piano cover)", 125, uploader="pianist")
    slowed = Candidate("d", "Aphex Twin - Avril 14th (slowed + reverb)", 140, uploader="x")
    wrong_len = Candidate("e", "Avril 14th", 600, uploader="x")
    other = Candidate("f", "Aphex Twin - Flim", 177, uploader="Aphex Twin - Topic")
    assert score_candidate(w, official) > score_candidate(w, fan) > 0
    assert score_candidate(w, cover) <= 0
    assert score_candidate(w, slowed) <= 0
    assert score_candidate(w, wrong_len) <= 0
    assert score_candidate(w, other) <= 0
    best, _ = choose_candidate(w, [fan, cover, official, slowed, wrong_len, other])
    assert best is official


def test_reject_words_allowed_when_in_work_title():
    w = Work("Come to Daddy (Pappy Mix)", "Aphex Twin", "Come to Daddy", 262.0)
    c = Candidate("a", "Aphex Twin - Come to Daddy (Pappy Mix)", 262, uploader="Aphex Twin - Topic")
    assert score_candidate(w, c) > 0
    w2 = Work("Flim", "Aphex Twin", "Come to Daddy", 177.0)
    assert score_candidate(w2, Candidate("b", "Flim (Aphex Twin remix)", 177)) <= 0


def test_alias_search_matches_saw2_numbered_titles():
    w = Work("Rhubarb", "Aphex Twin", "SAW II", 464.0, aliases=["#3"])
    assert score_candidate(w, Candidate("a", "Aphex Twin - #3", 465)) > 0


class FakeClient:
    """Stands in for YouTubeClient: canned search results, fake downloads."""

    sleep_between = 0

    def __init__(self, results, fail_ids=()):
        self.results = results
        self.fail_ids = set(fail_ids)
        self.downloads = []

    def search(self, query, n=8):
        return self.results.get(query, [])

    def download_audio(self, video_id, out_path_no_ext, codec="opus", quality="0"):
        if video_id in self.fail_ids:
            raise RuntimeError("HTTP Error 403")
        self.downloads.append(video_id)
        path = f"{out_path_no_ext}.{codec}"
        with open(path, "wb") as f:
            f.write(b"\x00" * 16)
        return path


def test_build_dataset_dedupes_videos_and_is_incremental(tmp_path):
    works = [
        Work("Xtal", "Aphex Twin", "SAW", 294.0),
        Work("Xtal (Remastered)", "Aphex Twin", "SAW reissue", 294.0),  # same key -> dropped by dedupe_works
        Work("Tha", "Aphex Twin", "SAW", 542.0),
        Work("Tha II", "Aphex Twin", "Single", 542.0, aliases=["Tha"]),  # distinct key, same video -> duplicate_video
        Work("Ptolemy", "Aphex Twin", "SAW", 432.0),
        Work("Nothing", "Aphex Twin", "SAW", 100.0),
    ]
    works = dedupe_works(works)
    results = {
        "Aphex Twin Xtal": [Candidate("v1", "Aphex Twin - Xtal", 294, uploader="Aphex Twin - Topic")],
        "Aphex Twin Tha II": [],
        "Aphex Twin Tha": [Candidate("v2", "Aphex Twin - Tha", 542)],
        "Aphex Twin Ptolemy": [Candidate("v3", "Aphex Twin - Ptolemy", 432)],
        "Aphex Twin Nothing": [],
    }
    client = FakeClient(results, fail_ids={"v3"})
    out = tmp_path / "ds"
    m = build_dataset(works, str(out), client, verbose=False)
    assert m.summary() == {"ok": 2, "duplicate_video": 1, "download_failed": 1, "not_found": 1}
    assert client.downloads == ["v1", "v2"]
    assert (out / "audio" / "Aphex_Twin_Xtal.opus").exists()
    assert m.entries["tha ii"].status == "duplicate_video"

    # second run: nothing re-downloaded unless --retry-failed
    client2 = FakeClient(results)
    m2 = build_dataset(works, str(out), client2, verbose=False)
    assert client2.downloads == []
    client3 = FakeClient(results)
    m3 = build_dataset(works, str(out), client3, retry_failed=True, verbose=False)
    assert client3.downloads == ["v3"]
    assert m3.summary()["ok"] == 3

    # manifest round-trips
    lines = [json.loads(l) for l in open(out / "manifest.jsonl")]
    assert {l["work_key"] for l in lines} == {w.key for w in works}
    assert Manifest(str(out / "manifest.jsonl")).downloaded_video_ids() == {"v1", "v2", "v3"}


def test_dry_run_downloads_nothing(tmp_path):
    works = [Work("Xtal", "Aphex Twin", "SAW", 294.0)]
    client = FakeClient({"Aphex Twin Xtal": [Candidate("v1", "Xtal", 294)]})
    m = build_dataset(works, str(tmp_path), client, dry_run=True, verbose=False)
    assert client.downloads == []
    assert m.entries["xtal"].status == "skipped"
    assert m.entries["xtal"].video_id == "v1"


@pytest.mark.network
def test_musicbrainz_live():
    """Only runs with -m network; needs musicbrainz.org to be reachable."""
    from musicfit.dataset.catalog import musicbrainz_tracks

    works = musicbrainz_tracks("Aphex Twin", albums=["Richard D. James Album"], verbose=False)
    assert {w.key for w in works} >= {"4", "cornish acid", "fingerbib", "girl boy song"}
