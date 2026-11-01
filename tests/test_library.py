"""The data library: listing what is cached, fitting it, and downloading more."""

import hashlib
import io
import json
import os
import sys
import tarfile
import threading
import time

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from musicfit.dataset import library as lib  # noqa: E402
from musicfit.dataset.catalog import Work  # noqa: E402
from musicfit.dataset.library import Library, fetch_zenodo, zenodo_record_id  # noqa: E402
from musicfit.dataset.youtube import Candidate, download_videos  # noqa: E402
from musicfit.web.app import create_app  # noqa: E402

SR = 16000


def _tone(path, seconds=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    sf.write(path, 0.3 * np.sin(2 * np.pi * 220 * t), SR)


def _manifest_dataset(root, name="yt"):
    d = os.path.join(root, name)
    _tone(os.path.join(d, "audio", "a.wav"))
    _tone(os.path.join(d, "audio", "gone_from_manifest.wav"))
    entries = [
        {"work_key": "a", "title": "Song", "artist": "Band", "album": None, "catalog_duration_s": None, "status": "ok",
         "video_id": "x", "file": "audio/a.wav", "video_duration_s": 61.0},
        {"work_key": "b", "title": "Lost", "artist": "Band", "album": None, "catalog_duration_s": None, "status": "not_found"},
    ]
    with open(os.path.join(d, "manifest.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(e) for e in entries) + "\n")


def _slakh_dataset(root):
    d = os.path.join(root, "babyslakh", "babyslakh_16k", "Track00001")
    _tone(os.path.join(d, "mix.wav"))
    _tone(os.path.join(d, "stems", "S00.wav"))
    with open(os.path.join(d, "metadata.yaml"), "w") as f:
        f.write("stems:\n  S00: {inst_class: Piano}\n  S01: {inst_class: Bass}\n")


def test_lists_each_layout_with_titles(tmp_path):
    root = str(tmp_path / "data")
    _manifest_dataset(root)
    _slakh_dataset(root)
    _tone(os.path.join(root, "plain", "take one.wav"))
    os.makedirs(os.path.join(root, "empty"))
    found = {d["id"]: d for d in Library(root).datasets()}
    assert set(found) == {"examples", "yt", "babyslakh", "plain"}
    assert [(t["title"], t["duration_s"]) for t in found["yt"]["tracks"]] == [("Band - Song", 61.0)]
    # S01 has no audio, whatever the metadata says
    assert [t["title"] for t in found["babyslakh"]["tracks"]] == ["Track00001: mix", "Track00001: Piano"]
    assert found["babyslakh"]["name"] == "BabySlakh"
    assert [t["title"] for t in found["plain"]["tracks"]] == ["take one"]
    assert found["plain"]["tracks"][0]["duration_s"] == pytest.approx(1.0)


def test_listing_survives_files_vanishing_during_a_download(tmp_path, monkeypatch):
    root = str(tmp_path / "data")
    _manifest_dataset(root)
    walk = os.walk
    monkeypatch.setattr(lib.os, "walk", lambda base: ((d, s, f + ["manifest.jsonl.tmp"]) for d, s, f in walk(base)))
    assert Library(root).datasets()[1]["id"] == "yt"


def test_a_manifest_with_nothing_finished_lists_nothing(tmp_path):
    root = tmp_path / "data"
    _tone(str(root / "yt" / "audio" / "unfinished.wav"))
    (root / "yt" / "manifest.jsonl").write_text(json.dumps({"work_key": "a", "title": "A", "artist": "B", "status": "download_failed"}) + "\n")
    assert [d["id"] for d in Library(str(root)).datasets()] == ["examples"]


def test_examples_ship_and_local_ones_need_the_download(tmp_path):
    root = str(tmp_path / "data")
    library = Library(root)
    examples = library.datasets()[0]
    assert examples["id"] == "examples" and "Sparse piano" in [t["title"] for t in examples["tracks"]]
    assert "Cornish Acid" not in [t["title"] for t in examples["tracks"]]
    assert all(os.path.isfile(library.resolve(t["id"])) for t in examples["tracks"])
    os.makedirs(os.path.join(root, "aphex_twin", "audio"))
    open(os.path.join(root, "aphex_twin", "audio", "Aphex_Twin_Cornish_Acid.opus"), "wb").close()
    acid = next(t for t in library.datasets()[0]["tracks"] if t["title"] == "Cornish Acid")
    assert acid["offset"] == 30.0 and acid["voices"] == 2


@pytest.mark.parametrize("bad", ["../secret.wav", "yt/../../secret.wav", "yt/audio/../../../secret.wav", "/etc/passwd",
                                 "yt/manifest.jsonl", "yt/", "yt", "", "yt/audio/missing.wav", "..%2Fsecret.wav"])
def test_ids_can_only_name_audio_inside_a_dataset(tmp_path, bad):
    root = tmp_path / "data"
    _manifest_dataset(str(root))
    _tone(str(tmp_path / "secret.wav"))
    with pytest.raises(ValueError):
        Library(str(root)).resolve(bad)


def test_ids_cannot_follow_a_link_out_of_the_library(tmp_path):
    root = tmp_path / "data"
    _manifest_dataset(str(root))
    _tone(str(tmp_path / "secret.wav"))
    os.symlink(tmp_path / "secret.wav", root / "yt" / "audio" / "link.wav")
    with pytest.raises(ValueError):
        Library(str(root)).resolve("yt/audio/link.wav")


class FakeYouTube:
    sleep_between = 0

    def __init__(self, videos=(), fail=()):
        self.videos, self.fail, self.downloaded = list(videos), set(fail), []

    def search(self, query, n=8):
        return self.videos[:n]

    def list_url(self, url, limit=None):
        return self.videos[:limit]

    def download_audio(self, video_id, base, codec="opus"):
        if video_id in self.fail:
            raise RuntimeError("Sign in to confirm you're not a bot")
        self.downloaded.append(video_id)
        _tone(base + ".wav")
        return base + ".wav"


def _videos(n=3):
    return [Candidate(f"id{i}", f"Video {i}", 120.0, "Someone") for i in range(n)]


def test_download_videos_resumes_and_skips_long_ones(tmp_path):
    videos = _videos() + [Candidate("long", "Ten hour mix", 36000.0, "Someone")]
    client = FakeYouTube(fail={"id1"})
    m = download_videos(videos, str(tmp_path), client, max_duration_s=1200)
    assert m.summary() == {"ok": 2, "download_failed": 1, "skipped": 1}
    again = FakeYouTube()
    assert download_videos(videos, str(tmp_path), again, max_duration_s=1200).summary() == {"ok": 3, "skipped": 1}
    assert again.downloaded == ["id1"]  # the others aren't fetched twice


@pytest.fixture()
def app(tmp_path):
    return create_app(str(tmp_path / "jobs"), str(tmp_path / "data"))


def _wait_download(client, download_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = next(x for x in client.get("/api/library").json()["downloads"] if x["id"] == download_id)
        if d["status"] not in ("queued", "running"):
            return d
        time.sleep(0.05)
    raise TimeoutError(d)


def _wait_job(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job)


def test_search_download_is_cached_and_can_be_fitted(app, tmp_path):
    client = TestClient(app)
    app.state.downloads.client_factory = lambda: FakeYouTube(_videos(2))
    d = client.post("/api/library/downloads", json={"source": "search", "query": "acid bass", "name": "Acid Bass", "limit": 2}).json()
    d = _wait_download(client, d["id"])
    assert d["status"] == "done" and (d["done"], d["total"]) == (2, 2) and d["dataset"] == "acid_bass"

    library = client.get("/api/library").json()
    dataset = next(x for x in library["datasets"] if x["id"] == "acid_bass")
    assert dataset["name"] == "Acid Bass" and dataset["source"] == "search" and dataset["query"] == "acid bass"
    assert len(dataset["tracks"]) == 2
    track = dataset["tracks"][0]["id"]
    assert client.get("/api/library/audio", params={"track": track}).status_code == 200
    assert client.get("/api/library/audio", params={"track": "acid_bass/../../jobs"}).status_code == 404

    job = client.post("/api/jobs", data={"track": track, "duration": "0.8", "roll_steps": "5", "note_steps": "3",
                                         "pitch_min": "50", "pitch_max": "62"}).json()
    assert job["kind"] == "library" and job["name"] == dataset["tracks"][0]["title"]
    job = _wait_job(client, job["id"])
    assert job["status"] == "done", job["error"]
    assert job["result"]["n_samples"] == int(0.8 * SR)

    # a new app on the same directory still has it, and knows the download's outcome
    again = TestClient(create_app(str(tmp_path / "jobs"), str(tmp_path / "data"))).get("/api/library").json()
    assert any(x["id"] == "acid_bass" for x in again["datasets"])
    assert next(x for x in again["downloads"] if x["id"] == d["id"])["status"] == "done"


def test_fit_a_shipped_example(app):
    client = TestClient(app)
    job = client.post("/api/jobs", data={"track": "examples/sparse_piano.flac", "duration": "1", "roll_steps": "5",
                                         "note_steps": "3"}).json()
    assert _wait_job(client, job["id"])["status"] == "done"


def test_fit_needs_a_file_or_a_real_track(app):
    client = TestClient(app)
    assert client.post("/api/jobs", data={"duration": "1"}).status_code == 422
    assert client.post("/api/jobs", data={"duration": "1", "track": "../../etc/passwd.wav"}).status_code == 422


def test_artist_download_uses_musicbrainz_and_the_limit(app, monkeypatch):
    client = TestClient(app)
    works = [Work("Song One", "Band", duration_s=120.0), Work("Song Two", "Band", duration_s=120.0)]
    monkeypatch.setattr("musicfit.web.downloads.musicbrainz_tracks", lambda artist, verbose=False: works)
    videos = [Candidate("one", "Band - Song One", 120.0, "Band")]
    app.state.downloads.client_factory = lambda: FakeYouTube(videos)
    d = client.post("/api/library/downloads", json={"source": "artist", "query": "Band", "limit": 1}).json()
    d = _wait_download(client, d["id"])
    assert d["status"] == "done" and d["total"] == 1
    assert [t["title"] for t in next(x for x in client.get("/api/library").json()["datasets"] if x["id"] == d["dataset"])["tracks"]] == ["Band - Song One"]


def test_failed_download_says_why_and_mentions_cookies(app):
    client = TestClient(app)
    app.state.downloads.client_factory = lambda: FakeYouTube(_videos(1), fail={"id0"})
    d = client.post("/api/library/downloads", json={"source": "search", "query": "x"}).json()
    d = _wait_download(client, d["id"])
    assert d["status"] == "failed" and "MUSICFIT_YT_COOKIES" in d["error"]


@pytest.mark.parametrize("source,query", [
    ("url", "https://example.com/video"), ("url", "file:///etc/passwd"), ("url", "http://169.254.169.254/"),
    ("url", "https://youtube.com.evil.example/watch?v=x"), ("zenodo", "https://example.com/records/1"), ("zenodo", "abc"),
    ("nonsense", "x"), ("search", "   "),
])
def test_only_known_sites_are_accepted(app, source, query):
    assert TestClient(app).post("/api/library/downloads", json={"source": source, "query": query}).status_code == 422


def test_a_cancelled_download_keeps_what_it_has_and_resumes(app):
    client = TestClient(app)
    manager = app.state.downloads
    first = FakeYouTube(_videos(3))
    cancel_after_first = first.download_audio

    def stop_after_first(video_id, base, codec="opus"):
        path = cancel_after_first(video_id, base, codec)
        manager.cancel(next(d.id for d in manager.records.values() if d.status == "running"))
        return path

    first.download_audio = stop_after_first
    manager.client_factory = lambda: first
    d = client.post("/api/library/downloads", json={"source": "search", "query": "x", "name": "Stops"}).json()
    d = _wait_download(client, d["id"])
    assert d["status"] == "cancelled" and first.downloaded == ["id0"]

    second = FakeYouTube(_videos(3))
    manager.client_factory = lambda: second
    d = _wait_download(client, client.post("/api/library/downloads", json={"source": "search", "query": "x", "name": "Stops"}).json()["id"])
    assert d["status"] == "done" and second.downloaded == ["id1", "id2"]


def test_records_can_be_saved_from_several_threads(app):
    manager, errors = app.state.downloads, []

    def save():
        for _ in range(200):
            try:
                manager._save(force=True)
            except OSError as e:
                errors.append(e)

    threads = [threading.Thread(target=save) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors


def test_a_failure_saving_records_doesnt_stop_later_downloads(app, monkeypatch):
    client, manager = TestClient(app), app.state.downloads
    manager.client_factory = lambda: FakeYouTube(_videos(1))
    run = manager._run
    calls = []

    def fails_once(d):
        calls.append(d.id)
        if len(calls) == 1:
            d.status = "failed"
            raise OSError("disk full")
        run(d)

    monkeypatch.setattr(manager, "_run", fails_once)
    client.post("/api/library/downloads", json={"source": "search", "query": "x", "name": "First"})
    second = client.post("/api/library/downloads", json={"source": "search", "query": "y", "name": "Second"}).json()
    assert _wait_download(client, second["id"])["status"] == "done"


def test_the_same_dataset_cant_download_twice_at_once(app):
    client = TestClient(app)
    manager = app.state.downloads
    release = []

    class Slow(FakeYouTube):
        def search(self, query, n=8):
            while not release:
                time.sleep(0.01)
            return super().search(query, n)

    manager.client_factory = lambda: Slow(_videos(1))
    d = client.post("/api/library/downloads", json={"source": "search", "query": "x", "name": "Once"}).json()
    assert client.post("/api/library/downloads", json={"source": "search", "query": "x", "name": "Once"}).status_code == 422
    release.append(True)
    assert _wait_download(client, d["id"])["status"] == "done"


# ---- Zenodo

class FakeResponse:
    def __init__(self, data=None, status=200, payload=None):
        self.data, self.status_code, self.payload = data, status, payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.payload

    def iter_content(self, n):
        for i in range(0, len(self.data), n):
            yield self.data[i:i + n]


def _archive(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _fake_zenodo(monkeypatch, blob, name="clips.tar.gz", checksum=None, calls=None):
    record = {"metadata": {"title": "Clips", "license": {"id": "cc-by-4.0"}},
              "files": [{"key": "readme.txt", "size": 5, "checksum": "md5:0", "links": {"self": "https://zenodo.org/f/readme"}},
                        {"key": name, "size": len(blob), "checksum": checksum or "md5:" + hashlib.md5(blob).hexdigest(),
                         "links": {"self": "https://zenodo.org/f/" + name}}]}

    def get(url, headers=None, stream=False, timeout=None):
        (calls if calls is not None else []).append((url, headers))
        if "api/records" in url:
            return FakeResponse(payload=record)
        if headers and "Range" in headers:
            start = int(headers["Range"].split("=")[1].rstrip("-"))
            return FakeResponse(blob[start:], status=206)
        return FakeResponse(blob)

    monkeypatch.setattr(lib.requests, "get", get)


def _wav_bytes():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(SR // 4), SR, format="WAV")
    return buf.getvalue()


def test_zenodo_downloads_unpacks_and_lists(tmp_path, monkeypatch):
    calls = []
    blob = _archive({"clips/one.wav": _wav_bytes(), "clips/notes.txt": b"hello"})
    _fake_zenodo(monkeypatch, blob, calls=calls)
    root = tmp_path / "data"
    info = fetch_zenodo("https://zenodo.org/records/123", str(root / "z"))
    assert info == {"title": "Clips", "license": "cc-by-4.0"}
    assert (root / "z" / "clips" / "one.wav").exists() and not (root / "z" / "clips.part").exists()
    assert [t["id"] for t in Library(str(root)).datasets()[1]["tracks"]] == ["z/clips/one.wav"]
    assert "readme" not in " ".join(url for url, _ in calls)  # only audio and archives are fetched
    # again: the file is verified and nothing is fetched or unpacked
    calls.clear()
    (root / "z" / "clips" / "marker").write_text("kept")
    fetch_zenodo("123", str(root / "z"))
    assert len(calls) == 1 and (root / "z" / "clips" / "marker").exists()


def test_zenodo_continues_a_partial_download(tmp_path, monkeypatch):
    calls = []
    blob = _archive({"clips/one.wav": _wav_bytes()})
    _fake_zenodo(monkeypatch, blob, calls=calls)
    out = tmp_path / "z"
    (out / "download").mkdir(parents=True)
    (out / "download" / "clips.tar.gz.part").write_bytes(blob[:10])
    fetch_zenodo("123", str(out))
    assert calls[-1][1] == {"Range": "bytes=10-"}
    assert (out / "clips" / "one.wav").exists()


def test_zenodo_rejects_bad_checksums_and_huge_records(tmp_path, monkeypatch):
    blob = _archive({"clips/one.wav": _wav_bytes()})
    _fake_zenodo(monkeypatch, blob, checksum="md5:" + "0" * 32)
    with pytest.raises(ValueError, match="checksum"):
        fetch_zenodo("123", str(tmp_path / "z"))
    assert not (tmp_path / "z" / "download" / "clips.tar.gz.part").exists()
    _fake_zenodo(monkeypatch, blob)
    monkeypatch.setattr(lib, "MAX_DOWNLOAD_GB", len(blob) / 2e9)
    with pytest.raises(ValueError, match="over the limit"):
        fetch_zenodo("123", str(tmp_path / "y"))


def test_archive_members_cannot_escape_the_dataset(tmp_path, monkeypatch):
    blob = _archive({"../../escaped.wav": _wav_bytes()})
    _fake_zenodo(monkeypatch, blob)
    with pytest.raises(tarfile.FilterError):
        fetch_zenodo("123", str(tmp_path / "data" / "z"))
    assert not list(tmp_path.rglob("escaped.wav"))


def test_archive_bomb_is_refused(tmp_path, monkeypatch):
    blob = _archive({"big.wav": b"\0" * 1000})
    _fake_zenodo(monkeypatch, blob)
    monkeypatch.setattr(lib, "MAX_UNPACK_RATIO", 1)
    monkeypatch.setattr(lib, "MAX_DOWNLOAD_GB", 500 / 1e9)
    with pytest.raises(ValueError, match="too much"):
        fetch_zenodo("123", str(tmp_path / "z"))


@pytest.mark.parametrize("text,record", [("4603870", "4603870"), (" 12 ", "12"), ("https://zenodo.org/records/9?preview=1", "9"),
                                         ("https://zenodo.org/record/9", "9")])
def test_zenodo_record_forms(text, record):
    assert zenodo_record_id(text) == record
