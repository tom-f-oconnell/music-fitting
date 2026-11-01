"""Background dataset downloads into the library, one at a time, recorded on disk.

The server has no login, so what a client can ask for is deliberately narrow: searches,
YouTube links and Zenodo record numbers, never another URL or a path on this machine.
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

from ..dataset.aphex_twin_catalog import aphex_twin_works
from ..dataset.catalog import musicbrainz_tracks, save_catalog
from ..dataset.library import PRESETS, Library, fetch_zenodo, slug, zenodo_record_id
from ..dataset.youtube import Manifest, YouTubeClient, build_dataset, download_videos

SOURCES = {
    "artist": {"label": "An artist's albums and EPs (MusicBrainz + YouTube)", "placeholder": "Artist name"},
    "search": {"label": "A YouTube search", "placeholder": "Search words"},
    "url": {"label": "A YouTube video or playlist", "placeholder": "https://www.youtube.com/..."},
    "zenodo": {"label": "A Zenodo record", "placeholder": "Record number or zenodo.org link"},
}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
MAX_VIDEO_S = 1200.0  # searches and playlists skip anything longer: long mixes are poor material, and large
KEEP_RECORDS = 50


class Cancelled(Exception):
    pass


@dataclass
class Download:
    id: str
    dataset: str
    name: str
    source: str
    query: str
    limit: int | None = None
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | failed | cancelled | interrupted
    done: int = 0
    total: int = 0
    unit: str = "tracks"  # or "bytes"
    message: str = ""
    error: str | None = None


def _check_query(source: str, query: str) -> str:
    query = query.strip()
    if source not in SOURCES and source != "aphex_twin":
        raise ValueError(f"unknown source: {source}")
    if not query:
        raise ValueError("enter what to download")
    if source == "url":
        u = urlparse(query)
        if u.scheme not in ("http", "https") or u.hostname not in YOUTUBE_HOSTS:
            raise ValueError("expected a youtube.com or youtu.be link")
    elif source == "zenodo":
        zenodo_record_id(query)
    return query


def _default_client() -> YouTubeClient:
    return YouTubeClient(cookies_file=os.environ.get("MUSICFIT_YT_COOKIES"))


class DownloadManager:
    """Runs downloads on one worker thread; a daemon, so stopping the server doesn't wait for them.

    Stopping loses nothing but the file being fetched: asking for the same dataset again resumes.
    """

    def __init__(self, library: Library):
        self.library = library
        self.client_factory = _default_client
        self.records_path = os.path.join(library.root, ".downloads.json")
        self.records: dict[str, Download] = {}
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self._saved_at = 0.0
        self._queue: queue.Queue[Download] = queue.Queue()
        self._load()
        threading.Thread(target=self._worker, daemon=True).start()

    # -- API -----------------------------------------------------------------

    def state(self) -> dict:
        """The sources and presets on offer, and the recent downloads, newest first."""
        have = {d["id"] for d in self.library.datasets()}
        return {
            "sources": [{"id": k, **v} for k, v in SOURCES.items()],
            "presets": [{"id": k, "cached": k in have, "complete": self._complete(k), **v} for k, v in PRESETS.items()],
            "downloads": [asdict(d) for d in sorted(self.records.values(), key=lambda d: -d.created)],
            "can_use_youtube": importlib.util.find_spec("yt_dlp") is not None,
        }

    def _complete(self, preset: str) -> bool:
        marker = PRESETS[preset].get("complete_when")
        return bool(marker) and os.path.isdir(os.path.join(self.library.root, preset, marker))

    def submit_preset(self, preset: str) -> dict:
        if preset not in PRESETS:
            raise ValueError(f"unknown dataset: {preset}")
        p = PRESETS[preset]
        return self._enqueue(preset, p["name"], p["source"], p["query"], None)

    def submit(self, source: str, query: str, name: str = "", limit: int | None = 25) -> dict:
        query = _check_query(source, query)
        if source == "aphex_twin":
            return self.submit_preset("aphex_twin")
        name = name.strip() or query
        # a Zenodo record has one form however it's written; the same request resumes
        key = zenodo_record_id(query) if source == "zenodo" else query
        return self._enqueue(slug(name if name != query else f"{source}_{key}"), name, source, query, limit)

    def cancel(self, download_id: str) -> bool:
        d = self.records.get(download_id)
        if d is None:
            return False
        if d.status == "queued":
            d.status = "cancelled"
            self._save(force=True)
        elif d.status == "running":
            self._cancel.add(download_id)
        return True

    # -- worker ----------------------------------------------------------------

    def _enqueue(self, dataset: str, name: str, source: str, query: str, limit: int | None) -> dict:
        with self._lock:
            if any(d.dataset == dataset and d.status in ("queued", "running") for d in self.records.values()):
                raise ValueError(f"{name} is already downloading")
            d = Download(uuid.uuid4().hex[:12], dataset, name, source, query, limit)
            self.records[d.id] = d
        self._save(force=True)
        self._queue.put(d)
        return asdict(d)

    def _worker(self) -> None:
        while True:
            d = self._queue.get()
            if d.status == "queued":  # else cancelled while waiting
                self._run(d)

    def _run(self, d: Download) -> None:
        d.status = "running"
        self._save(force=True)
        out_dir = os.path.join(self.library.root, d.dataset)
        try:
            self.library.write_info(d.dataset, {"name": d.name, "source": d.source, "query": d.query, "limit": d.limit})
            extra = self._fetch(d, out_dir)
            self.library.write_info(d.dataset, {"name": d.name, "source": d.source, "query": d.query, "limit": d.limit, **extra})
            d.status = "done"
        except Cancelled:
            d.status, d.message = "cancelled", f"Stopped at {d.done} of {d.total}; download it again to resume."
        except Exception as e:  # noqa: BLE001  (surfaced in the UI)
            d.status, d.error = "failed", _explain(f"{type(e).__name__}: {e}")
        finally:
            self._cancel.discard(d.id)
            self._save(force=True)

    def _fetch(self, d: Download, out_dir: str) -> dict:
        if d.source == "zenodo":
            d.unit = "bytes"

            def on_bytes(done: int, total: int, name: str) -> None:
                self._step(d, done, total, name)

            info = fetch_zenodo(d.query, out_dir, on_bytes)
            d.done, d.total, d.message = d.total or d.done, d.total, "Downloaded and unpacked."
            return info

        def on_track(i: int, n: int, entry) -> None:
            self._step(d, i, n, entry.title)

        os.makedirs(out_dir, exist_ok=True)
        client = self.client_factory()
        if d.source in ("aphex_twin", "artist"):
            d.message = "Looking up the catalogue on MusicBrainz"
            self._save(force=True)
            works = aphex_twin_works(log=lambda _: None) if d.source == "aphex_twin" else musicbrainz_tracks(d.query, verbose=False)
            if d.limit:
                works = works[: d.limit]
            if not works:
                raise ValueError(f"MusicBrainz has no albums or EPs for {d.query}")
            save_catalog(works, os.path.join(out_dir, "catalog.json"))
            manifest = build_dataset(works, out_dir, client, verbose=False, retry_failed=True, on_progress=on_track)
        else:
            d.message = "Finding videos"
            self._save(force=True)
            videos = client.search(d.query, d.limit or 25) if d.source == "search" else client.list_url(d.query, d.limit)
            if not videos:
                raise ValueError("found no videos")
            manifest = download_videos(videos, out_dir, client, max_duration_s=MAX_VIDEO_S, on_progress=on_track)
        return self._finish_tracks(d, manifest)

    def _finish_tracks(self, d: Download, manifest: Manifest) -> dict:
        counts = manifest.summary()
        d.message = ", ".join(f"{n} {status.replace('_', ' ')}" for status, n in sorted(counts.items()))
        if not counts.get("ok"):
            errors = [e.error for e in manifest.entries.values() if e.error]
            raise ValueError(errors[0] if errors else f"nothing could be downloaded ({d.message})")
        return {}

    def _step(self, d: Download, done: int, total: int, message: str) -> None:
        d.done, d.total, d.message = done, total, message
        self._save()
        if d.id in self._cancel:
            raise Cancelled()

    # -- persistence -------------------------------------------------------------

    def _save(self, force: bool = False) -> None:
        if not force and time.time() - self._saved_at < 1.0:
            return
        self._saved_at = time.time()
        with self._lock:
            newest = sorted(self.records.values(), key=lambda d: -d.created)[:KEEP_RECORDS]
            self.records = {d.id: d for d in newest}
            text = json.dumps([asdict(d) for d in newest], indent=1)
        tmp = self.records_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, self.records_path)

    def _load(self) -> None:
        try:
            with open(self.records_path) as f:
                items = json.load(f)
        except (OSError, ValueError):
            return
        for item in items:
            d = Download(**item)
            if d.status in ("queued", "running"):
                d.status, d.message = "interrupted", "The server stopped; download it again to resume."
            self.records[d.id] = d


def _explain(error: str) -> str:
    if "not a bot" in error or "Sign in" in error:
        return error + "\nYouTube wants a signed-in browser: export its cookies to a file and start the server with MUSICFIT_YT_COOKIES=/path/to/cookies.txt."
    return error
