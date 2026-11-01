"""The audio that fits can run on: shipped examples, and datasets cached on disk.

Each subdirectory of the library's root is a dataset, in whichever layout it came in:
a ``manifest.jsonl`` from the YouTube pipeline, BabySlakh's ``Track*/`` directories, or
plain audio files. Nothing here depends on how a dataset got there, so ones built with
``scripts/build_aphex_twin_dataset.py`` or by hand are listed too.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from typing import Callable

import requests
import soundfile as sf

AUDIO_EXTS = (".opus", ".ogg", ".oga", ".flac", ".wav", ".mp3", ".m4a", ".aac", ".webm")
ARCHIVE_EXTS = (".tar.gz", ".tgz", ".tar", ".zip")
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
EXAMPLES_ID = "examples"

ZENODO_RECORD_URL = "https://zenodo.org/api/records/{}"
MAX_DOWNLOAD_GB = float(os.environ.get("MUSICFIT_MAX_DOWNLOAD_GB", 10))
MAX_UNPACK_RATIO = 5  # an archive may unpack to this many times the download limit


@dataclass
class Track:
    id: str  # "<dataset>/<path inside it>"
    title: str
    duration_s: float | None = None
    offset: float = 0.0  # what to fill the form with when it's chosen
    length: float = 4.0
    voices: int = 1
    note: str = ""


# Clips cut from BabySlakh by scripts/make_examples.py (CC BY 4.0, see examples/ATTRIBUTION.md).
SHIPPED_EXAMPLES = [
    Track(f"{EXAMPLES_ID}/sparse_piano.flac", "Sparse piano", note="Few notes on one instrument: what the fitter does best."),
    Track(f"{EXAMPLES_ID}/busier_piano.flac", "Busier piano", note="One instrument with about twice as many notes."),
    Track(f"{EXAMPLES_ID}/bass_line.flac", "Bass line", note="A bass stem. Bass is harder: expect octave errors."),
    Track(f"{EXAMPLES_ID}/full_band.flac", "Full band",
          note="A dense mix. The fitter alone finds few notes here: try the automatic transcription under Advanced."),
]
# The clips used to develop the fitter. They are copyrighted, so they aren't in the repository;
# they are listed if the training set has been downloaded.
LOCAL_EXAMPLES = [
    Track("aphex_twin/audio/Aphex_Twin_aisatsana.opus", "aisatsana", offset=40.0, voices=2, note="Sparse piano, from your download."),
    Track("aphex_twin/audio/Aphex_Twin_Cornish_Acid.opus", "Cornish Acid", offset=30.0, voices=2,
          note="Acid line and drums, from your download. Try the mono lead option under Advanced."),
    Track("aphex_twin/audio/Aphex_Twin_produk_29.opus", "produk 29", offset=60.0, voices=2, note="Synths and drums, from your download."),
    Track("aphex_twin/audio/Aphex_Twin_Radiator.opus", "Radiator", offset=60.0, voices=2, note="Pads, from your download."),
]
# The data the project was developed on; see downloads.py for how each is fetched.
PRESETS = {
    "aphex_twin": {"name": "Aphex Twin", "source": "aphex_twin", "query": "Aphex Twin",
                   "description": "The YouTube training set: about 150 tracks from MusicBrainz's catalogue, about 1 GB."},
    "babyslakh": {"name": "BabySlakh", "source": "zenodo", "query": "4603870", "complete_when": "babyslakh_16k",
                  "description": "20 songs rendered from MIDI with stems and true notes, from Zenodo (CC BY 4.0), 883 MB."},
}


def slug(text: str) -> str:
    """A dataset directory name for this text."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "dataset"


class Library:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._cache: dict[str, tuple] = {}

    def resolve(self, track_id: str) -> str:
        """The path of a track, if the id names an audio file inside a dataset.

        Raises:
            ValueError: otherwise. Ids come from the client, so this is what keeps them in the library.
        """
        dataset, _, rel = track_id.partition("/")
        if not re.fullmatch(r"\w+(-\w+)*", dataset) or not rel:
            raise ValueError(f"no such track: {track_id}")
        base = os.path.realpath(EXAMPLES_DIR if dataset == EXAMPLES_ID else os.path.join(self.root, dataset))
        path = os.path.realpath(os.path.join(base, rel))
        if os.path.commonpath([base, path]) != base or not path.lower().endswith(AUDIO_EXTS) or not os.path.isfile(path):
            raise ValueError(f"no such track: {track_id}")
        return path

    def title(self, track_id: str) -> str:
        """A track's title, for naming the job that fits it."""
        for dataset in self.datasets():
            for track in dataset["tracks"]:
                if track["id"] == track_id:
                    return track["title"]
        return os.path.basename(track_id)

    def info(self, dataset: str) -> dict:
        """What a dataset was downloaded from, if it was downloaded by the web app."""
        try:
            with open(os.path.join(self.root, dataset, "dataset.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def write_info(self, dataset: str, info: dict) -> None:
        os.makedirs(os.path.join(self.root, dataset), exist_ok=True)
        with open(os.path.join(self.root, dataset, "dataset.json"), "w") as f:
            json.dump(info, f, indent=2)

    def datasets(self) -> list[dict]:
        """The examples, then every dataset with audio in it, as dicts for the web app."""
        shipped = [t for t in SHIPPED_EXAMPLES if os.path.isfile(os.path.join(EXAMPLES_DIR, t.id.partition("/")[2]))]
        local = [t for t in LOCAL_EXAMPLES if os.path.isfile(os.path.join(self.root, t.id))]
        out = [{"id": EXAMPLES_ID, "name": "Examples", "source": "", "query": "", "limit": None, "size_bytes": 0,
                "tracks": [asdict(t) for t in shipped + local]}]
        for name in sorted(os.listdir(self.root)):
            base = os.path.join(self.root, name)
            if name.startswith(".") or not os.path.isdir(base):
                continue
            tracks, size = self._scan(name, base)
            if not tracks:
                continue
            preset, info = PRESETS.get(name, {}), self.info(name)
            out.append({"id": name, "name": info.get("name") or preset.get("name") or name,
                        "source": info.get("source") or preset.get("source", ""),
                        "query": info.get("query") or preset.get("query", ""), "limit": info.get("limit"),
                        "size_bytes": size, "tracks": [asdict(t) for t in tracks]})
        return out

    def _scan(self, name: str, base: str) -> tuple[list[Track], int]:
        # listing a dataset reads every file's header, so it is only redone when something changed
        stamp = tuple(os.path.getmtime(p) if os.path.exists(p) else 0
                      for p in (base, os.path.join(base, "manifest.jsonl"), os.path.join(base, "audio")))
        cached = self._cache.get(name)
        if cached and cached[0] == stamp:
            return cached[1], cached[2]
        tracks, size = _scan_dataset(name, base)
        self._cache[name] = (stamp, tracks, size)
        return tracks, size


def _scan_dataset(name: str, base: str) -> tuple[list[Track], int]:
    tracks: list[Track] = []
    size = 0
    listed = _manifest_tracks(name, base)
    for d, subdirs, files in os.walk(base):
        subdirs[:] = sorted(s for s in subdirs if not s.endswith(".part") and s != "download")
        size += sum(os.path.getsize(os.path.join(d, f)) for f in files)
        if listed:
            continue
        rel = os.path.relpath(d, base)
        if "metadata.yaml" in files and "mix.wav" in files:
            tracks += _slakh_tracks(name, d, rel)
        elif os.path.basename(d) == "stems" and "metadata.yaml" in os.listdir(os.path.dirname(d)):
            continue  # listed with their track, by name
        else:
            for f in sorted(files):
                if f.lower().endswith(AUDIO_EXTS):
                    path = os.path.normpath(os.path.join(rel, f))
                    tracks.append(Track(f"{name}/{path}", os.path.splitext(f)[0], _duration(os.path.join(d, f))))
    return listed or tracks, size


def _manifest_tracks(name: str, base: str) -> list[Track]:
    path = os.path.join(base, "manifest.jsonl")
    if not os.path.exists(path):
        return []
    tracks = []
    with open(path) as f:
        for line in f:
            e = json.loads(line) if line.strip() else {}
            if e.get("status") == "ok" and e.get("file") and os.path.exists(os.path.join(base, e["file"])):
                artist, title = e.get("artist") or "", e.get("title") or e["file"]
                label = title if artist.lower() in title.lower() else f"{artist} - {title}"
                tracks.append(Track(f"{name}/{e['file']}", label, e.get("video_duration_s")))
    return sorted(tracks, key=lambda t: t.title.lower())


def _slakh_tracks(name: str, d: str, rel: str) -> list[Track]:
    import yaml

    track = os.path.basename(d)
    with open(os.path.join(d, "metadata.yaml")) as f:
        stems = (yaml.safe_load(f) or {}).get("stems") or {}
    out = [Track(f"{name}/{rel}/mix.wav", f"{track}: mix", _duration(os.path.join(d, "mix.wav")))]
    for stem, meta in sorted(stems.items()):
        wav = os.path.join(d, "stems", f"{stem}.wav")
        if os.path.exists(wav):  # the metadata's audio_rendered isn't reliable (design.md)
            out.append(Track(f"{name}/{rel}/stems/{stem}.wav", f"{track}: {meta.get('inst_class', stem)}", _duration(wav)))
    return out


def _duration(path: str) -> float | None:
    try:
        return sf.info(path).duration
    except (RuntimeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Zenodo
# ---------------------------------------------------------------------------


def zenodo_record_id(text: str) -> str:
    """The record number in "4603870" or in a link to a Zenodo record."""
    m = re.fullmatch(r"\s*(\d+)\s*", text) or re.fullmatch(r"\s*https?://(?:www\.)?zenodo\.org/records?/(\d+)\S*\s*", text)
    if not m:
        raise ValueError("expected a Zenodo record number or a zenodo.org/records/... link")
    return m.group(1)


def fetch_zenodo(record: str, out_dir: str, on_progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Download a Zenodo record's audio files and archives into a dataset, and unpack the archives.

    Files already there with the right checksum aren't fetched again, and an interrupted
    download continues where it stopped.

    Args:
        record: record number or link.
        out_dir: dataset directory.
        on_progress: called with (bytes done, bytes in all, file name); raising stops the download.

    Returns:
        The record's title and license.

    Raises:
        ValueError: nothing usable in the record, too large, or a bad checksum.
    """
    response = requests.get(ZENODO_RECORD_URL.format(zenodo_record_id(record)), timeout=30)
    response.raise_for_status()
    meta = response.json()
    files = [f for f in meta.get("files", []) if f["key"].lower().endswith(AUDIO_EXTS + ARCHIVE_EXTS)]
    if not files:
        raise ValueError("this record has no audio files or archives")
    total = sum(f["size"] for f in files)
    if total > MAX_DOWNLOAD_GB * 1e9:
        raise ValueError(f"this record is {total / 1e9:.1f} GB, over the limit of {MAX_DOWNLOAD_GB:g} GB "
                         "(set MUSICFIT_MAX_DOWNLOAD_GB to change it)")
    os.makedirs(os.path.join(out_dir, "download"), exist_ok=True)
    done = 0
    for f in files:
        key = os.path.basename(f["key"])
        audio = key.lower().endswith(AUDIO_EXTS)
        dest = os.path.join(out_dir, "audio" if audio else "download", key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        report = (lambda d, base=done, name=key: on_progress(base + d, total, name)) if on_progress else None
        _download_checked(f["links"]["self"], dest, f["checksum"], report)
        done += f["size"]
        if not audio:
            _unpack(dest, out_dir, MAX_UNPACK_RATIO * MAX_DOWNLOAD_GB * 1e9)
    return {"title": (meta.get("metadata") or {}).get("title"), "license": ((meta.get("metadata") or {}).get("license") or {}).get("id")}


def _digest(path: str, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _download_checked(url: str, dest: str, checksum: str, report: Callable[[int], None] | None) -> None:
    algorithm, _, expected = checksum.partition(":")
    if os.path.exists(dest) and _digest(dest, algorithm) == expected:
        return
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # already complete
            have = -1
        else:
            r.raise_for_status()
            have = have if r.status_code == 206 else 0
            with open(part, "ab" if have else "wb") as out:
                for chunk in r.iter_content(1 << 20):
                    out.write(chunk)
                    have += len(chunk)
                    if report:
                        report(have)
    if _digest(part, algorithm) != expected:
        os.unlink(part)
        raise ValueError(f"{os.path.basename(dest)} failed its checksum, so it was deleted; try again")
    os.replace(part, dest)


def _unpack(archive: str, out_dir: str, max_bytes: float) -> None:
    """Unpack into <out_dir>/<archive name>, atomically, unless that's already there.

    Archives are untrusted: the tar "data" filter and zipfile both refuse members that
    would land outside the destination, and nothing inside is ever executed.
    """
    name = os.path.basename(archive)
    for ext in ARCHIVE_EXTS:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    dest = os.path.join(out_dir, name)
    if os.path.isdir(dest):
        return
    part = dest + ".part"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    if archive.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > max_bytes:
                raise ValueError(f"{os.path.basename(archive)} unpacks to too much data")
            z.extractall(part)
    else:
        with tarfile.open(archive) as t:
            if sum(m.size for m in t.getmembers()) > max_bytes:
                raise ValueError(f"{os.path.basename(archive)} unpacks to too much data")
            t.extractall(part, filter="data")
    only = os.listdir(part)
    if only == [name] and os.path.isdir(os.path.join(part, name)):  # the archive has one top directory
        os.replace(os.path.join(part, name), dest)
        os.rmdir(part)
    else:
        os.replace(part, dest)
