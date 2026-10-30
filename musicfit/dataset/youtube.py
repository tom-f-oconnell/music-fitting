"""Find and download audio for catalogue works from YouTube with yt-dlp.

Pipeline per work:

1. ``search``   – ``ytsearchN:"<artist> <title>"`` via yt-dlp (metadata only).
2. ``choose``   – score the candidates: title must contain the normalised work
   title, duration must be within a tolerance of the catalogue duration,
   official/"Topic" uploads are preferred, and covers / remixes / live /
   slowed / reaction videos are rejected unless the work itself is one.
3. ``download`` – best audio stream, re-encoded by ffmpeg to the requested
   codec, saved as ``<out_dir>/audio/<slug>.<ext>``.
4. ``manifest`` – one JSON line per work in ``<out_dir>/manifest.jsonl``
   with the chosen video, duration, file path and status; re-running skips
   works already downloaded, and a video id is never downloaded twice.

Everything that talks to the network is behind :class:`YouTubeClient` so it
can be mocked in tests.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

from .catalog import Work, normalize_title

try:  # imported lazily so the fitting code does not need yt-dlp installed
    import yt_dlp
except ImportError:  # pragma: no cover
    yt_dlp = None

REJECT_WORDS = (
    "remix", "rmx", "edit", "cover", "live", "slowed", "reverb", "sped up", "8d", "nightcore", "reaction",
    "tutorial", "remake", "recreat", "how to", "bootleg", "mashup", "instrumental", "karaoke", "lyrics",
    "piano version", "guitar", "sample", "breakdown", "analysis", "full album", "mix)", " mix ", "megamix",
    "vinyl rip", "loop", "1 hour", "10 hours", "extended", "flip",
)
PREFERRED_UPLOADERS = ("aphex twin", "warp records", "warprecords", " - topic")


@dataclass
class Candidate:
    video_id: str
    title: str
    duration_s: float | None
    uploader: str = ""
    channel_id: str = ""
    view_count: int | None = None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass
class ManifestEntry:
    work_key: str
    title: str
    artist: str
    album: str | None
    catalog_duration_s: float | None
    status: str  # ok | not_found | duplicate_video | download_failed | skipped
    video_id: str | None = None
    video_title: str | None = None
    video_url: str | None = None
    uploader: str | None = None
    video_duration_s: float | None = None
    file: str | None = None
    score: float | None = None
    error: str | None = None
    candidates: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# candidate selection (pure, testable)
# ---------------------------------------------------------------------------


def _allowed_words(work: Work) -> set[str]:
    """Reject-words that appear in the work's own title are allowed in candidates."""
    own = normalize_title(work.title) + " " + " ".join(normalize_title(a) for a in work.aliases)
    return {w for w in REJECT_WORDS if normalize_title(w) and normalize_title(w) in own}


def score_candidate(work: Work, cand: Candidate, duration_tol: float = 0.08, min_duration_tol: float = 6.0) -> float:
    """Return a score > 0 for acceptable candidates, <= 0 for rejected ones."""
    ct = normalize_title(cand.title)
    keys = [work.key] + [normalize_title(a) for a in work.aliases]
    keys = [k for k in keys if k]
    # title must contain the work title as a whole-word phrase
    matched = None
    for k in keys:
        if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", ct):
            matched = k
            break
    if matched is None:
        return 0.0
    score = 1.0 + len(matched) / max(len(ct), 1)  # tighter titles score higher

    lower = cand.title.lower()
    allowed = _allowed_words(work)
    for w in REJECT_WORDS:
        if w in allowed:
            continue
        if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", lower):
            return -1.0

    if work.duration_s and cand.duration_s:
        tol = max(duration_tol * work.duration_s, min_duration_tol)
        diff = abs(cand.duration_s - work.duration_s)
        if diff > tol:
            return -0.5
        score += 1.0 - diff / tol
    elif cand.duration_s is None:
        score -= 0.5

    up = (cand.uploader or "").lower()
    if any(p in up for p in PREFERRED_UPLOADERS):
        score += 2.0
    if cand.view_count:
        score += min(cand.view_count, 10_000_000) / 10_000_000  # small tie-breaker
    return score


def choose_candidate(work: Work, candidates: list[Candidate], **kw) -> tuple[Candidate | None, float]:
    best, best_score = None, 0.0
    for c in candidates:
        s = score_candidate(work, c, **kw)
        if s > best_score:
            best, best_score = c, s
    return best, best_score


def slugify(text: str, max_len: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return s[:max_len] or "untitled"


# ---------------------------------------------------------------------------
# yt-dlp wrapper
# ---------------------------------------------------------------------------


class YouTubeClient:
    def __init__(self, cookies_file: str | None = None, sleep_between: float = 1.0, quiet: bool = True,
                 extra_opts: dict | None = None):
        if yt_dlp is None:
            raise ImportError("yt-dlp is required: pip install yt-dlp")
        self.cookies_file = cookies_file
        self.sleep_between = sleep_between
        self.quiet = quiet
        self.extra_opts = extra_opts or {}

    def _opts(self, **more) -> dict:
        opts = {"quiet": self.quiet, "no_warnings": self.quiet, "noplaylist": True, "ignoreerrors": True}
        if self.cookies_file:
            opts["cookiefile"] = self.cookies_file
        opts.update(self.extra_opts)
        opts.update(more)
        return opts

    def search(self, query: str, n: int = 8) -> list[Candidate]:
        with yt_dlp.YoutubeDL(self._opts(extract_flat="in_playlist", skip_download=True)) as ydl:
            info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
        out = []
        for e in (info or {}).get("entries") or []:
            if not e or not e.get("id"):
                continue
            out.append(Candidate(
                video_id=e["id"], title=e.get("title") or "", duration_s=e.get("duration"),
                uploader=e.get("uploader") or e.get("channel") or "", channel_id=e.get("channel_id") or "",
                view_count=e.get("view_count"),
            ))
        return out

    def download_audio(self, video_id: str, out_path_no_ext: str, codec: str = "opus", quality: str = "0") -> str:
        """Download best audio and convert with ffmpeg; returns the written file path."""
        pp = [{"key": "FFmpegExtractAudio", "preferredcodec": codec, "preferredquality": quality}]
        opts = self._opts(format="bestaudio/best", outtmpl=out_path_no_ext + ".%(ext)s", postprocessors=pp,
                          ignoreerrors=False)
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        ext = {"vorbis": "ogg", "alac": "m4a", "aac": "m4a"}.get(codec, codec)
        path = f"{out_path_no_ext}.{ext}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"yt-dlp finished but {path} is missing")
        return path


# ---------------------------------------------------------------------------
# dataset builder
# ---------------------------------------------------------------------------


class Manifest:
    def __init__(self, path: str):
        self.path = path
        self.entries: dict[str, ManifestEntry] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        e = ManifestEntry(**json.loads(line))
                        self.entries[e.work_key] = e

    def downloaded_video_ids(self) -> set[str]:
        return {e.video_id for e in self.entries.values() if e.status == "ok" and e.video_id}

    def add(self, entry: ManifestEntry) -> None:
        self.entries[entry.work_key] = entry
        self.rewrite()

    def rewrite(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            for e in self.entries.values():
                f.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries.values():
            out[e.status] = out.get(e.status, 0) + 1
        return out


def build_dataset(
    works: list[Work],
    out_dir: str,
    client: YouTubeClient,
    codec: str = "opus",
    n_search: int = 8,
    dry_run: bool = False,
    retry_failed: bool = False,
    verbose: bool = True,
    duration_tol: float = 0.08,
) -> Manifest:
    """Search, de-duplicate and download every work; returns the updated manifest."""
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    manifest = Manifest(os.path.join(out_dir, "manifest.jsonl"))
    seen_videos = manifest.downloaded_video_ids()

    for i, work in enumerate(works, start=1):
        prev = manifest.entries.get(work.key)
        if prev is not None and (prev.status == "ok" or (prev.status != "download_failed" and not retry_failed)
                                 or (prev.status == "download_failed" and not retry_failed)):
            if not dry_run:
                continue
        label = f"[{i}/{len(works)}] {work.artist} - {work.title}"
        entry = ManifestEntry(work.key, work.title, work.artist, work.album, work.duration_s, status="not_found")
        try:
            queries = [f"{work.artist} {work.title}"] + [f"{work.artist} {a}" for a in work.aliases[:2]]
            candidates: list[Candidate] = []
            for q in queries:
                candidates += client.search(q, n_search)
                best, score = choose_candidate(work, candidates, duration_tol=duration_tol)
                if best is not None and score >= 2.0:
                    break
            # dedupe candidates by id, keep for the record
            uniq = {c.video_id: c for c in candidates}
            best, score = choose_candidate(work, list(uniq.values()), duration_tol=duration_tol)
            entry.candidates = [{"id": c.video_id, "title": c.title, "duration_s": c.duration_s,
                                 "uploader": c.uploader, "score": round(score_candidate(work, c, duration_tol), 3)}
                                for c in list(uniq.values())[:n_search]]
            if best is None:
                if verbose:
                    print(f"{label}: no acceptable candidate among {len(uniq)}")
            else:
                entry.video_id, entry.video_title, entry.video_url = best.video_id, best.title, best.url
                entry.uploader, entry.video_duration_s, entry.score = best.uploader, best.duration_s, score
                if best.video_id in seen_videos:
                    entry.status = "duplicate_video"
                    if verbose:
                        print(f"{label}: video {best.video_id} already downloaded for another work")
                elif dry_run:
                    entry.status = "skipped"
                    if verbose:
                        print(f"{label}: would download {best.url} ({best.title!r}, {best.duration_s}s, score {score:.2f})")
                else:
                    base = os.path.join(out_dir, "audio", slugify(f"{work.artist}_{work.title}"))
                    try:
                        entry.file = os.path.relpath(client.download_audio(best.video_id, base, codec), out_dir)
                        entry.status = "ok"
                        seen_videos.add(best.video_id)
                        if verbose:
                            print(f"{label}: downloaded {entry.file} from {best.url}")
                    except Exception as e:  # noqa: BLE001
                        entry.status, entry.error = "download_failed", f"{type(e).__name__}: {e}"
                        if verbose:
                            print(f"{label}: download failed: {entry.error}")
        except Exception as e:  # noqa: BLE001  (search failure, network down, ...)
            entry.status, entry.error = "download_failed", f"{type(e).__name__}: {e}"
            if verbose:
                print(f"{label}: search failed: {entry.error}")
        if not dry_run or prev is None:
            manifest.add(entry)
        if client.sleep_between and not dry_run:
            time.sleep(client.sleep_between)
    return manifest
