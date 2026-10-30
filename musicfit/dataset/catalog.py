"""Track catalogues: which works to look for on YouTube.

Two sources:

* :func:`musicbrainz_tracks` queries the MusicBrainz web service for every
  official release of an artist (optionally restricted to a list of album
  titles) and returns its tracks with durations.
* A curated built-in list in :mod:`musicfit.dataset.aphex_twin_catalog`,
  used when MusicBrainz is unreachable or when ``--source curated`` is given.

Both return :class:`Work` records, de-duplicated by normalised title so that a
track appearing on several releases (album + compilation + remaster) becomes
one work.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field

import requests

MB_API = "https://musicbrainz.org/ws/2"
USER_AGENT = "musicfit/0.1 (https://github.com/ - research dataset builder)"


@dataclass
class Work:
    title: str
    artist: str
    album: str | None = None
    duration_s: float | None = None
    year: int | None = None
    track_no: int | None = None
    mbid: str | None = None
    aliases: list[str] = field(default_factory=list)  # other titles this work has been released under

    @property
    def key(self) -> str:
        return normalize_title(self.title)

    def as_dict(self) -> dict:
        return asdict(self)


_PAREN_NOISE = re.compile(
    r"\((?:remaster(?:ed)?|\d{4} remaster(?:ed)?|album version|original mix|official (?:audio|video)|hd|hq)[^)]*\)",
    re.I,
)
_BRACKET_NOISE = re.compile(r"\[(?:remaster(?:ed)?|official (?:audio|video)|hd|hq)[^\]]*\]", re.I)
_TEMPO = re.compile(r"\[\d{2,3}(?:\.\d+)?\]")  # Syro-style "[120.2]" bpm tags, absent on MusicBrainz
_ARTIST_PREFIX = re.compile(r"^(?:aphex twin|afx|polygon window|caustic window|the tuss)\s*[-–—:]\s*", re.I)


def normalize_title(title: str) -> str:
    """Normalise a track title for duplicate detection.

    Unicode-folds, lower-cases, strips "(2019 Remaster)"-style suffixes and an
    "Artist - " prefix, collapses punctuation/whitespace.  Deliberately keeps
    bracketed *content* that distinguishes real works (e.g. "Xtal" vs
    "Xtal (live)") only when it is not in the noise list above.
    """
    t = unicodedata.normalize("NFKD", title)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower().strip()
    t = _ARTIST_PREFIX.sub("", t)
    t = _PAREN_NOISE.sub("", t)
    t = _BRACKET_NOISE.sub("", t)
    t = _TEMPO.sub("", t)
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def release_key(title: str) -> str:
    """Looser key for matching release names across sources.

    On top of :func:`normalize_title`, "pt2"/"pt 2" becomes "part 2" and a
    trailing "EP"/"E.P." is dropped, so "Cheetah EP" matches "Cheetah".
    """
    t = re.sub(r"\bpt ?(\d+)", r"part \1", normalize_title(title))
    return re.sub(r"(?: e ?p)+$", "", t)


def dedupe_works(works: list[Work]) -> list[Work]:
    """Collapse works with identical normalised titles, keeping the first and recording aliases."""
    seen: dict[str, Work] = {}
    for w in works:
        k = w.key
        if not k:
            continue
        if k in seen:
            kept = seen[k]
            if w.title != kept.title and w.title not in kept.aliases:
                kept.aliases.append(w.title)
            if kept.duration_s is None and w.duration_s is not None:
                kept.duration_s = w.duration_s
        else:
            seen[k] = w
    return list(seen.values())


# ---------------------------------------------------------------------------
# MusicBrainz
# ---------------------------------------------------------------------------


class MusicBrainz:
    """Tiny MusicBrainz client that honours the 1 request/second rule."""

    def __init__(self, user_agent: str = USER_AGENT, min_interval: float = 1.1, timeout: float = 30.0):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.min_interval = min_interval
        self.timeout = timeout
        self._last = 0.0

    def get(self, path: str, **params) -> dict:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        params.setdefault("fmt", "json")
        for attempt in range(4):
            r = self.session.get(f"{MB_API}/{path}", params=params, timeout=self.timeout)
            self._last = time.time()
            if r.status_code == 503:  # rate limited
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}

    def artist_id(self, name: str) -> str:
        data = self.get("artist", query=f'artist:"{name}"', limit=5)
        for a in data.get("artists", []):
            if a["name"].lower() == name.lower():
                return a["id"]
        if data.get("artists"):
            return data["artists"][0]["id"]
        raise LookupError(f"artist not found on MusicBrainz: {name!r}")

    def release_groups(self, artist_mbid: str, types=("album", "ep")) -> list[dict]:
        out: list[dict] = []
        offset = 0
        while True:
            data = self.get(f"release-group", artist=artist_mbid, type="|".join(types), limit=100, offset=offset)
            out.extend(data.get("release-groups", []))
            offset += 100
            if offset >= data.get("release-group-count", 0):
                break
        return out

    def best_release(self, release_group_mbid: str) -> dict | None:
        """Pick the earliest official release of a group, with its recordings."""
        data = self.get("release", **{"release-group": release_group_mbid, "status": "official", "limit": 100,
                                      "inc": "recordings+media"})
        rels = [r for r in data.get("releases", []) if r.get("media")]
        if not rels:
            return None
        rels.sort(key=lambda r: (r.get("date") or "9999", -sum(m.get("track-count", 0) for m in r["media"])))
        # media list from the search endpoint lacks tracks; fetch the chosen release fully
        return self.get(f"release/{rels[0]['id']}", inc="recordings+media+release-groups")


def musicbrainz_tracks(
    artist: str,
    albums: list[str] | None = None,
    include_eps: bool = True,
    client: MusicBrainz | None = None,
    verbose: bool = True,
) -> list[Work]:
    """All tracks of an artist's official albums (and EPs) from MusicBrainz.

    With an explicit ``albums`` list, release names are matched with
    :func:`release_key`, and singles are searched too (some EPs, e.g.
    "Windowlicker", are typed as singles on MusicBrainz).  When several release
    groups match one name, only the first album, else EP, else single is used.
    """
    mb = client or MusicBrainz()
    aid = mb.artist_id(artist)
    wanted = {release_key(a) for a in albums} if albums else None
    works: list[Work] = []
    types = ("album", "ep") if include_eps else ("album",)
    if wanted is not None and include_eps:
        types += ("single",)
    rank = {"Album": 0, "EP": 1, "Single": 2}
    groups = sorted(mb.release_groups(aid, types), key=lambda g: rank.get(g.get("primary-type"), 3))
    done: set[str] = set()
    for g in groups:
        if "secondary-types" in g and g["secondary-types"]:
            continue  # skip live / compilation / remix / soundtrack groups
        key = release_key(g["title"])
        if wanted is not None and (key not in wanted or key in done):
            continue
        done.add(key)
        rel = mb.best_release(g["id"])
        if rel is None:
            continue
        year = int(rel["date"][:4]) if rel.get("date") else None
        if verbose:
            print(f"[musicbrainz] {g['title']} ({year}): ", end="", flush=True)
        n = 0
        for medium in rel.get("media", []):
            for t in medium.get("tracks", []):
                rec = t.get("recording", {})
                length = t.get("length") or rec.get("length")
                works.append(Work(
                    title=t.get("title") or rec.get("title"),
                    artist=artist,
                    album=g["title"],
                    duration_s=length / 1000.0 if length else None,
                    year=year,
                    track_no=t.get("position"),
                    mbid=rec.get("id"),
                ))
                n += 1
        if verbose:
            print(f"{n} tracks")
    return dedupe_works(works)


def save_catalog(works: list[Work], path: str) -> None:
    with open(path, "w") as f:
        json.dump([w.as_dict() for w in works], f, indent=2, ensure_ascii=False)


def load_catalog(path: str) -> list[Work]:
    with open(path) as f:
        return [Work(**d) for d in json.load(f)]
