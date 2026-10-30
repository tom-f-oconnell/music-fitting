"""Dataset building: track catalogues (MusicBrainz / curated) and YouTube download via yt-dlp."""

from .catalog import Work, dedupe_works, musicbrainz_tracks, normalize_title
from .youtube import Candidate, Manifest, YouTubeClient, build_dataset, choose_candidate, score_candidate

__all__ = ["Work", "dedupe_works", "musicbrainz_tracks", "normalize_title", "Candidate", "Manifest",
           "YouTubeClient", "build_dataset", "choose_candidate", "score_candidate"]
