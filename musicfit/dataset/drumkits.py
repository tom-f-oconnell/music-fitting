"""Recorded drum kits to fit drums with, downloaded into ``<data>/drumkits/``.

Each is a freely licensed multisampled acoustic kit, fetched from a fixed address, checked
against a fixed checksum and unpacked like any other archive here (untrusted: nothing in it is
run). Reading one picks a single recording per drum and mixes it to mono (:func:`read_kit`).
Any other folder under ``<data>/drumkits/`` is a kit too, read by file names like a kit
passed to ``--drum-kit``.
"""

from __future__ import annotations

import os
from typing import Callable

from .library import DRUMKITS_DIR, MAX_DOWNLOAD_GB, MAX_UNPACK_RATIO, _download_checked, _unpack

DATA_ROOT = os.environ.get("MUSICFIT_DATA", "data")

RECORDED = {
    "salamander": {
        "name": "Salamander Drumkit",
        "url": "https://archive.org/download/SalamanderDrumkit/salamanderDrumkit.tar.bz2",
        "checksum": "md5:af8e2067668a7f438e7d981877fb771f",
        "size": 387611727,
        "license": "CC BY-SA 3.0",
        "credit": "Salamander Drumkit by Alexander Holm (RytmenPinne), CC BY-SA 3.0",
        "description": "An acoustic kit recorded at 44.1 kHz: kick, snare, toms, hi-hat, crash, ride, cowbell, "
                       "tambourine and more, 388 MB (CC BY-SA 3.0).",
    },
    "drskit": {
        "name": "DRSKit",
        "url": "https://drumgizmo.org/kits/DRSKit/drskit_minimal-1.0.tar.gz",
        "checksum": "sha256:7b5a97fe75eda2ee23e9edc32d3e5f40e4bf3815c43cfd07969e4f546a3399ed",
        "size": 478213963,
        "license": "CC BY 4.0",
        "credit": "DRSKit by the DrumGizmo team and Jes Eiler (DRSDrums), CC BY 4.0",
        "description": "An acoustic kit recorded for DrumGizmo, for anything from jazz to rock: kick, snare, three toms, "
                       "hi-hat, two crashes and a ride, 478 MB (CC BY 4.0).",
    },
}


def kits_root(root: str | None = None) -> str:
    return os.path.join(root or DATA_ROOT, DRUMKITS_DIR)


def fetch_drumkit(kit: str, root: str | None = None, on_progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Download and unpack a recorded kit; one already there isn't fetched again.

    Args:
        kit: a key of :data:`RECORDED`.
        root: the data directory.
        on_progress: called with (bytes done, bytes in all, file name); raising stops it.

    Returns:
        Its title and license, for the dataset's record.
    """
    k = RECORDED[kit]
    out_dir = os.path.join(kits_root(root), kit)
    name = os.path.basename(k["url"])
    dest = os.path.join(out_dir, "download", name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    _download_checked(k["url"], dest, k["checksum"], (lambda done: on_progress(done, k["size"], name)) if on_progress else None)
    _unpack(dest, out_dir, MAX_UNPACK_RATIO * MAX_DOWNLOAD_GB * 1e9)
    return {"title": k["name"], "license": k["license"]}


# Where each recorded kit's samples are, and which recording stands for each drum.
#  "prefix" (Salamander): files <prefix>_<dynamic>_<take>.wav, tried at forte first, then louder,
#   then softer; the overhead microphones hold the whole kit, as a listener hears it.
#  "drumgizmo" (DRSKit): an instrument folder per drum, with samples/<n>-<instrument>-<mic>.wav,
#   numbered from softest to loudest; a hit about three quarters of the way up, all its mics summed.
_LAYOUT = {
    "salamander": {"kind": "prefix", "folder": "salamanderDrumkit/OH", "drums": {
        "kick": "kick_OH", "snare": "snare_OH", "rim": "snareStick_OH", "closed hat": "hihatClosed_OH",
        "open hat": "hihatOpen_OH", "low tom": "loTom_OH", "high tom": "hiTom_OH", "crash": "crash1_OH",
        "ride": "ride1_OH", "cowbell": "cowbell"}},
    "drskit": {"kind": "drumgizmo", "folder": "drskit_minimal-1.0/DRSKit", "drums": {
        "kick": "Kdrum_with_contact", "snare": "Snare", "closed hat": "Hihat_closed_shank", "open hat": "Hihat_semi_open",
        "high tom": "Tom1", "mid tom": "Tom2", "low tom": "Tom3", "crash": "Crash_left_shank", "ride": "Ride_tip"}},
}
_DYNAMICS = ("F", "FF", "MP", "P", "PP")


def installed(root: str | None = None) -> list[dict]:
    """The kits on disk: recorded ones that have been downloaded, and folders of samples.

    Returns:
        [{"id", "name", "kind": "recorded" or "folder", "credit"}], recorded ones first.
    """
    base = kits_root(root)
    out = [{"id": k, "name": v["name"], "kind": "recorded", "credit": v["credit"]}
           for k, v in RECORDED.items() if kit_path(k, root)]
    if os.path.isdir(base):
        from ..drumkit import _drum_of

        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if name in RECORDED or name.startswith(".") or not os.path.isdir(path):
                continue
            if any(_drum_of(f) for f in os.listdir(path)):
                out.append({"id": name, "name": name, "kind": "folder", "credit": ""})
    return out


def kit_path(kit: str, root: str | None = None) -> str | None:
    """Where a kit named ``kit`` is under the data directory, if it's there."""
    path = os.path.join(kits_root(root), kit)
    if kit in RECORDED:
        return path if os.path.isdir(os.path.join(path, _LAYOUT[kit]["folder"])) else None
    return path if os.path.basename(kit) == kit and not kit.startswith(".") and os.path.isdir(path) else None


def read_recorded(kit: str, sr: int, seconds: float, root: str | None = None) -> dict:
    """One recording per drum of a downloaded kit, mono, at ``sr``, cut to ``seconds``.

    The first read is saved next to the kit, as reading it means decoding hundreds of files.

    Raises:
        ValueError: the kit hasn't been downloaded.
    """
    import numpy as np

    path = kit_path(kit, root)
    if path is None:
        raise ValueError(f"the {RECORDED[kit]['name']} kit hasn't been downloaded: get it from the Data menu")
    cache = os.path.join(path, f"musicfit_{sr}_{seconds:g}s.npz")
    if os.path.exists(cache):
        with np.load(cache) as z:
            return {name.replace("_", " "): z[name] for name in z.files}
    layout = _LAYOUT[kit]
    folder = os.path.join(path, layout["folder"])
    read = _read_prefixed if layout["kind"] == "prefix" else _read_drumgizmo
    sounds = {drum: x for drum, where in layout["drums"].items() if (x := read(folder, where, sr, seconds)) is not None}
    np.savez(cache + ".part.npz", **{name.replace(" ", "_"): x for name, x in sounds.items()})
    os.replace(cache + ".part.npz", cache)
    return sounds


def _read_prefixed(folder: str, prefix: str, sr: int, seconds: float):
    import re

    from ..audio_io import load_audio

    files = os.listdir(folder)
    for dyn in _DYNAMICS:
        found = sorted(f for f in files if re.fullmatch(rf"{re.escape(prefix)}_{dyn}_\d+\.(wav|flac)", f))
        if found:
            return load_audio(os.path.join(folder, found[0]), sr, 0.0, seconds).numpy().astype("float64")
    return None


def _read_drumgizmo(folder: str, instrument: str, sr: int, seconds: float):
    import re

    import numpy as np

    from ..audio_io import load_audio

    samples = os.path.join(folder, instrument, "samples")
    if not os.path.isdir(samples):
        return None
    files = [f for f in os.listdir(samples) if re.fullmatch(rf"\d+-{re.escape(instrument)}-\w+\.wav", f)]
    numbers = sorted({int(f.split("-")[0]) for f in files})
    if not numbers:
        return None
    n = numbers[round(0.75 * (len(numbers) - 1))]
    mics = [load_audio(os.path.join(samples, f), sr, 0.0, seconds).numpy() for f in sorted(files) if f.startswith(f"{n}-")]
    length = max(len(m) for m in mics)
    return sum(np.pad(m, (0, length - len(m))) for m in mics).astype("float64")


def catalogue(root: str | None = None) -> list[dict]:
    """Every kit, for choosing: built-in ones, then those on disk, then those that can be downloaded.

    Returns:
        [{"id", "name", "kind": "machine", "imitation", "recorded" or "folder", "drums", "credit",
        "installed"}]; ``drums`` lists the drums it has, for kits that are installed.
    """
    from ..drumkit import BUILT_IN, BUILT_IN_NAMES, DRUMS, KIT_SECONDS, load_kit_folder, make_kit

    out = []
    for k in BUILT_IN:
        out.append({"id": k, "name": BUILT_IN_NAMES[k], "kind": "imitation" if "imitation" in BUILT_IN_NAMES[k] else "machine",
                    "drums": make_kit(k, 16000).names, "credit": "", "installed": True})
    for k in installed(root):
        try:
            sounds = read_recorded(k["id"], 16000, KIT_SECONDS, root) if k["kind"] == "recorded" else load_kit_folder(kit_path(k["id"], root), 8000)
        except (OSError, ValueError):
            continue
        out.append({**k, "drums": [d for d, _ in DRUMS if d in sounds], "installed": True})
    have = {k["id"] for k in out}
    out += [{"id": k, "name": v["name"], "kind": "recorded", "drums": [], "credit": v["credit"], "installed": False}
            for k, v in RECORDED.items() if k not in have]
    return out


def default_kits(root: str | None = None) -> list[str]:
    """The kits a fit uses unless told otherwise: the drum machines, whatever is downloaded.

    The recorded kits find hits no better on BabySlakh and fit electronic music far worse
    (design.md), so having them doesn't change the default.
    """
    from ..drumkit import DEFAULT_KITS

    return DEFAULT_KITS.split(",")
