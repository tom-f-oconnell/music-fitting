"""Curated Aphex Twin catalogue used when MusicBrainz is unavailable.

Covers the six major studio albums plus the best-known EPs.  Durations are
track lengths in seconds from MusicBrainz (the earliest official release of
each album/EP, as :func:`~musicfit.dataset.catalog.musicbrainz_tracks` picks;
checked October 2026), used for matching YouTube candidates with a tolerance.
The three bonus tracks absent from those releases (Stone in Focus, marchromt38
fast, pthex) have approximate lengths.

SAW II tracks are officially untitled; they are listed under the widely used
fan names (as on MusicBrainz), with the "#n" form the official YouTube uploads
use as an alias.
"""

from __future__ import annotations

from typing import Callable

from .catalog import Work, dedupe_works, musicbrainz_tracks, normalize_title

ARTIST = "Aphex Twin"

# album title -> year, [(track title, duration_s), ...]
ALBUMS: dict[str, tuple[int, list[tuple[str, int]]]] = {
    "Selected Ambient Works 85-92": (1992, [
        ("Xtal", 291), ("Tha", 541), ("Pulsewidth", 227), ("Ageispolis", 321), ("i", 73),
        ("Green Calx", 362), ("Heliosphan", 291), ("We Are the Music Makers", 462),
        ("Schottkey 7th Path", 307), ("Ptolemy", 432), ("Hedphelym", 362), ("Delphium", 336),
        ("Actium", 455),
    ]),
    "Selected Ambient Works Volume II": (1994, [
        ("Cliffs", 448), ("Radiator", 394), ("Rhubarb", 464), ("Hankie", 280), ("Grass", 535),
        ("Mould", 211), ("Curtains", 531), ("Blur", 308), ("Weathered Stone", 415), ("Tree", 598),
        ("Domino", 438), ("White Blur 1", 160), ("Blue Calx", 440), ("Parallel Stripes", 481),
        ("Shiny Metal Rods", 334), ("Grey Stripe", 286), ("Z Twig", 125), ("Windowsill", 437),
        ("Hexagon", 358), ("Lichen", 255), ("Spots", 430), ("Tassels", 450), ("White Blur 2", 687),
        ("Matchsticks", 339), ("Stone in Focus", 614),
    ]),
    "...I Care Because You Do": (1995, [
        ("Acrid Avid Jam Shred", 449), ("The Waxen Pith", 284), ("Wax the Nip", 253),
        ("Icct Hedral (Edit)", 359), ("Ventolin (Video Version)", 263), ("Come On You Slags!", 336),
        ("Start as You Mean to Go On", 359), ("Wet Tip Hen Ax", 312), ("Mookid", 226),
        ("Alberto Balsalm", 305), ("Cow Cud Is a Twin", 326), ("Next Heap With", 283),
    ]),
    "Richard D. James Album": (1996, [
        ("4", 216), ("Cornish Acid", 135), ("Peek 824545201", 185), ("Fingerbib", 229),
        ("Carn Marth", 153), ("To Cure a Weakling Child", 244), ("Goon Gumpas", 122),
        ("Yellow Calx", 183), ("Girl/Boy Song", 288), ("Logan Rock Witch", 212),
        # US / later editions
        ("Milkman", 249), ("Inkey$", 84), ("Girl/Boy Song (£18 Snare Rush Mix)", 118), ("Beetles", 91),
        ("Girl/Boy Song (Redruth Mix)", 98),
    ]),
    "Drukqs": (2001, [
        ("Jynweythek", 143), ("Vordhosbn", 291), ("Kladfvgbung Micshk", 127), ("Omgyjya-Switch7", 292),
        ("Strotha Tynhe", 132), ("Gwely Mernans", 308), ("Bbydhyonchord", 153), ("Cock/Ver10", 318),
        ("Avril 14th", 126), ("Mt Saint Michel + Saint Michaels Mount", 490), ("Gwarek2", 407),
        ("Orban Eq Trx4", 95), ("Aussois", 13), ("Hy a Scullyas Lyf a Dhagrow", 135), ("Kesson Dalef", 80),
        ("54 Cymru Beats", 367), ("Btoum-Roumada", 118), ("Lornaderek", 31), ("QKThr", 87),
        ("Meltphace 6", 384), ("Bit 4", 25), ("Prep Gwarlek 3b", 79), ("Father", 57),
        ("Taking Control", 434), ("Petiatil Cx Htdui", 131), ("Ruglen Holon", 109), ("Afx237 v.7", 263),
        ("Ziggomatic 17", 515), ("Beskhu3epnm", 130), ("Nanou2", 204),
    ]),
    "Syro": (2014, [
        ("minipops 67 [120.2][source field mix]", 288), ("XMAS_EVET10 [120][thanaton3 mix]", 631),
        ("produk 29 [101]", 304), ("4 bit 9d api+e+6 [126.26]", 269), ("180db_ [130]", 191),
        ("CIRCLONT6A [141.98][syrobonkus mix]", 360), ("fz pseudotimestretch+e+3 [138.85]", 59),
        ("CIRCLONT14 [152.97][shrymoming mix]", 441), ("syro u473t8+e [141.98][piezoluminescence mix]", 393),
        ("PAPAT4 [155][pineal mix]", 259), ("s950tx16wasr10 [163.97][earth portal mix]", 362),
        ("aisatsana [102]", 322),
    ]),
}

EPS: dict[str, tuple[int, list[tuple[str, int]]]] = {
    "Come to Daddy": (1997, [
        ("Come to Daddy (Pappy Mix)", 262), ("Flim", 177), ("Come to Daddy (Little Lord Faulteroy Mix)", 230),
        ("Bucephalus Bouncing Ball", 346), ("To Cure a Weakling Child (Contour Regard)", 310),
        ("Funny Little Man", 238), ("Come to Daddy (Mummy Mix)", 265), ("IZ-US", 177),
    ]),
    "Windowlicker": (1999, [
        ("Windowlicker", 368), ("Equation", 346), ("Nannou", 254),
    ]),
    "Computer Controlled Acoustic Instruments pt2": (2015, [
        ("diskhat ALL prepared1mixed 13", 322), ("snar2", 20), ("diskhat1", 146), ("piano un1 arpej", 50),
        ("DISKPREPT4", 112), ("hat 2b 2012b", 85), ("disk aud1_12", 9), ("0035 1-Audio", 27),
        ("diskhat2", 38), ("piano un10 it happened", 108), ("hat5c 0001 rec-4", 286),
        ("diskprept1", 209), ("disk prep calrec2 barn dance [slo]", 262),
    ]),
    "Cheetah": (2016, [
        ("CHEETAHT2 [Ld spectrum]", 353), ("CHEETAHT7b", 403), ("CHEETA1b ms800", 27), ("CHEETA2 ms800", 37),
        ("CIRKLON3 [Kolkhoznaya mix]", 493), ("CIRKLON 1", 437), ("2X202-ST5", 279), ("marchromt38 fast", 44),
    ]),
    "Collapse": (2018, [
        ("T69 Collapse", 322), ("1st 44", 369), ("MT1 t29r2", 364),
        ("abundance10edit [2 R8's, FZ20m & a 909]", 380), ("pthex", 247),
    ]),
    "Blackbox Life Recorder 21f / In a Room7 F760": (2023, [
        ("Blackbox Life Recorder 21f", 266), ("zin2 test5", 159), ("In a Room7 F760", 233),
        ("Blackbox Life Recorder 22 [parallax mix]", 212),
        ("blackbox life recorder 21f 760 omc", 271), ("zin2 test5 omc ma2", 159), ("in a room7 F760 omc s", 230),
        ("blackbox life recorder 22 parallax mix 760 omc ma2", 213),
    ]),
}

# Works whose YouTube titles usually differ from the catalogue title.
SEARCH_ALIASES: dict[str, list[str]] = {
    "Cliffs": ["#1", "Untitled 1"], "Radiator": ["#2"], "Rhubarb": ["#3"], "Hankie": ["#4"], "Grass": ["#5"],
    "Mould": ["#6", "Mold"], "Curtains": ["#7"], "Blur": ["#8"], "Weathered Stone": ["#9"], "Tree": ["#10"],
    "Domino": ["#11"], "White Blur 1": ["#12"], "Blue Calx": ["#13"], "Parallel Stripes": ["#14"],
    "Shiny Metal Rods": ["#15"], "Grey Stripe": ["#16"], "Z Twig": ["#17"], "Windowsill": ["#18"],
    # from Hexagon on, the official uploads use two numberings, one off from the other
    "Hexagon": ["#19", "#20"], "Lichen": ["#20", "#21"], "Spots": ["#21", "#22"], "Tassels": ["#22", "#23"],
    "White Blur 2": ["#23", "#24"], "Matchsticks": ["#24", "#25"],
    "Equation": ["[Formula]", "ΔMi−1 = −αΣn=1NDi[n]", "∆Mᵢ⁻¹ = −∂ ∑ Dᵢ[n] [∑ Fⱼᵢ[n−1] + F extᵢ[n⁻¹]]"],
    "Girl/Boy Song (£18 Snare Rush Mix)": ["Girl/Boy (£18 Snare Rush Mix)", "Girl Boy (£18 Snarerush Mix)"],
    "Cock/Ver10": ["Ver10"],  # YouTube returns nothing for searches containing "Cock"
    "Milkman": ["Milk Man"], "Nanou2": ["Nanou 2"], "CIRKLON3 [Kolkhoznaya mix]": ["CIRKLON3 [ Колхозная mix ]"],
    "4": ["Four"],
}


def curated_works(include_eps: bool = True) -> list[Work]:
    works: list[Work] = []
    sources = dict(ALBUMS)
    if include_eps:
        sources.update(EPS)
    for album, (year, tracks) in sources.items():
        for i, (title, dur) in enumerate(tracks, start=1):
            works.append(Work(title=title, artist=ARTIST, album=album, duration_s=float(dur), year=year,
                              track_no=i, aliases=list(SEARCH_ALIASES.get(title, []))))
    return dedupe_works(works)


def aphex_twin_works(source: str = "auto", all_releases: bool = False, include_eps: bool = True,
                     log: Callable[[str], None] = print) -> list[Work]:
    """The works of the training set, with search aliases from the curated list.

    Args:
        source: "musicbrainz", "curated", or "auto" for MusicBrainz, then curated on failure.
        all_releases: with MusicBrainz, every official album/EP rather than the major ones.
        include_eps: also take EPs, not only albums.
        log: where to report which catalogue was used.

    Returns:
        De-duplicated works.
    """
    works = None
    if source in ("auto", "musicbrainz"):
        try:
            albums = None if all_releases else list(ALBUMS) + (list(EPS) if include_eps else [])
            works = musicbrainz_tracks(ARTIST, albums=albums, include_eps=include_eps)
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
            log(f"MusicBrainz catalogue: {len(works)} distinct works")
        except Exception as e:  # noqa: BLE001
            if source == "musicbrainz":
                raise
            log(f"MusicBrainz unavailable ({type(e).__name__}: {e}); using curated catalogue")
    if works is None:
        works = curated_works(include_eps=include_eps)
        log(f"Curated catalogue: {len(works)} distinct works")
    return dedupe_works(works)
