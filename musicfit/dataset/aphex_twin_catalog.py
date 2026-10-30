"""Curated Aphex Twin catalogue used when MusicBrainz is unavailable.

Covers the six major studio albums plus the best-known EPs.  Durations are the
album-version lengths in seconds (approximate, used only for matching YouTube
candidates with a tolerance), taken from the standard CD editions.

SAW II tracks are officially untitled; they are listed under the widely used
fan names (which is how they are titled on YouTube), with the "#n" form as an
alias.
"""

from __future__ import annotations

from .catalog import Work, dedupe_works

ARTIST = "Aphex Twin"

# album title -> year, [(track title, duration_s), ...]
ALBUMS: dict[str, tuple[int, list[tuple[str, int]]]] = {
    "Selected Ambient Works 85-92": (1992, [
        ("Xtal", 294), ("Tha", 542), ("Pulsewidth", 227), ("Ageispolis", 321), ("i", 73),
        ("Green Calx", 364), ("Heliosphan", 294), ("We Are the Music Makers", 462),
        ("Schottkey 7th Path", 307), ("Ptolemy", 432), ("Hedphelym", 362), ("Delphium", 326),
        ("Actium", 456),
    ]),
    "Selected Ambient Works Volume II": (1994, [
        ("Cliffs", 445), ("Radiator", 383), ("Rhubarb", 464), ("Hankie", 341), ("Grass", 586),
        ("Mould", 330), ("Curtains", 582), ("Blur", 319), ("Weathered Stone", 354), ("Tree", 625),
        ("Domino", 243), ("White Blur 1", 324), ("Blue Calx", 434), ("Parallel Stripes", 428),
        ("Shiny Metal Rods", 254), ("Grey Stripe", 341), ("Z Twig", 164), ("Windowsill", 456),
        ("Spots", 374), ("Tassels", 393), ("White Blur 2", 679), ("Matchsticks", 234),
        ("Lichen", 317), ("Stone in Focus", 614), ("Hexagon", 444),
    ]),
    "...I Care Because You Do": (1995, [
        ("Acrid Avid Jam Shred", 464), ("The Waxen Pith", 289), ("Wax the Nip", 268),
        ("Icct Hedral (Edit)", 367), ("Ventolin (Video Version)", 258), ("Come On You Slags!", 349),
        ("Start as You Mean to Go On", 366), ("Wet Tip Hen Ax", 316), ("Mookid", 236),
        ("Alberto Balsalm", 310), ("Cow Cud Is a Twin", 342), ("Next Heap With", 283),
    ]),
    "Richard D. James Album": (1996, [
        ("4", 237), ("Cornish Acid", 134), ("Peek 824545201", 183), ("Fingerbib", 227),
        ("Carn Marth", 153), ("To Cure a Weakling Child", 243), ("Goon Gumpas", 123),
        ("Yellow Calx", 180), ("Girl/Boy Song", 288), ("Logan Rock Witch", 224),
        # US / later editions
        ("Milkman", 253), ("Inkey$", 266), ("Girl/Boy (£18 Snare Rush Mix)", 216), ("Beetles", 184),
    ]),
    "Drukqs": (2001, [
        ("Jynweythek", 134), ("Vordhosbn", 283), ("Kladfvgbung Micshk", 122), ("Omgyjya-Switch7", 290),
        ("Strotha Tynhe", 122), ("Gwely Mernans", 310), ("Bbydhyonchord", 155), ("Cock/Ver10", 318),
        ("Avril 14th", 125), ("Mt Saint Michel + Saint Michaels Mount", 480), ("Gwarek2", 400),
        ("Orban Eq Trx4", 94), ("Aussois", 16), ("Hy a Scullyas Lyf a Dhagrow", 131), ("Kesson Dalef", 73),
        ("54 Cymru Beats", 360), ("Btoum-Roumada", 116), ("Lornaderek", 30), ("QKThr", 85),
        ("Meltphace 6", 375), ("Bit 4", 20), ("Prep Gwarlek 3b", 81), ("Father", 48),
        ("Taking Control", 433), ("Petiatil Cx Htdui", 130), ("Ruglen Holon", 103), ("Afx237 v.7", 262),
        ("Ziggomatic 17", 514), ("Beskhu3epnm", 120), ("Nanou2", 203),
    ]),
    "Syro": (2014, [
        ("minipops 67 [120.2][source field mix]", 287), ("XMAS_EVET10 [120][thanaton3 mix]", 626),
        ("produk 29 [101]", 306), ("4 bit 9d api+e+6 [126.26]", 266), ("180db_ [130]", 190),
        ("CIRCLONT6A [141.98][syrobonkers mix]", 359), ("fz pseudotimestretch+e+3 [138.85]", 58),
        ("CIRCLONT14 [152.97][shrymoming mix]", 440), ("syro u473t8+e [141.98][piezoluminescence mix]", 374),
        ("PAPAT4 [155][pineal mix]", 263), ("s950tx16wasr10 [163.97][earth portal mix]", 364),
        ("aisatsana [102]", 321),
    ]),
}

EPS: dict[str, tuple[int, list[tuple[str, int]]]] = {
    "Come to Daddy": (1997, [
        ("Come to Daddy (Pappy Mix)", 262), ("Flim", 177), ("Come to Daddy (Little Lord Faulteroy Mix)", 232),
        ("Bucephalus Bouncing Ball", 345), ("To Cure a Weakling Child (Contour Regard)", 242),
        ("Funny Little Man", 236), ("Come to Daddy (Mummy Mix)", 237), ("IZ-US", 179),
    ]),
    "Windowlicker": (1999, [
        ("Windowlicker", 367), ("Equation", 343), ("Nannou", 258),
    ]),
    "Computer Controlled Acoustic Instruments pt2": (2015, [
        ("diskhat ALL prepared1mixed 13", 174), ("snar2", 90), ("diskhat1", 229), ("piano un1 arpej", 89),
        ("DISKPREPT4", 162), ("hat 2b 2012b", 73), ("disk aud1_12", 70), ("0035 1-Audio", 43),
        ("diskhat2", 174), ("piano un10 it happened", 90), ("hat5c 0001 rec-4", 135),
        ("diskprept1", 232), ("disk prep calrec2 barn dance [slo]", 133),
    ]),
    "Cheetah": (2016, [
        ("CHEETAHT2 [Ld spectrum]", 337), ("CHEETAHT7b", 357), ("CHEETA1b ms800", 51), ("CHEETA2 ms800", 48),
        ("CIRKLON3 [Kolkhoznaya mix]", 392), ("CIRKLON 1", 331), ("2X202-ST5", 383), ("marchromt38 fast", 44),
    ]),
    "Collapse": (2018, [
        ("T69 Collapse", 321), ("1st 44", 284), ("MT1 t29r2", 322),
        ("abundance10edit [2 R8's, FZ20m & a 909]", 289), ("pthex", 247),
    ]),
    "Blackbox Life Recorder 21f / In a Room7 F760": (2023, [
        ("Blackbox Life Recorder 21f", 260), ("zin2 test5", 252), ("In a Room7 F760", 162),
        ("Blackbox Life Recorder 22 [parallax mix]", 357),
    ]),
}

# Works whose YouTube titles usually differ from the catalogue title.
SEARCH_ALIASES: dict[str, list[str]] = {
    "Cliffs": ["#1", "Untitled 1"], "Radiator": ["#2"], "Rhubarb": ["#3"], "Hankie": ["#4"], "Grass": ["#5"],
    "Mould": ["#6"], "Curtains": ["#7"], "Blur": ["#8"], "Weathered Stone": ["#9"], "Tree": ["#10"],
    "Domino": ["#11"], "White Blur 1": ["#12"], "Blue Calx": ["#13"], "Parallel Stripes": ["#14"],
    "Shiny Metal Rods": ["#15"], "Grey Stripe": ["#16"], "Z Twig": ["#17"], "Windowsill": ["#18"],
    "Spots": ["#19"], "Tassels": ["#20"], "White Blur 2": ["#21"], "Matchsticks": ["#22"], "Lichen": ["#23"],
    "Stone in Focus": ["#24"], "Hexagon": ["#25"],
    "Equation": ["[Formula]", "ΔMi−1 = −αΣn=1NDi[n]"],
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
