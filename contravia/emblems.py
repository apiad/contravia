"""The map's city emblems and town names (docs/superpowers/specs/2026-11-20-map-v2-design.html, section 05).

Provincial capitals and cities over CITY_POP people get a medallion: a coloured disc with an ink silhouette, a paper
ring and an ink rim; a few have their own drawing (GLYPHS), the rest a generic town. Smaller towns get a paper dot.
Which show depends on the camera's height (SHOWN), their size grows as it comes down, and below FADE_TO they all
fade out for the real streets. `layout` places them in two passes, most populous first: the symbols, leaving out any
that would overlap one placed, then the names, each trying a few spots; a town whose name fits nowhere loses its
dot too. Pure arithmetic and SVG text, kept apart from the engine so it can be tested; positions are GUI design
pixels from the top left.
"""

import math
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from contravia import collect, mapview, ui

CITY_POP = 100_000  # people: a city this size gets a medallion even when it is no capital
SHOWN = ((600_000.0, 140_000, None), (200_000.0, 0, 40_000), (80_000.0, 0, 15_000), (0.0, 0, 0))
# (camera height from, least people for a medallion, least for a town dot or None for no dots), highest first
FADE_FROM, FADE_TO = 13_000.0, 9_000.0  # metres: the emblems fade out between these heights
NEAR_PLACES = 12_000.0  # metres: a medallion's chip counts the collectible places this close
ACCENTS = ("#e76f51", "#2a9d8f", "#e9b949", "#7cc59a", "#5fa8d3")  # coral, turquoise, mustard, mint, sky
GAP = 3.0  # design pixels kept clear round each symbol
EDGE = 4.0  # design pixels kept clear of the screen's edges

# 40 x 40 drawings, filled with ink, standing on y = 35.
GLYPHS = {
    "La Habana": "M5 33h30v3H5zM8 27h24v6H8zM10 28h2v4h-2zm4 0h2v4h-2zm4 0h2v4h-2zm4 0h2v4h-2zm4 0h2v4h-2zM14.5 21h11v6h"
                 "-11zM14 21.5a6 7.5 0 0 1 12 0zM19.3 8h1.4v6h-1.4z",  # the Capitolio
    "Santiago de Cuba": "M7 35V17l3.5-5 3.5 5v18zM26 35V17l3.5-5 3.5 5v18zM14 35V22l6-4 6 4v13zM18.5 35v-6h3v6z",  # cathedral
    "Camagüey": "M14 8h12v3h-1v2c5 2 8 6 8 11 0 6-5 11-13 11S7 30 7 24c0-5 3-9 8-11v-2h-1z",  # a tinajón
    "Holguín": "M3 35c6-10 11-15 17-15s11 5 17 15zM19 5h2v5h4v2h-4v9h-2v-9h-4v-2h4z",  # the Loma de la Cruz
    "Santa Clara": "M8 35v-7h24v7zM17 28V14l1.5-5h3L23 14v14zM23 15l5-3 1 2-5 3z",  # the Che's monument
    "Pinar del Río": "M3 35c0-11 3-17 8-17 4 0 4 6 7 6s3-12 9-12c5 0 9 9 10 23z",  # a mogote
    "Matanzas": "M3 21h34v3H3zM5 24v11h3c0-6 4-9 7-9s7 3 7 9h-1 4c0-6 3-9 6-9s6 3 6 9V24z",  # its bridges
    "Baracoa": "M3 35l9-14h16l9 14zM12 21l2-3h12l2 3z",  # El Yunque
    "Trinidad": "M17 35V14l3-7 3 7v21zM4 35v-8h10v8zM26 35v-8h10v8zM4 27l5-4 5 4zM26 27l5-4 5 4z",  # its tower
}
GENERIC = "M5 35v-9h8v9zM14 35V18l3-4 3 4v17zM21 35V23h7v12zM29 35v-7h6v7z"
CITY_FONT, TOWN_FONT = ui.TITLE, ui.BOLD


@dataclass(frozen=True)
class Emblem:
    """One town on the map: a medallion or a dot at (x, y), and its name's box when it has room."""

    name: str
    medallion: bool
    x: float  # design pixels, the symbol's centre
    y: float
    r: float  # the symbol's radius
    size: float  # the name's font size
    label: tuple[float, float, float, float] | None  # (left, top, width, height) of the name's image
    places: int  # collectible places within NEAR_PLACES, for the medallion's chip
    at: tuple[float, float]  # the town's world (x, z)


def scale(h: float) -> float:
    """How much bigger symbols and names draw with the camera h metres up: 1 from 1 000 km, 1.8 at about 17 km, in
    steps of 0.05, so the names' measures and drawings repeat while the camera moves."""
    s = min(1.8, max(1.0, 1.0 + 0.45 * math.log10(1_000_000.0 / max(h, 1.0))))
    return round(s * 20) / 20


def opacity(h: float) -> float:
    """1 above FADE_FROM, 0 below FADE_TO, linear between."""
    return min(1.0, max(0.0, (h - FADE_TO) / (FADE_FROM - FADE_TO)))


def has_medallion(town) -> bool:
    return town.capital or town.pop >= CITY_POP


def shown(towns, h: float) -> list:
    """The towns the camera h metres up may draw, largest first: medallions over the least population of its
    band, and dots over theirs."""
    _, least, dots = next(s for s in SHOWN if h >= s[0])
    return [t for t in towns if (t.pop >= least if has_medallion(t) else dots is not None and t.pop >= dots)]


@lru_cache(maxsize=4096)
def text_width(text: str, font: str, size: float) -> float:
    """The advance of text in font at size, in the same pixels as size."""
    from PIL import ImageFont

    return ImageFont.truetype(font, max(1, round(size * 4))).getlength(text) / 4


def label_text(town) -> str:
    return town.name.upper() if has_medallion(town) else town.name


def label_size(town, s: float) -> float:
    p = math.sqrt(max(town.pop, 1) / 1.75e6)
    return s * (12.5 + 9 * p) if has_medallion(town) else 10.5 + 2.2 * (s - 1)


HALO = 2.5  # design pixels of ink round a name's letters


def label_box(town, s: float) -> tuple[float, float]:
    """(width, height) of a town's name image at scale s, halo included."""
    size = label_size(town, s)
    font = CITY_FONT if has_medallion(town) else TOWN_FONT
    return text_width(label_text(town), font, size) + 2 * HALO + size * 0.15, size * 1.3 + 2 * HALO


def places_near(towns, places) -> dict[str, int]:
    """Each town's number of collectible places within NEAR_PLACES."""
    pts = np.array([[p.x, p.z] for p in places]).reshape(-1, 2)
    return {t.name: int((np.hypot(pts[:, 0] - t.x, pts[:, 1] - t.z) < NEAR_PLACES).sum()) for t in towns}


def _hit(box, boxes, own=None) -> bool:
    return any(b is not own and box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in boxes)


def layout(towns, cam: mapview.MapCam, w: float, h: float, near: dict[str, int] | None = None,
           avoid: tuple = ()) -> list[Emblem]:
    """The emblems the camera cam draws in a w x h frame, placed so no two symbols or names overlap, nor any of
    the (left, top, right, bottom) boxes in `avoid` (the screen's panels)."""
    if opacity(cam.h) <= 0.0:
        return []
    cand = shown(towns, cam.h)
    if not cand:
        return []
    eye, target = mapview.pose(cam)
    ndc = collect.project(eye, target, np.array([[t.x, 0.0, t.z] for t in cand]), 1.0, w / h)
    s = scale(cam.h)
    boxes, placed = list(avoid), []
    for t, (nx, ny, depth) in zip(cand, ndc, strict=True):  # pass 1: the symbols
        if depth <= 0:
            continue
        x, y = (nx + 1) / 2 * w, (1 - ny) / 2 * h
        med = has_medallion(t)
        r = s * (11 + 19 * math.sqrt(max(t.pop, 1) / 1.75e6)) if med else s * 3.2
        box = (x - r - GAP, y - r - GAP, x + r + GAP, y + r + GAP + (2 if med else 0))
        if box[0] < EDGE or box[1] < EDGE or box[2] > w - EDGE or box[3] > h - EDGE or _hit(box, boxes):
            continue
        boxes.append(box)
        placed.append((t, med, x, y, r, box))
    out = []
    for t, med, x, y, r, own in placed:  # pass 2: the names, in the same order
        bw, bh = label_box(t, s)
        spots = ([(x - bw / 2, y + r + 2), (x + r + 4, y - bh / 2), (x - r - 4 - bw, y - bh / 2), (x - bw / 2, y - r - 2 - bh)]
                 if med else
                 [(x + r + 3, y - bh / 2), (x - r - 3 - bw, y - bh / 2), (x - bw / 2, y - r - 2 - bh), (x - bw / 2, y + r + 2)])
        label = None
        for lx, ly in spots:
            box = (lx, ly, lx + bw, ly + bh)
            if lx >= EDGE and ly >= EDGE and box[2] <= w - EDGE and box[3] <= h - EDGE and not _hit(box, boxes, own):
                boxes.append(box)
                label = (lx, ly, bw, bh)
                break
        if label is None and not med:
            continue
        out.append(Emblem(t.name, med, x, y, r, label_size(t, s), label, (near or {}).get(t.name, 0), (t.x, t.z)))
    return out


def accent(name: str) -> str:
    """A medallion's disc colour: yellow for La Habana, else one of ACCENTS picked by the name, so it never changes."""
    return ui.YELLOW if name == "La Habana" else ACCENTS[zlib.crc32(name.encode()) % len(ACCENTS)]


CHIP_FROM = (15.0, 80_000.0)  # a medallion carries its places chip from this radius and camera height up


@lru_cache(maxsize=256)
def medallion_svg(name: str, chip: int = 0) -> str:
    """A medallion as SVG on a 100 x 106 box, with a yellow chip of `chip` places at its top right when it is not
    0: the GUI scales it to the emblem's size."""
    glyph = GLYPHS.get(name, GENERIC)
    tag = (f'<rect x="58" y="0" width="{30 + 10 * len(str(chip))}" height="24" rx="12" fill="{ui.YELLOW}" '
           f'stroke="{ui.INK}" stroke-width="3"/><path transform="translate(63 3) scale(0.75)" fill="{ui.INK}" '
           'd="M12 1l3.2 7 7.6.7-5.8 5 1.8 7.4L12 17.2 5.2 21.1 7 13.7 1.2 8.7 8.8 8z"/>'
           f'<text x="83" y="19" font-family="{_family(CITY_FONT)}" font-size="19" fill="{ui.INK}">{chip}</text>'
           ) if chip else ""
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 106">'
            f'<circle cx="50" cy="55" r="48" fill="{ui.INK}" opacity=".5"/>'
            f'<circle cx="50" cy="50" r="48" fill="{ui.INK}"/><circle cx="50" cy="50" r="40" fill="{ui.PAPER}"/>'
            f'<circle cx="50" cy="50" r="31" fill="{accent(name)}"/>'
            '<clipPath id="c"><circle cx="50" cy="50" r="31"/></clipPath>'
            f'<g clip-path="url(#c)"><g transform="translate(19 17) scale(1.55)" fill="{ui.INK}">'
            f'<path d="{glyph}"/></g></g>{tag}</svg>')


def _family(font: str) -> str:
    """The name SVG text matches a loaded font by: its file name without the extension."""
    return font.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def dot_svg() -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><circle cx="10" cy="10" r="8" '
            f'fill="{ui.PAPER}" stroke="{ui.INK}" stroke-width="3"/></svg>')


@lru_cache(maxsize=1024)
def label_svg(town_name: str, medallion: bool, size: float, w: float, h: float) -> str:
    """A name with an ink halo, as SVG of w x h design pixels: eight ink copies round the paper letters."""
    family = _family(CITY_FONT if medallion else TOWN_FONT)
    text = (town_name.upper() if medallion else town_name).replace("&", "&amp;").replace("<", "&lt;")
    fill = ui.YELLOW if town_name == "La Habana" else ui.PAPER
    x, y, k = HALO + size * 0.05, HALO + size * 1.0, HALO
    d = k * 0.7
    font = f'font-family="{family}" font-size="{size:.1f}"'  # on each text: thorvg does not inherit them from a group
    copies = "".join(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" {font} fill="{ui.INK}">{text}</text>'
                     for dx, dy in ((-k, 0), (k, 0), (0, -k), (0, k), (-d, -d), (d, -d), (-d, d), (d, d)))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}">{copies}'
            f'<text x="{x:.1f}" y="{y:.1f}" {font} fill="{fill}">{text}</text></svg>')
