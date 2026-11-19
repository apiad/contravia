"""Generic pieces for the collectible places OSM has only as a point, so the player frames something there.

docs/superpowers/specs/2026-11-19-collectible-places-design.html, section 09. places.json gives such a place the
default box of its type (POINT_BOX). Six types are not buildings and the city draws nothing for them: each gets a
piece seeded from its id, set off the streets (spot), with walls the car hits and its box as the place's
(Piece.boxes). A museum, church or hotel that is only a point usually has its building in OSM without a tag, and
the city already draws it. A place layer 5 models (collect.MODELLED) keeps its own mesh.
"""

import random

import numpy as np

from contravia import collect, geo
from contravia.landmarks.shapes import Piece, circle, dome, flat, prism, pyramid, rect
from contravia.terrain import seat

NAME = "Lugares de un punto"
POINT_BOX = {"monumento": {(8.0, 8.0, 10.0), (3.0, 3.0, 4.0)}, "fuente": {(6.0, 6.0, 4.0)},
             "plaza_parque": {(80.0, 80.0, 6.0)}, "estacion": {(20.0, 50.0, 10.0)}, "faro": {(8.0, 8.0, 30.0)},
             "sitio_historico": {(20.0, 20.0, 6.0)}}
RINGS = (6.0, 12.0, 20.0, 30.0)  # metres from the point where spot looks for room off the streets
KERB = 1.0  # metres between a piece and a street's edge
STONE, BRONZE, WHITE, WATER = (0.78, 0.76, 0.70), (0.30, 0.36, 0.30), (0.94, 0.93, 0.89), (0.25, 0.45, 0.55)
ROOF, RED, GLASS = (0.45, 0.20, 0.15), (0.75, 0.15, 0.12), (0.85, 0.88, 0.60)


def is_point(p) -> bool:
    """Whether a place is one OSM has only as a point, of a type the city draws nothing for."""
    return (p.w, p.d, p.h) in POINT_BOX.get(p.type, ())


def _gap(x, z, roads) -> tuple[float, np.ndarray]:
    """How far (x, z) is from the nearest street's edge, and that street's direction."""
    a, b, half = roads[:, 0:2], roads[:, 2:4], roads[:, 4] / 2
    ab = b - a
    t = np.clip(((np.array([x, z]) - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    d = np.hypot(*(a + ab * t[:, None] - (x, z)).T) - half
    k = int(np.argmin(d))
    return float(d[k]), ab[k] / (np.hypot(*ab[k]) or 1.0)


def spot(x: float, z: float, roads: np.ndarray, r: float):
    """(x, z, along) for a piece of radius r: the point itself, or the nearest of eight directions on RINGS where
    it clears every street by KERB, and the direction of the nearest street; None if no room is found."""
    near = roads[(np.minimum(roads[:, 0], roads[:, 2]) < x + 100) & (np.maximum(roads[:, 0], roads[:, 2]) > x - 100)
                 & (np.minimum(roads[:, 1], roads[:, 3]) < z + 100) & (np.maximum(roads[:, 1], roads[:, 3]) > z - 100)]
    if not len(near):
        return x, z, np.array([1.0, 0.0])
    for d in (0.0, *RINGS):
        for k in range(8 if d else 1):
            px, pz = x + d * np.cos(k * np.pi / 4), z + d * np.sin(k * np.pi / 4)
            gap, along = _gap(px, pz, near)
            if gap > r + KERB:
                return px, pz, along
    return None


def form(title: str, rng) -> str:
    """A monument's form: the one its title names (obelisk, bust, statue), else a seeded statue or obelisk."""
    t = title.lower()
    for word, f in (("obelisco", "obelisk"), ("busto", "bust"), ("estatua", "statue"), ("escultura", "statue")):
        if word in t:
            return f
    return "statue" if rng.random() < 0.5 else "obelisk"


def monument(mb, c, u, y, rng, shape="statue"):
    """A bust on a slim pedestal, a statue on a stepped base, or an obelisk."""
    if shape == "bust":
        base = rect(c, u, 0.6, 0.6)
        prism(mb, base, y, y + 1.4, STONE)
        dome(mb, c, 0.35, y + 1.4, 0.7, BRONZE, steps=4)
        return base
    base = rect(c, u, 2.2, 2.2)
    prism(mb, base, y, y + 0.5, STONE)
    if shape == "statue":
        prism(mb, rect(c, u, 1.3, 1.3), y + 0.5, y + 3.0, STONE)
        prism(mb, rect(c, u, 0.35, 0.5), y + 3.0, y + 4.6, BRONZE)
        dome(mb, c, 0.3, y + 4.6, 0.5, BRONZE, steps=4)
    else:
        top = y + rng.uniform(9.0, 14.0)
        prism(mb, rect(c, u, 0.8, 0.8), y + 0.5, top - 1.4, WHITE, roof=False)
        pyramid(mb, rect(c, u, 0.8, 0.8), top - 1.4, top, WHITE)
    return base


def fountain(mb, c, u, y, rng):
    """A round basin with water, a column and a bowl on it."""
    r = rng.uniform(2.6, 3.4)
    basin = circle(c, r, 16)
    prism(mb, basin, y, y + 0.6, STONE, roof=False)
    flat(mb, circle(c, r - 0.3, 16), y + 0.45, WATER)
    prism(mb, circle(c, 0.35, 8), y, y + 2.0, STONE)
    prism(mb, circle(c, 1.0, 12), y + 1.7, y + 1.9, STONE)
    return basin


def glorieta(mb, c, u, y, rng):
    """An octagonal bandstand: a platform, eight posts and a pointed roof."""
    r = rng.uniform(4.0, 5.0)
    floor = circle(c, r, 8)
    prism(mb, floor, y, y + 0.5, STONE)
    for k in range(8):
        a = k * np.pi / 4
        prism(mb, rect(c + (r - 0.4) * np.array([np.cos(a), np.sin(a)]), u, 0.15, 0.15), y + 0.5, y + 3.2, WHITE)
    pyramid(mb, circle(c, r + 0.5, 8), y + 3.2, y + 5.0, RED)
    return floor


def station(mb, c, u, y, rng):
    """A platform along the street with a shelter: posts and a flat roof."""
    a = rng.uniform(8.0, 11.0)
    floor = rect(c, u, a, 2.5)
    prism(mb, floor, y, y + 1.0, STONE)
    v = np.array([-u[1], u[0]])
    for s in np.linspace(-a + 1.5, a - 1.5, 4):
        prism(mb, rect(c + u * s + v * 1.6, u, 0.12, 0.12), y + 1.0, y + 3.6, WHITE)
    prism(mb, rect(c + v * 0.4, u, a - 0.5, 2.3), y + 3.6, y + 3.8, ROOF)
    return floor


def lighthouse(mb, c, u, y, rng):
    """A white tower with a red band, a lantern and a cap."""
    h = rng.uniform(12.0, 18.0)
    base = circle(c, 1.8, 12)
    prism(mb, base, y, y + h * 0.6, WHITE, roof=False)
    prism(mb, circle(c, 1.75, 12), y + h * 0.6, y + h * 0.75, RED, roof=False)
    prism(mb, circle(c, 1.7, 12), y + h * 0.75, y + h, WHITE)
    prism(mb, circle(c, 1.0, 8), y + h, y + h + 1.5, GLASS)
    dome(mb, c, 1.1, y + h + 1.5, 0.9, RED, steps=4)
    return base


def plaque(mb, c, u, y, rng):
    """A memorial stone: a low base and an upright slab facing the street."""
    base = rect(c, u, 1.2, 0.6)
    prism(mb, base, y, y + 0.4, STONE)
    prism(mb, rect(c, u, 0.9, 0.18), y + 0.4, y + 2.2, STONE)
    return base


BUILD = {"monumento": (monument, 3.2), "fuente": (fountain, 3.5), "plaza_parque": (glorieta, 5.6),
         "estacion": (station, 11.5), "faro": (lighthouse, 2.0), "sitio_historico": (plaque, 1.4)}  # (piece, radius)


def place(city, mb) -> Piece | None:
    """Add a piece for each point-only place in the city's cells; None when there is none."""
    cells = {tuple(k) for k in np.asarray(city.cells()).tolist()}
    roads = np.asarray(city.a["roads"], float).reshape(-1, 5)
    modelled = set(collect.MODELLED.values())
    piece = Piece()
    for p in collect.Places.load():
        if not is_point(p) or p.id in modelled or tuple(int(v) for v in geo.cell_of([p.x, p.z])) not in cells:
            continue
        build, r = BUILD[p.type]
        found = spot(p.x, p.z, roads, r)
        if found is None:
            continue
        x, z, along = found
        c = np.array([x, z])
        mark = mb.mark()
        ring = build(mb, c, along, seat(circle(c, r, 8), city.terrain), random.Random(p.id),
                     **({"shape": "bust" if p.w < 5 else form(p.title, random.Random(p.id))}
                        if build is monument else {}))
        piece.boxes[p.id] = mb.bounds_since(mark)
        piece.walls.append(ring)
        piece.clear.append(circle(c, r + KERB, 12))
    return piece if piece.boxes else None
