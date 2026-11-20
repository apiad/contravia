"""What the map names under and near its crosshair, with as much detail as its height allows.

The map's four stages (docs/superpowers/specs/2026-11-18-city-map-design.html, section 03) set what a player can
tell apart, so they set what the panel says. From the island stage up it names the province and the most important
places on screen; at the city stage the municipio and the important places near the crosshair; at the barrio stage
the places near it, weighed by importance and distance; at the street stage the street under it and the places
right there, nearest first.

A place's importance is its score in places.json (`scores["total"]`: history, culture and visits, 2 to 15).

Pure functions, kept apart from the engine so they can be tested: the lookups of province, municipio and street
are contravia/whereis.py's, and the panel that draws the result is the map screen's (contravia/menu.py).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    """How the panel speaks over one stage of the map."""

    name: str  # the map stage: "isla", "ciudad", "barrio" or "calle"
    above: float  # metres of camera height where the band starts
    count: int  # how many places it names at most
    reach: float  # the places it names are within reach * height metres of the crosshair
    floor: float  # and never less than this many metres
    near: float  # 0 ranks by importance alone, 1 by distance alone, between mixes both
    header: str  # the line above the places


BANDS = (  # highest first
    Band("isla", 100_000.0, 25, 1.3, 0.0, 0.0, "LO MÁS NOTABLE"),
    Band("ciudad", 13_000.0, 12, 0.6, 0.0, 0.0, "NOTABLE CERCA"),
    Band("barrio", 1_600.0, 8, 0.5, 800.0, 0.6, "CERCA"),
    Band("calle", 0.0, 5, 0.5, 300.0, 1.0, "AQUÍ"),
)
MOVED = 0.03  # the panel is chosen again once the crosshair moved this share of the height
ZOOMED = 0.05  # or the height changed by this share


def band(h: float) -> Band:
    """The band of a camera h metres up."""
    return next(b for b in BANDS if h >= b.above)


def importance(p) -> float:
    """How notable a place is: its total score in places.json, 0 when it has none."""
    return float(p.scores.get("total", 0.0))


@dataclass(frozen=True)
class Spot:
    """What lies under the crosshair, as contravia/whereis.py finds it; empty strings for what it did not."""

    province: str = ""
    municipio: str = ""
    street: str = ""
    line: str = ""  # "entre L y M", "esquina a L"...
    barrio: str = ""


@dataclass(frozen=True)
class Item:
    """One place the panel names."""

    id: str
    kind: str  # the place's type, for its icon
    title: str
    distance: str  # "350 m", "1,2 km", or empty from the island stage


@dataclass(frozen=True)
class Here:
    """What the panel says: a title, a line under it, a header and the places."""

    band: str
    title: str
    line: str
    header: str
    items: tuple[Item, ...]


def distance(metres: float) -> str:
    """A distance as the panel writes it: metres rounded to 10 below a kilometre, then kilometres with one
    decimal below 10 and none above, with a decimal comma."""
    if metres < 995:
        return f"{max(10, round(metres / 10) * 10)} m"
    if metres < 9_950:
        return f"{metres / 1000:.1f} km".replace(".", ",")
    return f"{round(metres / 1000)} km"


def places_near(places, x: float, z: float, h: float) -> list[tuple]:
    """(place, metres) of the places the band of height h names around (x, z), best first: within its reach,
    ranked by importance * (1 - near * d / reach), the nearest first on a tie, at most its count."""
    b = band(h)
    r = max(b.reach * h, b.floor)
    found = []
    for p in places:
        d = math.hypot(p.x - x, p.z - z)
        if d <= r:
            score = importance(p) * (1 - b.near * d / r) if b.near < 1 else -d
            found.append((-score, d, p.id, p))
    found.sort(key=lambda t: t[:3])
    return [(p, d) for _, d, _, p in found[:b.count]]


def choose(places, spot: Spot, x: float, z: float, h: float) -> Here:
    """The panel for the crosshair at (x, z) with the camera h metres up, given what lies under it."""
    b = band(h)
    sea = "Cuba" if b.name == "isla" else "En el mar"
    if b.name in ("isla", "ciudad"):
        title, line = spot.province or sea, spot.municipio if b.name == "ciudad" else ""
    elif b.name == "barrio" or not spot.street:
        title = spot.municipio or spot.province or sea
        line = " · ".join(n for n in (spot.barrio, spot.province if spot.municipio else "") if n)
    else:
        title = spot.street
        line = " · ".join(dict.fromkeys(n for n in (spot.line, spot.barrio, spot.municipio) if n))
    items = tuple(Item(p.id, p.type, p.title, "" if b.name == "isla" else distance(d))
                  for p, d in places_near(places, x, z, h))
    return Here(b.name, title, line, b.header if items else "", items)


def stale(was: tuple[float, float, float] | None, x: float, z: float, h: float) -> bool:
    """Whether the panel chosen with the camera at was = (x, z, h) must be chosen again for (x, z, h): the
    crosshair moved MOVED of the height, the height changed by ZOOMED, or the band changed."""
    if was is None:
        return True
    x0, z0, h0 = was
    return (math.hypot(x - x0, z - z0) > MOVED * h0 or abs(h - h0) > ZOOMED * h0
            or band(h).name != band(h0).name)
