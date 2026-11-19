"""The collectible places on the minimap and on the map: an icon per kind of place, grey until photographed and
yellow after, and where each falls on screen (docs/superpowers/specs/2026-11-19-collectible-places-design.html,
section 07). From CITY_H up the map joins a municipio's places into one mark with their number.

Pure arithmetic, kept apart from the engine so it can be tested. Positions are GUI design pixels from the top left.
"""

import math
from dataclasses import dataclass

import numpy as np

from contravia import collect, mapview, ui

CITY_H = 13_000.0  # metres of map camera height where the city stage starts and places join by municipio
CROSS_PX = 24.0  # a mark this close to the screen's centre is under the crosshair

# 24 x 24 drawings, filled with the icon's colour.
DRAWINGS = {
    "iglesia": "M11 3h2v3h3v2h-3v3l5 3v7h-4v-4h-4v4H6v-7l5-3V8H8V6h3z",
    "fortaleza": "M4 8h3v2h2V8h2v2h2V8h2v2h2V8h3v13H4z",
    "playa": "M3 15c3-2 4-2 6 0s4 2 6 0 4-2 6 0v3c-2-2-4-2-6 0s-4 2-6 0-4-2-6 0zM15 5a3 3 0 1 1 0 6 3 3 0 0 1 0-6z",
    "museo": "M12 3l9 5v2H3V8zM5 11h2v7H5zm4 0h2v7H9zm4 0h2v7h-2zm4 0h2v7h-2zM3 19h18v2H3z",
    "monumento": "M11 3h2l1 14h3v4H7v-4h3z",
    "plaza": "M12 3a6 6 0 0 1 2 11.7V21h-4v-6.3A6 6 0 0 1 12 3z",
    "teatro": "M4 4h16v6a8 8 0 0 1-16 0zm4 4a1.5 1.5 0 1 0 0 .1zm8 0a1.5 1.5 0 1 0 0 .1zM8 13h8a4 4 0 0 1-8 0z",
    "estacion": "M7 3h10a3 3 0 0 1 3 3v10a2 2 0 0 1-2 2l2 3h-3l-2-3H9l-2 3H4l2-3a2 2 0 0 1-2-2V6a3 3 0 0 1 3-3zM6 7v4h12V7z",
    "naturaleza": "M3 20l6-11 4 6 3-4 5 9z",
    "lugar": "M12 2a7 7 0 0 1 7 7c0 5-7 13-7 13S5 14 5 9a7 7 0 0 1 7-7zm0 4a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
}
GLYPH = {"iglesia": "iglesia", "cementerio": "iglesia", "fortaleza": "fortaleza", "sitio_historico": "fortaleza",
         "playa": "playa", "agua": "playa", "fuente": "playa", "museo": "museo", "casa_colonial": "museo",
         "edificio_historico": "museo", "universidad": "museo", "monumento": "monumento", "faro": "monumento",
         "plaza_parque": "plaza", "teatro": "teatro", "estadio": "teatro", "estacion": "estacion",
         "puente": "estacion", "obra_ingenieria": "estacion", "ingenio": "estacion", "area_natural": "naturaleza",
         "montana": "naturaleza", "cueva": "naturaleza"}


def glyph(kind: str) -> str:
    """The drawing a kind of place gets; hotels and anything new get the generic pin."""
    return GLYPH.get(kind, "lugar")


def icon(kind: str, got: bool, size: int = 24) -> str:
    """The SVG of a place's icon: its drawing on an ink disc, yellow once photographed and grey before."""
    colour = ui.YELLOW if got else ui.GREY
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 28 28">'
            f'<circle cx="14" cy="14" r="13" fill="{ui.INK}" stroke="{colour}" stroke-width="1.5"/>'
            f'<path transform="translate(2 2)" d="{DRAWINGS[glyph(kind)]}" fill="{colour}"/></svg>')


@dataclass(frozen=True)
class Mark:
    """One icon on the map: a place, or from CITY_H up a municipio's places joined."""

    x: float  # design pixels from the left
    y: float  # design pixels from the top
    kind: str  # the place's type, or the joined places' most common
    got: bool  # photographed; for a group, all of them
    ids: tuple[str, ...]
    label: str  # what the top-left note says with the crosshair on it
    at: tuple[float, float]  # world (x, z) of the place, or of the group's middle


def on_minimap(places, x: float, z: float, yaw: float, metres: float, radius_px: float):
    """(place, right, down) for each place inside the minimap's circle, in pixels from its centre: the car at
    (x, z) faces yaw (forward is (cos, sin)) at the top, and radius_px pixels span `metres` of ground."""
    k = radius_px / metres
    fx, fz = math.cos(yaw), math.sin(yaw)
    out = []
    for p in places.near(x, z, metres):
        dx, dz = p.x - x, p.z - z
        right, ahead = (-fz * dx + fx * dz) * k, (fx * dx + fz * dz) * k
        if math.hypot(right, ahead) <= radius_px:
            out.append((p, right, -ahead))
    return out


def on_map(places, has, cam: mapview.MapCam, w: float, h: float) -> list[Mark]:
    """The map's marks inside a w x h frame for the camera cam; has(id) says whether a place is photographed.
    From CITY_H up a municipio's places become one mark at their middle."""
    eye, target = mapview.pose(cam)
    every = list(places)
    if cam.h < CITY_H:
        groups = [[p] for p in every]
    else:
        by = {}
        for p in every:
            by.setdefault((p.province, p.municipio), []).append(p)
        groups = list(by.values())
    at = np.array([[np.mean([p.x for p in g]), 0.0, np.mean([p.z for p in g])] for g in groups])
    ndc = collect.project(eye, target, at, 1.0, w / h)
    marks = []
    for g, (nx, ny, depth), (ax, _, az) in zip(groups, ndc, at, strict=True):
        if not depth > 0 or abs(nx) > 1.05 or abs(ny) > 1.05:
            continue
        kinds = [p.type for p in g]
        got = sum(has(p.id) for p in g)
        if len(g) == 1:
            label = g[0].title
        else:
            label = f"{g[0].municipio} · {len(g)} lugares" + (f", {got} con foto" if got else "")
        marks.append(Mark(x=(nx + 1) / 2 * w, y=(1 - ny) / 2 * h, kind=max(set(kinds), key=kinds.count),
                          got=got == len(g), ids=tuple(p.id for p in g), label=label, at=(float(ax), float(az))))
    return marks


def under_cross(marks: list[Mark], w: float, h: float) -> Mark | None:
    """The mark nearest the screen's centre, if it is within CROSS_PX."""
    best = min(marks, key=lambda m: math.hypot(m.x - w / 2, m.y - h / 2), default=None)
    if best is None or math.hypot(best.x - w / 2, best.y - h / 2) > CROSS_PX:
        return None
    return best
