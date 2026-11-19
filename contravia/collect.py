"""Collectible places in the world: where each one stands, which are near the car, and how a camera frames one.

The list of places is `places.json`, written outside the repo from EcuRed and OpenStreetMap and reviewed by hand
(spec `docs/superpowers/specs/2026-11-19-collectible-places-design.html`). Each place has a box: the volume a
photograph must frame, given as a width `w` along an axis turned `yaw_deg` from east toward north, a depth `d`
across it and a height `h`. This module turns those boxes into world metres, finds the ones near a point through
the streamed cells, and measures how a camera frames one. It needs no engine and no window, so all of it is tested
on its own; the game passes in the ground's height and its camera.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contravia import geo
from contravia.mapview import FOV_DEG

PATH = Path(__file__).parent / "places.json"

# How a framing is judged. The step that adds the photo tunes these by playing.
SMALLEST = 0.30  # the box must fill at least this share of the frame, along its larger side on screen
LARGEST = 0.90  # and at most this share; above it the camera is too close
CENTRE = 0.60  # the box's centre must fall in this central share of the frame, on each axis
INSIDE = 0.80  # at least this share of the box's rectangle on screen must be inside the frame
ASPECT = 16 / 9  # the game window's width over its height


@dataclass(frozen=True)
class Place:
    """One collectible place, with its box in world metres."""

    id: str
    title: str
    type: str
    province: str
    municipio: str
    text: str
    photo_target: str
    lat: float
    lon: float
    w: float  # the box's side along `axis`, metres
    d: float  # the side across it
    h: float  # the height
    scores: dict
    x: float  # the box's centre in world metres (geo.to_local)
    z: float
    axis: tuple[float, float]  # the unit direction of the `w` side in the world, as (x, z)


def _axis(lat: float, lon: float, yaw_deg: float) -> tuple[float, float]:
    """The world direction of a side turned `yaw_deg` from east toward north at (lat, lon).

    The side is projected rather than rotated, so the map projection's turn of the meridians (about 3 degrees at
    the island's ends) is included.
    """
    a = math.radians(yaw_deg)
    step = 20.0  # metres along the side
    lat2 = lat + step * math.sin(a) / 110_540
    lon2 = lon + step * math.cos(a) / (111_320 * math.cos(math.radians(lat)))
    (x1, z1), (x2, z2) = geo.to_local([lat, lat2], [lon, lon2])
    n = math.hypot(x2 - x1, z2 - z1)
    return (x2 - x1) / n, (z2 - z1) / n


def _place(r: dict) -> Place:
    (x, z), b = geo.to_local([r["lat"]], [r["lon"]])[0], r["box"]
    return Place(id=r["id"], title=r["title"], type=r["type"], province=r["province"], municipio=r["municipio"],
                 text=r["text"], photo_target=r["photo_target"], lat=r["lat"], lon=r["lon"],
                 w=float(b["w"]), d=float(b["d"]), h=float(b["h"]), scores=r["scores"],
                 x=float(x), z=float(z), axis=_axis(r["lat"], r["lon"], b["yaw_deg"]))


class Places:
    """All the places, indexed by the streamed cell (geo.cell_of) their centre falls in."""

    def __init__(self, records: list[dict]):
        self._all = [_place(r) for r in records]
        self._ids = {p.id: p for p in self._all}
        self._cells: dict[tuple[int, int], list[Place]] = {}
        for p in self._all:
            i, j = geo.cell_of([p.x, p.z])
            self._cells.setdefault((int(i), int(j)), []).append(p)

    @classmethod
    def load(cls, path: Path | None = None) -> Places:
        """Read places.json, next to this module unless a path is given."""
        return cls(json.loads(Path(path or PATH).read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self._all)

    def __iter__(self):
        return iter(self._all)

    def by_id(self, pid: str) -> Place:
        return self._ids[pid]

    def near(self, x: float, z: float, r: float) -> list[Place]:
        """The places whose centre is less than `r` metres from (x, z), looking only at the cells the circle touches."""
        (i0, j0), (i1, j1) = geo.cell_of([[x - r, z - r], [x + r, z + r]])
        found = []
        for i in range(int(i0), int(i1) + 1):
            for j in range(int(j0), int(j1) + 1):
                found += [p for p in self._cells.get((i, j), ()) if math.hypot(p.x - x, p.z - z) < r]
        return found


def corners(p: Place, ground: float) -> np.ndarray:
    """The 8 corners (x, y, z) of a place's box, its base at height `ground`; the game passes World.height there."""
    ux, uz = p.axis
    vx, vz = -uz, ux
    out = [(p.x + a * ux * p.w / 2 + b * vx * p.d / 2, y, p.z + a * uz * p.w / 2 + b * vz * p.d / 2)
           for a in (-1, 1) for b in (-1, 1) for y in (ground, ground + p.h)]
    return np.array(out, dtype=float)


def project(eye, target, points, zoom: float = 1.0, aspect: float = ASPECT) -> np.ndarray:
    """World points to the screen of a camera at `eye` looking at `target`, with y up.

    Returns an (N, 3) array: x and y from -1 to 1 across the frame (x right, y up) and the depth in front of the
    camera, which is 0 or less for a point behind it. Zoom narrows the field of view: the tangent of half the
    angle is divided by it.
    """
    eye, target, pts = np.asarray(eye, float), np.asarray(target, float), np.atleast_2d(np.asarray(points, float))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = pts - eye
    depth = d @ f
    t = math.tan(math.radians(FOV_DEG) / 2) / zoom
    safe = np.where(depth > 1e-9, depth, np.nan)
    return np.stack([(d @ r) / (safe * t * aspect), (d @ u) / (safe * t), depth], axis=-1)


@dataclass(frozen=True)
class Framing:
    """How a camera frames a box: the share of the frame it fills, where its centre falls, how much is inside."""

    size: float  # the larger of the box's width and height on screen, as shares of the frame's
    centre: tuple[float, float]  # where the box's centre falls, -1 to 1 on each axis
    inside: float  # the share of the box's rectangle on screen that lies inside the frame
    verdict: str  # "good", "far", "near" or "off"


def framing(eye, target, box_corners, zoom: float = 1.0, aspect: float = ASPECT) -> Framing:
    """Judge how a camera frames a box given by its 8 corners (from `corners`)."""
    s = project(eye, target, box_corners, zoom, aspect)
    if (s[:, 2] <= 0).any():
        return Framing(0.0, (math.nan, math.nan), 0.0, "off")
    cx, cy, _ = project(eye, target, np.asarray(box_corners).mean(axis=0), zoom, aspect)[0]
    x0, y0 = s[:, 0].min(), s[:, 1].min()
    x1, y1 = s[:, 0].max(), s[:, 1].max()
    size = max(x1 - x0, y1 - y0) / 2
    clipped = max(0.0, min(x1, 1) - max(x0, -1)) * max(0.0, min(y1, 1) - max(y0, -1))
    inside = clipped / max((x1 - x0) * (y1 - y0), 1e-12)
    if size > LARGEST:
        verdict = "near"
    elif abs(cx) > CENTRE or abs(cy) > CENTRE or inside < INSIDE:
        verdict = "off"
    elif size < SMALLEST:
        verdict = "far"
    else:
        verdict = "good"
    return Framing(float(size), (float(cx), float(cy)), float(inside), verdict)
