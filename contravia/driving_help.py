"""Driving help: six levels from a real car (5) to almost autopilot (0), and where the car is on the road.

Each level sets the strengths of manifoldx's four chassis assists (steering, traction, anti-lock, stability) and,
from level 2 down, the road assists that read the streets (spec
docs/superpowers/specs/2026-11-20-driving-help-design.html). Every road assist starts from the same answer, which
`Road.locate` gives once a frame: the edge the car drives along, how far along it, how far right of its centreline,
and the lane that falls in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contravia.traffic import lane_geometry, lane_offsets

# (name in the menu, what it does) by level, 0 the most help
LEVELS = (
    ("Casi piloto", "maneja por ti: sigue al de delante, para en rojo y dobla donde le toques el timón"),
    ("Freno de emergencia", "te mantiene en la senda y frena solo antes de chocar"),
    ("Mantener senda", "con el timón suelto se centra en su senda"),
    ("Dirección asistida", "no se va de lado, no bloquea las ruedas y te corrige el timón"),
    ("Chasis", "frenos antibloqueo, control de tracción y de estabilidad, como un carro moderno"),
    ("Real", "sin ninguna ayuda"),
)
DEFAULT = 3
CHASSIS = 0.6  # strength of the chassis assists at level 4
REACH = 12.0  # metres: farthest a car can be from an edge's centreline and still be on it
ALONG = np.cos(np.radians(60))  # the car's heading must be within 60° of the edge's direction
CELL, PIECE = 16.0, 8.0  # the index's cells and the pieces segments are cut into for it
AROUND = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])  # the 3 × 3 cells around a point


def chassis_strengths(level: int, locked=()) -> np.ndarray:
    """The four chassis assists' strengths, in manifoldx's ASSISTS order, at this level; locked ones are 0."""
    from manifoldx.vehicles.assists import ASSISTS

    k = np.ones(len(ASSISTS)) if level <= 3 else np.full(len(ASSISTS), CHASSIS if level == 4 else 0.0)
    if level == 4:
        k[ASSISTS.index("steer")] = 0.0
    for name in locked:
        if name in ASSISTS:
            k[ASSISTS.index(name)] = 0.0
    return k


def level_from_realism(realism: float) -> int:
    """The level an old saved realism (0 easy to 100 real) becomes."""
    return 5 if realism >= 90 else 4 if realism >= 60 else 3


@dataclass
class Place:
    """Where the car is on the road: its edge, s metres along it, `right` metres right of its centreline, in
    `lane` (0 the rightmost), `dist` metres from the edge's polyline."""

    edge: int
    s: float
    right: float
    lane: int
    dist: float


class Road:
    """The road graph's polylines, indexed on a grid so a point finds the segments near it in a few lookups."""

    def __init__(self, graph):
        a = graph.a
        self.graph = graph
        self.lanes = a["g_lanes"].astype(np.int64)
        self.twoway, self.lane_w = lane_geometry(a)
        poly, starts = a["g_poly"].astype(np.float64), a["g_poly_start"].astype(np.int64)
        edge_of = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
        k = np.flatnonzero(np.arange(len(poly) - 1) + 1 < starts[edge_of[:-1] + 1])  # segments inside an edge
        self.p0, self.edge = poly[k], edge_of[k]
        d = poly[k + 1] - self.p0
        self.seg_len = np.maximum(np.hypot(*d.T), 1e-9)
        self.u = d / self.seg_len[:, None]
        seg = np.zeros(len(poly))
        seg[k + 1] = self.seg_len
        cum = np.cumsum(seg)
        self.s0 = cum[k] - cum[starts[self.edge]]  # distance along the edge where each segment starts
        # cut every segment into pieces of at most PIECE and file each piece by its midpoint's cell
        n = np.ceil(self.seg_len / PIECE).astype(np.int64)
        which = np.repeat(np.arange(len(k)), n)
        f = (np.arange(len(which)) - np.repeat(np.cumsum(n) - n, n) + 0.5) / n[which]
        mid = self.p0[which] + d[which] * f[:, None]
        key = self._key(np.floor(mid / CELL).astype(np.int64))
        order = np.argsort(key, kind="stable")
        self.keys, self.pieces = key[order], which[order]

    @staticmethod
    def _key(cell: np.ndarray) -> np.ndarray:
        return (cell[..., 0] + (1 << 20)) * (1 << 21) + (cell[..., 1] + (1 << 20))

    def _near(self, x: float, z: float) -> np.ndarray:
        """Segments with a piece in the 3 × 3 cells around (x, z): every segment within REACH of it, and some more."""
        cells = self._key(np.floor(np.array([x, z]) / CELL).astype(np.int64) + AROUND)
        lo, hi = np.searchsorted(self.keys, cells).tolist(), np.searchsorted(self.keys, cells, side="right").tolist()
        return np.concatenate([self.pieces[a:b] for a, b in zip(lo, hi)])  # a segment may repeat: harmless

    def locate(self, x: float, z: float, heading: float) -> Place | None:
        """Where a car at (x, z) driving along `heading` (atan2(dz, dx)) is on the road, or None off the road,
        the wrong way on a one-way street or crosswise to every street."""
        seg = self._near(x, z)
        if not len(seg):
            return None
        rel = np.array([x, z]) - self.p0[seg]
        t = np.clip((rel * self.u[seg]).sum(1), 0, self.seg_len[seg])
        foot = self.p0[seg] + self.u[seg] * t[:, None]
        dist = np.hypot(*(foot - (x, z)).T)
        dist = np.where(self.u[seg] @ np.array([np.cos(heading), np.sin(heading)]) >= ALONG, dist, np.inf)
        k = int(np.argmin(dist))
        if not np.isfinite(dist[k]):
            return None
        i, e = seg[k], int(self.edge[seg[k]])
        real = float(np.hypot(*(foot[k] - (x, z))))
        right = float(rel[k] @ np.array([-self.u[i, 1], self.u[i, 0]]))
        offs = lane_offsets(int(self.lanes[e]), float(self.lane_w[e]), bool(self.twoway[e]))
        lane = int(np.argmin(np.abs(offs - right)))
        if real > REACH or abs(offs[lane] - right) > self.lane_w[e] + 1.5:
            return None
        return Place(e, float(self.s0[i] + t[k]), right, lane, real)
