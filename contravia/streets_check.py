"""contravia streets-check: count what stands on a street, where the ground covers it and where its surface breaks.

A street is clean when a car can drive every metre of the surface the cells draw (contravia/streets.py), kerb to
kerb, without meeting anything it sees or hits. This walks the cells of a built island and, for each one, takes
the street triangles around it and counts, by kind:

- ground: a street triangle the drawn ground stands over (rules/ground.drawn, or the shore's grid near water),
  outside the open cuts, where the ground is left out, and less than clearance.UNDER deep, past which it is a
  covered tunnel;
- water: a street triangle under the sea's plane or an inland water's, where the shore's grid says water;
- step: a street triangle with another one over or under it by more than JUMP and less than HEAD: a step in the
  surface, two surfaces on one street, or a deck with no headroom;
- steep: a street triangle steeper than STEEP;
- building, landmark, park, tree, cemetery, tomb: a wall segment of that source the car hits (walls.parts) that
  reaches into a street;
- parapet: a wall of a bridge or tunnel (struct_walls) that crosses a street at a height where it stops the car;
- structure, walk, motif: a triangle of a bridge or tunnel's dressing, of a pavement or of a park's motifs
  (benches, lamps, fountains) drawn between LOW and HEAD over a street's asphalt.

"Into a street" is past EDGE from the surface's edge, since kerbs, bevels and parapets stand on that edge (a
pavement may reach WALK_EDGE: walks.py lets its outline cut a rounded corner by SIMPLIFY, and its bevel runs on); an
avenue's median (contravia/medians.py) is no part of the surface, so its kerb, lamps and wall are edges too. The
2D walls are tested against the streets drawn in the open, as clearance.py cuts the boxes; what has heights is
tested against every level, tunnels too. Each cell is tested against the streets of its 3 x 3 cells, since a
thing is filed under the cell of its middle. Every count must be 0.
"""

import hashlib
import json
import pickle
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np
import shapely

from contravia import regions
from contravia.clearance import UNDER, open_streets
from contravia.geo import CELL, to_latlon
from contravia.landmarks import cached
from contravia.mesh import lattice
from contravia.rules.ground import drawn
from contravia.shore import of as shore_of
from contravia.structures import KERB_TOP, LEAVES
from contravia.walks import BEVEL, KERB
from contravia.walks import SIMPLIFY as WALK_SIMPLIFY
from contravia.walls import parts

EDGE = 0.3  # metres a thing may reach past the street surface's edge: kerbs, bevels and parapets stand on it
HEAD = 3.5  # metres of room over the asphalt a street needs: a bus and some air
LOW = 0.1  # metres over the asphalt under which a drawn thing is paint or the foot of a kerb
JUMP = 0.15  # metres: two street surfaces further apart than this at one point are a step
STEEP = 0.3  # rise over run: no street triangle is steeper (a ramp climbs 8 %, a turn off a carried ramp 16 %)
COVER = 0.005  # metres the ground may stand over a street's triangle: rounding, not a patch of grass
REACH = 0.5  # metres under a structure wall's foot from where it stops the car (walls.Walls.near)
STEP = 1.0  # metres between the points a street triangle is tested at
FINE = 0.5  # metres between the points a drawn thing or a wall is tested at
SLIVER = 0.05  # m²: a street triangle smaller than this is not judged steep
WALK_EDGE = WALK_SIMPLIFY + KERB * BEVEL + 0.02  # metres a pavement and its bevel may reach into a street
KINDS = ("ground", "water", "step", "steep", "building", "landmark", "park", "tree", "cemetery", "tomb", "parapet",
         "structure", "walk", "motif")
_NONE = shapely.Polygon()
KEPT = "streets-check.kept"  # the last run's findings by cell, beside index.npz
KEPT_VERSION = 1
AROUND = 2  # cells round a cell whose data its findings may read (the check itself reads one): its fingerprint
# what the fingerprint of the island holds besides each province's relief: the files every cell reads
ISLAND = ("coast.src", "shore.src", "cover.src", "causeways.src", "water.src", "regions.src", "landmarks.pkl")
# the modules that build the data: their work is in each cell's arrays, so a change to them is seen there
BUILDERS = ("contravia.streets", "contravia.walks", "contravia.structures", "contravia.medians", "contravia.prepare",
            "contravia.source", "contravia.cli")


class Levels:
    """The street triangles round a cell, to ask which streets lie under a point and how high."""

    def __init__(self, tris: np.ndarray):
        t = np.asarray(tris, float).reshape(-1, 3, 3)
        a = t[:, 0, [0, 2]]
        u, v = t[:, 1, [0, 2]] - a, t[:, 2, [0, 2]] - a
        det = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
        ok = np.abs(det) > 1e-9  # an upright or empty triangle holds no point
        self.a, self.u, self.v, self.det, self.y = a[ok], u[ok], v[ok], det[ok], t[ok][:, :, 1]
        self.tree = shapely.STRtree(shapely.polygons(t[ok][:, :, [0, 2]])) if ok.any() else None

    def under(self, xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(K,) which of the (P, 2) points and (K,) how high, a pair for every street triangle a point lies in."""
        if self.tree is None or not len(xz):
            return np.zeros(0, np.int64), np.zeros(0)
        p, t = self.tree.query(shapely.points(xz), predicate="intersects")
        d = xz[p] - self.a[t]
        s = (d[:, 0] * self.v[t, 1] - self.v[t, 0] * d[:, 1]) / self.det[t]
        r = (self.u[t, 0] * d[:, 1] - d[:, 0] * self.u[t, 1]) / self.det[t]
        return p, self.y[t, 0] * (1 - s - r) + self.y[t, 1] * s + self.y[t, 2] * r


class Check:
    """A built island's streets, checked a cell at a time. cell(i, j) gives the cell's findings."""

    KEPT = 8000  # cells whose streets are held at once: a scan in order asks for each cell's neighbours again

    def __init__(self, city):
        self.city, self.landmarks, self.shore = city, cached(city), shore_of(city)
        self._streets: dict = {}  # cell -> (all street triangles, those in the open, its medians' footprints)
        self._inner: dict = {}  # cell -> (the streets' inside there at every level, in the open)
        self._digests: dict = {}  # cell -> the digest of its data

    def digest(self, i: int, j: int) -> bytes:
        """A digest of everything cell (i, j) holds, as the island gives it."""
        def make(i, j):
            h = hashlib.blake2b(digest_size=16)
            c = self.city.cell(i, j)
            for f in fields(c):
                v = getattr(c, f.name)
                for part in v if isinstance(v, list | tuple) else (v,):
                    h.update(b"-" if part is None else np.ascontiguousarray(part).tobytes()
                             if isinstance(part, np.ndarray) else repr(part).encode())
                h.update(f.name.encode())
            return h.digest()
        return self._held(self._digests, (i, j), make)

    def fingerprint(self, i: int, j: int) -> bytes:
        """What cell (i, j)'s findings were computed from: the data of the cells AROUND it. Two runs that give a
        cell the same fingerprint, on the same island (island_stamp) with the same code, find the same."""
        h = hashlib.blake2b(digest_size=16)
        for di in range(-AROUND, AROUND + 1):
            for dj in range(-AROUND, AROUND + 1):
                h.update(self.digest(i + di, j + dj))
        return h.digest()

    def _held(self, cache: dict, key, make):
        if key not in cache:
            if len(cache) >= self.KEPT:
                for old in list(cache)[: self.KEPT // 2]:
                    del cache[old]
            cache[key] = make(*key)
        return cache[key]

    def streets(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell (i, j)'s street triangles, (T, 3, 3): all of them and those drawn in the open; and the (M, 3, 2)
        footprints of the triangles of its avenues' medians."""
        def make(i, j):
            c = self.city.cell(i, j)
            t = c.street_tris
            if t is None or not len(t):
                return np.zeros((0, 3, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3, 2))
            median = np.zeros((0, 3, 2))
            if c.struct_tris is not None and len(c.struct_tris):
                paint = np.asarray(c.struct_paint)
                median = np.asarray(c.struct_tris, float)[(paint >= KERB_TOP) & (paint <= LEAVES)][:, :, [0, 2]]
            return np.asarray(t, float).reshape(-1, 3, 3), open_streets(self.city, i, j), median
        return self._held(self._streets, (i, j), make)

    def _around(self, i: int, j: int, which: int) -> np.ndarray:
        return np.concatenate([self.streets(i + di, j + dj)[which] for di in (-1, 0, 1) for dj in (-1, 0, 1)])

    def inner(self, i: int, j: int) -> tuple:
        """The part of cell (i, j) that lies inside a street by more than EDGE, as a prepared polygon: over the
        streets of every level, and over those in the open. The streets are joined across the cell's borders
        first, where the surface was cut, so a border is no edge."""
        def make(i, j):
            if not len(self.streets(i, j)[0]):
                return _NONE, _NONE
            cell = shapely.box(i * CELL, j * CELL, (i + 1) * CELL, (j + 1) * CELL)
            out = []
            for which in (0, 1):
                if which and all(len(self.streets(i + di, j + dj)[0]) == len(self.streets(i + di, j + dj)[1])
                                 for di in (-1, 0, 1) for dj in (-1, 0, 1)):
                    out.append(out[0])  # no tunnel near: one answer for both
                    continue
                t = self._around(i, j, which)[:, :, [0, 2]]
                near = shapely.union_all(shapely.make_valid(shapely.polygons(t))).intersection(cell.buffer(4 * EDGE))
                median = self._around(i, j, 2)
                if len(median):
                    near = near.difference(shapely.union_all(shapely.make_valid(shapely.polygons(median))))
                piece = near.buffer(-EDGE).intersection(cell)
                shapely.prepare(piece)
                out.append(piece)
            return tuple(out)
        return self._held(self._inner, (i, j), make)

    def _inside(self, i: int, j: int, which: int, geoms, deeper: float = 0.0) -> np.ndarray:
        """Whether each geometry reaches into a street of the 3 x 3 cells round (i, j); with deeper, whether it
        reaches that many metres further in than EDGE (a cell's border is no edge of a street)."""
        out = np.zeros(len(geoms), bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                piece = self.inner(i + di, j + dj)[which]
                if piece.is_empty:
                    continue
                hit = shapely.intersects(piece, geoms)
                if deeper and hit.any():
                    a, b = (i + di) * CELL, (j + dj) * CELL
                    edge = piece.boundary.difference(shapely.box(a, b, a + CELL, b + CELL).boundary.buffer(1e-6))
                    hit[hit] = shapely.distance(edge, geoms[hit]) > deeper
                out |= hit
        return out

    def cell(self, i: int, j: int) -> list[tuple[str, float, float, float]]:
        """What is wrong in cell (i, j): a (kind, x, z, how much) per triangle or wall segment at fault. How much
        is metres of ground over the street, of step, of height over the asphalt, of wall inside the street, or
        the slope of a steep triangle."""
        every = self._around(i, j, 0)
        if not len(every):
            return []
        levels = Levels(every)
        out = self._surface(i, j, levels)
        c = regions.current().parks(self.landmarks.keep(i, j, self.city.cell(i, j)))
        for kind, walls in parts(self.city, self.landmarks, i, j).items():
            if len(walls):
                lines = shapely.linestrings(walls.astype(float))
                for k in np.flatnonzero(self._inside(i, j, 1, lines)).tolist():
                    x, z = walls[k].mean(axis=0)
                    out.append((kind, float(x), float(z), self._length(i, j, 1, lines[k])))
        out += self._parapets(i, j, c.struct_walls, levels)
        for kind, tris in (("structure", c.struct_tris), ("walk", c.walk_tris), ("motif", c.park_tris)):
            out += self._drawn(i, j, kind, tris, levels)
        return out

    def _length(self, i, j, which, line) -> float:
        return float(sum(shapely.length(shapely.intersection(self.inner(i + di, j + dj)[which], line))
                         for di in (-1, 0, 1) for dj in (-1, 0, 1)))

    def _ground(self, i: int, j: int, x, z) -> tuple[np.ndarray, np.ndarray]:
        """What cell (i, j) draws at points (x, z): the height of its ground, or of the water where the shore's
        grid says water, and whether it is water."""
        f = self.shore.field(i, j) if self.shore is not None else None
        if f is None:
            return drawn(self.city.terrain, x, z), np.zeros(len(x), bool)
        wet = f.sample(f.d, x, z) < 0
        return np.where(wet, f.level(x, z), f.height(x, z)), wet

    def _surface(self, i: int, j: int, levels: Levels) -> list:
        """The cell's own street triangles: ground over them, steps between them and steep ones."""
        out = []
        tris = self.streets(i, j)[1]
        if not len(tris):
            return out
        e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
        n = np.cross(e1, e2)
        flat = np.hypot(n[:, 0], n[:, 2])
        for k in np.flatnonzero((np.linalg.norm(n, axis=1) / 2 > SLIVER) & (flat > STEEP * np.abs(n[:, 1]))).tolist():
            x, _, z = tris[k].mean(axis=0)
            out.append(("steep", float(x), float(z), float(flat[k] / max(abs(n[k, 1]), 1e-9))))
        pts, own = lattice(tris, STEP)
        # a point on the cell's far border belongs to the next cell's ground grid: pull each a hair toward its triangle
        pts = pts + (tris[own].mean(axis=1) - pts) * 1e-3
        ground, wet = self._ground(i, j, pts[:, 0], pts[:, 2])
        over = ground - pts[:, 1]
        cuts = self.city.cell(i, j).street_cuts
        if cuts is not None and len(cuts) and (over > COVER).any():
            hole = shapely.union_all(shapely.make_valid(shapely.polygons(np.asarray(cuts, float)))).buffer(1e-3)
            over[shapely.contains_xy(hole, pts[:, 0], pts[:, 2])] = 0.0
        over[~wet & (over > UNDER)] = 0.0  # the covered end of a triangle that starts in a cut: a tunnel
        out += self._worst("ground", pts, own, np.where(~wet & (over > COVER), over, 0.0))
        out += self._worst("water", pts, own, np.where(wet & (over > COVER), over, 0.0))
        p, h = levels.under(pts[:, [0, 2]])
        gap = np.abs(h - pts[p, 1])
        step = np.zeros(len(pts))
        np.maximum.at(step, p, np.where((gap > JUMP) & (gap < HEAD), gap, 0.0))
        return out + self._worst("step", pts, own, step)

    @staticmethod
    def _worst(kind: str, pts: np.ndarray, own: np.ndarray, amount: np.ndarray) -> list:
        """A finding per triangle with a point at fault: where its worst point is and how much."""
        bad = np.flatnonzero(amount > 0)
        if not len(bad):
            return []
        order = bad[np.lexsort((-amount[bad], own[bad]))]
        first = order[np.r_[True, own[order][1:] != own[order][:-1]]]
        return [(kind, float(pts[k, 0]), float(pts[k, -1]), float(amount[k])) for k in first.tolist()]

    def _parapets(self, i: int, j: int, walls, levels: Levels) -> list:
        """Bridge and tunnel walls that cross a street at a height where they stop the car."""
        if walls is None or not len(walls):
            return []
        w = np.asarray(walls, float).reshape(-1, 6)
        hit = np.flatnonzero(self._inside(i, j, 0, shapely.linestrings(w[:, :4].reshape(-1, 2, 2))))
        out = []
        for k in hit.tolist():
            a, b = w[k, 0:2], w[k, 2:4]
            t = np.linspace(0.0, 1.0, max(int(np.ceil(np.hypot(*(b - a)) / FINE)), 1) + 1)[:, None]
            pts = a + (b - a) * t
            pts = pts[self._inside(i, j, 0, shapely.points(pts))]
            _, h = levels.under(pts)
            if ((h >= w[k, 4] - REACH) & (h <= w[k, 5])).any():
                out.append(("parapet", float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2), len(pts) * FINE))
        return out

    def _drawn(self, i: int, j: int, kind: str, tris, levels: Levels) -> list:
        """Triangles of a drawn thing that stand between LOW and HEAD over a street's asphalt."""
        if tris is None or not len(tris):
            return []
        t = np.asarray(tris, float).reshape(-1, 3, 3)
        lo, hi = t[:, :, [0, 2]].min(axis=1), t[:, :, [0, 2]].max(axis=1)
        t = t[self._inside(i, j, 0, shapely.box(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1]))]
        pts, own = lattice(t, FINE)
        deep = self._inside(i, j, 0, shapely.points(pts[:, [0, 2]]), WALK_EDGE - EDGE if kind == "walk" else 0.0)
        pts, own = pts[deep], own[deep]
        p, h = levels.under(pts[:, [0, 2]])
        up = pts[p, 1] - h
        over = np.zeros(len(pts))
        np.maximum.at(over, p, np.where((up > LOW) & (up < HEAD), up, 0.0))
        return self._worst(kind, pts, own, over)


def count(city, cells=None) -> tuple[Counter, list]:
    """The findings over the given cells (all of the city's by default): how many of each kind, and each one as
    (kind, x, z, how much)."""
    check = Check(city)
    cells = city.cells() if cells is None else cells
    found = []
    for i, j in sorted(map(tuple, np.asarray(cells).reshape(-1, 2).tolist())):
        found += check.cell(i, j)
    counts = Counter({k: 0 for k in KINDS})
    counts.update(f[0] for f in found)
    return counts, found


def island_stamp(folder) -> bytes:
    """A digest of what every cell of the island in folder reads besides its own data: each province's relief and
    the files of ISLAND."""
    h = hashlib.blake2b(digest_size=16)
    folder = Path(folder)
    for path in sorted(folder.glob("*.npz")):
        if path.name == "index.npz":
            continue
        with np.load(path) as f:
            for key in ("heights", "heights_origin", "heights_step"):
                if key in f.files:
                    h.update(np.ascontiguousarray(f[key]).tobytes())
        h.update(path.name.encode())
    for name in ISLAND:
        h.update((folder / name).read_bytes() if (folder / name).exists() else b"-")
        h.update(name.encode())
    return h.digest()


def code_stamp(modules) -> str:
    """A digest of the sources of modules: the code a run's findings came from."""
    import importlib.util

    h = hashlib.blake2b(digest_size=16)
    for name in sorted(modules):
        spec = importlib.util.find_spec(name)
        h.update(Path(spec.origin).read_bytes() if spec is not None and spec.origin else b"-")
        h.update(name.encode())
    return h.hexdigest()


def load_kept(folder, island: bytes) -> dict:
    """{cell: (fingerprint, findings)} of the last run in folder, or {} when there is none, the island's relief or
    files changed since, or any module that run loaded did."""
    path = Path(folder) / KEPT
    if not path.exists():
        return {}
    try:
        kept = pickle.loads(path.read_bytes())
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError):
        return {}
    if (kept.get("version") != KEPT_VERSION or kept.get("island") != island
            or kept.get("code") != code_stamp(kept.get("modules", ()))):
        return {}
    return kept["cells"]


def save_kept(folder, island: bytes, modules, cells: dict) -> None:
    path = Path(folder) / KEPT
    part = path.with_name(path.name + ".part")
    part.write_bytes(pickle.dumps({"version": KEPT_VERSION, "island": island, "modules": sorted(modules),
                                   "code": code_stamp(modules), "cells": cells}))
    part.replace(path)


def loaded() -> list[str]:
    """The game's modules this process has loaded, less the BUILDERS: the code its findings depend on."""
    return sorted(m for m in sys.modules if m.split(".")[0] == "contravia" and m not in BUILDERS)


_check: Check | None = None
_kept: dict = {}


def _start(data: str, kept: dict | None = None) -> None:
    global _check, _kept
    from contravia.world import World

    _check, _kept = Check(World.load(data)), kept or {}


def _run(cells: list) -> tuple[list, dict, list]:
    """(findings, what to keep for the cells checked again, the modules loaded) over cells: a cell whose
    fingerprint is the kept one gives its kept findings."""
    found, fresh = [], {}
    for i, j in cells:
        if not len(_check._around(i, j, 0)):
            continue
        mark = _check.fingerprint(i, j)
        old = _kept.get((i, j))
        if old is not None and old[0] == mark:
            found += old[1]
            continue
        fresh[(i, j)] = (mark, _check.cell(i, j))
        found += fresh[(i, j)][1]
    return found, fresh, loaded()


def cells_of(world, province: str | None = None, around: tuple | None = None) -> list:
    """The cells to check, in order: the island's, one province's, or those within around's (x, z, radius)."""
    names = [n.lower() for n in world.names]
    ks = range(len(names)) if province is None else [names.index(province.lower())]
    cells = np.unique(np.concatenate([np.asarray(world.province(k).cells()).reshape(-1, 2) for k in ks]), axis=0)
    if around is not None:
        x, z, r = around
        mid = (cells + 0.5) * CELL
        cells = cells[np.hypot(mid[:, 0] - x, mid[:, 1] - z) <= r + CELL]
    return sorted(map(tuple, cells.tolist()))


def main(data: str, province: str | None = None, around: tuple | None = None, jobs: int = 1,
         out: str | None = None, fresh: bool = False) -> int:
    """Print the counts and the worst of each kind, write every finding to out as JSON; the exit status is 1
    when any count is above 0. An island in a folder keeps each cell's findings with the fingerprint of its data
    (KEPT): the next run checks again only the cells whose data changed, all of them when the island's relief or
    files or the code did, or with fresh."""
    from concurrent.futures import ProcessPoolExecutor

    from contravia.world import World

    world = World.load(data)
    cells = cells_of(world, province, around)
    island = island_stamp(world.folder) if world.folder is not None else None
    kept = {} if island is None or fresh else load_kept(world.folder, island)
    rows = max(1, len(cells) // (jobs * 16) or 1)
    chunks = [cells[k:k + rows] for k in range(0, len(cells), rows)]
    found, again, modules = [], {}, set()
    with ProcessPoolExecutor(jobs, initializer=_start, initargs=(data, kept)) as pool:
        for part, new, mods in pool.map(_run, chunks):
            found += part
            again.update(new)
            modules.update(mods)
    served = sum(c in kept and c not in again for c in cells)
    if island is not None:
        if fresh:  # the cells not asked for keep what an earlier run found, when it still holds
            kept = load_kept(world.folder, island)
        save_kept(world.folder, island, modules, {**kept, **again})
    counts = Counter({k: 0 for k in KINDS})
    counts.update(f[0] for f in found)
    print(f"{len(cells)} cells, {len(again)} checked, {served} kept from the last run")
    for kind in KINDS:
        worst = max((f for f in found if f[0] == kind), key=lambda f: f[3], default=None)
        where = "" if worst is None else "  worst {:.2f} at {:.5f},{:.5f}".format(
            worst[3], *to_latlon(np.array([[worst[1], worst[2]]]))[0])
        print(f"{kind:10s} {counts[kind]:7d} on a street{where}")
    if out:
        ll = to_latlon(np.array([[f[1], f[2]] for f in found]).reshape(-1, 2))
        with open(out, "w") as fh:
            json.dump([{"kind": f[0], "x": round(f[1], 2), "z": round(f[2], 2), "amount": round(f[3], 3),
                        "lat": round(float(p[0]), 6), "lon": round(float(p[1]), 6)}
                       for f, p in zip(found, ll, strict=True)], fh)
    return int(any(counts.values()))
