"""People on the pavements: the pool of bodies a preset builds (Figures), the lanes they walk (lay) and the crowd.

A crowd does not get one body per person: a body costs about 350 ms over its three levels and 280 ms of clips, so
300 unique people would take minutes. A preset builds a pool of bodies instead (Figures), each a
rules.people.body dressed for one zone, at levels 2, 1 and 0, with the clips of rules.people.CLIPS, and every walker
plays one of them. The work runs on a Builder thread of its own, in an order that lets the crowd show early: every
body's level 2 and walk clips first, then level 1, then level 0 and the other clips. The GPU uploads (a Mesh, a
baked clip) happen on the frame thread, a few per frame, as results arrive.

A clip is baked from the body's level 2 figure: the skeleton is the same at every level, so it plays on all three.
Results are kept across a save under contravia dev (contravia.kept) until rules/people.py or this file changes.

The lanes of a cell are lines along its walkable tops (contravia.walks): INSET metres in from a pavement's kerb,
SIDE_INSET in from the edge of an apron, a path or a plaza, cut where they come within BORDER of the cell's edge.
A lane cut there is open: a walker nearing its end looks both ways (the cross cycle), stands a moment and turns
back. The Crowd keeps its walkers in arrays, moves them all in one step, gives each a level by distance with
hysteresis and the preset's caps, and changes an entity's level as the graveyard does: the new level fades in
over the pixels the old one leaves (manifoldx's Fade), since a mesh cannot be swapped for one of another size.
Spec: docs/superpowers/specs/2026-11-22-pedestrians-in-game-design.html.
"""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from contravia import kept as keep
from contravia.garage import Builder
from contravia.rules import people as rules

SEED = 1000  # body i of the pool is rules.people.body(SEED + i)
ZONES = 3  # body i dresses for zone i % ZONES
LEVELS = (2, 1, 0)
UPLOADS = 2  # results put on the GPU per frame at most
FRAMES = 32  # poses per baked cycle


def work(i: int) -> list[tuple]:
    """Every job of body i, as (stage, item): stage orders the jobs of the whole pool."""
    walk = [("clip", i, "walk", v) for v in rules.WALK]
    rest = [("clip", i, k, v) for k, v in rules.CLIPS if k != "walk"]
    return ([(0, ("figure", i, 2))] + [(1, j) for j in walk] + [(2, ("figure", i, 1))]
            + [(3, ("figure", i, 0))] + [(4, j) for j in rest])


class Figures:
    """The pool of bodies of one preset: their meshes at three levels and their clips, built off the frame."""

    def __init__(self, engine, bodies: int):
        self.engine = engine
        stamp = (keep.source(rules.__file__), keep.source(__file__))
        self.store = keep.kept(engine.state, "people", stamp, dict)  # item -> what the frame thread made of it
        self._figs = {}  # body -> its level 2 Figure, written and read on the worker thread only
        self.builder = Builder(make=self._make, key_of=repr, cached=1 << 30, name="people")
        self.waiting: deque = deque()
        self.n = 0
        self.resize(bodies)

    def body(self, i: int):
        """Body i's HumanParams, Outfit and zone."""
        params = rules.body(SEED + i)
        zone = i % ZONES
        return params, rules.outfit(SEED + i, zone, params.build), zone

    def resize(self, bodies: int) -> None:
        """Hold a pool of this many bodies, queueing the jobs of the bodies not built yet. A smaller pool keeps
        what it built; its extra bodies are only not handed out."""
        new = range(self.n, bodies)
        self.n = bodies
        jobs = sorted((s, k, item) for k, i in enumerate(new) for s, item in work(i))
        for _, _, item in jobs:
            if item not in self.store:
                self.waiting.append(item)
                self.builder.want(item)

    def _make(self, item):
        """On the worker: a figure's geometry, or a clip's poses."""
        from manifoldx.humanoid.cycles import bake_cycle
        from manifoldx.modeling.assets.human import build_human

        if item[0] == "figure":
            _, i, level = item
            params, outfit, _ = self.body(i)
            f = build_human(params, outfit=outfit, level=level)
            if level == 2:
                self._figs[i] = f
            return f.mesh.to_geometry()
        _, i, kind, speed = item
        f = self._figs.get(i)
        if f is None:  # its level 2 job ran in an earlier life of this object
            params, outfit, _ = self.body(i)
            f = self._figs[i] = build_human(params, outfit=outfit, level=2)
        return bake_cycle(f, kind, speed, FRAMES)

    def step(self) -> int:
        """On the frame thread: put up to UPLOADS finished results on the GPU. The worker runs its jobs in the
        order they were queued, so only the head of the queue can be done. Returns how many were put up."""
        from manifoldx.components import Mesh

        done = 0
        while self.waiting and done < UPLOADS:
            item = self.waiting[0]
            got = self.builder.want(item)
            if got is None:
                break
            self.waiting.popleft()
            self.builder.cache.pop(repr(item), None)  # the store holds it now
            self.store[item] = Mesh(got) if item[0] == "figure" else self.engine.bake_poses(got)
            done += 1
        return done

    def nearest(self, i: int, level: int):
        """(level, Mesh) of body i at level, or at the nearest level built, coarser first; (-1, None) while none
        is."""
        for lv in sorted(LEVELS, key=lambda lv: (abs(lv - level), -lv)):
            m = self.store.get(("figure", i, lv))
            if m is not None:
                return lv, m
        return -1, None

    def mesh(self, i: int, level: int):
        """Body i's Mesh at level, or at the nearest level built, coarser first; None while none is."""
        return self.nearest(i, level)[1]

    def clip_table(self) -> np.ndarray:
        """(bodies, len(rules.CLIPS)) clip ids, -1 where not baked yet."""
        return np.array([[c if (c := self.clip(i, k, v)) is not None else -1 for k, v in rules.CLIPS]
                         for i in range(self.n)], np.int64).reshape(self.n, len(rules.CLIPS))

    def clip(self, i: int, kind: str, speed: float):
        """The id of body i's baked clip of kind at speed, or None while it is not baked."""
        return self.store.get(("clip", i, kind, speed))

    def ready(self, zone: int) -> list[int]:
        """The bodies of zone a walker can play now: one level built and every walking clip baked."""
        return [i for i in range(zone, self.n, ZONES)
                if self.mesh(i, 2) is not None and all(self.clip(i, "walk", v) is not None for v in rules.WALK)]

    def counts(self) -> dict:
        built = [k for k in self.store if k[1] < self.n]
        return {"bodies": self.n, "meshes": sum(k[0] == "figure" for k in built),
                "clips": sum(k[0] == "clip" for k in built), "waiting": len(self.waiting)}

    def close(self) -> None:
        self.builder.close()


# ------------------------------------------------------------------------------------------------- lanes

INSET = 0.7  # metres a lane runs in from a pavement's kerb
SIDE_INSET = 0.6  # metres in from the edge of an apron, a path or a plaza
BORDER = 0.8  # metres from the cell's edge where a lane is cut
MIN_LANE = 6.0  # metres: shorter lanes are dropped
STEP = 1.0  # metres between a lane's samples, at most


@dataclass
class Lanes:
    """A cell's lanes: samples (P, 3) of every lane one after another; per lane its first sample, its sample
    count, the metres between two samples, its length, whether it loops, its zone and its weight."""

    pts: np.ndarray
    start: np.ndarray
    count: np.ndarray
    step: np.ndarray
    length: np.ndarray
    loop: np.ndarray
    zone: np.ndarray
    weight: np.ndarray

    def __len__(self):
        return len(self.start)


def _empty() -> Lanes:
    z = np.zeros(0)
    return Lanes(np.zeros((0, 3)), z.astype(int), z.astype(int), z, z, z.astype(bool), z.astype(int), z)


def lay(city, i: int, j: int) -> Lanes:
    """Cell (i, j)'s lanes, from its walk triangles."""
    import shapely
    from shapely.geometry import LineString, Polygon

    from contravia import walks
    from contravia.geo import CELL

    c = city.cell(i, j)
    tris = getattr(c, "walk_tris", None)
    if tris is None or not len(tris):
        return _empty()
    tris, paint = np.asarray(tris, float).reshape(-1, 3, 3), np.asarray(c.walk_paint, int)
    part, zone = paint % walks.PARTS, np.minimum(paint // walks.PARTS, ZONES - 1)
    plaza = paint == walks.PLAZA_CODE
    pavement = (paint < 3 * walks.PARTS) & (part == walks.TOP)
    side = ((paint < 3 * walks.PARTS) & np.isin(part, (walks.APRON_TOP, walks.PATH_TOP, walks.DIRT_TOP))) | plaza
    zone = np.where(plaza, 1, zone)
    polys = shapely.polygons(tris[:, :, [0, 2]])
    ok = shapely.area(polys) > 1e-6
    inner = Polygon([(i * CELL + BORDER, j * CELL + BORDER), ((i + 1) * CELL - BORDER, j * CELL + BORDER),
                     ((i + 1) * CELL - BORDER, (j + 1) * CELL - BORDER), (i * CELL + BORDER, (j + 1) * CELL - BORDER)])
    lines, zones, sides = [], [], []
    for z in range(ZONES):
        for kind, inset in ((pavement, INSET), (side, SIDE_INSET)):
            pick = kind & (zone == z) & ok
            if not pick.any():
                continue
            area = shapely.union_all(polys[pick]).buffer(-inset)
            for poly in getattr(area, "geoms", [area]):
                if poly.is_empty or poly.geom_type != "Polygon":
                    continue
                ring = LineString(poly.exterior.coords)
                if ring.length < MIN_LANE:
                    continue
                if ring.within(inner):
                    pieces = [(ring, True)]
                else:
                    cut = ring.intersection(inner)
                    cut = shapely.line_merge(cut) if cut.geom_type == "MultiLineString" else cut
                    pieces = [(g, False) for g in getattr(cut, "geoms", [cut])]
                for g, loop in pieces:
                    if g.geom_type == "LineString" and g.length >= MIN_LANE:
                        lines.append((g, loop))
                        zones.append(z)
                        sides.append(kind is side)
    if not lines:
        return _empty()
    tree = shapely.STRtree(polys)
    pts, start, count, step, length = [], [], [], [], []
    n0 = 0
    for g, _ in lines:
        n = max(2, int(np.ceil(g.length / STEP)) + 1)
        d = np.linspace(0.0, g.length, n)
        xz = shapely.get_coordinates(shapely.line_interpolate_point(g, d))
        pts.append(np.column_stack([xz[:, 0], _heights(tree, tris, xz), xz[:, 1]]))
        start.append(n0)
        count.append(n)
        step.append(g.length / (n - 1))
        length.append(g.length)
        n0 += n
    zones = np.array(zones)
    weight = np.array(length) * np.array(rules.DENSITY)[zones] * np.where(sides, rules.SIDE, 1.0)
    return Lanes(np.concatenate(pts), np.array(start), np.array(count), np.array(step), np.array(length),
                 np.array([lp for _, lp in lines]), zones, weight)


def _heights(tree, tris: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """The height of the walk triangles under each point; the nearest triangle's plane where none is under it."""
    import shapely

    pts = shapely.points(xz)
    which = np.full(len(xz), -1)
    pi, ti = tree.query(pts, predicate="intersects")
    which[pi[::-1]] = ti[::-1]  # the first triangle found for each point
    miss = which < 0
    if miss.any():
        which[miss] = tree.query_nearest(pts[miss], return_distance=False)[1]
    t = tris[which]
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    v0, v1 = (b - a)[:, [0, 2]], (c - a)[:, [0, 2]]
    v2 = xz - a[:, [0, 2]]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    return a[:, 1] + u * (b[:, 1] - a[:, 1]) + v * (c[:, 1] - a[:, 1])


# ------------------------------------------------------------------------------------------------- the crowd

WALK, IDLE, RUN, CROSS = range(4)
KIND = ("walk", "idle", "run", "cross")
CLIP_IDLE = rules.CLIPS.index(("idle", 0.0))
CLIP_CROSS = rules.CLIPS.index(("cross", rules.CROSS))
HYSTERESIS = 3.0  # metres a walker goes past a band before it changes level
EVERY = 6  # frames between two passes over the levels
LINGER = 60  # frames a cell's lanes stay after the streamer stops drawing it in full
KEEP = 256  # closed cells whose lanes are kept
FADE = 0.3  # seconds a figure takes to fade in or out
SPAWNS = 12  # walkers spawned or sent away per frame at most
RATE = 0.05  # walkers per weighted lane metre: one every 20 m of an old centre pavement
LOOK = 3.0  # metres before an open end where a walker starts looking both ways
STAND = (1.0, 3.0)  # seconds a walker stands at an open end before it turns back
IDLE_FOR = (5.0, 20.0)  # seconds an idle walker stands
SHARES = (0.8, 0.1, 0.1)  # walk, idle, run
HIDDEN = 3  # the level of a walker too far to draw


def hour_factor(hour: float) -> float:
    """The share of the crowd out at this hour of the day."""
    if 8 <= hour < 20:
        return 1.0
    if 6 <= hour < 8 or 20 <= hour < 24:
        return 0.5
    return 0.15


def yaw_quat(d: np.ndarray) -> np.ndarray:
    """Quaternions (x, y, z, w) that turn a figure facing +Z toward each direction d (n, 3), about +Y."""
    a = np.arctan2(d[:, 0], d[:, 2])
    q = np.zeros((len(d), 4))
    q[:, 1], q[:, 3] = np.sin(a / 2), np.cos(a / 2)
    return q


# Each walker's arrays: clip is its walking or running bucket's column in rules.CLIPS, level the one it should
# draw at, shown the one its mesh is (a level not built yet shows the nearest), ent its entity and ghost the one
# fading out under it.
FIELDS = (("alive", bool), ("leaving", bool), ("lane", np.int64), ("local", np.int64), ("s", float),
          ("dir", float), ("mode", np.int64), ("timer", float), ("clip", np.int64), ("body", np.int64),
          ("phase", float), ("level", np.int64), ("shown", np.int64), ("ent", np.int64), ("ghost", np.int64),
          ("t0", float), ("fading", bool))


class Crowd:
    """The walkers of the cells drawn in full: where they are, what they play, and their entities."""

    def __init__(self, engine, city, q=None, seed: int = 0):
        import manifoldx as mx
        from manifoldx.components import Component, Transform
        from manifoldx.resources import StandardMaterial
        from manifoldx.types import Float

        class Person(Component):
            k: Float  # the walker this entity draws

        Person = engine.component(Person)  # registered now: the system below queries it before any person exists
        self.engine, self.city, self.Person = engine, city, Person
        self.material = StandardMaterial("#ffffff", roughness=0.8, vertex_colors=True, faded=True)
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="lanes")
        self._pending: dict[tuple[int, int], Future] = {}
        self.lanes: dict[tuple[int, int], Lanes] = {}
        self._kept: dict[tuple[int, int], Lanes] = {}
        self._absent: dict[tuple[int, int], int] = {}
        self._frame, self.clock, self.rng = 0, perf_counter, np.random.default_rng(seed)
        self.lod, self.cap, self.limit = (12.0, 45.0, 120.0), (300, 16, 80, 24), None
        self.figures = Figures(engine, 0)
        self.clips = np.zeros((0, len(rules.CLIPS)), np.int64)
        self.rates: dict = {}
        self.target, self.timings = 0, []
        self.cell: list = []
        self._alloc(0)
        self._table()
        self.configure(q) if q is not None else self._sized(self.cap)

        def move(qr, dt: float):
            k = np.rint(qr[Person].k.data[:, 0]).astype(np.int64)
            qr[Transform].pos = self.pos[k]
            qr[Transform].rot = self.rot[k]

        move.__annotations__ = {"qr": mx.Query[Transform, Person], "dt": float}
        engine.system(move)

    # -- sizes and knobs

    def _alloc(self, n: int) -> None:
        """Hold n walker slots, keeping the first ones."""
        had = len(getattr(self, "alive", ()))
        m = min(n, had)
        for name, kind in FIELDS:
            new = np.full(n, -1 if name in ("ent", "ghost", "level", "shown") else 0, dtype=kind)
            if had:
                new[:m] = getattr(self, name)[:m]
            setattr(self, name, new)
        for name, width in (("pos", 3), ("rot", 4), ("rate", 0)):
            new = np.zeros((n, width) if width else n)
            if name == "rot":
                new[:, 3] = 1.0
            if had:
                new[:m] = getattr(self, name)[:m]
            setattr(self, name, new)
        self.cell = (self.cell + [None] * n)[:n]

    def _sized(self, cap) -> None:
        walkers = cap[0] if self.limit is None else min(cap[0], self.limit)
        live = np.flatnonzero(self.alive)
        if len(live) > walkers:
            self._send_away(live[walkers:], now=True)
        kept = live[:walkers]
        self._alloc(max(walkers, int(kept.max()) + 1 if len(kept) else 0))
        self.figures.resize(cap[3])

    def configure(self, q) -> None:
        """Take a quality's bands and caps (contravia/quality.py): walkers, level 0 and 1 caps, bodies."""
        self.lod, self.cap = tuple(q.people_lod), tuple(q.people)
        self._sized(self.cap)

    def set_limit(self, n: int | None) -> None:
        """At most n walkers whatever the preset says (None: the preset's)."""
        self.limit = n
        self._sized(self.cap)

    # -- one frame

    def update(self, cells, eye, dt: float, hour: float = 12.0, rain: float = 0.0) -> None:
        """Follow the cells drawn in full, keep the walker count, move everyone, change levels, pose them."""
        self._frame += 1
        if self.figures.step():
            self.clips = self.figures.clip_table()
        eye = np.asarray(eye, float)
        changed = self._follow(set(cells))
        if changed:
            self._table()
        self._count(eye, hour, rain)
        self._advance(dt)
        if changed or self._frame % EVERY == 0:
            self._levels(eye)
        self._fade_step()
        self._pose()

    # -- cells and lanes

    def _follow(self, cells: set) -> bool:
        changed = False
        for c in cells:
            self._absent.pop(c, None)
        for c in [c for c in self.lanes if c not in cells]:
            if self._frame - self._absent.setdefault(c, self._frame) >= LINGER:
                del self._absent[c]
                self._kept[c] = self.lanes.pop(c)
                if len(self._kept) > KEEP:
                    del self._kept[next(iter(self._kept))]
                changed = True
        for c in [c for c in self._pending if c not in cells]:
            self._pending.pop(c).cancel()
        for c in cells:
            if c in self.lanes or c in self._pending:
                continue
            if c in self._kept:
                self.lanes[c] = self._kept.pop(c)
                changed = True
            else:
                self._pending[c] = self._pool.submit(self._lay, *c)
        for c, fut in list(self._pending.items()):
            if fut.done():
                del self._pending[c]
                self.lanes[c] = fut.result()
                changed = True
        return changed

    def _lay(self, i, j) -> Lanes:
        t = perf_counter()
        lanes = lay(self.city, i, j)
        self.timings.append(perf_counter() - t)
        return lanes

    def _table(self) -> None:
        """One table of every open cell's lanes; walkers whose cell closed leave."""
        cells = [c for c in self.lanes if len(self.lanes[c])]
        offs, n_lane, n_pt, parts = {}, 0, 0, []
        for c in cells:
            ln = self.lanes[c]
            offs[c] = n_lane
            parts.append((ln, n_pt))
            n_lane += len(ln)
            n_pt += len(ln.pts)
        if parts:
            self.t_pts = np.concatenate([ln.pts for ln, _ in parts])
            self.t_start = np.concatenate([ln.start + p0 for ln, p0 in parts])
            for k in ("count", "step", "length", "loop", "zone", "weight"):
                setattr(self, "t_" + k, np.concatenate([getattr(ln, k) for ln, _ in parts]))
            self.t_cell = np.concatenate([np.full(len(ln), i) for i, (ln, _) in enumerate(parts)])
        else:
            self.t_pts = np.zeros((0, 3))
            self.t_start, self.t_count, self.t_cell, self.t_zone = (np.zeros(0, np.int64) for _ in range(4))
            self.t_step, self.t_length, self.t_weight = np.zeros(0), np.zeros(0), np.zeros(0)
            self.t_loop = np.zeros(0, bool)
        self.cells_open, self._offs = cells, offs
        for w in np.flatnonzero(self.alive & ~self.leaving):
            c = self.cell[w]
            if c in offs:
                self.lane[w] = offs[c] + self.local[w]
            else:
                self._send_away([w], now=False)

    # -- how many

    def _count(self, eye, hour: float, rain: float) -> None:
        total = float(self.t_weight.sum())
        rain = min(max(rain, 0.0), 1.0)
        self.target = int(min(len(self.alive), RATE * total) * hour_factor(hour) * (1.0 - 0.7 * rain))
        live = np.flatnonzero(self.alive & ~self.leaving)
        if len(live) > self.target:
            d = np.linalg.norm(self.pos[live] - eye, axis=1)
            self._send_away(live[np.argsort(-d)[: min(SPAWNS, len(live) - self.target)]], now=False)
        elif len(live) < self.target and total > 0:
            for w in np.flatnonzero(~self.alive)[: min(SPAWNS, self.target - len(live))]:
                self._spawn(int(w), eye)

    def _spawn(self, w: int, eye) -> None:
        """Put walker w on a lane, the farthest from eye of four draws, on a ready body of its zone."""
        lanes = self.rng.choice(len(self.t_weight), size=4, p=self.t_weight / self.t_weight.sum())
        s = self.rng.random(4) * self.t_length[lanes]
        xyz, _ = self._at(lanes, s)
        k = int(np.argmax(np.linalg.norm(xyz - eye, axis=1)))
        lane = int(lanes[k])
        bodies = self.figures.ready(int(self.t_zone[lane]))
        if not bodies:
            return
        body = int(self.rng.choice(bodies))
        r = self.rng.random()
        mode = WALK if r < SHARES[0] else IDLE if r < SHARES[0] + SHARES[1] else RUN
        if mode == RUN and (body >= len(self.clips) or (self.clips[body, len(rules.WALK):len(rules.WALK) + len(rules.RUN)] < 0).any()):
            mode = WALK
        c = self.cells_open[int(self.t_cell[lane])]
        self.cell[w] = c
        self.alive[w], self.leaving[w], self.fading[w] = True, False, False
        self.lane[w], self.local[w], self.s[w] = lane, lane - self._offs[c], float(s[k])
        self.dir[w] = 1.0 if self.rng.random() < 0.5 else -1.0
        self.body[w], self.phase[w] = body, self.rng.random()
        if mode == RUN:
            self.clip[w] = len(rules.WALK) + int(self.rng.integers(len(rules.RUN)))
        else:
            self.clip[w] = int(self.rng.integers(len(rules.WALK)))
        self._mode(np.array([w]), mode)
        self.timer[w] = self.rng.uniform(*IDLE_FOR) if mode == IDLE else 0.0
        self.level[w], self.shown[w], self.ent[w], self.ghost[w] = -1, -1, -1, -1
        self._place(np.array([w]))

    def _mode(self, ws: np.ndarray, mode: int) -> None:
        """Set walkers ws to mode, keeping their walking or running bucket for when they walk on."""
        self.mode[ws] = mode
        for w in ws:
            self.rate[w] = self._rate(int(w))

    def _send_away(self, ws, now: bool) -> None:
        """Walkers ws leave: their figure fades out (or goes now) and their slot frees once it has."""
        for w in np.atleast_1d(np.asarray(ws, np.int64)):
            if not self.alive[w]:
                continue
            ents = [int(e) for e in (self.ent[w], self.ghost[w]) if e >= 0]
            if now or not ents:
                if ents:
                    self.engine.destroy(np.array(ents))
                self.alive[w] = self.leaving[w] = self.fading[w] = False
                self.ent[w] = self.ghost[w] = self.level[w] = self.shown[w] = -1
                continue
            if self.ghost[w] >= 0:
                self.engine.destroy(np.array([self.ghost[w]]))
            self.leaving[w], self.ghost[w], self.ent[w] = True, self.ent[w], -1
            self.t0[w], self.fading[w] = self.clock(), True

    # -- moving

    def _at(self, lanes, s):
        """Points s metres along lanes, and the direction each lane runs there."""
        n, h = self.t_count[lanes], self.t_step[lanes]
        f = np.clip(s / h, 0.0, n - 1 - 1e-9)
        i = f.astype(np.int64)
        a = self.t_pts[self.t_start[lanes] + i]
        b = self.t_pts[self.t_start[lanes] + i + 1]
        return a + (b - a) * (f - i)[:, None], b - a

    def _place(self, ws: np.ndarray) -> None:
        if not len(ws):
            return
        xyz, d = self._at(self.lane[ws], self.s[ws])
        self.pos[ws], self.rot[ws] = xyz, yaw_quat(d * self.dir[ws, None])

    def _speed(self, ws) -> np.ndarray:
        m = self.mode[ws]
        bucket = np.array(rules.CLIPS)[self.clip[ws], 1].astype(float)
        return np.where(m == IDLE, 0.0, np.where(m == CROSS, rules.CROSS, bucket))

    def _rate(self, w: int) -> float:
        from manifoldx.humanoid.cycles import cadence

        key = (int(self.body[w]), int(self.mode[w]), float(self._speed(np.array([w]))[0]))
        r = self.rates.get(key)
        if r is None:
            r = self.rates[key] = cadence(KIND[key[1]], key[2], self.figures.body(key[0])[0])
        return r

    def _advance(self, dt: float) -> None:
        live = np.flatnonzero(self.alive & ~self.leaving)
        if not len(live):
            return
        lane = self.lane[live]
        length, loop = self.t_length[lane], self.t_loop[lane]
        idle = self.mode[live] == IDLE
        w = live[idle]
        self.timer[w] -= dt
        up = w[self.timer[w] <= 0]
        if len(up):
            at_end = ~self.t_loop[self.lane[up]]
            start = at_end & (self.s[up] <= 0.01)
            end = at_end & (self.s[up] >= self.t_length[self.lane[up]] - 0.01)
            self.dir[up[start]], self.dir[up[end]] = 1.0, -1.0  # it stood at an open end: it turns back
            self._mode(up, WALK)
        w, length, loop = live[~idle], length[~idle], loop[~idle]
        s = self.s[w] + self.dir[w] * self._speed(w) * dt
        s = np.where(loop, np.mod(s, np.maximum(length, 1e-9)), s)
        left = np.where(self.dir[w] < 0, s, length - s)
        ended = ~loop & (left <= 0)
        looking = ~loop & ~ended & (left < LOOK) & (self.mode[w] == WALK)
        self.s[w] = np.clip(s, 0.0, length)
        if ended.any():
            self.timer[w[ended]] = self.rng.uniform(*STAND, size=int(ended.sum()))
            self._mode(w[ended], IDLE)
        if looking.any():
            self._mode(w[looking], CROSS)
        self.phase[live] = (self.phase[live] + self.rate[live] * dt) % 1.0
        self._place(live)

    # -- levels, entities, poses

    def _levels(self, eye) -> None:
        live = np.flatnonzero(self.alive & ~self.leaving)
        if not len(live):
            return
        d = np.linalg.norm(self.pos[live] - eye, axis=1)
        bands = np.asarray(self.lod)
        new = np.searchsorted(bands, d)
        old = self.level[live]
        near, far = np.searchsorted(bands, d - HYSTERESIS), np.searchsorted(bands, d + HYSTERESIS)
        lv = np.where((old >= 0) & (old >= near) & (old <= far), old, new)
        order = np.argsort(d)
        for level, cap in ((0, self.cap[1]), (1, self.cap[2])):
            at = order[lv[order] == level]
            lv[at[cap:]] = level + 1
        for w, level in zip(live, lv):
            level = int(min(level, HIDDEN))
            shown = self.figures.nearest(int(self.body[w]), level)[0] if level < HIDDEN else -1
            if level != self.level[w] or shown != self.shown[w]:
                self._swap(int(w), level)

    def _swap(self, w: int, level: int) -> None:
        """Walker w's figure becomes level (HIDDEN: none), fading in over the pixels the old one leaves."""
        from manifoldx.components import Fade, Material, Transform

        shown, mesh = self.figures.nearest(int(self.body[w]), level) if level < HIDDEN else (-1, None)
        if self.ghost[w] >= 0:
            self.engine.destroy(np.array([self.ghost[w]]))
        self.ghost[w], self.ent[w] = self.ent[w], -1
        self.level[w], self.shown[w] = level, shown
        if mesh is not None:
            h = self.engine.spawn(mesh, Material(self.material), Transform(pos=self.pos[w], rot=self.rot[w]),
                                  self.Person(k=float(w)), Fade(value=0.0))
            self.ent[w] = h.index
        if self.ent[w] >= 0 or self.ghost[w] >= 0:
            self.t0[w], self.fading[w] = self.clock(), True

    def _fade_step(self) -> None:
        ws = np.flatnonzero(self.fading)
        if not len(ws):
            return
        fade = self.engine.store._components["Fade"]
        t = np.minimum((self.clock() - self.t0[ws]) / FADE, 1.0)
        for w, tw in zip(ws, t):
            if self.ent[w] >= 0:
                fade[self.ent[w]] = tw
            if self.ghost[w] >= 0:
                fade[self.ghost[w]] = -(1.0 - tw) if self.ent[w] >= 0 else 1.0 - tw
            if tw >= 1.0:
                self.fading[w] = False
                if self.ghost[w] >= 0:
                    self.engine.destroy(np.array([self.ghost[w]]))
                    self.ghost[w] = -1
                if self.leaving[w]:
                    self.alive[w] = self.leaving[w] = False
                    self.level[w] = self.shown[w] = -1

    def _pose(self) -> None:
        ws = np.flatnonzero(self.alive & ((self.ent >= 0) | (self.ghost >= 0)))
        if not len(ws) or not len(self.clips):
            return
        m = self.mode[ws]
        col = np.where(m == IDLE, CLIP_IDLE, np.where(m == CROSS, CLIP_CROSS, self.clip[ws]))
        clip = self.clips[self.body[ws], col]
        clip = np.where(clip >= 0, clip, self.clips[self.body[ws], 0])
        ok = clip >= 0
        ws, clip = ws[ok], clip[ok]
        ents = np.concatenate([self.ent[ws], self.ghost[ws]])
        clips = np.concatenate([clip, clip])
        frames = np.concatenate([self.phase[ws], self.phase[ws]]) * FRAMES
        drawn = ents >= 0
        if drawn.any():
            self.engine.set_pose_frames(ents[drawn], clips[drawn], frames[drawn])

    # -- what an agent asks

    def reseed(self, seed: int) -> None:
        """Send everyone away at once and draw the crowd again from seed."""
        self._send_away(np.flatnonzero(self.alive), now=True)
        self.rng = np.random.default_rng(seed)

    def waiting(self) -> int:
        """Cells whose lanes are still being laid."""
        return len(self._pending)

    def counts(self) -> dict:
        live = self.alive & ~self.leaving
        lv = self.level[live]
        return {"walkers": int(live.sum()), "target": self.target, "cells": len(self.lanes),
                "lanes": len(self.t_start), **{f"lod{k}": int((lv == k).sum()) for k in range(3)},
                "hidden": int((lv == HIDDEN).sum()), "figures": self.figures.counts()}

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.figures.close()
