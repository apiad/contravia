"""The grass the game draws on the green near the camera: instanced patches of blades that sway in the shader.

Every cell drawn in full finds, on a thread, the spots of a STEP-metre grid that lie in its green (the parks, woods
and meadows it grows trees on) and clear of its streets, roads, pavements and park motifs. Each spot holds one
patch: a STEP-metre square of blades, one geometry for every patch of a level, dense within NEAR of the camera and
sparse out to FAR. The blades move with the wind and part round the car in manifoldx's vertex shader
(StandardMaterial's sway and the engine's benders), so the CPU only spawns and drops the patches whose level
changes when the camera has moved MOVE metres. A patch's turn, height and shade come from its spot, so a cell
opens with the same grass every time.
"""

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from contravia.clearance import roads_around
from contravia.rules import trees

STEP = 2.0  # metres between two patches' centres, and the side of a patch
NEAR, FAR = 20.0, 45.0  # metres from the camera: dense patches out to NEAR, sparse ones out to FAR
BLADES = (60, 16)  # blades in a dense and in a sparse patch
MOVE = 4.0  # metres the camera moves before the patches' levels are checked again
ROAD_CLEAR = 1.5  # metres between a patch's centre and a road's edge
CHUNK = 512  # spots tested at once against a cell's triangles, which bounds the arrays the test builds
BASE, TIP = (0.10, 0.22, 0.04), (0.45, 0.62, 0.18)  # linear colours of a blade's foot and tip
BENDER = (2.6, 0.9)  # radius and strength of the parting round the car


def patch(blades: int, height=(0.25, 0.55), width=0.05, seed=0) -> dict:
    """The geometry of a STEP-metre square of blades: each blade a strip of two segments (5 vertices, 3
    triangles) that narrows to its tip, dark at the foot and light at the tip, leaning a little its own way.
    Every vertex faces up, so a field is lit as one soft surface, and each triangle is drawn on both sides."""
    rng = np.random.default_rng(seed)
    base, tip = np.array(BASE), np.array(TIP)
    pos, col, idx = [], [], []
    for _ in range(blades):
        foot = np.array([*rng.uniform(-STEP / 2, STEP / 2, 1), 0.0, *rng.uniform(-STEP / 2, STEP / 2, 1)])
        h = rng.uniform(*height)
        a = rng.uniform(0, np.pi)
        side = np.array([np.cos(a), 0.0, np.sin(a)]) * width
        lean = np.array([rng.normal(0, 0.12), 0.0, rng.normal(0, 0.12)]) * h
        mid = foot + 0.4 * lean + (0.0, 0.55 * h, 0.0)
        top = foot + lean + (0.0, h, 0.0)
        k = len(pos)
        pos += [foot - side, foot + side, mid - 0.6 * side, mid + 0.6 * side, top]
        col += [base, base, (base + tip) / 2, (base + tip) / 2, tip]
        idx += [(k, k + 1, k + 2), (k + 1, k + 3, k + 2), (k + 2, k + 3, k + 4)]
    idx = np.array(idx, np.uint32)
    return {"positions": np.array(pos, np.float32), "normals": np.tile(np.float32([0, 1, 0]), (len(pos), 1)),
            "colors": np.array(col, np.float32), "indices": np.vstack([idx, idx[:, ::-1]]).ravel()}


def in_tris(tris) -> np.ndarray:
    """The (N, 2) points of the STEP grid, in world metres, inside the (T, 3, 2) triangles."""
    out = []
    for t in np.asarray(tris, float).reshape(-1, 3, 2):
        lo, hi = np.floor(t.min(0) / STEP), np.ceil(t.max(0) / STEP)
        gx, gz = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        p = np.stack([gx.ravel(), gz.ravel()], 1) * STEP
        a, b, c = t
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-9:
            continue
        u = ((b[1] - c[1]) * (p[:, 0] - c[0]) + (c[0] - b[0]) * (p[:, 1] - c[1])) / den
        v = ((c[1] - a[1]) * (p[:, 0] - c[0]) + (a[0] - c[0]) * (p[:, 1] - c[1])) / den
        out.append(p[(u >= 0) & (v >= 0) & (u + v <= 1)])
    return np.unique(np.vstack(out), axis=0) if out else np.zeros((0, 2))


def clear_of_tris(xz, tris) -> np.ndarray:
    """For each (x, z), whether it and the points a metre round it lie off every (T, 3, 3) triangle, tested
    CHUNK spots at a time against the triangles near them."""
    keep = np.ones(len(xz), bool)
    if tris is None or not len(tris) or not len(xz):
        return keep
    tris = np.asarray(tris, float)
    lo, hi = tris[:, :, [0, 2]].min(1), tris[:, :, [0, 2]].max(1)
    for s in range(0, len(xz), CHUNK):
        p = xz[s:s + CHUNK]
        near = ((lo <= p.max(0) + 1.0) & (hi >= p.min(0) - 1.0)).all(1)
        keep[s:s + CHUNK] = trees.off_streets(p, tris[near])
    return keep


def spots(city, i: int, j: int) -> np.ndarray:
    """(N, 2) the centres of cell (i, j)'s patches: grid spots in its green, off its streets, pavements and park
    motifs, and ROAD_CLEAR metres off every road's edge."""
    c = city.cell(i, j)
    if c.green_tris is None or not len(c.green_tris):
        return np.zeros((0, 2))
    p = in_tris(c.green_tris)
    for tris in (c.street_tris, c.walk_tris, c.park_tris):
        p = p[clear_of_tris(p, tris)]
    return p[trees.clear_of(p, roads_around(city, i, j), ROAD_CLEAR)]


class Lawn:
    """One cell's patches: their spots, the ground under them, each one's level (-1 not drawn, 0 dense, 1
    sparse) and the entity that draws it (-1 none)."""

    def __init__(self, xz: np.ndarray, terrain):
        self.xz = xz
        self.y = terrain.height(xz[:, 0], xz[:, 1]) + 0.02 if len(xz) else np.zeros(0)
        h = np.sin(xz[:, 0] * 12.9898 + xz[:, 1] * 78.233) * 43758.5453  # a hash of the spot, in [0, 1)
        h -= np.floor(h)
        self.turn = h * 2 * np.pi
        self.height = 0.8 + 0.5 * h
        self.tint = np.stack([0.85 + 0.3 * h, 0.9 + 0.2 * (1 - h), 0.8 + 0.2 * h], 1)
        self.level = np.full(len(xz), -1)
        self.ents = np.full(len(xz), -1, np.int64)
        self.centre = xz.mean(0) if len(xz) else np.zeros(2)
        self.reach = float(np.hypot(*(xz - self.centre).T).max()) if len(xz) else 0.0
        self.drawn = 0  # patches drawn at either level


class Grass:
    """The patches of the cells drawn in full, kept in step with the city's streamer."""

    def __init__(self, engine, city, workers: int = 1):
        from manifoldx.components import Mesh
        from manifoldx.resources import StandardMaterial

        self.engine, self.city = engine, city
        self.material = StandardMaterial("#ffffff", roughness=0.95, vertex_colors=True, tinted=True,
                                         sway=0.35, sway_reach=0.6)
        self.meshes = [Mesh(patch(BLADES[0], seed=1)), Mesh(patch(BLADES[1], height=(0.3, 0.6), width=0.09, seed=2))]
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="grass")
        self._pending: dict[tuple[int, int], Future] = {}
        self.lawns: dict[tuple[int, int], Lawn] = {}
        self._at = None  # where the camera stood when the levels were last checked

    def update(self, cells, eye, car=None) -> None:
        """Follow the set of cells drawn in full: lay the new ones' lawns on a thread, drop the cells no
        longer drawn, and when the camera has moved MOVE metres or a cell came in, give every patch its level.
        With car (x, z), the grass parts round it."""
        cells = set(cells)
        for c in [c for c in self.lawns if c not in cells]:
            self._drop(self.lawns.pop(c), slice(None))
        for c in [c for c in self._pending if c not in cells]:
            self._pending.pop(c).cancel()
        for c in cells - self.lawns.keys() - self._pending.keys():
            self._pending[c] = self._pool.submit(lambda c=c: Lawn(spots(self.city, *c), self.city.terrain))
        came = False
        for c, fut in list(self._pending.items()):
            if fut.done():
                del self._pending[c]
                self.lawns[c] = fut.result()
                came = True
        eye = np.asarray(eye, float)
        if came or self._at is None or np.hypot(*(eye[[0, 2]] - self._at)) >= MOVE:
            self._at = eye[[0, 2]].copy()
            for lawn in self.lawns.values():
                self.place(lawn, eye)
        if car is not None:
            self.engine.benders = [(float(car[0]), float(car[1]), *BENDER)]

    def place(self, lawn: Lawn, eye) -> None:
        """Give each of lawn's patches the level its distance from eye calls for, respawning those that change."""
        from manifoldx.components import Material, Tint, Transform

        if not lawn.drawn and np.hypot(*(lawn.centre - eye[[0, 2]])) - lawn.reach >= FAR:
            return  # a lawn wholly out of reach, none of it drawn, has no patch to move
        d = np.hypot(lawn.xz[:, 0] - eye[0], lawn.xz[:, 1] - eye[2])
        new = np.where(d < NEAR, 0, np.where(d < FAR, 1, -1))
        moved = np.flatnonzero(new != lawn.level)
        if not len(moved):
            return
        self._drop(lawn, moved)
        lawn.level[moved] = new[moved]
        lawn.drawn = int((lawn.level >= 0).sum())
        for k in (0, 1):
            idx = moved[new[moved] == k]
            if not len(idx):
                continue
            half = lawn.turn[idx] / 2
            zero = np.zeros(len(idx))
            h = self.engine.spawn(
                self.meshes[k], Material(self.material),
                Transform(pos=np.stack([lawn.xz[idx, 0], lawn.y[idx], lawn.xz[idx, 1]], 1),
                          rot=np.stack([zero, np.sin(half), zero, np.cos(half)], 1),
                          scale=np.stack([zero + 1, lawn.height[idx], zero + 1], 1)),
                Tint(rgb=lawn.tint[idx]), n=len(idx))
            lawn.ents[idx] = h.indices

    def _drop(self, lawn: Lawn, idx) -> None:
        """Destroy the entities of lawn's patches idx."""
        e = lawn.ents[idx]
        e = e[e >= 0]
        if len(e):
            self.engine.destroy(e)
        lawn.ents[idx] = -1
        lawn.level[idx] = -1
        lawn.drawn = int((lawn.level >= 0).sum())

    def counts(self) -> dict:
        """Patches held and drawn per level, for probes."""
        level = np.concatenate([lawn.level for lawn in self.lawns.values()]) if self.lawns else np.zeros(0, int)
        return {"patches": len(level), "dense": int((level == 0).sum()), "sparse": int((level == 1).sum())}

    def close(self) -> None:
        """Stop finding spots."""
        self._pool.shutdown(wait=False, cancel_futures=True)
