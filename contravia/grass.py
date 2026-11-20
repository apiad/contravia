"""The grass the game draws near the camera: instanced patches of blades that sway in the shader.

Every cell drawn in full finds, on a thread, the spots of a STEP-metre grid where grass grows: its green (the parks,
woods and meadows OSM maps) and the lawns of the parks contravia.parks laid out grow a lawn, and every other spot of
bare ground grows the tufts of a bare lot. Bare ground is what is left of the land once its streets, roads,
pavements, park motifs, bridges, cuts, buildings, filler houses and landmarks are taken out with a margin, and the
sea. Each spot holds one patch: a STEP-metre square of blades, one geometry for every patch of a level, dense within
NEAR of the camera and sparse out to FAR. The blades move with the wind and part round the car in manifoldx's vertex
shader (StandardMaterial's sway and the engine's benders), so the CPU only spawns and drops the patches whose level
changes when the camera has moved MOVE metres. A patch's turn, height and shade come from its spot, so a cell
opens with the same grass every time.

The city data holds no plazas, car parks, rail yards, inland water or industrial yards outside the blocks walks.py
paves, and the ground there is drawn as any other: grass grows on them as on a bare lot.
"""

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import shapely

from contravia import parks
from contravia.geo import CELL
from contravia.rules import trees
from contravia.rules.buildings import filler_rings
from contravia.rules.ground import SEA, relief

STEP = 2.0  # metres between two patches' centres, and the side of a patch
NEAR, FAR = 20.0, 45.0  # metres from the camera: dense patches out to NEAR, sparse ones out to FAR
BLADES = (60, 16)  # blades in a dense and in a sparse patch
MOVE = 4.0  # metres the camera moves before the patches' levels are checked again
CLEAR = 1.0  # metres between a patch's centre and a street, pavement, motif, building or landmark
ROAD_CLEAR = 1.5  # metres between a patch's centre and a road's edge
TUFT = 8.0  # metres: a bare lot's grass grows in tufts about this wide
BARE_SHARE = 0.55  # of a bare lot's spots, about this share holds a patch
YARD_STEP, YARD_SHARE = 12.0, 0.3  # metres between a yard tree's candidate spots, and the share that grow one
HOUSE_CLEAR = 3.0  # metres between a yard tree's trunk and a house, so its crown stands off the wall
TREE_CLEAR = 1.5  # metres between a yard tree's trunk and a street or pavement
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


def _hash(xz) -> np.ndarray:
    """A number in [0, 1) for each (x, z), the same every time."""
    h = np.sin(xz[:, 0] * 12.9898 + xz[:, 1] * 78.233) * 43758.5453
    return h - np.floor(h)


def grid(i: int, j: int) -> np.ndarray:
    """(N, 2) the points of the STEP grid in cell (i, j), from its corner up to but not on its far edges."""
    g = np.arange(0.0, CELL, STEP)
    gx, gz = np.meshgrid(i * CELL + g, j * CELL + g)
    return np.stack([gx.ravel(), gz.ravel()], 1)


def _hit(p, geoms, within: float = 0.0) -> np.ndarray:
    """For each (x, z) of p, whether it lies within `within` metres of any of the shapely geoms (on or in at 0)."""
    hit = np.zeros(len(p), bool)
    geoms = [g for g in geoms if g is not None]
    if not geoms or not len(p):
        return hit
    tree, pts = shapely.STRtree(geoms), shapely.points(p)
    k = tree.query(pts, predicate="dwithin", distance=within)[0] if within else tree.query(pts, "intersects")[0]
    hit[k] = True
    return hit


def _tris(tris) -> list:
    """The shapely triangles of (T, 3, 3) x, y, z or (T, 3, 2) x, z triangles, or none for None."""
    if tris is None or not len(tris):
        return []
    t = np.asarray(tris, float)
    t = t[:, :, [0, 2]] if t.shape[2] == 3 else t
    return list(shapely.polygons(np.concatenate([t, t[:, :1]], 1)))


def _rings(rings) -> list:
    """The shapely polygons of (K, 2) rings with three corners or more."""
    return [shapely.Polygon(r) for r in (np.asarray(r, float) for r in rings) if len(r) >= 3]


def lawns(city, i: int, j: int) -> list:
    """The polygons of cell (i, j)'s green and of the parks contravia.parks laid out that meet it; the laid parks
    have no OSM green under them, and their kerbs and paths are pavements, which open_ground() leaves out."""
    box = shapely.box(i * CELL, j * CELL, (i + 1) * CELL, (j + 1) * CELL)
    polys = _tris(city.cell(i, j).green_tris)
    for a in trees.arrays_of(city, i, j):
        if "park_kind" in a and len(a["park_kind"]):
            tree = parks._lookup(a)[0]
            polys += list(tree.geometries.take(tree.query(box)))
    return polys


def open_ground(city, i: int, j: int, p, landmarks=None, clear=CLEAR, house_clear=CLEAR,
                road_clear=ROAD_CLEAR) -> np.ndarray:
    """Whether each (x, z) of p, points of cell (i, j), stands on open land: `clear` metres off the cell's streets,
    pavements, park motifs, bridges and cuts, `house_clear` metres off the buildings and filler houses of it and
    its neighbours and off a landmark's rings, `road_clear` metres off a road's edge, and over the sea. A house is
    its whole ring: clearance.boxes only drops a ring or cuts it back off the street, never grows it."""
    p = np.asarray(p, float).reshape(-1, 2)
    if not len(p):
        return np.zeros(0, bool)
    c = city.cell(i, j)
    near = [city.cell(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
    reach = max(clear, house_clear, road_clear) + 1.0
    lo, hi = p.min(0) - reach, p.max(0) + reach
    houses = [r for n in near for r in n.buildings]
    houses += [r for n in near for block, seed in zip(n.fillers, n.filler_seeds, strict=True)
               for r in filler_rings(block, int(seed))]
    if landmarks is not None:
        houses += landmarks.clear + landmarks.wall_rings
    houses = [r for r in (np.asarray(r, float) for r in houses)
              if len(r) and (r.min(0) <= hi).all() and (r.max(0) >= lo).all()]
    roads = np.concatenate([n.roads for n in near]).reshape(-1, 5).astype(float)
    pad = roads[:, 4:5] / 2 + road_clear
    roads = roads[((np.minimum(roads[:, 0:2], roads[:, 2:4]) - pad <= hi).all(1)
                   & (np.maximum(roads[:, 0:2], roads[:, 2:4]) + pad >= lo).all(1))]
    lines = shapely.linestrings(roads[:, :4].reshape(-1, 2, 2)) if len(roads) else []
    # A road's edge stands road_clear from a spot when its strip widened by road_clear - clear is clear away.
    strips = list(shapely.buffer(lines, roads[:, 4] / 2 + road_clear - clear, quad_segs=2)) if len(roads) else []
    paved = (_tris(c.street_tris) + _tris(c.walk_tris) + _tris(c.park_tris) + _tris(c.struct_tris)
             + _tris(c.street_cuts) + strips)
    keep = ~_hit(p, paved, clear) & ~_hit(p, _rings(houses), house_clear)
    if relief(city.terrain):
        keep &= city.terrain.height(p[:, 0], p[:, 1]) >= SEA
    return keep


def spots(city, i: int, j: int, landmarks=None) -> tuple[np.ndarray, np.ndarray]:
    """The (N, 2) centres of cell (i, j)'s patches and whether each stands on a bare lot rather than a lawn.

    A lawn is the cell's green and its laid parks; every other spot of open ground (open_ground) is a bare lot,
    which keeps only its tufts (about BARE_SHARE of it)."""
    p = grid(i, j)
    lawn = _hit(p, lawns(city, i, j))
    tuft = 0.6 * _hash(np.floor(p / TUFT) * TUFT) + 0.4 * _hash(p)
    keep = (lawn | (tuft < BARE_SHARE)) & open_ground(city, i, j, p, landmarks)
    return p[keep], ~lawn[keep]


def yard_trees(city, i: int, j: int, landmarks=None) -> np.ndarray:
    """(N, 2) the trees of a built cell's yards and lots: about one per YARD_STEP square in YARD_SHARE of them,
    jittered and seeded from the cell, on open ground off its green and laid parks (which grow their own trees),
    HOUSE_CLEAR metres off every house and landmark, TREE_CLEAR off pavements and streets and WILD_CLEAR off a
    road's edge. A cell with no buildings or filler houses has none: it grows wild trees instead."""
    c = city.cell(i, j)
    if not (len(c.buildings) or len(c.fillers)):
        return np.zeros((0, 2))
    rng = np.random.default_rng(abs(i * 1_000_003 + j) + 29)
    step = CELL / round(CELL / YARD_STEP)  # a whole number of squares a side, so no jitter leaves the cell
    t = (np.arange(round(CELL / YARD_STEP)) + 0.5) * step
    gx, gz = np.meshgrid(i * CELL + t, j * CELL + t)
    p = np.stack([gx.ravel(), gz.ravel()], 1) + rng.uniform(-0.45, 0.45, (gx.size, 2)) * step
    p = p[rng.random(len(p)) < YARD_SHARE]
    p = p[~_hit(p, lawns(city, i, j))]
    return p[open_ground(city, i, j, p, landmarks, TREE_CLEAR, HOUSE_CLEAR, trees.WILD_CLEAR)]


class Lawn:
    """One cell's patches: their spots, the ground under them, each one's level (-1 not drawn, 0 dense, 1
    sparse) and the entity that draws it (-1 none). A bare lot's patches are shorter and drier than a lawn's."""

    def __init__(self, xz: np.ndarray, terrain, bare=None):
        self.xz = xz
        self.y = terrain.height(xz[:, 0], xz[:, 1]) + 0.02 if len(xz) else np.zeros(0)
        h = _hash(xz) if len(xz) else np.zeros(0)
        b = np.zeros(len(xz), bool) if bare is None else np.asarray(bare, bool)
        self.turn = h * 2 * np.pi
        self.height = np.where(b, 0.45 + 0.3 * h, 0.8 + 0.5 * h)
        self.tint = np.stack([0.85 + 0.3 * h, 0.9 + 0.2 * (1 - h), 0.8 + 0.2 * h], 1).reshape(-1, 3)
        self.tint[b] *= (1.25, 1.0, 0.6)  # straw over green
        self.level = np.full(len(xz), -1)
        self.ents = np.full(len(xz), -1, np.int64)
        self.centre = xz.mean(0) if len(xz) else np.zeros(2)
        self.reach = float(np.hypot(*(xz - self.centre).T).max()) if len(xz) else 0.0
        self.drawn = 0  # patches drawn at either level


class Grass:
    """The patches of the cells drawn in full, kept in step with the city's streamer."""

    def __init__(self, engine, city, landmarks=None, workers: int = 1):
        from manifoldx.components import Mesh
        from manifoldx.resources import StandardMaterial

        self.engine, self.city, self.landmarks = engine, city, landmarks
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
            self._pending[c] = self._pool.submit(self._lay, c)
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

    def _lay(self, c: tuple[int, int]) -> Lawn:
        """Cell c's lawn: its spots and the ground under them."""
        xz, bare = spots(self.city, *c, self.landmarks)
        return Lawn(xz, self.city.terrain, bare)

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
