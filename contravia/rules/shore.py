"""The shore of a coastal cell: what the ground looks like where it meets the sea, on a grid of STEP metres.

The relief's sea is where the ground lies under rules.ground.SEA. Without a shore the cell's 10 m ground stops
there and the flat water plane meets flat grass in a hard line. A cell with sea within REACH draws its ground
instead on a grid four times finer, and each node of it gets:

- `d`, its signed distance to the waterline: metres to the nearest sea node on land (positive), to the nearest
  land node at sea (negative), smoothed so the waterline curves between the nodes;
- a share of each shore TYPE, from what lies there: a beach where OSM maps beach or sand, a cliff where OSM
  maps one or the ground rises CLIFF_RISE metres within CLIFF_W of the water, mangrove where OSM maps a
  wetland or WorldCover sees mangrove or wetland, rocky shore (the limestone "diente de perro") elsewhere.
  The shares are blurred over about BLUR metres, so one kind of shore turns into the next;
- its height: each type's profile (a beach sloping gently under the water, a low rough rock shelf, a cliff
  standing up to the height inland and dropping straight into the sea, a muddy flat), mixed by the shares and
  back into the relief's ground away from the water. Near anything built (streets, pavements, parks, buildings)
  the ground keeps its height, so no street is buried and no house floats;
- its colour: sand, rock or mud on land, and in the water a band of shallows, light over sand, dark over rock,
  murky by the mangroves, fading into the sea's own colour at its outer edge. That band is drawn SHALLOW_Y over
  the water plane, so it hides the hard edge; beyond it the triangles are left out and the sea plane shows.

Everything is a function of world position and the data, so neighbouring cells agree along their edges, and a
cell away from the sea keeps the old 10 m ground, which a coastal cell's grid matches where it is untouched.
"""

from dataclasses import dataclass

import numpy as np

from contravia.geo import CELL
from contravia.rules.ground import SEA, WATER_Y, N, drawn, grass
from contravia.rules.noise import fbm

STEP = 2.5  # metres between the nodes of a shore grid
NODES = round(CELL / STEP) + 1  # nodes per side of a cell
REACH = 40.0  # metres: a cell draws a shore when the sea comes this close; distances are capped here
MARGIN = int(REACH / STEP) + 4  # nodes measured round a cell, so its edges see all the sea within REACH
BEACH, ROCK, CLIFF, MANGROVE = range(4)
TYPES = ("beach", "rock", "cliff", "mangrove")
BLUR = 25.0  # metres over which one type of shore turns into the next
CLIFF_W = 30.0  # metres inland of the water where a cliff's top is measured
CLIFF_RISE = 4.0  # metres the ground must rise within CLIFF_W of the water to make a cliff
CLIFF_MIN = 5.0  # metres a mapped cliff stands at least over the water, where the relief is flatter
CLIFF_FOOT = WATER_Y - 0.8  # a cliff's foot, under the water
SHALLOW_Y = WATER_Y + 0.04  # the shallows' band, just over the water plane
BEACH_SLOPE = 0.05  # rise over run of the sand from the water's edge
# metres from an OSM feature that still take its type: OSM draws them on its own coast, which can lie some way
# off the relief's, most of all along mangroves, whose tops the relief measures
NEAR = {"beach": 60.0, "cliff": 60.0, "mangrove": 150.0}
SHALLOWS = (30.0, 14.0, 10.0, 26.0)  # metres of lighter water off each type
BAND = ((16.0, 26.0), (5.0, 12.0), (4.0, 9.0), (6.0, 18.0))  # metres inland where each type's colour gives way
KEEP_NEAR, KEEP_FAR = 4.0, 12.0  # metres from anything built where the ground starts to take the shore's shape

SAND, WET_SAND = (0.96, 0.84, 0.52), (0.74, 0.62, 0.38)
ROCK_LIGHT, ROCK_DARK = (0.52, 0.48, 0.41), (0.33, 0.30, 0.26)
CLIFF_ROCK = (0.62, 0.58, 0.50)
MUD = (0.27, 0.24, 0.18)
SEA_COLOUR = (0.10, 0.24, 0.32)  # the water plane's colour (game.WATER): the shallows fade into it
SHALLOW = ((0.40, 0.70, 0.68), (0.18, 0.40, 0.44), (0.14, 0.34, 0.40), (0.22, 0.30, 0.22))
FOAM = (0.80, 0.86, 0.86)


def smooth(a, b, x) -> np.ndarray:
    """0 below a, 1 above b and a smoothstep between."""
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def coastal(terrain, i: int, j: int) -> bool:
    """Whether the ground of cell (i, j) or REACH round it dips under SEA anywhere, sampled every 5 m."""
    t = np.arange(-REACH, CELL + REACH + 0.1, 5.0)
    x, z = np.meshgrid(i * CELL + t, j * CELL + t)
    return bool((terrain.height(x.ravel(), z.ravel()) < SEA).any())


def distance(mask: np.ndarray, step: float = STEP, cap: float = REACH + STEP) -> np.ndarray:
    """The distance in metres from each node of a 2D grid to the nearest True node, exact up to cap and inf
    past it: the squared distance along each row to a True node within cap, then the least over the rows
    within cap of that plus the rows between."""
    k = int(np.ceil(cap / step))
    big = np.inf
    along = np.where(mask, 0.0, big)
    for s in range(1, k + 1):  # the nearest True node along the row, s nodes either side
        hit = np.zeros_like(mask)
        hit[:, s:] |= mask[:, :-s]
        hit[:, :-s] |= mask[:, s:]
        along = np.where(hit, np.minimum(along, float(s * s)), along)
    out = along.copy()
    for s in range(1, k + 1):  # then s rows up or down
        out[s:] = np.minimum(out[s:], along[:-s] + s * s)
        out[:-s] = np.minimum(out[:-s], along[s:] + s * s)
    d = np.sqrt(out) * step
    return np.where(d <= cap, d, np.inf)


def blur(a: np.ndarray, k: int) -> np.ndarray:
    """a averaged over a (2k + 1) square round each node of its first two axes, the edges held."""
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (k, k)
        p = np.pad(a, pad, mode="edge")
        c = np.cumsum(p, axis=axis, dtype=float)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        n = a.shape[axis]
        a = (np.take(c, np.arange(2 * k + 1, n + 2 * k + 1), axis=axis)
             - np.take(c, np.arange(0, n), axis=axis)) / (2 * k + 1)
    return a


def signed(sea: np.ndarray) -> np.ndarray:
    """Each node's signed distance to the waterline between sea and land nodes, capped at REACH: positive on
    land, negative at sea, half a step at the nodes either side of it, smoothed over three nodes."""
    land_d = np.minimum(distance(sea), REACH + STEP) - STEP / 2
    sea_d = np.minimum(distance(~sea), REACH + STEP) - STEP / 2
    d = np.where(sea, -sea_d, land_d)
    return np.clip(blur(d, 1), -REACH, REACH)


def kind(cliff_osm, beach_osm, mangrove, steep) -> np.ndarray:
    """The type of each node, by priority: a mapped cliff, a mapped beach, mangrove, a steep rise, else rock."""
    out = np.full(np.shape(steep), ROCK, np.int8)
    out[steep] = CLIFF
    out[mangrove] = MANGROVE
    out[beach_osm] = BEACH
    out[cliff_osm] = CLIFF
    return out


def shares(types: np.ndarray) -> np.ndarray:
    """(..., 4) shares of each type round each node: the types one-hot, blurred twice over BLUR."""
    onehot = (types[..., None] == np.arange(len(TYPES))).astype(float)
    k = max(1, round(BLUR / STEP / 4))
    return blur(blur(onehot, k), k)


def profiles(d, h, top, rough) -> np.ndarray:
    """(..., 4) heights each type gives nodes at signed distance d over the relief's drawn height h, with top
    the cliff's top inland and rough a noise in [0, 1). Each returns to h by the inner end of its band."""
    d, h = np.asarray(d, float), np.asarray(h, float)
    land = d >= 0
    beach = SHALLOW_Y + BEACH_SLOPE * np.maximum(d, 0)
    rock = SHALLOW_Y + 0.45 * smooth(0.0, 1.5, d) + 0.4 * rough * smooth(0.5, 2.5, d)
    cliff = np.maximum(h, top)
    mud = SHALLOW_Y + 0.12 + 0.02 * np.maximum(d, 0)
    out = np.stack([np.where(land, beach + (h - beach) * smooth(14.0, 24.0, d), SHALLOW_Y),
                    np.where(land, rock + (h - rock) * smooth(6.0, 12.0, d), SHALLOW_Y),
                    np.where(land, cliff + (h - cliff) * smooth(CLIFF_W - 10.0, CLIFF_W, d),
                             np.where(d > -1.2 * STEP, CLIFF_FOOT, SHALLOW_Y)),
                    np.where(land, mud + (h - mud) * smooth(8.0, 20.0, d), SHALLOW_Y)], axis=-1)
    return out


def colours(d, base, rough) -> np.ndarray:
    """(..., 4, 3) colours each type gives nodes at signed distance d over the ground's own colour base, with
    rough a noise in [0, 1). At sea, the type's shallows fading into the sea's colour, foam at the edge."""
    d, base, rough = np.asarray(d, float), np.asarray(base, float), np.asarray(rough, float)[..., None]
    sea = (d < 0)[..., None]
    land_c = [np.add(np.multiply(WET_SAND, 1 - smooth(1.0, 4.0, d)[..., None]),
                     np.multiply(SAND, smooth(1.0, 4.0, d)[..., None])),
              np.add(np.multiply(ROCK_DARK, 1 - rough), np.multiply(ROCK_LIGHT, rough)),
              np.add(np.multiply(CLIFF_ROCK, 0.8 + 0.3 * rough), 0.0),
              np.add(np.multiply(MUD, 0.9 + 0.2 * rough), 0.0)]
    out = []
    for t in range(len(TYPES)):
        a, b = BAND[t]
        g = smooth(a, b, d)[..., None]
        land = land_c[t] * (1 - g) + base * g
        w = smooth(0.0, SHALLOWS[t], -d)[..., None]
        water = np.multiply(SHALLOW[t], 1 - w) + np.multiply(SEA_COLOUR, w)
        if t in (BEACH, ROCK):
            foam = (1 - smooth(0.0, 1.5 if t == BEACH else 1.0, -d))[..., None] * 0.6
            water = water * (1 - foam) + np.multiply(FOAM, foam)
        if t == CLIFF:  # the face and its foot are rock down into the water
            water = np.where((d > -1.2 * STEP)[..., None], land_c[t], water)
        out.append(np.where(sea, water, land))
    return np.stack(out, axis=-2)


def shallows(w) -> np.ndarray:
    """How far out each node's band of shallows reaches, mixed by its shares w."""
    return np.asarray(w, float) @ np.asarray(SHALLOWS)


def keep(dist_built) -> np.ndarray:
    """How much of the shore's shape a node takes, from its distance to anything built: none within KEEP_NEAR."""
    return smooth(KEEP_NEAR, KEEP_FAR, dist_built)


def _coarse(i: int, j: int, terrain):
    """The 10 m ground of cell (i, j) and one ring round it as rules.ground draws it: (N + 2, N + 2) colours and
    normals at its vertices, so a shore grid can match it where the shore leaves it alone."""
    step = CELL / (N - 1)
    t = (np.arange(-1, N + 1)) * step
    x, z = np.meshgrid(i * CELL + t, j * CELL + t)
    x, z = x.ravel(), z.ravel()
    dx = (terrain.height(x + 1, z) - terrain.height(x - 1, z)) / 2
    dz = (terrain.height(x, z + 1) - terrain.height(x, z - 1)) / 2
    n = np.stack([-dx, np.ones_like(dx), -dz], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return grass(x, z, np.hypot(dx, dz)).reshape(N + 2, N + 2, 3), n.reshape(N + 2, N + 2, 3)


def on_coarse(values, i: int, j: int, x, z) -> np.ndarray:
    """values (N + 2, N + 2, k) at the 10 m vertices of cell (i, j) and its ring, at points (x, z) inside the
    cell: straight between them over the same two triangles per square the 10 m ground draws."""
    step = CELL / (N - 1)
    fx, fz = (np.asarray(x, float) - i * CELL) / step + 1, (np.asarray(z, float) - j * CELL) / step + 1
    c = np.clip(np.floor(fx), 0, N).astype(int)
    r = np.clip(np.floor(fz), 0, N).astype(int)
    tx, tz = (fx - c)[:, None], (fz - r)[:, None]
    v00, v01, v10, v11 = values[r, c], values[r, c + 1], values[r + 1, c], values[r + 1, c + 1]
    return np.where(tx + tz <= 1, v00 + tx * (v01 - v00) + tz * (v10 - v00),
                    v11 + (1 - tx) * (v10 - v11) + (1 - tz) * (v01 - v11))


@dataclass
class Field:
    """A coastal cell's shore grid, NODES x NODES from its corner (x0, z0): per node its signed distance d, its
    type shares w (4), how much shore shape it takes (keep), its height y, normal n and colour rgb, and whether
    the ground is left out there (out: past the shallows)."""

    x0: float
    z0: float
    d: np.ndarray
    w: np.ndarray
    keep: np.ndarray
    y: np.ndarray
    n: np.ndarray
    rgb: np.ndarray
    out: np.ndarray

    def _locate(self, x, z):
        fx = np.clip((np.asarray(x, float) - self.x0) / STEP, 0, NODES - 1 - 1e-9)
        fz = np.clip((np.asarray(z, float) - self.z0) / STEP, 0, NODES - 1 - 1e-9)
        c, r = np.floor(fx).astype(int), np.floor(fz).astype(int)
        return c, r, fx - c, fz - r

    def sample(self, values, x, z) -> np.ndarray:
        """values (NODES, NODES, ...) at points (x, z) of the cell, straight between the nodes over the grid's
        two triangles per square, as the mesh draws them."""
        c, r, tx, tz = self._locate(x, z)
        shape = (-1,) + (1,) * (values.ndim - 2)
        tx, tz = tx.reshape(shape), tz.reshape(shape)
        v00, v01, v10, v11 = values[r, c], values[r, c + 1], values[r + 1, c], values[r + 1, c + 1]
        return np.where(tx + tz <= 1, v00 + tx * (v01 - v00) + tz * (v10 - v00),
                        v11 + (1 - tx) * (v10 - v11) + (1 - tz) * (v01 - v11))

    def height(self, x, z) -> np.ndarray:
        """The drawn ground's height at points (x, z) of the cell."""
        return self.sample(self.y, x, z)

    def triangles(self) -> np.ndarray:
        """(T, 3) node indices of the grid's triangles, but those with all three nodes out."""
        a = (np.arange(NODES - 1)[:, None] * NODES + np.arange(NODES - 1)[None, :]).ravel()
        tris = np.concatenate([np.stack([a, a + NODES, a + 1], 1), np.stack([a + 1, a + NODES, a + NODES + 1], 1)])
        return tris[~self.out.ravel()[tris].all(axis=1)]


def field(shore, i: int, j: int) -> Field:
    """The shore Field of cell (i, j) from shore (a contravia.shore.Shore: its terrain, OSM features, cover and
    built world), measured over the cell and MARGIN nodes round it."""
    terrain = shore.terrain
    m = MARGIN
    t = (np.arange(-m, NODES + m)) * STEP
    x, z = np.meshgrid(i * CELL + t, j * CELL + t)
    sea = terrain.height(x, z) < SEA
    d = signed(sea)
    h = drawn(terrain, x, z)
    gz, gx = np.gradient(d, STEP)
    g = np.hypot(gx, gz)
    ux, uz = np.where(g > 1e-6, gx / np.maximum(g, 1e-6), 0.0), np.where(g > 1e-6, gz / np.maximum(g, 1e-6), 0.0)
    reach = np.maximum(CLIFF_W - d, 0.0)  # from each node to the point CLIFF_W inland of the water
    top = drawn(terrain, x + ux * reach, z + uz * reach)
    near = np.abs(d) < REACH
    steep = near & (top >= SHALLOW_Y + CLIFF_RISE)
    marks = {k: np.zeros(x.shape, bool) for k in ("cliff", "beach", "mangrove")}
    import shapely

    box = shapely.box(x.min() - 50, z.min() - 50, x.max() + 50, z.max() + 50)
    pts = None
    for k, mark in marks.items():
        geom = shore.near(k, box)
        if geom is None:
            continue
        if pts is None:
            pts = shapely.points(x[near], z[near])
        mark[near] = shapely.distance(geom, pts) <= NEAR[k]
    if shore.cover is not None:
        codes = shore.cover.sample(x, z)
        marks["mangrove"] |= near & np.isin(codes, (90, 95))
    types = kind(marks["cliff"], marks["beach"], marks["mangrove"], steep)
    w = shares(types)
    top = np.where(marks["cliff"], np.maximum(top, SHALLOW_Y + CLIFF_MIN), top)
    rough = fbm(x.ravel(), z.ravel(), 3.0, seed=11).reshape(x.shape)
    ys = profiles(d, h, top, rough)
    y = (ys * w).sum(axis=-1)
    built = np.full(x.shape, np.inf)
    shaped = near & (d >= 0)
    if shaped.any() and shore.built is not None:
        geom = built_near(shore.built, i, j)
        if geom is not None:
            built[shaped] = shapely.distance(geom, shapely.points(x[shaped], z[shaped]))
    k = keep(built)
    y = np.where(d >= 0, h + k * (y - h), y)
    out = d < -shallows(w)
    # normals: the shore grid's own slopes, giving way to the 10 m ground's past the bands
    sz, sx = np.gradient(y, STEP)
    n_fine = np.stack([-sx, np.ones_like(sx), -sz], axis=-1)
    s = slice(m, m + NODES)
    x, z, d, w, y, k, out, rough, n_fine = (a[s, s] for a in (x, z, d, w, y, k, out, rough, n_fine))
    colour, normal = _coarse(i, j, terrain)
    px, pz = x.ravel(), z.ravel()
    base = on_coarse(colour, i, j, px, pz).reshape(NODES, NODES, 3)
    n_base = on_coarse(normal, i, j, px, pz).reshape(NODES, NODES, 3)
    rgb = (colours(d, base, rough) * w[..., None]).sum(axis=-2)
    shaped_n = (1 - smooth(BAND[CLIFF][1], CLIFF_W, d))[..., None] * k[..., None]
    shaped_n = np.where((d < 0)[..., None], 1.0, shaped_n)
    n = n_base * (1 - shaped_n) + n_fine / np.linalg.norm(n_fine, axis=-1, keepdims=True) * shaped_n
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return Field(i * CELL, j * CELL, d, w, k, y, n, np.clip(rgb, 0, 1), out)


def built_near(world, i: int, j: int):
    """Everything built in the 3 x 3 cells round (i, j) as one shapely geometry (roads at their width,
    buildings, filler blocks, pavements and park motifs), or None."""
    import shapely

    parts = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            c = world.cell(i + a, j + b)
            r = np.asarray(c.roads, float).reshape(-1, 5)
            if len(r):
                lines = shapely.linestrings(np.stack([r[:, 0:2], r[:, 2:4]], axis=1))
                parts.extend(shapely.buffer(lines, r[:, 4] / 2, quad_segs=1))
            for ring in list(c.buildings) + list(c.fillers):
                ring = np.asarray(ring, float).reshape(-1, 2)
                if len(ring) >= 3:
                    parts.append(shapely.Polygon(ring))
            for tris in (c.walk_tris, c.park_tris, c.street_tris):
                if tris is not None and len(tris):
                    t = np.asarray(tris, float).reshape(-1, 3, 3)[:, :, [0, 2]]
                    parts.extend(shapely.polygons(t))
    if not parts:
        return None
    g = shapely.union_all(shapely.make_valid(np.array(parts, dtype=object)))
    return None if g.is_empty else g


# --- what stands on the shore: rocks on the rocky shore and at a cliff's foot, mangroves in the shallows ------

ROCK_EVERY = 3.0  # metres between candidate rock spots, before jitter
ROCK_SHARE = 0.4  # share of candidate spots on the rock band that hold a rock
MANGROVE_EVERY = 5.0  # metres between candidate mangrove spots, before jitter
MANGROVE_SHARE = 0.75


def spots(f: Field, every: float, salt: int) -> tuple[np.ndarray, np.ndarray]:
    """(xz, seeds): a jittered grid of candidate spots `every` metres apart over the cell, seeded by position,
    so a cell holds the same spots every time."""
    from contravia.rules.trees import point_seeds, unit

    n = round(CELL / every)
    t = (np.arange(n) + 0.5) * every
    x, z = np.meshgrid(f.x0 + t, f.z0 + t)
    xz = np.c_[x.ravel(), z.ravel()]
    seeds = point_seeds(xz + salt)
    xz = xz + (np.c_[unit(seeds, 21), unit(seeds, 22)] - 0.5) * every * 0.8
    return xz, seeds


def rocks(f: Field) -> tuple[np.ndarray, np.ndarray]:
    """(xz, seeds) of the rocks on cell f: on the rock band, 0.5 to 8 m from the water and a little into it, and
    at a cliff's foot, away from anything built."""
    from contravia.rules.trees import unit

    xz, seeds = spots(f, ROCK_EVERY, 7)
    d = f.sample(f.d, xz[:, 0], xz[:, 1])
    w = f.sample(f.w, xz[:, 0], xz[:, 1])
    k = f.sample(f.keep, xz[:, 0], xz[:, 1])
    on = (w[:, ROCK] > 0.5) & (d > -3.0) & (d < 8.0) | (w[:, CLIFF] > 0.5) & (d < 0) & (d > -6.0)
    keep_ = on & (k > 0.9) & (unit(seeds, 23) < ROCK_SHARE * (1 - smooth(2.0, 8.0, d) * 0.6))
    return xz[keep_], seeds[keep_]


def mangroves(f: Field) -> tuple[np.ndarray, np.ndarray]:
    """(xz, y): where mangroves stand on cell f, in the shallows and the mud behind them, and the height their
    trunks stand at: the water's on the water, the ground's on land."""
    from contravia.rules.trees import unit

    xz, seeds = spots(f, MANGROVE_EVERY, 13)
    d = f.sample(f.d, xz[:, 0], xz[:, 1])
    w = f.sample(f.w, xz[:, 0], xz[:, 1])
    k = f.sample(f.keep, xz[:, 0], xz[:, 1])
    on = (w[:, MANGROVE] > 0.5) & (d > -0.7 * SHALLOWS[MANGROVE]) & (d < 10.0) & (k > 0.8)
    on &= unit(seeds, 24) < MANGROVE_SHARE
    xz = xz[on]
    y = np.maximum(f.height(xz[:, 0], xz[:, 1]), WATER_Y) if len(xz) else np.zeros(0)
    return xz, y


def dress(mb, f: Field) -> None:
    """Add what the cell's mesh draws on its shore f: its rocks."""
    xz, seeds = rocks(f)
    rock_mesh(mb, xz, seeds, f.height(xz[:, 0], xz[:, 1]) if len(xz) else np.zeros(0))


def rock_mesh(mb, xz, seeds, y) -> None:
    """Add a lumpy low rock at each spot xz, standing on heights y, its size, turn and lumps from its seed."""
    from contravia.rules.trees import unit

    if not len(xz):
        return
    v, faces = _ico()
    size = 0.35 + 0.9 * unit(seeds, 31) ** 2
    flat = 0.45 + 0.3 * unit(seeds, 32)
    yaw = 2 * np.pi * unit(seeds, 33)
    tone = 0.75 + 0.35 * unit(seeds, 34)
    pos, nrm, col, tri = [], [], [], []
    base = 0
    for k in range(len(xz)):
        lump = 0.8 + 0.4 * unit(seeds[k] + np.arange(len(v), dtype=np.uint64), 35)
        p = v * lump[:, None] * (size[k], size[k] * flat[k], size[k] * (0.7 + 0.3 * flat[k]))
        cy, sy = np.cos(yaw[k]), np.sin(yaw[k])
        p = np.c_[p[:, 0] * cy - p[:, 2] * sy, p[:, 1], p[:, 0] * sy + p[:, 2] * cy]
        p += (xz[k, 0], y[k] - 0.15 * size[k] * flat[k], xz[k, 1])
        q = p[faces].reshape(-1, 3)  # flat-shaded: each face its own vertices
        fn = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True)
        pos.append(q)
        nrm.append(np.repeat(fn, 3, axis=0))
        shade = np.repeat(0.85 + 0.15 * fn[:, 1], 3)[:, None]
        col.append(np.clip(np.multiply(ROCK_LIGHT, tone[k]) * shade, 0, 1))
        tri.append(base + np.arange(len(q)).reshape(-1, 3))
        base += len(q)
    mb.add(np.vstack(pos), np.vstack(nrm), np.vstack(col), np.vstack(tri))


def _ico():
    """An icosahedron's 12 unit vertices and its 20 faces, turned outward."""
    from contravia.rules.species import _ICO, _ICO_FACES

    v = np.asarray(_ICO, float)
    f = np.asarray(_ICO_FACES, int)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = (n * v[f].mean(axis=1)).sum(axis=1) < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    return v, f
