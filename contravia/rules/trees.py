"""Where a cell's trees stand and which species each one is (the species: rules/species.py).

No tree is stored. A cell's trees are its parks' trees, the ones its green grows (in_green) and, on land OSM says
nothing about, the wild ones (wild): clumps where a broad noise is high, clearings where it is low, never on a
road or in the sea. Each tree's species comes from its zone (the coast, the old centre, Vedado, a park or the
rest) and a seed from its position; its size, turn, tone and seasonal shift come from the same seed. So the same
cell always grows the same trees, and they look as the game's date says.

A cell's trees are rows, not geometry: the game draws them as instances of their species' meshes
(rules/treegrammar.py) by level of detail (forest.py).
"""

import numpy as np

from contravia.geo import CELL
from contravia.rules.ground import SEA
from contravia.rules.noise import fbm

TREE_EVERY = 150.0  # m² of mapped green (park, wood, meadow) per tree, seeded when its cell opens
WILD_STEP = 13.0  # metres between the candidate spots of wild trees, before jitter
WILD_CLEAR = 3.0  # metres between a wild tree and a road's edge


CLEAR = 1.0  # metres between a tree's trunk and the drawn street
_clock = None


def set_clock(now) -> None:
    """Make today() read the date from now(), a function returning a datetime: the game's clock. The date also
    goes to CONTRAVIA_DATE, which the cell builders' processes inherit when the streamer spawns them."""
    import os

    global _clock
    _clock = now
    os.environ["CONTRAVIA_DATE"] = now().date().isoformat()


def today():
    """The date the trees are drawn for: the game's clock's; in a builder process, the CONTRAVIA_DATE it was
    spawned with; else the machine's."""
    import os
    from datetime import date

    if _clock is not None:
        return _clock().date()
    if os.environ.get("CONTRAVIA_DATE"):
        return date.fromisoformat(os.environ["CONTRAVIA_DATE"])
    return date.today()


def off_streets(xz, street_tris) -> np.ndarray:
    """For each (x, z), whether the trunk and the points CLEAR metres around it all lie off the street
    triangles (T, 3, 3) a cell draws: a junction is wider than the road strips the build keeps trees off."""
    xz = np.asarray(xz, float).reshape(-1, 2)
    if street_tris is None or not len(street_tris) or not len(xz):
        return np.ones(len(xz), bool)
    t = np.asarray(street_tris, float)[:, :, [0, 2]]
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    v0, v1 = b - a, c - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-9
    a, v0, v1, den = a[ok], v0[ok], v1[ok], den[ok]
    keep = np.ones(len(xz), bool)
    for dx, dz in ((0, 0), (CLEAR, 0), (-CLEAR, 0), (0, CLEAR), (0, -CLEAR)):
        p = xz + (dx, dz)
        d = p[:, None, :] - a[None]
        u = (d[..., 0] * v1[:, 1] - v1[:, 0] * d[..., 1]) / den
        v = (v0[:, 0] * d[..., 1] - d[..., 0] * v0[:, 1]) / den
        keep &= ~((u >= 0) & (v >= 0) & (u + v <= 1)).any(axis=1)
    return keep


def wild(i: int, j: int, terrain, roads: np.ndarray):
    """The wild trees of cell (i, j): their (N, 2) spots, sizes and leaf tones, clear of the (R, 5) roads."""
    rng = np.random.default_rng(abs(i * 1_000_003 + j) + 17)
    t = np.arange(WILD_STEP / 2, CELL, WILD_STEP)
    gx, gz = np.meshgrid(i * CELL + t, j * CELL + t)
    p = np.stack([gx.ravel(), gz.ravel()], axis=1) + rng.uniform(-0.45, 0.45, (gx.size, 2)) * WILD_STEP
    density = np.clip((fbm(p[:, 0], p[:, 1], 150.0, seed=3) - 0.4) * 2.5, 0, 0.75)
    keep = (rng.random(len(p)) < density) & (terrain.height(p[:, 0], p[:, 1]) > SEA)
    if keep.any():
        keep[np.flatnonzero(keep)[~clear_of(p[keep], roads)]] = False
    size = rng.uniform(0.7, 1.4, len(p))
    return p[keep], size[keep], 0.8 + 0.4 * fbm(p[keep, 0], p[keep, 1], 30.0, seed=4)



def in_green(tris, seeds) -> np.ndarray:
    """The trees of a cell's green: one per TREE_EVERY m² of each (T, 3, 2) triangle, at spots seeded by the
    green area's id and the triangle's place, so a cell grows the same trees every time it opens."""
    tris = np.asarray(tris, float).reshape(-1, 3, 2)
    if not len(tris):
        return np.zeros((0, 2))
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / 2
    out = []
    for k in range(len(tris)):
        centre = np.round(tris[k].mean(axis=0)).astype(np.int64)
        rng = np.random.default_rng([abs(int(seeds[k])), int(centre[0]) & 0xFFFFFFFF, int(centre[1]) & 0xFFFFFFFF])
        want = area[k] / TREE_EVERY
        n = int(want) + int(rng.random() < want - int(want))
        if not n:
            continue
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        out.append(a[k] + u[:, None] * (b[k] - a[k]) + v[:, None] * (c[k] - a[k]))
    return np.vstack(out) if out else np.zeros((0, 2))


def clear_of(xz, roads, clear: float = WILD_CLEAR) -> np.ndarray:
    """Whether each (N, 2) spot stands at least `clear` metres off every (R, 5) road's edge."""
    xz, r = np.asarray(xz, float).reshape(-1, 2), np.asarray(roads, float).reshape(-1, 5)
    if not len(r) or not len(xz):
        return np.ones(len(xz), bool)
    from contravia.clearance import _point_segment

    d = _point_segment(xz[:, None], r[None, :, 0:2], r[None, :, 2:4]) - r[None, :, 4] / 2
    return ~(d < clear).any(axis=1)


# --- species by zone (docs/superpowers/specs/2026-11-18-trees-design.html) -------------------------------------
# Nothing below is stored: a tree's species, size, turn and seasonal shift come from its position.

COAST, OLD, VEDADO, PARK, OUT = range(5)
MIX = {
    COAST: {"coco": 50, "almendro": 35, "laurel": 15},
    OLD: {"laurel": 75, "almendro": 15, "flamboyan": 10},
    VEDADO: {"laurel": 40, "flamboyan": 30, "almendro": 20, "palma": 10},
    PARK: {"laurel": 45, "flamboyan": 25, "palma": 20, "almendro": 10},
    OUT: {"palma": 45, "flamboyan": 20, "laurel": 20, "almendro": 10, "ceiba": 5},
}
COAST_REACH = 300.0  # metres: a tree this close to the sea grows coastal species
SCALE = (0.8, 1.25)
SHIFT_DAYS = 14  # a tree turns with its season up to this many days early or late


def _cumulative():
    from contravia.rules.species import SPECIES

    cum = np.zeros((len(MIX), len(SPECIES)))
    for z, mix in MIX.items():
        w = np.array([mix.get(s, 0) for s in SPECIES], float)
        cum[z] = np.cumsum(w / w.sum())
    cum[:, -1] = 1.0
    return cum


_CUM = _cumulative()


def point_seeds(xz) -> np.ndarray:
    """A 64-bit seed per (x, z), from the position rounded to the decimetre."""
    q = np.round(np.asarray(xz, float).reshape(-1, 2) * 10).astype(np.int64).view(np.uint64)
    return _mix64(q[:, 0] * np.uint64(0x9E3779B97F4A7C15) ^ q[:, 1])


def _mix64(x) -> np.ndarray:
    x = np.asarray(x, np.uint64).copy()
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def unit(seeds, salt: int) -> np.ndarray:
    """A uniform number in [0, 1) per seed, a different one for each salt."""
    with np.errstate(over="ignore"):
        s = _mix64(np.asarray(seeds, np.uint64) + np.uint64(salt) * np.uint64(0x9E3779B97F4A7C15))
    return (s >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def pick(zones, seeds) -> np.ndarray:
    """The species (an index into species.SPECIES) of each tree, drawn from its zone's MIX by its seed."""
    zones = np.asarray(zones, int)
    u = unit(seeds, 1)
    return np.minimum((u[:, None] >= _CUM[zones]).sum(axis=1), _CUM.shape[1] - 1).astype(np.int8)


def near_sea(xz, terrain) -> np.ndarray:
    """Whether the ground is sea at any of 8 points COAST_REACH metres round each (x, z)."""
    xz = np.asarray(xz, float).reshape(-1, 2)
    out = np.zeros(len(xz), bool)
    for a in np.arange(8) * np.pi / 4:
        out |= terrain.height(xz[:, 0] + COAST_REACH * np.cos(a), xz[:, 1] + COAST_REACH * np.sin(a)) < SEA
    return out


def cell_zone(c) -> int:
    """The zone of a cell's streets and green: from the commonest municipio of its buildings (OUT with none)."""
    from contravia.buildingtags import MUNI
    from contravia.walks import zone_of

    if c.building_attrs is None or not len(c.building_attrs):
        return OUT
    m = c.building_attrs[:, MUNI].astype(int)
    m = m[m >= 0]
    if not len(m):
        return OUT
    return (OLD, VEDADO, OUT)[zone_of(c.muni_names[int(np.bincount(m).argmax())])]


class Trees:
    """The trees of a cell: (N, 2) xz, and per tree its ground height, species index, scale, turn, the days
    its season is shifted by and its tone."""

    def __init__(self, xz, y, species, scale, yaw, shift, tone):
        self.xz, self.y, self.species, self.scale, self.yaw, self.shift = xz, y, species, scale, yaw, shift
        self.tone = tone  # in [0, 1): the leaves' shade, 0.85 to 1.15 times their colour

    def __len__(self):
        return len(self.xz)


def grow(xz, terrain, zone: int, arrays=()) -> Trees:
    """Species, size, turn and shift for trees at (N, 2) xz in a cell of `zone`, with the parks of the city arrays
    of each province holding the cell:
    the Prado is laurel, a paseo one species from its seed, a city park the PARK mix and its plaza's ceiba, an
    organic park the OUT mix; near the sea, the coastal mix, except on the Prado and the paseos."""
    from contravia import parks
    from contravia.rules.species import SPECIES

    xz = np.asarray(xz, float).reshape(-1, 2)
    seeds = point_seeds(xz)
    zones = np.full(len(xz), zone)
    fixed = np.full(len(xz), -1)
    for a in arrays if len(xz) else ():
        kind, pseed = parks.of_points(a, xz)
        zones[kind == parks.KINDS.index("city")] = PARK
        zones[kind == parks.KINDS.index("organic")] = OUT
        fixed[kind == parks.KINDS.index("prado")] = SPECIES.index("laurel")
        paseo = kind == parks.KINDS.index("paseo")
        fixed[paseo] = pick(np.full(paseo.sum(), PARK), _mix64(pseed[paseo].view(np.uint64)))
        fixed[parks.is_ceiba(a, xz)] = SPECIES.index("ceiba")
    free = fixed < 0
    zones[free & near_sea(xz, terrain)] = COAST
    species = np.where(free, pick(zones, seeds), fixed).astype(np.int8)
    scale = SCALE[0] + (SCALE[1] - SCALE[0]) * unit(seeds, 2)
    yaw = 2 * np.pi * unit(seeds, 3)
    shift = np.round((2 * unit(seeds, 4) - 1) * SHIFT_DAYS).astype(int)
    y = terrain.height(xz[:, 0], xz[:, 1]) if len(xz) else np.zeros(0)
    return Trees(xz, y, species, scale, yaw, shift, unit(seeds, 5))


def cell_trees(i: int, j: int, c, terrain, near, arrays=(), yards=None) -> Trees:
    """Every tree cell (i, j) draws and its car hits: its parks' trees, the trees its green grows, the (N, 2) yard
    trees of its built blocks (contravia.grass.yard_trees, already off the streets and houses) and, on land OSM says
    nothing about, the wild ones; each clear of the streets, with its species and look from grow()."""
    parts = [c.trees[off_streets(c.trees, c.street_tris)]]
    if yards is not None:
        parts.append(yards)
    has_green = c.green_tris is not None and len(c.green_tris)
    if has_green:
        g = in_green(c.green_tris, c.green_seeds)
        parts.append(g[off_streets(g, c.street_tris) & clear_of(g, near)])
    if not (len(c.buildings) or len(c.fillers) or len(c.trees) or has_green):
        spots, _, _ = wild(i, j, terrain, near)
        parts.append(spots[off_streets(spots, c.street_tris)])
    xz = np.vstack([np.asarray(p, float).reshape(-1, 2) for p in parts])
    return grow(xz, terrain, cell_zone(c), arrays)


def far_green(day) -> tuple:
    """The colour of distant tree cover on day: the leaves of the OUT mix, averaged by its weights; a bare tree
    counts as its bark."""
    from contravia.rules.species import BARK, SPECIES, season

    total, w = np.zeros(3), 0.0
    for s in SPECIES:
        weight = MIX[OUT].get(s, 0)
        look = season(s, day)
        total += weight * np.asarray(BARK[s] if look.crown == "bare" else look.leaf)
        w += weight
    return tuple(float(v) for v in total / w)


def arrays_of(city, i: int, j: int) -> list:
    """The raw arrays of each province holding cell (i, j) (a World), or the one city's (a CityData)."""
    if hasattr(city, "provinces_of"):
        return [city.province(k).a for k in city.provinces_of(i, j)]
    return [city.a]


def trunks(t) -> np.ndarray:
    """(4N, 2, 2) wall segments: a square round each trunk of the (Trees) t, of its species' collision radius."""
    from contravia.rules.species import SPECIES, TRUNK_R

    if not len(t):
        return np.zeros((0, 2, 2), np.float32)
    r = np.array([TRUNK_R[s] for s in SPECIES])[t.species] * t.scale
    corners = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], float)
    sq = t.xz[:, None, :] + corners[None] * r[:, None, None]
    return np.stack([sq, np.roll(sq, -1, axis=1)], axis=2).reshape(-1, 2, 2).astype(np.float32)
