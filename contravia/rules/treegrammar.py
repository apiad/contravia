"""The tree grammar: each species' form as rules, and the meshes they make at three levels of detail.

A Form says how a species grows: a trunk (its height, girth, taper and lean, buttress roots), then levels of
branches, each branch of one level growing `count` branches of the next at a pitch from its parent, and a crown of
leaf clusters at the last level's tips, some of them an accent (flowers, red leaves). A palm grows no branches:
its form is a Palm, fronds round a crownshaft or a cluster of nuts.

skeleton() grows a form once from a seed into trunk and branch segments and cluster spots, and every level of
detail reads that one skeleton, so a tree keeps its shape as the camera nears it: LOD 0 draws every branch and
cluster, LOD 1 the first level of branches under one cluster per branch, LOD 2 the trunk under one crown.
meshes() splits a level's mesh into its parts: wood keeps its own colours, leaves and accents are grey shades
the instance's tint (the day's leaf or accent colour, rules/species.season) multiplies on the GPU. Nothing is
stored: the forms are code and the meshes are cached per process.
"""

import functools
from dataclasses import dataclass

import numpy as np

from contravia.rules.species import ACCENT, BARK, CROWNSHAFT, FIXED, LEAF, NUT, TRUNK, _Mesh

LODS = 3  # levels of detail: 0 near, 1 middle, 2 far
VARIANTS = 2  # skeletons per species, picked by each tree's seed
PARTS = ("wood", "leaf", "accent")
TRUNK_SIDES = (8, 6, 4)  # sides of a trunk's tube at each level of detail
BRANCH_SIDES = (5, 4, 3)


@dataclass(frozen=True)
class Level:
    """One level of branches: every branch of the level before grows `count` of these."""

    count: int
    length: float  # metres at the first level; times the parent's length after it
    pitch: float  # degrees from the parent's direction (from vertical for the first level)
    start: float  # how far along the parent the first one grows, 0 its base and 1 its tip
    radius: float  # times the parent's radius where it forks


@dataclass(frozen=True)
class Crown:
    """The leaf clusters at the tips of the last level's branches."""

    clusters: int  # per tip, at full detail
    size: tuple[float, float, float]  # a cluster's radii (rx, ry, rz), metres
    spread: float  # metres the clusters scatter round their tip
    accent: float = 0.0  # share of clusters that are the accent
    accent_on_top: bool = False  # accent clusters rise over the leaves (flowers on a flamboyán)


@dataclass(frozen=True)
class Palm:
    """A palm's head: fronds of leaflets round a green crownshaft (royal palm) or a cluster of nuts."""

    fronds: int
    length: float
    rise: float
    droop: float
    leaflets: int  # pairs along a frond at full detail
    crownshaft: bool


@dataclass(frozen=True)
class Form:
    """How a species grows: its trunk, its branch levels and its crown, or its palm head."""

    height: float  # metres of trunk below the first branches (the whole stem for a palm)
    radius: float  # at the base
    taper: float  # the top radius over the base's
    lean: float = 0.0  # metres the top leans off the base, along x
    buttress: int = 0  # root flares round the base (the ceiba)
    roots: int = 0  # stilt roots arching from the trunk down into the water (the mangle)
    levels: tuple[Level, ...] = ()
    crown: Crown | None = None
    palm: Palm | None = None


FORMS = {
    # Ficus: a short thick trunk forking low into a dense round crown.
    "laurel": Form(3.0, 0.5, 0.75, levels=(Level(5, 3.6, 42, 0.85, 0.55), Level(3, 0.7, 34, 0.5, 0.6)),
                   crown=Crown(5, (1.35, 1.05, 1.35), 1.7)),
    # Flamboyán: branches spread almost flat from a short trunk into a wide umbrella, flowers on top in summer.
    "flamboyan": Form(2.8, 0.32, 0.7, levels=(Level(4, 3.8, 62, 0.9, 0.6), Level(3, 0.75, 38, 0.45, 0.6)),
                      crown=Crown(4, (1.6, 0.55, 1.6), 1.6, accent=0.45, accent_on_top=True)),
    # Almendro (Terminalia): a tall stem with whorls of flat tiers, the pagoda shape; red leaves in the dry season.
    "almendro": Form(9.0, 0.3, 0.5, levels=(Level(9, 3.2, 80, 0.3, 0.45), Level(2, 0.55, 30, 0.6, 0.6)),
                     crown=Crown(2, (1.4, 0.45, 1.4), 0.5, accent=0.3)),
    # Palma real: a tall grey column, a green crownshaft and a ball of arching fronds.
    "palma": Form(18.0, 0.36, 0.62, palm=Palm(12, 5.5, 1.6, 3.0, 9, crownshaft=True)),
    # Cocotero: a slender leaning stem, nuts under a head of drooping fronds.
    "coco": Form(11.0, 0.3, 0.6, lean=2.2, palm=Palm(10, 5.0, 1.2, 3.4, 8, crownshaft=False)),
    # Ceiba: buttress roots, a thick column and a few huge horizontal limbs under a flat crown.
    # Mangle rojo: a short trunk on arching stilt roots, standing in the shallows, under a low dense crown.
    "mangle": Form(1.8, 0.16, 0.7, roots=7, levels=(Level(5, 2.0, 58, 0.6, 0.6), Level(3, 0.6, 35, 0.5, 0.6)),
                   crown=Crown(4, (1.1, 0.75, 1.1), 1.1)),
    "ceiba": Form(12.0, 1.4, 0.45, buttress=5,
                  levels=(Level(5, 6.5, 70, 0.8, 0.45), Level(3, 0.6, 30, 0.5, 0.55)),
                  crown=Crown(5, (1.9, 0.8, 1.9), 2.2)),
}


@dataclass(frozen=True)
class Skeleton:
    """A grown form: wood segments (base, tip, base radius, tip radius, level), and leaf cluster spots
    (centre, radii, role, the first-level branch they hang from)."""

    wood: list
    clusters: list
    accent_on_top: bool = False


def _perp(d, az):
    """A unit vector at right angles to d, turned az radians round it."""
    u = np.cross(d, (0.0, 0.0, 1.0) if abs(d[2]) < 0.9 else (1.0, 0.0, 0.0))
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    return np.cos(az) * u + np.sin(az) * w


@functools.cache
def skeleton(species: str, variant: int = 0) -> Skeleton:
    """Grow species' form from a seed: the same species and variant grow the same tree in every process."""
    form = FORMS[species]
    rng = np.random.default_rng([sum(map(ord, species)), variant])
    top = np.array((form.lean, form.height, 0.0))
    wood = [(np.zeros(3), top, form.radius, form.radius * form.taper, 0)]
    clusters = []
    if form.palm is not None:
        return Skeleton(wood, clusters)
    # each parent: (base, tip, radius at base, first-level branch index)
    parents = [(np.zeros(3), top, form.radius, -1)]
    for depth, lv in enumerate(form.levels):
        grown = []
        for p, (base, tip, r, root) in enumerate(parents):
            axis = tip - base
            length = np.linalg.norm(axis)
            d = axis / length
            turn = rng.uniform(0, 2 * np.pi)
            for k in range(lv.count):
                f = lv.start + (1 - lv.start) * (k + 0.5) / lv.count if lv.count > 1 else lv.start
                at = base + axis * min(f, 1.0)
                az = turn + 2 * np.pi * k / lv.count + rng.uniform(-0.3, 0.3)
                pitch = np.radians(lv.pitch + rng.uniform(-8, 8))
                if depth == 0:
                    dirn = np.array((np.sin(pitch) * np.cos(az), np.cos(pitch), np.sin(pitch) * np.sin(az)))
                else:
                    dirn = np.cos(pitch) * d + np.sin(pitch) * _perp(d, az)
                    dirn[1] = max(dirn[1], -0.15)  # twigs never hang far below their branch
                    dirn /= np.linalg.norm(dirn)
                size = (lv.length if depth == 0 else lv.length * length) * rng.uniform(0.85, 1.15)
                rb = r * lv.radius
                end = at + dirn * size
                wood.append((at, end, rb, rb * 0.5, depth + 1))
                grown.append((at, end, rb, len(grown) if depth == 0 else root))
        parents = grown
    cr = form.crown
    for base, tip, _, root in parents:
        for _ in range(cr.clusters):
            c = tip + rng.normal(0, cr.spread / 2, 3) * (1, 0.5, 1)
            radii = np.array(cr.size) * rng.uniform(0.8, 1.2)
            clusters.append((c, radii, LEAF, root))
    if cr.accent:
        n = round(cr.accent * len(clusters))
        pick = rng.choice(len(clusters), n, replace=False)
        for k in pick:
            c, radii, _, root = clusters[k]
            if cr.accent_on_top:
                c = c + (0, radii[1] * 0.7, 0)
            clusters[k] = (c, radii, ACCENT, root)
    return Skeleton(wood, clusters, cr.accent_on_top)


def _merge(spots, role):
    """One cluster covering the spots (centre, radii, role, root) given: their mean, as wide as they reach."""
    c = np.mean([s[0] for s in spots], axis=0)
    reach = np.max([np.abs(s[0] - c) + s[1] for s in spots], axis=0)
    return c, reach * 0.9, role


def _crown(sk: Skeleton, lod: int, crown: str) -> list:
    """The clusters (centre, radii, role) drawn at a level of detail with a full, sparse or bare crown."""
    spots = sk.clusters
    if crown == "bare" or not spots:
        return []
    if crown == "sparse":
        spots = spots[::2]
    if lod == 0:
        return [(c, r, role) for c, r, role, _ in spots]
    if lod == 1:  # one cluster per first-level branch and role
        out = []
        for root in sorted({s[3] for s in spots}):
            for role in (LEAF, ACCENT):
                group = [s for s in spots if s[3] == root and s[2] == role]
                if group:
                    out.append(_merge(group, role))
        return out
    leaf = [s for s in spots if s[2] == LEAF] or spots
    top = [s for s in spots if s[2] == ACCENT]
    far = [_merge(leaf, LEAF)]
    if top and sk.accent_on_top:  # the flowers read from afar: a flat cap over the crown
        c, r, _ = _merge(top, ACCENT)
        far.append((c + (0, r[1] * 0.5, 0), r * (0.95, 0.6, 0.95), ACCENT))
    return far


def _frond(m: _Mesh, top, yaw, palm: Palm, up: bool, lod: int):
    """One frond from top: a spine rising then drooping, with pairs of leaflets at full detail."""
    rise, droop = (palm.rise, palm.droop) if up else (palm.rise * 0.4, palm.droop * 1.25)
    if lod > 0:
        m.frond(top, yaw, palm.length, rise, droop, 1.0, LEAF)
        return
    d = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    side = np.array([-d[2], 0.0, d[0]])
    n = palm.leaflets

    def spine(t):  # a parabola through rise at mid-length and rise - droop at the tip
        return top + d * palm.length * t + (0.0, (3 * rise + droop) * t - 2 * (rise + droop) * t * t, 0.0)

    for k in range(n):
        t0, t1 = k / n, (k + 1) / n
        a, b = spine(t0), spine(t1)
        w = 0.9 * np.sin(np.pi * (t0 + t1) / 2) + 0.15  # leaflets longest mid-frond
        for s in (1, -1):
            tip = (a + b) / 2 + side * s * w + (0.0, -0.35 * w, 0.0)
            m.two_sided(a, b, tip, LEAF)


def _palm_head(m: _Mesh, form: Form, lod: int, crown: str, species: str):
    palm = form.palm
    top = np.array((form.lean, form.height, 0.0))
    if palm.crownshaft:
        m.prism(top, top + (0, 2.0, 0), form.radius * form.taper * 1.05, form.radius * form.taper * 0.85,
                TRUNK_SIDES[lod], FIXED, CROWNSHAFT)
        top = top + (0, 2.0, 0)
    else:
        m.blob(top + (0.0, -0.3, 0.0), (0.55, 0.35, 0.55), FIXED, NUT)  # the cluster of nuts under the fronds
    n = {"full": palm.fronds, "sparse": palm.fronds // 2, "bare": 0}[crown]
    if lod == 2:
        n = n * 2 // 3
    for k in range(n):
        _frond(m, top, 2 * np.pi * k / max(n, 1) + 0.3, palm, k % 2 == 0, lod)


def _trunk(m: _Mesh, form: Form, lod: int, species: str):
    """The trunk: one tube for a straight tree, three bending sections for a leaning one, buttresses at its foot."""
    sides = TRUNK_SIDES[lod]
    r0, r1 = form.radius, form.radius * form.taper
    if form.lean:
        pts = [np.array((form.lean * (k / 3) ** 2, form.height * k / 3, 0.0)) for k in range(4)]
        for k in range(3):
            m.prism(pts[k], pts[k + 1], r0 + (r1 - r0) * k / 3, r0 + (r1 - r0) * (k + 1) / 3, sides, TRUNK,
                    BARK[species], level=True)
    elif lod == 2:
        m.prism((0, 0, 0), (0, form.height, 0), r0, r1, sides, TRUNK, BARK[species])
    else:
        mid = np.array((0.0, form.height * 0.45, 0.0))
        rm = r0 + (r1 - r0) * 0.45
        m.prism((0, 0, 0), mid, r0 * (1.25 if form.buttress else 1.0), rm, sides, TRUNK, BARK[species])
        m.prism(mid, (0, form.height, 0), rm, r1, sides, TRUNK, BARK[species])
    if form.buttress and lod < 2:
        for k in range(form.buttress):
            a = 2 * np.pi * k / form.buttress + 0.4
            out = np.array((np.cos(a), 0.0, np.sin(a)))
            foot, up = out * r0 * 2.4, out * r0 * 0.6 + (0, 2.6, 0)
            side = np.array((-out[2], 0.0, out[0])) * 0.18
            m.two_sided(foot, up + side, foot + side * 2, TRUNK, BARK[species])
            m.two_sided(foot, up - side, up + side, TRUNK, BARK[species])


def _stilts(m: _Mesh, form: Form, lod: int, species: str):
    """A mangle's stilt roots: each leaves the trunk part way up and arches out and down to the ground, in two
    bends near and one far."""
    count = form.roots if lod < 2 else 3
    sides = 3
    for k in range(count):
        a = 2 * np.pi * k / count + 0.3 * np.sin(3.1 * k)
        out = np.array((np.cos(a), 0.0, np.sin(a)))
        reach = 1.0 + 0.35 * ((k * 7) % 3)
        start = np.array((0.0, form.height * (0.55 + 0.1 * (k % 2)), 0.0))
        knee = start + out * reach * 0.55 + (0.0, 0.25, 0.0)
        foot = out * reach + (0.0, -0.3, 0.0)
        r = form.radius * 0.35
        if lod == 0:
            m.prism(start, knee, r, r * 0.9, sides, TRUNK, BARK[species])
            m.prism(knee, foot, r * 0.9, r * 0.8, sides, TRUNK, BARK[species])
        else:
            m.prism(start, foot, r, r * 0.8, sides, TRUNK, BARK[species])


def build(species: str, lod: int, crown: str = "full", variant: int = 0) -> _Mesh:
    """The mesh of a species at a level of detail, with a full, sparse or bare crown, standing at the origin."""
    form = FORMS[species]
    sk = skeleton(species, variant)
    m = _Mesh()
    _trunk(m, form, lod, species)
    if form.roots:
        _stilts(m, form, lod, species)
    if form.palm is not None:
        _palm_head(m, form, lod, crown, species)
        return m
    bark = BARK[species]
    for base, tip, rb, rt, level in sk.wood[1:]:
        if level == 1 and lod < 2 or level >= 2 and lod == 0:
            m.prism(base, tip, rb, rt, BRANCH_SIDES[lod], TRUNK, bark)
    for c, radii, role in _crown(sk, lod, crown):
        m.blob(c, radii, role)
    return m


@functools.cache
def meshes(species: str, lod: int, crown: str = "full", variant: int = 0) -> dict:
    """The geometry dicts (positions, normals, colours, indices) of each part of build(...) that has triangles:
    "wood" in its own colours, "leaf" and "accent" in grey shades for the instance's tint to colour."""
    t = build(species, lod, crown, variant).done()
    out = {}
    for part, roles in (("wood", (TRUNK, FIXED)), ("leaf", (LEAF,)), ("accent", (ACCENT,))):
        tri = np.flatnonzero(np.isin(t.role[t.triangles[:, 0]], roles))
        if not len(tri):
            continue
        v = t.triangles[tri].ravel()
        col = t.rgb[v] * t.shade[v, None] if part == "wood" else np.repeat(t.shade[v, None], 3, axis=1)
        out[part] = {"positions": t.positions[v].astype(np.float32), "normals": t.normals[v].astype(np.float32),
                     "colors": np.clip(col, 0, 1).astype(np.float32),
                     "indices": np.arange(len(v), dtype=np.uint32)}
    return out


def triangles(species: str, lod: int, crown: str = "full", variant: int = 0) -> int:
    """How many triangles a tree of species draws at a level of detail."""
    return sum(len(g["indices"]) // 3 for g in meshes(species, lod, crown, variant).values())
