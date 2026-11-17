"""Buildings at the cell's level of detail: every footprint becomes its facade plan's massing.

The plan (rules/facades.py) is seeded by the building's OSM id, so a building looks the same on every drive.
Its walls rise in three bands, the ground floor, the floors above and the parapet; the first two carry the
facade coordinates manifoldx draws windows and doors from, and the parapet shows its inner face over a flat
roof. Blocks that OSM left empty get filler boxes along their edges, seeded by the block, so they are
stable too.
"""

import numpy as np

from caiman.mesh import signed_area, triangulate
from caiman.rules import facades
from caiman.terrain import seat

GROUND_SHADE = 0.82  # the shade of a wall's inner face, under the parapet
SETBACK = 3.0  # metres between a block's edge and its filler boxes
FRONT = (8.0, 15.0)
DEPTH = (10.0, 15.0)
GAP = (0.0, 3.0)


def _rng(seed) -> np.random.Generator:
    return np.random.default_rng(abs(int(seed)))


def _portal(mb, p0, p1, out, base, ground, bays, color, kind, seed) -> None:
    """The ground floor of one wall as a portal: columns on the wall's line at every bay, a ceiling, and the
    shop fronts standing PORTAL_DEPTH back."""
    inward = -np.asarray(out)
    q0, q1 = p0 + inward * facades.PORTAL_DEPTH, p1 + inward * facades.PORTAL_DEPTH
    n, down = (out[0], 0.0, out[1]), (0.0, -1.0, 0.0)
    mb.add([(q0[0], base, q0[1]), (q1[0], base, q1[1]), (q1[0], ground, q1[1]), (q0[0], ground, q0[1])], [n] * 4,
           color, [(0, 1, 2), (0, 2, 3)], [(0, 0, kind, seed), (bays, 0, kind, seed), (bays, 1, kind, seed),
                                           (0, 1, kind, seed)])
    soffit = tuple(min(1.0, c * 1.1) for c in color)
    mb.add([(p0[0], ground, p0[1]), (p1[0], ground, p1[1]), (q1[0], ground, q1[1]), (q0[0], ground, q0[1])],
           [down] * 4, soffit, [(0, 1, 2), (0, 2, 3)])
    d = (p1 - p0) / max(np.hypot(*(p1 - p0)), 1e-9)
    s = facades.COLUMN / 2
    for k in range(bays + 1):
        c = p0 + (p1 - p0) * (k / bays) + inward * s * 1.2
        c = c - d * s * (1 if k == bays else 0) + d * s * (1 if k == 0 else 0)
        corners = [c + d * s + inward * s, c + d * s - inward * s, c - d * s - inward * s, c - d * s + inward * s]
        for m in range(4):
            u, v = corners[m], corners[(m + 1) % 4]
            e = v - u
            nn = np.array([e[1], -e[0]]) / max(np.hypot(*e), 1e-9)
            if np.dot(nn, (u + v) / 2 - c) < 0:
                nn = -nn
            mb.add([(u[0], base, u[1]), (v[0], base, v[1]), (v[0], ground, v[1]), (u[0], ground, u[1])],
                   [(nn[0], 0.0, nn[1])] * 4, soffit, [(0, 1, 2), (0, 2, 3)])


def box(mb, ring: np.ndarray, osm_id: int, base: float = 0.0, plan: facades.Plan | None = None,
        portals=None) -> bool:
    """Add a footprint's massing from base up: walls to the parapet, the parapet's inner face and a flat
    roof. plan is the building's facade plan (by default one from its id alone); portals marks the edges
    whose ground floor is a portal (facades.portal_walls). Returns False when the roof could not be cut."""
    ring = np.asarray(ring, float)
    p = plan or facades.plan(osm_id, area=abs(signed_area(ring)))
    color, low = p.color, tuple(c * GROUND_SHADE for c in p.color)
    ground, roof, top = base + min(p.ground, p.roof), base + p.roof, base + p.height
    kind, seed = facades.SHADER_TYPE[p.type], (abs(int(osm_id)) * 0.6180339887) % 1.0
    ccw = signed_area(ring) > 0
    for k in range(len(ring)):
        (x0, z0), (x1, z1) = ring[k], ring[(k + 1) % len(ring)]
        d = np.array([x1 - x0, z1 - z0])
        length = np.hypot(*d)
        if length < 1e-6:
            continue
        out = np.array([d[1], -d[0]]) / length  # the right-hand side is outside a counter-clockwise ring
        if not ccw:
            out = -out
        n, inner = (out[0], 0.0, out[1]), (-out[0], 0.0, -out[1])
        bays = max(1, round(length / p.bay))  # whole bays, so windows sit centred on the wall
        if portals is not None and portals[k] and p.portal and roof > ground:
            _portal(mb, np.array([x0, z0]), np.array([x1, z1]), out, base, ground, bays, color, kind, seed)
        else:
            mb.add([(x0, base, z0), (x1, base, z1), (x1, ground, z1), (x0, ground, z0)], [n] * 4, color,
                   [(0, 1, 2), (0, 2, 3)], [(0, 0, kind, seed), (bays, 0, kind, seed), (bays, 1, kind, seed),
                                            (0, 1, kind, seed)])
        if roof > ground:
            up = p.storeys
            mb.add([(x0, ground, z0), (x1, ground, z1), (x1, roof, z1), (x0, roof, z0)], [n] * 4, color,
                   [(0, 1, 2), (0, 2, 3)], [(0, 1, kind, seed), (bays, 1, kind, seed), (bays, up, kind, seed),
                                            (0, up, kind, seed)])
        mb.add([(x0, roof, z0), (x1, roof, z1), (x1, top, z1), (x0, top, z0)], [n] * 4, color,
               [(0, 1, 2), (0, 2, 3)])
        mb.add([(x0, roof, z0), (x1, roof, z1), (x1, top, z1), (x0, top, z0)], [inner] * 4, low,
               [(0, 2, 1), (0, 3, 2)])
    try:
        tris = triangulate(ring)
    except ValueError:
        return False
    mb.add([(x, roof, z) for x, z in ring], [(0, 1, 0)] * len(ring), p.roof_color, tris)
    return True


def _inside(p, ring) -> bool:
    """Even-odd rule: is point p inside the ring?"""
    x, z = p
    inside = False
    for k in range(len(ring)):
        (x0, z0), (x1, z1) = ring[k], ring[k - 1]
        if (z0 > z) != (z1 > z) and x < x0 + (z - z0) * (x1 - x0) / (z1 - z0):
            inside = not inside
    return inside


def _edge_distance(p, ring) -> float:
    """The shortest distance from p to any edge of the ring."""
    a, b = ring, np.roll(ring, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    return float(np.min(np.hypot(*(a + ab * t[:, None] - p).T)))


def _overlap(r, s) -> bool:
    """Separating axis test for two convex quads."""
    for poly in (r, s):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            axis = np.array([-e[1], e[0]])
            pr, ps = r @ axis, s @ axis
            if pr.max() <= ps.min() + 1e-6 or ps.max() <= pr.min() + 1e-6:
                return False
    return True


def filler_rings(block: np.ndarray, seed: int) -> list[np.ndarray]:
    """Footprints of filler boxes along the edges of a block, set back SETBACK metres."""
    block = np.asarray(block, float)
    if signed_area(block) < 0:
        block = block[::-1]
    rng = _rng(seed)
    placed = []
    for k in range(len(block)):
        a, b = block[k], block[(k + 1) % len(block)]
        length = np.hypot(*(b - a))
        if length < FRONT[0] + 2 * SETBACK:
            continue
        d = (b - a) / length
        inward = np.array([-d[1], d[0]])  # left of the edge is inside a counter-clockwise ring
        t = SETBACK
        while True:
            front, depth = rng.uniform(*FRONT), rng.uniform(*DEPTH)
            if t + front > length - SETBACK:
                break
            p0 = a + d * t + inward * SETBACK
            quad = np.array([p0, p0 + d * front, p0 + d * front + inward * depth, p0 + inward * depth])
            t += front + rng.uniform(*GAP)
            if not all(_inside(c, block) and _edge_distance(c, block) >= SETBACK - 1e-6 for c in quad):
                continue
            if any(_overlap(quad, q) for q in placed):
                continue
            placed.append(quad)
    return placed


def filler(mb, block: np.ndarray, seed: int, terrain=None) -> None:
    """Add the filler boxes of a block as buildings, each seeded from the block's seed and on the terrain."""
    for k, ring in enumerate(filler_rings(block, seed)):
        box(mb, ring, osm_id=abs(int(seed)) * 131 + k, base=seat(ring, terrain) if terrain else 0.0)
