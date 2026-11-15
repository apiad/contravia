"""Buildings in layer 0: every footprint becomes a box of random height and colour.

The randomness is seeded by the building's OSM id, so a building looks the same on every drive. Blocks
that OSM left empty get filler boxes along their edges, seeded by the block, so they are stable too.
"""

import colorsys

import numpy as np

from havana.mesh import signed_area, triangulate
from havana.terrain import seat

MIN_H, MAX_H = 6.0, 30.0
SETBACK = 3.0  # metres between a block's edge and its filler boxes
FRONT = (8.0, 15.0)
DEPTH = (10.0, 15.0)
GAP = (0.0, 3.0)


def _rng(seed) -> np.random.Generator:
    return np.random.default_rng(abs(int(seed)))


def style(osm_id: int) -> tuple[float, tuple]:
    """The height and colour of a building, drawn from its OSM id, so box and the summaries agree."""
    rng = _rng(osm_id)
    h = rng.uniform(MIN_H, MAX_H)
    return h, colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.25, 0.4), rng.uniform(0.8, 0.95))


def box(mb, ring: np.ndarray, osm_id: int, base: float = 0.0) -> bool:
    """Add walls from base up and a flat roof for a footprint; returns False when the roof could not be cut."""
    ring = np.asarray(ring, float)
    h, color = style(osm_id)
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
        n = (out[0], 0.0, out[1])
        top = base + h
        mb.add([(x0, base, z0), (x1, base, z1), (x1, top, z1), (x0, top, z0)], [n] * 4, color,
               [(0, 1, 2), (0, 2, 3)])
    try:
        tris = triangulate(ring)
    except ValueError:
        return False
    mb.add([(x, base + h, z) for x, z in ring], [(0, 1, 0)] * len(ring), color, tris)
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
