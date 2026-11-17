"""The detail ring's geometry: what a building wears up close, over its massing.

The cells around the car get a second mesh, drawn over the massing (rules/buildings.py): cornices on the floor
lines the shader paints, balconies with iron railings, window sills, the parapet's coping and water tanks on
the roofs. Everything comes from the same facade plan as the massing, so a cornice sits exactly on the floor
line the facade's windows are laid out from: floor line k (k from 1) stands ground + (k - 1) * floor above the
base, which is where the facade coordinate v equals k. Balconies and sills go only on walls that face a
street, and tanks are seeded by the building's id, so the ring looks the same on every drive.
"""

from dataclasses import dataclass

import numpy as np

from caiman.mesh import signed_area
from caiman.rules import facades

IRON = (0.07, 0.07, 0.08)
TANKS = ((0.10, 0.12, 0.17), (0.62, 0.62, 0.60), (0.15, 0.15, 0.16))  # black plastic, cement, dark plastic
CREAM = np.array((0.93, 0.90, 0.82))
STREET = 7.0  # metres at most between a wall and a road for the wall to count as a street front
RAIL = 1.0  # a railing's height over its balcony
BARS = 4  # railing bars across the front of one balcony
BALCONY_SHARE = 0.7  # the share of a street front's bays that have a balcony


@dataclass(frozen=True)
class Style:
    floor_cornice: tuple[float, float] | None  # depth and height of the cornice on every floor line
    roof_cornice: tuple[float, float] | None  # the one under the parapet
    balconies: str | None  # "bay": one per bay on every upper floor; "band": one along the wall per floor
    sills: bool  # a sill under every upper window on a street front
    tanks: tuple[int, int]  # how many water tanks a roof carries, at least and at most
    coping: bool  # a lip along the parapet's top


STYLES = {
    "colonial": Style((0.25, 0.25), (0.6, 0.6), "bay", False, (1, 2), True),
    "eclectic": Style((0.35, 0.35), (0.6, 0.55), "bay", False, (1, 2), True),
    "vedado": Style(None, (0.35, 0.3), None, True, (1, 1), True),
    "tower": Style(None, None, "band", False, (0, 0), False),
    "prefab": Style(None, None, None, False, (1, 3), False),
    "suburban": Style(None, None, None, True, (1, 1), False),
    "warehouse": Style(None, None, None, False, (0, 0), False),
}


def floor_lines(p: facades.Plan, base: float) -> list[float]:
    """The heights of the floor lines between storeys, lowest first: where facade v is 1, 2, ... storeys - 1."""
    return [base + p.ground + (k - 1) * p.floor for k in range(1, p.storeys)]


class _Parts:
    """One building's faces, gathered so they reach the MeshBuilder in one call: a call per face would cost
    more than building the faces."""

    def __init__(self):
        self.pos, self.nrm, self.col, self.idx, self.count = [], [], [], [], 0

    def add(self, positions, normals, color, indices) -> None:
        p = np.asarray(positions, float)
        self.pos.append(p)
        self.nrm.append(np.asarray(normals, float))
        self.col.append(np.broadcast_to(np.asarray(color, float), (len(p), 3)))
        self.idx.append(np.asarray(indices) + self.count)
        self.count += len(p)

    def flush(self, mb) -> None:
        if self.pos:
            mb.add(np.vstack(self.pos), np.vstack(self.nrm), np.vstack(self.col), np.vstack(self.idx))


def _face(mb, pts, n, color) -> None:
    """A flat convex polygon facing n, as a fan; MeshBuilder turns each triangle to face n."""
    mb.add(pts, [tuple(n)] * len(pts), color, [(0, k, k + 1) for k in range(1, len(pts) - 1)])


def _cuboid(mb, c, d, o, hd, ho, y0, y1, color, back=False, caps=True) -> None:
    """A box centred at c on the ground plane, half hd along the unit vector d and ho along o, from y0 to y1.
    Its face toward -o (against the wall) is left out unless back, its top and bottom unless caps."""
    sides = [(d, hd, o, ho), (-d, hd, -o, ho), (o, ho, -d, hd)] + ([(-o, ho, d, hd)] if back else [])
    for n, hn, t, ht in sides:
        a, b = c + n * hn - t * ht, c + n * hn + t * ht
        _face(mb, [(a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]), (a[0], y1, a[1])], (n[0], 0.0, n[1]), color)
    q = [c + d * hd + o * ho, c - d * hd + o * ho, c - d * hd - o * ho, c + d * hd - o * ho]
    for y, ny in ((y1, 1.0), (y0, -1.0)) if caps else ():
        _face(mb, [(x, y, z) for x, z in q], (0.0, ny, 0.0), color)


def _cylinder(mb, c, r, y0, y1, color, sides=8) -> None:
    """An upright cylinder of radius r standing at c from y0 to y1, with its lid."""
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = c + r * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    for k in range(sides):
        a, b = ring[k], ring[(k + 1) % sides]
        n = (a + b) / 2 - c
        n = n / np.hypot(*n)
        _face(mb, [(a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]), (a[0], y1, a[1])], (n[0], 0.0, n[1]), color)
    _face(mb, [(x, y1, z) for x, z in ring], (0.0, 1.0, 0.0), color)


def _walls(ring: np.ndarray):
    """Each edge of the ring as (p0, p1, unit along, unit outward, length), whatever the ring's winding."""
    ccw = signed_area(ring) > 0
    for k in range(len(ring)):
        p0, p1 = ring[k], ring[(k + 1) % len(ring)]
        length = float(np.hypot(*(p1 - p0)))
        if length < 1e-6:
            continue
        d = (p1 - p0) / length
        out = np.array([d[1], -d[0]]) if ccw else np.array([-d[1], d[0]])
        yield p0, p1, d, out, length


def _inside(p, ring) -> bool:
    x, z = p
    inside = False
    for k in range(len(ring)):
        (x0, z0), (x1, z1) = ring[k], ring[k - 1]
        if (z0 > z) != (z1 > z) and x < x0 + (z - z0) * (x1 - x0) / (z1 - z0):
            inside = not inside
    return inside


def dress(mb, ring, osm_id: int, base: float, p: facades.Plan, roads: np.ndarray, portals=None) -> None:
    """Add one building's detail: its cornices on every wall, its balconies and sills on its street fronts, the
    parapet's coping and its roof tanks. roads are the (ax, az, bx, bz, width) rows around it; portals marks the
    walls whose ground floor is a portal, as the massing drew them."""
    ring = np.asarray(ring, float)
    s = STYLES[p.type]
    builder, mb = mb, _Parts()  # every face goes to the batch, and the batch to the builder at the end
    rng = np.random.default_rng(abs(int(osm_id)) + 11)
    trim = tuple(np.asarray(p.color) * 0.3 + CREAM * 0.7)
    street = facades.portal_walls(ring, roads, min_width=0.0) if len(roads) else np.zeros(len(ring), bool)
    lines, roof, top = floor_lines(p, base), base + p.roof, base + p.height
    for k, (p0, _p1, d, out, length) in enumerate(_walls(ring)):
        if length < 2.0:
            continue
        mid, half = p0 + d * length / 2, length / 2
        if s.floor_cornice:
            depth, h = s.floor_cornice
            for y in lines:
                _cuboid(mb, mid + out * depth / 2, d, out, half + depth, depth / 2, y, y + h, trim)
        if s.roof_cornice and p.storeys > 0:
            depth, h = s.roof_cornice
            _cuboid(mb, mid + out * depth / 2, d, out, half + depth, depth / 2, roof - h, roof, trim)
        if s.coping and p.parapet > 0:
            _cuboid(mb, mid + out * 0.04, d, out, half + 0.08, 0.12, top, top + 0.1, trim, back=True)
        if not street[k] or length < 3.0:
            continue
        bays = max(1, round(length / p.bay))
        upper = lines  # the floors above the ground floor stand on these lines
        if portals is not None and portals[k] and p.portal and upper:
            upper = upper[1:] if s.balconies == "bay" else upper  # the portal's roof carries no balcony
        if s.balconies == "bay":
            for y in upper:
                for b in range(bays):
                    c = p0 + d * length * (b + 0.5) / bays
                    if rng.random() < BALCONY_SHARE:
                        _balcony(mb, c, d, out, 0.36 * length / bays, y, trim)
        elif s.balconies == "band":
            for y in upper:
                _cuboid(mb, mid + out * 0.45, d, out, half, 0.45, y - 0.15, y, trim)
                _cuboid(mb, mid + out * 0.88, d, out, half, 0.03, y + RAIL - 0.05, y + RAIL, IRON)
        if s.sills:
            for y in upper:
                for b in range(bays):
                    c = p0 + d * length * (b + 0.5) / bays + out * 0.08
                    _cuboid(mb, c, d, out, 0.3 * length / bays, 0.08, y + 0.85, y + 0.93, trim)
    _tanks(mb, ring, osm_id, roof, s)
    mb.flush(builder)


def _balcony(mb, c, d, out, half, y, trim) -> None:
    """One balcony at floor line y in front of the bay centred at c: a slab and an iron railing along its
    front with BARS bars."""
    depth = 0.8
    _cuboid(mb, c + out * depth / 2, d, out, half, depth / 2, y - 0.15, y, trim)
    front = c + out * (depth - 0.04)
    _cuboid(mb, front, d, out, half, 0.03, y + RAIL - 0.05, y + RAIL, IRON)
    for k in range(BARS + 1):
        b = front + d * half * (2 * k / BARS - 1)
        _cuboid(mb, b, d, out, 0.025, 0.025, y, y + RAIL - 0.05, IRON, caps=False)


def _tanks(mb, ring, osm_id: int, roof: float, s: Style) -> None:
    """The roof's water tanks, seeded by the building: up to s.tanks of them, on spots inside the footprint
    and at least a metre and a half from its edges."""
    lo, hi = s.tanks
    area = abs(signed_area(ring))
    if hi == 0 or area < 30:
        return
    rng = np.random.default_rng(abs(int(osm_id)) + 7)
    n = int(rng.integers(lo, hi + 1)) if area > 120 else lo
    x0, z0 = ring.min(axis=0)
    x1, z1 = ring.max(axis=0)
    placed = []
    for _ in range(12 * max(n, 1)):
        if len(placed) == n:
            break
        c = np.array([rng.uniform(x0, x1), rng.uniform(z0, z1)])
        if not _inside(c, ring) or any(np.hypot(*(c - q)) < 2.2 for q in placed):
            continue
        a, b = ring, np.roll(ring, -1, axis=0)
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", c - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
        if np.min(np.hypot(*(a + ab * t[:, None] - c).T)) < 1.5:
            continue
        placed.append(c)
        r, h = rng.uniform(0.55, 0.85), rng.uniform(1.0, 1.5)
        legs = 0.5 if rng.random() < 0.6 else 0.0
        if legs:
            _cuboid(mb, c, np.array([1.0, 0.0]), np.array([0.0, 1.0]), r * 0.8, r * 0.8, roof, roof + legs,
                    (0.45, 0.45, 0.44), back=True)
        _cylinder(mb, c, r, roof + legs, roof + legs + h, TANKS[int(rng.integers(len(TANKS)))])
