"""Props: the meshes of a park's motifs, from boxes, prisms and extruded outlines.

Each function adds one motif to a Props: its triangles (wound so their normal faces out, as the cells take a
normal from the winding), each triangle's paint (contravia.structures' PAINTS) and the wall segments the car
hits, as struct_walls rows (ax, az, bx, bz, bottom, top). Positions are world (x, z) metres, y the height the
motif stands on, a heading the radians from +x its front faces. The measures are those of
docs/superpowers/specs/2026-11-18-parks-design.html, section 05.
"""

import numpy as np
import shapely

from contravia.structures import (
    BLUE,
    BRONZE,
    IRON,
    LAMPS,
    RED,
    SAND,
    STONE,
    THATCH,
    TILE,
    WATER,
    WHITE,
    WOOD,
    YELLOW,
)

HIT = 0.4  # metres: a motif taller than this is a wall the car hits


class Props:
    """Triangles, paints and walls of the motifs added so far."""

    def __init__(self):
        self.tris, self.paint, self.walls = [], [], []

    def tri(self, a, b, c, want, paint):
        t = np.array([a, b, c], float)
        if np.cross(t[1] - t[0], t[2] - t[0]) @ np.asarray(want, float) < 0:
            t = t[[0, 2, 1]]
        self.tris.append(t)
        self.paint.append(paint)

    def quad(self, a, b, c, d, want, paint):
        self.tri(a, b, c, want, paint)
        self.tri(a, c, d, want, paint)

    def wall_ring(self, pts, y0, y1):
        """The closed outline pts (N, 2) as walls from y0 to y1."""
        p = np.asarray(pts, float)
        q = np.roll(p, -1, axis=0)
        self.walls += [(*a, *b, y0, y1) for a, b in zip(p, q, strict=True)]

    def extrude(self, pts, y0, y1, paint, top=None, hit=True):
        """A solid from the outline pts (N, 2), anticlockwise or not, from y0 to y1, its top in `top` paint."""
        p = np.asarray(pts, float)
        c = p.mean(axis=0)
        for a, b in zip(p, np.roll(p, -1, axis=0), strict=True):
            m = (a + b) / 2 - c
            self.quad((a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]), (a[0], y1, a[1]), (m[0], 0, m[1]), paint)
        self.flat(p, y1, paint if top is None else top)
        if hit and y1 - y0 > HIT:
            self.wall_ring(p, y0, y1)

    def flat(self, pts, y, paint, down=False):
        """The polygon pts (N, 2) as a flat face at height y, facing up (or down)."""
        poly = shapely.Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        t = shapely.constrained_delaunay_triangles(poly)
        for tr in shapely.get_coordinates(t).reshape(-1, 4, 2)[:, :3]:
            self.tri(*((x, y, z) for x, z in tr), (0, -1 if down else 1, 0), paint)

    def box(self, c, hx, hz, y0, y1, heading, paint, top=None, hit=True):
        """A box centred at c, hx along the heading and hz across it, from y0 to y1."""
        self.extrude(_rect(c, hx, hz, heading), y0, y1, paint, top, hit)

    def beam(self, a, b, half, y0, y1, paint, hit=False):
        """A box along the segment a-b, half wide."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        self.box((a + b) / 2, np.hypot(*d) / 2, half, y0, y1, np.arctan2(d[1], d[0]), paint, hit=hit)

    def prism(self, c, r, n, y0, y1, paint, top=None, r_top=None, cap=True, inward=False, turn=0.0):
        """An n-sided prism (or a frustum, with r_top) round c, from y0 to y1."""
        a = turn + np.arange(n) * 2 * np.pi / n
        c = np.asarray(c, float)
        lo = c + r * np.column_stack([np.cos(a), np.sin(a)])
        hi = c + (r if r_top is None else r_top) * np.column_stack([np.cos(a), np.sin(a)])
        for k in range(n):
            j = (k + 1) % n
            m = (lo[k] + lo[j]) / 2 - c
            want = (-m[0], 0, -m[1]) if inward else (m[0], (r - (r if r_top is None else r_top)) * np.hypot(*m) / max(y1 - y0, 1e-9), m[1])
            self.quad((*_xz(lo[k], y0),), (*_xz(lo[j], y0),), (*_xz(hi[j], y1),), (*_xz(hi[k], y1),), want, paint)
        if cap and (r_top is None or r_top > 0.01):
            self.flat(hi, y1, paint if top is None else top)
        return lo


def _xz(p, y):
    return (p[0], y, p[1])


def _rect(c, hx, hz, heading):
    u = np.array([np.cos(heading), np.sin(heading)])
    v = np.array([-u[1], u[0]])
    c = np.asarray(c, float)
    return np.array([c - u * hx - v * hz, c + u * hx - v * hz, c + u * hx + v * hz, c - u * hx + v * hz])


def _at(c, heading, du, dv):
    u = np.array([np.cos(heading), np.sin(heading)])
    return np.asarray(c, float) + u * du + np.array([-u[1], u[0]]) * dv


def glorieta(p: Props, c, r, heading, y):
    """An octagonal bandstand r metres to its columns: a stone floor 0.6 m up with steps at two entrances, eight
    columns, an iron rail between them but at the entrances, and a tiled roof to 6 m with a small lantern."""
    turn = heading + np.pi / 8
    floor = p.prism(c, r + 1, 8, y, y + 0.6, STONE, turn=turn)
    p.wall_ring(floor, y, y + 0.6)
    for s in (0, np.pi):
        for k, h in enumerate((0.2, 0.4)):
            p.box(_at(c, heading + s, r + 1.3 + 0.35 * (1 - k), 0), 0.2, 1.0, y, y + h, heading + s, STONE, hit=False)
    a = turn + np.arange(8) * np.pi / 4
    cols = np.asarray(c) + r * np.column_stack([np.cos(a), np.sin(a)])
    for q in cols:
        p.box(q, 0.13, 0.13, y + 0.6, y + 4.0, 0.0, WHITE, hit=False)
    entrances = {0, 4}  # the sides facing the heading and its back
    for k in range(8):
        if k in entrances:
            continue
        p.beam(cols[k], cols[(k + 1) % 8], 0.03, y + 1.4, y + 1.5, IRON)
        p.beam(cols[k], cols[(k + 1) % 8], 0.015, y + 0.7, y + 1.4, IRON)
    p.prism(c, r + 0.6, 8, y + 4.0, y + 4.25, WHITE, turn=turn)
    p.prism(c, r + 0.6, 8, y + 4.25, y + 6.0, TILE, r_top=0.5, turn=turn)
    p.prism(c, 0.5, 8, y + 6.0, y + 6.5, WHITE, top=TILE, turn=turn)


def fountain(p: Props, c, r, y):
    """A round basin r metres across inside, 0.5 m high, its water 0.1 m under the rim, a 2 m column with a plate."""
    outer = p.prism(c, r + 0.3, 20, y, y + 0.5, STONE, cap=False)
    p.prism(c, r, 20, y + 0.1, y + 0.5, STONE, cap=False, inward=True)
    a = np.arange(20) * 2 * np.pi / 20
    inner = np.asarray(c) + r * np.column_stack([np.cos(a), np.sin(a)])
    for k in range(20):
        j = (k + 1) % 20
        p.quad(_xz(outer[k], y + 0.5), _xz(outer[j], y + 0.5), _xz(inner[j], y + 0.5), _xz(inner[k], y + 0.5), (0, 1, 0), STONE)
    p.flat(inner, y + 0.4, WATER)
    p.wall_ring(outer, y, y + 0.5)
    p.prism(c, 0.3, 8, y + 0.4, y + 2.0, STONE)
    p.prism(c, 0.35 * r, 12, y + 1.5, y + 1.65, STONE, top=WATER)


def bust(p: Props, c, heading, y):
    """A white pedestal 1.2 x 1.2 x 1.6 m with a bronze bust on it."""
    p.box(c, 0.6, 0.6, y, y + 1.6, heading, WHITE)
    p.box(c, 0.7, 0.7, y + 1.6, y + 1.7, heading, WHITE, hit=False)
    p.box(c, 0.15, 0.32, y + 1.7, y + 2.0, heading, BRONZE, hit=False)
    p.box(c, 0.13, 0.12, y + 2.0, y + 2.35, heading, BRONZE, hit=False)


def bench(p: Props, c, facing, y):
    """A 1.8 m bench facing `facing`: a wooden seat at 0.45 m and a backrest on two concrete legs."""
    along = facing + np.pi / 2
    p.box(c, 0.9, 0.22, y + 0.4, y + 0.47, along, WOOD, hit=False)
    p.box(_at(c, facing, -0.22, 0), 0.9, 0.03, y + 0.5, y + 0.85, along, WOOD, hit=False)
    for s in (-0.75, 0.75):
        p.box(_at(c, along, s, 0), 0.05, 0.24, y, y + 0.4, along, STONE, hit=False)
    p.wall_ring(_rect(c, 0.9, 0.25, along), y, y + 0.85)


def domino(p: Props, c, heading, y):
    """A square concrete table 0.9 m across at 0.75 m on one leg, with four stools."""
    p.box(c, 0.12, 0.12, y, y + 0.7, heading, STONE, hit=False)
    p.box(c, 0.45, 0.45, y + 0.7, y + 0.78, heading, STONE, hit=False)
    p.wall_ring(_rect(c, 0.45, 0.45, heading), y, y + 0.78)
    for k in range(4):
        p.prism(_at(c, heading + k * np.pi / 2, 0.8, 0), 0.18, 6, y, y + 0.45, STONE)


def playground(p: Props, c, heading, y):
    """A 15 x 12 m sand pit with a low kerb: swings, a slide and a cachumbambé (seesaw)."""
    p.flat(_rect(c, 7.5, 6.0, heading), y + 0.12, SAND)  # over the ground mesh, which runs between its 10 m vertices
    for du, dv, hx, hz in ((0, -6, 7.6, 0.1), (0, 6, 7.6, 0.1), (-7.5, 0, 0.1, 6.0), (7.5, 0, 0.1, 6.0)):
        p.box(_at(c, heading, du, dv), hx, hz, y, y + 0.15, heading, STONE, hit=False)
    a, b = _at(c, heading, -5.5, -3), _at(c, heading, -1.5, -3)  # swings
    for q in (a, b):
        p.box(q, 0.08, 0.08, y, y + 2.3, heading, RED)
    p.beam(a, b, 0.06, y + 2.2, y + 2.32, RED)
    for du in (-4.5, -2.5):
        q = _at(c, heading, du, -3)
        p.box(q, 0.3, 0.12, y + 0.42, y + 0.48, heading, YELLOW, hit=False)
        for s in (-0.25, 0.25):
            p.box(_at(q, heading, s, 0), 0.01, 0.01, y + 0.48, y + 2.2, heading, IRON, hit=False)
    top, foot = _at(c, heading, 3, -4), _at(c, heading, 3, 0)  # slide
    p.box(top, 0.4, 0.3, y, y + 2.0, heading, BLUE)
    u = np.array([-np.sin(heading), np.cos(heading)]) * 0.35  # across the slide
    for s in (1, -1):
        p.quad(_xz(top - u, y + 2.0), _xz(top + u, y + 2.0), _xz(foot + u, y + 0.3), _xz(foot - u, y + 0.3),
               (0, s, 0), RED)
    p.wall_ring(_rect((top + foot) / 2, 0.4, 2.0, heading + np.pi / 2), y, y + 2.0)
    q = _at(c, heading, -3, 3)  # cachumbambé
    p.box(q, 0.15, 0.15, y, y + 0.4, heading, IRON)
    p.box(q, 1.8, 0.15, y + 0.4, y + 0.48, heading, YELLOW, hit=False)
    for s in (-1.5, 1.5):
        p.box(_at(q, heading, s, 0), 0.03, 0.12, y + 0.48, y + 0.8, heading, IRON, hit=False)


def ranchon(p: Props, c, heading, y):
    """A 6 x 4 m thatched shelter on four posts, its ridge at 4 m along the heading, its eaves at 2.4 m."""
    for du in (-2.7, 2.7):
        for dv in (-1.7, 1.7):
            p.box(_at(c, heading, du, dv), 0.1, 0.1, y, y + 2.6, heading, WOOD)
    e = [_at(c, heading, du, dv) for du, dv in ((-3.4, -2.4), (3.4, -2.4), (3.4, 2.4), (-3.4, 2.4))]
    r0, r1 = _at(c, heading, -3.4, 0), _at(c, heading, 3.4, 0)
    for a, b in ((e[0], e[1]), (e[3], e[2])):  # the two slopes, seen from above and from under the roof
        for up in (1, -1):
            p.quad(_xz(a, y + 2.4), _xz(b, y + 2.4), _xz(r1, y + 4.0), _xz(r0, y + 4.0), (0, up, 0), THATCH)
    for a, b, r in ((e[0], e[3], r0), (e[1], e[2], r1)):  # the gables, both faces
        m = (a + b) / 2 - np.asarray(c)
        for s in (1, -1):
            p.tri(_xz(a, y + 2.4), _xz(b, y + 2.4), _xz(r, y + 4.0), (s * m[0], 0, s * m[1]), THATCH)


def picnic(p: Props, c, heading, y):
    """A wooden picnic table 1.8 x 0.8 m with a bench on each long side."""
    p.box(c, 0.9, 0.4, y + 0.72, y + 0.78, heading, WOOD, hit=False)
    for du in (-0.7, 0.7):
        p.box(_at(c, heading, du, 0), 0.05, 0.35, y, y + 0.72, heading, WOOD, hit=False)
    for dv in (-0.65, 0.65):
        p.box(_at(c, heading, 0, dv), 0.9, 0.14, y + 0.42, y + 0.46, heading, WOOD, hit=False)
    p.wall_ring(_rect(c, 0.95, 0.85, heading), y, y + 0.78)


def lamp(p: Props, c, y):
    """A 4.5 m park lamp: a dark pole with a lantern."""
    p.box(c, 0.07, 0.07, y, y + 4.3, 0.0, IRON, hit=False)
    p.prism(c, 0.25, 6, y + 4.3, y + 4.75, LAMPS, top=IRON)
    p.wall_ring(_rect(c, 0.12, 0.12, 0.0), y, y + 4.3)


def lake(p: Props, water, yw):
    """A lake's water at yw inside a 0.4 m wide stone rim standing 0.4 m over it, down to 0.5 m under the water
    outside. The rim is a wall."""
    p.flat(np.asarray(water.exterior.coords)[:-1], yw, WATER)
    rim = water.buffer(0.4, join_style="mitre", mitre_limit=2.0)
    o, i = np.asarray(rim.exterior.coords)[:-1], np.asarray(water.exterior.coords)[:-1]
    top = yw + 0.4
    t = shapely.constrained_delaunay_triangles(rim.difference(water))
    for tr in shapely.get_coordinates(t).reshape(-1, 4, 2)[:, :3]:
        p.tri(*((x, top, z) for x, z in tr), (0, 1, 0), STONE)
    for ring, out, y0 in ((o, 1, yw - 0.5), (i, -1, yw - 0.1)):
        c = ring.mean(axis=0)
        for a, b in zip(ring, np.roll(ring, -1, axis=0), strict=True):
            m = (a + b) / 2 - c
            n = np.array([b[1] - a[1], a[0] - b[0]])  # one of the edge's normals; take the one away from (or to) the middle
            if (n @ m) * out < 0:
                n = -n
            p.quad(_xz(a, y0), _xz(b, y0), _xz(b, top), _xz(a, top), (n[0], 0, n[1]), STONE)
    p.wall_ring(o, yw - 0.5, top)


def footbridge(p: Props, a, b, y_a, y_b):
    """A 2.5 m wooden footbridge from a to b arching 1 m, with rails, and two bollards at each end that keep cars off."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.hypot(*d))
    u = d / max(L, 1e-9)
    v = np.array([-u[1], u[0]])
    n = max(4, int(L / 2))
    t = np.linspace(0, 1, n + 1)
    pts = a + t[:, None] * d
    ys = y_a + (y_b - y_a) * t + 1.0 * np.sin(np.pi * t)
    for k in range(n):
        q0, q1 = pts[k], pts[k + 1]
        lo = min(ys[k], ys[k + 1])
        for s in (1, -1):
            p.quad(_xz(q0 + v * 1.25, ys[k]), _xz(q1 + v * 1.25, ys[k + 1]), _xz(q1 - v * 1.25, ys[k + 1]),
                   _xz(q0 - v * 1.25, ys[k]), (0, s, 0), WOOD)
        for side in (1.2, -1.2):
            p.beam(q0 + v * side, q1 + v * side, 0.05, lo + 0.9, lo + 1.0, WOOD)
            p.box(q0 + v * side, 0.05, 0.05, ys[k], ys[k] + 1.0, 0.0, WOOD, hit=False)
        p.walls += [(*(q0 + v * 1.2), *(q1 + v * 1.2), lo - 1, lo + 1.0), (*(q0 - v * 1.2), *(q1 - v * 1.2), lo - 1, lo + 1.0)]
    for end, y in ((a, y_a), (b, y_b)):
        for s in (-0.6, 0.6):
            q = end + v * s - u * 0.3 * (1 if end is a else -1)
            p.prism(q, 0.12, 8, y, y + 0.9, STONE)
            p.wall_ring(_rect(q, 0.12, 0.12, 0.0), y, y + 0.9)


def pier(p: Props, a, b, yw):
    """A 2 m wooden pier from the shore a to its tip b, 0.3 m over the water, on posts."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    p.beam(a, b, 1.0, yw + 0.25, yw + 0.4, WOOD)
    d = b - a
    v = np.array([-d[1], d[0]]) / max(float(np.hypot(*d)), 1e-9)
    for t in np.linspace(0.2, 1.0, 3):
        for s in (-0.9, 0.9):
            p.box(a + d * t + v * s, 0.08, 0.08, yw - 0.5, yw + 0.25, 0.0, WOOD, hit=False)


BOAT_PAINTS = (BLUE, RED, WHITE, YELLOW)


def boat(p: Props, c, heading, yw, k: int):
    """A 3 x 1.2 m rowing boat on the water at yw: a pointed hull in one of four colours, a wooden floor and a thwart."""
    hull = np.array([(-1.5, -0.6), (1.0, -0.6), (1.5, 0.0), (1.0, 0.6), (-1.5, 0.6)])
    cs, sn = np.cos(heading), np.sin(heading)
    pts = np.asarray(c) + hull @ np.array([[cs, sn], [-sn, cs]])
    p.extrude(pts, yw - 0.1, yw + 0.3, BOAT_PAINTS[k % 4], top=WOOD, hit=False)
    p.box(c, 0.1, 0.55, yw + 0.3, yw + 0.38, heading, WOOD, hit=False)
