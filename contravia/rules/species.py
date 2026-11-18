"""Six Cuban trees as low-poly templates, and the colour of each one's crown on a given day.

A template is one tree at the origin, its trunk standing on y = 0, flat-shaded: every triangle has its own three
vertices and its face's normal. Each vertex carries a role (TRUNK, FIXED, LEAF, ACCENT): the first two keep the
template's own colour, LEAF and ACCENT take the colours season() gives for the day, times the vertex's shade
(darker under a crown, lighter on top), so a crown reads round without a texture.

A crown is full, sparse or bare: the dry season thins the flamboyán and strips the ceiba, and those templates
have fewer triangles, so a February city draws less. Nothing here is stored: a tree's look comes from its
species, its seed and the date.
"""

import functools
from dataclasses import dataclass
from datetime import date

import numpy as np

SPECIES = ("laurel", "flamboyan", "almendro", "palma", "coco", "ceiba")
CROWNS = ("full", "sparse", "bare")
TRUNK_R = {"laurel": 0.5, "flamboyan": 0.3, "almendro": 0.3, "palma": 0.25, "coco": 0.25, "ceiba": 1.5}  # collision
TRUNK, FIXED, LEAF, ACCENT = 0, 1, 2, 3

BARK = {"laurel": (0.33, 0.24, 0.17), "flamboyan": (0.36, 0.27, 0.19), "almendro": (0.30, 0.22, 0.16),
        "palma": (0.62, 0.60, 0.55), "coco": (0.45, 0.37, 0.27), "ceiba": (0.55, 0.52, 0.47)}
GREEN = {"laurel": (0.10, 0.30, 0.10), "flamboyan": (0.20, 0.42, 0.14), "almendro": (0.18, 0.40, 0.15),
         "palma": (0.20, 0.42, 0.16), "coco": (0.24, 0.45, 0.18), "ceiba": (0.14, 0.34, 0.12)}
FLOWER = (0.80, 0.22, 0.12)
AUTUMN = (0.82, 0.52, 0.16)
RED_LEAF = (0.72, 0.27, 0.14)
PALE = (0.45, 0.50, 0.28)
CROWNSHAFT = (0.30, 0.48, 0.22)
NUT = (0.38, 0.30, 0.12)


class _Mesh:
    def __init__(self):
        self.pos, self.nrm, self.role, self.rgb, self.shade = [], [], [], [], []

    def tri(self, a, b, c, role, rgb=(1.0, 1.0, 1.0), shade=1.0, outward=None):
        a, b, c = (np.asarray(v, float) for v in (a, b, c))
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            return
        n /= np.linalg.norm(n)
        if outward is not None and np.dot(n, (a + b + c) / 3 - outward) < 0:
            n, b, c = -n, c, b
        for v in (a, b, c):
            self.pos.append(v)
            self.nrm.append(n)
            self.role.append(role)
            self.rgb.append(rgb)
            # crowns darken toward their underside: shade from the face's normal
            self.shade.append(shade if role < LEAF else 0.8 + 0.25 * max(-1.0, min(1.0, n[1])) * (1 if n[1] > 0 else 0.8))

    def two_sided(self, a, b, c, role, rgb=(1.0, 1.0, 1.0)):
        """A thin blade seen from both sides: the engine culls back faces."""
        self.tri(a, b, c, role, rgb)
        self.tri(a, c, b, role, rgb)

    def prism(self, base, top, r0, r1, sides, role, rgb, level=False):
        """A tapered tube from point base (radius r0) to point top (radius r1), without caps; with level, its
        rings lie flat, as a leaning trunk meets the ground."""
        base, top = np.asarray(base, float), np.asarray(top, float)
        axis = (top - base) / np.linalg.norm(top - base)
        u = np.cross(axis, (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0))
        u /= np.linalg.norm(u)
        w = np.cross(axis, u)
        if level:
            u, w = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        ring = [np.cos(2 * np.pi * k / sides) * u + np.sin(2 * np.pi * k / sides) * w for k in range(sides)]
        mid = (base + top) / 2
        for k in range(sides):
            p0, p1 = base + r0 * ring[k], base + r0 * ring[(k + 1) % sides]
            q0, q1 = top + r1 * ring[k], top + r1 * ring[(k + 1) % sides]
            self.tri(p0, p1, q1, role, rgb, outward=mid)
            self.tri(p0, q1, q0, role, rgb, outward=mid)

    def blob(self, centre, radii, role, rgb=(1.0, 1.0, 1.0)):
        """An icosahedron of 20 faces stretched to radii (rx, ry, rz): one lump of a crown."""
        centre, radii = np.asarray(centre, float), np.asarray(radii, float)
        for f in _ICO_FACES:
            a, b, c = (centre + _ICO[k] * radii for k in f)
            self.tri(a, b, c, role, rgb, outward=centre)

    def frond(self, base, yaw, length, rise, droop, width, role, rgb=(1.0, 1.0, 1.0)):
        """A palm frond: a narrow diamond from base, rising then drooping, as 2 triangles seen from both sides."""
        base = np.asarray(base, float)
        d = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        side = np.array([-d[2], 0.0, d[0]]) * width / 2
        mid = base + d * length * 0.5 + (0.0, rise, 0.0)
        tip = base + d * length + (0.0, rise - droop, 0.0)
        self.two_sided(base, mid + side, mid - side, role, rgb)
        self.two_sided(mid + side, tip, mid - side, role, rgb)

    def done(self):
        p = np.array(self.pos, float).reshape(-1, 3)
        p[:, 1] = np.maximum(p[:, 1], 0.0)  # a leaning trunk's base ring would dip a little under the ground
        return Template(p, np.array(self.nrm, float).reshape(-1, 3), np.arange(len(p)).reshape(-1, 3),
                        np.array(self.role, np.int8), np.array(self.rgb, float).reshape(-1, 3),
                        np.array(self.shade, float))


def _icosahedron():
    t = (1 + 5 ** 0.5) / 2
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    return v, f


_ICO, _ICO_FACES = _icosahedron()


@dataclass(frozen=True)
class Template:
    """One tree at the origin, flat-shaded: (V, 3) positions and normals, (T, 3) triangles, and per vertex its
    role, its own colour (for TRUNK and FIXED) and its shade."""

    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    role: np.ndarray
    rgb: np.ndarray
    shade: np.ndarray


def _laurel(m: _Mesh, crown: str):
    m.prism((0, 0, 0), (0, 3.2, 0), 0.5, 0.4, 5, TRUNK, BARK["laurel"])
    lumps = [((-1.6, 5.6, 0.6), (2.8, 2.3, 2.8)), ((1.5, 5.8, -0.7), (2.7, 2.2, 2.6)), ((0.1, 7.6, 0.1), (3.0, 2.4, 3.0))]
    for c, r in lumps[: {"full": 3, "sparse": 2, "bare": 0}[crown]]:
        m.blob(c, r, LEAF)


def _flamboyan(m: _Mesh, crown: str):
    m.prism((0, 0, 0), (0, 3.0, 0), 0.3, 0.25, 5, TRUNK, BARK["flamboyan"])
    for yaw in (0.3, 2.4, 4.4):
        m.prism((0, 2.8, 0), (2.8 * np.cos(yaw), 5.4, 2.8 * np.sin(yaw)), 0.2, 0.1, 3, TRUNK, BARK["flamboyan"])
    if crown == "bare":
        return
    if crown == "sparse":
        m.blob((0, 6.0, 0), (4.0, 0.8, 4.0), LEAF)
        return
    m.blob((0, 6.0, 0), (5.5, 1.3, 5.5), LEAF)
    m.blob((0.2, 6.7, -0.1), (5.3, 1.1, 5.3), ACCENT)


def _almendro(m: _Mesh, crown: str):
    m.prism((0, 0, 0), (0, 9.0, 0), 0.3, 0.15, 5, TRUNK, BARK["almendro"])
    tiers = [((0, 3.8, 0), (4.4, 0.6, 4.4), ACCENT), ((0, 6.0, 0), (3.5, 0.55, 3.5), LEAF),
             ((0, 8.2, 0), (2.3, 0.5, 2.3), LEAF)]
    for c, r, role in tiers[: {"full": 3, "sparse": 2, "bare": 0}[crown]]:
        m.blob(c, r, role)


def _palm(m: _Mesh, crown: str, height: float, lean: float, fronds: int, nuts: bool):
    """A palm: a trunk of three bending sections, a green crownshaft (royal palm) or nuts (coconut), fronds."""
    pts = [np.array((lean * (k / 3) ** 2, height * k / 3, 0.0)) for k in range(4)]
    radii = (0.35, 0.3, 0.32, 0.22) if not nuts else (0.3, 0.25, 0.22, 0.18)
    sides = 5
    for k in range(3):
        m.prism(pts[k], pts[k + 1], radii[k], radii[k + 1], sides, TRUNK, BARK["coco" if nuts else "palma"], level=True)
    top = pts[3]
    if not nuts:
        m.prism(top, top + (0, 2.0, 0), 0.24, 0.2, 5, FIXED, CROWNSHAFT)
        top = top + (0, 2.0, 0)
    else:
        m.blob(top + (0.0, -0.3, 0.0), (0.55, 0.35, 0.55), FIXED, NUT)  # the cluster of nuts under the fronds
    n = fronds if crown == "full" else fronds // 2 if crown == "sparse" else 0
    for k in range(n):
        up = k % 2 == 0  # every other frond stands higher, so the crown reads as a ball, not a broom
        m.frond(top, 2 * np.pi * k / n + 0.3, 5.5 if not nuts else 5.0, 1.6 if up else 0.6, 2.6 if up else 3.4, 1.0, LEAF)


def _ceiba(m: _Mesh, crown: str):
    m.prism((0, 0, 0), (0, 3.0, 0), 1.5, 0.8, 6, TRUNK, BARK["ceiba"])
    m.prism((0, 3.0, 0), (0, 13.0, 0), 0.8, 0.6, 6, TRUNK, BARK["ceiba"])
    for yaw in (0.0, 2.1, 4.2):
        m.prism((0, 12.0, 0), (6.0 * np.cos(yaw), 15.0, 6.0 * np.sin(yaw)), 0.35, 0.15, 3, TRUNK, BARK["ceiba"])
    if crown == "bare":
        return
    r = 8.5 if crown == "full" else 6.0
    m.blob((0, 15.8, 0), (r, 2.2, r), LEAF)


@functools.cache
def template(species: str, crown: str = "full") -> Template:
    """The template of a species with a full, sparse or bare crown."""
    m = _Mesh()
    if species == "laurel":
        _laurel(m, crown)
    elif species == "flamboyan":
        _flamboyan(m, crown)
    elif species == "almendro":
        _almendro(m, crown)
    elif species == "palma":
        _palm(m, crown, 18.0, 0.0, 10, nuts=False)
    elif species == "coco":
        _palm(m, crown, 11.0, 2.2, 8, nuts=True)
    elif species == "ceiba":
        _ceiba(m, crown)
    else:
        raise ValueError(f"unknown species {species!r}")
    return m.done()


@dataclass(frozen=True)
class Look:
    """How a tree looks on a day: its crown, its leaves' colour, and its accent's (flowers, red leaves)."""

    crown: str
    leaf: tuple
    accent: tuple


def _mix(a, b, t: float) -> tuple:
    return tuple(float(x + (y - x) * t) for x, y in zip(a, b, strict=True))


def season(species: str, day: date, shift_days: int = 0) -> Look:
    """The look of a species on day, moved by shift_days so that neighbours do not all turn at once.

    Havana has no real autumn: a dry season from November to April, when some trees turn or drop their leaves,
    and a rainy summer when the flamboyán flowers. The months are approximate."""
    from datetime import timedelta

    m = (day + timedelta(days=int(shift_days))).month
    g = GREEN[species]
    if species == "flamboyan":
        if 5 <= m <= 8:
            return Look("full", _mix(g, FLOWER, 0.45), FLOWER)
        if m == 9:
            return Look("full", g, g)
        if m >= 10 or m == 1:
            return Look("full", _mix(g, AUTUMN, 0.7), _mix(g, AUTUMN, 0.9))
        return Look("sparse", PALE, PALE)
    if species == "almendro":
        if m == 12 or m <= 2:
            return Look("full", _mix(g, AUTUMN, 0.5), RED_LEAF)
        return Look("full", g, _mix(g, RED_LEAF, 0.15))
    if species == "ceiba":
        return Look("bare", g, g) if 1 <= m <= 3 else Look("full", g, g)
    leaf = _mix(g, PALE, 0.25) if m in (3, 4) else g
    return Look("full", leaf, leaf)
