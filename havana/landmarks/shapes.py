"""The shapes landmark mock-ups are made of, and the Piece each landmark returns.

Every shape adds triangles to a MeshBuilder in world metres. Rings are (K, 2) arrays of (x, z) corners in
any winding; MeshBuilder turns each triangle to face the way its normals point.
"""

from dataclasses import dataclass, field

import numpy as np

from havana.mesh import signed_area, triangulate


@dataclass
class Piece:
    """What one landmark puts in the city besides its mesh."""

    replaces: set = field(default_factory=set)  # OSM building ids it stands in for
    clear: list = field(default_factory=list)  # rings where no OSM building, filler box or tree is kept
    walls: list = field(default_factory=list)  # rings the car hits


def prism(mb, ring, y0: float, y1: float, color, roof: bool = True) -> None:
    """Walls from y0 to y1 around a ring, and a flat roof at y1 unless roof is False."""
    ring = np.asarray(ring, float)
    ccw = signed_area(ring) > 0
    for k in range(len(ring)):
        (x0, z0), (x1, z1) = ring[k], ring[(k + 1) % len(ring)]
        length = np.hypot(x1 - x0, z1 - z0)
        if length < 1e-6:
            continue
        out = np.array([z1 - z0, -(x1 - x0)]) / length  # outside a counter-clockwise ring
        if not ccw:
            out = -out
        n = (out[0], 0.0, out[1])
        mb.add([(x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0)], [n] * 4, color, [(0, 1, 2), (0, 2, 3)])
    if roof:
        mb.add([(x, y1, z) for x, z in ring], [(0, 1, 0)] * len(ring), color, triangulate(ring))


def pyramid(mb, ring, y0: float, y1: float, color) -> None:
    """Sloped faces from a ring at y0 up to one apex above its centre at y1: a spire or a hipped roof."""
    ring = np.asarray(ring, float)
    c = ring.mean(axis=0)
    apex = np.array([c[0], y1, c[1]])
    for k in range(len(ring)):
        a = np.array([ring[k][0], y0, ring[k][1]])
        b = np.array([ring[(k + 1) % len(ring)][0], y0, ring[(k + 1) % len(ring)][1]])
        n = np.cross(b - a, apex - a)
        mid = (a + b) / 2
        if np.dot(n[[0, 2]], mid[[0, 2]] - c) < 0:  # point the face away from the centre
            n = -n
        mb.add([a, b, apex], [n / (np.linalg.norm(n) or 1)] * 3, color, [(0, 1, 2)])


def dome(mb, c, r: float, y: float, h: float, color, steps: int = 8) -> None:
    """A dome of radius r and height h standing on y, as stacked rings that narrow toward the top."""
    for k in range(steps):
        t0, t1 = k / steps * np.pi / 2, (k + 1) / steps * np.pi / 2
        prism(mb, circle(c, r * np.cos(t1) + 0.01), y + h * np.sin(t0), y + h * np.sin(t1), color,
              roof=k == steps - 1)


def circle(c, r: float, n: int = 24) -> np.ndarray:
    """A ring of n corners around c."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], axis=1)


def flat(mb, ring, y: float, color) -> None:
    """A flat polygon facing up at height y: paving or water."""
    ring = np.asarray(ring, float)
    mb.add([(x, y, z) for x, z in ring], [(0, 1, 0)] * len(ring), color, triangulate(ring))


def frame(ring):
    """The oriented box of a footprint: centre, long axis u, short axis v and the half sizes along each."""
    ring = np.asarray(ring, float)
    c = ring.mean(axis=0)
    _, _, vt = np.linalg.svd(ring - c)
    u, v = vt[0], np.array([-vt[0][1], vt[0][0]])
    pu, pv = (ring - c) @ u, (ring - c) @ v
    c = c + u * (pu.max() + pu.min()) / 2 + v * (pv.max() + pv.min()) / 2
    return c, u, v, (pu.max() - pu.min()) / 2, (pv.max() - pv.min()) / 2


def rect(c, u, a: float, b: float) -> np.ndarray:
    """The ring of a rectangle centred on c, a along the unit axis u and b across it (half sizes)."""
    c, u = np.asarray(c, float), np.asarray(u, float)
    v = np.array([-u[1], u[0]])
    return np.array([c - u * a - v * b, c + u * a - v * b, c + u * a + v * b, c - u * a + v * b])


def panel(mb, p0, p1, y0: float, y1: float, out, color) -> None:
    """A vertical rectangle from p0 to p1 between y0 and y1, facing the (x, z) direction out: a mural."""
    (x0, z0), (x1, z1) = p0, p1
    n = (out[0], 0.0, out[1])
    mb.add([(x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0)], [n] * 4, color, [(0, 1, 2), (0, 2, 3)])
