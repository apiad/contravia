"""The meshes of a cemetery's plots and pantheons, one per kind and level of detail, built from boxes.

Each mesh stands on the origin with its length along +z (a cross or an angel stands at the +z end, a pantheon's
door faces +z) and is drawn in grey shades for the instance's Tint to colour in its stone. Level 0 is drawn near
the camera with its mouldings, level 1 as one or two plain boxes, and level 2 keeps a plot's flat top (two
triangles), so a cemetery seen from far off is still a field of pale stone, and an angel's or a pantheon's box.
"""

import functools

import numpy as np

from contravia.cemetery import KINDS, PANTHEON, PLOT

LODS = 3
_FACES = np.array([((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
                   ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
                   ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))], float)


class _Boxes:
    """Triangles of boxes, each face flat-shaded, in one geometry dict."""

    def __init__(self):
        self.pos, self.nrm, self.col = [], [], []

    def box(self, centre, size, shade: float = 1.0, top: float | None = None) -> None:
        """A box of size (x, y, z) round centre; with top, its upper face is top times as wide in x and z."""
        c, h = np.asarray(centre, float), np.asarray(size, float) / 2
        for n, a, b in _FACES:
            corners = [c + h * (n + s * a + t * b) for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            if top is not None:
                corners = [p if p[1] < c[1] else np.array([c[0] + (p[0] - c[0]) * top, p[1], c[2] + (p[2] - c[2]) * top])
                           for p in corners]
            for k in (0, 1, 2, 0, 2, 3):
                self.pos.append(corners[k])
                self.nrm.append(n)
                self.col.append((shade, shade, shade))

    def top(self, y: float, size, shade: float = 1.0) -> None:
        """One upward face of size (x, z) at height y."""
        h = np.asarray(size, float) / 2
        corners = [np.array([s * h[0], y, t * h[1]]) for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        for k in (0, 2, 1, 0, 3, 2):
            self.pos.append(corners[k])
            self.nrm.append((0.0, 1.0, 0.0))
            self.col.append((shade, shade, shade))

    def done(self) -> dict:
        pos, nrm = np.array(self.pos, np.float32), np.array(self.nrm, np.float32)
        tri = pos.reshape(-1, 3, 3)
        facing = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), nrm[::3])
        idx = np.arange(len(pos), dtype=np.uint32).reshape(-1, 3)
        idx[facing < 0] = idx[facing < 0][:, ::-1]
        return {"positions": pos, "normals": nrm, "colors": np.array(self.col, np.float32), "indices": idx.ravel()}


def _tomb(m: _Boxes, lod: int, high: float) -> None:
    w, length = PLOT
    if lod == 0:
        m.box((0, 0.08, 0), (w, 0.16, length), 0.82)
        m.box((0, 0.16 + high / 2, 0), (w - 0.2, high, length - 0.2), 1.0)
        m.box((0, 0.19 + high, 0), (w - 0.1, 0.06, length - 0.1), 0.95)
    else:
        m.box((0, (0.22 + high) / 2, 0), (w, 0.22 + high, length), 1.0)


def build(kind: str, lod: int) -> dict | None:
    """The geometry dict of a kind at a level of detail, or None where that level does not draw it."""
    w, length = PLOT
    m = _Boxes()
    if lod == 2 and kind in ("slab", "tomb", "cross"):
        m.top({"slab": 0.12, "tomb": 0.71, "cross": 0.56}[kind], PLOT, 0.95)
        return m.done()
    if kind == "slab":
        m.box((0, 0.06, 0), (w, 0.12, length), 0.95)
        if lod == 0:
            m.box((0, 0.13, 0.55), (0.5, 0.02, 0.35), 0.55)  # the plate
    elif kind == "tomb":
        _tomb(m, lod, 0.5)
    elif kind == "cross":
        _tomb(m, lod, 0.35)
        z = length / 2 - 0.25
        if lod == 0:
            m.box((0, 1.05, z), (0.12, 1.3, 0.12), 0.9)
            m.box((0, 1.3, z), (0.6, 0.12, 0.12), 0.9)
        else:
            m.box((0, 1.05, z), (0.15, 1.3, 0.15), 0.9)
    elif kind == "angel":
        z = length / 2 - 0.35
        if lod < 2:
            _tomb(m, lod, 0.4)
        m.box((0, 0.45, z), (0.6, 0.9, 0.6), 0.85)  # the pedestal
        if lod == 0:
            m.box((0, 1.4, z), (0.42, 1.0, 0.34), 1.0, top=0.55)  # the robe
            m.box((0, 2.0, z), (0.17, 0.2, 0.17), 1.0)  # the head
            for s in (-1, 1):  # the wings, folded behind
                m.box((s * 0.16, 1.55, z - 0.2), (0.06, 0.9, 0.35), 0.95, top=0.6)
        else:
            m.box((0, 1.5, z), (0.4, 1.2, 0.35), 1.0, top=0.6)
    elif kind == "pantheon":
        pw, pd = PANTHEON
        if lod == 0:
            m.box((0, 0.15, 0), (pw, 0.3, pd), 0.8)
            m.box((0, 1.7, 0), (pw - 0.3, 2.8, pd - 0.3), 1.0)
            m.box((0, 3.2, 0), (pw + 0.1, 0.25, pd + 0.1), 0.9)  # the cornice
            m.box((0, 3.6, 0), (pw - 0.6, 0.6, pd - 0.6), 0.95, top=0.7)  # the roof
            m.box((0, 4.15, 0), (0.1, 0.6, 0.1), 0.9)  # the cross on top
            m.box((0, 4.25, 0), (0.4, 0.1, 0.1), 0.9)
            m.box((0, 1.3, pd / 2 - 0.12), (1.0, 2.0, 0.06), 0.22)  # the door
            for s in (-1, 1):  # the columns beside it
                m.box((s * 0.85, 1.6, pd / 2 - 0.05), (0.28, 2.7, 0.28), 1.0)
        else:
            m.box((0, 1.6, 0), (pw, 3.2, pd), 1.0)
            m.box((0, 3.5, 0), (pw - 0.6, 0.6, pd - 0.6), 0.95, top=0.7)
    else:
        raise ValueError(f"no tomb kind {kind!r} (known: {', '.join(KINDS)})")
    return m.done()


@functools.cache
def mesh(kind: int, lod: int) -> dict | None:
    """build() of KINDS[kind], made once."""
    return build(KINDS[kind], lod)
