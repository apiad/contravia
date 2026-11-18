"""Medians: what stands between the two carriageways of the avenues named in STRETCHES.

Where the two one-way carriageways run close, contravia.streets pairs them into one ribbon; a median is drawn
on that ribbon's centreline, stopping GAP metres short of each junction:

- "kerb": Carlos III's concrete strip, KERB_W wide and KERB_H high, a two-armed street lamp every LAMP_EVERY.
- "paseo": Boyeros from the Plaza to the Ciudad Deportiva, a PASEO_W paved walk with a tree every TREE_EVERY.
- "wall": Boyeros south of the Ciudad Deportiva, a WALL_W concrete wall WALL_H high the car hits.

Where the carriageways run far apart (Paseo, Prado, 5ta Avenida), the ground between them is a block of its
own; contravia.walks paves it (boulevard()).

Everything comes out as contravia.structures triangles and paints, and walls as struct_walls rows.
"""

import re
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString

from contravia.geo import CELL

GAP = 4.0  # metres a median stops short of a junction: room for a crossing
STEP = 3.0  # metres between a median's cross-sections, so it follows the street's height
KERB_W, KERB_H = 1.0, 0.2
PASEO_W, PASEO_H = 3.0, 0.15
WALL_W, WALL_H = 0.5, 1.0
LAMP_EVERY, LAMP_H, LAMP_ARM = 30.0, 9.0, 1.6  # metres between lamps, their height and arm
TREE_EVERY = 12.0
MATCH = 0.7  # the share of a paired ribbon's points that must lie on a stretch's named edges
ACROSS = 45.0  # metres: a boulevard's median between its carriageways' centrelines is narrower than this
ROTONDA_Z = 3500.0  # the Ciudad Deportiva's roundabout on Boyeros: the paseo north of it, the wall south


@dataclass(frozen=True)
class Stretch:
    pattern: str  # a regex over the street name
    kind: str  # "kerb", "paseo", "wall" or "boulevard"
    zmin: float = -np.inf
    zmax: float = np.inf


STRETCHES = (
    Stretch(r"Carlos III", "kerb"),
    Stretch(r"Boyeros|Avenida de la Independencia", "paseo", zmax=ROTONDA_Z),
    Stretch(r"Boyeros|Avenida de la Independencia", "wall", zmin=ROTONDA_Z),
    Stretch(r"^Paseo$|Paseo del Prado|^5ta Avenida$", "boulevard"),
)


def named_lines(a, kind: str) -> list:
    """The graph's edge pieces, as LineStrings, of the streets the stretches of this kind name, within their z."""
    if "g_names" not in a:
        return []
    names = [str(n) for n in a["g_names"]]
    poly, starts, name = a["g_poly"].astype(float), a["g_poly_start"], a["g_name"]
    out = []
    for s in (s for s in STRETCHES if s.kind == kind):
        hit = {k for k, n in enumerate(names) if re.search(s.pattern, n)}
        for e in np.flatnonzero(np.isin(name, list(hit))):
            p = poly[starts[e]:starts[e + 1]]
            mid = p[len(p) // 2]
            if s.zmin <= mid[1] < s.zmax and len(p) >= 2:
                out.append(LineString(p))
    return out


def kinds(a, chains) -> dict:
    """{chain index: kind} for the paired ground chains that run along a stretch's named carriageways."""
    out = {}
    for kind in ("kerb", "paseo", "wall"):
        lines = named_lines(a, kind)
        if not lines:
            continue
        tree = shapely.STRtree(lines)
        for c, e in enumerate(chains):
            if not e.paired or e.layer or c in out:
                continue
            pts = shapely.points(e.p)
            near = tree.query(pts, predicate="dwithin", distance=e.w / 2 + 2.0)
            if len(np.unique(near[0])) >= MATCH * len(e.p):
                out[c] = kind
    return out


def _quad(a, b, c, d, want) -> list:
    """Two triangles of the quad a b c d, wound so their normal points along want."""
    t = np.array([[a, b, c], [a, c, d]], float)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = n @ np.asarray(want, float) < 0
    t[flip] = t[flip][:, [0, 2, 1]]
    return list(t)


def _box(p0, p1, half, y0, y1) -> list:
    """The triangles of a box along the segment p0 p1 (x, z), half wide, from y0 to y1: top and four sides."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    d = d / max(float(np.hypot(*d)), 1e-9)
    n = np.array([-d[1], d[0]]) * half
    c = [p0 + n, p1 + n, p1 - n, p0 - n]
    v = lambda q, y: (q[0], y, q[1])  # noqa: E731
    out = _quad(v(c[0], y1), v(c[1], y1), v(c[2], y1), v(c[3], y1), (0, 1, 0))
    for i in range(4):
        q0, q1 = c[i], c[(i + 1) % 4]
        mid = (q0 + q1) / 2 - (p0 + p1) / 2
        out += _quad(v(q0, y0), v(q1, y0), v(q1, y1), v(q0, y1), (mid[0], 0, mid[1]))
    return out


def _strip(p, y, half, h) -> list:
    """A raised strip along the polyline p (N, 2) on road heights y (N,), half wide and h high."""
    out = []
    for k in range(len(p) - 1):
        out += _box(p[k], p[k + 1], half, min(y[k], y[k + 1]) - 0.3, max(y[k], y[k + 1]) + h)
    return out


def _lamp(q, d, y) -> tuple[list, list]:
    """A two-armed lamp at q on the median heading d, its foot at y: (pole and arms, lamp heads)."""
    n = np.array([-d[1], d[0]])
    top = y + LAMP_H
    body = _box(q - d * 0.12, q + d * 0.12, 0.12, y, top)
    heads = []
    for s in (1, -1):
        end = q + n * s * LAMP_ARM
        body += _box(q, end, 0.06, top - 0.12, top)
        heads += _box(end - n * s * 0.35, end + n * s * 0.1, 0.18, top - 0.3, top - 0.1)
    return body, heads


def _tree(q, y) -> tuple[list, list]:
    """A street tree at q on y: (trunk, crown), as triangles of contravia.rules.trees' template at 0.7 size."""
    from contravia.rules.trees import _template

    out = []
    for pos, _, idx in _template():
        t = pos[idx] * 0.7 + np.array([q[0], y, q[1]])
        out.append(list(t))
    return out[0], out[1]


def build(ribbons, owner, axes, chains, terrain, a):
    """(tris (T, 3, 3), paint (T,), cell keys (T, 2), walls (W, 6), wall keys (W, 2)) of every median."""
    from contravia.streets import _level, cut, densify, length, tangents
    from contravia.structures import CONCRETE, KERB_TOP, LAMPS, LEAVES, PAVING, POLE, TRUNK

    which = kinds(a, chains)
    tris, paint, walls = [], [], []

    def add(ts, code):
        tris.extend(ts)
        paint.extend([code] * len(ts))

    for k in range(len(ribbons)):
        kind = which.get(owner[k])
        if kind is None:
            continue
        q = np.asarray(axes[k], float)
        L = length(q)
        if L < 2 * GAP + 4.0:
            continue
        p = densify(cut(q, GAP, L - GAP), STEP)
        e = chains[owner[k]]
        y = _level(terrain, p, np.full(len(p), e.w / 2))
        if kind == "kerb":
            add(_strip(p, y, KERB_W / 2, KERB_H), KERB_TOP)
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
            t = tangents(p)
            for at in np.arange(LAMP_EVERY / 2, s[-1], LAMP_EVERY):
                i = min(int(np.searchsorted(s, at)), len(p) - 1)
                body, heads = _lamp(p[i], t[i], y[i] + KERB_H)
                add(body, POLE)
                add(heads, LAMPS)
        elif kind == "paseo":
            add(_strip(p, y, PASEO_W / 2, PASEO_H), PAVING)
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
            for at in np.arange(TREE_EVERY / 2, s[-1], TREE_EVERY):
                i = min(int(np.searchsorted(s, at)), len(p) - 1)
                trunk, crown = _tree(p[i], y[i] + PASEO_H)
                add(trunk, TRUNK)
                add(crown, LEAVES)
        else:
            add(_strip(p, y, WALL_W / 2, WALL_H), CONCRETE)
            t = tangents(p)
            n = np.stack([-t[:, 1], t[:, 0]], axis=1) * (WALL_W / 2)
            for side in (n, -n):
                e_ = p + side
                for i in range(len(p) - 1):
                    walls.append((*e_[i], *e_[i + 1], min(y[i], y[i + 1]), max(y[i], y[i + 1]) + WALL_H))
    tris = np.array(tris, np.float32).reshape(-1, 3, 3)
    paint = np.array(paint, np.uint8)
    keys = np.floor(tris[:, :, [0, 2]].mean(axis=1) / CELL).astype(np.int64).reshape(-1, 2)
    walls = np.array(walls, np.float32).reshape(-1, 6)
    wkeys = np.floor((walls[:, 0:2] + walls[:, 2:4]) / 2 / CELL).astype(np.int64).reshape(-1, 2)
    return tris, paint, keys, walls, wkeys


def boulevard(a, blocks) -> np.ndarray:
    """Which of the blocks (polygons) are the ground between a boulevard's carriageways: thin (under ACROSS metres
    across) and bounded for half their outline or more by the boulevards' named edges."""
    lines = named_lines(a, "boulevard")
    out = np.zeros(len(blocks), bool)
    if not lines or not len(blocks):
        return out
    near = shapely.union_all(lines).buffer(2.0)
    rings = shapely.get_exterior_ring(blocks)
    share = shapely.length(shapely.intersection(rings, near)) / np.maximum(shapely.length(rings), 1e-9)
    across = 2 * shapely.area(blocks) / np.maximum(shapely.length(rings), 1e-9)
    return (share >= 0.5) & (across < ACROSS)
