"""Sidewalks: a raised pavement over each dense city block, and a platform with a path around buildings elsewhere.

A block is a face of the road network (as for the filler boxes), and its land is the block less the street
surface contravia.streets already laid. A dense block, one built over more than DENSE of its land or a block of
the old centre built over more than CENTRE_DENSE, rises KERB metres over the street as one pavement, kerb to
kerb, and its buildings stand on it. Anywhere else each building gets an APRON-wide platform around its
footprint and a path to the nearest street, if one is within PATH_REACH; the rest stays bare ground.

The pavement's top stands KERB over the higher of the ground and the nearest street, so it never dips under
either; its kerb hangs from the top to below the street. Everything is cut per cell like the streets and
stored as walk_tris, with walk_paint telling each triangle's zone and part (paint()), so the cells draw it and
the wheels stand on it (contravia/surface.py). `contravia walks <data.npz>` rebuilds it in place without reading
OSM again.
"""

import math

import numpy as np
import shapely
from shapely.geometry import MultiLineString, Polygon, box
from shapely.ops import polygonize, unary_union

from contravia.buildingtags import MUNI
from contravia.geo import CELL

KERB = 0.15  # metres a pavement stands over the street
APRON = 1.5  # the width of the platform around a building outside the dense blocks
PATH = 1.4  # the width of the path from a building to its street
PATH_REACH = 25.0  # metres at most from a building to the street its path reaches
DENSE = 0.3  # the share of a block's land built over that makes it one raised pavement
CENTRE_DENSE = 0.12  # the same share in the old centre's municipios
MAX_BLOCK = 30_000.0  # m², as in contravia.prepare: larger faces are not city blocks
SAMPLE = 12.0  # metres between the vertices along a block pavement's edges, so its top follows the ground
SIMPLIFY = 0.3  # metres a pavement's outline may move to drop the vertices of the streets' rounded corners
OFF_GROUND = 3.0  # street triangles this far off the ground (bridges, tunnels) do not cut the land
CENTRE = ("La Habana Vieja", "Centro Habana", "Cerro", "Diez de Octubre", "Regla", "Guanabacoa")
VEDADO = ("Plaza de la Revolución", "Playa")
TOP, KERB_FACE, APRON_TOP, APRON_EDGE = range(4)  # the parts of walk_paint


def paint(zone: int, part: int) -> int:
    """One walk_paint value: zone 0 is the old centre, 1 Vedado and Playa, 2 the rest; part is TOP to APRON_EDGE."""
    return zone * 4 + part


def zone_of(muni: str | None) -> int:
    return 0 if muni in CENTRE else 1 if muni in VEDADO else 2


def _streets_by_cell(a) -> dict:
    """Each cell's street triangles at ground level, as one polygon, and the heights of their corners."""
    tris, off, cells = a["street_tris"], a["street_tris_offsets"], a["cells"]
    out = {}
    for k, (i, j) in enumerate(cells.tolist()):
        t = np.asarray(tris[off[k]:off[k + 1]], float)
        if not len(t):
            continue
        out[(i, j)] = t
    return out


def _street_polys(by_cell, terrain) -> dict:
    """The union of each cell's ground-level street triangles."""
    out = {}
    for key, t in by_cell.items():
        xz = t[:, :, [0, 2]]
        ground = terrain.height(xz[:, :, 0].ravel(), xz[:, :, 1].ravel()).reshape(-1, 3)
        t2 = xz[np.abs(t[:, :, 1] - ground).max(axis=1) < OFF_GROUND]
        if len(t2):
            out[key] = shapely.union_all(shapely.polygons(t2)).buffer(0.01)
    return out


def _cells_of(bounds):
    x0, z0, x1, z1 = bounds
    return [(i, j) for i in range(math.floor(x0 / CELL), math.floor(x1 / CELL) + 1)
            for j in range(math.floor(z0 / CELL), math.floor(z1 / CELL) + 1)]


class _Heights:
    """The height a pavement's top stands at: KERB over the higher of the ground and the nearest street corner."""

    def __init__(self, by_cell, terrain):
        self.by_cell, self.terrain, self._near = by_cell, terrain, {}

    def _corners(self, i, j):
        key = (i, j)
        if key not in self._near:
            parts = [self.by_cell[(i + a, j + b)].reshape(-1, 3) for a in (-1, 0, 1) for b in (-1, 0, 1)
                     if (i + a, j + b) in self.by_cell]
            self._near[key] = np.unique(np.vstack(parts), axis=0) if parts else np.zeros((0, 3))
        return self._near[key]

    def top(self, xz: np.ndarray, lift: float = KERB) -> np.ndarray:
        xz = np.asarray(xz, float).reshape(-1, 2)
        y = self.terrain.height(xz[:, 0], xz[:, 1]).astype(float)
        cells = np.floor(xz / CELL).astype(int)
        for i, j in {tuple(c) for c in cells.tolist()}:
            m = (cells[:, 0] == i) & (cells[:, 1] == j)
            c = self._corners(i, j)
            if not len(c):
                continue
            d = (xz[m, None, 0] - c[None, :, 0]) ** 2 + (xz[m, None, 1] - c[None, :, 2]) ** 2
            y[m] = np.maximum(y[m], c[np.argmin(d, axis=1), 1])
        return y + lift


def _facing(t: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Triangles t (T, 3, 3) wound so their normal points along want (T, 3): the cells take a normal from the
    winding, and the engine culls the back."""
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = np.einsum("ij,ij->i", n, want) < 0
    t = t.copy()
    t[flip] = t[flip][:, [0, 2, 1]]
    return t


def _triangles(poly, heights: _Heights, lift: float, sample: float | None):
    """The top of a pavement polygon, cut per cell: (T, 3, 3) triangles and their (T, 2) cell keys. sample, if
    given, adds vertices along the edges that far apart."""
    tris, keys = [], []
    poly = poly.segmentize(sample) if sample else poly
    for i, j in _cells_of(poly.bounds):
        part = shapely.intersection(poly, box(i * CELL, j * CELL, (i + 1) * CELL, (j + 1) * CELL))
        for g in getattr(part, "geoms", [part]):
            if not isinstance(g, Polygon) or g.area < 0.05:
                continue
            t = shapely.get_coordinates(shapely.constrained_delaunay_triangles(g)).reshape(-1, 4, 2)[:, :3]
            if not len(t):
                continue
            y = heights.top(t.reshape(-1, 2), lift).reshape(-1, 3)
            t3 = np.stack([t[:, :, 0], y, t[:, :, 1]], axis=2)
            tris.append(_facing(t3, np.tile((0.0, 1.0, 0.0), (len(t3), 1))))
            keys.append(np.tile((i, j), (len(t), 1)))
    return tris, keys


def _edges(poly, heights: _Heights, lift: float, drop: float, sample: float | None):
    """The faces hanging from a pavement's edges down by drop: two triangles per piece of every ring (pieces at
    most sample long, if given), keyed by the cell of the piece's middle."""
    tris, keys = [], []
    poly = shapely.orient_polygons(poly)  # exteriors counter-clockwise, holes clockwise: the walk is on the left
    for g in getattr(poly, "geoms", [poly]):
        if not isinstance(g, Polygon):
            continue
        for ring in [g.exterior, *g.interiors]:
            p = shapely.get_coordinates(ring.segmentize(sample) if sample else ring)
            if len(p) < 2:
                continue
            a, b = p[:-1], p[1:]
            ya, yb = heights.top(a, lift), heights.top(b, lift)
            q = np.stack([np.c_[a[:, 0], ya, a[:, 1]], np.c_[b[:, 0], yb, b[:, 1]], np.c_[b[:, 0], yb - drop, b[:, 1]],
                          np.c_[a[:, 0], ya - drop, a[:, 1]]], axis=1)
            d = b - a
            out = np.c_[d[:, 1], np.zeros(len(d)), -d[:, 0]]  # the right of each piece: away from the walk
            tris.append(_facing(np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]), np.concatenate([out, out])))
            mid = np.floor((a + b) / 2 / CELL).astype(int)
            keys.append(np.concatenate([mid, mid]))
    return tris, keys


def _block_muni(cents_in, munis) -> str | None:
    names = [munis[k] for k in cents_in if munis[k] is not None]
    return max(set(names), key=names.count) if names else None


def build(a, terrain) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The walk triangles of prepared arrays a: (T, 3, 3) triangles, (T,) paint and (T, 2) cell keys, and counts."""
    by_cell = _streets_by_cell(a)
    streets = _street_polys(by_cell, terrain)
    heights = _Heights(by_cell, terrain)
    roads = a["roads"]
    blocks = [b for b in polygonize(unary_union(MultiLineString([((r[0], r[1]), (r[2], r[3])) for r in roads])))
              if b.area < MAX_BLOCK] if len(roads) else []
    rings = _building_rings(a)
    feet = [Polygon(r) for r in rings]
    feet = [f if f.is_valid else f.buffer(0) for f in feet]
    names = tuple(str(n) for n in a.get("building_muni_names", ()))
    attrs = a.get("building_attrs")
    munis = [None] * len(rings)
    if attrs is not None:
        col = attrs[:, MUNI]
        munis = [names[int(m)] if np.isfinite(m) and 0 <= int(m) < len(names) else None for m in col]
    cents = shapely.points(np.array([r.mean(axis=0) for r in rings]).reshape(-1, 2)) if rings else np.array([])
    tree = shapely.STRtree(cents)
    tris, keys, paints = [], [], []
    dense_blocks, open_feet = 0, set(range(len(rings)))

    def land(g):
        cut = [streets[c] for c in _cells_of(g.bounds) if c in streets]
        return shapely.difference(g, shapely.union_all(cut)) if cut else g

    def add(t, k, code):
        for tt, kk in zip(t, k, strict=True):
            tris.append(tt)
            keys.append(kk)
            paints.append(np.full(len(tt), code, np.uint8))

    for b in blocks:
        inside = tree.query(b, predicate="contains")
        ground = land(b).simplify(SIMPLIFY)
        if ground.is_empty or ground.area < 10:
            continue
        built = sum(feet[k].area for k in inside)
        muni = _block_muni(inside, munis)
        if built / ground.area < (CENTRE_DENSE if muni in CENTRE else DENSE):
            continue
        dense_blocks += 1
        open_feet -= set(inside.tolist())
        z = zone_of(muni)
        add(*_triangles(ground, heights, KERB, SAMPLE), paint(z, TOP))
        add(*_edges(ground, heights, KERB, KERB + 0.4, SAMPLE), paint(z, KERB_FACE))

    aprons = 0
    for k in sorted(open_feet):
        f = feet[k]
        if f.is_empty:
            continue
        pad = f.buffer(APRON, join_style="mitre")
        near = [streets[c] for c in _cells_of(pad.buffer(PATH_REACH).bounds) if c in streets]
        st = shapely.union_all(near) if near else None
        if st is not None and not st.is_empty:
            line = shapely.shortest_line(f, st)
            if line.length < PATH_REACH:
                pad = pad.union(line.buffer(PATH / 2, cap_style="flat"))
            pad = pad.difference(st)
        others = tree.query(pad)
        pad = pad.difference(shapely.union_all([feet[o] for o in others])).simplify(SIMPLIFY)
        if pad.is_empty or pad.area < 1:
            continue
        aprons += 1
        z = zone_of(munis[k])
        add(*_triangles(pad, heights, 0.12, None), paint(z, APRON_TOP))
        add(*_edges(pad, heights, 0.12, 0.35, None), paint(z, APRON_EDGE))

    if not tris:
        return np.zeros((0, 3, 3), np.float32), np.zeros(0, np.uint8), np.zeros((0, 2), np.int64), {}
    return (np.concatenate(tris).astype(np.float32), np.concatenate(paints), np.concatenate(keys).astype(np.int64),
            {"dense_blocks": dense_blocks, "aprons": aprons})


def _building_rings(a) -> list[np.ndarray]:
    """Every building ring of prepared arrays a, in the order of building_ids."""
    verts, starts = a["building_verts"], a["building_starts"]
    return [np.asarray(verts[starts[k]:starts[k + 1]], float) for k in range(len(starts) - 1)]


def add_to(city) -> dict:
    """Build the walks of a city with streets and store them in it, cut per cell. Returns counts."""
    for k in ("walk_tris", "walk_paint"):
        city.a.pop(k, None)
        city.a.pop(f"{k}_offsets", None)
    if "street_tris" not in city.a:
        return {}
    tris, paints, keys, report = build(city.a, city.terrain)
    city.add_kind("walk_tris", tris, keys)
    city.add_kind("walk_paint", paints, keys)
    report["walk_triangles"] = len(tris)
    return report


def main(path) -> dict:
    """Rebuild the walks of a prepared data file in place and print the counts."""
    import json
    import time

    from contravia.citydata import CityData

    t0 = time.perf_counter()
    city = CityData.load(path)
    report = add_to(city)
    city.save(path)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(report, indent=2))
    return report

