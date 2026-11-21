"""Sidewalks: a raised pavement over each dense city block, and a platform with a path around buildings elsewhere.

A block is a face of the road network (as for the filler boxes), and its land is the block less the street
surface contravia.streets already laid. A dense block, one built over more than DENSE of its land or a block of
the old centre built over more than CENTRE_DENSE, rises KERB metres over the street as one pavement, kerb to
kerb, and its buildings stand on it. Anywhere else each building gets an APRON-wide platform around its
footprint and a path to the nearest street, if one is within PATH_REACH; the rest stays bare ground.

The work runs over whole arrays, not block by block. Each cell's land (its square less its streets, and for the
platforms less its buildings too) is cut once; every block and every platform is intersected with the land of
the cells it touches in one vectorized call, which also cuts the pieces per cell as the streets are cut. The
kerbs are the pieces' edges, less those lying on a cell's border. A pavement's top stands KERB over the higher
of the ground and the nearest street, read from a raster per cell into which the streets' heights are painted
and spread a few metres (StreetHeights), so every vertex is looked up at once; that raster is the part a GPU
could take over.

Where a street's edge stands VERGE_MIN or more over the ground outside those pavements, platforms and parks, a
verge leads off it: a SHOULDER-wide strip at the edge's height, then a slope down to the natural ground RUN
metres further, so a car leaving the street rolls down a slope instead of dropping off a step. The ground is a
30 m raster drawn on a 10 m grid, too coarse to carry a ditch a metre or two wide, so the verge is laid over it
as triangles, like the pavements, and never digs below it. Every raised edge is bevelled: its top runs BEVEL
times its height outward down to the street or the ground, so a wheel rolls up a kerb instead of hitting it.

A park's block (contravia.parks) is paved by its layout instead: its kerb round the edge (or the whole Prado, or
a town plaza in its tiles) as a pavement, its paths and plazas PATH_LIFT over the lawn and its dirt paths DIRT_LIFT.

Everything is stored as walk_tris, with walk_paint telling each triangle's zone and part (paint()), so the cells
draw it and the wheels stand on it (contravia/surface.py). `contravia walks <data.npz>` rebuilds it in place
without reading OSM again.
"""

import numpy as np
import shapely
from shapely.geometry import MultiLineString
from shapely.ops import polygonize, unary_union

from contravia import parks
from contravia.buildingtags import MUNI
from contravia.geo import CELL
from contravia.medians import boulevard
from contravia.streets import pmap

KERB = 0.15  # metres a pavement stands over the street
APRON = 1.5  # the width of the platform around a building outside the dense blocks
APRON_LIFT = 0.12  # metres a platform stands over the ground
PATH = 1.4  # the width of the path from a building to its street
PATH_REACH = 25.0  # metres at most from a building to the street its path reaches
DENSE = 0.3  # the share of a block's land built over that makes it one raised pavement
CENTRE_DENSE = 0.12  # the same share in the old centre's municipios
MAX_BLOCK = 30_000.0  # m², as in contravia.prepare: larger faces are not city blocks
SAMPLE = 12.0  # metres between the vertices along a block pavement's edges, so its top follows the ground
SIMPLIFY = 0.3  # metres a pavement's outline may move to drop the vertices of the streets' rounded corners
SEAM = 0.05  # metres: gaps this narrow between street triangles (a junction's plate and a ribbon) are street, not land
OFF_GROUND = 3.0  # street triangles this far off the ground (bridges, tunnels) do not cut the land
GRID = 2.0  # metres per pixel of the street-height raster
SPREAD = 6  # pixels a street's height spreads beyond it (12 m): farther, a pavement follows the ground
MARGIN = 16.0  # metres the raster reaches past its cell
CENTRE = ("La Habana Vieja", "Centro Habana", "Cerro", "Diez de Octubre", "Regla", "Guanabacoa")
VEDADO = ("Plaza de la Revolución", "Playa")
PATH_LIFT, DIRT_LIFT = 0.05, 0.03  # metres a park's paths and its dirt paths stand over the ground
TOP, KERB_FACE, APRON_TOP, APRON_EDGE, PATH_TOP, PATH_EDGE, DIRT_TOP, DIRT_EDGE = range(8)  # the parts of walk_paint
PARTS = 8  # each top is followed by its edge
SHOULDER_CODE, SLOPE_CODE = 3 * PARTS, 3 * PARTS + 1  # walk_paint of a verge's shoulder and slope, past every zone
PLAZA_CODE = 3 * PARTS + 2  # walk_paint of a town plaza's tiles (contravia.parks.PLAZA), in every zone
BEVEL = 1.5  # metres outward per metre of height of a raised edge's bevel
VERGE_MIN = 0.1  # metres a street's edge stands over the ground for a verge to lead off it
SHOULDER = 0.5  # metres of a verge's shoulder, level with the street's edge
RUN = 2.5  # metres over which a verge slopes down from its shoulder to the natural ground


def paint(zone: int, part: int) -> int:
    """One walk_paint value: zone 0 is the old centre, 1 Vedado and Playa, 2 the rest; part is TOP to DIRT_EDGE."""
    return zone * PARTS + part


def zone_of(muni: str | None) -> int:
    return 0 if muni in CENTRE else 1 if muni in VEDADO else 2


def _chunked(fn, *arrays, size=2048) -> np.ndarray:
    """fn over aligned slices of arrays on the thread pool, the results concatenated in order."""
    n = len(arrays[0])
    if n <= size:
        return fn(*arrays)
    return np.concatenate(pmap(lambda i: fn(*(x[i:i + size] for x in arrays)), range(0, n, size), chunk=1))


def _ground_streets(a, terrain) -> dict:
    """Each cell's street triangles that lie on the ground (not bridges or tunnels), (T, 3, 3)."""
    tris, off, cells = a["street_tris"], a["street_tris_offsets"], a["cells"]
    out = {}
    for k, (i, j) in enumerate(cells.tolist()):
        t = np.asarray(tris[off[k]:off[k + 1]], float)
        if not len(t):
            continue
        ground = terrain.height(t[:, :, 0].ravel(), t[:, :, 2].ravel()).reshape(-1, 3)
        t = t[np.abs(t[:, :, 1] - ground).max(axis=1) < OFF_GROUND]
        if len(t):
            out[(i, j)] = t
    return out


class StreetHeights:
    """A pavement's height: KERB (or a platform's lift) over the higher of the ground and the street nearby.

    Per cell, a raster of GRID-metre pixels reaching MARGIN past the cell holds the highest street point in each
    pixel, painted from the triangles' edges, then spread SPREAD pixels outward; pixels no street reaches hold
    nothing, and a vertex there stands on the ground alone."""

    def __init__(self, streets: dict, terrain):
        self.streets, self.terrain, self._rasters = streets, terrain, {}

    def _raster(self, i: int, j: int) -> np.ndarray:
        key = (i, j)
        if key not in self._rasters:
            self._rasters[key] = self._paint(i, j)
        return self._rasters[key]

    def _paint(self, i: int, j: int) -> np.ndarray:
        n = round((CELL + 2 * MARGIN) / GRID)
        r = np.full((n, n), -np.inf)
        parts = [self.streets[(i + a, j + b)] for a in (-1, 0, 1) for b in (-1, 0, 1) if (i + a, j + b) in self.streets]
        if parts:
            t = np.concatenate(parts)
            a_, b_ = t.reshape(-1, 3), t[:, [1, 2, 0]].reshape(-1, 3)  # every edge, a to b
            steps = np.maximum(1, np.ceil(np.hypot(b_[:, 0] - a_[:, 0], b_[:, 2] - a_[:, 2]) / GRID)).astype(int)
            s = (np.arange(steps.sum()) - np.repeat(np.cumsum(steps) - steps, steps)) / np.repeat(steps, steps)
            p = np.repeat(a_, steps, axis=0) + (np.repeat(b_ - a_, steps, axis=0)) * s[:, None]
            o = np.array([i * CELL - MARGIN, j * CELL - MARGIN])
            ix, iz = np.floor((p[:, [0, 2]] - o) / GRID).astype(int).T
            ok = (ix >= 0) & (ix < n) & (iz >= 0) & (iz < n)
            np.maximum.at(r, (iz[ok], ix[ok]), p[ok, 1])
            for _ in range(SPREAD):  # each empty pixel takes the highest of its eight neighbours
                g = np.pad(r, 1, constant_values=-np.inf)
                nb = np.max([g[1 + dz:1 + dz + n, 1 + dx:1 + dx + n] for dz in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
                r = np.where(np.isfinite(r), r, nb)
        return r

    def top(self, xz: np.ndarray, lift) -> np.ndarray:
        """The height of each point (N, 2) with lift (one or one per point) over the ground or the street."""
        xz = np.asarray(xz, float).reshape(-1, 2)
        y = self.terrain.height(xz[:, 0], xz[:, 1]).astype(float)
        cell = np.floor(xz / CELL).astype(np.int64)
        key = cell[:, 0] * 1_000_003 + cell[:, 1]
        order = np.argsort(key, kind="stable")
        bounds = np.flatnonzero(np.diff(key[order])) + 1
        n = round((CELL + 2 * MARGIN) / GRID)
        groups = np.split(order, bounds)
        new = list({(int(cell[g[0], 0]), int(cell[g[0], 1])) for g in groups if len(g)} - self._rasters.keys())
        self._rasters.update(zip(new, pmap(lambda c: self._paint(*c), new, chunk=8)))
        for group in groups:
            if not len(group):
                continue
            i, j = cell[group[0]]
            r = self._raster(int(i), int(j))
            o = np.array([i * CELL - MARGIN, j * CELL - MARGIN])
            ix, iz = np.clip(np.floor((xz[group] - o) / GRID).astype(int), 0, n - 1).T
            y[group] = np.maximum(y[group], r[iz, ix])
        return y + lift


def _cell_boxes(cells: np.ndarray) -> np.ndarray:
    c = np.asarray(cells, float)
    return shapely.box(c[:, 0] * CELL, c[:, 1] * CELL, (c[:, 0] + 1) * CELL, (c[:, 1] + 1) * CELL)


def _pieces(geoms: np.ndarray, land: np.ndarray, boxes: np.ndarray):
    """Every geometry cut by the land of each cell it touches: (pieces, the geometry's index, the cell's index),
    polygons only. A cell's land can come out of its differences with one shell inside another (one cell of
    Holguín), which the cut fails on: its shells are joined first."""
    land = shapely.make_valid(land, method="structure", keep_collapsed=False)
    gi, ci = shapely.STRtree(boxes).query(geoms)
    cut = _chunked(shapely.intersection, geoms[gi], land[ci])
    parts, k = shapely.get_parts(cut, return_index=True)
    keep = (shapely.get_type_id(parts) == 3) & (shapely.area(parts) > 0.05)
    return parts[keep], gi[k[keep]], ci[k[keep]]


def _tops(parts: np.ndarray, sample: float | None):
    """The triangles (T, 3, 2) of every polygon and the polygon each belongs to."""
    if sample:
        parts = shapely.segmentize(parts, sample)
    tri = shapely.constrained_delaunay_triangles(parts)
    xy, k = shapely.get_coordinates(tri, return_index=True)
    return xy.reshape(-1, 4, 2)[:, :3], k[::4]


def _sides(parts: np.ndarray, sample: float | None):
    """The edges (E, 2, 2) of every polygon's rings, the walk on their left, less those on a cell's border, and the
    polygon each belongs to."""
    rings, k = shapely.get_rings(shapely.orient_polygons(parts), return_index=True)
    if sample:
        rings = shapely.segmentize(rings, sample)
    xy, r = shapely.get_coordinates(rings, return_index=True)
    same = r[:-1] == r[1:]
    a, b = xy[:-1][same], xy[1:][same]
    owner = k[r[:-1][same]]

    def on_line(u, v):
        m = np.round(u / CELL) * CELL
        return (np.abs(u - m) < 1e-3) & (np.abs(v - m) < 1e-3)

    border = on_line(a[:, 0], b[:, 0]) | on_line(a[:, 1], b[:, 1])
    return np.stack([a, b], axis=1)[~border], owner[~border]


def _facing(t: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Triangles t (T, 3, 3) wound so their normal points along want (T, 3): the cells take a normal from the
    winding, and the engine culls the back."""
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = np.einsum("ij,ij->i", n, want) < 0
    t = t.copy()
    t[flip] = t[flip][:, [0, 2, 1]]
    return t


def _mesh(parts, cells_of_part, codes, lift, drop, sample, heights: StreetHeights, cells: np.ndarray):
    """The tops and the hanging edges of polygon pieces: triangles (T, 3, 3), their paint and their cell keys."""
    tri, k = _tops(parts, sample)
    y = heights.top(tri.reshape(-1, 2), np.repeat(lift[k], 3)).reshape(-1, 3)
    top = _facing(np.stack([tri[:, :, 0], y, tri[:, :, 1]], axis=2), np.tile((0.0, 1.0, 0.0), (len(tri), 1)))
    e, m = _sides(parts, sample)
    ya = heights.top(e[:, 0], lift[m])
    yb = heights.top(e[:, 1], lift[m])
    a, b = e[:, 0], e[:, 1]
    d = b - a
    away = np.c_[d[:, 1], -d[:, 0]] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]  # right: off the walk
    run = away * (BEVEL * lift[m])[:, None]
    fa, fb = a + run, b + run  # the bevel's foot, at the street or the ground the walk stands on
    za, zb = ya - lift[m], yb - lift[m]
    bevel = np.stack([np.c_[a[:, 0], ya, a[:, 1]], np.c_[b[:, 0], yb, b[:, 1]], np.c_[fb[:, 0], zb, fb[:, 1]],
                      np.c_[fa[:, 0], za, fa[:, 1]]], axis=1)
    face = np.stack([bevel[:, 3], bevel[:, 2], np.c_[fb[:, 0], yb - drop[m], fb[:, 1]],
                     np.c_[fa[:, 0], ya - drop[m], fa[:, 1]]], axis=1)  # upright under the foot, hiding the gap
    out = np.c_[away[:, 0], np.zeros(len(d)), away[:, 1]]
    up = out + (0.0, 1.0, 0.0)  # the bevel faces out and up
    side = _facing(np.concatenate([bevel[:, [0, 1, 2]], bevel[:, [0, 2, 3]], face[:, [0, 1, 2]], face[:, [0, 2, 3]]]),
                   np.concatenate([up, up, out, out]))
    side_codes = np.tile(codes[m], 4) + 1  # each part's edge paint follows its top's
    tris = np.concatenate([top, side])
    paint_ = np.concatenate([codes[k], side_codes]).astype(np.uint8)
    keys = cells[np.concatenate([cells_of_part[k]] + [cells_of_part[m]] * 4)]
    return tris, paint_, keys


def _every_street(a, cells: np.ndarray, street: np.ndarray, streets: dict) -> np.ndarray:
    """Per cell, the footprint of all its street triangles, those of bridges and tunnels too: the ground streets'
    own (street) wherever the cell holds no other."""
    tris, off = a["street_tris"], a["street_tris_offsets"]
    out = street.copy()
    for k, c in enumerate(map(tuple, cells.tolist())):
        if off[k + 1] - off[k] > (len(streets[c]) if c in streets else 0):
            t = np.asarray(tris[off[k]:off[k + 1]], float)[:, :, [0, 2]]
            out[k] = shapely.union_all(shapely.make_valid(shapely.polygons(t))).buffer(SEAM)
    return out


def _free_edges(streets: dict, street: np.ndarray, index: dict, terrain):
    """The edges of the ground streets' triangles with no street past them that stand VERGE_MIN or more over the
    ground at either end: (E, 2, 3) with their outward unit normals (E, 2) in (x, z). street holds per cell the
    footprint of every street, so the edge a bridge or a tunnel goes on from is not free."""
    edges, normals, homes = [], [], []
    for cell, t in streets.items():
        a, b, c = t, t[:, [1, 2, 0]], t[:, [2, 0, 1]]  # every edge a to b, c the triangle's third corner
        a, b, c = a.reshape(-1, 3), b.reshape(-1, 3), c.reshape(-1, 3)
        over = np.maximum(a[:, 1] - terrain.height(a[:, 0], a[:, 2]), b[:, 1] - terrain.height(b[:, 0], b[:, 2]))
        keep = over >= VERGE_MIN
        a, b, c = a[keep], b[keep], c[keep]
        d = b[:, [0, 2]] - a[:, [0, 2]]
        n = np.c_[d[:, 1], -d[:, 0]] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
        n *= np.where(np.einsum("ij,ij->i", c[:, [0, 2]] - a[:, [0, 2]], n) > 0, -1.0, 1.0)[:, None]
        edges.append(np.stack([a, b], axis=1)), normals.append(n), homes.append(np.full(len(a), index[cell]))
    if not edges:
        return np.zeros((0, 2, 3)), np.zeros((0, 2))
    e, n, home = np.concatenate(edges), np.concatenate(normals), np.concatenate(homes)
    probe = e[:, :, [0, 2]].mean(axis=1) + n * 0.1
    cell = np.floor(probe / CELL).astype(np.int64)
    there = np.array([index.get(c, -1) for c in map(tuple, cell.tolist())], int).reshape(-1)
    inside = np.zeros(len(e), bool)
    for k in (home, there):  # the edge's own cell's streets, and those of the cell the probe falls in
        for c in np.unique(k[k >= 0]):
            m = np.flatnonzero(k == c)
            inside[m] |= shapely.contains_xy(street[c], probe[m, 0], probe[m, 1])
    return e[~inside], n[~inside]


def _verges(streets, street, cells, boxes, cover, terrain, every=None):
    """The verges off the raised edges of the ground streets (see the module): triangles (T, 3, 3), their paint and
    their cell keys. cover holds per cell what else lies on the land there (pavements, platforms, parks,
    buildings), which a verge leaves alone."""
    index = {c: k for k, c in enumerate(map(tuple, cells.tolist()))}
    e, n = _free_edges(streets, street if every is None else every, index, terrain)
    if not len(e):
        return np.zeros((0, 3, 3)), np.zeros(0, np.uint8), np.zeros((0, 2), np.int64)
    a, b = e[:, 0, [0, 2]], e[:, 1, [0, 2]]
    # each end's normal mitred with the next edge's, so the strips of a curving edge neither gap nor overlap
    ends = np.round(np.concatenate([a, b]), 2)
    _, key = np.unique(ends, axis=0, return_inverse=True)
    vn = np.zeros((key.max() + 1, 2))
    np.add.at(vn, key, np.concatenate([n, n]))
    vn = vn / np.maximum(np.hypot(vn[:, 0], vn[:, 1]), 1e-9)[:, None]
    na, nb = vn[key[:len(e)]], vn[key[len(e):]]
    na = na / np.maximum(np.einsum("ij,ij->i", na, n), 0.5)[:, None]
    nb = nb / np.maximum(np.einsum("ij,ij->i", nb, n), 0.5)[:, None]
    quads = []
    for lo, hi in ((0.0, SHOULDER), (SHOULDER, SHOULDER + RUN)):
        quads.append(np.stack([a + na * lo, b + nb * lo, b + nb * hi, a + na * hi], axis=1))
    quads = np.concatenate(quads)
    kind = np.repeat([SHOULDER_CODE, SLOPE_CODE], len(e))
    src = np.tile(np.arange(len(e)), 2)
    geoms = shapely.make_valid(shapely.polygons(quads))
    touched = np.unique(shapely.STRtree(boxes).query(geoms)[1])
    free = np.full(len(boxes), shapely.Polygon(), dtype=object)
    free[touched] = shapely.difference(shapely.difference(boxes[touched], street[touched]), cover[touched])
    parts, gi, ci = _pieces(geoms, free, boxes)
    tri, k = _tops(parts, None)
    q = gi[k]  # the quad of each triangle
    p = tri.reshape(-1, 2)
    qv = np.repeat(q, 3)
    i = src[qv]
    ab = b[i] - a[i]
    t = np.clip(np.einsum("ij,ij->i", p - a[i], ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    edge_y = e[i, 0, 1] + t * (e[i, 1, 1] - e[i, 0, 1])
    ground = terrain.height(p[:, 0], p[:, 1])
    out = np.einsum("ij,ij->i", p - a[i], n[i])
    f = np.where(kind[qv] == SLOPE_CODE, np.clip((out - SHOULDER) / RUN, 0, 1), 0.0)
    y = np.maximum(edge_y + (ground - edge_y) * f, ground).reshape(-1, 3)
    tris = _facing(np.stack([tri[:, :, 0], y, tri[:, :, 1]], axis=2), np.tile((0.0, 1.0, 0.0), (len(tri), 1)))
    return tris, kind[q].astype(np.uint8), cells[ci[k]]


def _rings(a) -> list[np.ndarray]:
    verts, starts = a["building_verts"], a["building_starts"]
    return [np.asarray(verts[starts[k]:starts[k + 1]], float) for k in range(len(starts) - 1)]


def build(a, terrain) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The walk triangles of prepared arrays a: (T, 3, 3) triangles, (T,) paint and (T, 2) cell keys, and counts."""
    empty = np.zeros((0, 3, 3), np.float32), np.zeros(0, np.uint8), np.zeros((0, 2), np.int64), {}
    roads = a["roads"]
    if not len(roads):
        return empty
    streets = _ground_streets(a, terrain)
    heights = StreetHeights(streets, terrain)
    cells = np.asarray(a["cells"], np.int64).reshape(-1, 2)
    boxes = _cell_boxes(cells)
    street = np.array(pmap(lambda c: shapely.union_all(shapely.polygons(streets[c][:, :, [0, 2]])) if c in streets
                           else None, list(map(tuple, cells.tolist())), chunk=16), dtype=object)
    street = np.where(shapely.is_missing(street), shapely.Polygon(), street)
    # Closed: the plate and the ribbons are triangulated apart, and the hairline seam between them would be land that
    # rises as a pavement strip, bevelled both sides, across the street at every junction.
    street = shapely.buffer(shapely.buffer(street, SEAM, join_style="mitre"), -SEAM, join_style="mitre")
    land = _chunked(lambda b, s: shapely.simplify(shapely.difference(b, s), SIMPLIFY), boxes, street, size=256)

    rings = _rings(a)
    starts = np.asarray(a["building_starts"])
    feet = (shapely.make_valid(shapely.polygons(shapely.linearrings(
        np.asarray(a["building_verts"], float), indices=np.repeat(np.arange(len(rings)), np.diff(starts)))))
            if rings else np.array([], dtype=object))
    names = tuple(str(n) for n in a.get("building_muni_names", ()))
    attrs = a.get("building_attrs")
    muni = np.full(len(rings), -1)
    if attrs is not None and len(attrs):
        col = np.asarray(attrs[:, MUNI], float)
        muni = np.where(np.isfinite(col), col, -1).astype(int)
    zones = np.array([zone_of(n) for n in names] + [2])  # index -1: no municipio

    blocks = np.array([b for b in polygonize(unary_union(MultiLineString([((r[0], r[1]), (r[2], r[3])) for r in roads])))
                       if b.area < MAX_BLOCK], dtype=object)
    tris, paints, keys = [], [], []
    covers = []  # (pieces, their cells' indices) already laid on the land, which no verge may cover
    laid = parks.laid(a)
    park_rings = [np.asarray(a["park_verts"][a["park_starts"][k]:a["park_starts"][k + 1]], float)
                  for k in range(len(laid))]
    report_parks = 0
    dense = np.zeros(len(blocks), bool)
    in_dense = np.zeros(len(rings), bool)
    if len(blocks):
        parts, bi, ci = _pieces(blocks, land, boxes)
        area = np.bincount(bi, shapely.area(parts), minlength=len(blocks))
        bk, fk = shapely.STRtree(shapely.centroid(feet)).query(blocks, predicate="contains") if len(rings) else ([], [])
        bk, fk = np.asarray(bk, int), np.asarray(fk, int)
        built = np.bincount(bk, shapely.area(feet[fk]) if len(fk) else None, minlength=len(blocks))
        m = muni[fk]
        nm = len(names) + 1
        votes = np.bincount(bk * nm + np.where(m >= 0, m, nm - 1), minlength=len(blocks) * nm).reshape(-1, nm)
        votes[:, -1] = 0
        block_muni = np.where(votes.max(axis=1) > 0, votes.argmax(axis=1), -1)
        centre = np.isin(block_muni, [k for k, n in enumerate(names) if n in CENTRE])
        dense = (area > 10) & (built > np.where(centre, CENTRE_DENSE, DENSE) * np.maximum(area, 1e-9))
        dense |= (area > 10) & boulevard(a, blocks)  # Paseo, Prado, 5ta: paved between the carriageways
        if laid:  # a park's block is paved by its layout below
            dense &= ~shapely.contains(shapely.union_all(shapely.make_valid([shapely.Polygon(r) for r in park_rings])),
                                       shapely.point_on_surface(blocks))
        in_dense[fk[dense[bk]]] = True
        keep = dense[bi]
        if keep.any():
            codes = np.array([paint(z, TOP) for z in zones[block_muni[bi[keep]]]])
            n = int(keep.sum())
            t, p, k = _mesh(parts[keep], ci[keep], codes, np.full(n, KERB), np.full(n, KERB + 0.4), SAMPLE, heights,
                            cells)
            tris.append(t), paints.append(p), keys.append(k)
            covers.append((parts[keep], ci[keep]))

    if laid:
        geoms, codes, lifts, drops = [], [], [], []
        for park, zone in laid:
            for g, part, lift, drop in ((park.kerb, TOP, KERB, KERB + 0.4),
                                        (shapely.union_all(park.paved) if park.paved else None, PATH_TOP, PATH_LIFT, 0.3),
                                        (shapely.union_all(park.dirt) if park.dirt else None, DIRT_TOP, DIRT_LIFT, 0.2)):
                if g is not None and not g.is_empty:
                    geoms.append(shapely.make_valid(g))
                    code = PLAZA_CODE if park.kind == parks.PLAZA and part == TOP else paint(zone, part)
                    codes.append(code), lifts.append(lift), drops.append(drop)
        if geoms:
            parts, gi, ci = _pieces(np.array(geoms, dtype=object), land, boxes)
            t, p, k = _mesh(parts, ci, np.array(codes)[gi], np.array(lifts)[gi], np.array(drops)[gi], SAMPLE, heights,
                            cells)
            tris.append(t), paints.append(p), keys.append(k)
            covers.append((parts, ci))
            report_parks = len(laid)

    open_ = np.flatnonzero(~in_dense & ~shapely.is_empty(feet)) if len(rings) else np.zeros(0, int)
    aprons = 0
    if len(open_):
        f = feet[open_]
        pad = shapely.buffer(f, APRON, join_style="mitre")
        index = {c: k for k, c in enumerate(map(tuple, cells.tolist()))}
        home = np.floor(shapely.get_coordinates(shapely.centroid(f)) / CELL).astype(np.int64)
        own = np.array([index.get(c, -1) for c in map(tuple, home.tolist())])
        st = np.where(own >= 0, street[np.maximum(own, 0)], shapely.Polygon())
        line = shapely.shortest_line(f, st)
        reach = ~shapely.is_empty(st) & (shapely.length(line) < PATH_REACH)
        pad[reach] = shapely.union(pad[reach], shapely.buffer(line[reach], PATH / 2, cap_style="flat"))
        fi, fc = shapely.STRtree(boxes).query(feet)
        built_cell = np.array([None] * len(cells), dtype=object)
        order = np.argsort(fc, kind="stable")
        uc, first = np.unique(fc[order], return_index=True)
        built_cell[uc] = pmap(lambda g: shapely.union_all(feet[fi[g]]), np.split(order, first[1:]), chunk=16)
        built_cell = np.where(shapely.is_missing(built_cell), shapely.Polygon(), built_cell)
        apron_land = shapely.difference(land, built_cell)
        parts, pi, ci = _pieces(pad, apron_land, boxes)
        aprons = len(np.unique(pi))
        codes = np.array([paint(z, APRON_TOP) for z in zones[muni[open_[pi]]]])
        n = len(parts)
        t, p, k = _mesh(parts, ci, codes, np.full(n, APRON_LIFT), np.full(n, 0.35), None, heights, cells)
        tris.append(t), paints.append(p), keys.append(k)
        covers.append((parts, ci))
    if len(rings):
        fi, fc = shapely.STRtree(boxes).query(feet)
        covers.append((feet[fi], fc))

    cover = np.full(len(cells), shapely.Polygon(), dtype=object)
    if covers:
        geoms = np.concatenate([g for g, _ in covers])
        owner = np.concatenate([c for _, c in covers])
        order = np.argsort(owner, kind="stable")
        uc, first = np.unique(owner[order], return_index=True)
        # the union of valid pieces can still come out with nested shells, which the verges' overlay rejects
        cover[uc] = pmap(lambda g: shapely.make_valid(shapely.union_all(shapely.make_valid(geoms[g]))),
                         np.split(order, first[1:]), chunk=16)
    t, p, k = _verges(streets, street, cells, boxes, cover, terrain, _every_street(a, cells, street, streets))
    if len(t):
        tris.append(t), paints.append(p), keys.append(k)

    if not tris:
        return empty
    return (np.concatenate(tris).astype(np.float32), np.concatenate(paints), np.concatenate(keys).astype(np.int64),
            {"dense_blocks": int(dense.sum()), "aprons": aprons, "paved_parks": report_parks,
             "verge_triangles": len(t)})


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
    covered = parks.drop_covered(city)  # before the walks, which pave the parks left
    report = {**covered, **add_to(city), **parks.add_to(city)}
    city.save(path)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(report, indent=2))
    return report
