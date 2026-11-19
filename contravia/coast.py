"""The island's coast: the land OSM's natural=coastline encloses, in local metres, and its part in each quadtree node.

The map draws the land of a far node from this coast instead of from the relief, which is 90 m outside La Habana
and measures the tops of the mangroves (docs/superpowers/specs/2026-11-19-map-sea-coast-design.html). OSM draws
every coastline way with the land on its left, so a closed ring that turns counter-clockwise in (lon, lat)
holds land, and one that turns clockwise holds water inside land. `read` joins the ways into rings, keeps the
land and simplifies it to TOLERANCE; `save` and `load` keep it as WKB in one zstd file, coast.src, which the game
ships beside the province sources. `lands` cuts it per node when the index is written: each node keeps its part,
simplified to what a pixel spans when the node is drawn, and a leaf keeps it at TOLERANCE so the nodes below it,
made while playing, cut theirs from it (`below`). `shelf` grows it into the shallows, one mesh for the island,
which the game shows only in the map.
"""

import pickle
from compression import zstd
from pathlib import Path

import numpy as np

from contravia.geo import CELL

VERSION = 1
NAME = "coast.src"  # the file's name, in the sources and in the built island
TOLERANCE = 10.0  # metres the stored coast is simplified to
NODE_PIXELS = 1024  # a node is drawn at most about this many pixels wide
SHELF = 800.0  # metres of clear, shallow water around the land
SHELF_TOLERANCE = 50.0  # metres the shallows' outline is simplified to
SHELF_FILE = "shelf.npy"  # the shallows' triangles, beside the index


def tolerance(L: int) -> float:
    """The coast's error in a level-L node: its side over NODE_PIXELS, never finer than TOLERANCE."""
    return max(TOLERANCE, CELL * (1 << L) / NODE_PIXELS)


def node_box(L: int, a: int, b: int, origin):
    """The shapely box of node (L, a, b)."""
    import shapely

    x0, z0 = (origin[0] + (a << L)) * CELL, (origin[1] + (b << L)) * CELL
    side = (1 << L) * CELL
    return shapely.box(x0, z0, x0 + side, z0 + side)


def read(pbf) -> tuple:
    """(land, report) from an OSM file: the land as a shapely (Multi)Polygon in local metres, simplified to
    TOLERANCE, and how many coastline ways joined into rings and how many stayed open (left out)."""
    import osmium
    import shapely

    lines = []

    class Handler(osmium.SimpleHandler):
        def way(self, w):
            if w.tags.get("natural") == "coastline":
                try:
                    lines.append([(n.lon, n.lat) for n in w.nodes])
                except osmium.InvalidLocationError:
                    pass

    Handler().apply_file(str(pbf), locations=True)
    lines = [p for p in lines if len(p) > 1]
    if not lines:  # an extract with no coast: no land
        return shapely.Polygon(), {"ways": 0, "rings": 0, "open": 0, "open_km": 0.0, "land_km2": 0.0, "points": 0}
    merged = shapely.line_merge(shapely.multilinestrings([shapely.linestrings(p) for p in lines]),
                                directed=True)
    parts = shapely.get_parts(merged)
    closed = shapely.is_closed(parts) & (shapely.get_num_coordinates(parts) >= 4)
    if closed.any():
        coords, which = shapely.get_coordinates(parts[closed], return_index=True)
        rings = shapely.linearrings(coords, indices=which)
    else:
        rings = []
    ccw = shapely.is_ccw(rings) if len(rings) else np.zeros(0, bool)
    land = shapely.union_all(shapely.polygons(rings[ccw])) if ccw.any() else shapely.Polygon()
    if (~ccw).any():
        land = shapely.difference(land, shapely.union_all(shapely.polygons(rings[~ccw])))
    land = shapely.simplify(_local(land), TOLERANCE, preserve_topology=True)
    report = {"ways": len(lines), "rings": len(rings), "open": int((~closed).sum()),
              "open_km": round(float(shapely.length(_local(shapely.multilinestrings(parts[~closed])))) / 1000, 1)
              if (~closed).any() else 0.0,
              "land_km2": round(float(land.area) / 1e6, 1), "points": int(shapely.get_num_coordinates(land))}
    return land, report


def _local(geom):
    """geom from (lon, lat) degrees to local (x, z) metres."""
    import shapely

    from contravia.geo import to_local

    return shapely.transform(geom, lambda ll: to_local(ll[:, 1], ll[:, 0]))


def save(land, path) -> int:
    """Write the land to path; returns its size in bytes."""
    import shapely

    data = zstd.compress(pickle.dumps({"version": VERSION, "wkb": shapely.to_wkb(land)}), level=19)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return len(data)


def load(path):
    """The land saved at path."""
    import shapely

    body = pickle.loads(zstd.decompress(Path(path).read_bytes()))
    if body.get("version") != VERSION:
        raise ValueError(f"{path}: coast version {body.get('version')}, this game reads {VERSION}")
    return shapely.from_wkb(body["wkb"])


def _keep(piece, tol: float):
    """piece simplified to tol, without the bits smaller than a tol x tol square."""
    import shapely

    piece = shapely.simplify(piece, tol, preserve_topology=True)
    parts = shapely.get_parts(piece)
    parts = parts[shapely.area(parts) >= tol * tol]
    return shapely.union_all(parts) if len(parts) else shapely.Polygon()


def lands(land, nodes, leaves, origin) -> dict:
    """{node: WKB of its land} for every node in nodes (a set of (L, a, b), each with its parent but the root's),
    cut from the parent's full-detail piece down from the root, so each cut is small. A leaf keeps TOLERANCE."""
    import shapely

    out = {}
    root = max(nodes)

    def visit(key, geom):
        L, a, b = key
        piece = shapely.intersection(geom, node_box(L, a, b, origin))
        out[key] = shapely.to_wkb(_keep(piece, TOLERANCE if key in leaves else tolerance(L)))
        for kid in _children(key) if L > 0 else []:
            if kid in nodes:
                visit(kid, piece)

    visit(root, land)
    return out


def shelf(land) -> np.ndarray:
    """(T, 3, 2) float32 triangles of the shallows: the land grown by SHELF, simplified to SHELF_TOLERANCE. They
    run under the land too, which stands over them, so one mesh for the whole island has no seams."""
    import shapely

    band = shapely.simplify(shapely.buffer(shapely.simplify(land, SHELF_TOLERANCE), SHELF, quad_segs=2),
                            SHELF_TOLERANCE)
    parts = shapely.get_parts(band)
    if not len(parts):
        return np.zeros((0, 3, 2), np.float32)
    tris = shapely.get_coordinates(shapely.constrained_delaunay_triangles(parts)).reshape(-1, 4, 2)[:, :3]
    return tris.astype(np.float32)


def _children(key):
    L, a, b = key
    return [(L - 1, 2 * a + da, 2 * b + db) for da in (0, 1) for db in (0, 1)]


def below(leaf_land: bytes, L: int, a: int, b: int, origin) -> bytes:
    """The land of node (L, a, b), which lies inside a leaf, cut from that leaf's land."""
    import shapely

    piece = shapely.intersection(shapely.from_wkb(leaf_land), node_box(L, a, b, origin))
    return shapely.to_wkb(_keep(piece, tolerance(L)))
