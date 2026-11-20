"""What OSM says the shore is made of, and each coastal cell's shore as the full-detail ground draws it.

The relief says where the sea is, but not what the land is like where it meets the water. OSM maps beaches and
sand (natural=beach, natural=sand), cliffs (natural=cliff, a line with the drop on its right) and mangroves and
other wetlands (natural=wetland) along much of Cuba's coast. `read` takes those lying within REACH of the coast
(contravia/coast.py) out of an extract, in local metres; `save` and `load` keep them in one small zstd file,
shore.src, which the game ships beside coast.src. ESA WorldCover's mangrove and wetland classes
(contravia/cover.py), where cover.src is there, add to OSM's.

`Shore` answers for a built island: which cells are near the sea, and for each of those its `Field`, the 2.5 m
grid of heights, normals, colours and types that contravia/rules/shore.py makes of the relief and these
features. The cells draw that grid (rules/ground.py), the wheels stand on it (contravia/surface.py) and the
mangroves and rocks grow from it. Fields are cached per process; the same cell gives the same field in each.
"""

import pickle
import threading
import weakref
from compression import zstd
from pathlib import Path

import numpy as np

from contravia.geo import CELL

VERSION = 1
NAME = "shore.src"  # the file's name, beside coast.src in the sources and in the built island
REACH = 300.0  # metres from the coast a feature may lie and still be kept
TOLERANCE = 2.0  # metres the features are simplified to
KINDS = ("beach", "cliff", "mangrove")
KEEP = 256  # coastal cells whose fields a process keeps


def kind_of(tags) -> str | None:
    """The shore kind an OSM object's tags make it: beach, cliff or mangrove (any wetland), or None."""
    natural = tags.get("natural")
    if natural in ("beach", "sand"):
        return "beach"
    if natural == "cliff":
        return "cliff"
    if natural == "wetland" or tags.get("wetland") == "mangrove":
        return "mangrove"
    return None


def read(pbf, land) -> tuple[dict, dict]:
    """({kind: shapely geometry in local metres}, report) from an OSM file: the beaches and wetlands as areas,
    the cliffs as lines, each kept only within REACH of the coast of land (a shapely geometry, coast.load)."""
    import osmium
    import shapely

    found = {k: [] for k in KINDS}
    wkb = osmium.geom.WKBFactory()

    class Handler(osmium.SimpleHandler):
        def way(self, w):  # cliffs are lines; an area is built from its way or relation by area()
            if kind_of(w.tags) == "cliff":
                try:
                    found["cliff"].append(shapely.from_wkb(wkb.create_linestring(w)))
                except (osmium.InvalidLocationError, RuntimeError):
                    pass

        def area(self, a):
            kind = kind_of(a.tags)
            if kind in ("beach", "mangrove"):
                try:
                    found[kind].append(shapely.from_wkb(wkb.create_multipolygon(a)))
                except (osmium.InvalidLocationError, RuntimeError):
                    pass

    Handler().apply_file(str(pbf), locations=True)
    from contravia.coast import _local

    edge = shapely.buffer(shapely.boundary(land), REACH, quad_segs=2) if not land.is_empty else None
    out, report = {}, {}
    for kind in KINDS:
        geoms = np.array(found[kind], dtype=object)
        if len(geoms) and edge is not None:
            geoms = _local(shapely.make_valid(geoms))
            shapely.prepare(edge)
            geoms = geoms[shapely.intersects(edge, geoms)]
            geom = shapely.intersection(shapely.union_all(geoms), edge) if len(geoms) else shapely.Polygon()
            geom = shapely.simplify(geom, TOLERANCE, preserve_topology=True)
        else:
            geom = shapely.Polygon()
        out[kind] = geom
        report[kind] = {"found": len(found[kind]), "kept": len(geoms) if edge is not None else 0,
                        "km" if kind == "cliff" else "km2":
                        round(float(shapely.length(geom)) / 1000, 1) if kind == "cliff"
                        else round(float(shapely.area(geom)) / 1e6, 1)}
    return out, report


def save(features: dict, path) -> int:
    """Write the features to path; returns its size in bytes."""
    import shapely

    body = {"version": VERSION, **{k: shapely.to_wkb(features[k]) for k in KINDS}}
    data = zstd.compress(pickle.dumps(body), level=19)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return len(data)


def load(path) -> dict:
    """The features saved at path, {kind: shapely geometry}."""
    import shapely

    body = pickle.loads(zstd.decompress(Path(path).read_bytes()))
    if body.get("version") != VERSION:
        raise ValueError(f"{path}: shore version {body.get('version')}, this game reads {VERSION}")
    return {k: shapely.from_wkb(body[k]) for k in KINDS}


class Shore:
    """The shore of a world: OSM's features (from shore.src, if its folder has one), WorldCover's classes (from
    cover.src, if there) and its terrain; `field(i, j)` makes and keeps a coastal cell's Field."""

    def __init__(self, terrain, features=None, cover=None, built=None):
        import shapely

        self.terrain, self.cover, self.built = terrain, cover, built
        self.features = {k: g for k, g in (features or {}).items() if g is not None and not g.is_empty}
        self._trees = {k: shapely.STRtree(shapely.get_parts(g)) for k, g in self.features.items()}
        self._coastal: dict[tuple[int, int], bool] = {}
        self._fields: dict[tuple[int, int], object] = {}
        self._lock = threading.Lock()

    def near(self, kind: str, box):
        """The parts of OSM's kind of feature within box (a shapely geometry), as one geometry or None."""
        import shapely

        tree = self._trees.get(kind)
        if tree is None:
            return None
        hit = tree.query(box)
        return shapely.union_all(tree.geometries.take(hit)) if len(hit) else None

    def coastal(self, i: int, j: int) -> bool:
        """Whether the sea comes within rules.shore.REACH of cell (i, j): only those cells draw a shore."""
        from contravia.rules import shore as rules

        key = (i, j)
        if key not in self._coastal:
            self._coastal[key] = rules.coastal(self.terrain, i, j)
        return self._coastal[key]

    def field(self, i: int, j: int):
        """The Field of cell (i, j) (rules.shore.field), or None for a cell away from the sea."""
        from contravia.rules import shore as rules

        key = (i, j)
        if key in self._fields:
            return self._fields[key]
        f = rules.field(self, i, j) if self.coastal(i, j) else None
        with self._lock:
            if len(self._fields) >= KEEP:
                self._fields.pop(next(iter(self._fields)))
            self._fields[key] = f
        return f

    def height(self, x, z) -> np.ndarray:
        """The drawn ground's height at points (x, z): the shore's where a cell has one, the terrain elsewhere."""
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        out = self.terrain.height(x, z).astype(float)
        ci, cj = np.floor(x / CELL).astype(np.int64), np.floor(z / CELL).astype(np.int64)
        for i, j in set(zip(ci.tolist(), cj.tolist(), strict=True)):
            f = self.field(i, j)
            if f is not None:
                k = (ci == i) & (cj == j)
                out[k] = f.height(x[k], z[k])
        return out


_shores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def of(city) -> Shore | None:
    """The Shore of a World with real heights (cached per world), or None for anything else: a single province
    being prepared, a flat island or a test city keep the ground as it was."""
    from contravia.rules.ground import relief

    key = city
    if key not in _shores:
        folder = getattr(city, "folder", None)
        if folder is None or not hasattr(city, "provinces_of") or not relief(city.terrain):
            _shores[key] = None
        else:
            from contravia import cover

            features = load(folder / NAME) if (folder / NAME).exists() else None
            grid = cover.load(folder / cover.NAME) if (folder / cover.NAME).exists() else None
            _shores[key] = Shore(city.terrain, features, grid, built=city)
    return _shores[key]
