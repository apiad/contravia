"""Regions: the generation parameters that change by place, read from contravia/regions.toml.

A region is a shape (where), a priority and the groups of parameters it changes; it leaves out what it does not
change. Each item a cell builds (a building or a filler box) takes its parameters from four layers, later layers
winning field by field and, inside a layer, the higher priority winning:

1. the defaults, Params() below;
2. the municipio regions, matched by the municipio name each building already carries;
3. the tag regions (where = { tag = "k=v" }), polygons from regions.src;
4. the named regions: barrio, OSM id, place or polygon.

regions.src, beside coast.src in the sources and in the built island, holds the OSM polygons the file names by tag
or OSM id, every landuse=cemetery included, in the island's metres: `contravia regions <pbf>` writes it (`read`,
`save`). Without it, the regions it would place are left out; with it, a tag or id it lacks stops the load.

A field is replaced whole: a weights table is never mixed with another. The merged Params are cached by the set
of regions that made them, so the thousands of houses of a cell share a handful of objects.

Polygons are projected into the island's metres (geo.to_local) when the file loads and kept, part by part (a
tag's hundreds of cemeteries are one region), in one STRtree. A
cell asks the tree once for the polygons that touch the box around its items: when none has an edge inside that
box, each polygon covers all of the cell or none of it, and the whole cell takes one set of regions (the fast
path). Otherwise each item's centroid is tested against the polygons that cross the box, one vectorised
shapely.contains_xy call per polygon.

A region's blocks group takes filler houses (blocks.fill, read by clearance.boxes) and the parks contravia
prepare laid (blocks.parks, read through `parks` by the cells, the walls, the forest and the grass) away from it.

Every process that builds cells loads the file once, on first use. Under contravia dev a reload imports this
module afresh and starts new builder processes, so a reload always reads the file again.
"""

import os
import pickle
import tomllib
from compression import zstd
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import shapely

from contravia.geo import CELL

PATH = Path(__file__).with_name("regions.toml")
NAME = "regions.src"  # the OSM polygons the file names, beside coast.src in the sources and in the built island
VERSION = 1
TAGS = ("landuse=cemetery",)  # tags regions.src always holds, whether the file names them or not
TOLERANCE = 0.5  # metres the polygons of regions.src are simplified to
SHAPES = ("municipio", "barrio", "osm", "tag", "place", "polygon")  # the keys of where
LAYER = {"municipio": 1, "tag": 2, "barrio": 3, "osm": 3, "place": 3, "polygon": 3}
BUILDINGS = ("types", "storeys", "palette", "wear", "portal")  # the keys of a region's buildings group
BLOCKS = {"fill": ("houses", "none"), "parks": ("keep", "drop")}  # the keys of a region's blocks group, and values
MARGIN = 100.0  # metres around a cell's square that its items' footprints may reach
CASE = ("kind", "above", "below", "types", "storeys")  # the keys of one case of buildings.types


class RegionError(ValueError):
    """A regions file that cannot be used, with the region's name and the key at fault."""


@dataclass(frozen=True)
class Case:
    """One case of buildings.types: when the building's OSM kind is `kind` and its footprint is larger than
    `above` m² and smaller than `below` (each left out to match any), its type is drawn from `types`, a tuple
    of (type, weight)."""

    types: tuple[tuple[str, float], ...]
    kind: str | None = None
    above: float | None = None
    below: float | None = None
    storeys: tuple[int, int] | None = None  # the storeys this case draws, over the type's and the region's

    def matches(self, kind: str, area: float) -> bool:
        """Whether a building of this OSM kind and footprint takes this case."""
        return ((self.kind is None or kind == self.kind) and (self.above is None or area > self.above)
                and (self.below is None or area < self.below))

    def draw(self, rng) -> str:
        """The type: the only one without a draw, else one rng.random() over the weights in the file's order."""
        if len(self.types) == 1:
            return self.types[0][0]
        r = rng.random() * sum(w for _, w in self.types)
        for name, w in self.types:
            if r < w:
                return name
            r -= w
        return self.types[-1][0]


@dataclass(frozen=True)
class Params:
    """The generation parameters of one item, merged from its regions. Overrides left None keep the type's row."""

    types: tuple[Case, ...] = (Case((("prefab", 1.0),), kind="apartments"), Case((("suburban", 1.0),)))
    storeys: tuple[int, int] | None = None  # over the type's row, for a type a region chose
    palette: tuple[str, ...] | None = None
    wear: tuple[float, float] | None = None
    portal: bool = False  # eclectic and colonial buildings of two storeys or more stand on columns by main roads
    fill: str = "houses"  # what a block left empty gets: filler houses, or nothing
    parks: str = "keep"  # whether the parks contravia prepare laid stand, or are dropped
    regions: tuple[str, ...] = ()  # the names of the regions merged into these, lowest layer first

    def pick(self, kind: str, area: float) -> Case:
        """The first case of types this building takes; the last case matches every building."""
        return next(c for c in self.types if c.matches(kind, area))


@dataclass(frozen=True)
class Region:
    """One [[region]] of the file: its name, shape, priority and the fields it sets, as Params field values."""

    name: str
    shape: str
    value: object
    priority: int
    fields: dict = field(hash=False)
    order: int = 0  # its place in the file, which breaks ties of layer and priority

    @property
    def rank(self) -> tuple:
        return LAYER[self.shape], self.priority, self.order


def _fail(region: str, key: str, why: str):
    raise RegionError(f"region {region!r}, {key}: {why}")


def _pair(name, key, v, kind):
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and v[0] <= v[1]):
        _fail(name, key, "must be [low, high] with low <= high")
    return (kind(v[0]), kind(v[1]))


def _weights(name, key, v, types) -> tuple:
    if not isinstance(v, dict) or not v:
        _fail(name, key, "must be a table of weights such as { colonial = 0.7, eclectic = 0.3 }")
    for t, w in v.items():
        if t not in types:
            _fail(name, key, f"{t!r} is not a type ({', '.join(types)})")
        if not isinstance(w, (int, float)) or w <= 0:
            _fail(name, key, f"the weight of {t!r} must be a positive number")
    return tuple((t, float(w)) for t, w in v.items())


def _types(name, v, types) -> tuple[Case, ...]:
    """buildings.types: one weights table, or a list of cases whose last one has no condition."""
    if isinstance(v, dict):
        return (Case(_weights(name, "buildings.types", v, types)),)
    if not isinstance(v, list) or not v:
        _fail(name, "buildings.types", "must be a weights table or a list of cases")
    cases = []
    for k, c in enumerate(v):
        key = f"buildings.types[{k}]"
        if not isinstance(c, dict):
            _fail(name, key, "must be a table")
        for bad in set(c) - set(CASE):
            _fail(name, f"{key}.{bad}", f"unknown key (known: {', '.join(CASE)})")
        if "types" not in c:
            _fail(name, key, "needs types")
        for bound in ("above", "below"):
            if bound in c and not isinstance(c[bound], (int, float)):
                _fail(name, f"{key}.{bound}", "must be a number of m²")
        if "kind" in c and not isinstance(c["kind"], str):
            _fail(name, f"{key}.kind", "must be an OSM building kind")
        cases.append(Case(_weights(name, f"{key}.types", c["types"], types), c.get("kind"),
                          float(c["above"]) if "above" in c else None, float(c["below"]) if "below" in c else None,
                          _pair(name, f"{key}.storeys", c["storeys"], int) if "storeys" in c else None))
    if (cases[-1].kind, cases[-1].above, cases[-1].below) != (None, None, None):
        _fail(name, "buildings.types", "the last case must have no kind, above or below, so every building takes one")
    return tuple(cases)


def _buildings(name, group, types) -> dict:
    if not isinstance(group, dict):
        _fail(name, "buildings", "must be a table")
    out = {}
    for key, v in group.items():
        if key == "types":
            out["types"] = _types(name, v, types)
        elif key == "storeys":
            out["storeys"] = _pair(name, "buildings.storeys", v, int)
            if out["storeys"][0] < 1:
                _fail(name, "buildings.storeys", "a building has at least one storey")
        elif key == "palette":
            if not (isinstance(v, list) and v and all(isinstance(c, str) and len(c) == 7 and c[0] == "#" for c in v)):
                _fail(name, "buildings.palette", 'must be a list of colours such as "#efece4"')
            out["palette"] = tuple(v)
        elif key == "wear":
            out["wear"] = _pair(name, "buildings.wear", v, float)
        elif key == "portal":
            if not isinstance(v, bool):
                _fail(name, "buildings.portal", "must be true or false")
            out["portal"] = v
        else:
            _fail(name, f"buildings.{key}", f"unknown key (known: {', '.join(BUILDINGS)})")
    return out


def _blocks(name, group, shape) -> dict:
    if not isinstance(group, dict):
        _fail(name, "blocks", "must be a table")
    out = {}
    for key, v in group.items():
        if key not in BLOCKS:
            _fail(name, f"blocks.{key}", f"unknown key (known: {', '.join(BLOCKS)})")
        if v not in BLOCKS[key]:
            _fail(name, f"blocks.{key}", f"must be one of {', '.join(map(repr, BLOCKS[key]))}")
        out[key] = v
    if shape == "municipio" and "parks" in out:  # a laid park carries no municipio; parks() places it by shape
        _fail(name, "blocks.parks", "a municipio region cannot set it: name the place by a shape")
    return out


def _polygon(name, where) -> shapely.Polygon:
    from contravia.geo import to_local

    pts = where["polygon"]
    if not (isinstance(pts, list) and len(pts) >= 3 and all(isinstance(p, list) and len(p) == 2 for p in pts)):
        _fail(name, "where.polygon", "must be three or more [lat, lon] points")
    ll = np.asarray(pts, float)
    poly = shapely.make_valid(shapely.Polygon(to_local(ll[:, 0], ll[:, 1])))
    if poly.is_empty or poly.area <= 0:
        _fail(name, "where.polygon", "encloses no area")
    return poly


def _place(name, where) -> shapely.Polygon:
    from contravia import collect

    radius = where.get("radius")
    if not isinstance(radius, (int, float)) or radius <= 0:
        _fail(name, "where.radius", "a place needs a radius in metres")
    found = [p for p in collect.load() if p.id == where["place"]]
    if not found:
        _fail(name, "where.place", f"no place {where['place']!r} in places.json")
    return shapely.Point(found[0].x, found[0].z).buffer(float(radius))


def _shape(name, where, src):
    """The shape's kind, its value, and its polygon (None for a municipio, matched by name)."""
    if not isinstance(where, dict):
        _fail(name, "where", f"must be a table with one of {', '.join(SHAPES)}")
    keys = [k for k in where if k in SHAPES]
    extra = set(where) - set(SHAPES) - ({"radius"} if "place" in where else set())
    if len(keys) != 1 or extra:
        _fail(name, "where", f"must name exactly one of {', '.join(SHAPES)}")
    shape = keys[0]
    if shape == "municipio":
        if not isinstance(where["municipio"], str) or not where["municipio"]:
            _fail(name, "where.municipio", "must be a municipio's name")
        return shape, where["municipio"], None
    if shape == "polygon":
        return shape, None, _polygon(name, where)
    if shape == "place":
        return shape, where["place"], _place(name, where)
    # barrio, osm and tag read their polygons from regions.src, which `contravia regions <pbf>` writes
    value = where[shape]
    if src is None:
        return shape, value, None
    found = src.get(f"{shape}:{value}")
    if found is None:
        _fail(name, f"where.{shape}", f"{value!r} resolves to no polygon in regions.src")
    return shape, value, found


def _key(names) -> frozenset:
    return frozenset(names)


class Regions:
    """The regions of one file: municipio regions by name, the others in an STRtree, and the merged Params."""

    def __init__(self, regions: list[Region], polygons: list[shapely.Geometry], version: str = ""):
        self.regions = regions
        self.version = version  # a hash of the file's bytes and the side file's time
        self.by_muni: dict[str, list[int]] = {}
        for k, r in enumerate(regions):
            if r.shape == "municipio":
                self.by_muni.setdefault(r.value, []).append(k)
        shaped = [k for k, r in enumerate(regions) if r.shape != "municipio"]  # aligned with polygons
        parts = [[g for g in shapely.get_parts(p) if g.geom_type == "Polygon" and g.area > 0] for p in polygons]
        self.shaped = [k for k, ps in zip(shaped, parts, strict=True) for _ in ps]  # each part's region
        self.polygons = np.array([g for ps in parts for g in ps], dtype=object)
        self.edges = np.array([p.boundary for p in self.polygons], dtype=object)
        for p in self.polygons:
            shapely.prepare(p)
        self.tree = shapely.STRtree(self.polygons) if len(self.polygons) else None
        self.dropping = any(regions[k].fields.get("parks") == "drop" for k in shaped)
        self.emptying = any(r.fields.get("fill") == "none" for r in regions)
        self._cache: dict[frozenset, Params] = {}

    @classmethod
    def load(cls, path: Path | str = PATH, src: dict | None = None, stamp: str = "") -> Regions:
        """Read and check a regions file. src maps "tag:k=v", "osm:way/…" and "barrio:…" to polygons in the
        island's metres (regions.src, `load_src`); a region whose shape src lacks stops the load, and without
        src such regions are left out. stamp (the side file's time) joins the version."""
        import hashlib

        data = Path(path).read_bytes()
        return cls.parse(data.decode(), src, hashlib.sha1(data + stamp.encode()).hexdigest()[:12])

    @classmethod
    def parse(cls, text: str, src: dict | None = None, version: str = "") -> Regions:
        """The regions of a TOML text, checked: every error names the region and the key."""
        from contravia.rules.facades import TYPES

        try:
            doc = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise RegionError(f"regions file: {e}") from None
        for bad in set(doc) - {"region"}:
            raise RegionError(f"regions file: unknown table {bad!r}")
        regions, polygons, names = [], [], set()
        for k, raw in enumerate(doc.get("region", [])):
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                raise RegionError(f"region #{k + 1}: needs a name")
            if name in names:
                _fail(name, "name", "two regions have this name")
            names.add(name)
            for bad in set(raw) - {"name", "where", "priority", "buildings", "blocks"}:
                _fail(name, bad, "unknown key (known: where, priority, buildings, blocks)")
            priority = raw.get("priority", 0)
            if not isinstance(priority, int):
                _fail(name, "priority", "must be a whole number")
            shape, value, poly = _shape(name, raw.get("where"), src)
            fields = _buildings(name, raw.get("buildings", {}), tuple(TYPES)) | _blocks(name, raw.get("blocks", {}), shape)
            if shape in ("barrio", "osm", "tag") and poly is None:  # no regions.src: nothing places it
                continue
            regions.append(Region(name, shape, value, priority, fields, k))
            if poly is not None:
                polygons.append(poly)
        return cls(regions, polygons, version)

    def municipios(self) -> set[str]:
        """The municipio names the file's regions match."""
        return set(self.by_muni)

    def check(self, municipios) -> None:
        """Stop with the region's name when a municipio region names no municipio of the data."""
        for muni, ks in self.by_muni.items():
            if muni not in municipios:
                _fail(self.regions[ks[0]].name, "where.municipio", f"{muni!r} is no municipio of the data")

    def params(self, regions: frozenset) -> Params:
        """The Params of a set of region indexes, merged by layer and priority and cached by the set."""
        p = self._cache.get(regions)
        if p is None:
            p = Params()
            for r in sorted((self.regions[k] for k in regions), key=lambda r: r.rank):
                p = replace(p, **r.fields)
            p = replace(p, regions=tuple(r.name for r in sorted((self.regions[k] for k in regions),
                                                                 key=lambda r: r.rank)))
            self._cache[regions] = p
        return p

    def municipio(self, muni: str | None) -> Params:
        """The Params of an item that only its municipio places."""
        return self.params(_key(self.by_muni.get(muni, ()) if muni else ()))

    def around(self, x0: float, z0: float, x1: float, z1: float) -> tuple[list[int], list[int]]:
        """The shaped regions that cover the whole box, and those whose edge crosses it."""
        if self.tree is None:
            return [], []
        box = shapely.box(x0, z0, x1, z1)
        hits = self.tree.query(box, predicate="intersects")
        if not len(hits):
            return [], []
        crossing = shapely.intersects(self.edges[hits], box)
        return [int(h) for h in hits[~crossing]], [int(h) for h in hits[crossing]]

    def assign(self, points: np.ndarray, munis: list) -> list[Params]:
        """The Params of each item, from its (x, z) centroid and its municipio's name (or None)."""
        points = np.asarray(points, float).reshape(-1, 2)
        if not len(points):
            return []
        whole, cut = self.around(*points.min(axis=0), *points.max(axis=0))
        return self._assign(whole, cut, lambda: points, munis)

    def _assign(self, whole, cut, points, munis) -> list[Params]:
        base = [self.shaped[h] for h in whole]
        if not cut:  # the fast path: one Params per municipio
            by = {m: self.params(_key(base + self.by_muni.get(m, []))) for m in set(munis)}
            return [by[m] for m in munis]
        pts = points()
        inside = [shapely.contains_xy(self.polygons[h], pts[:, 0], pts[:, 1]) for h in cut]
        return [self.params(_key(base + self.by_muni.get(m, []) + [self.shaped[h] for h, s in zip(cut, inside) if s[n]]))
                for n, m in enumerate(munis)]

    def cell(self, cell) -> tuple[list[Params], list[Params]]:
        """The Params of a cell's buildings and of its filler blocks. A building is placed by its footprint's
        mean corner and its municipio; a filler block by its ring's mean corner and the cell's most common
        municipio, which is what filler houses took before regions. The tree is asked once, for the cell's
        square grown by MARGIN (a footprint may reach past its cell); only when a region's edge crosses that
        square are the items' corners averaged and tested."""
        from contravia.rules.facades import municipio

        attrs = cell.building_attrs if cell.building_attrs is not None else [None] * len(cell.buildings)
        munis = [municipio(a, cell.muni_names) for a in attrs]
        zone = max(set(munis), key=munis.count) if munis else None
        rings = list(cell.buildings) + list(cell.fillers)
        if not rings:
            return [], []
        x, z = np.asarray(rings[0], float)[0] // CELL * CELL
        whole, cut = self.around(x - MARGIN, z - MARGIN, x + CELL + MARGIN, z + CELL + MARGIN)
        params = self._assign(whole, cut, lambda: np.array([np.asarray(r, float).mean(axis=0) for r in rings]),
                              munis + [zone] * len(cell.fillers))
        return params[:len(munis)], params[len(munis):]

    def filled(self, cell) -> list[bool]:
        """Whether each filler block of the cell gets its houses (blocks.fill), as clearance.boxes decides."""
        if not self.emptying:
            return [True] * len(cell.fillers)
        return [p.fill == "houses" for p in self.cell(cell)[1]]

    def kept(self, xz) -> np.ndarray:
        """Whether each (N, 2) point of a laid park stands where its regions keep parks (blocks.parks). Only
        shaped regions place a park, as a laid park carries no municipio."""
        xz = np.asarray(xz, float).reshape(-1, 2)
        if not self.dropping or not len(xz):
            return np.ones(len(xz), bool)
        return np.array([p.parks == "keep" for p in self.assign(xz, [None] * len(xz))], bool)

    def parks(self, cell):
        """The cell without the laid parks' motifs, walls and trees that stand where a region drops parks: each
        motif triangle and wall placed by its middle, each tree by its spot. The cell itself where none does."""
        if not self.dropping:
            return cell
        trees = self.kept(cell.trees)
        if cell.park_tris is None:
            return replace(cell, trees=cell.trees[trees]) if not trees.all() else cell
        tris = self.kept(np.asarray(cell.park_tris, float)[:, :, [0, 2]].mean(axis=1))
        walls = np.asarray(cell.park_walls, float).reshape(-1, 6)
        walls = self.kept((walls[:, 0:2] + walls[:, 2:4]) / 2)
        if trees.all() and tris.all() and walls.all():
            return cell
        return replace(cell, trees=cell.trees[trees], park_tris=cell.park_tris[tris], park_paint=cell.park_paint[tris],
                     park_walls=cell.park_walls[walls])


def read(pbf, names=()) -> tuple[dict, dict]:
    """({"tag:k=v" or "osm:way/…": shapely geometry in local metres}, report) from an OSM file: every area with
    one of TAGS or of the tags in names, and every way or relation names gives by id (a list of "tag:k=v" and
    "osm:way/…" keys, as `named` gives)."""
    import osmium

    from contravia.coast import _local

    tags = {k.split(":", 1)[1] for k in names if k.startswith("tag:")} | set(TAGS)
    tags = {tuple(t.split("=", 1)) for t in tags}
    ids = {k.split(":", 1)[1] for k in names if k.startswith("osm:")}
    found: dict[str, list] = {f"tag:{k}={v}": [] for k, v in tags} | {f"osm:{i}": [] for i in ids}
    wkb = osmium.geom.WKBFactory()

    class Handler(osmium.SimpleHandler):
        def area(self, a):
            osm = f"{'way' if a.from_way() else 'relation'}/{a.orig_id()}"
            keys = [f"tag:{k}={v}" for k, v in tags if a.tags.get(k) == v] + ([f"osm:{osm}"] if osm in ids else [])
            if keys:
                try:
                    g = shapely.from_wkb(wkb.create_multipolygon(a))
                except (osmium.InvalidLocationError, RuntimeError):
                    return
                for key in keys:
                    found[key].append(g)

    Handler().apply_file(str(pbf), locations=True)
    out, report = {}, {}
    for key, geoms in found.items():
        if not geoms:
            report[key] = {"found": 0}
            continue
        g = shapely.simplify(_local(shapely.make_valid(shapely.union_all(geoms))), TOLERANCE, preserve_topology=True)
        out[key] = g
        report[key] = {"found": len(geoms), "parts": len(shapely.get_parts(g)), "ha": round(g.area / 1e4, 1)}
    return out, report


def named(path: Path | str = PATH) -> list[str]:
    """The "tag:k=v" and "osm:way/…" keys the regions of a file name, which regions.src must hold."""
    doc = tomllib.loads(Path(path).read_text())
    return sorted({f"{k}:{r['where'][k]}" for r in doc.get("region", []) for k in ("tag", "osm")
                   if isinstance(r.get("where"), dict) and k in r["where"]})


def save(shapes: dict, path) -> int:
    """Write the shapes to path; returns its size in bytes."""
    body = {"version": VERSION, "shapes": {k: shapely.to_wkb(g) for k, g in shapes.items()}}
    data = zstd.compress(pickle.dumps(body), level=19)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return len(data)


def load_src(path) -> dict:
    """The shapes saved at path, {"tag:k=v" or "osm:way/…": shapely geometry}."""
    body = pickle.loads(zstd.decompress(Path(path).read_bytes()))
    if body.get("version") != VERSION:
        raise ValueError(f"{path}: regions version {body.get('version')}, this game reads {VERSION}")
    return {k: shapely.from_wkb(g) for k, g in body["shapes"].items()}


_current: Regions | None = None


def current() -> Regions:
    """This process's regions, read on first use from regions.toml and, where there is one, the regions.src of the
    island HAVANA_DATA names (the folder, or a province file's folder)."""
    global _current
    if _current is None:
        data = Path(os.environ.get("HAVANA_DATA", "data/cuba"))
        side = (data if data.is_dir() else data.parent) / NAME
        if side.exists():
            _current = Regions.load(src=load_src(side), stamp=str(side.stat().st_mtime_ns))
        else:
            _current = Regions.load()
    return _current
