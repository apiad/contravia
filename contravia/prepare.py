"""prepare: read an OSM extract once and write the city as data, grouped by cell.

Reading OSM is slow and needs pyosmium and shapely, so it happens once, ahead of play. The output keeps
only what the rules need: road centrelines with a width, building footprints with their OSM id, the city
blocks that OSM left empty (they get filler boxes, or become parks: contravia.parks), and tree points.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from contravia import buildingtags, parks
from contravia.citydata import CityData
from contravia.geo import CELL, HAVANA_BOX, cell_of, to_local
from contravia.graph import build_graph, road_width
from contravia.rules.trees import TREE_EVERY  # m² of green per tree; the trees grow when their cell opens
from contravia.terrain import Terrain

# Road width in metres by OSM highway class; anything not listed is not drivable and is dropped.
WIDTHS = {
    "motorway": 14.0, "trunk": 10.0, "primary": 10.0, "secondary": 8.0,
    "tertiary": 6.0, "unclassified": 6.0, "residential": 6.0, "living_street": 6.0, "service": 4.0,
}
for _k in ("motorway", "trunk", "primary", "secondary", "tertiary"):
    WIDTHS[f"{_k}_link"] = WIDTHS[_k]

ROAD_TAGS = ("oneway", "lanes", "maxspeed", "junction", "bridge", "tunnel", "layer", "name")  # what the graph reads from a road
CONTROLS = {"traffic_signals": "signal", "stop": "stop", "give_way": "give_way"}
PLACES = ("suburb", "neighbourhood", "quarter")  # place values that name a barrio

MAX_SEGMENT = 50.0  # metres; longer segments are split so each piece lands in the cell it crosses
MAX_BLOCK = 30_000.0  # m², 3 ha: bigger faces of the road network are not city blocks
GREEN_ACROSS = 15.0  # metres: a green area narrower than this (2 area / perimeter) gets no scattered trees

GREEN = {("leisure", "park"), ("landuse", "forest"), ("landuse", "grass"), ("landuse", "recreation_ground"),
         ("landuse", "meadow"), ("landuse", "village_green"), ("natural", "wood"), ("natural", "scrub"),
         ("natural", "grassland")}
NOT_BUILT = GREEN | {("natural", "water"), ("landuse", "reservoir"), ("landuse", "basin"),
                     ("landuse", "farmland"), ("landuse", "farmyard"), ("landuse", "orchard"),
                     ("waterway", "riverbank")}
NO_PARK = {("landuse", v) for v in ("cemetery", "industrial", "military", "railway", "construction", "commercial",
                                    "retail", "port", "garages", "brownfield", "landfill", "quarry")} | {
    ("amenity", v) for v in ("parking", "grave_yard", "school", "hospital", "university", "college")} | {
    ("leisure", v) for v in ("pitch", "stadium", "sports_centre", "track", "golf_course", "marina")} | {
    ("aeroway", v) for v in ("aerodrome", "apron", "runway", "taxiway")} | {("military", "barracks")}
NO_PARK_SHARE = 0.3  # a block this much covered by such land is never made a park


@dataclass
class Raw:
    """What read_osm keeps, still in (lat, lon) degrees."""

    roads: list = field(default_factory=list)  # (highway, [(lat, lon), ...])
    buildings: list = field(default_factory=list)  # (osm_id, [(lat, lon), ...]) outer ring
    building_tags: dict = field(default_factory=dict)  # osm_id -> the tags buildingtags.TAGS keeps
    green: list = field(default_factory=list)  # (osm_id, ring) areas that get scattered trees
    not_built: list = field(default_factory=list)  # rings of areas that never get filler
    no_park: list = field(default_factory=list)  # rings of land that is never made a park (NO_PARK)
    trees: list = field(default_factory=list)  # (lat, lon)
    boundary: list = field(default_factory=list)  # outer rings of the province
    ways: list = field(default_factory=list)  # (way id, highway, node ids, [(lat, lon), ...], tags) per road
    controls: dict = field(default_factory=dict)  # node id -> "signal", "stop" or "give_way"
    restrictions: list = field(default_factory=list)  # (from way, via node, to way, kind)
    municipios: list = field(default_factory=list)  # (name, outer ring) of admin_level 6 boundaries
    places: list = field(default_factory=list)  # (name, outer ring) of place=suburb|neighbourhood|quarter areas
    place_points: list = field(default_factory=list)  # (name, (lat, lon), place) of such places mapped as points
    consejos: list = field(default_factory=list)  # (name, outer ring) of consejos populares (admin_level 9)
    built_up: object = None  # contravia.builtup.BuiltUp over the province's box: where the satellite sees buildings


def _tagged(tags, pairs) -> bool:
    return any(tags.get(k) == v for k, v in pairs)


def read_osm(path, boundary: str = "La Habana", box=HAVANA_BOX, keep_ways: bool = True) -> Raw:
    """Read the features the rules need from an OSM file (.pbf or .osm).

    Anything with no point inside `box` is dropped while reading, so a country-wide extract never sits in
    memory as Python objects; build_city then clips to the province's real outline. With keep_ways False
    the traffic graph's inputs (ways, controls, restrictions) are not kept, which the island's tiles skip.
    """
    import osmium

    raw = Raw()
    s, w, n, e = box

    def near(pts) -> bool:
        return any(s <= lat <= n and w <= lon <= e for lat, lon in pts)

    class Handler(osmium.SimpleHandler):
        def node(self, n):
            if n.tags.get("natural") == "tree" and near([(n.location.lat, n.location.lon)]):
                raw.trees.append((n.location.lat, n.location.lon))
            if n.tags.get("place") in PLACES and "name" in n.tags and near([(n.location.lat, n.location.lon)]):
                raw.place_points.append((n.tags["name"], (n.location.lat, n.location.lon), n.tags["place"]))
            kind = CONTROLS.get(n.tags.get("highway")) if keep_ways else None
            if kind and near([(n.location.lat, n.location.lon)]):
                raw.controls[n.id] = kind

        def way(self, w):
            tags = w.tags
            hw = tags.get("highway")
            if hw in WIDTHS:
                valid = [nd for nd in w.nodes if nd.location.valid()]
                pts = [(nd.lat, nd.lon) for nd in valid]
                if len(pts) >= 2 and near(pts):
                    raw.roads.append((hw, pts))
                    if keep_ways:
                        raw.ways.append((w.id, hw, [nd.ref for nd in valid], pts,
                                         {k: tags.get(k) for k in ROAD_TAGS if k in tags}))
            if "building" in tags and w.is_closed():
                pts = [(nd.lat, nd.lon) for nd in w.nodes if nd.location.valid()]
                if near(pts):
                    raw.buildings.append((w.id, pts))
                    raw.building_tags[w.id] = buildingtags.kept(tags)

        def area(self, a):
            tags = a.tags
            rings = [[(n.lat, n.lon) for n in ring] for ring in a.outer_rings()]
            if not rings or not near([p for r in rings for p in r]):
                return
            if "building" in tags and not a.from_way():
                raw.buildings.append((a.orig_id(), rings[0]))
                raw.building_tags[a.orig_id()] = buildingtags.kept(tags)
            if _tagged(tags, GREEN):
                raw.green.extend((a.orig_id(), r) for r in rings)
            if _tagged(tags, NOT_BUILT):
                raw.not_built.extend(rings)
            if _tagged(tags, NO_PARK):
                raw.no_park.extend(rings)
            if (tags.get("boundary") == "administrative" and tags.get("admin_level") == "4"
                    and tags.get("name") == boundary and not a.from_way()):
                raw.boundary.extend(rings)
            if (tags.get("boundary") == "administrative" and tags.get("admin_level") == "6"
                    and not a.from_way() and "name" in tags):
                raw.municipios.extend((tags["name"], r) for r in rings)
            if tags.get("place") in PLACES and "name" in tags:
                raw.places.extend((tags["name"], r) for r in rings)
            if (tags.get("boundary") == "administrative" and tags.get("admin_level") == "9"
                    and not a.from_way() and "name" in tags):
                raw.consejos.extend((tags["name"], r) for r in rings)

        def relation(self, r):
            if not keep_ways or r.tags.get("type") != "restriction" or "restriction" not in r.tags:
                return
            got = {}
            for m in r.members:
                if m.role in ("from", "to") and m.type == "w" or m.role == "via" and m.type == "n":
                    got[m.role] = m.ref
            if len(got) == 3:
                raw.restrictions.append((got["from"], got["via"], got["to"], r.tags["restriction"]))

    Handler().apply_file(str(path), locations=True)
    return raw


def _local(points) -> np.ndarray:
    p = np.asarray(points, float).reshape(-1, 2)
    return to_local(p[:, 0], p[:, 1])


def _locals(items) -> list[np.ndarray]:
    """_local of each list of points, projected in one call."""
    arrs = [np.asarray(p, float).reshape(-1, 2) for p in items]
    if not arrs:
        return []
    return np.split(_local(np.concatenate(arrs)), np.cumsum([len(a) for a in arrs])[:-1])


def _split(line: np.ndarray, width: float) -> np.ndarray:
    """Cut a polyline into segments no longer than MAX_SEGMENT, as rows (ax, az, bx, bz, width)."""
    return _split_all([line], [width])


def _split_all(lines, widths) -> np.ndarray:
    """_split of every line, stacked in order, computed at once: the same numbers as np.linspace gives."""
    a = np.concatenate([ln[:-1] for ln in lines])
    b = np.concatenate([ln[1:] for ln in lines])
    w = np.repeat(np.asarray(widths, float), [len(ln) - 1 for ln in lines])
    n = np.maximum(1, np.ceil(np.hypot(*(b - a).T) / MAX_SEGMENT)).astype(np.int64)
    seg = np.repeat(np.arange(len(n)), n)
    j = np.arange(len(seg)) - np.repeat(np.cumsum(n) - n, n)
    step = 1.0 / n[seg]
    t0 = (j * step)[:, None]
    t1 = np.where(j + 1 == n[seg], 1.0, (j + 1) * step)[:, None]
    d = (b - a)[seg]
    return np.hstack([a[seg] + d * t0, a[seg] + d * t1, w[seg][:, None]])


def _closed(rings) -> np.ndarray:
    """Whether each ring's last point is np.allclose to its first."""
    if not rings:
        return np.zeros(0, bool)
    first = np.array([r[0] for r in rings])
    last = np.array([r[-1] for r in rings])
    return np.isclose(first, last).all(axis=1)


def _valid_rings(rings) -> np.ndarray:
    """Whether each ring has three distinct points and makes a polygon shapely calls valid, all at once."""
    import shapely

    ok = np.zeros(len(rings), bool)
    if not rings:
        return ok
    n = np.array([len(r) for r in rings])
    rid = np.repeat(np.arange(len(rings)), n)
    xz = np.concatenate(rings)
    distinct = np.bincount(np.unique(np.column_stack([rid, xz]), axis=0)[:, 0].astype(np.int64),
                           minlength=len(rings))
    three = distinct >= 3
    if three.any():
        sel = np.repeat(three, n)
        dense = (np.cumsum(three) - 1)[rid[sel]]
        polys = shapely.polygons(shapely.linearrings(xz[sel], indices=dense))
        ok[np.flatnonzero(three)] = shapely.is_valid(polys)
    return ok


def _block_seed(c) -> int:
    """A block's seed from its centroid rounded to the metre, so the same block gets the same boxes."""
    return round(c.x) * 1_000_003 + round(c.y)


def on_road(points: np.ndarray, roads: np.ndarray, clear: float = 1.0) -> np.ndarray:
    """For each (x, z) point, whether it stands within `clear` metres of a road strip's edge or on it."""
    import shapely

    out = np.zeros(len(points), bool)
    if not len(points) or not len(roads):
        return out
    lines = shapely.linestrings(np.asarray(roads[:, :4], float).reshape(-1, 2, 2))
    pts = shapely.points(points)
    pi, ri = shapely.STRtree(lines).query(pts, predicate="dwithin", distance=float(roads[:, 4].max()) / 2 + clear)
    hit = shapely.distance(pts[pi], lines[ri]) < roads[ri, 4] / 2 + clear
    out[pi[hit]] = True
    return out


def choose_fillers(blocks, cents: np.ndarray, no_build: list, nb_tree) -> tuple[list, list]:
    """The blocks that get filler boxes, as rings, and their seeds: no building centroid and no green inside."""
    import shapely

    blocks = np.asarray(blocks, dtype=object)
    free = np.ones(len(blocks), bool)
    if len(cents) and len(blocks):
        _, hit = shapely.STRtree(blocks).query(shapely.points(cents), predicate="within")
        free[hit] = False
    if len(no_build) and free.any():
        bi, ki = nb_tree.query(blocks[free])
        big = shapely.area(shapely.intersection(blocks[free][bi], np.asarray(no_build, dtype=object)[ki])) > 1.0
        free[np.flatnonzero(free)[bi[big]]] = False
    fillers = [np.asarray(b.exterior.coords)[:-1] for b in blocks[free]]
    seeds = [_block_seed(b.centroid) for b in blocks[free]]
    return fillers, seeds


def _rings_arrays(rings) -> tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate([[0], np.cumsum([len(r) for r in rings])]).astype(np.int64)
    verts = np.concatenate(rings).astype(np.float32) if rings else np.zeros((0, 2), np.float32)
    return verts, starts


def find_parks(raw: Raw, faces, rings, cents, graph, roads) -> tuple[list, list, list, list]:
    """The faces that become parks: the rings of their land (the face, whose edges are the roads' centrelines,
    less the roads' strips), their kinds (contravia.parks), zones (contravia.walks.zone_of) and the faces' seeds."""
    import shapely
    from shapely.geometry import Polygon

    from contravia.medians import ACROSS, boulevard
    from contravia.walks import CENTRE, VEDADO, zone_of

    if not faces:
        return [], [], [], []
    faces = np.asarray(faces, dtype=object)
    lands = _lands(faces, roads)
    built, count = np.zeros(len(faces)), np.zeros(len(faces), int)
    if len(cents):
        fi, bi = shapely.STRtree(faces).query(shapely.points(cents), predicate="within")[::-1]
        np.add.at(count, fi, 1)
        np.add.at(built, fi, [Polygon(rings[k]).area for k in bi])

    def cover(rings_):
        polys = [Polygon(_local(r)) for r in rings_ if len(r) >= 4]
        polys = [p if p.is_valid else p.buffer(0) for p in polys]
        out = np.zeros(len(faces))
        if polys:
            fi, pi = shapely.STRtree(np.asarray(polys, dtype=object)).query(faces, predicate="intersects")
            np.add.at(out, fi, shapely.area(shapely.intersection(faces[fi], np.asarray(polys, dtype=object)[pi])))
        return out

    green = cover([r for _, r in raw.green])
    blocked = cover(raw.no_park) > NO_PARK_SHARE * shapely.area(faces)
    names = [None] * len(faces)
    spots = shapely.point_on_surface(faces)
    for name, ring in raw.municipios:
        m = Polygon(_local(ring))
        if not m.is_valid:
            m = m.buffer(0)
        inside_m = shapely.contains(m, spots)
        for k in np.flatnonzero(inside_m):
            names[k] = name
    city = np.array([n in CENTRE or n in VEDADO for n in names])
    prado = _named_between(graph, faces, r"Paseo del Prado|Paseo de Mart[ií]", ACROSS)
    seen = raw.built_up.share(faces) if raw.built_up is not None else None
    kinds = parks.classify(lands, built, count, green, city, boulevard(graph, faces), prado, seen)
    kinds[blocked & ~prado] = parks.NONE
    kinds = parks.in_town(kinds, lands, city, raw.built_up)  # the other towns' parks are urban, by the satellite
    keep = np.flatnonzero(kinds != parks.NONE)
    return ([np.asarray(lands[k].exterior.coords)[:-1] for k in keep], [int(kinds[k]) for k in keep],
            [zone_of(names[k]) for k in keep], [_block_seed(faces[k].centroid) for k in keep])


def _lands(faces, roads) -> np.ndarray:
    """Each face less the strips of the roads round it, as wide as they are: its largest polygon (or the face, for
    one too small to be a park, which classify drops anyway)."""
    import shapely

    out = faces.copy()
    if not len(roads):
        return out
    strips = shapely.buffer(shapely.linestrings(np.asarray(roads[:, :4], float).reshape(-1, 2, 2)),
                            np.asarray(roads[:, 4], float) / 2, cap_style="round", quad_segs=2)
    tree = shapely.STRtree(strips)
    big = np.flatnonzero(shapely.area(faces) >= parks.MIN_AREA)
    fi, si = tree.query(faces[big], predicate="intersects")
    order = np.argsort(fi, kind="stable")
    fi, si = fi[order], si[order]
    cut, first = np.unique(fi, return_index=True)
    for f, group in zip(cut, np.split(si, first[1:]), strict=True):
        land = shapely.difference(faces[big[f]], shapely.union_all(strips[group]))
        parts = shapely.get_parts(land)
        parts = parts[shapely.get_type_id(parts) == 3]
        if len(parts):
            out[big[f]] = parts[np.argmax(shapely.area(parts))]
    return out


def _named_between(graph, faces, pattern, across) -> np.ndarray:
    """Which faces lie between the carriageways of the streets whose name matches pattern, like medians.boulevard."""
    import re

    import shapely
    from shapely.geometry import LineString

    out = np.zeros(len(faces), bool)
    if "g_names" not in graph:
        return out
    names = [str(n) for n in graph["g_names"]]
    hit = [k for k, n in enumerate(names) if re.search(pattern, n)]
    poly, starts = graph["g_poly"].astype(float), graph["g_poly_start"]
    lines = [LineString(poly[starts[e]:starts[e + 1]]) for e in np.flatnonzero(np.isin(graph["g_name"], hit))
             if starts[e + 1] - starts[e] >= 2]
    if not lines:
        return out
    near = shapely.union_all(lines).buffer(2.0)
    ext = shapely.get_exterior_ring(faces)
    share = shapely.length(shapely.intersection(ext, near)) / np.maximum(shapely.length(ext), 1e-9)
    wide = 2 * shapely.area(faces) / np.maximum(shapely.length(ext), 1e-9)
    return (share >= 0.5) & (wide < across)


def build_city(raw: Raw, terrain: Terrain | None = None) -> tuple[CityData, dict]:
    """Project, clip to the province and assemble the city data, with its heights, and its report."""
    city, info = base_city(raw, terrain)
    streets = add_streets(city) if info["edges"] else {}
    walks = add_walks(city) if info["edges"] else {}
    return city, report(city, info, {**streets, **walks})


def add_streets(city: CityData) -> dict:
    """The street surface, and the road's height at every graph point, so traffic need not ask each frame."""
    from contravia.streets import add_to

    return add_to(city, WIDTHS)


def add_walks(city: CityData) -> dict:
    """The pavements and the buildings' platforms, over the streets, and the parks' motifs on them."""
    from contravia import walks

    covered = parks.drop_covered(city)  # a park the street surface covers is asphalt: not paved, no motifs
    return {**covered, **walks.add_to(city), **parks.add_to(city)}


def green_pieces(green) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Green areas, as (polygon, seed) in local metres, cut along the cell grid and triangulated: (T, 3, 2)
    triangles, the seed of each and its (T, 2) cell."""
    import shapely

    tris, seeds, keys = [], [], []
    for poly, seed in green:
        if poly.is_empty:
            continue  # all of it was a park
        x0, z0, x1, z1 = poly.bounds
        ii, jj = np.meshgrid(np.arange(np.floor(x0 / CELL), np.floor(x1 / CELL) + 1),
                             np.arange(np.floor(z0 / CELL), np.floor(z1 / CELL) + 1))
        ii, jj = ii.ravel().astype(np.int64), jj.ravel().astype(np.int64)
        boxes = shapely.box(ii * CELL, jj * CELL, (ii + 1) * CELL, (jj + 1) * CELL)
        shapely.prepare(poly)
        pieces = shapely.intersection(boxes, poly)
        for k in np.flatnonzero(~shapely.is_empty(pieces) & (shapely.area(pieces) > 1.0)):
            for part in shapely.get_parts(pieces[k]):
                if part.geom_type != "Polygon" or part.area <= 1.0:
                    continue  # an edge or a corner the box only touches
                for t in shapely.get_parts(shapely.constrained_delaunay_triangles(part)):
                    c = shapely.get_coordinates(shapely.get_exterior_ring(t))[:3]
                    if len(c) == 3:
                        tris.append(c)
                        seeds.append(seed)
                        keys.append((ii[k], jj[k]))
    return (np.array(tris, np.float32).reshape(-1, 3, 2), np.array(seeds, np.int64),
            np.array(keys, np.int64).reshape(-1, 2))


def base_city(raw: Raw, terrain: Terrain | None = None) -> tuple[CityData, dict]:
    """The city data before its streets and walks: roads, buildings, fillers, trees, the road graph and the
    buildings' tags; and what report needs of the build: counts, the road segments and the building centres."""
    import shapely
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union

    if raw.boundary:
        province = unary_union([Polygon(_local(r)) for r in raw.boundary if len(r) >= 3])
    else:
        province = None

    def inside(xz: np.ndarray) -> np.ndarray:
        if province is None or len(xz) == 0:
            return np.ones(len(xz), bool)
        return shapely.contains_xy(province, xz[:, 0], xz[:, 1])

    # Roads: split, then keep the segments whose midpoint is in the province.
    road_xz = _locals(pts for _, pts in raw.roads)
    lines = [LineString(line) for line in road_xz]
    if len(raw.ways) == len(raw.roads):  # read with the ways: as wide as their lanes, like the drawn streets
        widths = [road_width(hw, tags, WIDTHS) for _, hw, _, _, tags in raw.ways]
    else:
        widths = [WIDTHS[hw] for hw, _ in raw.roads]
    roads = _split_all(road_xz, widths) if road_xz else np.zeros((0, 5))
    roads = roads[inside((roads[:, 0:2] + roads[:, 2:4]) / 2)]

    # Buildings: skip rings shapely calls invalid or with fewer than three distinct points.
    rings = _locals(pts for _, pts in raw.buildings)
    rings = [r[:-1] if len(r) > 1 and c else r
             for r, c in zip(rings, _closed(rings))]
    ok = _valid_rings(rings)
    skipped = int((~ok).sum())
    rings = [r for r, k in zip(rings, ok) if k]
    ids = [i for (i, _), k in zip(raw.buildings, ok) if k]
    cents = np.array([r.mean(axis=0) for r in rings]).reshape(-1, 2)
    keep = inside(cents)
    rings = [r for r, k in zip(rings, keep) if k]
    ids = [i for i, k in zip(ids, keep) if k]
    cents = cents[keep]

    # Blocks: the faces of the road network. Filler goes to small ones with no building and no green.
    faces = [b for b in polygonize(unary_union(lines)) if b.area <= parks.PARK_MAX] if lines else []
    faces = [b for b in faces if province is None or province.contains(b.representative_point())]
    blocks = [b for b in faces if b.area < MAX_BLOCK]
    no_build = [Polygon(_local(r)) for r in raw.not_built if len(r) >= 4]
    no_build = [p if p.is_valid else p.buffer(0) for p in no_build]
    fillers, seeds = choose_fillers(blocks, cents, no_build, shapely.STRtree(no_build))

    # Green: the areas that grow trees, kept as triangles per cell; the trees themselves are seeded when the
    # cell opens (rules.trees.in_green), so no tree is stored. OSM's tree points are left out: they are mapped
    # unevenly and stand in medians and on roads.
    green = []
    for osm_id, ring in raw.green:
        poly = Polygon(_local(ring))
        if not poly.is_valid or poly.area < TREE_EVERY or 2 * poly.area / poly.length < GREEN_ACROSS:
            continue  # a strip of grass in a median or along a road gets no trees
        if province is not None and not province.contains(poly):
            poly = poly.intersection(province)
        if not poly.is_empty:
            green.append((poly, abs(int(osm_id))))
    trees = np.zeros((0, 2))

    graph = build_graph(raw.ways, raw.controls, raw.restrictions, raw.municipios, rings,
                        lambda lat, lon: to_local(lat, lon), WIDTHS, inside,
                        places=(raw.places, raw.place_points, raw.consejos))

    # Parks: blocks with few or no buildings (contravia.parks). They get no filler, and their own trees.
    park_rings, kinds, park_zone, face_seeds = find_parks(raw, faces, rings, cents, graph, roads)
    park_seeds = [parks.seed_of(Polygon(r)) for r in park_rings]
    if park_rings:
        gone = set(face_seeds)
        keep = [k for k, sd in enumerate(seeds) if sd not in gone]
        fillers, seeds = [fillers[k] for k in keep], [seeds[k] for k in keep]
        polys = [Polygon(r) for r in park_rings]
        parks_all = shapely.union_all(polys)
        green = [(g.difference(parks_all), gid) if g.intersects(parks_all) else (g, gid) for g, gid in green]
        laid = parks.laid({"park_verts": _rings_arrays(park_rings)[0], "park_starts": _rings_arrays(park_rings)[1],
                           "park_kind": kinds, "park_seed": park_seeds, "park_zone": park_zone})
        trees = np.vstack([trees] + [p.trees for p, _ in laid if len(p.trees)])
        trees = trees[~on_road(trees, roads)]

    if terrain is not None and terrain.heights.size > 4:  # the relief past the Malecón's wall goes under the sea
        from contravia.landmarks import malecon

        malecon.carve(terrain, roads)
    city = CityData.from_features(roads, rings, np.array(ids, np.int64), fillers, np.array(seeds, np.int64),
                                  trees, terrain)
    city.a.update(graph)
    tris, gseeds, gkeys = green_pieces(green)
    city.add_kind("green_tris", tris, gkeys)
    city.add_kind("green_seeds", gseeds, gkeys)
    if park_rings:  # a park overlapping a landmark's ground is left to the landmark
        from contravia.landmarks import Landmarks

        clear = [Polygon(r).buffer(0) for r in Landmarks(city).clear if len(r) >= 3]
        if clear:
            hit = shapely.intersects(np.array([Polygon(r) for r in park_rings], dtype=object), shapely.union_all(clear))
            park_rings = [r for r, h in zip(park_rings, hit, strict=True) if not h]
            kinds = [k for k, h in zip(kinds, hit, strict=True) if not h]
            park_zone = [z for z, h in zip(park_zone, hit, strict=True) if not h]
            park_seeds = [s for s, h in zip(park_seeds, hit, strict=True) if not h]
            face_seeds = [s for s, h in zip(face_seeds, hit, strict=True) if not h]
    pv, ps = _rings_arrays(park_rings)
    city.a.update(park_verts=pv, park_starts=ps, park_kind=np.asarray(kinds, np.uint8).reshape(-1),
                  park_zone=np.asarray(park_zone, np.uint8).reshape(-1), park_seed=np.asarray(park_seeds, np.int64))
    city.a.update(buildingtags.encode(city, raw.building_tags, raw.municipios))
    info = {"roads": len(raw.roads), "segments": roads, "buildings": len(rings), "buildings_skipped": skipped,
            "blocks": len(blocks), "filler_blocks": len(fillers), "trees": len(trees), "centres": cents,
            "parks": {parks.KINDS[k]: int(np.sum(np.asarray(kinds) == k)) for k in range(1, len(parks.KINDS))},
            "edges": len(graph.get("g_from", ())), "junctions": len(graph.get("g_junction", ())),
            "turns": len(graph.get("g_turn_in", ()))}
    return city, info


def report(city: CityData, info: dict, street_report: dict) -> dict:
    """The build's report: base_city's counts, the cells, the heights and the streets' and walks' counts."""
    roads, cents = info["segments"], info["centres"]
    road_cells = {tuple(c) for c in cell_of((roads[:, 0:2] + roads[:, 2:4]) / 2).tolist()}
    built_cells = {tuple(c) for c in cell_of(cents).tolist()}
    sizes = {tuple(c): 0 for c in city.cells().tolist()}
    for c in city.cells().tolist():
        cell = city.cell(*c)
        sizes[tuple(c)] = 4 * len(cell.roads) + sum(len(r) for r in cell.buildings) + 4 * len(cell.trees)
    largest = max(sizes.items(), key=lambda kv: kv[1]) if sizes else ((0, 0), 0)
    return {
        **{k: v for k, v in info.items() if k not in ("segments", "centres", "edges", "junctions", "turns")},
        "segments": len(roads),
        "cells": len(city.cells()),
        "largest_cell": {"cell": list(largest[0]), "vertices": largest[1]},
        "cells_with_roads_no_buildings": round(len(road_cells - built_cells) / max(1, len(road_cells)), 3),
        "edges": info["edges"], "junctions": info["junctions"], "turns": info["turns"],
        "heights_m": [round(float(city.terrain.heights.min()), 1), round(float(city.terrain.heights.max()), 1)],
        **street_report,
    }


BOX_MARGIN = 0.03  # degrees around a province's outline that its reading and its relief cover
TERRAIN_STRIDE = {"La Habana": 1}  # every how many 30 m samples a province's relief keeps; 3 (90 m) elsewhere


def province_boxes(pbf) -> dict:
    """{name: (south, west, north, east)} of every province's outline (admin_level 4) in the extract, with
    BOX_MARGIN around it; one quick pass over the file."""
    import osmium

    boxes: dict = {}

    class Handler(osmium.SimpleHandler):
        def area(self, a):
            t = a.tags
            if t.get("boundary") != "administrative" or t.get("admin_level") != "4" or a.from_way():
                return
            pts = [(n.lat, n.lon) for ring in a.outer_rings() for n in ring]
            if pts:
                lat, lon = zip(*pts, strict=True)
                s, w, n, e = boxes.get(t.get("name"), (90.0, 180.0, -90.0, -180.0))
                boxes[t.get("name")] = (min(s, min(lat)), min(w, min(lon)), max(n, max(lat)), max(e, max(lon)))

    Handler().apply_file(str(pbf), locations=True)
    return {k: (s - BOX_MARGIN, w - BOX_MARGIN, n + BOX_MARGIN, e + BOX_MARGIN) for k, (s, w, n, e) in boxes.items()}


def _province(pbf, out, dem, name: str, slug: str, box, worldcover=None) -> dict:
    """Build one province into out/<slug>.npz and return its report; a worker process runs it when jobs > 1.

    The file is written under a temporary name and renamed when complete, so a crash never leaves a half-written
    province that resume would take as built."""
    import time

    from contravia.terrain import read_terrain

    t0 = time.perf_counter()
    terrain = read_terrain(dem, box, stride=TERRAIN_STRIDE.get(name, 3)) if dem else None
    raw = read_osm(pbf, boundary=name, box=box)
    if worldcover:
        from contravia import builtup

        raw.built_up = builtup.read(worldcover, box)
    city, report = build_city(raw, terrain)
    path = Path(out) / f"{slug}.npz"
    part = Path(out) / f"{slug}.part.npz"
    city.save(part)
    part.replace(path)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    return report


def main(pbf, out, dem=None, provinces=None, resume: bool = False, jobs: int = 1, worldcover=None) -> dict:
    """Build each province (all of contravia.geo.PROVINCES by default) into out/<slug>.npz: read its part of the
    extract and, from the GLO-30 tiles in dem (downloaded there if missing), its cleaned relief; with worldcover,
    a folder of WorldCover tiles (downloaded there if missing), where the satellite sees buildings. With resume, a
    province whose file is there is left as it is. jobs provinces are built at once, each in its own process
    (about 2.5 GB of memory each). Returns the reports by province."""
    import json
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from contravia import coast, shore
    from contravia.geo import PROVINCES

    boxes = province_boxes(pbf)
    reports = {}
    if not provinces or not (Path(out) / coast.NAME).exists():  # the map's coast, for the whole island
        land, reports["coast"] = coast.read(pbf)
        if not land.is_empty:
            reports["coast"]["bytes"] = coast.save(land, Path(out) / coast.NAME)
            features, reports["shore"] = shore.read(pbf, land)  # what the shore is made of, along that coast
            reports["shore"]["bytes"] = shore.save(features, Path(out) / shore.NAME)
        print("coast", json.dumps(reports["coast"]), flush=True)
    todo = []
    for name, slug in PROVINCES:
        if provinces and name not in provinces and slug not in provinces:
            continue
        if name not in boxes:
            raise ValueError(f"{pbf}: no province named {name!r} (admin_level 4)")
        if resume and (Path(out) / f"{slug}.npz").exists():
            continue
        todo.append((name, slug))
    Path(out).mkdir(parents=True, exist_ok=True)
    if jobs <= 1:
        for name, slug in todo:
            reports[name] = _province(pbf, out, dem, name, slug, boxes[name], worldcover)
            print(name, json.dumps(reports[name]), flush=True)
        return reports
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        running = {pool.submit(_province, pbf, out, dem, name, slug, boxes[name], worldcover): name
                   for name, slug in todo}
        for done in as_completed(running):
            name = running[done]
            reports[name] = done.result()
            print(name, json.dumps(reports[name]), flush=True)
    return reports
