"""prepare: read an OSM extract once and write the city as data, grouped by cell.

Reading OSM is slow and needs pyosmium and shapely, so it happens once, ahead of play. The output keeps
only what the rules need: road centrelines with a width, building footprints with their OSM id, the city
blocks that OSM left empty (they get filler boxes), and tree points.
"""

from dataclasses import dataclass, field

import numpy as np

from contravia import buildingtags
from contravia.citydata import CityData
from contravia.geo import HAVANA_BOX, cell_of, to_local
from contravia.graph import build_graph
from contravia.terrain import Terrain, clean, read_tiles

# Road width in metres by OSM highway class; anything not listed is not drivable and is dropped.
WIDTHS = {
    "motorway": 14.0, "trunk": 10.0, "primary": 10.0, "secondary": 8.0,
    "tertiary": 6.0, "unclassified": 6.0, "residential": 6.0, "living_street": 6.0, "service": 4.0,
}
for _k in ("motorway", "trunk", "primary", "secondary", "tertiary"):
    WIDTHS[f"{_k}_link"] = WIDTHS[_k]

ROAD_TAGS = ("oneway", "lanes", "maxspeed", "junction", "bridge", "tunnel", "layer")  # what the graph reads from a road
CONTROLS = {"traffic_signals": "signal", "stop": "stop", "give_way": "give_way"}

MAX_SEGMENT = 50.0  # metres; longer segments are split so each piece lands in the cell it crosses
MAX_BLOCK = 30_000.0  # m², 3 ha: bigger faces of the road network are not city blocks
TREE_EVERY = 150.0  # m² of green area per scattered tree

GREEN = {("leisure", "park"), ("landuse", "forest"), ("landuse", "grass"), ("landuse", "recreation_ground"),
         ("landuse", "meadow"), ("landuse", "village_green"), ("natural", "wood"), ("natural", "scrub"),
         ("natural", "grassland")}
NOT_BUILT = GREEN | {("natural", "water"), ("landuse", "reservoir"), ("landuse", "basin"),
                     ("landuse", "farmland"), ("landuse", "farmyard"), ("landuse", "orchard"),
                     ("waterway", "riverbank")}


@dataclass
class Raw:
    """What read_osm keeps, still in (lat, lon) degrees."""

    roads: list = field(default_factory=list)  # (highway, [(lat, lon), ...])
    buildings: list = field(default_factory=list)  # (osm_id, [(lat, lon), ...]) outer ring
    building_tags: dict = field(default_factory=dict)  # osm_id -> the tags buildingtags.TAGS keeps
    green: list = field(default_factory=list)  # (osm_id, ring) areas that get scattered trees
    not_built: list = field(default_factory=list)  # rings of areas that never get filler
    trees: list = field(default_factory=list)  # (lat, lon)
    boundary: list = field(default_factory=list)  # outer rings of the province
    ways: list = field(default_factory=list)  # (way id, highway, node ids, [(lat, lon), ...], tags) per road
    controls: dict = field(default_factory=dict)  # node id -> "signal", "stop" or "give_way"
    restrictions: list = field(default_factory=list)  # (from way, via node, to way, kind)
    municipios: list = field(default_factory=list)  # (name, outer ring) of admin_level 6 boundaries


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
            if (tags.get("boundary") == "administrative" and tags.get("admin_level") == "4"
                    and tags.get("name") == boundary and not a.from_way()):
                raw.boundary.extend(rings)
            if (tags.get("boundary") == "administrative" and tags.get("admin_level") == "6"
                    and not a.from_way() and "name" in tags):
                raw.municipios.extend((tags["name"], r) for r in rings)

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
    """The pavements and the buildings' platforms, over the streets."""
    from contravia import walks

    return walks.add_to(city)


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
    roads = _split_all(road_xz, [WIDTHS[hw] for hw, _ in raw.roads]) if road_xz else np.zeros((0, 5))
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
    blocks = [b for b in polygonize(unary_union(lines)) if b.area < MAX_BLOCK] if lines else []
    blocks = [b for b in blocks if province is None or province.contains(b.representative_point())]
    no_build = [Polygon(_local(r)) for r in raw.not_built if len(r) >= 4]
    no_build = [p if p.is_valid else p.buffer(0) for p in no_build]
    fillers, seeds = choose_fillers(blocks, cents, no_build, shapely.STRtree(no_build))

    # Trees: tagged points, plus one per TREE_EVERY m² of green, seeded by the area's id.
    trees = [_local(raw.trees)] if raw.trees else []
    for osm_id, ring in raw.green:
        poly = Polygon(_local(ring))
        if not poly.is_valid or poly.area < TREE_EVERY:
            continue
        n = int(poly.area // TREE_EVERY)
        rng = np.random.default_rng(abs(int(osm_id)))
        x0, z0, x1, z1 = poly.bounds
        got = []
        while sum(len(g) for g in got) < n:
            p = rng.uniform((x0, z0), (x1, z1), size=(2 * n, 2))
            got.append(p[shapely.contains_xy(poly, p[:, 0], p[:, 1])])
        trees.append(np.vstack(got)[:n])
    trees = np.vstack(trees) if trees else np.zeros((0, 2))
    trees = trees[inside(trees)]

    city = CityData.from_features(roads, rings, np.array(ids, np.int64), fillers, np.array(seeds, np.int64),
                                  trees, terrain)
    graph = build_graph(raw.ways, raw.controls, raw.restrictions, raw.municipios, rings,
                        lambda lat, lon: to_local(lat, lon), WIDTHS, inside)
    city.a.update(graph)
    city.a.update(buildingtags.encode(city, raw.building_tags, raw.municipios))
    info = {"roads": len(raw.roads), "segments": roads, "buildings": len(rings), "buildings_skipped": skipped,
            "blocks": len(blocks), "filler_blocks": len(fillers), "trees": len(trees), "centres": cents,
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


def main(pbf, out, dem=None) -> dict:
    """Read the extract and, from the GLO-30 tiles in dem, the cleaned heights; write the file and the report."""
    import json
    import time

    t0 = time.perf_counter()
    terrain = None
    if dem:
        raw = read_tiles(dem, HAVANA_BOX)
        terrain = Terrain(clean(raw.heights), raw.origin, raw.step)
    city, report = build_city(read_osm(pbf), terrain)
    city.save(out)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(report, indent=2))
    return report
