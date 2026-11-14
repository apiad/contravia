"""prepare: read an OSM extract once and write the city as data, grouped by cell.

Reading OSM is slow and needs pyosmium and shapely, so it happens once, ahead of play. The output keeps
only what the rules need: road centrelines with a width, building footprints with their OSM id, the city
blocks that OSM left empty (they get filler boxes), and tree points.
"""

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from havana.citydata import CityData
from havana.geo import cell_of, to_local

# Road width in metres by OSM highway class; anything not listed is not drivable and is dropped.
WIDTHS = {
    "motorway": 14.0, "trunk": 10.0, "primary": 10.0, "secondary": 8.0,
    "tertiary": 6.0, "unclassified": 6.0, "residential": 6.0, "living_street": 6.0, "service": 4.0,
}
for _k in ("motorway", "trunk", "primary", "secondary", "tertiary"):
    WIDTHS[f"{_k}_link"] = WIDTHS[_k]

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
    green: list = field(default_factory=list)  # (osm_id, ring) areas that get scattered trees
    not_built: list = field(default_factory=list)  # rings of areas that never get filler
    trees: list = field(default_factory=list)  # (lat, lon)
    boundary: list = field(default_factory=list)  # outer rings of the province


def _tagged(tags, pairs) -> bool:
    return any(tags.get(k) == v for k, v in pairs)


def read_osm(path, boundary: str = "La Habana") -> Raw:
    """Read the features the rules need from an OSM file (.pbf or .osm)."""
    import osmium

    raw = Raw()

    class Handler(osmium.SimpleHandler):
        def node(self, n):
            if n.tags.get("natural") == "tree":
                raw.trees.append((n.location.lat, n.location.lon))

        def way(self, w):
            tags = w.tags
            hw = tags.get("highway")
            if hw in WIDTHS:
                pts = [(nd.lat, nd.lon) for nd in w.nodes if nd.location.valid()]
                if len(pts) >= 2:
                    raw.roads.append((hw, pts))
            if "building" in tags and w.is_closed():
                raw.buildings.append((w.id, [(nd.lat, nd.lon) for nd in w.nodes if nd.location.valid()]))

        def area(self, a):
            tags = a.tags
            rings = [[(n.lat, n.lon) for n in ring] for ring in a.outer_rings()]
            if not rings:
                return
            if "building" in tags and not a.from_way():
                raw.buildings.append((a.orig_id(), rings[0]))
            if _tagged(tags, GREEN):
                raw.green.extend((a.orig_id(), r) for r in rings)
            if _tagged(tags, NOT_BUILT):
                raw.not_built.extend(rings)
            if (tags.get("boundary") == "administrative" and tags.get("admin_level") == "4"
                    and tags.get("name") == boundary and not a.from_way()):
                raw.boundary.extend(rings)

    Handler().apply_file(str(path), locations=True)
    return raw


def _local(points) -> np.ndarray:
    p = np.asarray(points, float).reshape(-1, 2)
    return to_local(p[:, 0], p[:, 1])


def _split(line: np.ndarray, width: float) -> np.ndarray:
    """Cut a polyline into segments no longer than MAX_SEGMENT, as rows (ax, az, bx, bz, width)."""
    rows = []
    for a, b in itertools.pairwise(line):
        n = max(1, math.ceil(np.hypot(*(b - a)) / MAX_SEGMENT))
        t = np.linspace(0, 1, n + 1)[:, None]
        pts = a + (b - a) * t
        rows.append(np.hstack([pts[:-1], pts[1:], np.full((n, 1), width)]))
    return np.vstack(rows)


def _block_seed(c) -> int:
    """A block's seed from its centroid rounded to the metre, so the same block gets the same boxes."""
    return round(c.x) * 1_000_003 + round(c.y)


def build_city(raw: Raw) -> tuple[CityData, dict]:
    """Project, clip to the province and assemble the city data and its report."""
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
    lines, seg_rows = [], []
    for hw, pts in raw.roads:
        line = _local(pts)
        lines.append(LineString(line))
        seg_rows.append(_split(line, WIDTHS[hw]))
    roads = np.vstack(seg_rows) if seg_rows else np.zeros((0, 5))
    roads = roads[inside((roads[:, 0:2] + roads[:, 2:4]) / 2)]

    # Buildings: skip rings shapely calls invalid or with fewer than three distinct points.
    rings, ids, skipped = [], [], 0
    for osm_id, pts in raw.buildings:
        ring = _local(pts)
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if len(np.unique(ring, axis=0)) < 3 or not Polygon(ring).is_valid:
            skipped += 1
            continue
        rings.append(ring)
        ids.append(osm_id)
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
    nb_tree = shapely.STRtree(no_build)
    fillers, seeds = [], []
    for b in blocks:
        if len(cents) and shapely.contains_xy(b, cents[:, 0], cents[:, 1]).any():
            continue
        if any(b.intersection(no_build[k]).area > 1.0 for k in nb_tree.query(b)):
            continue
        fillers.append(np.asarray(b.exterior.coords)[:-1])
        seeds.append(_block_seed(b.centroid))

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
                                  trees)
    road_cells = {tuple(c) for c in cell_of((roads[:, 0:2] + roads[:, 2:4]) / 2).tolist()}
    built_cells = {tuple(c) for c in cell_of(cents).tolist()}
    sizes = {tuple(c): 0 for c in city.cells().tolist()}
    for c in city.cells().tolist():
        cell = city.cell(*c)
        sizes[tuple(c)] = 4 * len(cell.roads) + sum(len(r) for r in cell.buildings) + 4 * len(cell.trees)
    largest = max(sizes.items(), key=lambda kv: kv[1]) if sizes else ((0, 0), 0)
    report = {
        "roads": len(raw.roads),
        "segments": len(roads),
        "buildings": len(rings),
        "buildings_skipped": skipped,
        "blocks": len(blocks),
        "filler_blocks": len(fillers),
        "trees": len(trees),
        "cells": len(city.cells()),
        "largest_cell": {"cell": list(largest[0]), "vertices": largest[1]},
        "cells_with_roads_no_buildings": round(len(road_cells - built_cells) / max(1, len(road_cells)), 3),
    }
    return city, report


def main(pbf, out) -> dict:
    """Read the extract, write the city data file and print the report."""
    import json
    import time

    t0 = time.perf_counter()
    city, report = build_city(read_osm(pbf))
    city.save(out)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(report, indent=2))
    return report
