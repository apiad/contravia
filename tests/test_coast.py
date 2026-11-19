import json
import shutil

import numpy as np
import pytest
import shapely

from contravia import coast
from contravia.geo import CELL, to_local
from contravia.rules.ground import WATER_Y
from contravia.summary import LIFT, summarize, summary_mesh

ISLAND = [(-82.40, 23.10), (-82.37, 23.10), (-82.37, 23.12), (-82.40, 23.12)]  # (lon, lat), counter-clockwise
LAGOON = [(-82.39, 23.105), (-82.39, 23.11), (-82.38, 23.11), (-82.38, 23.105)]  # clockwise: water in the land
CAY = [(-82.36, 23.10), (-82.359, 23.10), (-82.3585, 23.1005), (-82.359, 23.101), (-82.36, 23.101)]  # 6 points


def _osm(path):
    """An OSM file with the island in two coastline ways, a lagoon, a cay and a way that never closes."""
    nodes, ways, k = [], [], 0

    def way(points, close=True):
        nonlocal k
        ids = []
        for lon, lat in points:
            k += 1
            nodes.append(f'<node id="{k}" version="1" lat="{lat}" lon="{lon}"/>')
            ids.append(k)
        return ids + ([ids[0]] if close else [])

    island = way(ISLAND)
    halves = [island[:3], island[2:]]  # joined again at node island[2]
    rings = [*halves, way(LAGOON), way(CAY), way([(-82.30, 23.0), (-82.29, 23.0)], close=False)]
    for w, ids in enumerate(rings, start=1):
        refs = "".join(f'<nd ref="{i}"/>' for i in ids)
        ways.append(f'<way id="{w}" version="1">{refs}<tag k="natural" v="coastline"/></way>')
    path.write_text('<?xml version="1.0" encoding="UTF-8"?><osm version="0.6">' + "".join(nodes + ways) + "</osm>")
    return path


def _local(ring):
    lon, lat = np.array(ring).T
    return shapely.Polygon(to_local(lat, lon))


@pytest.fixture(scope="module")
def land(tmp_path_factory):
    return coast.read(_osm(tmp_path_factory.mktemp("coast") / "coast.osm"))


def test_read_joins_the_ways_keeps_the_land_and_leaves_out_what_stays_open(land):
    geom, report = land
    assert report["rings"] == 3 and report["open"] == 1
    want = _local(ISLAND).difference(_local(LAGOON)).union(_local(CAY))
    assert geom.area == pytest.approx(want.area, rel=1e-3)
    assert geom.symmetric_difference(want).area < 1e-3 * want.area
    assert not geom.contains(_local(LAGOON).centroid)  # the lagoon is water


def test_save_and_load_keep_the_land(land, tmp_path):
    geom, _ = land
    coast.save(geom, tmp_path / coast.NAME)
    assert coast.load(tmp_path / coast.NAME).equals(geom)


def _tree_over(geom, L=6):
    """A node (L, 0, 0) over the land, its four children and their children: the nodes lands() walks."""
    x0, z0, _, _ = geom.bounds
    origin = (int(x0 // CELL) - 2, int(z0 // CELL) - 2)
    nodes = {(L, 0, 0)}
    for level in range(L - 1, L - 3, -1):
        nodes |= {(level, a, b) for a in range(1 << (L - level)) for b in range(1 << (L - level))}
    leaves = {k for k in nodes if k[0] == L - 2}
    return origin, nodes, leaves


def test_each_node_keeps_its_part_of_the_land_at_its_levels_error(land):
    geom, _ = land
    origin, nodes, leaves = _tree_over(geom)
    out = coast.lands(geom, nodes, leaves, origin)
    assert set(out) == nodes
    parts = {k: shapely.from_wkb(v) for k, v in out.items()}
    assert parts[(6, 0, 0)].area == pytest.approx(geom.area, rel=0.02)
    for key, piece in parts.items():
        if piece.is_empty:
            continue
        box = coast.node_box(*key, origin)
        tol = coast.TOLERANCE if key in leaves else coast.tolerance(key[0])
        assert box.buffer(1e-6).contains(piece)
        assert piece.hausdorff_distance(geom.intersection(box)) <= tol + 1e-6
    leaf = next(k for k in sorted(leaves) if parts[k].area > 0.5 * coast.node_box(*k, origin).area)
    kid = (leaf[0] - 1, 2 * leaf[1], 2 * leaf[2])
    below = shapely.from_wkb(coast.below(out[leaf], *kid, origin))
    assert not below.is_empty and coast.node_box(*kid, origin).buffer(1e-6).contains(below)


class _Coastal:
    """A terrain with relief somewhere, at 0.1 m everywhere here: land the relief would call sea."""

    heights = np.array([[0.0, 30.0]])

    def height(self, x, z, near=None):
        return np.full(np.shape(x), 0.1)


def test_the_ground_follows_the_coast_and_stands_over_the_water(land):
    geom, _ = land
    origin, nodes, leaves = _tree_over(geom)
    L, a, b = 5, 0, 0
    s = summarize([], L, a, b, origin)
    piece = shapely.from_wkb(coast.lands(geom, nodes, leaves, origin)[(L, a, b)])
    assert not piece.is_empty
    s["land"] = np.frombuffer(shapely.to_wkb(piece), np.uint8)
    m = summary_mesh(s, L, a, b, origin, _Coastal())
    corner = coast.node_box(L, a, b, origin).bounds[:2]
    p = np.asarray(m.positions, float) + (corner[0], 0.0, corner[1])  # positions are relative to the corner
    n = np.asarray(m.normals, float)
    tri = np.asarray(m.indices).reshape(-1, 3)
    ground = n[tri[:, 0], 1] > 0.5  # skirts, where the land reaches the node's edge, stand upright
    assert ground.any()
    assert (p[tri[ground].ravel(), 1] >= WATER_Y + LIFT - 1e-4).all()
    centres = p[tri[ground]].mean(axis=1)[:, [0, 2]]
    assert shapely.contains_xy(piece.buffer(1.0), centres[:, 0], centres[:, 1]).all()
    land_area = sum(0.5 * abs(np.cross(t[1] - t[0], t[2] - t[0])[1]) for t in p[tri[ground]])
    assert land_area == pytest.approx(piece.area, rel=1e-3)


def test_a_node_the_land_crosses_hems_its_edges_with_a_skirt(land):
    geom, _ = land
    origin, nodes, leaves = _tree_over(geom)
    out = coast.lands(geom, nodes, leaves, origin)
    key = next(k for k in sorted(leaves) if shapely.from_wkb(out[k]).area > 0.5 * coast.node_box(*k, origin).area)
    s = summarize([], *key, origin)
    s["land"] = np.frombuffer(out[key], np.uint8)
    m = summary_mesh(s, *key, origin, _Coastal())
    n = np.asarray(m.normals, float)
    assert (np.abs(n[:, 1]) < 1e-6).any()  # upright skirt quads


def test_the_shallows_run_around_every_island_and_under_it(land):
    geom, _ = land
    t = coast.shelf(geom)
    band = shapely.union_all(shapely.polygons(np.concatenate([t, t[:, :1]], axis=1).astype(float)))
    assert band.buffer(1.0).contains(geom)
    cay = _local(CAY)
    assert band.contains(shapely.Point(cay.centroid.x + coast.SHELF * 0.8, cay.centroid.y))
    assert not band.contains(shapely.Point(cay.centroid.x + coast.SHELF * 1.5, cay.centroid.y))


def test_the_shallows_show_only_in_the_map_between_the_sunken_sea_and_the_land():
    from types import SimpleNamespace

    from contravia import game, mapview

    engine = SimpleNamespace(h=720)
    driving = SimpleNamespace(mode="drive", map=None, engine=engine, _water_y=lambda: game.BASE_Y)
    assert game.Game._shelf_y(driving) == game.HIDDEN_Y
    m = SimpleNamespace(mode="map", map=mapview.MapCam(0.0, 0.0, 100_000.0, 0.0), engine=engine)
    m._water_y = lambda: game.Game._water_y(m)
    assert game.Game._water_y(m) < game.Game._shelf_y(m) < game.BASE_Y < WATER_Y + LIFT


def test_an_island_with_a_coast_indexes_its_land_and_an_older_index_is_rewritten(island_dir, tmp_path):
    from contravia.world import SUMMARY_VERSION, World

    folder = tmp_path / "cuba"
    shutil.copytree(island_dir, folder, ignore=shutil.ignore_patterns("index.npz", "summary"))
    x, z = World.load(folder).road_points()[0]
    coast.save(shapely.box(x - 3000, z - 3000, x + 3000, z + 3000), folder / coast.NAME)
    with np.load(folder / "index.npz") as f:  # an index from before the coast: no summary_version
        old = {k: f[k] for k in f.files}
    info = json.loads(str(old["info"]))
    info.pop("summary_version")
    old["info"] = np.array(json.dumps(info))
    np.savez(folder / "index.npz", **old)
    w = World.load(folder)
    with np.load(folder / "index.npz") as f:
        info = json.loads(str(f["info"]))
    assert info["summary_version"] == SUMMARY_VERSION and info["coast"]
    files = list((folder / "summary").glob("*.npz"))
    assert files and all("land" in np.load(p).files for p in files)
    assert len(np.load(folder / coast.SHELF_FILE))
    leaf = max(w.tree.leaves)
    if leaf[0] > 1:  # a node inside a leaf, made while playing, cuts its land from the leaf's
        kid = (leaf[0] - 1, 2 * leaf[1], 2 * leaf[2])
        if w.exists(*kid):
            assert "land" in w.summary(*kid)
