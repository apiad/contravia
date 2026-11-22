"""Causeways: roads over the sea found by their geometry, kept as land with riprap down to the water, the streets
raised onto their crown, and the real causeway to Cayo Coco driven end to end without sinking
(contravia/causeway.py, the riprap of contravia/rules/shore.py)."""

import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import shapely

from contravia import causeway, sea
from contravia.causeway import BASE, CROWN, LIFT, SHOULDER, SLOPE, Causeways
from contravia.citydata import Cell
from contravia.geo import CELL, to_latlon
from contravia.rules import shore as R
from contravia.rules.ground import WATER_Y
from contravia.shore import Shore
from contravia.terrain import Terrain

DATA = Path(os.environ.get("HAVANA_DATA", "data/cuba"))
SAVED = Path(os.environ.get("CONTRAVIA_CAUSEWAYS", DATA / causeway.NAME))  # another causeways.src to try
# the mainland west of x = 0, a cay east of x = 1500, the sea between; z runs 0 to 3000
MAINLAND, CAY = shapely.box(-3000, 0, 0, 3000), shapely.box(1500, 0, 4000, 3000)
LAND = shapely.union(MAINLAND, CAY)


def line(*points):
    return shapely.LineString(points)


def test_a_road_over_the_sea_is_a_causeway_that_climbs_from_the_land():
    road = line((-200, 1000), (1700, 1000))
    pieces, report = causeway.detect([road], [10.0], [], LAND)
    assert report["causeways"] == 1 and len(pieces) == 1
    xz, crown, width = pieces[0]
    assert width == 10.0
    assert xz[:, 0].min() <= 0 and xz[:, 0].max() >= 1500  # it reaches onto the land at both ends
    mid = np.abs(xz[:, 0] - 750) < 100
    assert np.allclose(crown[mid], CROWN)  # over the water, the full crown
    assert crown[np.argmin(xz[:, 0])] < BASE + 1.0 and crown[np.argmax(xz[:, 0])] < BASE + 1.0  # ramps at both ends


def test_a_causeway_mapped_in_pieces_keeps_the_ones_far_out_at_sea():
    """The middle of a long causeway is kilometres from any coastline, and OSM often maps it as ways of its own."""
    far_cay = shapely.box(9000, 0, 12000, 3000)
    land = shapely.union(MAINLAND, far_cay)
    ways = [line((-200, 1000), (1000, 1000)), line((1000, 1000), (8000, 1000)), line((8000, 1000), (9200, 1000))]
    pieces, report = causeway.detect(ways, [10.0] * 3, [], land)
    assert report["causeways"] == 1 and report["km"] == pytest.approx(9.0, abs=0.1)
    middle = [xz for xz, _, _ in pieces if xz[:, 0].min() >= 1000 and xz[:, 0].max() <= 8000]
    assert middle and np.allclose(middle[0][np.abs(middle[0][:, 0] - 4500) < 100, 1], 1000)


def test_the_ground_streets_are_built_on_has_the_bank_on_it():
    """Banked answers the higher of the ground and the bank, so a street or a bridge built on it lies on the crown."""
    pieces, _ = causeway.detect([line((-200, 1000), (1700, 1000))], [10.0], [], LAND)
    cw = Causeways(pieces)
    sea_floor = SimpleNamespace(height=lambda x, z: np.full(np.broadcast(x, z).shape, -2.0))
    banked = causeway.Banked(sea_floor, cw)
    rng = np.random.default_rng(1)
    x, z = rng.uniform(-300, 1800, 4000), rng.uniform(900, 1100, 4000)
    assert np.allclose(banked.height(x, z), np.maximum(-2.0, cw.top(x, z)))
    assert banked.height(np.array([750.0]), np.array([1000.0]))[0] == pytest.approx(CROWN)
    assert banked.height(np.array([750.0]), np.array([2500.0]))[0] == -2.0  # far from it, the ground


def test_a_road_on_land_and_one_along_the_shore_are_not():
    inland = line((-2000, 500), (-500, 2500))
    shore_road = line((-15, 200), (-15, 2800))  # on the land, the sea only on its east side
    pieces, report = causeway.detect([inland, shore_road], [8.0, 8.0], [], LAND)
    assert pieces == [] and report["causeways"] == 0


def test_a_road_on_a_strip_the_coast_wraps_is_a_causeway():
    strip = shapely.box(-10, 1990, 1510, 2010)  # OSM's coastline drawn round the causeway: 20 m of land
    pieces, _ = causeway.detect([line((-200, 2000), (1700, 2000))], [10.0], [], shapely.union(LAND, strip))
    assert len(pieces) == 1 and np.ptp(pieces[0][0][:, 0]) > 1400


def test_a_road_the_relief_drowns_is_a_causeway():
    wide = shapely.union(LAND, shapely.box(-10, 1500, 1510, 2500))  # the coast says land, 1 km across
    road = line((-200, 2000), (1700, 2000))
    assert causeway.detect([road], [10.0], [], wide)[0] == []
    pieces, _ = causeway.detect([road], [10.0], [], wide, height=lambda x, z: np.where(
        (np.asarray(x) > 0) & (np.asarray(x) < 1500), 0.0, 2.0))
    assert len(pieces) == 1


def test_a_service_road_is_a_causeway_on_a_dike_and_a_pier_over_the_water():
    """A service road over open water is a pier or a dock and stays out; on a strip of land the coast wraps (a
    dike, a breakwater) or on land the relief drowns it is a causeway like any road."""
    strip = shapely.box(0, 1990, 1500, 2010)
    dike, pier = line((-200, 2000), (1700, 2000)), line((-200, 1000), (1700, 1000))
    land = shapely.union(LAND, strip)
    pieces, _ = causeway.detect([dike, pier], [4.0, 4.0], [], land, piers=np.array([True, True]))
    assert len(pieces) >= 1 and all(np.allclose(xz[:, 1], 2000) for xz, _, _ in pieces)
    assert min(xz[:, 0].min() for xz, _, _ in pieces) <= 0 and max(xz[:, 0].max() for xz, _, _ in pieces) >= 1500
    wide = shapely.union(LAND, shapely.box(-10, 1500, 1510, 2500))  # the coast says land, 1 km across
    low = lambda x, z: np.where((np.asarray(x) > 0) & (np.asarray(x) < 1500), 0.0, 5.0)
    drowned, _ = causeway.detect([dike], [4.0], [], wide, height=low, piers=np.array([True]))
    assert len(drowned) == 1 and drowned[0][0][:, 0].min() <= 0 and drowned[0][0][:, 0].max() >= 1500


def test_no_ramp_where_the_causeway_meets_a_bridge():
    a, b = line((-200, 1000), (700, 1000)), line((760, 1000), (1700, 1000))
    bridge = line((700, 1000), (760, 1000))
    pieces, report = causeway.detect([a, b], [10.0, 10.0], [bridge], LAND)
    assert report["causeways"] == 2
    for xz, crown, _ in pieces:
        at_bridge = np.argmin(np.abs(xz[:, 0] - 730))
        assert crown[at_bridge] == pytest.approx(CROWN)  # full height up to the bridge's abutment


def osm(path, ways):
    """Write an OSM XML file of ways [(points in local metres, {tag: value})] to path."""
    nodes, rows, n = [], [], 1
    for pts, tags in ways:
        refs = []
        for lat, lon in to_latlon(np.asarray(pts, float)):
            nodes.append(f'<node id="{n}" version="1" lat="{lat:.8f}" lon="{lon:.8f}"/>')
            refs.append(n)
            n += 1
        nd = "".join(f'<nd ref="{r}"/>' for r in refs)
        tg = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
        rows.append(f'<way id="{len(rows) + 1}" version="1">{nd}{tg}</way>')
    path.write_text('<?xml version="1.0"?><osm version="0.6">' + "".join(nodes) + "".join(rows) + "</osm>")
    return path


def test_read_takes_drivable_roads_over_the_sea_and_leaves_bridges_out(tmp_path):
    pbf = osm(tmp_path / "x.osm", [
        ([(-200, 1000), (700, 1000)], {"highway": "primary", "name": "Pedraplén"}),
        ([(700, 1000), (760, 1000)], {"highway": "primary", "bridge": "yes", "layer": "1", "name": "Pedraplén"}),
        ([(760, 1000), (1700, 1000)], {"highway": "primary", "name": "Pedraplén"}),
        ([(-200, 2000), (1700, 2000)], {"highway": "primary", "bridge": "yes", "name": "Puente"}),
        ([(-200, 2500), (1700, 2500)], {"highway": "footway"}),
        ([(-200, 2700), (1700, 2700)], {"highway": "service"}),
    ])
    pieces, report = causeway.read(pbf, LAND)
    assert report["causeways"] == 2
    assert [n for n, _ in report["named"]] == ["Pedraplén"]
    z = np.concatenate([xz[:, 1] for xz, _, _ in pieces])
    assert np.allclose(z, 1000, atol=0.5)  # the bridge, the footway and the service road are not causeways
    bridge = shapely.box(700.5, 990, 759.5, 1010)
    assert not any(shapely.intersects(bridge, shapely.LineString(xz)) for xz, _, _ in pieces)
    causeway.save(pieces, tmp_path / causeway.NAME)
    back = causeway.load(tmp_path / causeway.NAME)
    assert len(back) == len(pieces) and np.allclose(back[0][0], pieces[0][0])


def sea_shore():
    """A Shore over flat sea with one causeway along x = 300 from z = -2000 to 2000, 10 m of road."""
    h = np.zeros((101, 101), np.float32)
    terrain = Terrain(h, (-1000.0, -2000.0), (40.0, 40.0), fade=0.0)
    pts = np.array([(300.0, z) for z in np.arange(-2000.0, 2001.0, 100.0)], np.float32)
    cw = Causeways([(pts, np.full(len(pts), CROWN, np.float32), 10.0)])
    return Shore(terrain, causeways=cw), cw


def test_the_bank_is_land_with_riprap_and_grass_on_its_shoulders():
    s, _cw = sea_shore()
    i, j = int(300 // CELL), 0
    assert s.coastal(i, j)
    f = s.field(i, j)
    z = np.full(7, 100.0)
    off = np.array([0.0, 4.0, 6.5, 7.5 + 1.0, 7.5 + SLOPE * 1.0, 7.5 + SLOPE * (CROWN - WATER_Y) + 1.0, 40.0])
    x = 300.0 + off
    y, d, w = f.height(x, z), f.sample(f.d, x, z), f.sample(f.w, x, z)
    assert y[0] == pytest.approx(CROWN, abs=0.02) and y[1] == pytest.approx(CROWN, abs=0.02)  # road and shoulder
    assert y[3] < CROWN - 0.3 and y[4] < y[3]  # the riprap goes down at SLOPE
    assert (d[:4] > 0).all() and d[5] < 0  # land down to the water (within the grid's 2.5 m), sea past it
    assert (w[:6, R.RIPRAP] > 0.9).all()
    assert R.vegetated(f, x[:3], z[:3])[2]  # grass on the shoulder's earth
    assert not R.vegetated(f, x[3:5], z[3:5]).any()  # rock on the slope
    floor = s.floor(x, z)
    assert not any(sea.in_sea(float(fl), float(fl)) for fl in floor[:5])  # nowhere deep on the bank
    assert sea.in_sea(float(floor[-1]), float(floor[-1]))  # off the causeway the sea is deep


def test_riprap_lays_big_rocks_on_the_slope_and_none_on_the_road():
    s, _cw = sea_shore()
    f = s.field(int(300 // CELL), 0)
    xz, seeds = R.riprap(f)
    assert len(xz) > 50
    r = np.abs(xz[:, 0] - 300.0)
    assert (r > 5.0 + SHOULDER - 0.5).all()  # off the road and its shoulder
    assert (r < 5.0 + SHOULDER + SLOPE * (CROWN - WATER_Y) + 4.0).all()
    assert len(R.rocks(f)[0]) == 0  # no rocky shore's small rocks along a causeway
    xz2, seeds2 = R.riprap(s.field(int(300 // CELL), 0))
    assert np.array_equal(xz, xz2) and np.array_equal(seeds, seeds2)  # seeded: the same rocks every time


def test_lift_raises_streets_on_a_causeway_only():
    _, cw = sea_shore()
    tri = np.array([[[298.0, 0.05, 100.0], [302.0, 0.05, 100.0], [300.0, 0.05, 110.0]],
                    [[150.0, 0.05, 100.0], [154.0, 0.05, 100.0], [152.0, 0.05, 110.0]],
                    [[298.0, 6.0, 300.0], [302.0, 6.0, 300.0], [300.0, 6.0, 310.0]]], np.float32)
    c = cw.lift(Cell(street_tris=tri), 1, 0)
    y = c.street_tris[:, :, 1]
    assert np.allclose(y[0], CROWN + LIFT) and np.allclose(y[1], 0.05) and np.allclose(y[2], 6.0)
    assert np.allclose(tri[0, :, 1], 0.05)  # the province's own arrays are left as they were
    far = Cell(street_tris=tri)
    assert cw.lift(far, 40, 40) is far  # a cell no causeway reaches is not copied


def cayo_coco():
    """The Cayo Coco causeways of the real island, the world and its surface, with causeways.src."""
    from contravia.surface import Surface
    from contravia.world import World

    world = World.load(DATA)
    world._causeways = Causeways(causeway.load(SAVED))
    box = shapely.box(*np.r_[_local(22.27, -78.56), _local(22.53, -78.25)].reshape(2, 2).min(0),
                      *np.r_[_local(22.27, -78.56), _local(22.53, -78.25)].reshape(2, 2).max(0))
    pieces = [p for p in causeway.load(SAVED) if box.intersects(shapely.LineString(p[0]))]
    return world, Surface(world, world.terrain), pieces


def _local(lat, lon):
    from contravia.geo import to_local

    return to_local(np.array([lat]), np.array([lon]))[0]


needs_island = pytest.mark.skipif(not (DATA / "index.npz").exists() or not SAVED.exists(), reason="needs the island and its causeways")


@needs_island
def test_the_real_causeway_to_cayo_coco_stands_over_the_water_end_to_end():
    _world, surface, pieces = cayo_coco()
    assert sum(shapely.LineString(p).length for p, _, _ in pieces) > 20_000
    wet = []
    for xz, _, width in pieces:
        g = shapely.LineString(xz)
        for d in np.arange(0.0, g.length, 50.0):
            a = np.array(g.interpolate(d).coords[0])
            b = np.array(g.interpolate(min(d + 1.0, g.length)).coords[0])
            t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            for off in (0.0, width / 2 - 1.0, 1.0 - width / 2):  # the middle and both lanes' outer wheels
                q = a + np.array([-t[1], t[0]]) * off
                y = float(surface.height(q[0], q[1])[0])
                floor = float(surface.shore.floor(np.array([q[0]]), np.array([q[1]]))[0])
                if sea.in_sea(floor, y) or y < WATER_Y + sea.DRY:
                    wet.append((round(d), off, y))
    assert wet == []


@needs_island
@pytest.mark.slow
def test_a_car_drives_the_cayo_coco_causeway_without_sinking():
    from contravia.car import CarState, Controls, settle
    from contravia.game import Game

    _world, surface, pieces = cayo_coco()
    xz = max(pieces, key=lambda p: shapely.LineString(p[0]).length)[0].astype(float)
    route = shapely.LineString(xz)
    g = Game.__new__(Game)
    g.engine = SimpleNamespace(state=SimpleNamespace(), origin=(0.0, 0.0), set_origin=lambda x, z: None)
    g.surface = surface
    g.walls = SimpleNamespace(nearest_road=lambda x, z: None)
    a, b = xz[0], np.array(route.interpolate(10.0).coords[0])
    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    lane = a + np.array([-math.sin(yaw), math.cos(yaw)]) * 2.0
    g._place(settle(CarState(lane[0], lane[1], yaw, 0.0, y=CROWN), surface), 10.0)
    dt, lowest = 1 / 60, np.inf
    for _ in range(int(240 / dt)):
        s = route.project(shapely.Point(g.car.x, g.car.z))
        if s > route.length - 30.0:
            break
        ahead = np.array(route.interpolate(s + 15.0).coords[0])
        want = math.atan2(ahead[1] - g.car.z, ahead[0] - g.car.x)
        steer = float(np.clip(2.0 * math.remainder(want - g.car.yaw, math.tau), -1, 1))
        fast = g.car.speed > 15.0
        g.car = g.driven.step(Controls(0.0 if fast else 0.5, 0.2 if fast else 0.0, steer, False), dt,
                              np.zeros((0, 2, 2)), None)
        g._at_sea(dt)
        assert g.sinking is None, f"sank {s:.0f} m along at ({g.car.x:.0f}, {g.car.z:.0f})"
        lowest = min(lowest, g.car.y)
    assert s > min(3000.0, route.length - 60.0)  # it drove kilometres along it
    assert lowest > WATER_Y + sea.DRY  # out of the water all along, from the ramp off the shore on
