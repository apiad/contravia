import numpy as np

from contravia.citydata import CityData
from contravia.geo import cell_of
from contravia.landmarks import MODULES, Landmarks, capitolio, generic, inside, plaza
from contravia.walls import Walls


def square(c, r):
    x, z = c
    return np.array([(x - r, z - r), (x + r, z - r), (x + r, z + r), (x - r, z + r)], np.float32)


def city_with(ids_and_rings, fillers=(), trees=()):
    ids = [i for i, _ in ids_and_rings]
    rings = [r for _, r in ids_and_rings]
    return CityData.from_features(np.zeros((0, 5)), rings, np.array(ids), list(fillers),
                                  np.arange(len(fillers)), np.asarray(trees, float).reshape(-1, 2))


def test_inside_uses_the_even_odd_rule():
    ring = square((0, 0), 10)
    assert inside([(0, 0), (9, -9), (11, 0), (0, 30)], ring).tolist() == [True, True, False, False]


def test_a_landmark_missing_from_the_data_is_skipped(city):
    names = Landmarks(city).names
    assert "Capitolio" not in names and "Malecón" in names   # the fixture holds no landmark footprint


def test_keep_drops_the_replaced_footprint_and_keeps_the_rest():
    cap, other = square((30, 30), 40), square((150, 150), 10)
    lm = Landmarks(city_with([(capitolio.OSM_ID, cap), (7, other)]))
    i, j = cell_of(np.array([[30.0, 30.0]]))[0]
    city = city_with([(capitolio.OSM_ID, cap), (7, other)])
    kept = lm.keep(i, j, city.cell(i, j))
    assert capitolio.OSM_ID not in kept.building_ids.tolist() and 7 in kept.building_ids.tolist()


def test_keep_carries_the_tags_of_the_buildings_it_keeps():
    """The facade plans read each building's tags and municipio: a kept cell must still hold them."""
    from dataclasses import replace

    cap, other = square((30, 30), 40), square((150, 150), 10)
    city = city_with([(capitolio.OSM_ID, cap), (7, other)])
    lm = Landmarks(city)
    i, j = cell_of(np.array([[30.0, 30.0]]))[0]
    cell = city.cell(i, j)
    attrs = np.arange(len(cell.buildings) * 5, dtype=float).reshape(-1, 5)
    kept = lm.keep(i, j, replace(cell, building_attrs=attrs, muni_names=("Centro Habana",)))
    k = cell.building_ids.tolist().index(7)
    assert kept.muni_names == ("Centro Habana",)
    assert kept.building_attrs.tolist() == [attrs[k].tolist()]


def test_the_car_hits_the_landmark_where_the_box_was():
    cap = square((30, 30), 40)
    walls = Walls(city_with([(capitolio.OSM_ID, cap)])).near(30, 30)
    assert len(walls) == 4                                # the footprint's four edges, once
    assert np.isclose(walls[:, :, 0].min(), -10) and np.isclose(walls[:, :, 0].max(), 70)


def test_the_sea_behind_the_malecon_has_no_trees_or_filler():
    trees = [(-1000, -900), (-1000, -600)]                # north of the wall, then south of it
    block = square((-1000, -900), 15)
    city = city_with([(7, square((-1000, -600), 5))], fillers=[block], trees=trees)
    lm = Landmarks(city)
    i, j = cell_of(np.array([[-1000.0, -900.0]]))[0]
    sea = lm.keep(i, j, city.cell(i, j))
    assert len(sea.trees) == 0 and len(sea.fillers) == 0
    i, j = cell_of(np.array([[-1000.0, -600.0]]))[0]
    land = lm.keep(i, j, city.cell(i, j))
    assert len(land.trees) == 1 and len(land.buildings) == 1


def test_every_landmark_builds_when_its_footprints_are_there():
    ids = [capitolio.OSM_ID, 19355360, 23918434, 23918435, 23866797, 23866811, 289803271,
           plaza.TOWER, plaza.BASE, plaza.MINCOM]
    rings = [(i, square((k * 300.0, 2000.0), 30)) for k, i in enumerate(ids)]
    lm = Landmarks(city_with(rings))
    assert lm.names == [m.NAME for m in MODULES if m is not generic]  # no point-only place in these cells
    assert len(lm.mesh.indices) > 0 and np.isfinite(lm.mesh.positions).all()


def test_landmarks_stand_on_the_terrain():
    from contravia.terrain import Terrain, seat

    _, c = np.mgrid[0:400, 0:600]
    t = Terrain(0.02 * (-3000 + 10.0 * c) + 60, (-3000, -1000), (10, 10))      # rises 2 m per 100 m eastward
    ids = [capitolio.OSM_ID, plaza.TOWER, plaza.BASE]
    rings = [square((0.0, 0.0), 40), square((-2744.0, 1397.0), 14), square((-2742.0, 1398.0), 42)]
    city = CityData.from_features(np.zeros((0, 5)), rings, np.array(ids), [], np.zeros(0), np.zeros((0, 2)), t)
    lm = Landmarks(city)
    p = lm.mesh.positions
    cap = p[(np.abs(p[:, 0]) < 45) & (np.abs(p[:, 2]) < 45)]
    assert np.isclose(cap[:, 1].min(), seat(rings[0], t), atol=1e-3)
    paving = p[np.all(np.isclose(lm.mesh.colors, plaza.PAVE), axis=1)]
    assert len(paving) and np.allclose(paving[:, 1], t.height(paving[:, 0], paving[:, 2]) + plaza.PAVING_LIFT, atol=1e-3)


def test_the_malecon_wall_stands_a_sidewalk_past_the_road_edge():
    from contravia.landmarks import malecon

    # a 21 m wide coast road along z = -702 at x around -1000, where LINE runs flat
    road = np.array([[-1100.0, -702.0, -900.0, -702.0, 21.0]])
    edge, wall, out = malecon.shore(road)
    k = int(np.argmin(np.abs(edge[:, 0] + 1000)))
    assert np.isclose(edge[k, 1], -702 - 10.5, atol=0.5)        # the road's sea edge: half its width north
    assert np.isclose(wall[k, 1], edge[k, 1] - malecon.SIDEWALK, atol=0.5)
    assert out[k, 1] < -0.99                                    # seaward is north


def test_the_relief_past_the_malecon_goes_under_the_sea_but_not_under_the_road():
    from contravia.landmarks import malecon
    from contravia.terrain import Terrain

    t = Terrain(np.full((60, 40), 3.0), (-1200.0, -1200.0), (10.0, 10.0), fade=0)
    road = np.array([[-1100.0, -702.0, -900.0, -702.0, 21.0]])
    malecon.carve(t, road)
    assert t.height(-1000.0, -702.0) == 3.0                     # under the road
    assert t.height(-1000.0, -702.0 - 10.5 - malecon.CARVE - 15) == malecon.SEABED


def test_each_landmark_knows_the_box_of_its_own_mesh():
    from contravia import collect

    ids = [capitolio.OSM_ID, 19355360, 23918434, 23918435, 23866797, 23866811, 289803271,
           plaza.TOWER, plaza.BASE, plaza.MINCOM]
    lm = Landmarks(city_with([(i, square((k * 300.0, 2000.0), 30)) for k, i in enumerate(ids)]))
    lo, hi = lm.bounds["Capitolio"]
    p = lm.mesh.positions
    cap = p[(np.abs(p[:, 0]) < 150) & (np.abs(p[:, 2] - 2000.0) < 150)]
    assert np.allclose(lo, cap.min(axis=0), atol=1e-3) and np.allclose(hi, cap.max(axis=0), atol=1e-3)
    assert set(collect.MODELLED) <= set(lm.bounds)


def test_a_point_place_is_one_with_its_types_default_box_and_not_a_building():
    from types import SimpleNamespace as P

    from contravia.landmarks.generic import is_point

    assert is_point(P(type="monumento", w=8.0, d=8.0, h=10.0))
    assert not is_point(P(type="monumento", w=120.0, d=52.0, h=10.0))  # an outline in OSM
    assert not is_point(P(type="museo", w=25.0, d=30.0, h=12.0))  # the city draws its building


def test_spot_moves_a_piece_off_the_street_and_faces_along_it():
    import numpy as np

    from contravia.landmarks.generic import KERB, spot

    roads = np.array([[-100.0, 0.0, 100.0, 0.0, 10.0]])  # a 10 m street along x through the point
    _, z, along = spot(0.0, 0.0, roads, 2.5)
    assert abs(z) >= 5 + KERB + 2.5 and abs(along[0]) == 1.0
    assert spot(0.0, 40.0, roads, 2.5)[:2] == (0.0, 40.0)  # room where it is
    assert spot(0.0, 0.0, np.array([[-100.0, 0.0, 100.0, 0.0, 200.0]]), 2.5) is None


def test_each_type_gets_a_piece_and_the_title_names_a_monuments_form():
    import random

    import numpy as np

    from contravia.landmarks import generic
    from contravia.mesh import MeshBuilder

    for kind, (build, r) in generic.BUILD.items():
        mb = MeshBuilder()
        mark = mb.mark()
        ring = build(mb, np.array([10.0, 20.0]), np.array([1.0, 0.0]), 3.0, random.Random(kind))
        lo, hi = mb.bounds_since(mark)
        assert lo[1] == 3.0 and hi[1] > 4.0 and np.hypot(*(ring - (10.0, 20.0)).T).max() <= r + 0.01, kind
    assert generic.form("Obelisco de Marianao", random.Random(0)) == "obelisk"
    assert generic.form("Busto de Martí", random.Random(0)) == "bust"


def test_place_puts_a_piece_on_each_point_place_in_the_citys_cells_with_its_box():
    from types import SimpleNamespace

    import numpy as np

    from contravia import collect, geo
    from contravia.landmarks import generic
    from contravia.mesh import MeshBuilder

    neptuno = collect.Places.load().by_id("L0557")
    cell = [int(v) for v in geo.cell_of([neptuno.x, neptuno.z])]
    flat = SimpleNamespace(height=lambda x, z: np.zeros(np.shape(x)))
    city = SimpleNamespace(cells=lambda: np.array([cell]), a={"roads": np.zeros((0, 5), np.float32)}, terrain=flat)
    piece = generic.place(city, MeshBuilder())
    assert "L0557" in piece.boxes and len(piece.walls) == len(piece.boxes) == len(piece.clear)
    lo, hi = piece.boxes["L0557"]
    assert lo[0] < neptuno.x < hi[0] and hi[1] - lo[1] < 3.0  # a fountain, on the point: no street in the way
    moved = collect.Places.load().with_meshes({}, piece.boxes).by_id("L0557")
    assert moved.w == hi[0] - lo[0] and moved.base == lo[1]
