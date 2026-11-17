import numpy as np

from caiman.citydata import CityData
from caiman.geo import cell_of
from caiman.landmarks import MODULES, Landmarks, capitolio, inside, plaza
from caiman.walls import Walls


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
    assert lm.names == [m.NAME for m in MODULES]
    assert len(lm.mesh.indices) > 0 and np.isfinite(lm.mesh.positions).all()


def test_landmarks_stand_on_the_terrain():
    from caiman.terrain import Terrain, seat

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
