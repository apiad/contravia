"""The World: the island as a folder of provinces, read lazily, or one province in memory."""

import shutil

import numpy as np
import pytest

from contravia.citydata import CityData
from contravia.quadtree import LEVELS
from contravia.world import World

FAR_EAST = 900_000.0  # a road this far east of the Capitolio, in the second province only


@pytest.fixture(scope="session")
def island(tmp_path_factory, city_npz, city):
    """La Habana from the fixture, and Artemisa: one far-east road and a building, plus a building in one of
    La Habana's cells, as a feature that lies across the border would."""
    out = tmp_path_factory.mktemp("cuba")
    shutil.copy(city_npz, out / "la-habana.npz")
    shared = city.a["roads"][0]
    mx, mz = (shared[0] + shared[2]) / 2 + 1.0, (shared[1] + shared[3]) / 2 + 1.0
    roads = np.array([(FAR_EAST - 40, 0.0, FAR_EAST + 40, 0.0, 6.0)], np.float32)
    rings = [np.array([(FAR_EAST, 20), (FAR_EAST + 10, 20), (FAR_EAST + 10, 30), (FAR_EAST, 30)], np.float32),
             np.array([(mx, mz), (mx + 1, mz), (mx + 1, mz + 1), (mx, mz + 1)], np.float32)]
    CityData.from_features(roads, rings, np.array([7, 8], np.int64), [], np.zeros(0, np.int64),
                           np.zeros((0, 2), np.float32)).save(out / "artemisa.npz")
    return out, (int(mx // 200), int(mz // 200))


def test_one_province_in_memory_is_that_city(city):
    w = World(city)
    for i, j in city.cells().tolist()[:5]:
        assert np.array_equal(w.cell(i, j).roads, city.cell(i, j).roads)
    assert w.exists(*next(iter(w.tree.leaves))) and w.graph is city.graph


def test_a_folder_of_provinces_is_one_island(island):
    folder, border = island
    w = World.load(folder)
    assert sorted(w.names) == ["Artemisa", "La Habana"] and (folder / "index.npz").exists()
    assert (w.cell(int(FAR_EAST // 200), 0).building_ids == 7).all()
    both = w.cell(*border)
    assert 8 in both.building_ids.tolist() and len(w.provinces_of(*border)) == 2
    assert len(set(both.building_ids.tolist())) == len(both.building_ids)  # joined, nothing doubled
    assert w.nearest_road(FAR_EAST + 300, 0.0)[0] == pytest.approx(FAR_EAST)
    assert w.nearest_road(450_000.0, 300_000.0) is None
    assert w.summary(LEVELS, 0, 0) is not None and any((folder / "summary").iterdir())


def test_a_province_read_ahead_is_the_one_a_start_there_opens(island):
    folder, _ = island
    w = World.load(folder)
    k = w.province_at(FAR_EAST, 0.0)
    assert k not in w._open
    w.open_soon(FAR_EAST, 0.0)
    read = w._reading[k]
    assert w.province(k) is read.result() and k in w._open and not w._reading
    assert w.nearest_road(FAR_EAST + 300, 0.0)[0] == pytest.approx(FAR_EAST)
    w.open_soon(FAR_EAST, 0.0)  # open already: nothing more is read
    assert not w._reading


def test_focus_moves_the_graph_to_the_province_of_the_car(island):
    folder, _ = island
    w = World.load(folder)
    assert w.focused == "La Habana"
    assert w.focus(FAR_EAST, 0.0) and w.focused == "Artemisa"
    assert not w.focus(FAR_EAST + 50, 0.0)
    assert w.focus(0.0, 0.0) and w.focused == "La Habana"


def two_reliefs():
    """Province 0 holds cell (0, 0) on a small grid at 10 m; province 1 holds cell (1, 0) on a grid at 20 m that
    reaches far past both, so every point here lies deeper inside it."""
    from types import SimpleNamespace

    from contravia.terrain import Terrain
    from contravia.world import IslandTerrain, _keys

    grids = [Terrain(np.full((11, 11), 10.0, np.float32), (-500.0, -500.0), (100.0, 100.0), fade=0.0),
             Terrain(np.full((101, 101), 20.0, np.float32), (-5000.0, -5000.0), (100.0, 100.0), fade=0.0)]
    world = SimpleNamespace(top=20.0, _grids=[World._grid_box(t) for t in grids], _terrain=lambda k: grids[k],
                            _region_keys=[_keys(np.array([[0, 0]])), _keys(np.array([[1, 0]]))])
    world._regions_over = lambda x, z: World._regions_over(world, x, z)
    return IslandTerrain(world)


def test_the_ground_of_a_cell_is_the_relief_of_the_province_that_holds_it():
    from contravia.world import BLEND

    t = two_reliefs()
    assert t.owner(np.array([0, 1, 7, 900]), np.array([0, 0, 7, 0])).tolist() == [0, 1, 1, -1]
    x = np.array([100.0, 200.0 - BLEND, 200.0, 200.0 + BLEND, 300.0])
    h = t.height(x, np.full(5, 100.0))
    assert h.tolist() == pytest.approx([10.0, 10.0, 15.0, 20.0, 20.0])  # held, pure to BLEND of the edge, mixed on it
    ramp = t.height(np.linspace(200.0 - BLEND, 200.0 + BLEND, 41), np.full(41, 100.0))
    assert (np.diff(ramp) > 0).all() and np.abs(np.diff(ramp)).max() < 0.3  # no step across the edge
    assert t.height(1500.0, 1500.0) == pytest.approx(20.0)  # bare land: the grid it lies deepest inside
    assert t.height(9000.0, 0.0) == 0.0 and t.height(np.zeros((2, 3)), np.zeros((2, 3))).shape == (2, 3)


def test_an_island_of_one_province_keeps_its_relief(city):
    w = World(city)
    xz = city.a["roads"][:50, :2].astype(float)
    assert np.allclose(w.terrain.height(xz[:, 0], xz[:, 1]), city.terrain.height(xz[:, 0], xz[:, 1]))


def test_heights_of_many_points_agree_with_each_point_asked_alone():
    """Points spread over held cells, bare land, the mixing bands round the edges and the open sea."""
    t = two_reliefs()
    t.world._terrain(0).heights[:] = np.random.default_rng(1).uniform(5.0, 15.0, (11, 11)).astype(np.float32)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-800.0, 1200.0, 300), 200.0 + rng.uniform(-25.0, 25.0, 100), [9000.0]])
    z = np.concatenate([rng.uniform(-800.0, 800.0, 300), rng.uniform(-25.0, 225.0, 100), [0.0]])
    together = t.height(x, z)
    alone = np.array([float(t.height(a, b)) for a, b in zip(x, z, strict=True)])
    assert together == pytest.approx(alone, abs=1e-9) and len(np.unique(np.round(together, 3))) > 50
