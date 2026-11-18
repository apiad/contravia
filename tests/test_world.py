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


def test_focus_moves_the_graph_to_the_province_of_the_car(island):
    folder, _ = island
    w = World.load(folder)
    assert w.focused == "La Habana"
    assert w.focus(FAR_EAST, 0.0) and w.focused == "Artemisa"
    assert not w.focus(FAR_EAST + 50, 0.0)
    assert w.focus(0.0, 0.0) and w.focused == "La Habana"
