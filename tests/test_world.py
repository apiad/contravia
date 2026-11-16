"""The World: Havana's data where it has it, the island's tiles elsewhere."""

import numpy as np
import pytest

from havana.quadtree import Tree
from havana.tiles import Tiles, write_tiles
from havana.world import World

FAR_EAST = 900_000.0  # a road this far east of the Capitolio, in the tiles only


@pytest.fixture(scope="session")
def island(tmp_path_factory, city):
    """Tiles holding the fixture's roads plus one far-east road and a building beside it."""
    out = tmp_path_factory.mktemp("tiles")
    roads = np.vstack([city.a["roads"], [(FAR_EAST - 40, 0.0, FAR_EAST + 40, 0.0, 6.0)]]).astype(np.float32)
    ring = np.array([(FAR_EAST, 20), (FAR_EAST + 10, 20), (FAR_EAST + 10, 30), (FAR_EAST, 30)], np.float32)
    weights = {(int(i), int(j)): 1.0 for i, j in np.floor(((roads[:, 0:2] + roads[:, 2:4]) / 2) / 200.0).tolist()}
    weights[(int(FAR_EAST // 200), 0)] = 1.0
    write_tiles(out, Tree.build(weights), roads, [ring], np.array([7], np.int64), [], np.zeros(0, np.int64),
                np.zeros((0, 2)), {})
    return out


def test_core_only_world_is_the_city(city):
    w = World(city)
    for i, j in city.cells().tolist()[:5]:
        assert np.array_equal(w.cell(i, j).roads, city.cell(i, j).roads)
    assert w.terrain is city.terrain and w.exists(*next(iter(w.tree.leaves)))


def test_tiles_fill_outside_the_core_and_sea_has_no_road(city, island):
    w = World(city, Tiles(island))
    assert (w.cell(int(FAR_EAST // 200), 0).building_ids == 7).all()
    assert w.nearest_road(FAR_EAST + 300, 0.0)[0] == pytest.approx(FAR_EAST)
    assert w.nearest_road(450_000.0, 300_000.0) is None
