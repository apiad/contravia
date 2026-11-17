import numpy as np

from contravia.citydata import CityData


def sample():
    roads = np.array([[10, 10, 30, 10, 6], [390, 10, 410, 10, 6]], np.float32)   # cells (0,0) and (2,0)
    square = np.array([[50, 50], [60, 50], [60, 60], [50, 60]], np.float32)
    return CityData.from_features(roads, [square], np.array([42]), [square + 200], np.array([7]),
                                  np.array([[5, 5]], np.float32))


def test_cell_returns_exactly_its_features(tmp_path):
    cd = sample()
    cd.save(tmp_path / "c.npz")
    cd = CityData.load(tmp_path / "c.npz")
    c = cd.cell(0, 0)
    assert len(c.roads) == 1 and c.building_ids.tolist() == [42] and len(c.trees) == 1
    assert cd.cell(1, 1).filler_seeds.tolist() == [7]
    assert len(cd.cell(2, 0).roads) == 1
    assert len(cd.cell(9, 9).roads) == 0 and cd.cell(9, 9).buildings == []


def test_rings_survive_the_round_trip(tmp_path):
    cd = sample()
    cd.save(tmp_path / "c.npz")
    ring = CityData.load(tmp_path / "c.npz").cell(1, 1).fillers[0]
    assert ring.tolist() == [[250, 250], [260, 250], [260, 260], [250, 260]]


def test_road_points_are_segment_midpoints():
    assert sorted(sample().road_points()[:, 0].tolist()) == [20.0, 400.0]
