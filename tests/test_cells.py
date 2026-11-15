from havana import cells
from havana.citydata import CityData


def test_every_fixture_cell_builds(city_npz):
    cd = CityData.load(city_npz)
    for i, j in cd.cells():
        md = cells.build(int(i), int(j))
        assert md.indices.max() < len(md.positions)
        assert len(md.positions) > 4


def test_empty_cell_is_ground_only(city_npz):
    md = cells.build(500, 500)
    assert len(md.positions) == 21 * 21 and len(md.indices) == 3 * 2 * 20 * 20   # the ground grid alone


def test_positions_are_relative_to_the_cell_corner(city_npz):
    md = cells.build(0, 0)
    assert md.positions[:, [0, 2]].min() > -60 and md.positions[:, [0, 2]].max() < 260
