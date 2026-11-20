import numpy as np

from contravia.prepare import build_city, read_osm

FIX = "tests/fixtures/mini.osm"


def city():
    return build_city(read_osm(FIX))


def all_roads(cd):
    return np.concatenate([cd.cell(*c).roads for c in cd.cells()])


def test_footways_are_dropped_and_widths_follow_class():
    cd, _ = city()
    assert set(all_roads(cd)[:, 4].tolist()) == {6.0, 10.0}


def test_long_segments_are_split_to_50_m():
    r = all_roads(city()[0])
    assert np.hypot(r[:, 2] - r[:, 0], r[:, 3] - r[:, 1]).max() <= 50.0 + 1e-3


def test_filler_goes_only_to_empty_blocks_outside_parks():
    _, rep = city()
    assert rep["blocks"] >= 4
    assert rep["filler_blocks"] == rep["blocks"] - 2   # minus the built block and the park block
    # the green block is the one park; the empty blocks' land, less the streets round them, is under
    # parks.OUT_MIN, so outside the city they keep their filler
    assert rep["parks"]["organic"] == 1 and sum(rep["parks"].values()) == 1


def test_degenerate_building_is_skipped_and_counted():
    _, rep = city()
    assert rep["buildings"] == 1 and rep["buildings_skipped"] == 1


def test_park_trees_are_seeded():
    a, _ = city()
    b, _ = city()
    ta = np.concatenate([a.cell(*c).trees for c in a.cells()])
    tb = np.concatenate([b.cell(*c).trees for c in b.cells()])
    assert len(ta) > 1 and np.array_equal(ta, tb)


def test_report_has_every_key():
    _, rep = city()
    for k in ("roads", "segments", "buildings", "buildings_skipped", "blocks", "filler_blocks", "trees",
              "cells", "largest_cell", "cells_with_roads_no_buildings"):
        assert k in rep


def test_features_outside_the_box_are_not_read():
    raw = read_osm(FIX, box=(22.0, -81.0, 22.5, -80.5))   # a box far from the fixture
    assert raw.roads == [] and raw.buildings == [] and raw.trees == [] and raw.green == []


def test_heights_from_glo30_tiles_reach_the_city_data(tmp_path):
    import tifffile

    from contravia.citydata import CityData
    from contravia.prepare import main
    from contravia.terrain import _tile_name

    dem = tmp_path / "dem"
    dem.mkdir()
    for lat in (22, 23):
        tifffile.imwrite(dem / _tile_name(lat, -83), np.full((36, 36), 12.0, np.float32),
                         predictor=3, compression="deflate")
    report = main(FIX, tmp_path / "cuba", dem=dem, provinces=["La Habana"])["La Habana"]
    assert report["heights_m"][1] == 12.0  # the lowest is the sea the Malecón's wall carves, -2 m
    assert np.isclose(CityData.load(tmp_path / "cuba" / "la-habana.npz").terrain.height(0.0, 0.0), 12.0)


def test_the_relief_far_away_follows_the_projection_not_a_degree_grid(tmp_path):
    """Near Santiago the meridians turn about 3 degrees against the local axes: a height read through the
    projection lands on the right tile's sample, where a degree grid scaled at the Capitolio misses by kilometres."""
    import tifffile

    from contravia.geo import to_local
    from contravia.terrain import _tile_name, read_terrain

    n = 360  # 10 arc-seconds a sample
    grid = np.zeros((n, n), np.float32)
    grid[:, n // 2:] = 100.0  # the tile's east half, from -75.5 degrees, stands at 100 m
    tifffile.imwrite(tmp_path / _tile_name(20, -76), grid)
    t = read_terrain(tmp_path, (20.2, -75.8, 20.8, -75.2), stride=1, fetch=False)
    west, east = to_local(np.array([20.5, 20.5]), np.array([-75.55, -75.45]))
    assert t.height(*west) < 1.0 and t.height(*east) > 99.0


def test_a_tree_never_stands_on_a_road():
    from contravia.prepare import on_road

    roads = np.array([[0.0, 0.0, 100.0, 0.0, 10.0]])  # 10 m wide along z = 0
    pts = np.array([[50.0, 0.0], [50.0, 5.5], [50.0, 6.5], [150.0, 0.0]])
    assert on_road(pts, roads).tolist() == [True, True, False, False]


def test_with_the_satellite_seeing_houses_everywhere_only_green_blocks_stay_parks(monkeypatch):
    from contravia import parks
    from contravia.builtup import BuiltUp

    classify, given = parks.classify, []
    monkeypatch.setattr(parks, "classify", lambda *a: given.append(a[7]) or classify(*a))
    raw = read_osm(FIX)
    _, plain = build_city(raw)
    raw.built_up = BuiltUp.of(np.ones((400, 400), bool), (-4000.0, -4000.0))
    _, seen = build_city(raw)
    assert sum(seen["parks"].values()) == sum(plain["parks"].values()) == 1  # the fixture's one park is green in OSM
    assert plain["parks"]["organic"] == 1 and seen["parks"]["city"] == 1  # in a town the satellite sees, it is urban
    assert seen["filler_blocks"] == plain["filler_blocks"]
    assert given[0] is None and np.allclose(given[1], 1.0)  # every block seen built over
