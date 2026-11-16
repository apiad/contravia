"""havana tiles on the mini fixture: the same cells as build_city, and the same files twice."""

import numpy as np

from havana.prepare import build_city, read_osm
from havana.tiles import Tiles
from havana.tiles_prepare import main


def test_tiles_match_build_city_and_repeat(tmp_path):
    main("tests/fixtures/mini.osm", tmp_path / "a", split=2.0)
    main("tests/fixtures/mini.osm", tmp_path / "b", split=2.0)
    city, _ = build_city(read_osm("tests/fixtures/mini.osm", boundary=""))
    tiles = Tiles(tmp_path / "a")
    assert len(tiles.tree.leaves) > 1
    for i, j in city.cells().tolist():
        a, b = city.cell(i, j), tiles.cell(i, j)
        assert np.array_equal(a.roads, b.roads)
        assert np.array_equal(a.building_ids, b.building_ids)
        assert sorted(a.filler_seeds.tolist()) == sorted(b.filler_seeds.tolist())
        assert len(a.trees) == len(b.trees)
    for f in sorted((tmp_path / "a").rglob("*.npz")):
        if f.name != "index.npz":  # the index carries the run's seconds
            assert f.read_bytes() == (tmp_path / "b" / f.relative_to(tmp_path / "a")).read_bytes()
