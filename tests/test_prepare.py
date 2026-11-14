import numpy as np

from havana.prepare import build_city, read_osm

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
