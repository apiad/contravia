"""The map's roads as lines: widths and alpha by height, ends on the ground, layers in order, set again when the
render frame moves."""

import time

import numpy as np
import pytest
from manifoldx.lines import Lines

from contravia import maproads


class _Flat:
    def height(self, x, z, near=None):
        return np.full(np.shape(x), 7.0)


class _World:
    terrain = _Flat()

    def __init__(self):
        main = np.array([[0, 0, 100, 0, 14.0], [100, 0, 200, 0, 14.0]])
        street = np.array([[0, 50, 0, 150, 6.0]])
        self.by = {(13, 0, 0): {"roads": main}, (3, 1, 1): {"roads": np.r_[street, main]}}

    def summary(self, L, a, b):
        return self.by.get((L, a, b), {"roads": np.zeros((0, 5))})


class _Engine:
    def __init__(self):
        self.lines = Lines()
        self.origin = (0.0, 0.0)


def _settle(roads, h, drawn=()):
    for _ in range(100):
        roads.update(True, h, drawn)
        if len(roads.engine.lines.layers["main"]) and (not drawn or len(roads.engine.lines.layers["streets"])):
            return
        time.sleep(0.02)
    raise AssertionError("the layers never landed")


def test_widths_grow_and_streets_come_in_as_the_camera_comes_down():
    assert maproads.scale(1_000_000) < maproads.scale(200_000) < maproads.scale(20_000) < maproads.scale(2_000)
    assert maproads.alphas(500_000) == (1.0, 0.0)
    assert maproads.alphas(50_000) == (1.0, 1.0)
    assert maproads.alphas(1_000) == (0.0, 0.0)


def test_the_layers_land_in_order_on_the_ground_and_main_roads_are_not_streets():
    e = _Engine()
    roads = maproads.MapRoads(e, _World(), (13, 0, 0))
    _settle(roads, 50_000, {(3, 1, 1), (9, 0, 0)})
    assert list(e.lines.layers) == list(maproads.KEYS)
    main, streets = e.lines.layers["main"], e.lines.layers["streets"]
    assert len(main) == 2 and len(streets) == 1  # the 14 m roads stay out of the streets
    assert main[:, 1] == pytest.approx(7.0 - 0.3 + maproads.LIFT)  # the summary ground's height, then LIFT
    assert e.lines.looks["streets"][1] == 1.0


def test_the_layers_move_with_the_render_frame_and_hide_off_the_map():
    e = _Engine()
    roads = maproads.MapRoads(e, _World(), (13, 0, 0))
    _settle(roads, 500_000)
    e.origin = (100.0, 0.0)
    roads.update(True, 500_000, ())
    assert e.lines.layers["main"][0, 0] == pytest.approx(-100.0)
    roads.update(False, 0.0, ())
    assert all(e.lines.looks[k][1] == 0.0 for k in maproads.KEYS)
