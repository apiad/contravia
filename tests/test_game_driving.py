"""The game drives the manifoldx car: teleports, a change of vehicle and a reload keep it where it was."""

from types import SimpleNamespace

import numpy as np
import pytest

from contravia.game import Game
from contravia.rules import fleet


def bare_game(city):
    from contravia.surface import Surface
    from contravia.walls import Walls

    g = Game.__new__(Game)
    g.engine = SimpleNamespace(state=SimpleNamespace(), origin=(0.0, 0.0), set_origin=lambda x, z: None)
    g.walls, g.terrain = Walls(city), city.terrain
    g.surface = Surface(city, g.terrain)
    return g


def test_a_new_vehicle_stands_where_the_old_one_was(city):
    from manifoldx.modeling.assets.bus import build_bus

    g = bare_game(city)
    g.teleport(100.0, 50.0)
    g.driven.launch(8.0)
    g.car = g.driven.state
    x, z, yaw = g.car.x, g.car.z, g.car.yaw
    g.model, g.family = build_bus(fleet.metroguagua()), "metroguagua"
    g._rebuild()
    assert (g.car.x, g.car.z) == pytest.approx((x, z), abs=0.05) and g.car.yaw == pytest.approx(yaw, abs=1e-3)
    assert g.car.speed == pytest.approx(0.0, abs=0.05)
    assert g.driven.spec.mass > 9000


def test_a_reload_keeps_place_and_speed(city):
    g = bare_game(city)
    g.teleport(100.0, 50.0)
    x, z, yaw = g.car.x, g.car.z, g.car.yaw
    g._resume({"pose": [x, z, yaw, 12.0], "eye": [x, 3.0, z]})
    assert (g.car.x, g.car.z) == pytest.approx((x, z), abs=0.05)
    assert g.car.speed == pytest.approx(12.0, abs=0.1)


def test_strengths_follow_the_realism(city):
    g = bare_game(city)
    g.realism, g.locked_assists = 30.0, ("abs",)
    np.testing.assert_allclose(g.strengths(), [0.7, 0.7, 0.0, 0.7])
