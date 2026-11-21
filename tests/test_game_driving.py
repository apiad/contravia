"""The game drives the manifoldx car: teleports, a change of vehicle and a reload keep it where it was."""

from types import SimpleNamespace

import numpy as np
import pytest

from contravia.car import Controls
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


def test_strengths_follow_the_help_level(city):
    g = bare_game(city)
    g.help_level, g.locked_assists = 4, ("abs",)
    np.testing.assert_allclose(g.strengths(), [0.0, 0.6, 0.0, 0.6])


class TwoLevels:
    """A surface with a deck 6 m over the ground everywhere: height answers the level nearest `near`."""

    def __init__(self, inner):
        self.inner = inner

    def height(self, x, z, near=None):
        base = np.asarray(self.inner.height(x, z, near=None), float)
        if near is None:
            return base
        near = np.broadcast_to(np.asarray(near, float), np.shape(base))
        return np.where(np.abs(near - (base + 6.0)) < np.abs(near - base), base + 6.0, base)


def test_respawn_on_a_deck_stays_on_the_deck(city):
    g = bare_game(city)
    g.surface = TwoLevels(g.surface)
    g.teleport(100.0, 50.0, y=g.surface.inner.height(np.array([100.0]), np.array([50.0]))[0] + 6.0)
    deck = g.car.y
    g.teleport(g.car.x, g.car.z, y=g.car.y)  # R
    g.driven.run(Controls(0, 0, 0, False), 1.0, np.zeros((0, 2, 2)), g.strengths())
    assert g.driven.state.y == pytest.approx(deck, abs=0.1)
    assert abs(g.driven.state.speed) < 0.1


def test_a_reload_keeps_the_height_the_gear_and_the_slide(city):
    g = bare_game(city)
    g.surface = TwoLevels(g.surface)
    ground = g.surface.inner.height(np.array([100.0]), np.array([50.0]))[0]
    g.teleport(100.0, 50.0, y=ground + 6.0)
    g.driven.launch(30.0)
    f = g.driven.fleet
    f.angular[0, 1] = 0.4  # turning
    f.velocity[0] += np.array([0.0, 0.0, 2.0])  # and sliding
    g.car = g.driven._read()
    saved = g.saved_car()
    g2 = bare_game(city)
    g2.surface = TwoLevels(g2.surface)
    g2._resume(saved)
    assert g2.car.y == pytest.approx(g.car.y, abs=0.05)
    assert g2.car.gear >= 3
    assert g2.driven.fleet.angular[0, 1] == pytest.approx(0.4)
    assert g2.car.vy == pytest.approx(g.car.vy, abs=1e-6)


def test_a_gps_destination_survives_a_reload(city):
    from contravia.world import World

    g = bare_game(city)
    g.city = World(city)
    g._make_gps()
    g._gps_event({"x": 150.0, "z": 80.0, "title": "Capitolio"})
    assert g.engine.state.gps_to == [150.0, 80.0, "Capitolio"] and g.gps.to == (150.0, 80.0)
    again = bare_game(city)
    again.engine, again.city = g.engine, g.city
    again._make_gps()
    assert again.gps.to == (150.0, 80.0) and again.gps.title == "Capitolio"
    again.gps_clear({})
    assert again.gps.to is None and again.engine.state.gps_to is None


def test_following_a_route_reads_the_focused_province_s_edge(city):
    from contravia.driving_help import Road
    from contravia.gps import RouteGraph
    from contravia.world import World

    g = bare_game(city)
    g.city = World(city)
    g._make_gps()
    g.teleport(100.0, 50.0)
    g._dt = 1 / 60
    g.gps.router.rg = RouteGraph.from_graphs([g.city.graph])
    g.place = Road(g.city.graph).locate(g.car.x, g.car.z, g.car.yaw)
    g.gps_to(g.car.x + 300.0, g.car.z)
    g._follow_route()
    assert g.engine.state.gps["to"] == [g.car.x + 300.0, g.car.z]
