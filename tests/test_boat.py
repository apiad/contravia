"""The boat's model against the spec's numbers (section 08), on a stand-in water with a beach."""

import math

import numpy as np
import pytest

from contravia.boat import DEEP, Boat, Water, nearest_water, rotation
from contravia.car import Controls

LEVEL = 0.1
LINE = 190.0  # x where the beach's floor stands 0.6 m under the water


class Beach:
    """Flat water at LEVEL; the floor 5 m deep up to x = 100, then rising 5 cm a metre toward +x."""

    def level(self, x, z):
        return np.full(np.shape(np.atleast_1d(x)), LEVEL)

    def floor(self, x, z):
        x = np.atleast_1d(np.asarray(x, float))
        return np.where(x < 100.0, -5.0, -5.0 + 0.05 * (x - 100.0))


FULL = Controls(throttle=1.0, brake=0.0, steer=0.0, handbrake=False)
IDLE = Controls(throttle=0.0, brake=0.0, steer=0.0, handbrake=False)


def test_full_throttle_reaches_twelve_metres_a_second():
    b = Boat(Beach(), -2000.0, 0.0, 0.0)
    assert 11.5 <= b.run(FULL, 60).speed <= 12.5


def test_coasting_from_full_speed_drops_under_one_metre_a_second_in_12_to_18_s():
    b = Boat(Beach(), -3000.0, 0.0, 0.0)
    b.run(FULL, 60)
    t = 0.0
    while b.state.speed > 1.0:
        b.step(IDLE, 1 / 60)
        t += 1 / 60
    assert 12.0 <= t <= 18.0


def test_full_rudder_at_full_speed_draws_a_circle_of_40_to_60_m():
    b = Boat(Beach(), -3000.0, 0.0, 0.0)
    b.run(FULL, 60)
    b.run(Controls(1.0, 0.0, 1.0, False), 20)  # settle into the turn
    xs, zs = [], []
    for _ in range(60 * 30):
        s = b.step(Controls(1.0, 0.0, 1.0, False), 1 / 60)
        xs.append(s.x)
        zs.append(s.z)
    radius = (np.ptp(xs) + np.ptp(zs)) / 4
    assert 40.0 <= radius <= 60.0


def test_the_rudder_does_not_turn_a_boat_at_rest():
    b = Boat(Beach(), -3000.0, 0.0, 0.3)
    b.run(Controls(0.0, 0.0, 1.0, False), 5)
    assert abs(math.degrees(b.state.yaw - 0.3)) < 1.0


def test_a_hard_turn_slides_the_stern_out():
    b = Boat(Beach(), -3000.0, 0.0, 0.0)
    b.run(FULL, 60)
    slip = max(abs(b.step(Controls(1.0, 0.0, 1.0, False), 1 / 60).vy) for _ in range(120))
    assert slip > 0.3


def test_driven_at_a_beach_it_stops_within_5_m_past_the_depth_line_and_backs_off():
    b = Boat(Beach(), 0.0, 0.0, 0.0)
    s = b.run(FULL, 40)
    assert b.aground
    assert abs(s.speed) < 0.2
    assert LINE - 1.0 <= s.x <= LINE + 5.0  # the hull's middle, where the model runs it aground
    stopped = s.x
    s = b.run(Controls(0.0, 1.0, 0.0, False), 10)
    assert s.x < stopped - 5.0


def test_the_hull_floats_on_the_water():
    b = Boat(Beach(), -3000.0, 0.0, 0.0)
    s = b.run(IDLE, 2)
    assert abs(s.y - LEVEL) <= b.spec.swell + 1e-6
    assert abs(s.pitch) < 0.05 and abs(s.roll) < 0.1


def test_snapshot_and_restore_keep_the_motion():
    b = Boat(Beach(), -3000.0, 0.0, 0.0)
    b.run(Controls(1.0, 0.0, 0.5, False), 10)
    snap = b.snapshot()
    a = b.run(IDLE, 1)
    c = Boat(Beach(), 0.0, 0.0, 0.0)
    c.restore(snap)
    assert c.run(IDLE, 1).x == pytest.approx(a.x)


def test_rotation_turns_forward_along_the_games_yaw():
    x, y, z, w = rotation(math.pi / 2, 0.0, 0.0)
    q = np.array([x, y, z])
    v = np.array([1.0, 0.0, 0.0])
    turned = v + 2 * w * np.cross(q, v) + 2 * np.cross(q, np.cross(q, v))
    assert turned == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)  # yaw toward +z


def test_nearest_water_finds_deep_enough_water():
    x, _ = nearest_water(Beach(), 250.0, 0.0, depth=1.0)
    assert x <= 182.0 + 1e-6  # floor 0.9 m under the water at x 182


class FakeShore:
    """Relief 0 at sea everywhere; a shore cell only at cell (0, 0), whose floor is 2 m under the water."""

    class terrain:
        @staticmethod
        def height(x, z):
            return np.zeros(np.shape(np.atleast_1d(x)))

    def field(self, i, j):
        return object() if (i, j) == (0, 0) else None

    def floor(self, x, z):
        x = np.atleast_1d(np.asarray(x, float))
        return np.where((x >= 0) & (x < 10) & (np.atleast_1d(z) >= 0), -1.9, 0.0)

    def level(self, x, z):
        return np.full(np.shape(np.atleast_1d(x)), LEVEL)


def test_water_reads_the_shore_near_it_and_deep_sea_away_from_it():
    w = Water(FakeShore(), sea_below=0.3)
    assert w.floor([5.0], [5.0])[0] == pytest.approx(-1.9)  # a shore cell: its own floor
    assert w.floor([5000.0], [5000.0])[0] == DEEP  # open sea: the relief's 0 is not the floor
