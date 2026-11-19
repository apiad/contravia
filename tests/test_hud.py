import math

import pytest

from contravia.car import CarState
from contravia.hud import LAMPS, bearing, gearbox, lamps_lit, north_offset, speed_text


@pytest.mark.parametrize("yaw, deg", [(0.0, 90), (math.pi / 2, 180), (math.pi, 270), (-math.pi / 2, 0)])
def test_bearing_is_clockwise_from_north_with_z_south(yaw, deg):
    assert bearing(yaw) == pytest.approx(deg)


def test_gearbox_shows_the_cars_own_gear_and_revs():
    assert gearbox(CarState(0, 0, 0, 0.0, gear=1, rpm=850), 850, 6900)[0] == "N"  # standing still
    assert gearbox(CarState(0, 0, 0, -3.0, gear=-1, rpm=2000), 850, 6900)[0] == "R"
    assert gearbox(CarState(0, 0, 0, 20.0, gear=3, rpm=4000), 850, 6900)[0] == "3"
    low = gearbox(CarState(0, 0, 0, 20.0, gear=3, rpm=2000), 850, 6900)[1]
    high = gearbox(CarState(0, 0, 0, 20.0, gear=3, rpm=6000), 850, 6900)[1]
    assert 0 <= low < high <= 1


def test_the_lamps_light_with_the_assists():
    car = CarState(0, 0, 0, 20.0, telemetry={"lit": {"steer": False, "traction": False, "abs": True,
                                                       "stability": False}})
    assert lamps_lit(car) == [False, False, True, False]
    assert len(LAMPS) == 4
    assert lamps_lit(CarState(0, 0, 0, 0.0)) == [False] * 4


def test_speed_text_in_kmh_and_mph():
    assert speed_text(10.0) == "36"
    assert speed_text(-10.0, "mph") == "22"


def test_north_badge_sits_on_top_facing_north_and_below_facing_south():
    side, size = 190, 26
    north_x, north_y = north_offset(-math.pi / 2, side, size)  # facing north: the badge at the top
    south_x, south_y = north_offset(math.pi / 2, side, size)  # facing south: at the bottom
    assert north_x == pytest.approx(south_x)
    assert north_y < south_y
    east_x, _ = north_offset(0.0, side, size)  # facing east, north is on the map's left: farther from the right edge
    assert east_x > north_x


def test_the_minimap_redraws_only_when_the_car_moved_or_turned():
    from contravia.hud import MINIMAP_EVERY, MINIMAP_IDLE, MINIMAP_MOVE, MINIMAP_TURN, minimap_stale

    last = (100.0, 200.0, 0.3)
    assert minimap_stale(None, last, 1)
    assert not minimap_stale(last, last, MINIMAP_EVERY)
    assert not minimap_stale(last, (100.0 + MINIMAP_MOVE * 0.9, 200.0, 0.3), MINIMAP_EVERY)
    assert minimap_stale(last, (100.0, 200.0 + MINIMAP_MOVE * 1.1, 0.3), MINIMAP_EVERY)
    assert minimap_stale(last, (100.0, 200.0, 0.3 + math.radians(MINIMAP_TURN * 1.1)), MINIMAP_EVERY)
    assert not minimap_stale(last, (100.0, 200.0, 0.3 + math.radians(MINIMAP_TURN * 0.9)), MINIMAP_EVERY)
    assert not minimap_stale(last, (150.0, 200.0, 0.3), MINIMAP_EVERY - 1)  # never more often than before
    assert minimap_stale((0.0, 0.0, math.pi - 0.01), (0.0, 0.0, -math.pi + 0.01), MINIMAP_EVERY) is False
    assert minimap_stale(last, last, MINIMAP_IDLE)  # the traffic around a still car moves on
