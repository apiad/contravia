import math

import pytest

from contravia.hud import bearing, gearbox, north_offset, speed_text


@pytest.mark.parametrize("yaw, deg", [(0.0, 90), (math.pi / 2, 180), (math.pi, 270), (-math.pi / 2, 0)])
def test_bearing_is_clockwise_from_north_with_z_south(yaw, deg):
    assert bearing(yaw) == pytest.approx(deg)


def test_gearbox_climbs_with_speed_and_knows_reverse_and_neutral():
    assert gearbox(0.0)[0] == "N"
    assert gearbox(-3.0)[0] == "R"
    gears = [gearbox(kmh / 3.6)[0] for kmh in (10, 30, 50, 70, 100, 150)]
    assert gears == ["1", "2", "3", "4", "5", "6"]
    assert gearbox(20 / 3.6)[1] < gearbox(33 / 3.6)[1]  # the revs climb inside a gear


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
