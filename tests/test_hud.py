import math

import pytest

from contravia.hud import bearing, heading_text


@pytest.mark.parametrize("yaw, deg", [(0.0, 90), (math.pi / 2, 180), (math.pi, 270), (-math.pi / 2, 0)])
def test_bearing_is_clockwise_from_north_with_z_south(yaw, deg):
    assert bearing(yaw) == pytest.approx(deg)


def test_heading_text_names_the_point_degrees_and_speed():
    assert heading_text(-math.pi / 4, 10.0) == "NE  45°   36 km/h"
