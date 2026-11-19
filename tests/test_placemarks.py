"""Where the collectible places' icons fall on the minimap and on the map, and how they join from the city stage."""

import math

import pytest

from contravia import collect, mapview, placemarks
from tests.test_collect import at


def places(*recs):
    return collect.Places(list(recs))


def test_every_kind_in_places_json_has_a_drawing_and_the_colour_says_photographed():
    for p in collect.Places.load():
        assert placemarks.glyph(p.type) in placemarks.DRAWINGS
    assert "#" in placemarks.icon("iglesia", True)
    assert placemarks.icon("iglesia", True) != placemarks.icon("iglesia", False)


@pytest.mark.parametrize("yaw", [0.0, 1.0, -2.3])
def test_a_place_ahead_of_the_car_is_above_the_minimap_centre(yaw):
    ps = places(at(100 * math.cos(yaw), 100 * math.sin(yaw), "a"))
    [(p, right, down)] = placemarks.on_minimap(ps, 0.0, 0.0, yaw, 173.0, 95.0)
    assert abs(right) < 1.0 and down == pytest.approx(-100 * 95 / 173, abs=1.0)


def test_a_place_to_the_cars_right_is_right_of_centre_and_far_ones_are_left_out():
    # facing east (yaw 0, forward +x), the right hand is south, +z
    ps = places(at(0, 50, "right"), at(0, -50, "left"), at(150, 150, "out"))
    got = {p.id: (r, d) for p, r, d in placemarks.on_minimap(ps, 0.0, 0.0, 0.0, 173.0, 95.0)}
    assert set(got) == {"right", "left"} and got["right"][0] > 20 and got["left"][0] < -20


def test_the_map_centres_the_place_under_its_camera_and_drops_what_is_off_frame():
    ps = places(at(0, 0, "here"), at(50_000, 0, "far"))
    cam = mapview.MapCam(0.0, 0.0, 1000.0, 0.0)
    marks = placemarks.on_map(ps, lambda pid: False, cam, 1600, 900)
    assert [m.ids for m in marks] == [("here",)]
    assert marks[0].x == pytest.approx(800, abs=1) and marks[0].y == pytest.approx(450, abs=1)
    assert placemarks.under_cross(marks, 1600, 900) is marks[0]


def test_north_is_up_on_the_map():
    ps = places(at(0, -100, "north"))
    [m] = placemarks.on_map(ps, lambda pid: False, mapview.MapCam(0.0, 0.0, 1000.0, 0.0), 1600, 900)
    assert m.y < 450 and m.x == pytest.approx(800, abs=1)


def test_from_the_city_stage_a_municipios_places_join_in_one_mark():
    recs = [at(0, 0, "a"), at(300, 0, "b"), at(0, 300, "c")]
    ps = places(*recs)
    low = placemarks.on_map(ps, lambda pid: pid == "a", mapview.MapCam(0.0, 0.0, 2000.0, 0.0), 1600, 900)
    assert len(low) == 3 and [m.got for m in low].count(True) == 1
    high = placemarks.on_map(ps, lambda pid: pid == "a", mapview.MapCam(0.0, 0.0, placemarks.CITY_H, 0.0), 1600, 900)
    [m] = high
    assert set(m.ids) == {"a", "b", "c"} and not m.got
    assert m.label == "Centro Habana · 3 lugares, 1 con foto"


def test_nothing_under_the_cross_when_the_nearest_mark_is_too_far():
    m = placemarks.Mark(800 + placemarks.CROSS_PX + 1, 450, "museo", False, ("a",), "a", (0.0, 0.0))
    assert placemarks.under_cross([m], 1600, 900) is None
