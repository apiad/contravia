import math
from itertools import pairwise

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
                                                       "stability": False, "lane": True, "brake": False, "pilot": True}})
    assert lamps_lit(car) == [False, False, True, False, True, False, True]
    assert [label for _, label in LAMPS] == ["DIR", "TRC", "ABS", "EST", "SENDA", "FRENO", "PILOTO"]
    assert lamps_lit(CarState(0, 0, 0, 0.0)) == [False] * 7


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
    assert minimap_stale(last, last, MINIMAP_EVERY, loading=True)  # cells land around a still car
    assert not minimap_stale(last, last, MINIMAP_EVERY - 1, loading=True)


class _Engine:
    """Just enough engine for the HUD to build its panels: a GUI root and a texture that is a plain SVG."""

    def __init__(self):
        from manifoldx.gui.widgets import _GuiRoot

        self.gui, self.started = _GuiRoot(), []

    def on(self, _event):
        return self.started.append

    def render_to_texture(self, *_args, **_kwargs):
        from contravia import ui

        return ui.icon("sol")

    def refresh_texture(self, _texture):
        pass


def test_the_hud_writes_a_widget_only_when_what_it_shows_changed():
    import numpy as np

    from contravia.hud import AREA_SHOWN, Hud

    engine = _Engine()
    hud = Hud(engine)
    engine.started[0]()
    car = CarState(0, 0, 0, 10.0, gear=2, rpm=3000)
    hud.update(car, np.zeros(3))
    hud.speed_text.text = "stale"  # a write from here on would replace it
    hud.update(CarState(1, 0, 0, 10.05, gear=2, rpm=3000), np.zeros(3))  # still 36 km/h
    assert hud.speed_text.text == "stale"
    hud.update(CarState(2, 0, 0, 20.0, gear=2, rpm=3000), np.zeros(3))
    assert hud.speed_text.text == "72"

    hud.set_area("Centro Habana", 0.1)
    shown = hud.area_text.style_overrides = {"fg": "stale"}
    hud.set_area("Centro Habana", 0.1)  # still fully shown
    assert hud.area_text.style_overrides is shown
    hud.set_area("Centro Habana", AREA_SHOWN)  # fading
    assert hud.area_text.style_overrides is not shown


def test_the_minimap_renders_once_more_when_loading_ends():
    import numpy as np

    from contravia.hud import MINIMAP_EVERY, Hud

    engine = _Engine()
    renders = []
    engine.refresh_texture = renders.append
    hud = Hud(engine)
    engine.started[0]()
    car = CarState(0, 0, 0, 0.0)
    for _ in range(MINIMAP_EVERY + 1):
        hud.update(car, np.zeros(3), loading=True)
    assert len(renders) == 2  # the first frame, then MINIMAP_EVERY frames later while loading
    hud.update(car, np.zeros(3), loading=False)
    assert len(renders) == 3  # the cells that landed since
    for _ in range(MINIMAP_EVERY * 2):
        hud.update(car, np.zeros(3), loading=False)
    assert len(renders) == 3  # a still car on a loaded world


def test_the_photo_hint_shows_the_place_only_while_there_is_one():
    from contravia.hud import Hud

    engine = _Engine()
    hud = Hud(engine)
    engine.started[0]()
    hud.show(True)
    assert hud.spot_panel not in engine.gui._panels
    hud.set_spot("Teatro Sauto")
    assert hud.spot_panel in engine.gui._panels and hud.spot_text.text == "Teatro Sauto"
    hud.show(False)
    assert hud.spot_panel not in engine.gui._panels
    hud.show(True)
    assert hud.spot_panel in engine.gui._panels
    hud.set_spot("")
    assert hud.spot_panel not in engine.gui._panels


def test_the_minimap_marks_a_place_ahead_of_the_car_above_its_centre_and_hides_the_spare_icons():
    import numpy as np

    from contravia import collect
    from contravia.hud import MARGIN, MARK_PX, MINIMAP_SHOWN, OFF, Hud
    from tests.test_collect import at

    engine = _Engine()
    hud = Hud(engine)
    engine.started[0]()
    hud.set_places(collect.Places([at(100, 0, "a")]), lambda pid: pid == "a")
    hud.update(CarState(0, 0, 0, 0.0), np.zeros(3))  # yaw 0 faces +x
    (first, image), (second, _) = hud.marks[:2]
    centre = MARGIN + MINIMAP_SHOWN / 2 - MARK_PX / 2
    assert first.offset[0] == round(centre) and first.offset[1] < centre - 30
    assert "#" in image.svg and second.offset == OFF


def test_the_pilot_line_says_its_speed_and_turn():
    from contravia.hud import pilot_line

    assert pilot_line(None, None) == ""
    assert pilot_line(11.1, 0.1) == "40 km/h · recto"
    assert pilot_line(11.1, 1.4) == "40 km/h · dobla a la derecha"
    assert pilot_line(11.1, -1.4, "mph") == "25 mph · dobla a la izquierda"


def test_the_minimap_rises_with_speed_between_a_city_street_and_a_highway():
    from contravia.hud import MINIMAP_AHEAD, MINIMAP_CAR, MINIMAP_LEAD, MINIMAP_SPAN, minimap_target

    speeds = [s / 3.6 for s in range(0, 260, 5)]
    heights = [minimap_target(s) for s in speeds]
    assert all(a <= b for a, b in pairwise(heights))  # never lower when faster
    assert heights[0] == minimap_target(30 / 3.6) < minimap_target(100 / 3.6) < heights[-1]
    ahead = [h * MINIMAP_SPAN * (1 + MINIMAP_CAR) for h in heights]  # metres from the car to the disc's top
    assert ahead[0] == pytest.approx(MINIMAP_AHEAD[0]) and ahead[-1] == pytest.approx(MINIMAP_AHEAD[1])
    assert minimap_target(-20.0) == minimap_target(20.0)  # reversing looks the same way
    assert minimap_target(25.0) * MINIMAP_SPAN * (1 + MINIMAP_CAR) == pytest.approx(25.0 * MINIMAP_LEAD)


def test_the_minimap_eases_toward_its_height_at_any_frame_rate():
    from contravia.hud import MINIMAP_EASE, ease

    assert ease(200.0, 500.0, MINIMAP_EASE) == pytest.approx(500.0 - 300.0 / math.e)
    assert ease(500.0, 200.0, 0.0) == 500.0
    h60, h20 = 200.0, 200.0
    for _ in range(60):
        h60 = ease(h60, 500.0, 1 / 60)
        assert 200.0 < h60 < 500.0  # moves every frame, never past the target
    for _ in range(20):
        h20 = ease(h20, 500.0, 1 / 20)
    assert h60 == pytest.approx(h20)
    assert ease(300.0, 600.0, 1 / 60) - 300.0 < 0.02 * 300.0  # no frame jumps


def test_the_minimap_redraws_when_its_height_changed_and_steps_by_its_pixels():
    from contravia.hud import MINIMAP_EVERY, MINIMAP_LOW, MINIMAP_MOVE, MINIMAP_ZOOM, minimap_stale

    last = (0.0, 0.0, 0.0)
    assert not minimap_stale(last, last, MINIMAP_EVERY, height=MINIMAP_LOW * (1 + MINIMAP_ZOOM * 0.5))
    assert minimap_stale(last, last, MINIMAP_EVERY, height=MINIMAP_LOW * (1 + MINIMAP_ZOOM * 1.5))
    high = MINIMAP_LOW * 3
    step = (MINIMAP_MOVE * 2, 0.0, 0.0)  # one pixel low down, a fraction of one from three times as high
    assert minimap_stale(last, step, MINIMAP_EVERY)
    assert not minimap_stale(last, step, MINIMAP_EVERY, height=high, drawn=high)


def test_the_minimap_zooms_out_smoothly_as_the_car_speeds_up():
    import numpy as np

    from contravia.hud import MINIMAP_LOW, Hud, minimap_target

    engine = _Engine()
    engine.refresh_texture = lambda t: None
    hud = Hud(engine)
    engine.started[0]()
    car = CarState(0, 0, 0, 0.0)
    hud.update(car, np.zeros(3), dt=1 / 60)
    assert hud.height == MINIMAP_LOW and hud.camera.far == MINIMAP_LOW + 200.0
    car.speed = 120 / 3.6
    seen = []
    for _ in range(240):
        hud.update(car, np.zeros(3), dt=1 / 60)
        seen.append(hud.height)
    assert all(b > a for a, b in pairwise(seen))
    jump = (minimap_target(car.speed) - MINIMAP_LOW) / 60  # what one frame of a 1 s ease can cover at most
    assert max(b - a for a, b in pairwise(seen)) <= jump
    assert seen[-1] == pytest.approx(minimap_target(car.speed), rel=0.03)
    assert hud.camera.far > MINIMAP_LOW + 200.0  # the far plane follows the height


def test_what_is_left_of_a_route_reads_as_the_panel_s_second_line():
    from datetime import datetime

    from contravia.hud import route_left

    at = datetime(2026, 11, 20, 10, 30)
    assert route_left(720, 8400, at, None, "kmh") == "12 min · 8,4 km · llegas 10:42"
    assert route_left(720, 8400, at, 11.1, "kmh") == "40 km/h · 12 min · 8,4 km · llegas 10:42"
    assert route_left(30, 400, at, None, "kmh") == "1 min · 400 m · llegas 10:30"


def test_each_manoeuvre_has_its_own_icon():
    import math

    from contravia import ui

    icons = {ui.manoeuvre(k, a, 34) for k, a in (("turn", 1.5), ("turn", -1.5), ("straight", 0.0),
                                                  ("roundabout", 0.5), ("arrive", 0.0), ("slight", math.radians(45)))}
    assert len(icons) == 6 and all(i.startswith("<svg") for i in icons)


def test_the_gps_panel_shows_while_a_route_is_set_and_hides_the_pilot_line():
    from contravia.hud import Hud

    engine = _Engine()
    hud = Hud(engine)
    engine.started[0]()
    hud.show(True)
    hud.set_pilot("40 km/h · recto")
    assert hud.pilot_panel in engine.gui._panels
    hud.set_route("200 m", "Dobla a la derecha en Galiano", "12 min · 8,4 km · llegas 10:42", "turn", 1.5)
    assert hud.route_panel in engine.gui._panels and hud.pilot_panel not in engine.gui._panels
    assert (hud.route_far.text, hud.route_what.text) == ("200 m", "Dobla a la derecha en Galiano")
    hud.set_pilot("40 km/h · recto")
    assert hud.pilot_panel not in engine.gui._panels
    hud.show(False)
    assert hud.route_panel not in engine.gui._panels
    hud.show(True)
    hud.set_route("", "", "", "", 0.0)
    assert hud.route_panel not in engine.gui._panels
    hud.set_pilot("40 km/h · recto")
    assert hud.pilot_panel in engine.gui._panels


def test_a_new_gps_band_renders_the_minimap_again_on_the_next_frame():
    import numpy as np

    from contravia.hud import MINIMAP_EVERY, Hud

    engine = _Engine()
    renders = []
    engine.refresh_texture = renders.append
    hud = Hud(engine)
    engine.started[0]()
    car = CarState(0, 0, 0, 0.0)
    for _ in range(MINIMAP_EVERY + 2):
        hud.update(car, np.zeros(3))
    before = len(renders)
    hud.redraw_minimap()
    hud.update(car, np.zeros(3))
    assert len(renders) == before + 1
