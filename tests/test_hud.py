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
