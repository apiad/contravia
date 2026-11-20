"""The car in the sea: the floor goes down past the shallows, deep water is told from a ford, a bridge or a tunnel,
the sink's timeline, the shore it comes back on, and a car driven off a beach that ends on land
(contravia/sea.py, Shore.floor, Game._at_sea)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import shapely

from contravia import sea
from contravia.car import Controls
from contravia.game import Game
from contravia.rules.ground import SEA, WATER_Y
from contravia.rules.shore import SHALLOW_Y, SHALLOWS
from contravia.shore import Shore
from contravia.terrain import Terrain

COAST = 1100.0


def beach():
    """A beach along x = COAST: ground rising 2 % to the east, the sea to the west, OSM's sand over the line."""
    x = np.arange(0.0, 3001.0, 10.0)
    h = np.tile((x - COAST) * 0.02 + SEA, (301, 1)).astype(np.float32)
    terrain = Terrain(h, (0.0, 0.0), (10.0, 10.0), fade=0.0)
    return Shore(terrain, {"beach": shapely.box(COAST - 60, 0, COAST + 150, 3000)})


def test_the_floor_is_the_ground_on_land_and_goes_down_past_the_shallows():
    reach = SHALLOWS[0]
    assert sea.depth(5.0, reach) == 0.0
    d = -np.arange(0.0, 41.0, 1.0)
    floor = sea.floor(np.full(d.shape, SHALLOW_Y), d, reach)
    assert (np.diff(floor) <= 0).all()  # always down, never a step up
    assert floor[0] == pytest.approx(SHALLOW_Y)
    assert not sea.in_sea(float(floor[int(reach)]), float(floor[int(reach)]))  # wading at the band's edge
    assert sea.in_sea(float(floor[-1]), float(floor[-1]))
    assert SHALLOW_Y - floor.min() <= sea.MAX_DEPTH + 1e-9


@pytest.mark.parametrize("floor,wheels,deep", [
    (SHALLOW_Y, SHALLOW_Y, False),  # the band of shallows
    (WATER_Y - 0.4, WATER_Y - 0.4, False),  # fording a river mouth
    (WATER_Y - 1.5, WATER_Y - 1.5, True),  # past the band
    (WATER_Y - 0.8, WATER_Y - 0.75, True),  # at a cliff's foot
    (WATER_Y - 2.0, 6.0, False),  # on a bridge over the sea
    (WATER_Y - 2.0, -14.0, False),  # in a tunnel under it
])
def test_in_sea_only_on_a_deep_floor(floor, wheels, deep):
    assert sea.in_sea(floor, wheels) is deep


def test_a_river_mouth_is_never_deep():
    for reach in SHALLOWS:  # a river 20 m wide: no water is more than 10 m from land
        f = float(sea.floor(SHALLOW_Y, -10.0, reach))
        assert not sea.in_sea(f, f)


def test_the_sink_eases_down_tilts_the_nose_and_ends_in_two_to_three_seconds():
    assert 2.0 <= sea.BACK <= 3.0 and sea.SINK < sea.BACK
    drop0, pitch0, _ = sea.sink(0.0)
    assert drop0 == pytest.approx(0.0, abs=1e-9) and pitch0 == pytest.approx(0.0)
    drops = [sea.sink(t)[0] for t in np.arange(0.0, sea.SINK, 0.25)]
    assert drops[-1] > drops[1] > -sea.BOB
    drop, pitch, roll = sea.sink(sea.SINK)
    assert drop == pytest.approx(sea.DROP) and pitch == pytest.approx(-sea.TILT) and abs(roll) < sea.TILT
    assert sea.sink(sea.BACK) == pytest.approx(sea.sink(sea.SINK))
    assert sea.drag(1.0) < 0.06


def test_the_sunk_car_is_lower_with_its_nose_down():
    at, q = sea.sunk((10.0, 1.0, 5.0), (0.0, 0.0, 0.0, 1.0), sea.SINK)
    assert at == pytest.approx((10.0, 1.0 - sea.DROP, 5.0))
    x, y, z, w = q
    nose_y = 2 * (x * y + w * z)  # the car's +x axis turned by q, its height
    assert nose_y == pytest.approx(-math.sin(sea.TILT), abs=1e-6)


def test_away_faces_off_the_point():
    assert sea.away(0.0, 0.0, 10.0, 0.0, math.pi) == pytest.approx(0.0, abs=1e-9)
    assert sea.away(0.0, 0.0, 10.0, 0.0, 0.3) == pytest.approx(0.3)


def land_east(xs, zs):
    """Ground rising east of x = 0, under the water west of it."""
    return np.atleast_1d(np.asarray(xs, float)) * 0.05


def test_ashore_takes_a_dry_road_within_reach_facing_away_from_the_water():
    road = lambda x, z: (x + 40.0, z, math.pi)
    x, z, yaw, y = sea.ashore(-30.0, 0.0, road, land_east)
    assert (x, z, y) == (10.0, 0.0, None) and math.cos(yaw) > 0.99


def test_ashore_without_a_road_picks_dry_land_facing_inland():
    for road in (lambda x, z: None, lambda x, z: (x + 5000.0, z, 0.0), lambda x, z: (x, z, 0.0)):
        x, z, yaw, y = sea.ashore(-30.0, 7.0, road, land_east)
        assert land_east(x, z)[0] >= WATER_Y + sea.DRY and y == pytest.approx(land_east(x, z)[0])
        assert x - -30.0 <= 50.0 and math.cos(yaw) > 0.9  # the nearest dry land, facing east, inland


def test_ashore_with_no_land_says_so():
    assert sea.ashore(0.0, 0.0, lambda x, z: None, lambda xs, zs: np.full(np.shape(xs), -5.0)) is None


def test_the_shore_floor_is_the_drawn_ground_on_land_and_deep_out_at_sea():
    s = beach()
    z = np.full(3, 1500.0)
    x = np.array([COAST + 60.0, COAST - 10.0, COAST - 80.0])
    floor, drawn = s.floor(x, z), s.height(x, z)
    assert floor[0] == pytest.approx(drawn[0])
    assert not sea.in_sea(floor[1], floor[1])  # wading in the band
    assert sea.in_sea(floor[2], floor[2]) and drawn[2] > WATER_Y  # drawn over the water, deep for the wheels


def test_a_car_driven_off_a_beach_sinks_and_ends_on_land(city):
    from contravia.surface import Surface

    g = Game.__new__(Game)
    g.engine = SimpleNamespace(state=SimpleNamespace(), origin=(0.0, 0.0), set_origin=lambda x, z: None)
    shore = beach()
    g.surface = Surface(city, shore.terrain)
    g.surface.shore = shore  # the fixture city is flat: give it the beach
    g.walls = SimpleNamespace(nearest_road=lambda x, z: None)
    from contravia.car import CarState, settle

    g._place(settle(CarState(COAST + 40.0, 1500.0, math.pi, 0.0, y=1.0), g.surface), 10.0)
    g.eye = np.array([COAST + 50.0, 4.0, 1500.0])
    sank, dt = None, 1 / 60
    for k in range(int(20 / dt)):
        c = Controls(0.4, 0.0, 0.0, False) if g.sinking is None else Controls(0.0, 1.0, 0.0, True)
        g.car = g.driven.step(c, dt, np.zeros((0, 2, 2)), None)
        g._at_sea(dt)
        if g.sinking is not None and sank is None:
            sank = k * dt
            assert g.car.x < COAST - SHALLOWS[0]  # it waded through the shallows before it sank
        if sank is not None and g.sinking is None:
            break
    assert sank is not None and sank < 12.0, "the car never went into the sea"
    assert g.sinking is None and k * dt - sank == pytest.approx(sea.BACK, abs=2 * dt)
    assert float(shore.floor(g.car.x, g.car.z)[0]) >= WATER_Y + sea.DRY  # on dry land
    assert math.cos(g.car.yaw) > 0.9 and abs(g.car.speed) < 0.1  # facing inland, stopped
    assert g.eye.tolist() == [COAST + 50.0, 4.0, 1500.0]  # the chase eases in from where the camera was
