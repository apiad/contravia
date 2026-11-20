import math
from itertools import pairwise

import numpy as np
import pytest

from contravia import flight, mapview
from contravia.camera import chase
from contravia.car import CarState


def _chase_pose(x=100.0, z=-40.0, yaw=0.7):
    _, eye, target = chase(None, CarState(x, z, yaw, 0.0, y=3.0), 0.0)
    return eye, target


def _fly(f, dt=1 / 60, ready=lambda f: True, end=None):
    """Step a flight to its end; the poses on the way and the flight landed."""
    poses = [f.pose(end)]
    while not f.done:
        f = flight.step(f, dt, ready(f))
        poses.append(f.pose(end))
    return poses, f


def test_a_flight_up_starts_on_the_chase_camera_and_ends_on_the_map():
    start = _chase_pose()
    cam = mapview.MapCam(100.0, -40.0, flight.UP_H, 0.0)
    end = mapview.pose(cam)
    f = flight.up(start, end)
    poses, landed = _fly(f, end=end)
    assert np.allclose(poses[0][0], start[0]) and np.allclose(poses[0][1], start[1])
    assert np.allclose(poses[-1][0], end[0]) and np.allclose(poses[-1][1], end[1])
    assert flight.SHORTEST <= landed.elapsed <= flight.LONGEST + 0.02
    heights = [float(e[1]) for e, _ in poses]
    assert all(b >= a - 1e-6 for a, b in pairwise(heights))  # it only climbs


def test_a_flight_down_ends_on_the_street_pose_and_only_falls():
    start = mapview.pose(mapview.MapCam(0.0, 0.0, 2500.0, 0.4))
    end = _chase_pose(30.0, 20.0, -2.0)
    f = flight.down(start, end, (30.0, 20.0))
    poses, landed = _fly(f)
    assert np.allclose(poses[0][0], start[0]) and np.allclose(poses[-1][0], end[0])
    assert np.allclose(poses[-1][1], end[1]) and landed.dest == (30.0, 20.0)
    heights = [float(e[1]) for e, _ in poses]
    assert all(b <= a + 1e-6 for a, b in pairwise(heights))
    assert landed.elapsed == pytest.approx(f.seconds, abs=0.02)  # ground ready: no wait


def test_the_pose_changes_smoothly_frame_to_frame():
    start = mapview.pose(mapview.MapCam(0.0, 0.0, 1_200_000.0, 0.0))
    end = _chase_pose(0.0, 0.0, 3.0)
    poses, _ = _fly(flight.down(start, end, (0.0, 0.0)))
    dist = [float(np.linalg.norm(t - e)) for e, t in poses]
    ratios = [b / a for a, b in pairwise(dist)]
    assert min(ratios) > 0.85  # at most 15 % closer in one frame at 60 fps, even from the whole island


def test_progress_only_grows_and_the_duration_is_bounded():
    for h in (flight.UP_H, 2500.0, 1_200_000.0):
        f = flight.down(mapview.pose(mapview.MapCam(0.0, 0.0, h, 0.0)), _chase_pose(), (0.0, 0.0))
        assert flight.SHORTEST <= f.seconds <= flight.LONGEST and f.cap == f.seconds + flight.HOLD
        s = [f.s]
        while not f.done:
            f = flight.step(f, 0.05, False)
            s.append(f.s)
        assert all(b > a for a, b in pairwise(s))


def test_a_descent_slows_near_the_end_while_the_ground_builds_and_never_past_its_cap():
    start, end = mapview.pose(mapview.MapCam(0.0, 0.0, 2500.0, 0.0)), _chase_pose()
    f = flight.down(start, end, (0.0, 0.0))
    _, never = _fly(f, ready=lambda f: False)
    assert never.elapsed <= f.cap + 1e-9 and never.elapsed > f.seconds + 1.0  # it waited, up to the cap
    _, late = _fly(f, ready=lambda f: f.elapsed > f.seconds)  # the ground lands a little late
    assert f.seconds < late.elapsed < never.elapsed
    early = f
    while early.s < flight.SLOW_FROM:
        early = flight.step(early, 1 / 60, False)
    assert early.elapsed == pytest.approx(flight.SLOW_FROM * f.seconds, abs=0.02)  # no wait before SLOW_FROM


def test_the_heading_turns_the_short_way():
    start = _chase_pose(yaw=math.pi - 0.1)
    end = _chase_pose(yaw=-math.pi + 0.1)
    mid = flight.between(start, end, 0.5)
    d = mid[1] - mid[0]
    assert math.cos(math.atan2(d[2], d[0]) - math.pi) > 0.99


def test_view_follows_the_eye_height_for_far_and_fog():
    low = flight.view(_chase_pose())
    high = flight.view(mapview.pose(mapview.MapCam(0.0, 0.0, 5000.0, 0.0)))
    assert high.h == pytest.approx(5000.0)
    assert mapview.far_and_fog(low)[0] < mapview.far_and_fog(high)[0]


def _street_between_houses():
    """A street running north-south on a 60 m high plateau, with 25 m houses close on both sides of it."""
    from contravia.citydata import CityData
    from contravia.surface import Surface
    from contravia.terrain import Terrain

    t = Terrain(np.full((101, 101), 60.0), (-1000.0, -1000.0), (20.0, 20.0), fade=40.0)
    road = np.array([[0.0, -400.0, 0.0, 400.0, 10.0]], np.float32)
    rings, tops = [], []
    for z in range(-300, 300, 24):
        for x0, x1 in ((8.0, 60.0), (-60.0, -8.0)):
            rings.append(np.array([[x0, z], [x1, z], [x1, z + 20.0], [x0, z + 20.0]]))
            tops.append(60.0 + 25.0)
    city = CityData.from_features(road, [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
    return Surface(city, t), rings, tops


def _hits(poses, surface, rings, tops):
    """The eyes of a flight that stand below the street or the ground, or inside a house under its roof."""
    import shapely

    houses = [shapely.Polygon(r) for r in rings]
    bad = []
    for eye, _ in poses:
        x, y, z = (float(v) for v in eye)
        if y < float(surface.height(x, z)[0]) + 1.0:
            bad.append(("below the street", round(x), round(y), round(z)))
        for house, top in zip(houses, tops, strict=True):
            if y < top and house.contains(shapely.Point(x, z)):
                bad.append(("in a house", round(x), round(y), round(z)))
    return bad


@pytest.mark.parametrize("turn", [math.pi, math.radians(170), math.radians(-150)])
def test_a_wide_turn_flies_over_the_houses_and_never_under_the_street(turn):
    from contravia.car import settle

    surface, rings, tops = _street_between_houses()
    clear = flight.clearance(surface, rings, tops)
    cam = mapview.MapCam(0.0, 0.0, flight.UP_H, 0.0)  # looks north, along -z
    look = math.atan2(-1.0, 0.0) + turn  # the car's heading, turned that far from the map's
    car = settle(CarState(0.0, 0.0, look, 0.0, y=0.0), surface)
    _, eye, target = chase(None, car, 0.0, surface)
    down, _ = _fly(flight.down(mapview.pose(cam), (eye, target), (0.0, 0.0), clear))
    assert np.allclose(down[-1][0], eye) and np.allclose(down[-1][1], target)
    assert _hits(down, surface, rings, tops) == []
    up_end = mapview.pose(cam)
    up, _ = _fly(flight.up((eye, target), up_end, clear), end=up_end)
    assert np.allclose(up[0][0], eye) and np.allclose(up[-1][0], up_end[0])
    assert _hits(up, surface, rings, tops) == []


def test_a_lift_found_late_eases_in_from_where_the_flight_is():
    surface, rings, tops = _street_between_houses()
    start = mapview.pose(mapview.MapCam(0.0, 0.0, flight.UP_H, 0.0))
    _, eye, target = chase(None, CarState(0.0, 0.0, 0.3, 0.0, y=61.0), 0.0)
    f = flight.down(start, (eye, target), (0.0, 0.0))
    while f.s < 0.3:
        f = flight.step(f, 1 / 60)
    lift = flight.lift_for(f, flight.clearance(surface, rings, tops))
    assert lift  # this heading looks across the houses: the low part needs lifting
    late = flight.with_lift(f, lift)
    assert np.allclose(late.pose()[0], f.pose()[0])  # no jump where it arrives
    poses, _ = _fly(late)
    assert _hits(poses, surface, rings, tops) == []
