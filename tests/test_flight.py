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
