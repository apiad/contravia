import math

import pytest

from contravia import mapview


def test_pose_looks_down_at_the_pitch_from_the_south_when_yaw_is_zero():
    eye, target = mapview.pose(mapview.MapCam(100.0, -50.0, 1000.0, 0.0))
    assert target == (100.0, 0.0, -50.0)
    assert eye[1] == pytest.approx(1000.0)
    assert eye[0] == pytest.approx(100.0)
    assert eye[2] > -50.0  # behind the target is south, +z; north is -z
    run = eye[2] - target[2]
    assert math.degrees(math.atan2(eye[1], run)) == pytest.approx(mapview.PITCH_DEG)


def test_zoom_multiplies_the_height_and_stays_in_range():
    cam = mapview.MapCam(0.0, 0.0, 1000.0, 0.0)
    assert mapview.zoom(cam, 2.0).h == pytest.approx(2000.0)
    assert mapview.zoom(cam, 1e-9).h == mapview.MIN_H
    assert mapview.zoom(cam, 1e9).h == mapview.MAX_H


def test_pan_moves_the_ground_under_the_pointer_with_it():
    cam = mapview.MapCam(0.0, 0.0, 1000.0, 0.0)
    per_px = mapview.metres_per_pixel(cam, 720)
    # Dragging right drags the ground right: the camera's target goes west (-x); dragging down goes north (-z).
    right = mapview.pan(cam, 100.0, 0.0, 720)
    assert right.x == pytest.approx(-100 * per_px)
    down = mapview.pan(cam, 0.0, 100.0, 720)
    assert down.z < 0.0


def test_pan_follows_the_turn():
    cam = mapview.MapCam(0.0, 0.0, 1000.0, math.pi / 2)
    moved = mapview.pan(cam, 100.0, 0.0, 720)
    assert abs(moved.x) < 1e-6 and moved.z != 0.0


def test_metres_per_pixel_grows_with_height():
    low, high = mapview.MapCam(0, 0, 500.0, 0), mapview.MapCam(0, 0, 5000.0, 0)
    assert mapview.metres_per_pixel(high, 720) == pytest.approx(10 * mapview.metres_per_pixel(low, 720))


def test_far_and_fog_cover_what_the_camera_sees():
    cam = mapview.MapCam(0, 0, 40000.0, 0)
    far, (fog_start, fog_end) = mapview.far_and_fog(cam)
    eye, target = mapview.pose(cam)
    assert far > math.dist(eye, target) * 3
    assert fog_start < fog_end <= far


def test_exaggeration_is_none_low_and_grows_to_twenty_over_the_island():
    assert mapview.exaggeration(150.0) == 1.0
    assert mapview.exaggeration(10_000.0) == 1.0
    assert mapview.exaggeration(100_000.0) == pytest.approx(10.5)  # half way in the logarithm
    assert mapview.exaggeration(1_000_000.0) == pytest.approx(20.0)
    assert mapview.exaggeration(mapview.MAX_H) == pytest.approx(20.0)


def test_the_game_stretches_heights_only_in_the_map():
    from types import SimpleNamespace

    from contravia.game import Game

    assert Game._exaggeration(SimpleNamespace(mode="drive", map=None)) == 1.0
    high = SimpleNamespace(mode="map", map=mapview.MapCam(0.0, 0.0, 1_000_000.0, 0.0))
    assert Game._exaggeration(high) == pytest.approx(20.0)
