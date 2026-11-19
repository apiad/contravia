"""Collectible places in world metres: the cell index, the box corners and the framing of a camera."""

import math

import numpy as np
import pytest

from contravia import collect, geo


def record(pid="T1", lat=geo.ORIGIN[0], lon=geo.ORIGIN[1], w=20.0, d=20.0, h=20.0, yaw=0.0):
    """One place as places.json writes it."""
    return {"id": pid, "title": pid, "type": "monumento", "province": "La Habana", "municipio": "Centro Habana",
            "text": "", "photo_target": "", "lat": lat, "lon": lon,
            "box": {"w": w, "d": d, "h": h, "yaw_deg": yaw},
            "scores": {"h": 1.0, "c": 1.0, "v": 1.0, "total": 3.0, "tier": "débil"}}


def at(x, z, pid):
    """A record whose centre projects to world (x, z)."""
    lat, lon = geo.to_latlon([[x, z]])[0]
    return record(pid, lat, lon)


def test_a_place_on_the_origin_is_at_zero():
    p = collect.Places([record()]).by_id("T1")
    assert p.x == pytest.approx(0.0, abs=1e-6) and p.z == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("yaw, axis", [(0.0, (1.0, 0.0)), (90.0, (0.0, -1.0))])
def test_the_w_side_points_east_at_yaw_0_and_north_at_yaw_90(yaw, axis):
    p = collect.Places([record(yaw=yaw)]).by_id("T1")
    assert p.axis == pytest.approx(axis, abs=1e-3)


def test_near_returns_the_places_inside_the_radius_and_none_beyond():
    ps = collect.Places([at(50, 0, "a"), at(0, 140, "b"), at(260, 0, "c"), at(-900, 900, "d")])
    assert sorted(p.id for p in ps.near(0.0, 0.0, 150.0)) == ["a", "b"]
    assert sorted(p.id for p in ps.near(0.0, 0.0, 300.0)) == ["a", "b", "c"]


def test_near_only_looks_at_the_cells_the_circle_touches():
    ps = collect.Places([at(50, 0, "a"), at(5000, 5000, "far")])

    asked = []

    class Counting(dict):
        def get(self, key, default=None):
            asked.append(key)
            return super().get(key, default)

    ps._cells = Counting(ps._cells)
    ps.near(100.0, 100.0, 150.0)  # from x, z = -50 to 250: cells -1, 0 and 1 on each axis
    assert sorted(set(asked)) == [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]


def test_corners_stand_on_the_ground_given():
    p = collect.Places([record(w=40.0, d=10.0, h=15.0)]).by_id("T1")
    c = collect.corners(p, ground=12.0)
    assert c.shape == (8, 3)
    assert sorted({round(y, 6) for y in c[:, 1]}) == [12.0, 27.0]
    assert c[:, 0].max() - c[:, 0].min() == pytest.approx(40.0, abs=0.1)  # yaw 0: w runs along x
    assert c[:, 2].max() - c[:, 2].min() == pytest.approx(10.0, abs=0.1)


def box(**kw):
    """The corners of a test box centred on the origin, base on the ground at 0."""
    return collect.corners(collect.Places([record(**kw)]).by_id("T1"), ground=0.0)


def from_south(gap, zoom=1.0, **kw):
    """A camera 1.5 m up, `gap` metres south of the box's near face, looking at the box's centre."""
    half_depth = kw.get("d", 20.0) / 2
    eye, target = (0.0, 1.5, half_depth + gap), (0.0, kw.get("h", 20.0) / 2, 0.0)
    return collect.framing(eye, target, box(**kw), zoom=zoom)


@pytest.mark.parametrize("gap, zoom, verdict", [(400, 1, "far"), (50, 1, "good"), (8, 1, "near"),
                                                (400, 3, "far"), (150, 3, "good")])
def test_a_box_is_far_good_or_near_with_distance_and_zoom(gap, zoom, verdict):
    assert from_south(gap, zoom).verdict == verdict


def test_a_box_turned_a_quarter_seen_from_the_east_frames_like_its_twin_seen_from_the_south():
    south = from_south(60, w=40.0, d=10.0, h=15.0)
    eye, target = (5.0 + 60, 1.5, 0.0), (0.0, 7.5, 0.0)
    east = collect.framing(eye, target, box(w=40.0, d=10.0, h=15.0, yaw=90.0))
    assert east.verdict == south.verdict == "good"
    assert east.size == pytest.approx(south.size, abs=0.01)
    assert east.inside == pytest.approx(south.inside, abs=0.01)


def test_a_box_behind_the_camera_is_off():
    assert collect.framing((0.0, 1.5, 60.0), (0.0, 1.5, 200.0), box()).verdict == "off"


def test_a_box_off_to_one_side_is_off():
    f = collect.framing((0.0, 1.5, 60.0), (45.0, 10.0, 0.0), box())
    assert f.centre[0] < -collect.CENTRE and f.verdict == "off"


def test_project_puts_the_target_in_the_middle_and_up_on_top():
    pts = collect.project((0.0, 1.5, 50.0), (0.0, 1.5, 0.0), [(0.0, 1.5, 0.0), (0.0, 1.5 + math.tan(math.radians(30)) * 50, 0.0)])
    assert pts[0][:2] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert pts[1][1] == pytest.approx(1.0)  # half the field of view up is the top edge


def test_the_shipped_list_loads_and_finds_the_capitolio_from_the_origin():
    ps = collect.Places.load()
    assert len(ps) >= 700
    assert "Capitolio" in " ".join(p.title for p in ps.near(0.0, 0.0, 200.0))


def test_in_reach_is_six_largest_sides_or_300_m_whichever_is_less():
    small = collect.Places([record(w=20.0, d=10.0, h=8.0)]).by_id("T1")  # 6 x 20 = 120 m
    assert collect.in_reach(small, 0.0, 110.0) and not collect.in_reach(small, 0.0, 130.0)
    big = collect.Places([record(w=200.0, d=100.0, h=30.0)]).by_id("T1")  # 6 x 200 is over 300 m
    assert collect.in_reach(big, 290.0, 0.0) and not collect.in_reach(big, 310.0, 0.0)


def test_a_wall_between_the_camera_and_the_box_hides_it_and_the_box_own_outline_does_not():
    eye = (0.0, 1.5, 60.0)
    c = box()  # 20 x 20 x 20 on the origin: the near face is at z = 10
    assert collect.in_sight(eye, c, np.zeros((0, 2, 2)))
    across = np.array([[[-30.0, 30.0], [30.0, 30.0]]])  # a building front between them
    assert not collect.in_sight(eye, c, across)
    own = np.array([[[-10.0, 10.0], [10.0, 10.0]], [[10.0, 10.0], [10.0, -10.0]]])  # the place's own walls
    assert collect.in_sight(eye, c, own)
    narrow = np.array([[[-1.0, 30.0], [1.0, 30.0]]])  # a lamp post's worth of wall blocks one ray of three
    assert collect.in_sight(eye, c, narrow)


def test_framing_says_hidden_when_walls_stand_in_the_way():
    across = np.array([[[-30.0, 50.0], [30.0, 50.0]]])
    eye, target = (0.0, 1.5, 60.0), (0.0, 10.0, 0.0)
    assert collect.framing(eye, target, box(), walls=across).verdict == "hidden"
    assert collect.framing(eye, target, box(), walls=np.zeros((0, 2, 2))).verdict == "good"


def test_start_zoom_fills_sixty_percent_within_one_and_three():
    def zoom_at(gap):
        eye, target = (0.0, 1.5, 10.0 + gap), (0.0, 10.0, 0.0)
        return collect.start_zoom(eye, target, box())
    assert zoom_at(20.0) == 1.0  # already over 60 %
    z = zoom_at(50.0)
    assert 1.0 < z < 3.0
    eye, target = (0.0, 1.5, 60.0), (0.0, 10.0, 0.0)
    assert collect.framing(eye, target, box(), zoom=z).size == pytest.approx(0.6, abs=0.02)
    assert zoom_at(800.0) == 3.0


def test_a_modelled_landmark_takes_the_box_of_its_mesh():
    ps = collect.Places([record("L0333", w=30.0, d=30.0, h=30.0), record("X")])
    got = ps.with_meshes({"Capitolio": ((-100.0, 2.0, -50.0), (100.0, 94.0, 70.0))})
    cap = got.by_id("L0333")
    assert (cap.w, cap.d, cap.h, cap.x, cap.z) == (200.0, 120.0, 92.0, 0.0, 10.0)
    assert cap.base == 2.0 and got.by_id("X").base is None
    assert got.by_id("X").w == 20.0 and len(got) == 2
    assert [p.id for p in got.near(0.0, 10.0, 5.0)] == ["L0333"]
