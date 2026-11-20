"""The traffic heard around the car: the four loudest engines, a bed, horns and air brakes."""

import numpy as np
from euterpe.sound import render

from contravia.audio.traffic import Honks, bed, brakes, cheap_engine, honk, loudest, virtual_rpm


def fixed(values):
    return lambda name: values.get(name, 0.0)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def test_loudest_picks_near_and_heavy():
    xyz = np.array([[10.0, 0, 0], [12.0, 0, 0], [200.0, 0, 0], [11.0, 0, 0], [60.0, 0, 0], [13.0, 0, 0]])
    speed = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 0.0])
    kinds = np.array(["car", "bus", "truck", "car", "truck", "car"])
    got = loudest(xyz, speed, kinds, np.zeros(3), k=4)
    assert len(got) == 4 and got[0] == 1 and 2 not in got


def test_loudest_with_fewer_cars_than_slots():
    assert len(loudest(np.zeros((2, 3)), np.ones(2), np.array(["car", "car"]), np.zeros(3), k=4)) == 2


def test_virtual_rpm_stays_between_idle_and_red_and_drops_at_a_shift():
    speeds = np.arange(0, 30, 0.25)
    r = np.array([virtual_rpm(v, 0.5, "car") for v in speeds])
    assert r.min() >= 800 and r.max() <= 6000 and (np.diff(r) < -500).sum() >= 2
    assert virtual_rpm(10.0, 2.0, "car") > virtual_rpm(10.0, 0.0, "car")


def test_trucks_shift_sooner():
    v = np.arange(0, 20, 0.25)
    assert (np.diff([virtual_rpm(s, 0.5, "truck") for s in v]) < -300).sum() > \
        (np.diff([virtual_rpm(s, 0.5, "car") for s in v]) < -300).sum()


def test_a_car_honks_once_per_wait():
    h = Honks()
    near = np.array([True, True, False])
    assert list(h.due(np.array([3.0, 0.0, 9.0]), near)) == []
    assert list(h.due(np.array([4.5, 0.0, 9.0]), near)) == [0]
    assert list(h.due(np.array([6.0, 0.0, 9.0]), near)) == []
    assert list(h.due(np.array([0.0, 0.0, 9.0]), near)) == []
    assert list(h.due(np.array([4.2, 0.0, 9.0]), near)) == [0]


def test_the_bed_follows_the_count():
    a = rms(render(bed(fixed({"traffic.count": 2.0})), 0.5))
    b = rms(render(bed(fixed({"traffic.count": 20.0})), 0.5))
    assert rms(render(bed(fixed({})), 0.5)) < 1e-4 and b > 3 * a > 0


def test_a_slot_is_silent_when_off_and_follows_its_rpm():
    off = render(cheap_engine(0, fixed({"traffic.0.rpm": 2000.0, "traffic.0.cyl": 4.0})), 0.5)
    on = render(cheap_engine(0, fixed({"traffic.0.rpm": 2000.0, "traffic.0.cyl": 4.0, "traffic.0.on": 1.0,
                                       "traffic.0.bright": 1500.0})), 1.0)
    assert rms(off) < 1e-4 and rms(on) > 1e-3


def test_one_shots_end():
    for u in (brakes(1), honk("guagua_vieja", 1), honk("ruso", 2)):
        assert 0.2 < u.duration < 2.0 and rms(render(u, u.duration)) > 1e-3


def test_the_frame_fills_the_slots_and_counts_the_rest():
    from test_sound import Recorder, fake_game

    from contravia.audio import CONTROLS, Sound
    from contravia.rules.traffic import looks

    s = Sound(Recorder(), CONTROLS)
    s.audio.controls = type("C", (), {"get": staticmethod(lambda name: 0.0)})()
    g = fake_game(1000.0)
    n = 9
    xyz = np.c_[np.arange(n) * 8.0 + 5, np.zeros(n), np.zeros(n)]
    car_look = next(i for i, lk in enumerate(looks()) if lk[0] == "car")
    g.traffic_now = (xyz, np.zeros(n), np.full(n, 8.0), np.full(n, car_look), np.zeros(n))
    s.frame(g)
    v = s.audio.values
    assert [v[f"traffic.{i}.on"] for i in range(4)] == [1.0] * 4 and v["traffic.count"] == 5.0
    assert v["traffic.0.rpm"] > 0
