"""Tyres, wind, crashes and the horn: graphs read from the car's controls, and one-shots."""

import numpy as np
from euterpe.sound import render

from contravia.audio.car import CRASH_MIN, GLASS_MIN, bell, bump, crash, horn, roll, squeal, wind


def fixed(values):
    return lambda name: values.get(name, 0.0)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def test_squeal_is_silent_below_the_peak_and_loud_above():
    assert rms(render(squeal(fixed({"car.slip": 0.0})), 0.5)) < 1e-4
    assert rms(render(squeal(fixed({"car.slip": 0.5})), 0.5)) > 1e-2


def test_a_wet_road_squeals_less():
    dry = rms(render(squeal(fixed({"car.slip": 0.5})), 0.5))
    wet = rms(render(squeal(fixed({"car.slip": 0.5, "car.wet": 1.0})), 0.5))
    assert wet < 0.6 * dry


def test_roll_rises_with_speed():
    slow = rms(render(roll(fixed({"car.speed": 3.0})), 0.5))
    fast = rms(render(roll(fixed({"car.speed": 25.0})), 0.5))
    assert fast > 2 * slow > 0


def test_each_surface_rolls_differently():
    x = [render(roll(fixed({"car.speed": 15.0, "car.surface": float(k)})), 0.5) for k in range(5)]
    assert len({round(rms(a), 4) for a in x}) >= 4


def test_wind_rises_with_speed_and_drops_in_the_cabin():
    slow = rms(render(wind(fixed({"car.speed": 5.0})), 0.5))
    fast = rms(render(wind(fixed({"car.speed": 35.0})), 0.5))
    inside = rms(render(wind(fixed({"car.speed": 35.0, "car.cabin": 1.0})), 0.5))
    assert fast > 4 * slow and inside < 0.5 * fast


def test_twice_the_impulse_is_louder_and_a_crash_ends():
    a, b = crash(CRASH_MIN * 2, seed=1), crash(CRASH_MIN * 4, seed=1)
    assert a.duration < 3.0 and rms(render(b, 1.0)) > rms(render(a, 1.0))


def test_a_crash_is_seeded():
    np.testing.assert_array_equal(render(crash(900.0, seed=7), 0.5), render(crash(900.0, seed=7), 0.5))


def test_glass_only_past_its_threshold():
    def highs(x):
        w = np.abs(np.fft.rfft(x))
        f = np.fft.rfftfreq(len(x), 1 / 48000)
        return w[f > 5000].sum() / w.sum()

    assert highs(render(crash(GLASS_MIN * 1.2, seed=1), 0.5)) > 2 * highs(render(crash(GLASS_MIN * 0.8, seed=1), 0.5))


def test_the_horn_sounds_only_while_held():
    for family in ("almendron", "ruso", "guagua_vieja", "moto_dos_tiempos"):
        assert rms(render(horn(family, fixed({"car.horn": 0.0})), 0.3)) < 1e-4
        assert rms(render(horn(family, fixed({"car.horn": 1.0})), 0.3)) > 1e-2


def test_bump_and_bell_are_short_and_audible():
    for u in (bump(0.05, seed=1), bell(seed=1)):
        assert u.duration < 2.0 and rms(render(u, u.duration)) > 1e-3
