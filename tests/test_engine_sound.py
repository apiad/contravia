"""Engines synthesized from the rpm and the throttle, one sound row per fleet family."""

import time

import numpy as np
import pytest
from euterpe.sound import render
from scipy.signal import find_peaks

from contravia.audio.engine import cycle, engine, pulses
from contravia.rules.fleet import FAMILIES
from contravia.rules.sound import FAMILY_SOUND

RATE = 48000


def fixed(values):
    return lambda name: values.get(name, 0.0)


def peak_hz(x, lo, hi):
    w = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    band = (f >= lo) & (f <= hi)
    return float(f[band][np.argmax(w[band])])


def centroid(x):
    w = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    return float((w * f).sum() / w.sum())


def test_every_family_has_a_sound_row():
    assert set(FAMILIES) <= set(FAMILY_SOUND)


def test_a_cycle_holds_one_pulse_per_cylinder():
    assert len(find_peaks(cycle(8, 0.35, seed=1), height=0.2)[0]) == 8
    assert len(find_peaks(cycle(4, 0.08, seed=1), height=0.2)[0]) == 4


def test_a_cycle_is_seeded():
    np.testing.assert_array_equal(cycle(4, .08, 3), cycle(4, .08, 3))
    assert not np.array_equal(cycle(4, .3, 3), cycle(4, .3, 4))


@pytest.mark.parametrize("fam", sorted(f for f, s in FAMILY_SOUND.items() if s.strokes))
def test_the_firing_frequency_follows_the_rpm(fam):
    s = FAMILY_SOUND[fam]
    x = render(pulses(s, fixed({"car.rpm": 2400.0, "car.throttle": 0.5}), seed=1), 2.0)
    want = 2400 * s.cylinders / (60 * s.strokes / 2)
    assert abs(peak_hz(x, 10, 1000) / want - 1) < 0.03


def test_more_throttle_is_louder_and_brighter():
    s = FAMILY_SOUND["almendron"]
    lo = render(engine(s, 600, 4600, fixed({"car.rpm": 3000.0, "car.throttle": 0.1}), seed=1), 1.0)
    hi = render(engine(s, 600, 4600, fixed({"car.rpm": 3000.0, "car.throttle": 1.0}), seed=1), 1.0)
    assert np.sqrt(np.mean(hi**2)) > np.sqrt(np.mean(lo**2)) and centroid(hi) > centroid(lo)


@pytest.mark.parametrize("fam", sorted(FAMILY_SOUND))
def test_every_family_renders(fam):
    x = render(engine(FAMILY_SOUND[fam], 700, 4000, fixed({"car.rpm": 2000.0, "car.throttle": 0.5,
                                                          "car.speed": 8.0}), seed=2), 0.5)
    assert np.isfinite(x).all() and 1e-4 < np.abs(x).max() <= 1.0


def test_an_engine_is_seeded():
    s = FAMILY_SOUND["ruso"]
    ctl = fixed({"car.rpm": 2500.0, "car.throttle": 0.0})
    np.testing.assert_array_equal(render(engine(s, 850, 6000, ctl, seed=5), 0.5),
                                  render(engine(s, 850, 6000, ctl, seed=5), 0.5))


@pytest.mark.slow
def test_the_players_engine_fits_its_budget():
    u = engine(FAMILY_SOUND["almendron"], 600, 4600, fixed({"car.rpm": 3000.0, "car.throttle": 0.7}), seed=1)
    t = time.perf_counter()
    render(u, 10.0, block=512)
    assert time.perf_counter() - t < 1.5
