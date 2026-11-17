"""The sky's positions against PyEphem.

Reference values were computed once with PyEphem 4.2 (pressure 0, so no refraction; horizon -0:50 for
sunrise and sunset) for Havana at 23.136 N, 82.359 W. PyEphem is not a dependency: the numbers are below.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from contravia import sky


def _crossings(day: datetime, body=sky.sun, horizon=-0.833):
    """Local times (HH:MM) at which the body crosses the horizon during the day, by one-minute steps."""
    out, prev = [], None
    for m in range(24 * 60 + 1):
        t = day + timedelta(minutes=m)
        alt = body(t).alt
        if prev is not None and (prev < horizon) != (alt < horizon):
            out.append(m)
        prev = alt
    return out


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


@pytest.mark.parametrize("day,rise,set_,noon_alt", [
    (datetime(2026, 3, 9), "07:42", "19:37", 62.57),     # first full day of summer time
    (datetime(2026, 6, 21), "06:43", "20:19", 89.70),
    (datetime(2026, 11, 15), "06:41", "17:46", 48.25),
    (datetime(2026, 12, 21), "07:04", "17:50", 43.42),
])
def test_sun_rises_sets_and_culminates_as_pyephem_says(day, rise, set_, noon_alt):
    r, s = _crossings(day, horizon=-0.833 - 0.267)  # PyEphem times the upper limb at -0:50: the centre is 16' lower
    assert abs(r - _minutes(rise)) <= 2 and abs(s - _minutes(set_)) <= 2
    best = max(sky.sun(day + timedelta(minutes=m)).alt for m in range(11 * 60, 15 * 60))
    assert best == pytest.approx(noon_alt, abs=0.2)


def test_the_morning_sun_comes_from_the_east_of_the_map():
    s = sky.sun(datetime(2026, 11, 15, 8, 0))
    assert s.alt == pytest.approx(15.29, abs=0.2) and s.az == pytest.approx(118.41, abs=0.3)
    d = sky.direction(s.alt, s.az)
    assert d[0] > 0.5 and d[1] > 0  # east is +x, up is +y


@pytest.mark.parametrize("when,alt,az,lit", [
    (datetime(2026, 11, 15, 20, 0), 33.00, 223.22, 0.365),
    (datetime(2026, 6, 21, 22, 0), 44.92, 237.98, 0.521),
])
def test_moon_position_and_phase(when, alt, az, lit):
    m = sky.moon(when)
    assert m.alt == pytest.approx(alt, abs=0.5) and m.az == pytest.approx(az, abs=0.6)
    assert m.lit == pytest.approx(lit, abs=0.03)


def test_moon_rises_and_sets_on_time():
    r, s = _crossings(datetime(2026, 11, 15), sky.moon, horizon=-0.27)  # the moon's half disc, no refraction
    assert abs(r - _minutes("11:57")) <= 10 and abs(s - _minutes("22:58")) <= 10


def test_moon_is_full_and_new_on_the_right_days():
    assert sky.moon(datetime(2026, 11, 24, 9, 53)).lit > 0.99  # full moon 14:53 UTC
    assert sky.moon(datetime(2026, 11, 9, 2, 2)).lit < 0.01  # new moon 07:02 UTC


@pytest.mark.parametrize("name,when,alt,az", [
    ("venus", datetime(2026, 11, 15, 5, 30), 13.01, 107.18),
    ("jupiter", datetime(2026, 11, 15, 5, 30), 69.92, 115.06),
    ("mars", datetime(2026, 11, 15, 5, 30), 70.42, 111.78),
    ("saturn", datetime(2026, 11, 15, 20, 0), 59.82, 134.37),
])
def test_planets_are_where_pyephem_puts_them(name, when, alt, az):
    p = sky.planets(when)[name]
    assert p.alt == pytest.approx(alt, abs=0.5)
    assert (p.az - az + 180) % 360 - 180 == pytest.approx(0, abs=0.8)


def test_venus_and_jupiter_are_the_brightest_planets():
    ps = sky.planets(datetime(2026, 11, 15, 5, 30))
    assert ps["venus"].mag == pytest.approx(-4.3, abs=0.4) and ps["jupiter"].mag == pytest.approx(-1.9, abs=0.4)


# J2000 places from the Hipparcos catalogue.
STARS = {"sirius": (101.287155, -16.716116), "polaris": (37.954561, 89.264109),
         "betelgeuse": (88.792939, 7.407064), "vega": (279.234735, 38.783689)}


@pytest.mark.parametrize("name,when,alt,az", [
    ("sirius", datetime(2026, 11, 15, 23, 0), 11.93, 114.24),
    ("polaris", datetime(2026, 11, 15, 23, 0), 23.74, 0.17),
    ("betelgeuse", datetime(2026, 11, 15, 23, 0), 33.85, 96.76),
    ("vega", datetime(2026, 6, 21, 22, 0), 35.99, 57.86),
])
def test_stars_after_precession(name, when, alt, az):
    ra, dec = sky.precess(*STARS[name], when)
    a, z = sky.altaz(ra, dec, when)
    assert float(a) == pytest.approx(alt, abs=0.2)
    assert (float(z) - az + 180) % 360 - 180 == pytest.approx(0, abs=0.3)


def test_rotation_matches_altaz_for_a_catalogue():
    when = datetime(2026, 11, 15, 23, 0)
    ra, dec = np.array([10.0, 101.3, 250.0]), np.array([-40.0, -16.7, 60.0])
    by_matrix = sky.unit_vectors(ra, dec) @ sky.equatorial_to_local(when).T
    by_altaz = sky.direction(*sky.altaz(ra, dec, when))
    assert np.allclose(by_matrix, by_altaz, atol=1e-9)


def test_polaris_stays_at_havanas_latitude():
    for h in (0, 6, 12, 18):
        ra, dec = sky.precess(*STARS["polaris"], datetime(2026, 3, 1, h))
        assert float(sky.altaz(ra, dec, datetime(2026, 3, 1, h))[0]) == pytest.approx(sky.LAT, abs=1.0)


def test_stars_rise_in_the_east():
    ra, dec = sky.precess(*STARS["sirius"], datetime(2026, 11, 15))
    a1 = float(sky.altaz(ra, dec, datetime(2026, 11, 15, 22, 0))[0])
    a2 = float(sky.altaz(ra, dec, datetime(2026, 11, 15, 23, 0))[0])
    assert a2 > a1


def test_no_jump_at_the_clock_change():
    # Cuba moved its clocks from 00:00 to 01:00 on 8 March 2026, so 23:59 and 01:00 are one real minute
    # apart: the sun must barely move between them.
    before = sky.sun(datetime(2026, 3, 7, 23, 59))
    after = sky.sun(datetime(2026, 3, 8, 1, 0))
    assert abs(after.alt - before.alt) < 1.0
