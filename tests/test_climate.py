"""havana climate's readers and the weather table, on small made-up inputs."""

import struct

import numpy as np
import pytest

from havana.climate import build_table, read_bsc5, sample_milky_way


def _bsc5(records):
    head = struct.pack("<7i", 0, 1, -len(records), 1, 1, 1, 32)
    body = b"".join(struct.pack("<fdd2shff", n, np.radians(ra), np.radians(dec), sp, int(mag * 100), 0.0, 0.0)
                    for n, ra, dec, sp, mag in records)
    return head + body


def test_bsc5_keeps_visible_stars_with_a_colour():
    s = read_bsc5(_bsc5([(1, 101.29, -16.72, b"A1", -1.46), (2, 88.79, 7.41, b"M1", 0.5), (3, 10.0, 5.0, b"G2", 7.2)]))
    assert len(s) == 2
    assert s["ra"][0] == pytest.approx(101.29, abs=1e-4) and s["dec"][0] == pytest.approx(-16.72, abs=1e-4)
    assert s["mag"][0] == pytest.approx(-1.46) and s["bv"][1] > s["bv"][0]  # Betelgeuse redder than Sirius


def test_milky_way_samples_land_on_the_bright_pixel():
    img = np.full((4, 8, 3), 10, np.uint8)
    img[1, 6] = 250  # column 6 of 8: u = 0.8125 -> RA = 180 - 292.5 = -112.5 -> 247.5; row 1 of 4: Dec 22.5..45
    pts = sample_milky_way(img, n=200)
    assert np.all((pts["ra"] > 225) & (pts["ra"] < 270)) and np.all((pts["dec"] > 0) & (pts["dec"] < 45))
    assert pts["w"].min() > 0.9


def _hours(years=2):
    t = np.arange(np.datetime64("2001-01-01T00"), np.datetime64(f"{2001 + years}-01-01T00"), np.timedelta64(1, "h"))
    month = t.astype("datetime64[M]").astype(int) % 12
    hour = (t - t.astype("datetime64[D]")).astype(int)
    rain = np.where((month == 5) & (hour == 16), 5.0, 0.0)  # June: one wet hour at 16:00 every day
    cloud = np.where(month == 5, 70.0, 10.0)
    return t, cloud, rain


def test_table_finds_junes_afternoon_rain_and_a_dry_january():
    t = build_table(*_hours(), gauge={})
    assert t["check"]["peak_hour"][5] == 16 and t["check"]["rain_days"][5] == pytest.approx(30, abs=0.5)
    assert t["check"]["rain_days"][0] == 0
    T = np.array(t["transitions"])
    assert T.shape == (12, 24, 5, 5) and np.allclose(T.sum(axis=3), 1.0)
    assert T[5, 15, 2, 3] + T[5, 15, 2, 4] > 0.9  # in June, overcast at 15:00 turns to rain at 16:00


def test_gauge_lowers_rain_days():
    t, cloud, rain = _hours()
    rain = rain.copy()
    month = t.astype("datetime64[M]").astype(int) % 12
    day = (t.astype("datetime64[D]").astype(int))
    rain[(month == 5) & (day % 2 == 0) & (rain > 0)] = 0.5  # every other June day only drizzles
    table = build_table(t, cloud, rain, gauge={6: 15.0})
    assert table["check"]["rain_days"][5] == pytest.approx(15, abs=1.0) and table["threshold_mm"][5] > 0.4
