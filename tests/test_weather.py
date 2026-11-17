"""The weather chain: same date same weather, no seam at midnight, settings, and the climate it reproduces."""

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from contravia.weather import STATES, SkyConfig, Weather, load_table

TABLE = load_table()


def test_same_date_same_weather_and_the_seed_changes_it():
    a, b = Weather(TABLE), Weather(TABLE)
    days = [date(2026, 9, 1) + timedelta(days=k) for k in range(20)]
    assert all((a.day(d) == b.day(d)).all() for d in days)
    c = Weather(TABLE, SkyConfig(seed=7))
    assert any((a.day(d) != c.day(d)).any() for d in days)


def test_no_seam_at_midnight():
    w = Weather(TABLE)
    same_midnight = same_evening = 0
    for k in range(365):
        d = date(2025, 1, 1) + timedelta(days=k)
        prev, cur = w.day(d - timedelta(days=1)), w.day(d)
        same_midnight += prev[23] == cur[0]
        same_evening += cur[22] == cur[23]
    assert abs(same_midnight - same_evening) / 365 < 0.1


def test_forced_state_holds_until_it_ends():
    t = datetime(2026, 2, 10, 12, 0)
    w = Weather(TABLE, SkyConfig(force="storm", force_until=t + timedelta(minutes=30)))
    assert w.at(t).state == "storm" and w.at(t).rain > 0.5
    w.configure(w.config.updated(force=None))
    assert w.at(t).state in STATES


def test_settings_refuse_bad_values():
    with pytest.raises(ValueError, match="unknown"):
        SkyConfig().updated(fog=3)
    with pytest.raises(ValueError, match="rain_factor"):
        SkyConfig().updated(rain_factor=[1, 2])
    with pytest.raises(ValueError, match="force"):
        SkyConfig().updated(force="snow")
    assert SkyConfig().updated(rain_factor=2).rain_factor == (2.0,) * 12


def test_rain_factor_brings_more_rain():
    base, wet = Weather(TABLE), Weather(TABLE, SkyConfig().updated(rain_factor=3))
    days = [date(2026, 7, 1) + timedelta(days=k) for k in range(60)]
    assert sum((wet.day(d) >= 3).sum() for d in days) > 1.5 * sum((base.day(d) >= 3).sum() for d in days)


def test_storm_blends_in_and_roads_dry_after():
    w = Weather(TABLE)
    for k in range(400):  # find a dry hour followed by a rain hour
        d = date(2026, 6, 1) + timedelta(days=k // 24)
        h = k % 24
        if h < 23 and w.day(d)[h] < 3 and w.day(d)[h + 1] >= 3:
            start = datetime(d.year, d.month, d.day, h + 1)
            assert w.at(start).rain < w.at(start + timedelta(minutes=10)).rain
            assert w.at(start + timedelta(minutes=30)).wet == 1.0
            break
    else:
        pytest.fail("no rain in June and July")


@pytest.mark.slow
def test_thirty_years_reproduce_the_table():
    w = Weather(TABLE)
    states, months, hours, days_ = [], [], [], []
    for k in range(30 * 365):
        d = date(1991, 1, 1) + timedelta(days=k)
        s = w.day(d)
        states.append(s)
        months.append(np.full(24, d.month - 1))
        days_.append(np.full(24, k))
        hours.append(np.arange(24))
    s, m, h, dd = map(np.concatenate, (states, months, hours, days_))
    check = TABLE["check"]
    for mo in range(12):
        sel = m == mo
        rain_days = len(np.unique(dd[sel & (s >= 3)])) / 30
        assert rain_days == pytest.approx(check["rain_days"][mo], rel=0.15, abs=0.8), mo
        cloud = np.array(TABLE["cloud"])[mo][s[sel]].mean()
        assert cloud == pytest.approx(check["cloud"][mo], abs=0.06), mo
        peak = int(np.argmax([(s[sel & (h == hh)] >= 3).mean() for hh in range(24)]))
        if check["rain_days"][mo] > 6:
            assert abs(peak - check["peak_hour"][mo]) <= 1, mo
