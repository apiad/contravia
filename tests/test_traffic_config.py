from datetime import datetime

import pytest

from havana.rules.traffic import TrafficConfig, day_type, hour_share, moving, vehicle_trips

TUE_8 = datetime(2026, 11, 17, 8, 30)


def test_default_weekday_trips_match_the_spec_estimate():
    t = vehicle_trips(TrafficConfig(), "weekday")
    assert 480_000 < sum(t.values()) < 510_000
    assert round(t["car"]) == pytest.approx(166_907, rel=2e-3)
    assert round(t["moto"]) == pytest.approx(111_272, rel=2e-3)
    assert round(t["taxi"]) == pytest.approx(113_705, rel=2e-3)


def test_moving_at_peak_night_and_sunday():
    cfg = TrafficConfig()
    peak = sum(moving(cfg, TUE_8).values())
    assert 9_000 < peak < 12_000
    assert sum(moving(cfg, datetime(2026, 11, 17, 3, 30)).values()) < 500
    assert sum(moving(cfg, datetime(2026, 11, 22, 8, 30)).values()) < peak / 2


def test_no_jump_at_midnight_or_on_the_hour():
    cfg = TrafficConfig()
    a = hour_share(cfg, datetime(2026, 11, 22, 23, 59, 59))
    b = hour_share(cfg, datetime(2026, 11, 23, 0, 0, 1))
    assert abs(a - b) < 0.001
    before = hour_share(cfg, datetime(2026, 11, 17, 7, 59, 59))
    after = hour_share(cfg, datetime(2026, 11, 17, 8, 0, 1))
    assert abs(before - after) < 0.001


def test_settings_scale_and_refuse_nonsense():
    cfg = TrafficConfig()
    double = cfg.updated(population=2 * cfg.population)
    for kind in ("car", "taxi", "moto"):  # buses are their own setting and do not follow population
        assert moving(double, TUE_8)[kind] == pytest.approx(2 * moving(cfg, TUE_8)[kind])
    for bad in ({"populaton": 1}, {"population": -1}, {"hours": {"weekday": [1] * 23}}):
        with pytest.raises(ValueError):
            cfg.updated(**bad)


def test_day_types():
    days = [day_type(datetime(2026, 11, d)) for d in (16, 19, 20, 21, 22)]
    assert days == ["weekday", "weekday", "friday", "saturday", "sunday"]
