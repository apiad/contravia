from datetime import datetime

import pytest

import numpy as np

from contravia.rules.fleet import FAMILIES, KINDS
from contravia.rules.traffic import (
    FAMILY_SHARE,
    MIX,
    TrafficConfig,
    day_type,
    hour_share,
    look_kinds,
    look_shares,
    looks,
    moving,
    pick,
    vehicle_trips,
)

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
    for mode in ("car", "taxi", "moto"):  # buses are their own setting and do not follow population
        assert vehicle_trips(double, "weekday")[mode] == pytest.approx(2 * vehicle_trips(cfg, "weekday")[mode])
    for bad in ({"populaton": 1}, {"population": -1}, {"hours": {"weekday": [1] * 23}}, {"mix": {"tank": 1}},
                {"families": {"car": 1}}, {"mix": {"car": -1}}):
        with pytest.raises(ValueError):
            cfg.updated(**bad)


def test_day_types():
    days = [day_type(datetime(2026, 11, d)) for d in (16, 19, 20, 21, 22)]
    assert days == ["weekday", "weekday", "friday", "saturday", "sunday"]


def test_the_mix_covers_the_fleet_and_sums_to_100():
    assert set(MIX) == set(KINDS) and sum(MIX.values()) == pytest.approx(100)
    assert set(FAMILY_SHARE) == set(FAMILIES)
    for kind in KINDS:
        assert sum(FAMILY_SHARE[f.name] for f in FAMILIES.values() if f.kind == kind) == pytest.approx(100), kind


def test_moving_splits_the_demand_by_the_mix():
    cfg = TrafficConfig()
    now = moving(cfg, TUE_8)
    assert list(now) == list(KINDS)
    assert now["moto"] / now["car"] == pytest.approx(30 / 26)


def test_look_shares_follow_the_mix_and_one_share_can_change():
    cfg = TrafficConfig()
    p, kind = look_shares(cfg), look_kinds()
    assert p.sum() == pytest.approx(1)
    for k, name in enumerate(KINDS):
        assert p[kind == k].sum() == pytest.approx(MIX[name] / 100), name
    no_bikes = look_shares(cfg.updated(mix={"bike": 0}))
    assert no_bikes[kind == KINDS.index("bike")].sum() == 0
    assert no_bikes[kind == KINDS.index("car")].sum() > p[kind == KINDS.index("car")].sum()
    coaches = [i for i, look in enumerate(looks()) if look[1] == "interprovincial"]
    assert p[coaches].sum() == 0  # no coaches in the city


def test_the_same_street_and_minute_draw_the_same_looks():
    cfg = TrafficConfig()
    a = pick(cfg, 42, TUE_8, 50)
    assert np.array_equal(a, pick(cfg, 42, TUE_8.replace(second=40), 50))
    assert not np.array_equal(a, pick(cfg, 43, TUE_8, 50))
    many = pick(cfg, 7, TUE_8, 20_000)
    share = np.bincount(look_kinds()[many], minlength=len(KINDS)) / len(many)
    assert share[KINDS.index("moto")] == pytest.approx(0.30, abs=0.02)
