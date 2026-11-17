import pytest

from contravia.input import Driver

DT = 1 / 60


def hold(d, secs, left=False, right=False, up=False, down=False):
    c = None
    for _ in range(round(secs / DT)):
        c = d.update(left, right, up, down, False, DT)
    return c


def test_the_wheel_reaches_full_lock_in_the_rise_time_and_not_before():
    d = Driver()
    assert hold(d, 0.15, right=True).steer == pytest.approx(0.5, abs=0.02)
    assert hold(d, 0.15, right=True).steer == pytest.approx(1.0)
    assert hold(d, 1.0, right=True).steer == 1.0


def test_the_wheel_returns_to_centre_on_its_own():
    d = Driver()
    hold(d, 0.5, left=True)
    assert d.steer == -1.0
    assert hold(d, 0.1).steer == pytest.approx(-1 + 0.1 / 0.18)
    assert hold(d, 0.1).steer == 0.0


def test_the_other_key_returns_at_the_return_rate_then_turns_at_the_rise_rate():
    d = Driver()
    hold(d, 0.5, right=True)
    assert hold(d, 0.18, left=True).steer == pytest.approx(0.0, abs=0.02)
    assert hold(d, 0.30, left=True).steer == pytest.approx(-1.0, abs=0.02)


def test_both_keys_together_centre_the_wheel():
    d = Driver()
    hold(d, 0.5, right=True)
    assert hold(d, 0.5, left=True, right=True).steer == 0.0


def test_the_pedals_ramp_up_and_drop_at_once():
    d = Driver()
    assert hold(d, 0.1, up=True).throttle == pytest.approx(0.5, abs=0.02)
    assert hold(d, 0.2, up=True).throttle == 1.0
    c = hold(d, DT, down=True)
    assert c.throttle == 0.0
    assert 0 < c.brake < 0.1
