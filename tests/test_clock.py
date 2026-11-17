from datetime import datetime, timedelta

import pytest

from caiman.clock import Clock, FlowRunner, parse_at
from caiman.flow import Flow
from caiman.prepare import build_city, read_osm
from caiman.rules.traffic import TrafficConfig

TUE_8 = datetime(2026, 11, 17, 8, 30)


class FakeClock:
    def __init__(self, at):
        self.t = at

    def now(self):
        return self.t


@pytest.fixture(scope="module")
def graph():
    return build_city(read_osm("tests/fixtures/traffic.osm"))[0].graph


def test_tick_steps_once_per_simulated_second(graph):
    clock = FakeClock(TUE_8)
    runner = FlowRunner(Flow(graph, TrafficConfig()), clock)
    runner.warm_up(minutes=1)
    clock.t += timedelta(seconds=5)
    assert runner.tick() == 5
    assert runner.view.when == clock.t


def test_submitted_calls_run_in_order_before_the_next_step(graph):
    clock = FakeClock(TUE_8)
    runner = FlowRunner(Flow(graph, TrafficConfig()), clock)
    runner.warm_up(minutes=1)
    seen = []
    runner.submit(lambda flow: seen.append(("a", flow.when)))
    runner.submit(lambda flow: seen.append(("b", flow.when)))
    clock.t += timedelta(seconds=2)
    runner.tick()
    assert [s[0] for s in seen] == ["a", "b"] and seen[0][1] == TUE_8


def test_warm_up_ends_at_the_clock(graph):
    clock = FakeClock(TUE_8)
    runner = FlowRunner(Flow(graph, TrafficConfig()), clock)
    runner.warm_up(minutes=30)
    assert runner.view.when == TUE_8
    assert runner.view.n.sum() > 0


def test_a_long_pause_does_not_step_for_hours(graph):
    clock = FakeClock(TUE_8)
    runner = FlowRunner(Flow(graph, TrafficConfig()), clock)
    runner.warm_up(minutes=1)
    clock.t += timedelta(hours=5)
    assert runner.tick() <= FlowRunner.MAX_STEPS
    assert runner.view.when == clock.t


def test_parse_at_gives_that_weekday_and_time():
    t = parse_at("Sun 08:00")
    assert t.weekday() == 6 and (t.hour, t.minute) == (8, 0)
    assert parse_at("tue 17:45").weekday() == 1


def test_clock_follows_the_machine_or_a_fixed_start():
    assert abs((Clock().now() - datetime.now()).total_seconds()) < 1
    fixed = Clock(at=TUE_8, warp=60)
    assert fixed.now() >= TUE_8
    assert fixed.now() - TUE_8 < timedelta(minutes=1)


def test_a_bare_time_is_today_at_that_time():
    from datetime import datetime

    at = parse_at("10:00")
    assert (at.date(), at.hour, at.minute) == (datetime.now().date(), 10, 0)
