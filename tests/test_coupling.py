from datetime import datetime, timedelta

import numpy as np
import pytest

from havana.clock import FlowRunner
from havana.flow import Flow
from havana.prepare import build_city, read_osm
from havana.rules.traffic import TrafficConfig
from havana.traffic import TrafficLayer

TUE_8 = datetime(2026, 11, 17, 8, 30)


class FakeClock:
    def __init__(self, at):
        self.t = at

    def now(self):
        return self.t


@pytest.fixture(scope="module")
def graph():
    return build_city(read_osm("tests/fixtures/traffic.osm"))[0].graph


def setup(graph, fill=0.3, radius=120.0):
    clock = FakeClock(TUE_8)
    flow = Flow(graph, TrafficConfig())
    flow.demand = False
    flow.n[:] = fill * flow.jam
    runner = FlowRunner(flow, clock)
    runner.view = flow.view()
    flow.when = TUE_8
    return clock, flow, runner, TrafficLayer(graph, runner, radius=radius, seed=2)


def advance(clock, runner, layer, x, z, seconds=1.0, dt=0.1):
    for _ in range(int(seconds / dt)):
        clock.t += timedelta(seconds=dt)
        layer.update(x, z, dt, [])
        runner.tick()


def test_an_edge_entering_range_gets_the_flow_vehicles(graph):
    _clock, flow, runner, layer = setup(graph)
    expected = np.floor(flow.n.copy())
    layer.update(100.0, 100.0, 0.0, [])
    runner.tick()
    layer.update(100.0, 100.0, 0.0, [])
    for e in np.flatnonzero(layer.owned):
        assert (layer.cars.edge == e).sum() == expected[e]
    queued = layer.cars.v == 0
    assert queued.any()


def test_moving_across_and_back_conserves_vehicles(graph):
    clock, _flow, runner, layer = setup(graph)
    before = layer.total()
    for x in (0.0, 100.0, 200.0, 2000.0, 100.0, -1500.0, 100.0):
        advance(clock, runner, layer, x, 100.0, seconds=3)
    assert layer.total() == pytest.approx(before, abs=1e-6)


def test_counts_of_owned_edges_reach_the_flow(graph):
    clock, flow, runner, layer = setup(graph)
    advance(clock, runner, layer, 100.0, 100.0, seconds=2)
    layer.update(100.0, 100.0, 0.0, [])
    runner.tick()
    owned = np.flatnonzero(layer.owned)
    counts = np.bincount(layer.cars.edge, minlength=len(flow.n))[owned]
    assert np.allclose(flow.n[owned], counts + layer.carry[owned] + layer.waiting[owned] + flow.arrived[owned])


def test_a_car_leaving_the_circle_returns_to_the_flow(graph):
    clock, flow, runner, layer = setup(graph, fill=0.0, radius=50.0)
    layer.update(100.0, 100.0, 0.0, [])
    runner.tick()
    layer.update(100.0, 100.0, 0.0, [])
    exits = [(int(e), o) for e in np.flatnonzero(layer.owned) for o in layer.cars.turns[e] if not layer.owned[o]]
    assert exits, "the fixture needs an edge in the circle that leads out of it"
    e, out = exits[0]
    outside = [out]
    layer.cars.add([e], [layer.cars.len[e] - 0.5], [5.0], [0])
    layer.cars.next_edge[-1] = outside[0]
    layer.cars.cleared[-1] = True
    before = flow.n[outside[0]]
    advance(clock, runner, layer, 100.0, 100.0, seconds=0.5)
    assert len(layer.cars) == 0
    assert flow.n[outside[0]] == pytest.approx(before + 1, abs=0.2)


def test_a_teleport_enters_a_limited_number_of_edges_per_update(graph):
    _clock, _flow, runner, layer = setup(graph)
    layer.MAX_ENTER = 3
    before = layer.total()
    layer.update(5000.0, 5000.0, 0.0, [])
    runner.tick()
    layer.update(100.0, 100.0, 0.0, [])
    assert layer.stats()["edges"] <= 3
    for _ in range(20):
        layer.update(100.0, 100.0, 0.0, [])
        runner.tick()
    assert layer.stats()["edges"] > 3
    assert layer.total() == pytest.approx(before, abs=1e-6)
