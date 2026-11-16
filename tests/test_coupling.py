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
        assert (layer.cars.edge == e).sum() in (expected[e], expected[e] + 1)  # rounded at random
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
    transit = np.zeros(len(flow.n))  # arrivals the flow handed over that the layer has not read yet
    for kind, edges, k in (m[:3] for m in layer.inbox):
        if kind == "arrive":
            np.add.at(transit, edges, k)
    assert np.allclose(flow.n[owned], counts + layer.carry[owned] + layer.waiting[owned] + flow.arrived[owned]
                       + transit[owned])


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


def test_accident_cars_stay_while_the_player_is_away(graph):
    clock, _flow, runner, layer = setup(graph)
    advance(clock, runner, layer, 100.0, 100.0, seconds=1)
    e = int(np.bincount(layer.cars.edge).argmax())
    layer.accident(e, minutes=40)
    held = int((layer.cars.edge == e).sum())
    before = layer.total()
    advance(clock, runner, layer, 5000.0, 5000.0, seconds=1)  # away
    assert len(layer.cars) == 0 and sum(len(v) for v in layer.stored.values()) == held
    advance(clock, runner, layer, 100.0, 100.0, seconds=1)  # back
    assert ((layer.cars.edge == e) & layer.cars.stopped).sum() == held
    assert layer.total() == pytest.approx(before, abs=1e-6)
    clock.t += timedelta(minutes=41)
    advance(clock, runner, layer, 100.0, 100.0, seconds=0.2)
    assert not layer.cars.stopped.any()


def test_locate_finds_the_edge_and_direction_of_a_vehicle(graph):
    _, _, _, layer = setup(graph)
    a = graph.a
    e = int(np.flatnonzero(a["g_len"] > 50)[0])
    x, z, heading = graph.points(np.array([e]), np.array([30.0]), np.array([1.5]))
    got = layer.locate(float(x[0]), float(z[0]), float(heading[0]))
    assert got is not None and got[0] == e and got[2] == pytest.approx(30.0, abs=1.0)
    assert layer.locate(float(x[0]) + 500, float(z[0]) + 500, float(heading[0])) is None


def test_edges_with_fractions_of_a_vehicle_still_get_cars(graph):
    _clock, flow, runner, layer = setup(graph, fill=0.0)
    flow.n[:] = 0.3  # less than one vehicle on every edge, as on most streets of Havana
    before = layer.total()
    layer.update(100.0, 100.0, 0.0, [])
    runner.tick()
    layer.update(100.0, 100.0, 0.0, [])
    owned = int(layer.owned.sum())
    assert len(layer.cars) >= 0.3 * owned / 3  # about 0.3 per edge, not zero
    assert layer.total() == pytest.approx(before, abs=1e-6)
