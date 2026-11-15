import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from havana.flow import Flow
from havana.prepare import build_city, read_osm
from havana.rules.traffic import TrafficConfig, moving

TUE_8 = datetime(2026, 11, 17, 8, 30)


@pytest.fixture(scope="module")
def graph():
    city, _ = build_city(read_osm("tests/fixtures/traffic.osm"))
    return city.graph


def cfg_for(total: float) -> TrafficConfig:
    """Settings whose vehicles moving at TUE_8 add up to `total`."""
    base = TrafficConfig()
    return base.updated(level=total / sum(moving(base, TUE_8).values()))


def run(flow, seconds, start=TUE_8):
    for k in range(seconds):
        flow.step(1.0, start + timedelta(seconds=k))


def test_without_demand_vehicles_are_conserved_and_bounded(graph):
    flow = Flow(graph, cfg_for(40))
    flow.demand = False
    flow.n[:] = flow.jam * np.linspace(0.1, 0.9, len(flow.n))
    total = flow.n.sum()
    for k in range(600):
        flow.step(1.0, TUE_8 + timedelta(seconds=k))
        assert (flow.n >= -1e-9).all() and (flow.n <= flow.jam + 1e-6).all()
    assert flow.n.sum() == pytest.approx(total, rel=1e-6)


LAG = timedelta(minutes=15)  # the flow follows demand about one trip time late


def test_warm_flow_holds_the_vehicles_moving(graph):
    flow = Flow(graph, cfg_for(40))
    run(flow, 3600, start=TUE_8 - timedelta(hours=1))
    assert flow.n.sum() == pytest.approx(sum(moving(flow.config, TUE_8 - LAG).values()), rel=0.05)


def approaches(graph, closed):
    a = graph.a
    feeders = a["g_turn_in"][a["g_turn_out"] == closed]
    return np.unique(feeders)


def test_closing_an_edge_blocks_and_fills_its_approaches(graph):
    flow = Flow(graph, cfg_for(40))
    flow.demand = False
    flow.n[:] = 0.5 * flow.jam
    closed = int(np.argmax(graph.a["g_len"]))
    app = approaches(graph, closed)
    before_closed, before_app = flow.n[closed], flow.n[app].sum()
    flow.close([closed], until=TUE_8 + timedelta(minutes=40))
    run(flow, 300)
    assert flow.n[closed] < before_closed  # it empties and nothing enters
    assert flow.n[app].sum() >= before_app  # blocked: they do not drain
    assert flow.view().queue[app].sum() > 0
    filled = flow.n[app].sum()
    run(flow, 300, start=TUE_8 + timedelta(minutes=41))  # reopened
    assert flow.n[app].sum() < filled


def test_junction_before_sends_less_into_a_blocked_approach(graph):
    closed = int(np.argmax(graph.a["g_len"]))
    app = approaches(graph, closed)
    flow = Flow(graph, cfg_for(40))
    into_app = np.isin(flow.t_out, app) & ~np.isin(flow.t_in, app)
    mask = np.zeros(len(flow.n), bool)
    open_share = flow.shares(mask)[into_app]
    mask[closed] = True
    closed_share = flow.shares(mask)[into_app]
    assert into_app.any() and (closed_share < open_share).all()


def test_new_settings_reach_the_streets_within_three_trip_times(graph):
    flow = Flow(graph, cfg_for(40))
    run(flow, 3600, start=TUE_8 - timedelta(hours=1))
    before = flow.n.sum() / sum(moving(flow.config, TUE_8 - LAG).values())
    flow.configure(flow.config.updated(level=2 * flow.config.level))
    end = TUE_8 + timedelta(minutes=45)
    run(flow, 45 * 60, start=TUE_8)
    after = flow.n.sum() / (sum(moving(flow.config, end - LAG).values()) / 2)
    assert after / before > 1.9


def test_view_is_a_copy_with_speeds(graph):
    flow = Flow(graph, cfg_for(40))
    run(flow, 60)
    v = flow.view()
    flow.n[:] = 0
    assert v.n.sum() > 0
    assert (v.speed >= 1.4 - 1e-6).all() and (v.speed <= graph.a["g_vfree"] + 1e-6).all()


@pytest.mark.slow
def test_one_step_on_a_city_sized_graph_is_fast():
    from havana.graph import Graph

    rng = np.random.default_rng(0)
    E, T = 80_000, 320_000
    a = {"g_from": rng.integers(0, 35_000, E).astype(np.int32), "g_to": rng.integers(0, 35_000, E).astype(np.int32),
         "g_len": rng.uniform(20, 200, E).astype(np.float32), "g_lanes": rng.integers(1, 3, E).astype(np.int8),
         "g_vfree": np.full(E, 11.1, np.float32), "g_class": np.full(E, 6, np.int8),
         "g_control": np.zeros(E, np.int8), "g_built": rng.uniform(0, 5000, E).astype(np.float32),
         "g_muni": np.zeros(E, np.int16), "g_muni_names": np.array(["Plaza de la Revolución"]),
         "g_turn_in": rng.integers(0, E, T).astype(np.int32), "g_turn_out": rng.integers(0, E, T).astype(np.int32),
         "g_turn_w": np.ones(T, np.float32)}
    flow = Flow(Graph(a), TrafficConfig())
    run(flow, 5)
    t0 = time.perf_counter()
    run(flow, 20, start=TUE_8 + timedelta(minutes=1))
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"flow step {ms:.1f} ms on {E} edges, {T} turns")
    assert ms < 40


def test_short_edges_do_not_create_vehicles():
    from graphs import make_graph

    # A ring of 5 m edges: shorter than one second of driving at free speed.
    ring = [(0, 0), (5, 0), (5, 5), (0, 5)]
    g = make_graph(ring, [(0, 1), (1, 2), (2, 3), (3, 0)])
    flow = Flow(g, TrafficConfig())
    flow.demand = False
    flow.n[:] = [0.3, 0.0, 0.0, 0.0]
    for k in range(100):
        flow.step(1.0, TUE_8 + timedelta(seconds=k))
    assert flow.n.sum() == pytest.approx(0.3, rel=1e-6)
