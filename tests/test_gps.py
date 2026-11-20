import numpy as np
import pytest
import time

from graphs import grid, make_graph, straight

from contravia.driving_help import STRAIGHT
from contravia.gps import Router, RouteGraph, arc_cost, route


def rg_joined(rg) -> bool:
    """Whether any arc goes from province 0's edges to province 1's."""
    a, b = rg.offsets[1], rg.offsets[2]
    return bool(np.any((rg.arc_in < a) & (rg.arc_out >= a) & (rg.arc_out < b)))


def test_a_left_costs_more_than_a_right_and_straight_costs_nothing():
    c = arc_cost(np.full(3, 100.0), np.full(3, 10.0), np.zeros(3, int), np.zeros(3, int),
                 np.radians([0.0, 90.0, -90.0]))
    assert c.tolist() == pytest.approx([10.0, 13.0, 18.0])


def test_controls_and_service_roads_add_their_cost():
    from contravia.graph import CLASSES

    service = CLASSES.index("service")
    c = arc_cost(np.full(4, 100.0), np.full(4, 10.0), np.array([0, 0, 0, service]), np.array([1, 2, 3, 0]),
                 np.zeros(4))
    assert c.tolist() == pytest.approx([20.0, 14.0, 12.0, 15.0])


def test_two_provinces_sharing_a_node_join_and_ids_map_both_ways():
    west = make_graph([(0, 0), (100, 0)], [(0, 1), (1, 0)])
    east = make_graph([(100, 0), (200, 0)], [(0, 1), (1, 0)])
    east.a["g_junction_node"] = np.array([1, 7])  # its junction 0 is west's junction 1
    rg = RouteGraph.from_graphs([west, east], join_m=0.0)
    a, b = rg.island(0, 0), rg.island(1, 0)
    assert b in rg.arc_out[rg.arc_in == a]
    assert rg.local(b) == (1, 0)


def test_without_a_shared_node_the_15_m_rule_joins_them():
    west = make_graph([(0, 0), (100, 0)], [(0, 1), (1, 0)])
    east = make_graph([(108, 0), (200, 0)], [(0, 1), (1, 0)])
    east.a["g_junction_node"] = np.array([50, 51])
    assert rg_joined(RouteGraph.from_graphs([west, east]))
    assert not rg_joined(RouteGraph.from_graphs([west, east], join_m=0.0))


def test_a_u_turn_is_not_an_arc_and_a_saved_graph_loads_the_same(tmp_path):
    rg = RouteGraph.from_graphs([straight()])
    assert len(rg.arc_in) and not np.any(np.abs(rg.angle) > np.radians(150))
    rg.save(tmp_path / "route.npz")
    back = RouteGraph.load(tmp_path / "route.npz")
    assert np.array_equal(back.arc_out, rg.arc_out) and np.allclose(back.cost, rg.cost)
    assert np.array_equal(back.poly_start, rg.poly_start) and np.allclose(back.poly, rg.poly)


def edge(g, f, t) -> int:
    """The edge of graph g from junction f to junction t."""
    return int(np.flatnonzero((g.a["g_from"] == f) & (g.a["g_to"] == t))[0])


def turn_angles(rg, r) -> np.ndarray:
    return np.array([rg.angle[(rg.arc_in == a) & (rg.arc_out == b)][0] for a, b in zip(r.edges[:-1], r.edges[1:])])


def brute_force_seconds(rg, start, end) -> float:
    """Bellman-Ford over the arcs, the same costs as route()."""
    v = rg.vfree.astype(float)
    d = np.full(len(rg.length), np.inf)
    d[start[0]] = (rg.length[start[0]] - start[1]) / v[start[0]]
    for _ in range(len(d)):
        new = d.copy()
        np.minimum.at(new, rg.arc_out, d[rg.arc_in] + rg.cost)
        if np.array_equal(new, d):
            break
        d = new
    return float(d[end[0]] - (rg.length[end[0]] - end[1]) / v[end[0]])


def sample_pairs(rg, n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, b = rng.choice(len(rg.length), 2, replace=False)
        yield (int(a), float(rng.uniform(0, rg.length[a]))), (int(b), float(rng.uniform(0, rg.length[b])))


def wait_for(fn, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = fn()
        if got:
            return got
        time.sleep(0.01)
    raise AssertionError("timed out")


def test_the_route_respects_one_way_streets():
    # a ring 0 → 1 → 2 → 3 → 0, one way: from edge 0 back to junction 0 must go all the way round
    g = make_graph([(0, 0), (100, 0), (100, 100), (0, 100)], [(0, 1), (1, 2), (2, 3), (3, 0)])
    rg = RouteGraph.from_graphs([g])
    r = route(rg, (0, 10.0), [(3, 90.0)])
    assert r.edges.tolist() == [0, 1, 2, 3]
    assert r.metres == pytest.approx(90.0 + 100.0 + 100.0 + 90.0)


def test_a_destination_behind_on_the_same_one_way_street_goes_round_the_block():
    g = make_graph([(0, 0), (100, 0), (100, 100), (0, 100)], [(0, 1), (1, 2), (2, 3), (3, 0)])
    r = route(RouteGraph.from_graphs([g]), (0, 60.0), [(0, 20.0)])
    assert r.edges.tolist() == [0, 1, 2, 3, 0] and r.metres == pytest.approx(360.0)


def test_with_left_turns_dear_it_goes_three_rights_round():
    g = grid(3)
    rg = RouteGraph.from_graphs([g])
    left = route(rg, (edge(g, 3, 4), 50.0), [(edge(g, 4, 1), 50.0)])
    assert np.any(turn_angles(rg, left) < -STRAIGHT)  # cheapest is the left at 4
    rg.cost = np.where(rg.angle < -STRAIGHT, rg.cost + 1000.0, rg.cost)
    rg._csr = None
    r = route(rg, (edge(g, 3, 4), 50.0), [(edge(g, 4, 1), 50.0)])
    assert not np.any(turn_angles(rg, r) < -STRAIGHT)


def test_the_route_is_the_cheapest_by_brute_force():
    rg = RouteGraph.from_graphs([grid(3)])
    for start, end in sample_pairs(rg, 20, seed=1):
        r = route(rg, start, [end])
        assert r.seconds == pytest.approx(brute_force_seconds(rg, start, end))


def test_an_unreachable_destination_gives_none():
    g = make_graph([(0, 0), (100, 0), (500, 0), (600, 0)], [(0, 1), (1, 0), (2, 3), (3, 2)])
    rg = RouteGraph.from_graphs([g])
    assert route(rg, (0, 10.0), [(2, 50.0)]) is None


def test_ends_near_a_point_and_nothing_far_from_every_street():
    rg = RouteGraph.from_graphs([grid(3)])
    ends = rg.ends_at(50.0, 5.0)
    assert {rg.local(e)[1] for e, _ in ends} >= {0, 1}  # both directions of the street
    assert all(s == pytest.approx(50.0, abs=1.0) or s == pytest.approx(150.0, abs=1.0) for _, s in ends)
    assert rg.ends_at(5000.0, 5000.0) == []


def test_a_start_off_the_road_takes_the_nearest_edge_along_the_heading():
    g = grid(3)
    rg = RouteGraph.from_graphs([g])
    assert rg.start_at(50.0, 8.0, 0.0)[0] == edge(g, 0, 1)
    assert rg.start_at(50.0, 8.0, np.pi)[0] == edge(g, 1, 0)


def test_the_router_answers_on_its_thread():
    r = Router(lambda: RouteGraph.from_graphs([grid(3)]))
    r.ask((10.0, 0.0, 0.0), (200.0, 150.0))
    got = wait_for(r.take, 5.0)
    assert got != "none" and got.edges[-1] in {edge(grid(3), 5, 8), edge(grid(3), 8, 5)}
    assert r.busy is False


def test_the_router_says_none_when_no_street_is_near():
    r = Router(lambda: RouteGraph.from_graphs([grid(3)]))
    r.ask((10.0, 0.0, 0.0), (5000.0, 5000.0))
    assert wait_for(r.take, 5.0) == "none"
