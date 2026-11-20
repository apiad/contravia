import numpy as np
import pytest
from graphs import make_graph, straight

from contravia.gps import RouteGraph, arc_cost


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
