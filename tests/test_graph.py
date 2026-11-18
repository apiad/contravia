import numpy as np
import pytest

from contravia.graph import CLASSES, Graph
from contravia.prepare import build_city, read_osm

FIX = "tests/fixtures/traffic.osm"


@pytest.fixture(scope="module")
def g():
    city, _ = build_city(read_osm(FIX))
    return city.graph


def edge(g, a, b):
    """The edge from OSM node a to OSM node b, or None."""
    nodes = g.a["g_junction_node"]
    hit = np.flatnonzero((nodes[g.a["g_from"]] == a) & (nodes[g.a["g_to"]] == b))
    return int(hit[0]) if len(hit) else None


def turn(g, e_in, e_out) -> bool:
    return bool(((g.a["g_turn_in"] == e_in) & (g.a["g_turn_out"] == e_out)).any())


def test_two_way_gives_two_edges_and_one_way_one(g):
    assert edge(g, 7, 8) is not None and edge(g, 8, 7) is not None
    assert edge(g, 4, 5) is not None and edge(g, 5, 4) is None


def test_no_u_turn_where_there_is_a_choice(g):
    assert not turn(g, edge(g, 7, 8), edge(g, 8, 7))
    assert turn(g, edge(g, 7, 8), edge(g, 8, 9))


def test_turn_restriction_removes_only_that_turn(g):
    assert not turn(g, edge(g, 5, 2), edge(g, 2, 1))
    assert turn(g, edge(g, 5, 2), edge(g, 2, 3))


def test_car_park_that_cannot_be_left_is_dropped(g):
    assert edge(g, 9, 20) is None
    assert not np.isin(g.a["g_junction_node"], [20, 21, 22]).any()


def test_controls_at_edge_ends(g):
    control = g.a["g_control"]
    for a in (4, 2, 8):
        assert control[edge(g, a, 5)] == 1  # signal at node 5
    assert control[edge(g, 5, 6)] == 0
    assert control[edge(g, 7, 8)] == 2  # stop at node 8


def test_built_area_only_near_the_building(g):
    built = g.a["g_built"]
    assert built[edge(g, 2, 5)] > 300
    assert built[edge(g, 7, 4)] == 0


def test_lanes_speeds_and_classes(g):
    a = g.a
    primary = edge(g, 1, 2)
    assert a["g_lanes"][primary] == 2 and a["g_vfree"][primary] == pytest.approx(50 / 3.6)
    assert CLASSES[a["g_class"][primary]] == "primary"
    assert a["g_lanes"][edge(g, 7, 8)] == 1 and a["g_vfree"][edge(g, 7, 8)] == pytest.approx(40 / 3.6)
    assert a["g_lanes"][edge(g, 4, 5)] == 1
    assert a["g_len"][primary] == pytest.approx(100, abs=3)


def test_municipio_of_every_edge(g):
    names = list(g.a["g_muni_names"])
    assert names == ["Habana Vieja"]
    assert (g.a["g_muni"] == 0).all()


def test_points_follow_the_edge_offset_to_the_right(g):
    e = edge(g, 7, 8)  # heading east along row 2
    j = g.a["g_junction"]
    start, end = j[g.a["g_from"][e]], j[g.a["g_to"][e]]
    x, z, heading = g.points(np.array([e, e]), np.array([0.0, g.a["g_len"][e]]), np.array([2.0, 0.0]))
    assert x[0] == pytest.approx(start[0], abs=1e-3) and z[0] == pytest.approx(start[1] + 2.0, abs=1e-3)
    assert x[1] == pytest.approx(end[0], abs=1e-3) and z[1] == pytest.approx(end[1], abs=1e-3)
    assert heading[0] == pytest.approx(0.0, abs=1e-3)


def test_edges_near(g):
    e = edge(g, 7, 8)
    mx, mz = g.a["g_mid"][e]
    assert e in g.edges_near(mx, mz, 10.0)


def test_old_data_has_an_empty_graph(city):
    from contravia.citydata import CityData

    old = CityData({k: v for k, v in city.a.items() if not k.startswith("g_")})
    assert len(old.graph) == 0
    assert len(Graph.from_arrays({})) == 0


def test_edges_are_sampled_every_10_m_with_road_heights(g):
    a = g.a
    seg = np.hypot(*np.diff(a["g_poly"], axis=0).T)
    inside = np.ones(len(seg), bool)
    inside[a["g_poly_start"][1:-1] - 1] = False  # the step from one edge to the next is not a segment
    assert seg[inside].max() <= 10.0 + 1e-3
    assert len(a["g_poly_y"]) == len(a["g_poly"])


def test_road_width_follows_the_lanes_tag():
    from contravia.graph import LANE, road_width
    from contravia.prepare import WIDTHS

    assert road_width("trunk", {"lanes": "6"}, WIDTHS) == 6 * LANE + 1  # the Malecón, three each way
    assert road_width("primary", {"lanes": "3", "oneway": "yes"}, WIDTHS) == 3 * LANE + 1  # one side of Boyeros
    assert road_width("residential", {"lanes": "2"}, WIDTHS) == WIDTHS["residential"] + 1.6
    assert road_width("trunk", {}, WIDTHS) == WIDTHS["trunk"]
    assert road_width("residential", {"oneway": "yes"}, WIDTHS) == LANE + 1
