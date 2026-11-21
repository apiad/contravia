"""A road that crosses a province border keeps its pieces up to the border, in exactly one province each."""

from datetime import datetime, timedelta

import numpy as np
from graphs import make_graph

from contravia.flow import Flow
from contravia.graph import build_graph
from contravia.prepare import WIDTHS
from contravia.rules.traffic import TrafficConfig
from contravia.traffic import Cars


def ident(lat, lon):
    return np.stack([np.asarray(lat, float), np.asarray(lon, float)], axis=1)


def way(way_id, refs, pts, oneway=False, highway="trunk"):
    return (way_id, highway, refs, pts, {"oneway": "yes"} if oneway else {})


# A dual carriageway runs east-west across the border at x = 0: eastbound along z = 0, westbound along z = 20,
# each a one-way street. The carriageways are joined at x = ±500 and x = ±1000, so each province has its own
# loop, and the pieces between the crossovers at ±500 lead only into the other province. The piece over the
# border (the bridge) has its midpoint at x = 50, in the east. A one-way spur leaves the east province 300 m
# into the west and ends there. Points are (x, z) metres: the test's to_local is the identity.
WAYS = [
    way(1, [1, 2, 3], [(-1000, 0), (-500, 0), (-100, 0)], oneway=True),
    way(2, [3, 4], [(-100, 0), (200, 0)], oneway=True),
    way(3, [4, 5, 6], [(200, 0), (500, 0), (1000, 0)], oneway=True),
    way(4, [11, 12, 13], [(1000, 20), (500, 20), (200, 20)], oneway=True),
    way(5, [13, 14], [(200, 20), (-100, 20)], oneway=True),
    way(6, [14, 15, 16], [(-100, 20), (-500, 20), (-1000, 20)], oneway=True),
    way(7, [2, 15], [(-500, 0), (-500, 20)]),
    way(8, [5, 12], [(500, 0), (500, 20)]),
    way(9, [1, 16], [(-1000, 0), (-1000, 20)]),
    way(10, [6, 11], [(1000, 0), (1000, 20)]),
    way(11, [5, 30, 31], [(500, 0), (100, -300), (-300, -300)], oneway=True),
]
MARGIN = 2000.0


def province(side, margin=MARGIN):
    """The graph of the west (-1) or east (+1) half plane, with the roads within margin of it read too."""
    return build_graph(WAYS, {}, [], [], [], ident, WIDTHS, inside=lambda xz: side * xz[:, 0] > 0,
                       near=(lambda xz: side * xz[:, 0] > -margin) if margin else None)


def pieces(a) -> set:
    """The (from OSM node, to OSM node) of every edge."""
    nodes = a["g_junction_node"]
    return set(zip(nodes[a["g_from"]].tolist(), nodes[a["g_to"]].tolist(), strict=True))


def test_one_way_pieces_up_to_the_border_are_kept():
    west, east = pieces(province(-1)), pieces(province(+1))
    assert {(2, 3), (14, 15)} <= west
    assert {(3, 4), (4, 5), (12, 13), (13, 14)} <= east


def test_without_the_margin_they_lead_nowhere_and_are_dropped():
    assert not {(2, 3), (14, 15)} & pieces(province(-1, margin=0))
    assert not {(3, 4), (13, 14)} & pieces(province(+1, margin=0))


def test_the_provinces_hold_every_edge_of_the_road_exactly_once():
    whole = pieces(build_graph(WAYS, {}, [], [], [], ident, WIDTHS))
    west, east = pieces(province(-1)), pieces(province(+1))
    assert not west & east
    assert west | east == whole


def test_the_provinces_share_the_osm_node_where_the_road_is_cut():
    west, east = province(-1), province(+1)
    shared = set(west["g_junction_node"].tolist()) & set(east["g_junction_node"].tolist())
    assert shared == {3, 14}


def test_a_road_that_leads_nowhere_past_the_border_is_still_dropped():
    assert not {(5, 30), (30, 31)} & (pieces(province(-1)) | pieces(province(+1)))


def test_the_piece_into_the_border_has_no_turn_and_the_piece_from_it_none_into_it():
    a = province(-1)
    nodes = a["g_junction_node"]
    out = int(np.flatnonzero((nodes[a["g_from"]] == 2) & (nodes[a["g_to"]] == 3))[0])
    back = int(np.flatnonzero((nodes[a["g_from"]] == 14) & (nodes[a["g_to"]] == 15))[0])
    assert out not in a["g_turn_in"] and back not in a["g_turn_out"]


def border_road():
    """Junctions 0 → 1 → 2 one way: the last edge ends at the province's border, with no turn out of it."""
    return make_graph([(0, 0), (200, 0), (400, 0)], [(0, 1), (1, 2)])


def test_a_car_drives_off_the_end_of_a_road_that_leaves_the_province():
    cars = Cars(border_road(), seed=1)
    cars.add([0, 1], [50.0, 100.0], [11.0, 11.0], [0, 0])
    for k in range(1200):
        cars.step(0.1, k * 0.1, [])
    assert len(cars) == 0


def test_the_flow_does_not_pile_up_where_a_road_leaves_the_province():
    flow = Flow(border_road(), TrafficConfig())
    flow.demand = False
    flow.n[:] = flow.jam * 0.5
    for k in range(600):
        flow.step(1.0, datetime(2026, 11, 17, 8, 30) + timedelta(seconds=k))
    assert flow.n.sum() < 0.01
