import math

import numpy as np
import pytest

from contravia.address import _and, describe, locate
from contravia.graph import Graph, build_graph
from contravia.prepare import WIDTHS


def ident(lat, lon):
    return np.stack([np.asarray(lat, float), np.asarray(lon, float)], axis=1)


def way(i, hw, refs, pts, name=None):
    return (i, hw, refs, pts, {"name": name} if name else {})


@pytest.fixture(scope="module")
def graph():
    # Calle 23 runs east along z = 0 and is crossed by L (x = 100), M (x = 200) and N (x = 400); at x = 300 only
    # an unnamed service road meets it. Points are (x, z) metres: the test's to_local is the identity.
    ways = [
        way(1, "residential", [1, 2, 3, 4, 5], [(0, 0), (100, 0), (200, 0), (300, 0), (400, 0)], "Calle 23"),
        way(2, "residential", [10, 2, 11], [(100, -100), (100, 0), (100, 100)], "L"),
        way(3, "residential", [20, 3, 21], [(200, -100), (200, 0), (200, 100)], "M"),
        way(4, "service", [30, 4], [(300, -50), (300, 0)]),
        way(5, "residential", [40, 5, 41], [(400, -100), (400, 0), (400, 100)], "N"),
    ]
    places = ([("Vedado", [(-10, -150), (250, -150), (250, 150), (-10, 150)])], [("Carmelo", (350, 0), "neighbourhood")],
              [("Plaza", [(-500, -500), (900, -500), (900, 500), (-500, 500)])])
    a = build_graph(ways, {}, [], [], [], ident, WIDTHS, places=places)
    return Graph.from_arrays(a)


def at(graph, x, z, yaw):
    return describe(graph.a, *locate(graph, x, z, yaw))


def test_between_the_streets_crossing_the_block(graph):
    assert at(graph, 150, 1, 0.0) == ("Calle 23", "entre L y M", "Vedado")


def test_the_other_way_says_them_the_other_way(graph):
    assert at(graph, 150, 1, math.pi)[1] == "entre M y L"


def test_an_unnamed_road_is_passed_by_to_the_next_named_street(graph):
    assert at(graph, 250, 1, 0.0)[1] == "entre M y N"


def test_at_the_corner(graph):
    assert at(graph, 195, 1, 0.0)[1] == "esquina a M"
    assert at(graph, 205, 1, 0.0)[1] == "esquina a M"


def test_one_crossing_known(graph):
    assert at(graph, 50, 1, 0.0)[1] == "hasta L"
    assert at(graph, 100, 50, math.pi / 2)[:2] == ("L", "desde 23")


def test_driving_across_an_edge_does_not_take_it(graph):
    # at the crossing of 23 and L, heading north: L, not 23
    assert at(graph, 101, 3, math.pi / 2)[0] == "L"


def test_an_unnamed_road_has_no_street_line(graph):
    assert at(graph, 300, -30, math.pi / 2)[:2] == ("", "")


def test_barrio_by_area_then_nearest_point(graph):
    assert at(graph, 150, 1, 0.0)[2] == "Vedado"
    assert at(graph, 350, 1, 0.0)[2] == "Carmelo"


def test_y_turns_to_e_before_the_sound_i():
    assert _and("23", "Infanta") == "23 e Infanta"
    assert _and("23", "Hidalgo") == "23 e Hidalgo"
    assert _and("23", "Hierro") == "23 y Hierro"
    assert _and("L", "M") == "L y M"


def test_a_name_after_entre_drops_its_parenthesis():
    from contravia.address import short

    assert short("Avenida de los Presidentes (Calle G)") == "Avenida de los Presidentes"
    assert short("Calle 23") == "23"


def test_the_street_name_drops_its_parenthesis():
    a = {"g_names": np.array(["Paseo de Martí (Paseo del Prado)"]), "g_name": np.array([0]),
         "g_between": np.array([[-1, -1]]), "g_len": np.array([100.0])}
    assert describe(a, 0, 50.0, True)[0] == "Paseo de Martí"
