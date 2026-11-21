import numpy as np
import pytest
import time

from graphs import grid, make_graph, straight

from contravia.driving_help import STRAIGHT
from contravia.gps import Guide, Router, RouteGraph, Step, arc_cost, distance_text, route, say, steps


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


@pytest.mark.parametrize("m, text", [(12, "Ahora"), (64, "60 m"), (180, "200 m"), (940, "950 m"),
                                     (1240, "1,2 km"), (24400, "24,4 km")])
def test_distances_round_as_the_spec_says(m, text):
    assert distance_text(m) == text


@pytest.mark.parametrize("deg, words", [(90, "Dobla a la derecha en Galiano"), (-90, "Dobla a la izquierda en Galiano"),
                                        (45, "Dobla ligeramente a la derecha en Galiano")])
def test_the_turn_s_angle_picks_the_verb(deg, words):
    step = Step(200.0, "turn" if abs(deg) > 60 else "slight", np.radians(deg), "Galiano")
    assert say(step, 200.0, None) == ("200 m", words)


def test_an_unnamed_street_gets_no_name():
    _, what = say(Step(200.0, "turn", np.radians(90), None), 200.0, None)
    assert what == "Dobla a la derecha"


def test_a_long_stretch_says_follow_along_the_street():
    assert say(Step(2600.0, "turn", np.radians(90), "Paseo"), 2400.0, "Malecón") == ("2,4 km", "Sigue 2,4 km por Malecón")
    assert say(Step(2600.0, "turn", np.radians(90), "Paseo"), 2400.0, None) == ("2,4 km", "Dobla a la derecha en Paseo")


def test_straight_through_with_the_same_name_is_not_a_step():
    g = make_graph([(0, 0), (100, 0), (200, 0), (100, 100)], [(0, 1), (1, 2), (1, 3)],
                   names=["Neptuno", "Neptuno", "Galiano"])
    rg = RouteGraph.from_graphs([g])
    assert [s.kind for s in steps(rg, route(rg, (0, 0.0), [(1, 90.0)]))] == ["arrive"]


def test_a_turn_is_a_step_at_its_junction_with_the_new_street_s_name():
    g = make_graph([(0, 0), (100, 0), (200, 0), (100, 100)], [(0, 1), (1, 2), (1, 3)],
                   names=["Neptuno", "Neptuno", "Galiano"])
    rg = RouteGraph.from_graphs([g])
    first, last = steps(rg, route(rg, (0, 20.0), [(2, 60.0)]))
    assert (first.kind, first.name, first.at) == ("turn", "Galiano", pytest.approx(80.0))
    assert last.kind == "arrive" and last.at == pytest.approx(140.0)


def test_rural_roads_with_no_names_still_merge_and_never_say_en_nothing():
    rg = RouteGraph.from_graphs([grid(3)])
    for start, end in sample_pairs(rg, 20, seed=2):
        r = route(rg, start, [end])
        for st in steps(rg, r):
            words = say(st, 300.0, None)[1]
            assert not words.endswith(" en") and " en " not in words and "None" not in words


def roundabout_fixture():
    """A one-way ring (one way id) round the origin, counterclockwise seen from above with north up, and four
    two-way arms: ring junctions 0 east, 1 north, 2 west, 3 south; arm ends 4 to 7 in the same order."""
    ring = [(20, 0), (0, -20), (-20, 0), (0, 20)]
    arms = [(100, 0), (0, -100), (-100, 0), (0, 100)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for k in range(4):
        edges += [(k + 4, k), (k, k + 4)]
    g = make_graph(ring + arms, edges)
    g.a["g_way"][:4] = 999999
    return g


def test_a_roundabout_says_which_exit():
    g = roundabout_fixture()
    rg = RouteGraph.from_graphs([g])
    r = route(rg, (edge(g, 4, 0), 30.0), [(edge(g, 2, 6), 50.0)])
    first = steps(rg, r)[0]
    assert first.kind == "roundabout" and first.exit == 2
    assert say(first, 80.0, None)[1].startswith("En la rotonda, toma la segunda salida")


def car_at(g, e, s):
    from types import SimpleNamespace

    x, z, h = g.points([e], [s], [0.0])
    return SimpleNamespace(x=float(x[0]), z=float(z[0]), yaw=float(h[0]), speed=10.0)


def points_of(g):
    return lambda e, s: tuple(float(v[0]) for v in g.points([e], [s], [0.0])[:2])


def guide_on(g, to, start=(0, 10.0)):
    """A guide on graph g toward `to`, its first route found from a car `start` (edge, s) along."""
    gd = Guide(Router(lambda: RouteGraph.from_graphs([g])))
    gd.go(*to)
    wait_for(lambda: (gd.update(start[0], start[1], car_at(g, *start), 1 / 60), gd.route)[1] is not None, 5.0)
    return gd


def settle(gd):
    wait_for(lambda: not gd.router.busy, 5.0)


def along_route(g, gd, step=2.0):
    for k, e in enumerate(gd.route.edges.tolist()):
        lo = gd.route.s0 if k == 0 else 0.0
        hi = gd.route.s1 if k == len(gd.route.edges) - 1 else float(g.a["g_len"][e])
        for s in np.arange(lo, hi + 1e-9, step):
            yield e, float(s)


def test_the_guide_advances_step_by_step_and_arrives():
    g = grid(3)
    gd = guide_on(g, (200.0, 150.0))
    seen = set()
    for e, s in list(along_route(g, gd)):
        gd.update(e, s, car_at(g, e, s), 1 / 60)
        if gd.step is not None:
            seen.add((gd.step.kind, round(gd.step.at)))
    assert gd.route is None and gd.message.startswith("Llegaste") and gd.reroutes == 0
    assert len(seen) >= 2  # it went through the turn and then the arrival


def test_off_the_route_for_a_second_a_new_route_starts_from_there():
    g = grid(3)
    gd = guide_on(g, (200.0, 150.0))
    wrong = next(e for e in range(len(g)) if e not in gd.route.edges)
    for _ in range(70):
        gd.update(wrong, 10.0, car_at(g, wrong, 10.0), 1 / 60)
    settle(gd)
    gd.update(wrong, 10.0, car_at(g, wrong, 10.0), 1 / 60)
    assert gd.route.edges[0] == wrong and gd.reroutes == 1
    assert gd.message == "Ruta nueva"


def test_a_teleport_routes_again_once():
    g = grid(3)
    gd = guide_on(g, (200.0, 150.0))
    far = next(e for e in range(len(g)) if e not in gd.route.edges)
    for _ in range(300):
        gd.update(far, 10.0, car_at(g, far, 10.0), 1 / 60)
        settle(gd)
    assert gd.reroutes == 1


def test_no_route_says_so_and_drops_the_destination():
    g = grid(3)
    gd = Guide(Router(lambda: RouteGraph.from_graphs([g])))
    gd.go(5000.0, 5000.0)
    wait_for(lambda: (gd.update(0, 10.0, car_at(g, 0, 10.0), 1 / 60), gd.to)[1] is None, 5.0)
    assert gd.message == "No hay ruta" and gd.route is None


def test_the_look_ahead_turns_the_corner_before_the_car():
    g = grid(3)
    gd = guide_on(g, (200.0, 100.0), start=(edge(g, 1, 2), 10.0))  # east along z = 0, then right (south) at x = 200
    x, z = gd.ahead(22.0, points_of(g), from_edge_s=(edge(g, 1, 2), 82.0))  # 18 m before the corner
    assert x == pytest.approx(200.0, abs=0.5) and z == pytest.approx(4.0, abs=0.5)
    assert gd.next_edge(edge(g, 1, 2)) == edge(g, 2, 5)


def test_the_arrow_eases_toward_its_heading_the_short_way_round():
    from contravia.gps import arrow_yaw

    assert arrow_yaw(None, 1.0, 1 / 60) == 1.0
    assert arrow_yaw(0.0, 1.0, 0.15) == pytest.approx(1 - np.exp(-1), abs=1e-6)
    assert arrow_yaw(3.0, -3.0, 0.05) > 3.0  # across ±π it turns on, not back through 0


def test_the_arrow_is_faint_far_from_the_next_step():
    from contravia.gps import arrow_opacity

    assert arrow_opacity(400.0) == 0.55 and arrow_opacity(120.0) == 1.0


def test_the_arrow_mesh_is_1_6_m_long_along_x():
    from contravia.gps import arrow_mesh

    m = arrow_mesh()
    x = m.positions[:, 0]
    assert x.max() - x.min() == pytest.approx(1.6, abs=0.1) and abs(x.max() + x.min()) < 0.05
    assert len(m.indices) % 3 == 0 and len(m.positions) == len(m.colors)


def test_a_point_along_an_island_edge_follows_its_polyline():
    g = make_graph([(0, 0), (100, 0), (100, 100)], [(0, 1), (1, 2)])
    rg = RouteGraph.from_graphs([g])
    assert rg.point(0, 30.0) == pytest.approx((30.0, 0.0))
    assert rg.point(1, 100.0) == pytest.approx((100.0, 100.0))


def test_the_arrow_s_faces_wind_as_the_shadow_s_that_shows_from_above():
    from contravia.gps import arrow_mesh

    m = arrow_mesh()
    p, i = m.positions.astype(float), m.indices.reshape(-1, 3)
    geo = np.cross(p[i[:, 1]] - p[i[:, 0]], p[i[:, 2]] - p[i[:, 0]])
    said = m.normals[i[:, 0]]
    # wound as the car's shadow, which shows from above: its declared normal agrees with its right-hand winding
    assert np.all((geo * said).sum(1) > 0)


def _rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))


@pytest.mark.parametrize("yaw", [0.0, 1.2, -2.5])
def test_the_arrow_points_along_its_yaw_and_leans_its_top_toward_the_camera(yaw):
    from contravia.gps import ARROW_TILT, arrow_rotation

    to_camera = np.array([-np.cos(0.4), -np.sin(0.4)])  # the camera behind the car, seen from above
    q = arrow_rotation(yaw, to_camera)
    nose, top = _rotate(q, np.array([1.0, 0, 0])), _rotate(q, np.array([0, 1.0, 0]))
    ground = nose[[0, 2]] / np.hypot(nose[0], nose[2])
    assert ground @ np.array([np.cos(yaw), np.sin(yaw)]) > 0.75  # the tilt foreshortens it, never turns it away
    assert top[1] == pytest.approx(np.cos(ARROW_TILT), abs=1e-6)
    assert top[[0, 2]] @ to_camera == pytest.approx(np.sin(ARROW_TILT), abs=1e-6)


def test_the_band_is_most_of_a_lane_in_town_and_widens_for_the_fast_minimap():
    from contravia.gps import band_width
    from contravia.hud import MINIMAP_SIZE, MINIMAP_SPAN, minimap_target

    assert band_width(3.3, 6.6, 0.0) == pytest.approx(2.64)
    per_px = 2 * minimap_target(30.0) * MINIMAP_SPAN / MINIMAP_SIZE
    assert band_width(3.3, 20.0, 30.0) == pytest.approx(2.5 * per_px)
    assert band_width(3.3, 6.6, 40.0) == pytest.approx(6.6)


def test_before_a_left_turn_the_band_moves_to_the_leftmost_lane_and_stays_right_before_a_right():
    from contravia.gps import band_targets

    along = np.arange(0.0, 300.0, 2.0)
    right, left = np.full(len(along), 4.95), np.full(len(along), 1.65)
    t = band_targets(along, [(200.0, -1)], right, left)
    assert np.allclose(t[along <= 120.0], 4.95) and np.allclose(t[(along >= 150.0) & (along <= 200.0)], 1.65)
    assert np.allclose(t[along > 200.0], 4.95)
    assert np.allclose(band_targets(along, [(200.0, 1)], right, left), 4.95)


def test_the_band_lies_on_its_lane_5_cm_over_the_road():
    from contravia.gps import band_mesh
    from contravia.traffic import lane_geometry

    g = grid(3, lanes=2)
    g.a["g_poly_y"] = np.full(len(g.a["g_poly"]), 2.0, np.float32)
    twoway, lane_w = lane_geometry(g.a)
    e = edge(g, 0, 1)
    m = band_mesh(g, [e], 10.0, 90.0, [], 0.8 * lane_w[e])
    p = m.positions.astype(float)
    half = 2 * lane_w[e]  # two lanes right of the centreline
    assert p[:, 2].min() >= -0.1 and p[:, 2].max() <= half + 0.1  # heading east, right is +z
    assert np.allclose(p[:, 1], 2.05, atol=1e-4)
    assert 10.0 - 0.1 <= p[:, 0].min() and p[:, 0].max() <= 90.0 + 0.1


def test_the_band_s_faces_wind_as_the_shadow_s():
    from contravia.gps import band_mesh

    g = grid(3)
    m = band_mesh(g, [edge(g, 0, 1), edge(g, 1, 4)], 10.0, 50.0, [(90.0, 1)], 2.0)
    p, i = m.positions.astype(float), m.indices.reshape(-1, 3)
    geo = np.cross(p[i[:, 1]] - p[i[:, 0]], p[i[:, 2]] - p[i[:, 0]])
    assert np.all((geo * m.normals[i[:, 0]]).sum(1) > 0)
