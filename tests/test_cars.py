import numpy as np
import pytest
from graphs import four_way, make_graph, straight

from contravia.traffic import LANE_SPEED, S0, Cars

LOOK = 0  # the first look is a car, 4.5 m


def run(cars, seconds, t0=0.0, obstacles=(), dt=0.1):
    for k in range(int(seconds / dt)):
        cars.step(dt, t0 + k * dt, list(obstacles))


def gaps_on_edge(cars, edge):
    on = np.flatnonzero(cars.edge == edge)
    order = on[np.argsort(cars.s[on])]
    return cars.s[order[1:]] - cars.length[order[1:]] - cars.s[order[:-1]]


def test_two_cars_in_a_lane_keep_their_distance():
    cars = Cars(straight(), seed=1)
    cars.add([0, 0], [10.0, 40.0], [11.0, 2.0], [LOOK, LOOK])
    smallest = np.inf
    for k in range(1200):
        cars.step(0.1, k * 0.1, [])
        same = cars.edge[0] == cars.edge[1]
        if same:
            smallest = min(smallest, abs(cars.s[1] - cars.s[0]) - 4.5)
    assert smallest >= S0 - 0.1


def test_a_car_stops_behind_a_stopped_car():
    cars = Cars(straight(), seed=1)
    cars.add([0, 0], [0.0, 150.0], [11.0, 0.0], [LOOK, LOOK])
    cars.stopped[1] = True
    run(cars, 60)
    assert cars.v[0] < 0.05 and cars.edge[0] == 0
    assert gaps_on_edge(cars, 0).min() >= S0 - 0.1


def test_a_car_stops_before_an_obstacle():
    cars = Cars(straight(), seed=1)
    cars.add([0], [0.0], [11.0], [LOOK])
    run(cars, 60, obstacles=[(0, 0, 100.0, 0.0)])
    assert cars.v[0] < 0.05 and cars.s[0] <= 100.0 - 4.5 - S0 + 0.1


def test_four_way_junction_without_control_empties():
    g = four_way()
    cars = Cars(g, seed=3)
    approaches = [e for e in range(len(g)) if g.a["g_to"][e] == 0]
    for e in approaches:
        cars.add([e] * 3, [130.0, 120.0, 110.0], [0.0] * 3, [LOOK] * 3)
    run(cars, 120)
    assert not np.isin(cars.edge, approaches).any()


def test_signal_lets_one_axis_go_at_a_time():
    g = four_way(control={0: 1})
    cars = Cars(g, seed=3)
    approaches = [e for e in range(len(g)) if g.a["g_to"][e] == 0]
    for e in approaches:
        cars.add([e], [125.0], [0.0], [LOOK])
    run(cars, 20, t0=0.0)
    heading = np.array([g.a["g_junction"][g.a["g_to"][e]] - g.a["g_junction"][g.a["g_from"][e]] for e in approaches])
    east_west = np.abs(heading[:, 0]) > np.abs(heading[:, 1])
    crossed = ~np.isin(cars.edge, approaches)
    assert crossed[east_west].all() and not crossed[~east_west].any()


def test_a_car_takes_its_next_edge_and_draws_another():
    g = straight()
    cars = Cars(g, seed=1)
    cars.add([0], [190.0], [11.0], [LOOK])
    first_next = cars.next_edge[0]
    run(cars, 3)
    assert cars.edge[0] == first_next
    turns_out = g.a["g_turn_out"][g.a["g_turn_in"] == first_next]
    assert cars.next_edge[0] in turns_out


def test_a_junction_keeps_up_with_steady_arrivals():
    g = four_way(arm=150.0)
    cars = Cars(g, seed=5)
    approaches = [e for e in range(len(g)) if g.a["g_to"][e] == 0]
    added = 0
    dt = 0.1
    for k in range(int(180 / dt)):
        t = k * dt
        if k % int(8 / dt) == 0:  # one car per approach every 8 s: 1800 an hour into the junction
            for e in approaches:
                on = cars.edge == e
                if not on.any() or cars.s[on].min() > 10:
                    cars.add([e], [0.0], [8.0], [LOOK])
                    added += 1
        cars.step(dt, t, [])
        cars.remove(np.isin(cars.edge, approaches, invert=True) & (cars.s > 100))  # gone down an exit
    waiting = int(np.isin(cars.edge, approaches).sum())
    assert waiting <= 12, f"{waiting} of {added} still on the approaches"


def test_cars_take_height_and_pitch_from_the_edge_profile():
    g = straight(length=400.0)
    a = g.a
    a["g_poly_y"] = (a["g_poly"][:, 0] * 0.1).astype(np.float32)  # rises 1 m every 10 m going east
    cars = Cars(g, seed=1)
    cars.add([0], [100.0], [0.0], [LOOK])
    xyz, _heading, pitch = cars.world()
    assert xyz[0, 1] == pytest.approx((100.0 - 4.5 / 2) * 0.1, abs=1e-3)
    assert pitch[0] == pytest.approx(np.arctan(0.1), abs=1e-3)


def test_kept_heights_match_the_graph_profile(city):
    """world() reads heights from the table _index keeps every 2 m; on a bumpy profile over the fixture city's
    graph they stay within a few cm of the graph's own lookup, and so does the pitch."""
    from contravia.graph import Graph

    a = dict(city.graph.a)
    x, z = a["g_poly"][:, 0].astype(np.float64), a["g_poly"][:, 1].astype(np.float64)
    a["g_poly_y"] = (3 * np.sin(x / 40) + 2 * np.cos(z / 25)).astype(np.float32)  # hills up to 10 % steep
    g = Graph(a)
    cars = Cars(g, seed=1)
    rng = np.random.default_rng(0)
    e = rng.integers(0, len(g), 2000)
    cars.add(e, rng.uniform(-0.2, 1.2, len(e)) * cars.len[e], np.zeros(len(e)), np.full(len(e), LOOK))
    xyz, _heading, pitch = cars.world()
    L = cars.length
    old_y = g.heights(cars.edge, cars.s - L / 2)
    old_pitch = np.arctan2(g.heights(cars.edge, cars.s) - g.heights(cars.edge, cars.s - L), L)
    assert np.abs(xyz[:, 1] - old_y).max() < 0.03
    assert np.abs(pitch - old_pitch).max() < np.radians(0.5)


def two_lanes(length=600.0):
    """A one-way road east with two lanes, 0 → 1 → 2, turned around at the ends."""
    return make_graph([(0, 0), (length / 2, 0), (length, 0)], [(0, 1), (1, 2), (2, 1), (1, 0)], lanes=2,
                      uturns=True)


def test_a_turning_car_moves_smoothly_through_the_junction():
    g = four_way()
    cars = Cars(g, seed=2)
    east_in = next(e for e in range(len(g)) if g.a["g_from"][e] == 1 and g.a["g_to"][e] == 0)
    cars.add([east_in], [100.0], [8.0], [LOOK])
    xyz, heading, _ = cars.world()
    biggest_step, biggest_turn, edges = 0.0, 0.0, {east_in}
    for k in range(150):
        cars.step(0.05, k * 0.05, [])
        now, h, _ = cars.world()
        biggest_step = max(biggest_step, float(np.hypot(*(now - xyz)[0, [0, 2]])) / max(cars.v[0] * 0.05, 0.01))
        biggest_turn = max(biggest_turn, abs(float(np.angle(np.exp(1j * (h[0] - heading[0]))))))
        xyz, heading = now, h
        edges.add(int(cars.edge[0]))
    assert len(edges) > 1  # it went through
    assert biggest_step < 1.3  # never a jump: each step at most a little over what its speed covers
    assert biggest_turn < 0.12  # radians per 0.05 s: no snap of the heading at the junction


def test_the_drawn_offset_eases_toward_a_new_lane():
    cars = Cars(two_lanes(), seed=1)
    cars.add([0], [50.0], [5.0], [LOOK])
    cars.step(0.1, 0.0, [])
    before = cars.off[0]
    cars.lane[0] = 1 - cars.lane[0]
    cars.step(0.1, 0.1, [])
    assert abs(cars.off[0] - before) <= LANE_SPEED * 0.1 + 1e-9
    run(cars, 5.0, t0=0.2)
    assert cars.off[0] == pytest.approx(cars.offsets()[0])


def test_a_fast_car_overtakes_a_slow_one_by_changing_lane():
    cars = Cars(two_lanes(), seed=1)
    cars.add([0, 0], [20.0, 60.0], [11.0, 3.0], [LOOK, LOOK])
    cars.lane[:] = 0
    cars.want[:] = -1
    cars.vfree = cars.vfree.copy()
    slow = cars.v[1]
    passed = False
    for k in range(300):
        cars.v[1] = slow  # the car ahead keeps crawling
        cars.step(0.1, k * 0.1, [])
        if cars.edge[0] == cars.edge[1] and cars.s[0] > cars.s[1] + 5:
            passed = True
            break
    assert passed and cars.lane[0] == 1


def test_a_car_reaches_its_turn_lane_before_the_junction():
    cars = Cars(two_lanes(), seed=1)
    cars.add([0], [10.0], [8.0], [LOOK])
    cars.lane[0], cars.want[0] = 1, 0
    run(cars, 12.0)
    assert cars.edge[0] == 0 and cars.lane[0] == 0
