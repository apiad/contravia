import numpy as np
from graphs import four_way, straight

from havana.traffic import S0, Cars

KIND = 0  # car, 4.5 m


def run(cars, seconds, t0=0.0, obstacles=(), dt=0.1):
    for k in range(int(seconds / dt)):
        cars.step(dt, t0 + k * dt, list(obstacles))


def gaps_on_edge(cars, edge):
    on = np.flatnonzero(cars.edge == edge)
    order = on[np.argsort(cars.s[on])]
    return cars.s[order[1:]] - cars.length[order[1:]] - cars.s[order[:-1]]


def test_two_cars_in_a_lane_keep_their_distance():
    cars = Cars(straight(), seed=1)
    cars.add([0, 0], [10.0, 40.0], [11.0, 2.0], [KIND, KIND])
    smallest = np.inf
    for k in range(1200):
        cars.step(0.1, k * 0.1, [])
        same = cars.edge[0] == cars.edge[1]
        if same:
            smallest = min(smallest, abs(cars.s[1] - cars.s[0]) - 4.5)
    assert smallest >= S0 - 0.1


def test_a_car_stops_behind_a_stopped_car():
    cars = Cars(straight(), seed=1)
    cars.add([0, 0], [0.0, 150.0], [11.0, 0.0], [KIND, KIND])
    cars.stopped[1] = True
    run(cars, 60)
    assert cars.v[0] < 0.05 and cars.edge[0] == 0
    assert gaps_on_edge(cars, 0).min() >= S0 - 0.1


def test_a_car_stops_before_an_obstacle():
    cars = Cars(straight(), seed=1)
    cars.add([0], [0.0], [11.0], [KIND])
    run(cars, 60, obstacles=[(0, 0, 100.0, 0.0)])
    assert cars.v[0] < 0.05 and cars.s[0] <= 100.0 - 4.5 - S0 + 0.1


def test_four_way_junction_without_control_empties():
    g = four_way()
    cars = Cars(g, seed=3)
    approaches = [e for e in range(len(g)) if g.a["g_to"][e] == 0]
    for e in approaches:
        cars.add([e] * 3, [130.0, 120.0, 110.0], [0.0] * 3, [KIND] * 3)
    run(cars, 120)
    assert not np.isin(cars.edge, approaches).any()


def test_signal_lets_one_axis_go_at_a_time():
    g = four_way(control={0: 1})
    cars = Cars(g, seed=3)
    approaches = [e for e in range(len(g)) if g.a["g_to"][e] == 0]
    for e in approaches:
        cars.add([e], [125.0], [0.0], [KIND])
    run(cars, 20, t0=0.0)
    heading = np.array([g.a["g_junction"][g.a["g_to"][e]] - g.a["g_junction"][g.a["g_from"][e]] for e in approaches])
    east_west = np.abs(heading[:, 0]) > np.abs(heading[:, 1])
    crossed = ~np.isin(cars.edge, approaches)
    assert crossed[east_west].all() and not crossed[~east_west].any()


def test_a_car_takes_its_next_edge_and_draws_another():
    g = straight()
    cars = Cars(g, seed=1)
    cars.add([0], [190.0], [11.0], [KIND])
    first_next = cars.next_edge[0]
    run(cars, 3)
    assert cars.edge[0] == first_next
    turns_out = g.a["g_turn_out"][g.a["g_turn_in"] == first_next]
    assert cars.next_edge[0] in turns_out
