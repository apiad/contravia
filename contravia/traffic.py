"""The car layer: cars near the player, driving on their lanes.

Each car is a row in a few arrays, never an object: the edge it is on, how far along it its front is (s), its
lane (0 is the rightmost), speed, look (and so kind), the edge it takes next, and how long it has waited at a junction.

Every frame each car finds what is ahead of it in its lane (another car, a stop line, the player) and
accelerates by the intelligent driver model: toward the street's free speed, keeping a gap that grows with
speed, braking smoothly. At the end of an edge a car may enter the junction if its signal is green, or, without
a signal, if no car with priority is about to enter: the bigger street goes first, a stop or give-way sign
loses, and between equal streets the car from the right goes first. A car that has waited 8 s goes anyway,
so a junction never locks up.
"""

import numpy as np

from contravia.graph import CLASSES, Graph
from contravia.rules.traffic import KINDS, look_kinds

LENGTH = {"car": 4.5, "taxi": 5.0, "bus": 12.0, "truck": 8.0, "moto": 2.0, "tricycle": 2.8, "bicitaxi": 2.6,
          "bike": 1.8, "police": 4.5, "fire": 8.6}
WIDTH = {"car": 1.8, "taxi": 1.95, "bus": 2.5, "truck": 2.5, "moto": 0.8, "tricycle": 1.2, "bicitaxi": 1.1,
         "bike": 0.6, "police": 1.8, "fire": 2.5}
ACCEL, BRAKE, S0, HEADWAY = 1.5, 2.0, 2.0, 1.2  # intelligent driver model: m/s², m/s², m, s
LANE_WIDTH = 3.0  # metres, at most; narrower when the street is narrow
WIDTHS = {"motorway": 14.0, "trunk": 10.0, "primary": 10.0, "secondary": 8.0, "tertiary": 6.0,
          "unclassified": 6.0, "residential": 6.0, "living_street": 6.0, "service": 4.0}
RANK = {c: 9 - k for k, c in enumerate(CLASSES)}  # bigger streets rank higher
GREEN, AMBER = 30.0, 3.0  # seconds per signal group; the cycle is 2 · (GREEN + AMBER)
CYCLE = 2 * (GREEN + AMBER)
PATIENCE = 8.0  # seconds a car waits at a junction before it goes anyway
LOOK = 30.0  # metres before the end of an edge where a car starts asking to enter the junction
NEAR = 25.0  # metres: a car this close to the junction on another approach counts as about to enter
REFRESH = 10.0  # metres the player moves before the circle is worked out again
ARRIVING = 3.0  # seconds: a moving car this close in time to the junction is about to enter it
BOX = 6.0  # metres: a car this far into an edge leaving the junction is still inside it
SPACING_QUEUE = 7.0  # metres between queued cars placed from the flow


class Cars:
    """The cars near the player, as arrays, on one road graph."""

    def __init__(self, graph: Graph, seed: int = 0, tables: dict | None = None):
        self.g = graph
        self.rng = np.random.default_rng(seed)
        if tables is None:
            self._index(graph)
        else:  # built before, by an earlier Cars on the same graph (contravia dev keeps them across saves)
            self.__dict__.update(tables)
        self._clear()

    def tables(self) -> dict:
        """What _index derived from the graph, which another Cars on the same graph can take as it is."""
        return {k: v for k, v in vars(self).items() if k not in ("g", "rng")}

    def _index(self, graph: Graph) -> None:
        """The graph's lanes, turns, directions and signal groups, as arrays and lists by edge and junction."""
        a = graph.a
        E = len(graph)
        self.lanes = a["g_lanes"].astype(np.int64)
        self.len = a["g_len"].astype(np.float64)
        self.vfree = a["g_vfree"].astype(np.float64)
        self.control = a["g_control"].astype(np.int64)
        self.to = a["g_to"].astype(np.int64)
        self.rank = np.array([RANK[CLASSES[c]] for c in a["g_class"]]) - np.where(np.isin(self.control, (2, 3)), 10, 0)
        # Turns of each edge, as lists, for drawing the next edge.
        self.turns: list[list[int]] = [[] for _ in range(E)]
        self.weights: list[list[float]] = [[] for _ in range(E)]
        for i, o, w in zip(a["g_turn_in"].tolist(), a["g_turn_out"].tolist(), a["g_turn_w"].tolist()):
            self.turns[i].append(o)
            self.weights[i].append(w)
        # Direction of travel at the start and end of each edge.
        poly, starts = a["g_poly"].astype(np.float64), a["g_poly_start"]
        d_end = poly[starts[1:] - 1] - poly[starts[1:] - 2]
        d_start = poly[starts[:-1] + 1] - poly[starts[:-1]]
        self.end_dir = d_end / np.maximum(np.hypot(*d_end.T), 1e-9)[:, None]
        self.start_dir = d_start / np.maximum(np.hypot(*d_start.T), 1e-9)[:, None]
        # Lane offsets: a street with a reverse edge is two-way, its lanes right of the centreline.
        pair = {(int(f), int(t), int(w)) for f, t, w in zip(a["g_from"], a["g_to"], a["g_way"])}
        self.twoway = np.array([(int(t), int(f), int(w)) in pair for f, t, w in zip(a["g_from"], a["g_to"], a["g_way"])],
                               bool)
        width = (a["g_width"].astype(np.float64) if "g_width" in a
                 else np.array([WIDTHS[CLASSES[c]] for c in a["g_class"]]))
        per_dir = np.where(self.twoway, width / 2, width) / np.maximum(self.lanes, 1)
        self.lane_w = np.minimum(LANE_WIDTH, per_dir)
        # Signal groups: at each signal, edges on the axis of the first edge entering it are group 0.
        self.group = np.zeros(E, np.int64)
        self.offset = np.zeros(E)
        for j in np.unique(self.to[self.control == 1]):
            into = np.flatnonzero(self.to == j)
            ref = self.end_dir[into[0]]
            self.group[into] = (np.abs(self.end_dir[into] @ ref) < np.cos(np.pi / 4)).astype(np.int64)
            self.offset[into] = (int(a["g_junction_node"][j]) * 7919) % CYCLE
        self.into_node: dict[int, np.ndarray] = {}
        self.out_of_node: dict[int, np.ndarray] = {}
        for j in range(len(a["g_junction"])):
            self.into_node[j] = np.flatnonzero(self.to == j)
            self.out_of_node[j] = np.flatnonzero(a["g_from"] == j)
        self.kind_len = np.array([LENGTH[k] for k in KINDS])
        self.look_kind = look_kinds()

    def _clear(self) -> None:
        self.edge = np.zeros(0, np.int64)
        self.s = np.zeros(0)
        self.lane = np.zeros(0, np.int64)
        self.v = np.zeros(0)
        self.look = np.zeros(0, np.int64)  # an index into rules.traffic.looks()
        self.next_edge = np.zeros(0, np.int64)
        self.wait = np.zeros(0)
        self.stopped = np.zeros(0, bool)
        self.cleared = np.zeros(0, bool)  # allowed into the junction at the end of its edge

    def __len__(self) -> int:
        return len(self.edge)

    @property
    def kind(self) -> np.ndarray:
        return self.look_kind[self.look]

    @property
    def length(self) -> np.ndarray:
        return self.kind_len[self.kind]

    def _draw_next(self, edges: np.ndarray) -> np.ndarray:
        out = np.full(len(edges), -1, np.int64)
        for k, e in enumerate(edges.tolist()):
            if self.turns[e]:
                w = np.asarray(self.weights[e])
                out[k] = self.turns[e][self.rng.choice(len(w), p=w / w.sum())]
        return out

    def _lane_for(self, edges: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        """The lane of the coming turn: rightmost for a right turn, leftmost for a left, any for straight on."""
        lanes = self.lanes[edges]
        lane = self.rng.integers(0, np.maximum(lanes, 1))
        ok = nxt >= 0
        d0, d1 = self.end_dir[edges], self.start_dir[np.where(ok, nxt, 0)]
        cross = d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]  # > 0: turning right (x east, z south)
        lane = np.where(ok & (cross > 0.5), 0, lane)
        lane = np.where(ok & (cross < -0.5), lanes - 1, lane)
        return lane

    def add(self, edges, s, v, looks) -> None:
        """Add cars on these edges at these distances and speeds; their next edge and lane are drawn."""
        edges = np.asarray(edges, np.int64)
        nxt = self._draw_next(edges)
        self.edge = np.r_[self.edge, edges]
        self.s = np.r_[self.s, np.asarray(s, float)]
        self.lane = np.r_[self.lane, self._lane_for(edges, nxt)]
        self.v = np.r_[self.v, np.asarray(v, float)]
        self.look = np.r_[self.look, np.asarray(looks, np.int64)]
        self.next_edge = np.r_[self.next_edge, nxt]
        self.wait = np.r_[self.wait, np.zeros(len(edges))]
        self.stopped = np.r_[self.stopped, np.zeros(len(edges), bool)]
        self.cleared = np.r_[self.cleared, np.zeros(len(edges), bool)]

    def remove(self, mask) -> None:
        """Drop the cars where mask is True."""
        keep = ~np.asarray(mask, bool)
        for name in ("edge", "s", "lane", "v", "look", "next_edge", "wait", "stopped", "cleared"):
            setattr(self, name, getattr(self, name)[keep])

    def offsets(self) -> np.ndarray:
        """Each car's distance to the right of its edge's centreline."""
        e = self.edge
        w, lanes = self.lane_w[e], self.lanes[e]
        right_of_centre = (lanes - self.lane - 0.5) * w
        return np.where(self.twoway[e], right_of_centre, right_of_centre - lanes * w / 2)

    def _signal_open(self, edges: np.ndarray, t: float, dist: np.ndarray, v: np.ndarray) -> np.ndarray:
        phase = (t + self.offset[edges]) % CYCLE
        start = self.group[edges] * (GREEN + AMBER)
        into = (phase - start) % CYCLE
        green = into < GREEN
        amber = (into >= GREEN) & (into < GREEN + AMBER)
        cannot_stop = dist < v ** 2 / (2 * BRAKE) + 1.0
        return green | (amber & cannot_stop)

    def _may_enter(self, i: int, t: float) -> bool:
        """Whether car i, at the head of its lane near the end of its edge, may enter the junction now."""
        e = int(self.edge[i])
        if self.next_edge[i] < 0:
            return False
        if self.control[e] == 1:  # a signal is obeyed however long the wait
            return bool(self._signal_open(np.array([e]), t, np.array([self.len[e] - self.s[i]]),
                                          np.array([self.v[i]]))[0])
        if self.wait[i] > PATIENCE:
            return True
        j = int(self.to[e])
        # Inside the junction: a car just past it on another edge leaving it, that is not where I go.
        leaving = self.out_of_node[j]
        box = np.isin(self.edge, leaving) & (self.s < BOX) & (self.edge != self.next_edge[i])
        if box.any():
            return False
        # About to enter: heads of the other approaches within NEAR of the junction that have priority.
        others = self.into_node[j]
        others = others[others != e]
        to_end = self.len[self.edge] - self.s
        arriving = (self.v > 1.0) & (to_end / np.maximum(self.v, 1e-3) < ARRIVING)
        # Only cars that will be in the junction soon, or were let through, count: a car waiting at its own line
        # does not block anyone, or every approach would wait for every other.
        near = np.flatnonzero(np.isin(self.edge, others) & (to_end < NEAR) & ~self.stopped
                              & (arriving | self.cleared))
        if not len(near):
            return True
        me = self.end_dir[e]
        for k in near.tolist():
            f = int(self.edge[k])
            if self.rank[f] > self.rank[e]:
                return False
            if self.rank[f] == self.rank[e]:
                d = self.end_dir[f]
                from_right = me[0] * d[1] - me[1] * d[0] < -0.5  # it comes from my right
                if from_right:
                    return False
        return True

    def step(self, dt: float, t: float, obstacles: list) -> None:
        """Advance dt seconds. t: seconds since midnight (signal phases). obstacles: (edge, lane, s[, v]) of
        things that are not cars, such as the player, each taken as a 4.5 m vehicle whose front is at s."""
        N = len(self)
        if N == 0 or dt <= 0:
            return
        L = self.length
        # Leaders: sort by (edge, lane, s); the next row in the same edge and lane is the car ahead.
        ob = np.array([(o[0], o[1], o[2], o[3] if len(o) > 3 else 0.0) for o in obstacles], float).reshape(-1, 4)
        e_all = np.r_[self.edge, ob[:, 0].astype(np.int64)]
        l_all = np.r_[self.lane, ob[:, 1].astype(np.int64)]
        s_all = np.r_[self.s, ob[:, 2]]
        v_all = np.r_[np.where(self.stopped, 0.0, self.v), ob[:, 3]]
        len_all = np.r_[L, np.full(len(ob), 4.5)]
        order = np.lexsort((s_all, l_all, e_all))
        gap = np.full(N + len(ob), np.inf)
        lead_v = np.zeros(N + len(ob))
        same = (e_all[order[1:]] == e_all[order[:-1]]) & (l_all[order[1:]] == l_all[order[:-1]])
        back, front = order[:-1][same], order[1:][same]
        gap[back] = s_all[front] - len_all[front] - s_all[back]
        lead_v[back] = v_all[front]
        gap, lead_v = gap[:N], lead_v[:N]

        # Heads of their lane near the edge's end: may they enter? If not, the stop line is their leader.
        head = np.isinf(gap)
        to_end = self.len[self.edge] - self.s
        asking = np.flatnonzero(head & (to_end < LOOK) & ~self.cleared)
        for i in asking.tolist():
            if self._may_enter(i, t):
                self.cleared[i] = True
        held = head & ~self.cleared
        gap = np.where(held, to_end - 0.5, gap)
        lead_v = np.where(held, 0.0, lead_v)
        # A cleared head follows the last car on its next edge, if any.
        go = np.flatnonzero(head & self.cleared)
        for i in go.tolist():
            nxt = self.next_edge[i]
            on = np.flatnonzero((self.edge == nxt) & (self.lane == min(self.lane[i], self.lanes[nxt] - 1)))
            if len(on):
                k = on[np.argmin(self.s[on])]
                gap[i] = to_end[i] + self.s[k] - L[k]
                lead_v[i] = 0.0 if self.stopped[k] else self.v[k]

        # Intelligent driver model.
        v0 = self.vfree[self.edge]
        dv = self.v - lead_v
        s_star = S0 + np.maximum(0.0, self.v * HEADWAY + self.v * dv / (2 * np.sqrt(ACCEL * BRAKE)))
        acc = ACCEL * (1 - (self.v / v0) ** 4 - (s_star / np.maximum(gap, 0.1)) ** 2)
        v_new = np.maximum(0.0, self.v + acc * dt)
        v_new = np.where(self.stopped, 0.0, v_new)
        advance = np.minimum(v_new * dt, np.maximum(gap - S0 * 0.5, 0.0))  # never into the car ahead
        self.v = np.where(advance < v_new * dt, advance / dt, v_new)
        self.s = self.s + advance
        self.wait = np.where(held & (to_end < LOOK) & (self.v < 0.5), self.wait + dt, 0.0)

        # Cars past the end of their edge move onto the next one.
        over = np.flatnonzero(self.s > self.len[self.edge])
        if len(over):
            rest = self.s[over] - self.len[self.edge[over]]
            nxt = self.next_edge[over]
            self.edge[over] = nxt
            self.s[over] = rest
            self.next_edge[over] = self._draw_next(nxt)
            self.lane[over] = self._lane_for(nxt, self.next_edge[over])
            self.cleared[over] = False
            self.wait[over] = 0.0

    def world(self, surface=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each car's centre (x, y, z) in metres, its heading and its pitch (positive nose up)."""
        L = self.length
        x, z, heading = self.g.points(self.edge, self.s - L / 2, self.offsets())
        if "g_poly_y" in self.g.a:  # the prepared road profile: cheap, and exact on level-across roads
            y = self.g.heights(self.edge, self.s - L / 2)
            pitch = np.arctan2(self.g.heights(self.edge, self.s) - self.g.heights(self.edge, self.s - L), L)
            return np.stack([x, y, z], axis=1), heading, pitch
        if surface is None:
            return np.stack([x, np.zeros_like(x), z], axis=1), heading, np.zeros_like(x)
        y = surface.height(x, z)
        fx, fz = x + np.cos(heading) * L / 2, z + np.sin(heading) * L / 2
        bx, bz = x - np.cos(heading) * L / 2, z - np.sin(heading) * L / 2
        pitch = np.arctan2(surface.height(fx, fz) - surface.height(bx, bz), L)
        return np.stack([x, y, z], axis=1), heading, pitch


class TrafficLayer:
    """Cars on the edges near the player, kept consistent with the flow layer.

    On those edges the cars are the truth; everywhere else the flow is. A vehicle is never in both. Changes to
    the flow run in its thread (runner.submit); the flow answers through `inbox`: which edges it handed over
    and with how many vehicles, and how many vehicles it sent into edges the cars own.
    """

    MAX_ENTER = 200  # edges taken over per update, nearest first, so a teleport does not stall one frame

    def __init__(self, graph: Graph, runner, radius: float = 400.0, seed: int = 0, tables: dict | None = None):
        from collections import deque

        from contravia.geo import CELL

        self.g, self.runner, self.radius = graph, runner, radius
        self.cars = Cars(graph, seed, tables)
        E = len(graph)
        self.centre = (graph.a["g_cell"].astype(np.float64) + 0.5) * CELL if E else np.zeros((0, 2))
        self.owned = np.zeros(E, bool)  # edges the cars own or have asked for
        self.carry = np.zeros(E)  # the fraction of a vehicle each owned edge holds on top of its cars
        self.waiting = np.zeros(E)  # vehicles that arrived on an owned edge but have no room at its start yet
        self.inbox = deque()
        self.counts = {"spawned": 0, "removed": 0}
        self.closed_until: dict[int, object] = {}  # edge -> datetime when its accident clears
        self.stored: dict[int, list] = {}  # edge -> [(s, lane, look)] of accident cars while the edge is away
        self.kind_width = np.array([WIDTH[k] for k in KINDS])
        self._rng = np.random.default_rng(seed + 1)
        self.debt = 0.0  # vehicles owed to the flow by rounding (never handed over as a negative count)
        self._at, self._backlog = None, False  # where the circle was last worked out, and whether edges wait
        self._owned_idx = np.zeros(0, np.int64)

    # Messages run in the flow's thread.
    def _hand_over(self, edges):
        def fn(flow):
            flow.owned[edges] = True
            view = flow.view()
            self.inbox.append(("enter", edges, view.n[edges], view.queue[edges], view.speed[edges]))
        return fn

    def _take_arrivals(self, flow):
        idx = np.flatnonzero(flow.owned & (flow.arrived >= 1.0))
        if len(idx):
            k = np.floor(flow.arrived[idx])
            flow.arrived[idx] -= k
            self.inbox.append(("arrive", idx, k))

    def total(self) -> float:
        """Every vehicle once: the flow's own edges, the cars, and what is in transit between the layers."""
        flow = self.runner.flow
        transit = sum(m[2].sum() for m in self.inbox)
        return float(flow.n[~flow.owned].sum() + flow.arrived[flow.owned].sum() + len(self.cars)
                     + self.carry.sum() + self.waiting.sum() + transit
                     + sum(len(v) for v in self.stored.values()) + self.debt)

    def _looks(self, edge: int, k: int) -> np.ndarray:
        from contravia.rules.traffic import pick

        return pick(self.runner.flow.config, edge, self.runner.clock.now(), k)

    def _enter(self, edges, n, queue, speed) -> None:
        for e, ne, qe, ve in zip(edges.tolist(), n.tolist(), queue.tolist(), speed.tolist()):
            if not self.owned[e]:  # left again before the flow answered: give the vehicles back
                self.runner.submit(lambda flow, e=e, ne=ne: self._give_back(flow, [e], [ne]))
                continue
            if e in self.stored:
                kept = self.stored.pop(e)
                self.cars.add([e] * len(kept), [c[0] for c in kept], [0.0] * len(kept), [c[2] for c in kept])
                self.cars.stopped[-len(kept):] = True
                self.cars.lane[-len(kept):] = [c[1] for c in kept]
            # Most streets hold a fraction of a vehicle: round at random so a block with 0.2 gets a car one time
            # in five, and keep the difference (which may be negative) so the total stays exact.
            frac = ne - np.floor(ne)
            k = int(np.floor(ne)) + int(self._rng.random() < frac)
            self.carry[e] += ne - k
            if k == 0:
                continue
            q = min(k, int(np.floor(qe)))
            length = self.cars.len[e]
            s_queue = length - 0.5 - SPACING_QUEUE * np.arange(q)
            free = k - q
            room = max(length - SPACING_QUEUE * q, 0.0)
            s_free = (np.arange(free) + 0.5) * room / max(free, 1)
            s = np.clip(np.r_[s_queue, s_free], 0.0, length)
            v = np.r_[np.zeros(q), np.full(free, ve)]
            self.cars.add(np.full(k, e), s, v, self._looks(e, k))
            self.counts["spawned"] += k

    def _arrive(self, edges, k) -> None:
        self.waiting[edges] += k
        for e in np.flatnonzero(self.waiting >= 1.0).tolist():
            if not self.owned[e]:
                continue
            on = self.cars.edge == e
            if on.any() and self.cars.s[on].min() < SPACING_QUEUE:
                continue  # no room at the start yet
            self.waiting[e] -= 1
            self.cars.add([e], [0.0], [self.cars.vfree[e] * 0.5], self._looks(e, 1))
            self.counts["spawned"] += 1

    def _settle(self, counts: np.ndarray) -> np.ndarray:
        """Counts to give back, none negative, with what rounding owes taken from them or kept as debt."""
        total = float(counts.sum()) + self.debt
        positive = np.maximum(counts, 0.0)
        if total <= 0 or positive.sum() == 0:
            self.debt = total if positive.sum() == 0 or total <= 0 else 0.0
            return np.zeros_like(counts) if total <= 0 else positive
        self.debt = 0.0
        return positive * (total / positive.sum())

    @staticmethod
    def _give_back(flow, edges, counts):
        edges = np.asarray(edges, np.int64)
        flow.n[edges] = np.asarray(counts, float) + flow.arrived[edges]
        flow.arrived[edges] = 0.0
        flow.owned[edges] = False

    def update(self, x: float, z: float, dt: float, obstacles: list) -> None:
        """One frame: hand edges over as the player moves, place arriving vehicles, drive, and report counts."""
        if len(self.g) == 0:
            return
        while self.inbox:
            msg = self.inbox.popleft()
            if msg[0] == "enter":
                self._enter(*msg[1:])
            else:
                self._arrive(*msg[1:])
        # Which edges belong to the circle changes only when the player moves; checking every edge costs ms.
        if self._at is None or np.hypot(x - self._at[0], z - self._at[1]) > REFRESH or self._backlog:
            self._circle(x, z)
        self._drive(dt, obstacles)

    def _circle(self, x: float, z: float) -> None:
        """Give back the edges that left the circle and ask for those that entered it, nearest first."""
        E = len(self.g)
        self._at = (x, z)
        want = np.hypot(*(self.centre - (x, z)).T) <= self.radius
        leaving = np.flatnonzero(self.owned & ~want)
        if len(leaving):
            crashed = np.isin(self.cars.edge, leaving) & self.cars.stopped
            for i in np.flatnonzero(crashed).tolist():  # accident cars wait here, out of the flow, until it clears
                e = int(self.cars.edge[i])
                self.stored.setdefault(e, []).append((float(self.cars.s[i]), int(self.cars.lane[i]),
                                                      int(self.cars.look[i])))
            self.cars.remove(crashed)
            on = np.isin(self.cars.edge, leaving)
            counts = (np.bincount(self.cars.edge[on], minlength=E)[leaving] + self.carry[leaving]
                      + self.waiting[leaving])
            self.cars.remove(on)
            self.counts["removed"] += int(on.sum())
            self.carry[leaving] = 0.0
            self.waiting[leaving] = 0.0
            counts = self._settle(counts)
            self.owned[leaving] = False
            self.runner.submit(lambda flow, e=leaving, c=counts: self._give_back(flow, e, c))
        entering = np.flatnonzero(want & ~self.owned)
        self._backlog = len(entering) > self.MAX_ENTER
        if len(entering):
            d = np.hypot(*(self.centre[entering] - (x, z)).T)
            entering = entering[np.argsort(d)][: self.MAX_ENTER]
            self.owned[entering] = True
            self.runner.submit(self._hand_over(entering))
        self._owned_idx = np.flatnonzero(self.owned)

    def _drive(self, dt: float, obstacles: list) -> None:
        """Clear finished accidents, move the cars, return those that left the circle, and report counts."""
        when = self.runner.clock.now()
        for e, until in list(self.closed_until.items()):
            if until <= when:
                del self.closed_until[e]
                self.cars.stopped[self.cars.edge == e] = False
                if e in self.stored:
                    k = len(self.stored.pop(e))
                    self.runner.submit(lambda flow, e=e, k=k: flow.n.__setitem__(e, flow.n[e] + k))
        t = when.hour * 3600 + when.minute * 60 + when.second + when.microsecond / 1e6
        self.cars.step(dt, t, obstacles)

        out = ~self.owned[self.cars.edge]
        if out.any():
            gone = self.cars.edge[out]
            self.cars.remove(out)
            self.counts["removed"] += len(gone)
            self.runner.submit(lambda flow, g=gone: np.add.at(flow.n, g, 1.0))

        owned = self._owned_idx  # every car is on one of these now
        on = np.bincount(np.searchsorted(owned, self.cars.edge), minlength=len(owned))
        counts = on + self.carry[owned] + self.waiting[owned]

        def report(flow, idx=owned, c=counts):
            mine = flow.owned[idx]
            flow.n[idx[mine]] = c[mine] + flow.arrived[idx[mine]]
            self._take_arrivals(flow)

        self.runner.submit(report)

    def accident(self, edge: int, minutes: float = 40.0) -> None:
        """Close an edge for `minutes`: its cars stop where they are and stay until it clears."""
        from datetime import timedelta

        until = self.runner.clock.now() + timedelta(minutes=minutes)
        self.closed_until[int(edge)] = until
        self.cars.stopped[self.cars.edge == edge] = True
        self.runner.submit(lambda flow: flow.close([edge], until))

    def locate(self, x: float, z: float, heading: float, within: float = 8.0):
        """(edge, lane, s) of a vehicle at (x, z) driving along `heading`, or None when it is off the streets."""
        a = self.g.a
        cand = self.g.edges_near(x, z, 120.0)
        if not len(cand):
            return None
        p0, p1 = a["g_junction"][a["g_from"][cand]].astype(float), a["g_junction"][a["g_to"][cand]].astype(float)
        d = p1 - p0
        L2 = np.maximum((d ** 2).sum(1), 1e-9)
        t = np.clip(((np.array([x, z]) - p0) * d).sum(1) / L2, 0, 1)
        foot = p0 + d * t[:, None]
        dist = np.hypot(*(foot - (x, z)).T)
        u = d / np.sqrt(L2)[:, None]
        along = u @ np.array([np.cos(heading), np.sin(heading)])
        dist = np.where(along > 0.3, dist, np.inf)
        k = int(np.argmin(dist))
        if dist[k] > within:
            return None
        e = int(cand[k])
        right = (np.array([x, z]) - foot[k]) @ np.array([-u[k, 1], u[k, 0]])  # metres right of the centreline
        lanes = int(self.cars.lanes[e])
        w = self.cars.lane_w[e]
        offs = (lanes - np.arange(lanes) - 0.5) * w - (0 if self.cars.twoway[e] else lanes * w / 2)
        lane = int(np.argmin(np.abs(offs - right)))
        return e, lane, float(t[k] * self.cars.len[e])

    def boxes_near(self, x: float, z: float, r: float = 30.0) -> np.ndarray:
        """The outlines of the cars within r metres as (W, 2, 2) wall segments, for the player's collisions."""
        if not len(self.cars):
            return np.zeros((0, 2, 2))
        xyz, heading, _ = self.cars.world()
        near = np.flatnonzero(np.hypot(xyz[:, 0] - x, xyz[:, 2] - z) < r)
        segs = []
        for i in near.tolist():
            c, s = np.cos(heading[i]), np.sin(heading[i])
            hl, hw = self.cars.length[i] / 2, self.kind_width[self.cars.kind[i]] / 2
            corners = [(xyz[i, 0] + c * fx - s * fz, xyz[i, 2] + s * fx + c * fz)
                       for fx, fz in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            segs += [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
        return np.array(segs, float).reshape(-1, 2, 2)

    def stats(self) -> dict:
        return {"cars": len(self.cars), "edges": int(self.owned.sum()), **self.counts}
