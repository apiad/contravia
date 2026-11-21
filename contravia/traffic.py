"""The car layer: cars near the player, driving on their lanes.

Each car is a row in a few arrays, never an object: the edge it is on, how far along it its front is (s), its
lane (0 is the rightmost), speed, look (and so kind), the edge it takes next, and how long it has waited at a junction.

A car's drawn path is smooth: its sideways offset eases toward its lane's at LANE_SPEED, and the last metres of
an edge and the first of the next are joined by a quadratic curve, so it neither jumps across a junction nor
turns at every vertex of the street. On a street with several lanes a car changes lane by MOBIL: when the other
lane lets it go faster than it loses the car behind there, or to reach the lane of its coming turn.

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
HEIGHT_STEP = 2.0  # metres between the road heights kept along each edge
JUNCTION = 7.0  # metres before and after a junction over which a car's path curves from one edge to the next
LANE_SPEED = 1.1  # m/s: how fast a car's offset moves toward its lane's, so a lane change takes about 3 s
POLITE, GAIN, SAFE = 0.3, 0.4, 3.0  # MOBIL: politeness, m/s² of gain to change, the most braking it may impose
SETTLE = 4.0  # seconds after a lane change before a car considers another
DECIDE = 0.2  # seconds between lane-change decisions, each toward one side, the next toward the other


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
        self.twoway, self.lane_w = lane_geometry(a)
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
        self._index_heights(graph)

    def _index_heights(self, graph: Graph) -> None:
        """The road's height every HEIGHT_STEP metres along each edge (and at its end), from the graph's profile,
        so world() reads a car's height by index instead of searching the polylines every frame."""
        self.h_start = None
        if "g_poly_y" not in graph.a:
            return
        n = np.ceil(self.len / HEIGHT_STEP).astype(np.int64) + 1  # samples per edge: 0, 2, 4, ... and the end
        n = np.maximum(n, 2)
        self.h_start = np.concatenate([[0], np.cumsum(n)])
        edge = np.repeat(np.arange(len(n)), n)
        k = np.arange(self.h_start[-1]) - self.h_start[edge]
        self.h = graph.heights(edge, np.minimum(k * HEIGHT_STEP, self.len[edge])).astype(np.float32)

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
        self.prev_edge = np.zeros(0, np.int64)  # the edge it came from, -1 when it was placed here
        self.want = np.zeros(0, np.int64)  # the lane of its coming turn, -1 when any lane goes there
        self.off = np.zeros(0)  # its drawn offset right of the centreline, easing toward offsets(); NaN: not yet
        self.since = np.zeros(0)  # seconds since its last lane change

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

    def _turn_lane(self, edges: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        """The lane of the coming turn: rightmost for a right turn, leftmost for a left, -1 (any) for straight on."""
        lanes = self.lanes[edges]
        ok = nxt >= 0
        d0, d1 = self.end_dir[edges], self.start_dir[np.where(ok, nxt, 0)]
        cross = d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]  # > 0: turning right (x east, z south)
        want = np.where(ok & (cross > 0.5), 0, -1)
        return np.where(ok & (cross < -0.5), lanes - 1, want)

    def _lane_for(self, edges: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        """A lane for a car placed on edges: that of its coming turn, or any for straight on."""
        want = self._turn_lane(edges, nxt)
        return np.where(want >= 0, want, self.rng.integers(0, np.maximum(self.lanes[edges], 1)))

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
        self.prev_edge = np.r_[self.prev_edge, np.full(len(edges), -1, np.int64)]
        self.want = np.r_[self.want, self._turn_lane(edges, nxt)]
        self.off = np.r_[self.off, np.full(len(edges), np.nan)]
        self.since = np.r_[self.since, np.full(len(edges), SETTLE)]

    def remove(self, mask) -> None:
        """Drop the cars where mask is True."""
        keep = ~np.asarray(mask, bool)
        for name in ("edge", "s", "lane", "v", "look", "next_edge", "wait", "stopped", "cleared", "prev_edge", "want",
                     "off", "since"):
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
        ob = np.array([(o[0], o[1], o[2], o[3] if len(o) > 3 else 0.0) for o in obstacles], float).reshape(-1, 4)
        self._decide = getattr(self, "_decide", 0.0) + dt
        if self._decide >= DECIDE:
            self._side = -getattr(self, "_side", 1)
            self._change_lanes(self._decide, ob, self._side)
            self._decide = 0.0
        L = self.length
        # Leaders: sort by (edge, lane, s); the next row in the same edge and lane is the car ahead.
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
        acc = _idm(self.v, self.vfree[self.edge], gap, lead_v)
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
            self.prev_edge[over] = self.edge[over]
            self.edge[over] = nxt
            self.s[over] = rest
            self.next_edge[over] = self._draw_next(nxt)
            self.lane[over] = np.minimum(self.lane[over], self.lanes[nxt] - 1)  # it keeps its side of the street
            self.want[over] = self._turn_lane(nxt, self.next_edge[over])
            self.cleared[over] = False
            self.wait[over] = 0.0

        target = self.offsets()
        off = np.where(np.isnan(self.off), target, self.off)
        self.off = off + np.clip(target - off, -LANE_SPEED * dt, LANE_SPEED * dt)

    def _change_lanes(self, dt: float, ob: np.ndarray, side: int) -> None:
        """MOBIL, toward the left (side 1, a higher lane) or the right (-1) only, so two cars never take the same
        gap from both sides at once. A car with a turn ahead moves toward its lane whenever it is safe and takes
        it anyway LOOK metres before the end; any other car changes when its gain, less POLITE times what the car
        behind there loses, passes GAIN. Never into a gap under S0, or making the car behind brake past SAFE."""
        self.since += dt
        to_end = self.len[self.edge] - self.s
        force = (self.want >= 0) & (self.lane != self.want) & (to_end < LOOK)
        self.lane = np.where(force, self.want, self.lane)
        self.since[force] = 0.0
        target = self.lane + side
        heading = np.where(self.want >= 0, np.sign(self.want - self.lane) == side, True)
        cand = np.flatnonzero((self.lanes[self.edge] > 1) & (target >= 0) & (target < self.lanes[self.edge])
                              & heading & ~self.stopped & ~self.cleared & (to_end > LOOK) & (self.since > SETTLE))
        if not len(cand):
            return
        # Every vehicle, the obstacles too, sorted by (edge, lane, s) as one key.
        e_all = np.r_[self.edge, ob[:, 0].astype(np.int64)]
        l_all = np.r_[self.lane, ob[:, 1].astype(np.int64)]
        s_all = np.r_[self.s, ob[:, 2]]
        v_all = np.r_[np.where(self.stopped, 0.0, self.v), ob[:, 3]]
        len_all = np.r_[self.length, np.full(len(ob), 4.5)]
        group = e_all * 64 + l_all
        order = np.argsort(group * 1e5 + s_all)
        keys, groups = (group * 1e5 + s_all)[order], group[order]

        def around(g, at):
            """The rows just ahead of and just behind position at in lane group g, -1 where there is none."""
            k = np.searchsorted(keys, g * 1e5 + at, side="right")
            ahead = np.where((k < len(keys)) & (groups[np.minimum(k, len(keys) - 1)] == g),
                             order[np.minimum(k, len(keys) - 1)], -1)
            k = k - 1  # in its own lane that is the car itself: only other lanes are asked for the car behind
            behind = np.where((k >= 0) & (groups[np.maximum(k, 0)] == g), order[np.maximum(k, 0)], -1)
            return ahead, behind

        def gap_to(front, back_s):
            return np.where(front >= 0, s_all[front] - len_all[front] - back_s, np.inf)

        s, v, v0, ln = self.s[cand], self.v[cand], self.vfree[self.edge[cand]], self.length[cand]
        own_ahead, _ = around(group[cand], s)
        new_ahead, new_behind = around(self.edge[cand] * 64 + target[cand], s)
        a_now = _idm(v, v0, gap_to(own_ahead, s), np.where(own_ahead >= 0, v_all[own_ahead], 0.0))
        gap_new = gap_to(new_ahead, s)
        a_new = _idm(v, v0, gap_new, np.where(new_ahead >= 0, v_all[new_ahead], 0.0))
        has_f = new_behind >= 0
        f = np.where(has_f, new_behind, 0)
        f_s, f_v, f_v0 = s_all[f], v_all[f], self.vfree[e_all[f]]
        f_gap_new = np.where(has_f, s - ln - f_s, np.inf)
        a_f_new = np.where(has_f, _idm(f_v, f_v0, f_gap_new, v), 0.0)
        a_f_old = np.where(has_f, _idm(f_v, f_v0, gap_to(new_ahead, f_s),
                                       np.where(new_ahead >= 0, v_all[new_ahead], 0.0)), 0.0)
        safe = (gap_new > S0) & (f_gap_new > S0) & (a_f_new > -SAFE)
        gain = a_new - a_now + POLITE * (a_f_new - a_f_old)
        go = safe & np.where(self.want[cand] >= 0, True, (gain > GAIN) & (v > 1.0))
        moved = cand[go]
        self.lane[moved] = target[moved]
        self.since[moved] = 0.0

    def _height(self, s: np.ndarray) -> np.ndarray:
        """The road's height at distance s along each car's edge, between the two kept heights around it."""
        e = self.edge
        length = self.len[e]
        s = np.clip(s, 0.0, length)
        k = np.minimum((s / HEIGHT_STEP).astype(np.int64), self.h_start[e + 1] - self.h_start[e] - 2)
        i = self.h_start[e] + k
        span = np.maximum(np.minimum(HEIGHT_STEP, length - k * HEIGHT_STEP), 1e-9)  # the last span is shorter
        f = np.clip((s - k * HEIGHT_STEP) / span, 0.0, 1.0)
        return self.h[i] + (self.h[i + 1] - self.h[i]) * f

    def _span(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """How many metres before and after the junction from edge a to edge b the path curves."""
        return np.minimum(JUNCTION, 0.45 * np.minimum(self.len[a], self.len[b]))

    def path(self, s: np.ndarray, off: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, z, heading) of the point s metres along each car's edge (s < 0 is on the edge it came from), off
        metres right of the centreline (its drawn offset by default). Within _span of a junction the point lies on
        the quadratic curve from the edge before it to the edge after, so a car turns smoothly through it."""
        if off is None:
            off = np.where(np.isnan(self.off), self.offsets(), self.off)
        e, prev, nxt = self.edge.copy(), self.prev_edge.copy(), self.next_edge.copy()
        s = np.asarray(s, float).copy()
        back = (s < 0) & (prev >= 0)  # behind its edge's start: on the edge it came from
        nxt[back], e[back], s[back] = e[back], prev[back], s[back] + self.len[prev[back]]
        prev[back] = -1
        x, z, heading = self.g.points(e, s, off)
        a, b, u = np.full(len(e), -1), np.zeros(len(e), np.int64), np.zeros(len(e))
        tail = nxt >= 0
        j = self._span(e, np.where(tail, nxt, e))
        tail &= s > self.len[e] - j
        a[tail], b[tail], u[tail] = e[tail], nxt[tail], (s[tail] - self.len[e[tail]] + j[tail]) / (2 * j[tail])
        head = ~tail & (prev >= 0)
        j = self._span(np.where(head, prev, e), e)
        head &= s < j
        a[head], b[head], u[head] = prev[head], e[head], (s[head] + j[head]) / (2 * j[head])
        k = np.flatnonzero(a >= 0)
        if len(k):
            x[k], z[k], heading[k] = _curve(self.g, a[k], b[k], self._span(a[k], b[k]), np.clip(u[k], 0, 1),
                                            off[k], self.len)
        return x, z, heading

    def world(self, surface=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each car's centre (x, y, z) in metres, its heading and its pitch (positive nose up)."""
        L = self.length
        x, z, heading = self.path(self.s - L / 2)
        if self.h_start is not None:  # the prepared road profile: exact on level-across roads
            y = self._height(self.s - L / 2)
            pitch = np.arctan2(self._height(self.s) - self._height(self.s - L), L)
            return np.stack([x, y, z], axis=1), heading, pitch
        if surface is None:
            return np.stack([x, np.zeros_like(x), z], axis=1), heading, np.zeros_like(x)
        y = surface.height(x, z)
        fx, fz = x + np.cos(heading) * L / 2, z + np.sin(heading) * L / 2
        bx, bz = x - np.cos(heading) * L / 2, z - np.sin(heading) * L / 2
        pitch = np.arctan2(surface.height(fx, fz) - surface.height(bx, bz), L)
        return np.stack([x, y, z], axis=1), heading, pitch


def lane_geometry(a: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each edge's two-way flag and lane width, from the graph's arrays: a street with a reverse edge is two-way,
    its lanes right of the centreline."""
    pair = {(int(f), int(t), int(w)) for f, t, w in zip(a["g_from"], a["g_to"], a["g_way"])}
    twoway = np.array([(int(t), int(f), int(w)) in pair for f, t, w in zip(a["g_from"], a["g_to"], a["g_way"])], bool)
    width = (a["g_width"].astype(np.float64) if "g_width" in a
             else np.array([WIDTHS[CLASSES[c]] for c in a["g_class"]]))
    per_dir = np.where(twoway, width / 2, width) / np.maximum(a["g_lanes"].astype(np.int64), 1)
    return twoway, np.minimum(LANE_WIDTH, per_dir)


def lane_offsets(lanes: int, lane_w: float, twoway: bool) -> np.ndarray:
    """The centres of an edge's lanes, 0 the rightmost, in metres right of its centreline."""
    offs = (lanes - np.arange(lanes) - 0.5) * lane_w
    return offs if twoway else offs - lanes * lane_w / 2


def _idm(v, v0, gap, lead_v):
    """The intelligent driver model's acceleration at speed v toward v0, gap metres behind a car at lead_v."""
    s_star = S0 + np.maximum(0.0, v * HEADWAY + v * (v - lead_v) / (2 * np.sqrt(ACCEL * BRAKE)))
    return ACCEL * (1 - (v / v0) ** 4 - (s_star / np.maximum(gap, 0.1)) ** 2)


def _curve(g, a, b, span, u, off, length):
    """(x, z, heading) at u in [0, 1] along the quadratic curve from span metres before the end of edge a to span
    metres into edge b, off metres right of both. Its middle point is where the two directions meet; when they
    are nearly parallel (straight on, or a U-turn) it is the middle of the ends pushed span/2 ahead."""
    x0, z0, h0 = g.points(a, length[a] - span, off)
    x2, z2, h2 = g.points(b, span, off)
    p0, p2 = np.stack([x0, z0], 1), np.stack([x2, z2], 1)
    t0, t2 = np.stack([np.cos(h0), np.sin(h0)], 1), np.stack([np.cos(h2), np.sin(h2)], 1)
    cross = t0[:, 0] * t2[:, 1] - t0[:, 1] * t2[:, 0]
    d = p2 - p0
    ahead = (d[:, 0] * t2[:, 1] - d[:, 1] * t2[:, 0]) / np.where(np.abs(cross) > 1e-6, cross, 1.0)
    meet = p0 + t0 * ahead[:, None]
    bent = (np.abs(cross) > 0.2) & (ahead > 0) & (ahead < 3 * span)
    p1 = np.where(bent[:, None], meet, (p0 + p2) / 2 + (t0 - t2) * (span / 4)[:, None])
    w = u[:, None]
    p = (1 - w) ** 2 * p0 + 2 * w * (1 - w) * p1 + w ** 2 * p2
    tangent = 2 * (1 - w) * (p1 - p0) + 2 * w * (p2 - p1)
    return p[:, 0], p[:, 1], np.arctan2(tangent[:, 1], tangent[:, 0])


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
        self._asked = 0  # edges asked of the flow whose vehicles have not come back yet
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
                self._asked -= len(msg[1])
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
            self._asked += len(entering)
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
        self.cars.step(dt, self._seconds(when), obstacles)

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

    @staticmethod
    def _seconds(when) -> float:
        """The time of day in seconds, which the signals' phases run on."""
        return when.hour * 3600 + when.minute * 60 + when.second + when.microsecond / 1e6

    def signal_open(self, edge: int, dist: float, v: float) -> bool:
        """Whether a vehicle dist metres before the end of edge at v m/s may go on: no signal there, green, or
        amber too close to stop, as for the traffic."""
        if self.cars.control[edge] != 1:
            return True
        t = self._seconds(self.runner.clock.now())
        return bool(self.cars._signal_open(np.array([edge]), t, np.array([dist]), np.array([v]))[0])

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
        return self.moving_boxes_near(x, z, r)[0]

    def moving_boxes_near(self, x: float, z: float, r: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
        """The outlines of the cars within r metres as (W, 2, 2) segments, and each segment's car's velocity
        (W, 2) in m/s, for what must know where they are going."""
        if not len(self.cars):
            return np.zeros((0, 2, 2)), np.zeros((0, 2))
        xyz, heading, _ = self.cars.world()
        near = np.flatnonzero(np.hypot(xyz[:, 0] - x, xyz[:, 2] - z) < r)
        segs, vel = [], []
        for i in near.tolist():
            c, s = np.cos(heading[i]), np.sin(heading[i])
            hl, hw = self.cars.length[i] / 2, self.kind_width[self.cars.kind[i]] / 2
            corners = [(xyz[i, 0] + c * fx - s * fz, xyz[i, 2] + s * fx + c * fz)
                       for fx, fz in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            segs += [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
            vel += [(c * self.cars.v[i], s * self.cars.v[i])] * 4
        return np.array(segs, float).reshape(-1, 2, 2), np.array(vel, float).reshape(-1, 2)

    def unplaced(self) -> int:
        """Edges near the player not placed yet: those asked of the flow and not answered, and one while the
        circle is not worked out or has edges left to ask for."""
        return self._asked + int(self._at is None or self._backlog)

    def stats(self) -> dict:
        return {"cars": len(self.cars), "edges": int(self.owned.sum()), **self.counts}
