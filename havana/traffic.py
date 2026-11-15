"""The car layer: cars near the player, driving on their lanes.

Each car is a row in a few arrays, never an object: the edge it is on, how far along it its front is (s), its
lane (0 is the rightmost), speed, kind, the edge it takes next, and how long it has waited at a junction.

Every frame each car finds what is ahead of it in its lane (another car, a stop line, the player) and
accelerates by the intelligent driver model: toward the street's free speed, keeping a gap that grows with
speed, braking smoothly. At the end of an edge a car may enter the junction if its signal is green, or, without
a signal, if no car with priority is about to enter: the bigger street goes first, a stop or give-way sign
loses, and between equal streets the car from the right goes first. A car that has waited 8 s goes anyway,
so a junction never locks up.
"""

import numpy as np

from havana.graph import CLASSES, Graph
from havana.rules.traffic import KINDS

LENGTH = {"car": 4.5, "taxi": 5.0, "moto": 2.0, "bus": 12.0, "truck": 8.0}
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
BOX = 6.0  # metres: a car this far into an edge leaving the junction is still inside it


class Cars:
    """The cars near the player, as arrays, on one road graph."""

    def __init__(self, graph: Graph, seed: int = 0):
        self.g = graph
        self.rng = np.random.default_rng(seed)
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
        width = np.array([WIDTHS[CLASSES[c]] for c in a["g_class"]])
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
        self._clear()

    def _clear(self) -> None:
        self.edge = np.zeros(0, np.int64)
        self.s = np.zeros(0)
        self.lane = np.zeros(0, np.int64)
        self.v = np.zeros(0)
        self.kind = np.zeros(0, np.int64)
        self.next_edge = np.zeros(0, np.int64)
        self.wait = np.zeros(0)
        self.stopped = np.zeros(0, bool)
        self.cleared = np.zeros(0, bool)  # allowed into the junction at the end of its edge

    def __len__(self) -> int:
        return len(self.edge)

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

    def add(self, edges, s, v, kinds) -> None:
        """Add cars on these edges at these distances and speeds; their next edge and lane are drawn."""
        edges = np.asarray(edges, np.int64)
        nxt = self._draw_next(edges)
        self.edge = np.r_[self.edge, edges]
        self.s = np.r_[self.s, np.asarray(s, float)]
        self.lane = np.r_[self.lane, self._lane_for(edges, nxt)]
        self.v = np.r_[self.v, np.asarray(v, float)]
        self.kind = np.r_[self.kind, np.asarray(kinds, np.int64)]
        self.next_edge = np.r_[self.next_edge, nxt]
        self.wait = np.r_[self.wait, np.zeros(len(edges))]
        self.stopped = np.r_[self.stopped, np.zeros(len(edges), bool)]
        self.cleared = np.r_[self.cleared, np.zeros(len(edges), bool)]

    def remove(self, mask) -> None:
        """Drop the cars where mask is True."""
        keep = ~np.asarray(mask, bool)
        for name in ("edge", "s", "lane", "v", "kind", "next_edge", "wait", "stopped", "cleared"):
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
        near = np.flatnonzero(np.isin(self.edge, others) & (self.len[self.edge] - self.s < NEAR) & ~self.stopped)
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
        if N == 0:
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
        if surface is None:
            return np.stack([x, np.zeros_like(x), z], axis=1), heading, np.zeros_like(x)
        y = surface.height(x, z)
        fx, fz = x + np.cos(heading) * L / 2, z + np.sin(heading) * L / 2
        bx, bz = x - np.cos(heading) * L / 2, z - np.sin(heading) * L / 2
        pitch = np.arctan2(surface.height(fx, fz) - surface.height(bx, bz), L)
        return np.stack([x, y, z], axis=1), heading, pitch
