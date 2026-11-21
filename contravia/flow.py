"""The flow layer: how many vehicles are on every street of the province, one step per second.

Each directed edge holds one number, n, the vehicles on it. A step moves vehicles from edges to the edges their
turns lead to, as many as the edge can send (its demand) and the receiving edges have room for (their supply).
New vehicles start on streets with buildings near them and others park, at rates the demand settings give
(contravia/rules/traffic.py), so the province holds about as many vehicles as are moving at that hour.

Junctions are first in, first out: when one of an edge's turns cannot take vehicles, the whole edge waits, as
the car at its head blocks the ones behind. That is how a closed street makes a queue that grows backwards.
While an edge is blocked by a closure, the junctions before it send only a tenth of their usual share into it,
so traffic goes around.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from contravia.graph import CLASSES, Graph
from contravia.rules.traffic import TrafficConfig, moving

CAPACITY = 0.5  # vehicles per second per lane leaving an edge
CAPACITY_CONTROLLED = 0.3  # the same at an edge ending at a signal or a stop
SPACING = 7.0  # metres per vehicle per lane in a jam
WALKING = 1.4  # m/s, the speed of a jammed edge
AVOID = 0.1  # share of the usual turn weight into an edge blocked by a closure
SOURCE_FACTOR = {"residential": 1.0, "living_street": 1.0, "unclassified": 1.0, "tertiary": 0.6,
                 "secondary": 0.4, "primary": 0.2, "trunk": 0.2, "motorway": 0.0, "service": 0.5}
MORNING, AFTERNOON = (6, 10), (15, 19)  # hours when `attract` pulls trips in, then sends them out


@dataclass(frozen=True)
class FlowView:
    """What the flow layer publishes after a step: per edge, vehicles, speed in m/s, and vehicles queued."""

    n: np.ndarray
    speed: np.ndarray
    queue: np.ndarray
    when: datetime | None
    arrived: np.ndarray  # vehicles sent into owned edges so far, cumulative; the car layer turns them into cars


class Flow:
    """Vehicles per edge over the whole graph, stepped by step(dt, when)."""

    def __init__(self, graph: Graph, config: TrafficConfig):
        a = graph.a
        self.graph = graph
        self.length = a["g_len"].astype(np.float64)
        self.lanes = a["g_lanes"].astype(np.float64)
        self.vfree = a["g_vfree"].astype(np.float64)
        self.jam = self.lanes * self.length / SPACING
        controlled = np.isin(a["g_control"], (1, 2))
        self.cap = self.lanes * np.where(controlled, CAPACITY_CONTROLLED, CAPACITY)
        self.t_in, self.t_out = a["g_turn_in"].astype(np.int64), a["g_turn_out"].astype(np.int64)
        self.t_w = a["g_turn_w"].astype(np.float64)
        # An edge with no turn out of it goes on into the next province: what reaches its end leaves the
        # graph.
        self.leaves = np.bincount(self.t_in, minlength=len(graph)) == 0
        self.n = np.zeros(len(graph))
        self.demand = True  # False: no vehicles start or park (tests of the moving part alone)
        # Edges the car layer owns (near the player): the flow does not move their vehicles, and what it sends
        # into them is counted in `arrived` for the car layer to place as cars.
        self.owned = np.zeros(len(graph), bool)
        self.arrived = np.zeros(len(graph))
        self.closures: list[tuple[np.ndarray, datetime]] = []
        classes = np.array([SOURCE_FACTOR[c] for c in CLASSES])[a["g_class"]]
        built = a["g_built"].astype(np.float64) * classes
        self.source_weight = built if built.sum() > 0 else self.length * classes
        self.when: datetime | None = None
        self.configure(config)

    def configure(self, config: TrafficConfig) -> None:
        """Use these demand settings from the next step on."""
        self.config = config
        names = list(self.graph.a.get("g_muni_names", ()))
        muni = self.graph.a.get("g_muni", np.full(len(self.n), -1))
        factor = np.array([config.attract.get(name, 1.0) for name in names] + [1.0])
        self.attract = factor[muni]  # -1 (outside every municipio) takes the trailing 1.0

    def close(self, edges, until: datetime) -> None:
        """Close these edges until `until`: nothing enters them and their vehicles stay where they are."""
        self.closures.append((np.asarray(edges, np.int64), until))

    def is_closed(self, when: datetime) -> np.ndarray:
        """A mask of the edges closed at `when`."""
        self.closures = [(e, until) for e, until in self.closures if until > when]
        mask = np.zeros(len(self.n), bool)
        for edges, _ in self.closures:
            mask[edges] = True
        return mask

    def set_counts(self, edges, counts) -> None:
        """Overwrite the vehicles on these edges (the car layer reports what is near the player)."""
        self.n[np.asarray(edges, np.int64)] = counts

    def shares(self, closed: np.ndarray) -> np.ndarray:
        """The share of each turn in its edge's outflow, with these edges closed.

        A closed edge keeps its share from its own approaches (their drivers do not know yet, and wait), but
        junctions one step further back send only AVOID of the usual share into those blocked approaches.
        """
        E = len(self.n)
        blocked = np.zeros(E, bool)
        if closed.any():
            blocked[np.unique(self.t_in[closed[self.t_out]])] = True
        w = self.t_w * np.where(blocked[self.t_out], AVOID, 1.0)
        return w / np.maximum(np.bincount(self.t_in, w, E), 1e-12)[self.t_in]

    def step(self, dt: float, when: datetime) -> None:
        """Advance dt seconds, at time `when`."""
        E = len(self.n)
        self.when = when
        closed = self.is_closed(when)
        if E == 0:
            return
        key = closed.tobytes() if closed.any() else b""
        if getattr(self, "_shares_key", None) != key:  # shares change only when the closures do
            self._shares, self._shares_key = self.shares(closed), key
        share = self._shares

        # What an edge can send: the vehicles that reach its end this step, never more than it holds.
        demand = np.minimum(self.n * np.minimum(1.0, self.vfree * dt / self.length), self.cap * dt)
        demand[self.owned | closed] = 0.0  # the car layer moves owned edges; a closed one holds still
        supply = np.where(closed, 0.0, np.maximum(self.jam - self.n, 0.0))
        want = demand[self.t_in] * share
        into = np.bincount(self.t_out, want, E)
        ratio = np.minimum(1.0, supply / np.maximum(into, 1e-12))
        # First in, first out: an edge sends at the pace of its most constrained turn. Only the few turns into
        # an edge short of room limit anything, so the minimum is taken over those alone.
        fifo = np.ones(E)
        limited = np.flatnonzero((ratio[self.t_out] < 1.0) & (share > 0))
        np.minimum.at(fifo, self.t_in[limited], ratio[self.t_out[limited]])
        moved = want * fifo[self.t_in]
        inflow = np.bincount(self.t_out, moved, E)
        self.n += inflow - np.bincount(self.t_in, moved, E) - np.where(self.leaves, demand, 0.0)
        self.arrived += np.where(self.owned, inflow, 0.0)

        if self.demand:
            self._demand(dt, when, closed)
        np.clip(self.n, 0.0, self.jam, out=self.n)

    def _demand(self, dt: float, when: datetime, closed: np.ndarray) -> None:
        trip_s = self.config.trip_minutes * 60
        target = sum(moving(self.config, when).values())
        hour = when.hour + when.minute / 60
        weight = self.source_weight * np.where(AFTERNOON[0] <= hour < AFTERNOON[1], self.attract, 1.0)
        weight = np.where(closed | self.owned, 0.0, weight)
        if weight.sum() > 0:
            room = np.maximum(self.jam - self.n, 0.0)
            self.n += np.minimum(target / trip_s * dt * weight / weight.sum(), room)
        pull = np.where(MORNING[0] <= hour < MORNING[1], self.attract, 1.0)
        total = self.n.sum()
        if total > 0:
            pull = pull * total / max(float((self.n * pull).sum()), 1e-12)
        self.n -= np.where(closed | self.owned, 0.0, self.n * pull * dt / trip_s)

    def view(self) -> FlowView:
        """A copy of the state for the car layer and the map: vehicles, speed and queue per edge."""
        fill = np.where(self.jam > 0, self.n / np.maximum(self.jam, 1e-12), 0.0)
        speed = np.maximum(WALKING, self.vfree * (1.0 - fill ** 2))
        queue = np.maximum(0.0, self.n - self.cap * self.length / self.vfree)
        return FlowView(self.n.copy(), speed, queue, self.when, self.arrived.copy())
