"""GPS: a route from anywhere to anywhere on the island, and the guides that follow it.

The provinces' road graphs (contravia/graph.py) are joined into one routing graph for the island (RouteGraph), kept
in route.npz beside index.npz. A node of the search is an edge; an arc is an allowed turn, costed in seconds (spec
docs/superpowers/specs/2026-11-20-gps-design.html).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contravia.driving_help import STRAIGHT
from contravia.graph import CLASSES, Graph

ROUTE_VERSION = 1  # bump when route.npz's arrays change
RIGHT, LEFT = 3.0, 8.0  # seconds a turn past STRAIGHT costs, right and left (across the oncoming traffic)
CONTROL_COST = np.array([0.0, 10.0, 4.0, 2.0])  # seconds by g_control: none, signal, stop, give way
SERVICE = 1.5  # service roads (car parks, lanes) take this much longer than their speed says
UTURN = np.radians(150)  # a turn sharper than this is not an arc, as the pilot never takes one
ROAD_NEAR = 300.0  # metres: farther than this from every street, a point has no route
SAME_STREET = 10.0  # metres: edges this much farther than the nearest from a destination are not its street
JOIN = 15.0  # metres: a dead end of one province this close to a start of another is the same junction
THIN = 50.0  # metres: route.npz keeps a polyline point at least this often
ARRAYS = ("offsets", "length", "vfree", "cls", "control", "way", "name", "names", "head0", "head1", "poly",
          "poly_start", "arc_in", "arc_out", "angle", "cost")


def arc_cost(length, vfree, cls_out, control_in, angle) -> np.ndarray:
    """Seconds an arc costs: driving the edge entered (length, vfree, its class cls_out), the control at the end of
    the edge left (control_in) and the turn's signed angle (right positive)."""
    length, vfree, angle = (np.asarray(v, float) for v in (length, vfree, angle))
    drive = length / np.maximum(vfree, 0.1) * np.where(np.asarray(cls_out) == CLASSES.index("service"), SERVICE, 1.0)
    turn = np.where(angle > STRAIGHT, RIGHT, 0.0) + np.where(angle < -STRAIGHT, LEFT, 0.0)
    return drive + turn + CONTROL_COST[np.asarray(control_in, np.int64)]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RouteGraph:
    """The island's routing graph: per island edge its length, speed, class, control, way, name and headings, its
    polyline thinned to a point every THIN metres; per arc (an allowed turn) its edges, signed angle and cost.
    An island edge id is its province's offset plus its id in the province's graph."""

    def __init__(self, **arrays):
        for k in ARRAYS:
            setattr(self, k, arrays[k])
        self._csr = self._tree = None

    @classmethod
    def from_graphs(cls, graphs: list[Graph], join_m: float = JOIN) -> RouteGraph:
        """The routing graph of these provinces' graphs, in order, joined where they share an OSM node or a dead end
        of one lies within join_m of a start of another."""
        sizes = [len(g) for g in graphs]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        parts: dict[str, list] = {k: [] for k in ("length", "vfree", "cls", "control", "way", "name", "head0",
                                                 "head1", "poly", "count", "arc_in", "arc_out", "end", "start",
                                                 "node_end", "node_start")}
        names, name_off = [], 0
        for k, g in enumerate(graphs):
            a, E, off = g.a, sizes[k], offsets[k]
            if not E:
                continue
            e = np.arange(E)
            length = a["g_len"].astype(np.float64)
            x0, z0, h0 = g.points(e, np.zeros(E), np.zeros(E))
            x1, z1, h1 = g.points(e, length, np.zeros(E))
            n = np.maximum(np.ceil(length / THIN).astype(np.int64) + 1, 2)
            rep = np.repeat(e, n)
            f = (np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n)) / (n[rep] - 1)
            px, pz, _ = g.points(rep, length[rep] * f, np.zeros(len(rep)))
            name = a["g_name"] if "g_name" in a else np.full(E, -1)
            table = list(a["g_names"]) if "g_names" in a else []
            for key, v in (("length", length), ("vfree", a["g_vfree"]), ("cls", a["g_class"]),
                           ("control", a["g_control"]), ("way", a["g_way"]),
                           ("name", np.where(name >= 0, name + name_off, -1)), ("head0", h0), ("head1", h1),
                           ("poly", np.stack([px, pz], 1)), ("count", n), ("arc_in", a["g_turn_in"] + off),
                           ("arc_out", a["g_turn_out"] + off), ("end", np.stack([x1, z1], 1)),
                           ("start", np.stack([x0, z0], 1)),
                           ("node_end", a["g_junction_node"][a["g_to"]]),
                           ("node_start", a["g_junction_node"][a["g_from"]])):
                parts[key].append(np.asarray(v))
            names += table
            name_off += len(table)
        cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()}
        prov = np.repeat(np.arange(len(graphs)), sizes)
        arc_in, arc_out = cat["arc_in"].astype(np.int64), cat["arc_out"].astype(np.int64)
        # provinces meet where they share a junction's OSM node: every edge ending there in one province turns into
        # every edge starting there in the other
        if len(prov):
            ends = {}
            for e, (node, p) in enumerate(zip(cat["node_end"].tolist(), prov.tolist())):
                ends.setdefault(node, []).append((e, p))
            extra = [(e, o) for o, (node, q) in enumerate(zip(cat["node_start"].tolist(), prov.tolist()))
                     for e, p in ends.get(node, ()) if p != q]
            if extra:
                arc_in = np.concatenate([arc_in, [x[0] for x in extra]])
                arc_out = np.concatenate([arc_out, [x[1] for x in extra]])
        if join_m > 0 and len(prov):  # dead ends of one province near starts of another
            from scipy.spatial import cKDTree

            no_out = np.setdiff1d(np.arange(len(prov)), arc_in)
            no_in = np.setdiff1d(np.arange(len(prov)), arc_out)
            if len(no_out) and len(no_in):
                tree = cKDTree(cat["start"][no_in])
                pairs = [(int(e), int(no_in[j])) for e, near in zip(no_out, tree.query_ball_point(cat["end"][no_out],
                                                                                                    join_m))
                         for j in near if prov[no_in[j]] != prov[e]]
                if pairs:
                    arc_in = np.concatenate([arc_in, [p[0] for p in pairs]])
                    arc_out = np.concatenate([arc_out, [p[1] for p in pairs]])
        head0, head1 = cat["head0"].astype(np.float32), cat["head1"].astype(np.float32)
        angle = _wrap(head0[arc_out].astype(np.float64) - head1[arc_in])
        ok = np.abs(angle) <= UTURN
        arc_in, arc_out, angle = arc_in[ok], arc_out[ok], angle[ok]
        length, vfree = cat["length"].astype(np.float32), cat["vfree"].astype(np.float32)
        cls_, control = cat["cls"].astype(np.int8), cat["control"].astype(np.int8)
        cost = arc_cost(length[arc_out], vfree[arc_out], cls_[arc_out], control[arc_in], angle)
        count = cat["count"].astype(np.int64)
        return cls(offsets=offsets, length=length, vfree=vfree, cls=cls_, control=control,
                   way=cat["way"].astype(np.int64), name=cat["name"].astype(np.int32),
                   names=np.array(names, dtype=str), head0=head0, head1=head1,
                   poly=cat["poly"].reshape(-1, 2).astype(np.float32),
                   poly_start=np.concatenate([[0], np.cumsum(count)]).astype(np.int64),
                   arc_in=arc_in.astype(np.int32), arc_out=arc_out.astype(np.int32), angle=angle.astype(np.float32),
                   cost=cost.astype(np.float32))

    def save(self, path) -> None:
        np.savez_compressed(path, version=np.array(ROUTE_VERSION), **{k: getattr(self, k) for k in ARRAYS})

    @classmethod
    def load(cls, path) -> RouteGraph:
        with np.load(path) as f:
            return cls(**{k: f[k] for k in ARRAYS})

    @classmethod
    def for_world(cls, world) -> RouteGraph:
        """The world's routing graph: route.npz beside its index when it is current, else built (and saved when the
        world was read from a folder). Reads only the g_* arrays of each province file, never the World's cache."""
        folder = world.folder
        if folder is None or any(p is None for p in world.paths):
            return cls.from_graphs([world.province(k).graph for k in range(len(world.names))])
        path = Path(folder) / "route.npz"
        if path.exists() and path.stat().st_mtime >= max(Path(p).stat().st_mtime for p in world.paths):
            with np.load(path) as f:
                current = int(f["version"]) == ROUTE_VERSION
            if current:
                return cls.load(path)
        graphs = []
        for p in world.paths:
            with np.load(p) as f:
                graphs.append(Graph({k: f[k] for k in f.files if k.startswith("g_")}))
        rg = cls.from_graphs(graphs)
        rg.save(path)
        return rg

    def island(self, k: int, local: int) -> int:
        """The island id of edge `local` of province k."""
        return int(self.offsets[k] + local)

    def local(self, edge: int) -> tuple[int, int]:
        """(province, edge in its graph) of an island edge."""
        k = int(np.searchsorted(self.offsets, edge, side="right") - 1)
        return k, int(edge - self.offsets[k])

    def drive(self, edge, metres) -> np.ndarray:
        """Seconds driving `metres` of edge at its speed, a service road taking SERVICE times longer."""
        edge = np.asarray(edge, np.int64)
        slow = np.where(self.cls[edge] == CLASSES.index("service"), SERVICE, 1.0)
        return np.asarray(metres, float) / np.maximum(self.vfree[edge], 0.1) * slow

    def csr(self):
        """The arcs as a sparse matrix, the cheapest of any repeated (in, out) pair; built once."""
        if self._csr is None:
            from scipy.sparse import csr_matrix

            order = np.lexsort((self.cost, self.arc_out, self.arc_in))
            a, b, c = self.arc_in[order], self.arc_out[order], self.cost[order].astype(np.float64)
            first = np.r_[True, (a[1:] != a[:-1]) | (b[1:] != b[:-1])] if len(a) else np.zeros(0, bool)
            n = len(self.length)
            self._csr = csr_matrix((np.maximum(c[first], 1e-6), (a[first], b[first])), shape=(n, n))
        return self._csr

    def near(self, x: float, z: float, within: float) -> list[tuple[int, float, float]]:
        """(edge, s, distance) of every edge whose polyline passes within `within` metres of (x, z), at the foot of
        the point on it; s is scaled from the thinned polyline to the edge's length."""
        from scipy.spatial import cKDTree

        if self._tree is None:
            self._tree = cKDTree(self.poly)
            self._edge_of = np.repeat(np.arange(len(self.length)), np.diff(self.poly_start))
        hit = self._tree.query_ball_point((x, z), within + THIN)
        if not len(hit):
            return []
        edges = np.unique(self._edge_of[np.asarray(hit)])
        out = []
        for e in edges.tolist():
            pts = self.poly[self.poly_start[e]:self.poly_start[e + 1]].astype(np.float64)
            d = np.diff(pts, axis=0)
            seg = np.maximum(np.hypot(*d.T), 1e-9)
            t = np.clip(((np.array([x, z]) - pts[:-1]) * d).sum(1) / seg ** 2, 0, 1)
            foot = pts[:-1] + d * t[:, None]
            dist = np.hypot(*(foot - (x, z)).T)
            k = int(np.argmin(dist))
            if dist[k] <= within:
                along = (np.concatenate([[0], np.cumsum(seg)])[k] + t[k] * seg[k]) / seg.sum()
                out.append((int(e), float(along * self.length[e]), float(dist[k])))
        return out

    def start_at(self, x: float, z: float, heading: float) -> tuple[int, float] | None:
        """Where a car at (x, z) facing `heading` joins the graph: the nearest edge within ROAD_NEAR running within
        90° of its heading, else the nearest; None with no edge that near."""
        near = self.near(x, z, ROAD_NEAR)
        if not near:
            return None
        e, s, _ = min(near, key=lambda n: n[2])
        along = [n for n in near if np.cos(self._heading_at(n[0], n[1]) - heading) > 0]
        if along:
            e, s, _ = min(along, key=lambda n: n[2])
        return e, s

    def _heading_at(self, e: int, s: float) -> float:
        pts = self.poly[self.poly_start[e]:self.poly_start[e + 1]].astype(np.float64)
        k = min(int(s / max(self.length[e], 1e-9) * (len(pts) - 1)), len(pts) - 2)
        d = pts[k + 1] - pts[k]
        return float(np.arctan2(d[1], d[0]))

    def ends_at(self, x: float, z: float) -> list[tuple[int, float]]:
        """The edges a route to (x, z) may end on: the street nearest the point (both its directions, every edge
        within SAME_STREET of the nearest), when it is within ROAD_NEAR; [] past it."""
        near = self.near(x, z, ROAD_NEAR)
        if not near:
            return []
        reach = min(n[2] for n in near) + SAME_STREET
        return [(e, s) for e, s, d in near if d <= reach]


@dataclass
class Route:
    """A way through the island's edges: from s0 along the first to s1 along the last, its metres and seconds,
    and where it goes (to, and the place's title when it was one)."""

    edges: np.ndarray
    s0: float
    s1: float
    metres: float
    seconds: float
    to: tuple[float, float] | None = None
    title: str | None = None

    def polyline(self, rg: RouteGraph) -> np.ndarray:
        """The route's points in metres, from the thinned polylines, cut at s0 and s1."""
        parts = []
        for k, e in enumerate(self.edges.tolist()):
            pts = rg.poly[rg.poly_start[e]:rg.poly_start[e + 1]].astype(np.float64)
            n = len(pts) - 1
            lo = self.s0 / max(rg.length[e], 1e-9) * n if k == 0 else 0.0
            hi = self.s1 / max(rg.length[e], 1e-9) * n if k == len(self.edges) - 1 else float(n)
            idx = np.arange(np.floor(lo), np.ceil(hi) + 1).astype(int).clip(0, n)
            f = np.clip(np.r_[lo, idx[1:-1], hi], 0, n)
            i0 = np.minimum(np.floor(f).astype(int), n - 1)
            parts.append(pts[i0] + (pts[i0 + 1] - pts[i0]) * (f - i0)[:, None])
        return np.concatenate(parts) if parts else np.zeros((0, 2))


def route(rg: RouteGraph, start: tuple[int, float], ends: list[tuple[int, float]]) -> Route | None:
    """The quickest route from s along edge start[0] to any of the ends (edge, s); None when none is reachable."""
    from scipy.sparse.csgraph import dijkstra

    e0, s0 = start
    base = float(rg.drive(e0, rg.length[e0] - s0))
    d, pred = dijkstra(rg.csr(), indices=e0, return_predecessors=True)
    best = None  # (seconds, end edge, s1, back into e0 by a loop: the edge before it)
    for e, s1 in ends:
        if e == e0 and s1 >= s0:
            cand = (float(rg.drive(e0, s1 - s0)), e, s1, None)
        elif e == e0:  # behind on the same edge: round a loop back into it
            into = rg.arc_out == e0
            if not into.any():
                continue
            k = np.argmin(d[rg.arc_in[into]] + rg.cost[into])
            loop = float(d[rg.arc_in[into]][k] + rg.cost[into][k])
            cand = (base + loop - float(rg.drive(e0, rg.length[e0] - s1)), e, s1, int(rg.arc_in[into][k]))
        else:
            cand = (base + float(d[e]) - float(rg.drive(e, rg.length[e] - s1)), e, s1, None)
        if np.isfinite(cand[0]) and (best is None or cand[0] < best[0]):
            best = cand
    if best is None:
        return None
    seconds, e1, s1, loop = best
    path, k = [e1], (loop if loop is not None else e1)
    if loop is not None:
        path.append(k)
    while k != e0:
        k = int(pred[k])
        if k < 0:
            return None
        path.append(k)
    if loop is not None and path[-1] == e0 and len(path) >= 2 and path[1] == e0:
        path = path[1:]
    edges = np.array(path[::-1], np.int64)
    metres = float(rg.length[edges].sum() - s0 - (rg.length[e1] - s1)) if len(edges) > 1 else float(s1 - s0)
    return Route(edges, float(s0), float(s1), metres, float(seconds))


class Router:
    """Finds routes on a daemon thread, the routing graph built by rg_factory on the first ask. ask() replaces any
    ask not started yet; take() gives the last answer once: a Route, "none" (no road near or no way there), or
    None while nothing new is in."""

    def __init__(self, rg_factory):
        self._factory, self.rg = rg_factory, None
        self._lock = threading.Lock()
        self._pending = self._result = None
        self.busy = False

    def ask(self, start: tuple[float, float, float], to: tuple[float, float], title: str | None = None) -> None:
        with self._lock:
            self._pending = (start, to, title)
            if self.busy:
                return
            self.busy = True
        threading.Thread(target=self._run, name="gps-router", daemon=True).start()

    def take(self):
        with self._lock:
            got, self._result = self._result, None
        return got

    def _run(self) -> None:
        while True:
            with self._lock:
                job, self._pending = self._pending, None
            if job is not None:
                if self.rg is None:
                    self.rg = self._factory()
                got = self._solve(*job)
            with self._lock:
                if job is not None:
                    self._result = got
                if self._pending is None:
                    self.busy = False
                    return

    def _solve(self, start, to, title):
        rg = self.rg
        s = rg.start_at(*start)
        ends = rg.ends_at(*to)
        r = route(rg, s, ends) if s is not None and ends else None
        if r is None:
            return "none"
        r.to, r.title = (float(to[0]), float(to[1])), title
        return r


SLIGHT = np.radians(60)  # past STRAIGHT and under this, a turn is slight
LONG = 1000.0  # metres: a step farther than this says to follow the street the car is on
ORDINALS = ("primera", "segunda", "tercera", "cuarta", "quinta", "sexta", "séptima", "octava")


@dataclass
class Step:
    """What to do at a junction `at` metres from the route's start: kind is "straight", "slight", "turn",
    "roundabout" (exit counts the exits, the one taken included) or "arrive"; name is the street taken."""

    at: float
    kind: str
    angle: float
    name: str | None
    exit: int = 0


def roundabouts(rg: RouteGraph) -> np.ndarray:
    """Whether each edge is part of a roundabout: a way whose edges are all one way and close a loop."""
    first, last = rg.poly[rg.poly_start[:-1]], rg.poly[rg.poly_start[1:] - 1]
    key = lambda p: [tuple(q) for q in np.round(p, 1).tolist()]  # noqa: E731
    starts, ends = key(first), key(last)
    out = np.zeros(len(rg.length), bool)
    order = np.argsort(rg.way, kind="stable")
    bounds = np.flatnonzero(np.r_[True, rg.way[order][1:] != rg.way[order][:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        es = order[lo:hi].tolist()
        if len(es) < 2:
            continue
        s, e = {starts[i] for i in es}, [ends[i] for i in es]
        one_way = len(s) == len(es) and not ({(starts[i], ends[i]) for i in es} & {(ends[i], starts[i]) for i in es})
        if one_way and all(p in s for p in e) and {starts[i] for i in es} == set(e):
            out[es] = True
    return out


def _arc(rg: RouteGraph, a: int, b: int) -> int:
    return int(np.flatnonzero((rg.arc_in == a) & (rg.arc_out == b))[0])


def steps(rg: RouteGraph, route: Route) -> list[Step]:
    """The route's steps: a junction where it turns, goes round a roundabout, or goes straight on into another named
    street where another exit is as straight; any other straight on is not a step. The last step is "arrive"."""
    ring = getattr(rg, "_ring", None)
    if ring is None:
        ring = rg._ring = roundabouts(rg)
    names = lambda e: str(rg.names[rg.name[e]]) if rg.name[e] >= 0 else None  # noqa: E731
    edges = route.edges.tolist()
    out, at, k = [], float(rg.length[edges[0]] - route.s0), 0
    while k < len(edges) - 1:
        a, b = edges[k], edges[k + 1]
        angle = float(rg.angle[_arc(rg, a, b)])
        if ring[b] and not ring[a]:  # into a roundabout: count the exits until the route leaves it
            here, exits, j = at, 0, k + 1
            while j < len(edges) - 1 and ring[edges[j]]:
                outs = rg.arc_out[rg.arc_in == edges[j]]
                exits += int(np.sum(~ring[outs]))
                at += float(rg.length[edges[j]])
                j += 1
            if j < len(edges) and not ring[edges[j]]:
                out.append(Step(here, "roundabout", angle, names(edges[j]), exits))
                k = j
                continue
            at = here
        others = rg.angle[(rg.arc_in == a) & (rg.arc_out != b)]
        fork = bool(np.any(np.abs(others) < STRAIGHT)) and names(b) is not None and names(b) != names(a)
        if abs(angle) >= STRAIGHT or fork:
            kind = "straight" if abs(angle) < STRAIGHT else "slight" if abs(angle) < SLIGHT else "turn"
            out.append(Step(at, kind, angle, names(b)))
        at += float(rg.length[b])
        k += 1
    out.append(Step(float(route.metres), "arrive", 0.0, None))
    return out


def distance_text(m: float) -> str:
    """Metres as the panel writes them: «Ahora» under 30, 10 m steps under 100, 50 m steps under a kilometre, then
    kilometres with one decimal and a comma."""
    if m < 30:
        return "Ahora"
    if m < 100:
        return f"{round(m / 10) * 10:.0f} m"
    if round(m / 50) * 50 < 1000:
        return f"{round(m / 50) * 50:.0f} m"
    return f"{m / 1000:.1f}".replace(".", ",") + " km"


def say(step: Step, metres: float, along: str | None) -> tuple[str, str]:
    """(distance, words) of a step `metres` ahead: «Dobla a la derecha en Galiano»; past LONG, on a named street,
    «Sigue 2,4 km por Malecón»."""
    far = distance_text(metres)
    if metres > LONG and along and step.kind != "arrive":
        return far, f"Sigue {far} por {along}"
    side = "derecha" if step.angle > 0 else "izquierda"
    if step.kind == "arrive":
        return far, "Llegas a tu destino"
    if step.kind == "roundabout":
        nth = ORDINALS[min(step.exit, len(ORDINALS)) - 1] if step.exit > 0 else "primera"
        return far, f"En la rotonda, toma la {nth} salida" + (f" hacia {step.name}" if step.name else "")
    if step.kind == "straight":
        return far, "Sigue recto" + (f" por {step.name}" if step.name else "")
    verb = "Dobla ligeramente a la " if step.kind == "slight" else "Dobla a la "
    return far, verb + side + (f" en {step.name}" if step.name else "")
