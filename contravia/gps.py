"""GPS: a route from anywhere to anywhere on the island, and the guides that follow it.

The provinces' road graphs (contravia/graph.py) are joined into one routing graph for the island (RouteGraph), kept
in route.npz beside index.npz. A node of the search is an edge; an arc is an allowed turn, costed in seconds (spec
docs/superpowers/specs/2026-11-20-gps-design.html).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contravia.driving_help import STRAIGHT
from contravia.graph import CLASSES, Graph

ROUTE_VERSION = 1  # bump when route.npz's arrays change
RIGHT, LEFT = 3.0, 8.0  # seconds a turn past STRAIGHT costs, right and left (across the oncoming traffic)
CONTROL_COST = np.array([0.0, 10.0, 4.0, 2.0])  # seconds by g_control: none, signal, stop, give way
SERVICE = 1.5  # service roads (car parks, lanes) take this much longer than their speed says
UTURN = np.radians(150)  # a turn sharper than this is not an arc, as the pilot never takes one
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
