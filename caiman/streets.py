"""Streets: a clean drivable surface for the whole city, built once in `caiman prepare` from the road graph.

OSM draws a street as a centreline through nodes, and a divided avenue as two one-way lines. Laying a ribbon
on each line leaves notches, overlaps and slivers at every junction. This module rebuilds the network first
and draws it after:

1. pair_carriageways: the two one-way carriageways of an avenue that run close enough to touch are pulled
   onto their shared centreline and split where they share a node, so they become one street.
2. network: nodes whose junctions would overlap merge into one cluster, centred on the straightest street
   through it so that street crosses without a kink; parallel edges between two clusters collapse; edges
   chain through a node where only two streets of one width meet; each chain is simplified and its corners
   rounded, while straight runs stay straight.
3. trims: each end is cut back past every neighbour's ribbon. An edge too short for its two trims merges
   its clusters, and steps 2 and 3 run again until none is left.
4. pieces: a ribbon per edge between its trims, its width easing into the street it continues, and one
   polygon per junction that meets its ribbons exactly.
5. mesh: one height per junction, ribbons level across that ease into their junctions' heights, the pieces
   cut per cell and triangulated, plus the edges where the road stands clear of the ground, for skirts.

The work runs on a thread pool: this Python has no GIL. Shapely prepares a geometry lazily the first time a
predicate needs it, which is not safe from several threads at once, so every shared tree is built and
prepared up front (built_tree).
"""

import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.strtree import STRtree

from caiman.geo import CELL
from caiman.graph import CLASSES

HULL = 4.0  # metres past the widest half width within which a junction is convex
QS = 4  # segments per quarter circle in round joins
PAIR_SLACK = 3.0  # metres: carriageways closer than their half widths plus this become one avenue
STEP = 2.0  # metres between samples along a rounded corner
RADIUS = 25.0  # metres: the radius a corner of a street is rounded to, when its runs are long enough
LANE = 3.3  # metres per lane of a one-way carriageway
MIN_WIDTH = 7.5  # metres: room for three cars side by side, so every street can be driven
TAPER = 30.0  # metres over which a street's width eases into the width of the street it continues
DP = 1.0  # metres: Douglas-Peucker tolerance before rounding corners
CORNER = 2.0  # metres added to every trim, room for the corner arcs
MERGE_SHORT = 30.0  # an edge shorter than this that cannot fit its trims merges its two junctions
GAP = 25.0  # m²: a hole between a junction and its neighbours smaller than this is absorbed by the junction
SELF_LOOP = 80.0  # metres: a loop shorter than this from a junction back to itself is a U-turn inside it
WALL_GAP = 2.0  # metres of a service road left out on each side of a landmark wall it crosses
ROUNDS = 8  # at most this many rounds of network and trims
LIFT = 0.15  # metres above the ground, as caiman.rules.roads
EASE = 20.0  # metres over which a ribbon's height eases into its junction's
SAMPLE = 8.0  # metres between the boundary points a ribbon is triangulated on; a junction is flat
CLEAR = 0.2  # metres: an edge of the road this far above the ground gets a skirt
MARK = 0.15  # metres: half the width of a painted line
MARK_LIFT = 0.02  # metres above the asphalt
GRADE = 0.08  # the steepest a ramp onto a bridge or down into a tunnel climbs, 8 %
BRIDGE_CLEAR = 6.0  # metres from a street up to the bridge crossing it: 5 m of headroom and the deck
BRIDGE_STEP = 2.0  # metres between the samples of a bridge's profile, so no street under it is missed
TUNNEL_DEPTH = 8.0  # metres of a tunnel under the ground: its headroom, roof and cover
END_FREE = 8.0  # metres at each end of a bridge where the ground street it leaves from lies under it
SEA_DEPTH = 14.0  # metres under sea level where a tunnel runs beneath the sea
DASH, DASH_GAP = 3.0, 6.0  # metres of paint and of gap along a dashed centre line, and of each painted piece
PAINTED = {c for k in ("motorway", "trunk", "primary", "secondary", "tertiary") for c in (k, f"{k}_link")}

# What caiman.structures adds per cell: triangles and their paint, walls with heights, cuts in the ground, roofs.
STRUCTURE_KINDS = ("struct_tris", "struct_paint", "struct_walls", "street_cuts", "street_roofs")

_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(os.cpu_count())
    return _POOL


def unit(v):
    """v scaled to length 1 along its last axis."""
    return v / np.maximum(np.hypot(v[..., 0], v[..., 1]), 1e-9)[..., None]


def length(p):
    """The length of polyline p."""
    d = p[1:] - p[:-1]
    return float(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2).sum())


def arclen(p):
    """The distance along polyline p to each of its points."""
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


def resample(p, n):
    """n points evenly spaced along polyline p, ends included."""
    s = arclen(p)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def densify(p, step):
    """Polyline p resampled about every `step` metres."""
    s = arclen(p)
    return resample(p, max(2, math.ceil(s[-1] / step) + 1)) if s[-1] > 0 else p


def tangents(p):
    """The unit direction of polyline p at each of its points."""
    d = np.vstack([p[1] - p[0], p[2:] - p[:-2], p[-1] - p[-2]]) if len(p) > 2 else np.repeat([p[-1] - p[0]], len(p), 0)
    return unit(d)


def cut(p, a, b):
    """The polyline between arclength a and arclength b (b measured from the start)."""
    s = arclen(p)
    def at(t):
        return np.array([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])
    inner = p[(s > a + 1e-6) & (s < b - 1e-6)]
    return np.vstack([at(a), inner, at(b)])


def fillet(p, step=STEP, radius=RADIUS):
    """Straight runs stay straight; each corner of p becomes a quadratic Bezier tangent to both runs, as wide as a
    circle of `radius` would need but never past half of either run, sampled about every `step` metres."""
    if len(p) < 3:
        return p
    d = np.diff(p, axis=0)
    L = np.hypot(*d.T)
    u = d / np.maximum(L, 1e-9)[:, None]
    out = [p[:1]]
    for i in range(1, len(p) - 1):
        cos = float(np.clip(u[i - 1] @ u[i], -1, 1))
        turn = math.acos(cos)
        t = min(radius * math.tan(turn / 2), 0.5 * L[i - 1], 0.5 * L[i]) if turn > 1e-3 else 0.0
        if t < 0.05:
            out.append(p[i:i + 1])
            continue
        a, c = p[i] - u[i - 1] * t, p[i] + u[i] * t
        n = max(2, math.ceil(2 * t / step))
        k = np.linspace(0, 1, n + 1)[:, None]
        out.append((1 - k) ** 2 * a + 2 * (1 - k) * k * p[i] + k ** 2 * c)
    out.append(p[-1:])
    q = np.vstack(out)
    keep = np.concatenate([[True], np.hypot(*np.diff(q, axis=0).T) > 1e-6])
    return q[keep]


def offset(p, h):
    """Left and right kerbs of a ribbon of half width h along p; the end normals are the end segments' own."""
    d = unit(np.diff(p, axis=0))
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    nv = np.vstack([n[:1], n[:-1] + n[1:], n[-1:]])
    nv = unit(nv)
    cos = np.ones(len(p))
    if len(p) > 2:
        cos[1:-1] = np.sum(nv[1:-1] * n[1:], axis=1)
    nv = nv / np.maximum(cos, 0.5)[:, None]
    h = np.broadcast_to(np.asarray(h, float), (len(p),))[:, None]
    return p + nv * h, p - nv * h



class Edge:
    """A street between clusters f and t along polyline p, w metres wide, of OSM class cls."""

    __slots__ = ("cls", "f", "layer", "oneway", "p", "paired", "t", "w")

    def __init__(self, f, t, p, w, cls, oneway, paired=False, layer=0):
        self.f, self.t, self.p, self.w, self.cls, self.oneway, self.paired = f, t, p, w, cls, oneway, paired
        self.layer = layer  # above 0 on a bridge, below 0 in a tunnel (caiman.graph.layer)


def load(a, widths):
    """The junctions and the undirected edges of the road graph in the arrays a, each as wide as its class,
    or as its lanes when it is one way."""
    J = a["g_junction"].astype(float)
    starts, poly = a["g_poly_start"], a["g_poly"].astype(float)
    F, T, W = a["g_from"].tolist(), a["g_to"].tolist(), a["g_way"].tolist()
    layers = a["g_layer"].tolist() if "g_layer" in a else [0] * len(F)
    directed = set(zip(F, T, W))
    seen, edges = set(), []
    for e in range(len(F)):
        key = (min(F[e], T[e]), max(F[e], T[e]), W[e])
        if key in seen:
            continue
        seen.add(key)
        cls = CLASSES[a["g_class"][e]]
        p = poly[starts[e]:starts[e + 1]].copy()
        p[0], p[-1] = J[F[e]], J[T[e]]
        oneway = (T[e], F[e], W[e]) not in directed
        w = widths[cls]
        if oneway:  # a one-way carriageway is as wide as its lanes, not as a whole street of its class
            w = min(w, max(1, int(a["g_lanes"][e])) * LANE + 1.0)
        w = max(w, MIN_WIDTH)
        edges.append(Edge(F[e], T[e], p, w, cls, oneway, layer=layers[e]))
    return J, edges


def pair_carriageways(J, edges, stops=frozenset()):
    """Pull the two one-way carriageways of an avenue onto their centreline where their ribbons would touch.

    Each sample of a one-way edge looks for the nearest anti-parallel one-way edge beside it (not ahead of it)
    within the two half widths plus PAIR_SLACK, and moves half way towards it; the partner does the same, so
    both land on the centreline. The pair's width becomes the separation plus both half widths. Then every
    endpoint of a paired edge that now lies on another paired edge splits it there, so both carriageways share
    their nodes and collapse into one edge later."""
    # a carriageway that reaches a bridge or tunnel (stops) stays apart: each goes on into its own one
    one = [k for k, e in enumerate(edges) if e.oneway and e.f not in stops and e.t not in stops]
    lines = [LineString(edges[k].p) for k in one]
    tree = built_tree(lines)
    moves = defaultdict(list)
    newp = {}

    def search(i):
        k = one[i]
        e = edges[k]
        p = densify(e.p, 3.0)
        tg = tangents(p)
        pts = shapely.points(p)
        best = np.full(len(p), np.inf)
        bq = p.copy()
        bw = np.zeros(len(p))
        for c in tree.query(lines[i].buffer(e.w / 2 + 7 + PAIR_SLACK)):
            k2 = one[c]
            e2 = edges[k2]
            if k2 == k or (e.cls == "service") != (e2.cls == "service"):
                continue
            L2 = lines[c]
            s = shapely.line_locate_point(L2, pts)
            q = shapely.get_coordinates(shapely.line_interpolate_point(L2, s))
            ta = shapely.get_coordinates(shapely.line_interpolate_point(L2, s + 1.0))
            tb = shapely.get_coordinates(shapely.line_interpolate_point(L2, s - 1.0))
            t2 = unit(ta - tb)
            d = np.hypot(*(q - p).T)
            side = np.abs(np.sum(unit(q - p) * tg, axis=1))
            ok = ((d < (e.w + e2.w) / 2 + PAIR_SLACK) & (d > 0.3) & (np.sum(tg * t2, axis=1) < -0.7)
                  & (side < 0.5) & (d < best))
            best = np.where(ok, d, best)
            bq[ok] = q[ok]
            bw[ok] = d[ok] + (e.w + e2.w) / 2
        hit = np.isfinite(best)
        if hit.mean() < 0.5:
            return None
        disp = np.where(hit[:, None], (bq - p) / 2, 0.0)
        pad = np.vstack([np.repeat(disp[:1], 3, 0), disp, np.repeat(disp[-1:], 3, 0)])
        k5 = np.ones(7) / 7
        disp = np.stack([np.convolve(pad[:, 0], k5, "valid"), np.convolve(pad[:, 1], k5, "valid")], axis=1)
        return k, p + disp, float(np.median(bw[hit])), disp[0], disp[-1]

    for r in pmap(search, range(len(one)), chunk=16):
        if r is None:
            continue
        k, q, w, d0, d1 = r
        e = edges[k]
        newp[k] = q
        e.w = w
        e.paired = True
        moves[e.f].append(d0)
        moves[e.t].append(d1)
    J = J.copy()
    for n, ds in moves.items():
        J[n] += np.mean(ds, axis=0)
    for k, e in enumerate(edges):
        if k in newp:
            e.p = newp[k]
        e.p[0], e.p[-1] = J[e.f], J[e.t]
    # split paired edges where another paired edge's endpoint now lies on them
    paired = [k for k, e in enumerate(edges) if e.paired]
    lines = [LineString(edges[k].p) for k in paired]
    tree = built_tree(lines)
    splits = defaultdict(list)
    for k in paired:
        for n in (edges[k].f, edges[k].t):
            pt = shapely.Point(J[n])
            for c in tree.query(pt.buffer(2.5)):
                k2 = paired[c]
                e2 = edges[k2]
                if k2 == k or n in (e2.f, e2.t):
                    continue
                L2 = lines[c]
                if L2.distance(pt) > 2.5:
                    continue
                s = L2.project(pt)
                if 1.5 < s < L2.length - 1.5:
                    splits[k2].append((s, n))
    out = []
    for k, e in enumerate(edges):
        if k not in splits:
            out.append(e)
            continue
        cuts = sorted(set(splits[k]))
        p, s0, f = e.p, 0.0, e.f
        for s, n in cuts + [(arclen(e.p)[-1], e.t)]:
            if s - s0 < 0.5 and n != e.t:
                continue
            q = cut(p, s0, s)
            q[0], q[-1] = J[f], J[n]
            out.append(Edge(f, n, q, e.w, e.cls, True, True))
            s0, f = s, n
    return J, out, len(splits)


def network(J, edges, forced, cache, stops=frozenset()):
    """Cluster, collapse, chain and smooth: the clean edges between junction centres. No chain runs through a
    node in stops: a street ends where a bridge or tunnel leaves it."""
    N = len(J)
    parent = list(range(N))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    maxw = np.zeros(N)
    for e in edges:
        maxw[e.f] = max(maxw[e.f], e.w)
        maxw[e.t] = max(maxw[e.t], e.w)
    for e in edges:
        if length(e.p) < 0.5 * (maxw[e.f] + maxw[e.t]):
            parent[root(e.f)] = root(e.t)
    for a, b in forced:
        parent[root(a)] = root(b)
    pn = sorted({n for e in edges if e.paired for n in (e.f, e.t)})
    if pn:
        pts = shapely.points(J[pn])
        for a, b in zip(*built_tree(pts).query(pts, predicate="dwithin", distance=4.0)):
            if a != b:
                parent[root(pn[a])] = root(pn[b])
    members = defaultdict(list)
    for i in range(N):
        members[root(i)].append(i)
    centre = {r: J[m].mean(axis=0) for r, m in members.items()}
    # a merged cluster's centre sits on its straightest through pair, so that street crosses without a kink
    out = defaultdict(list)  # cluster -> [(point 10 m out, outward direction, width)]
    for e in edges:
        f, t = root(e.f), root(e.t)
        if f == t:
            continue
        for r, q in ((f, e.p), (t, e.p[::-1])):
            if len(members[r]) > 1:
                s_ = arclen(q)
                k = min(int(np.searchsorted(s_, 10.0)), len(q) - 1)
                out[r].append((q[k], unit(q[k] - q[0]), e.w))
    for r, ends_ in out.items():
        best = None
        for i in range(len(ends_)):
            for j in range(i + 1, len(ends_)):
                cos = float(ends_[i][1] @ ends_[j][1])
                score = cos - 0.01 * (ends_[i][2] + ends_[j][2])
                if cos < -0.8 and (best is None or score < best[0]):
                    best = (score, ends_[i][0], ends_[j][0])
        if best:
            a, b = best[1], best[2]
            d = b - a
            t = float(np.clip((centre[r] - a) @ d / max(d @ d, 1e-9), 0, 1))
            centre[r] = a + d * t
    # re-end every edge on its clusters' centres; drop short self-loops
    es = []
    for e in edges:
        f, t = root(e.f), root(e.t)
        p = e.p.copy()
        p[0], p[-1] = centre[f], centre[t]
        if f == t and (length(p) < SELF_LOOP or np.hypot(*(p - centre[f]).T).max() < 40.0):
            continue
        es.append(Edge(f, t, p, e.w, e.cls, e.oneway, e.paired))
    # collapse parallel edges between the same two clusters
    groups = defaultdict(list)
    for e in es:
        groups[(min(e.f, e.t), max(e.f, e.t))].append(e)
    es = []
    for (f, t), g in groups.items():
        kept = []
        for e in g:
            q = e.p if e.f == f else e.p[::-1]
            Lq = length(q)
            for k in kept:
                Lk = length(k.p)
                if f != t and max(Lq, Lk) < 1.6 * min(Lq, Lk) + 5:
                    rq, rk = resample(q, 24), resample(k.p, 24)
                    if np.mean(np.hypot(*(rq - rk).T)) < 0.5 * max(k.w, e.w) + 2:
                        k.p = (rq + rk) / 2
                        k.p[0], k.p[-1] = centre[f], centre[t]
                        k.w = max(k.w, e.w)
                        break
            else:
                kept.append(Edge(f, t, q, e.w, e.cls, e.oneway, e.paired))
        es += kept
    # chain through clusters where exactly two edges of one width meet
    at = defaultdict(list)
    for k, e in enumerate(es):
        at[e.f].append(k)
        at[e.t].append(k)
    stopped = {root(n) for n in stops}

    def through(node):
        if node in stopped:
            return False
        ks = at[node]
        return len(ks) == 2 and ks[0] != ks[1] and abs(es[ks[0]].w - es[ks[1]].w) < 0.5
    used, chains = set(), []
    for k0 in range(len(es)):
        if k0 in used:
            continue
        used.add(k0)
        e0 = es[k0]
        pts, a_end, b_end = [e0.p], e0.f, e0.t
        for forward in (True, False):
            node, k = (b_end if forward else a_end), k0
            while through(node):
                k = next(x for x in at[node] if x != k)
                if k in used:
                    break
                used.add(k)
                e = es[k]
                q = e.p if e.f == node else e.p[::-1]
                nxt = e.t if e.f == node else e.f
                if forward:
                    pts.append(q[1:])
                    b_end = nxt
                else:
                    pts.insert(0, q[::-1][1:][::-1])
                    a_end = nxt
                node = nxt
        chains.append(Edge(a_end, b_end, np.vstack(pts), e0.w, e0.cls, e0.oneway, e0.paired))

    def smooth(p):
        key = p.tobytes()
        if key not in cache:
            q = np.asarray(LineString(p).simplify(DP, preserve_topology=False).coords) if len(p) > 2 else p
            cache[key] = fillet(q)
        return cache[key]

    for e, p in zip(chains, pmap(smooth, [e.p for e in chains], chunk=128)):
        e.p = p
    return chains, centre

def built_tree(geoms):
    """An STRtree built and prepared now: neither is safe to do lazily from several threads at once."""
    t = STRtree(geoms)
    t.query(shapely.Point(0.0, 0.0))
    shapely.prepare(t.geometries)  # a predicate query prepares them lazily, in place: racy across threads
    return t


def pmap(fn, items, chunk=32):
    """fn over items on the thread pool (this Python has no GIL), results in order."""
    items = list(items)
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    out = []
    for r in _pool().map(lambda ch: [fn(x) for x in ch], chunks):
        out += r
    return out


def end_widths(chains, ends):
    """The width each end eases into at its junction: the mean of its own and that of the street it continues
    straight into, if any."""
    dirs = {}
    for es in ends.values():
        for c, which in es:
            q = chains[c].p if which == 0 else chains[c].p[::-1]
            s = arclen(q)
            k = min(int(np.searchsorted(s, 10.0)), len(q) - 1)
            dirs[(c, which)] = unit(q[k] - q[0])
    ew = {}
    for es in ends.values():
        for x in es:
            w = chains[x[0]].w
            cos, y = min(((float(dirs[x] @ dirs[y]), y) for y in es if y != x), default=(1.0, None))
            ew[x] = (w + chains[y[0]].w) / 2 if cos < -0.8 else w
    return ew


def trims(chains, cache):
    """How far each end is cut back on its smooth curve, past every neighbour's ribbon, and the pairs of
    clusters whose edge is too short for its two trims."""
    ends = defaultdict(list)  # node -> [(chain, which end)]
    for c, e in enumerate(chains):
        ends[e.f].append((c, 0))
        ends[e.t].append((c, 1))
    lens = [arclen(e.p)[-1] for e in chains]
    ew = end_widths(chains, ends)

    def one(es):
        key = tuple((chains[c].p.tobytes(), chains[c].w, ew[(c, which)], which, chains[c].f == chains[c].t) for c, which in es)
        if key in cache:
            return [(es[k], t) for k, t in cache[key]]
        out = compute(es)
        index = {x: k for k, x in enumerate(es)}
        cache[key] = [(index[x], t) for x, t in out]
        return out

    def compute(es):
        geo = []
        for c, which in es:
            e = chains[c]
            q = e.p if which == 0 else e.p[::-1]
            reach = lens[c] * (0.95 if e.f != e.t else 0.3)
            line = LineString(cut(q, 0, reach))
            geo.append((line, line.buffer(max(e.w, ew[(c, which)]) / 2, cap_style="flat")))
        out = []
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                if not geo[i][1].intersects(geo[j][1]):
                    continue
                inter = geo[i][1].intersection(geo[j][1])
                if inter.is_empty or inter.area < 1e-3:
                    continue
                pts = shapely.points(shapely.get_coordinates(inter))
                for k in (i, j):
                    out.append((es[k], float(np.max(shapely.line_locate_point(geo[k][0], pts))) + CORNER))
        return out

    trim = defaultdict(float)
    for res in pmap(one, [es for es in ends.values() if len(es) >= 2], chunk=16):
        for key, t in res:
            trim[key] = max(trim[key], t)
    merge = [(e.f, e.t) for c, e in enumerate(chains)
             if trim[(c, 0)] + trim[(c, 1)] > lens[c] - 1.0 and lens[c] < MERGE_SHORT and e.f != e.t]
    return ends, lens, trim, merge, ew


def pieces(chains, centre, ends, lens, trim, ew):
    """Ribbons between the trims and one polygon per junction that meets them exactly: the ribbons, the chain
    each belongs to, its centreline, the junction polygons and the cluster each belongs to."""
    def ribbon(c):
        e = chains[c]
        a, b = trim[(c, 0)], trim[(c, 1)]
        L = lens[c]
        if a + b > L - 1.0:
            s = (L - 1.0) / max(a + b, 1e-9)
            a, b = a * s, b * s
            if L < 1.5:
                return None
        q = cut(e.p, a, L - b)
        if len(q) < 2 or arclen(q)[-1] < 0.2:
            return None
        if len(q) > 3:
            q = np.asarray(LineString(q).simplify(0.03, preserve_topology=False).coords)
        w0, w1 = ew.get((c, 0), e.w), ew.get((c, 1), e.w)
        if abs(w0 - e.w) > 0.1 or abs(w1 - e.w) > 0.1:  # eases into its neighbours' widths
            q = densify(q, STEP)
            sq = a + arclen(q)
            def ease(x):
                x = np.clip(x, 0, 1)
                return 1 - x * x * (3 - 2 * x)
            w = (e.w + (w0 - e.w) * ease(sq / (TAPER + 3 * abs(w0 - e.w)))
                 + (w1 - e.w) * ease((L - sq) / (TAPER + 3 * abs(w1 - e.w))))
            left, right = offset(q, w / 2)
            # One piece per segment and a disc at each bend, joined: a single outline offset from a curve folds
            # over itself on the inside of a tight bend when the street is wide.
            segs = [LineString(q[i:i + 2]).buffer((w[i] + w[i + 1]) / 4, cap_style="flat") for i in range(len(q) - 1)]
            bends = [shapely.Point(q[i]).buffer(w[i] / 2, QS) for i in range(1, len(q) - 1)]
            rib = largest(shapely.union_all(segs + bends))
            h0, h1 = w[0] / 2, w[-1] / 2
        else:
            left, right = offset(q, e.w / 2)
            rib = largest(LineString(q).buffer(e.w / 2, quad_segs=QS, cap_style="flat", join_style="round"))
            h0 = h1 = e.w / 2
        if rib is None:
            return None
        u0, u1 = unit(q[1] - q[0]), unit(q[-2] - q[-1])
        # outward from the junction: at the start the ribbon runs along u0, at the end back along u1
        return (c, rib, q, (q[0], u0, h0, right[0], left[0], stub(e.p, a, q[0], u0)),
                (q[-1], u1, h1, left[-1], right[-1], stub(e.p[::-1], b, q[-1], u1)))

    ribbons, owner, axes, cs = [], [], [], {}  # cs: (chain, which) -> (point, outward tangent, half width, right, left, stub)
    for r in pmap(ribbon, range(len(chains))):
        if r is None:
            continue
        c, rib, q, c0, c1 = r
        ribbons.append(rib)
        owner.append(c)
        axes.append(q)
        cs[(c, 0)], cs[(c, 1)] = c0, c1
    rtree = built_tree(ribbons)

    def junction(item):
        node, es = item
        es = [x for x in es if x in cs]
        if len(es) < 2:
            return None
        c0 = centre[node]
        es.sort(key=lambda x: math.atan2(*(cs[x][0] - c0)[::-1]))
        parts = [shapely.Point(c0).buffer(0.9 * min(cs[x][2] for x in es), QS)]
        stubs = []
        for i, x in enumerate(es):
            P, u, h, _, Lk, sl = cs[x]
            stubs.append(sl.buffer(h, quad_segs=QS, cap_style="flat", join_style="round"))
            parts.append(stubs[-1])
            P2, u2, h2, R2, _, _ = cs[es[(i + 1) % len(es)]]
            gap = (math.atan2(*(P2 - c0)[::-1]) - math.atan2(*(P - c0)[::-1])) % (2 * math.pi)
            m = np.array([-u, u2]).T
            arc = None
            if abs(np.linalg.det(m)) > 0.15:
                s_, r_ = np.linalg.solve(m, R2 - Lk)
                span = np.hypot(*(R2 - Lk))
                if 0 < s_ < 3 * span + 5 and 0 < r_ < 3 * span + 5:
                    C = Lk - s_ * u
                    arc = [(1 - t) ** 2 * Lk + 2 * (1 - t) * t * C + t * t * R2 for t in np.linspace(0, 1, 9)]
            if arc is None and gap < math.pi and len(es) > 2:
                arc = [Lk, R2]
            same = x[0] == es[(i + 1) % len(es)][0]  # the two ends of one loop
            near_centre = max(np.hypot(*(Lk - c0)), np.hypot(*(R2 - c0))) < 4 * (h + h2) + 10
            if arc is not None and not same and near_centre and np.hypot(*(R2 - Lk)) < 3 * (h + h2):
                fill = Polygon([c0] + arc)
                parts.append(fill if fill.is_valid else shapely.make_valid(fill))
        # near the centre the junction is convex: no notch between two streets that a car would have to steer around
        r = max(cs[x][2] for x in es) + HULL
        parts.append(shapely.intersection(shapely.union_all(stubs), shapely.Point(c0).buffer(r, QS)).convex_hull)
        poly = shapely.union_all(parts).buffer(1.5, quad_segs=QS, join_style="round").buffer(-1.5, quad_segs=QS, join_style="round")
        near = [ribbons[k] for k in rtree.query(poly)]
        for i, x in enumerate(es):  # nothing past an end's cross-section that no other stub covers
            P, u, h, _, _, _ = cs[x]
            n = np.array([-u[1], u[0]])
            beyond = Polygon([P - n * (h + 1.5), P + n * (h + 1.5), P + n * (h + 1.5) + 4 * u, P - n * (h + 1.5) + 4 * u])
            others = [st for j, st in enumerate(stubs) if j != i and st.intersects(beyond)]
            near.append(beyond.difference(shapely.union_all(others)) if others else beyond)
        cutter = shapely.union_all(near)
        poly = largest(poly.difference(cutter))
        if poly is None:
            return None
        if poly.interiors:
            poly = largest(Polygon(poly.exterior).difference(cutter))
        return (node, poly) if poly is not None else None

    rings, rnode = [], []
    items = list(ends.items())
    for r in pmap(junction, items, chunk=16):
        if r is not None:
            rnode.append(r[0])
            rings.append(r[1])
    # where two junctions still overlap, the larger keeps the shared part
    jt = built_tree(rings)
    order = sorted(range(len(rings)), key=lambda k: -rings[k].area)
    done = set()
    for k in order:
        prev = [rings[j] for j in jt.query(rings[k], predicate="intersects") if j in done]
        if prev:
            rings[k] = largest(rings[k].difference(shapely.union_all(prev))) or rings[k]
        done.add(k)
    ribbons = pmap(snap, ribbons, chunk=256)
    rings = pmap(lambda r: snap(r.simplify(0.02)), rings, chunk=256)
    # a junction absorbs the small holes left between it and its neighbours
    rt, jt = built_tree(ribbons), built_tree(rings)
    snapshot = list(rings)

    def absorb(k):
        ring = snapshot[k]
        neigh = ([ribbons[j] for j in rt.query(ring, predicate="dwithin", distance=25.0)]
                 + [snapshot[j] for j in jt.query(ring, predicate="dwithin", distance=25.0) if j != k])
        u = shapely.union_all(neigh + [ring], grid_size=0.01)
        holes = [Polygon(h) for part in getattr(u, "geoms", [u]) if isinstance(part, Polygon) for h in part.interiors]
        return [h for h in holes if h.area < GAP and h.distance(ring) < 0.05]

    taken = set()
    for k, add in enumerate(pmap(absorb, range(len(rings)), chunk=64)):
        add = [h for h in add if (key := tuple(np.round(h.centroid.coords[0], 2))) not in taken and not taken.add(key)]
        if add:
            rings[k] = largest(shapely.union_all([rings[k]] + add)) or rings[k]
    return ribbons, owner, axes, rings, rnode


def stub(p, t, P, u):
    """The centreline from the junction centre to the cut at arclength t, ending exactly at P along u."""
    q = cut(p, 0, max(t, 0.05))
    s = arclen(q)
    q = q[s < s[-1] - 0.3] if len(q) > 2 else q[:1]
    if len(q) == 0:
        q = p[:1]
    return LineString(np.vstack([q, P - 0.2 * u, P + 0.5 * u]))


def snap(g):
    """g on a 1 cm grid, unless snapping would split it."""
    s = shapely.set_precision(g, 0.01)
    return s if isinstance(s, Polygon) and not s.is_empty else g


def largest(g):
    """The largest polygon in g, or None when it holds none."""
    parts = [x for x in getattr(g, "geoms", [g]) if isinstance(x, Polygon) and not x.is_empty]
    return max(parts, key=lambda x: x.area) if parts else None


def build(a, widths, walls=()):
    """The street pieces of the road graph in the arrays a: (ribbons, owner, axes, rings, rnode, chains).

    walls are rings of landmark walls: where a service road crosses one, it is a path through the landmark's
    grounds, and the stretch within WALL_GAP of the wall is left out, as caiman.landmarks leaves out the
    old ribbons that cross it. The rest of the road stays, ending at a new node.

    Each level is built on its own, the ground first: a bridge does not meet the street it passes over, so
    neither joins, trims nor pairs with the other. Where a bridge or tunnel leaves the ground, its end is moved
    onto the end of the ground street there, so the two pieces meet. Every chain keeps its layer; a junction of
    a bridge or tunnel is named (layer, cluster) in rnode, a ground junction by its cluster alone."""
    J, edges = load(a, widths)
    if walls:
        J, edges = _cut_at_walls(J, edges, walls)
    levels = defaultdict(list)
    for e in edges:
        levels[e.layer].append(e)
    ground_nodes = {n for e in levels.get(0, ()) for n in (e.f, e.t)}
    portals = frozenset(n for L, es in levels.items() if L for e in es for n in (e.f, e.t)) & ground_nodes
    out = [[], [], [], [], [], []]
    ends = np.zeros((0, 2))
    for layer in sorted(levels, key=lambda x: (x != 0, abs(x), x)):
        Jl = J if layer == 0 else _onto_ground(J, levels[layer], ground_nodes, ends)
        ribbons, owner, axes, rings, rnode, chains = _level_pieces(Jl, levels[layer], portals if layer == 0 else frozenset())
        for c in chains:
            c.layer = layer
        if layer == 0 and chains:
            ends = np.array([p for c in chains for p in (c.p[0], c.p[-1])])
        n = len(out[5])
        out[0] += list(ribbons)
        out[1] += [o + n for o in owner]
        out[2] += list(axes)
        out[3] += list(rings)
        out[4] += [r if layer == 0 else (layer, r) for r in rnode]
        out[5] += chains
    return tuple(out)


def _onto_ground(J, edges, ground_nodes, ends):
    """J with each node where these edges meet the ground moved onto the nearest end of a ground street
    within 20 m, where the ground's own clusters put it."""
    J = J.copy()
    if not len(ends):
        return J
    for n in {n for e in edges for n in (e.f, e.t)} & ground_nodes:
        d = np.hypot(*(ends - J[n]).T)
        k = int(np.argmin(d))
        if d[k] < 20.0:
            J[n] = ends[k]
    for e in edges:
        e.p = e.p.copy()
        e.p[0], e.p[-1] = J[e.f], J[e.t]
    return J


def _level_pieces(J, edges, stops=frozenset()):
    """The pieces of one level's edges: pair, cluster, trim and cut them as build describes."""
    J, edges, _ = pair_carriageways(J, edges, stops)
    cache, forced = {}, []
    for _ in range(ROUNDS):
        chains, centre = network(J, edges, forced, cache, stops)
        ends, lens, trim, merge, ew = trims(chains, cache)
        if not merge:
            break
        forced += merge
    ribbons, owner, axes, rings, rnode = pieces(chains, centre, ends, lens, trim, ew)
    return ribbons, owner, axes, rings, rnode, chains


def _cut_at_walls(J, edges, walls):
    """Service edges with the stretches near a landmark wall removed; each remaining stretch keeps the
    original node at an end it still reaches and gets a new node at an end it does not."""
    fence = shapely.union_all([Polygon(w).exterior for w in walls if len(w) >= 3])
    zone = fence.buffer(WALL_GAP)
    J, out = list(J), []
    for e in edges:
        line = LineString(e.p)
        if e.cls != "service" or not line.intersects(zone):
            out.append(e)
            continue
        rest = line.difference(zone)
        for part in getattr(rest, "geoms", [rest]):
            if part.is_empty or part.length < 2 * WALL_GAP:
                continue
            q = np.asarray(part.coords)
            ends = []
            for pt, node in ((q[0], e.f), (q[-1], e.t)):
                if np.hypot(*(pt - J[node])) > 0.01:
                    J.append(pt)
                    node = len(J) - 1
                ends.append(node)
            out.append(Edge(ends[0], ends[1], q, e.w, e.cls, e.oneway, e.paired, e.layer))
    return np.array(J), out


def _level(terrain, p, half):
    """The road's height at centre points p (N, 2) with half widths half (N,): the highest ground under the
    left kerb, the centre and the right kerb, plus LIFT, from the tangent of the polyline p."""
    side = np.stack([-tangents(p)[:, 1], tangents(p)[:, 0]], axis=1) * np.asarray(half)[..., None]
    h = [terrain.height(q[:, 0], q[:, 1]) for q in (p - side, p, p + side)]
    return np.maximum.reduce(h) + LIFT


def _ease(x):
    """1 at x = 0 down to 0 at x >= 1, smoothly."""
    x = np.clip(x, 0, 1)
    return 1 - x * x * (3 - 2 * x)


def heights(ribbons, owner, axes, rings, rnode, chains, terrain):
    """One height per junction and a height profile per ribbon.

    A junction is flat at the highest ground under it plus LIFT, and never below the ribbons that meet it.
    A ribbon is level across: at each point of its centreline, the highest ground under its kerbs and centre
    plus LIFT, raised near each end so it meets its junction's height exactly and eases away from it over
    EASE metres (half the ribbon when it is shorter). Bridges and tunnels then take their own profiles
    (structure_profile), and the ground streets they leave from ramp up or down to meet them (_ramp_to).
    Returns (junction heights, [(arclengths, heights)])."""
    layer = [chains[owner[k]].layer for k in range(len(axes))]

    def junction(poly):
        pts = shapely.get_coordinates(poly.segmentize(SAMPLE))
        x0, z0, x1, z1 = poly.bounds
        gx, gz = np.meshgrid(np.arange(x0, x1, SAMPLE), np.arange(z0, z1, SAMPLE))
        grid = np.stack([gx.ravel(), gz.ravel()], axis=1)
        pts = np.vstack([pts, grid[shapely.contains_xy(poly, grid[:, 0], grid[:, 1])]])
        return float(terrain.height(pts[:, 0], pts[:, 1]).max()) + LIFT

    hj = dict(zip(rnode, pmap(junction, rings, chunk=64)))

    def base(k):
        q = densify(axes[k], SAMPLE)
        return arclen(q), _level(terrain, q, np.full(len(q), chains[owner[k]].w / 2))

    profiles = pmap(base, range(len(axes)), chunk=64)
    for k, (s, h) in enumerate(profiles):
        e = chains[owner[k]]
        if layer[k]:
            continue
        for node, y in ((e.f, h[0]), (e.t, h[-1])):
            if node in hj:
                hj[node] = max(hj[node], float(y))
    out = []
    for k, (s, h) in enumerate(profiles):
        e = chains[owner[k]]
        span = min(EASE, s[-1] / 2) or 1.0
        h = h.copy()
        if not layer[k]:
            if e.f in hj:
                h += (hj[e.f] - h[0]) * _ease(s / span)
            if e.t in hj:
                h += (hj[e.t] - h[-1]) * _ease((s[-1] - s) / span)
        out.append((s, h))
    if not any(layer):
        return hj, out
    ground = [k for k in range(len(axes)) if not layer[k]]
    gring = [i for i, n in enumerate(rnode) if not isinstance(n, tuple)]
    near = _Ends(ground, axes, [chains[owner[k]].w for k in ground], [rings[i] for i in gring],
                 [rnode[i] for i in gring])
    for L in sorted({x for x in layer if x}, key=lambda x: (abs(x), x)):
        # what this level passes over or under: the ground and the levels nearer to it, already final
        below = [k for k in range(len(axes)) if (layer[k] < L if L > 0 else layer[k] > L) and layer[k] * L >= 0]
        lower = [ribbons[k] for k in below] + [rings[i] for i in gring]
        tree = built_tree(lower) if lower else None

        def y_lower(xz, below=below, tree=tree):
            """The height of the lower pieces at points xz, NaN off them."""
            y = np.full(len(xz), np.nan)
            if tree is None:
                return y
            pi, gi = tree.query(shapely.points(xz), predicate="within")
            for p_, g_ in zip(pi, gi):
                if g_ < len(below):
                    k2 = below[g_]
                    s2, h2 = out[k2]
                    v = float(np.interp(LineString(axes[k2]).project(shapely.Point(xz[p_])), s2, h2))
                else:
                    v = hj[rnode[gring[g_ - len(below)]]]
                y[p_] = v if np.isnan(y[p_]) else max(y[p_], v)
            return y

        for k in [k for k in range(len(axes)) if layer[k] == L]:
            out[k] = structure_profile(axes[k], chains[owner[k]].w, L, terrain, y_lower)
        for i, n in enumerate(rnode):  # a junction on a bridge or in a tunnel: at its streets' mean end height
            if isinstance(n, tuple) and n[0] == L:
                ys = [out[k][1][0 if chains[owner[k]].f == n[1] else -1] for k in range(len(axes))
                      if layer[k] == L and n[1] in (chains[owner[k]].f, chains[owner[k]].t)]
                if ys:
                    hj[n] = float(np.mean(ys))
        for k in [k for k in range(len(axes)) if layer[k] == L]:
            s, h = out[k]
            for end, y in ((axes[k][0], h[0]), (axes[k][-1], h[-1])):
                _ramp_to(end, float(y), L > 0, near, hj, out)
    return hj, out


def structure_profile(axis, width, layer, terrain, y_lower):
    """(arclengths, heights) of a bridge (layer > 0) or a tunnel (layer < 0) along its centreline.

    A bridge never dips below the straight line between its ends nor below the ground, and stands
    BRIDGE_CLEAR above every street it crosses, climbing to it at GRADE. A tunnel never rises above the
    straight line between its ends, and runs TUNNEL_DEPTH under the ground or SEA_DEPTH under the sea,
    going down to it at GRADE; its ends sit as deep as the ground there allows, and the streets it leaves
    from go down to them."""
    from caiman.rules.ground import SEA

    q = densify(axis, BRIDGE_STEP if layer > 0 else SAMPLE)
    s = arclen(q)
    g = _level(terrain, q, np.full(len(q), width / 2))
    lin = g[0] + (g[-1] - g[0]) * s / max(s[-1], 1e-9)
    d = np.abs(s[:, None] - s[None, :])
    if layer > 0:
        under = y_lower(q)
        hit = ~np.isnan(under) & (s > END_FREE) & (s < s[-1] - END_FREE)  # not the street it leaves from
        h = np.maximum(lin, g)
        if hit.any():
            h = np.maximum(h, (under[hit][None, :] + BRIDGE_CLEAR - GRADE * d[:, hit]).max(axis=1))
        return s, h
    raw = terrain.height(q[:, 0], q[:, 1])
    need = np.where(raw < SEA, -SEA_DEPTH, raw - TUNNEL_DEPTH)
    return s, np.minimum(lin, (need[None, :] + GRADE * d).min(axis=1))


class _Ends:
    """The ground streets' ends and junctions, found by point: where bridges and tunnels leave the ground."""

    def __init__(self, ground, axes, widths, rings, nodes):
        self.k = ground
        self.axes = [axes[k] for k in ground]
        self.reach = np.maximum(np.asarray(widths, float), 4.0)
        pts = np.array([q[i] for q in self.axes for i in (0, -1)]).reshape(-1, 2)
        self.ends = built_tree(shapely.points(pts)) if len(pts) else None
        self.rings, self.nodes = rings, nodes
        self.rtree = built_tree(rings) if rings else None

    def at(self, point):
        """The ground junctions at point, and (ribbon, whether its start is the end there) for every ground
        street ending at point or at the edge of one of those junctions."""
        if self.ends is None:
            return [], []
        p = shapely.Point(point)
        rings = [] if self.rtree is None else list(self.rtree.query(p, predicate="dwithin", distance=1.0))
        zone = shapely.union_all([p.buffer(float(self.reach.max()))] + [self.rings[i].buffer(1.0) for i in rings])
        out = []
        for g in self.ends.query(zone):
            n, at_start = int(g) // 2, int(g) % 2 == 0
            e = self.axes[n][0 if at_start else -1]
            if (np.hypot(*(e - point)) <= self.reach[n]
                    or any(self.rings[i].distance(shapely.Point(e)) < 1.0 for i in rings)):
                out.append((self.k[n], at_start))
        return [self.nodes[i] for i in rings], out


def _ramp_to(point, y, up, near, hj, out):
    """The ground streets that end at a bridge's or tunnel's end `point` climb (up) or go down to its height y
    at GRADE, easing back to their own profile before their far end; a ground junction there takes y."""
    nodes, streets = near.at(point)
    for node in nodes:
        hj[node] = y
    for k, at_start in streets:
        s, h = out[k]
        dist = s if at_start else s[-1] - s
        fade = np.clip((s[-1] - dist) / min(EASE, s[-1] / 2 or 1.0), 0, 1)
        want = y - GRADE * dist if up else y + GRADE * dist
        gap = np.maximum(want - h, 0) if up else np.minimum(want - h, 0)
        out[k] = (s, h + gap * fade)


def _on(tris, xz, fallback):
    """The height at points xz (N, 2) of the triangles (T, 3, 3) they fall in; `fallback` where none."""
    if not len(tris):
        return np.array(fallback, float)
    a = tris[:, 0, [0, 2]]
    u, v = tris[:, 1, [0, 2]] - a, tris[:, 2, [0, 2]] - a
    det = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
    ok = np.abs(det) > 1e-9
    a, u, v, det, y = a[ok], u[ok], v[ok], det[ok], tris[ok][:, :, 1]
    d = xz[:, None, :] - a[None]
    b1 = (d[..., 0] * v[:, 1] - d[..., 1] * v[:, 0]) / det
    b2 = (u[:, 0] * d[..., 1] - u[:, 1] * d[..., 0]) / det
    inside = (b1 > -1e-4) & (b2 > -1e-4) & (b1 + b2 < 1 + 1e-4)
    out = np.array(fallback, float)
    hit = inside.any(axis=1)
    t = inside.argmax(axis=1)[hit]
    w1, w2 = b1[hit, t], b2[hit, t]
    out[hit] = (1 - w1 - w2) * y[t, 0] + w1 * y[t, 1] + w2 * y[t, 2]
    return out


def _strip(q, s, h, off, keep, tris):
    """Triangles (K, 3, 3) of a painted line MARK each side of offset `off` from centreline q, over the
    stretches of q where keep(arclength) is True, on the ribbon's triangles `tris` (its profile, arclengths
    s and heights h, where a point falls outside them)."""
    q = densify(q, DASH)
    sq = arclen(q)
    a, b = offset(q, off + MARK)[0], offset(q, off - MARK)[0]
    mid = (sq[1:] + sq[:-1]) / 2
    k = np.flatnonzero(keep(mid))
    if not len(k):
        return np.zeros((0, 3, 3))
    ya = _on(tris, a, np.interp(sq, s, h)) + MARK_LIFT
    yb = _on(tris, b, np.interp(sq, s, h)) + MARK_LIFT
    pa, pb = np.c_[a[:, 0], ya, a[:, 1]], np.c_[b[:, 0], yb, b[:, 1]]
    quads = [pa[k], pa[k + 1], pb[k + 1], pb[k]]
    return np.concatenate([np.stack(quads[:3], axis=1), np.stack([quads[0], quads[2], quads[3]], axis=1)])


def marks(axes, owner, chains, prof, tris):
    """The painted lines of every ribbon as (K, 3, 3) triangles: a dashed centre line on a two-way street, a
    solid one on a divided avenue or a wide two-way street, and a solid line along each edge of a street
    10 m wide or more. Only the classes in PAINTED are painted: Havana's side streets mostly are not. tris[k] are ribbon k's
    triangles, which the paint lies on."""
    def one(k):
        e, q = chains[owner[k]], axes[k]
        if e.cls not in PAINTED or len(q) < 2:
            return np.zeros((0, 3, 3))
        s, h = prof[k]
        out = []
        if not e.oneway or e.paired:
            solid = e.paired or e.w >= 10
            out.append(_strip(q, s, h, 0.0, (lambda m: np.ones(len(m), bool)) if solid else (lambda m: m % (DASH + DASH_GAP) < DASH),
                              tris[k]))
        if e.w >= 10:
            for side in (1, -1):
                out.append(_strip(q, s, h, side * (e.w / 2 - 0.6), lambda m: np.ones(len(m), bool), tris[k]))
        return np.concatenate(out) if out else np.zeros((0, 3, 3))

    return np.concatenate(pmap(one, range(len(axes)), chunk=128)).reshape(-1, 3, 3)


def mesh(ribbons, owner, axes, rings, rnode, chains, terrain, keep=None):
    """The street surface cut per cell: (triangles (T, 3, 3) as x, y, z, the cell of each (T, 2), skirt edges
    (E, 2, 3) with the road on their left in (x, z), the cell of each (E, 2), the painted lines' triangles
    (M, 3, 3) and the cell of each (M, 2)).

    Each piece is cut by the cells it crosses and each part triangulated on its boundary points, spaced at most
    SAMPLE metres apart along a ribbon. A point of a junction takes its height; a point of a ribbon takes the height of its
    profile at the point of the centreline beside it, so the ribbon is level across. A skirt edge is a stretch
    of a piece's boundary no other piece touches, where the road stands more than CLEAR above the ground, and
    not on a bridge or in a tunnel, which caiman.structures dresses instead. A dict `keep` gets the junction
    heights and the ribbons' profiles as "hj" and "prof"."""
    hj, prof = heights(ribbons, owner, axes, rings, rnode, chains, terrain)
    if keep is not None:
        keep["hj"], keep["prof"] = hj, prof
    pieces_ = list(ribbons) + list(rings)
    nr = len(ribbons)
    lines = [LineString(q) for q in axes]

    def y_of(k, xz):
        if k >= nr:
            return np.full(len(xz), hj[rnode[k - nr]])
        s, h = prof[k]
        return np.interp(shapely.line_locate_point(lines[k], shapely.points(xz)), s, h)

    def cut_cells(k):
        poly = shapely.orient_polygons(pieces_[k].segmentize(SAMPLE) if k < nr else pieces_[k])
        if not poly.is_valid:  # a piece can come out of the unions a hair invalid; cutting it would fail
            poly = shapely.orient_polygons(largest(shapely.make_valid(poly)) or poly.buffer(0))
        x0, z0, x1, z1 = poly.bounds
        tris, keys = [], []
        for i in range(math.floor(x0 / CELL), math.floor(x1 / CELL) + 1):
            for j in range(math.floor(z0 / CELL), math.floor(z1 / CELL) + 1):
                part = shapely.intersection(poly, box(i * CELL, j * CELL, (i + 1) * CELL, (j + 1) * CELL))
                for g in getattr(part, "geoms", [part]):
                    if not isinstance(g, Polygon) or g.area < 1e-4:
                        continue
                    t = shapely.get_coordinates(shapely.constrained_delaunay_triangles(g)).reshape(-1, 4, 2)[:, :3]
                    tris.append(t)
                    keys.append(np.tile((i, j), (len(t), 1)))
        if not tris:
            return np.zeros((0, 3, 3)), np.zeros((0, 2), int), poly
        t = np.concatenate(tris)
        y = y_of(k, t.reshape(-1, 2)).reshape(-1, 3)
        return np.stack([t[..., 0], y, t[..., 1]], axis=-1), np.concatenate(keys), poly

    cut = pmap(cut_cells, range(len(pieces_)), chunk=64)
    tree = built_tree(pieces_)

    def skirt(k):
        if (chains[owner[k]].layer if k < nr else isinstance(rnode[k - nr], tuple)):
            return np.zeros((0, 2, 3))
        ring = shapely.get_coordinates(cut[k][2].exterior)
        a, b = ring[:-1], ring[1:]
        mid = (a + b) / 2
        point, piece = tree.query(shapely.points(mid), predicate="dwithin", distance=0.05)
        shared = np.zeros(len(mid), bool)
        shared[point[piece != k]] = True
        a, b = a[~shared], b[~shared]
        if not len(a):
            return np.zeros((0, 2, 3))
        ya, yb = y_of(k, a), y_of(k, b)
        clear = (ya - terrain.height(a[:, 0], a[:, 1]) > CLEAR) | (yb - terrain.height(b[:, 0], b[:, 1]) > CLEAR)
        a, b, ya, yb = a[clear], b[clear], ya[clear], yb[clear]
        return np.stack([np.c_[a[:, 0], ya, a[:, 1]], np.c_[b[:, 0], yb, b[:, 1]]], axis=1)

    edges = pmap(skirt, range(len(pieces_)), chunk=64)
    lines = marks(axes, owner, chains, prof, [cut[k][0] for k in range(nr)]).astype(np.float32)
    mkeys = np.floor(lines[:, :, [0, 2]].mean(axis=1) / CELL).astype(np.int64)
    tris = np.concatenate([c[0] for c in cut]).astype(np.float32)
    tkeys = np.concatenate([c[1] for c in cut]).astype(np.int64)
    edges = np.concatenate(edges).reshape(-1, 2, 3).astype(np.float32)
    ekeys = np.floor(edges[:, :, [0, 2]].mean(axis=1) / CELL).astype(np.int64)
    return tris, tkeys, edges, ekeys, lines, mkeys


def add_to(city, widths) -> dict:
    """Build the streets of the city's road graph and store them in it, cut per cell; then the height of every
    graph point, so traffic drives on the same surface. Returns counts for the report."""
    from caiman.landmarks import Landmarks
    from caiman.surface import Surface

    from caiman import structures

    for k in ("street_tris", "street_edges", "street_marks") + STRUCTURE_KINDS:
        city.a.pop(k, None)
        city.a.pop(f"{k}_offsets", None)
    ribbons, owner, axes, rings, rnode, chains = build(city.a, widths, Landmarks(city).wall_rings)
    keep = {}
    tris, tkeys, edges, ekeys, lines, mkeys = mesh(ribbons, owner, axes, rings, rnode, chains, city.terrain, keep)
    city.add_kind("street_tris", tris, tkeys)
    city.add_kind("street_edges", edges, ekeys)
    city.add_kind("street_marks", lines, mkeys)
    ground = [r for k, r in enumerate(ribbons) if not chains[owner[k]].layer]
    ground += [r for r, n in zip(rings, rnode) if not isinstance(n, tuple)]
    st, paint, skeys, walls, wkeys, cuts, ckeys, roofs, rkeys = structures.build(
        ribbons, owner, axes, chains, keep["prof"], city.terrain, built_tree(ground) if ground else None)
    city.add_kind("struct_tris", st, skeys)
    city.add_kind("struct_paint", paint, skeys)
    city.add_kind("struct_walls", walls, wkeys)
    city.add_kind("street_cuts", cuts, ckeys)
    city.add_kind("street_roofs", roofs, rkeys)
    poly = city.a["g_poly"].astype(float)
    city.a["g_poly_y"] = Surface(city, city.terrain).height(poly[:, 0], poly[:, 1]).astype(np.float32)
    return {"streets": len(chains), "bridges_and_tunnels": sum(1 for c in chains if c.layer),
            "structure_triangles": len(st), "structure_walls": len(walls), "cut_triangles": len(cuts),
            "junction_pieces": len(rings), "street_triangles": len(tris),
            "street_skirt_edges": len(edges), "street_mark_triangles": len(lines)}


def main(path) -> dict:
    """Rebuild the streets of a prepared data file in place and print the counts."""
    import json
    import time

    from caiman.citydata import CityData
    from caiman.prepare import WIDTHS

    t0 = time.perf_counter()
    city = CityData.load(path)
    report = add_to(city, WIDTHS)
    city.save(path)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(report, indent=2))
    return report
