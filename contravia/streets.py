"""Streets: a clean drivable surface for the whole city, built once in `contravia prepare` from the road graph.

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
5. mesh: a plane per junction fitted to the ground under it, ribbons that follow the ground and ease into
   their junctions' planes, the pieces cut per cell and triangulated, plus the edges where the road stands
   clear of the ground, for skirts.

The work runs on a thread pool: this Python has no GIL. Shapely prepares a geometry lazily the first time a
predicate needs it, which is not safe from several threads at once, so every shared tree is built and
prepared up front (built_tree).
"""

import math
from collections import defaultdict

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.strtree import STRtree

from contravia.geo import CELL
from contravia.graph import CLASSES

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
LIFT = 0.05  # metres above the ground; contravia.rules.roads keeps 0.15 for data without a street surface
EASE = 20.0  # metres over which a ribbon's height eases into its junction's
SAMPLE = 8.0  # metres between the boundary points a ribbon is triangulated on; a junction is a plane
FIT = 4.0  # metres between the ground points a junction's plane is fitted to
CLEAR = 0.2  # metres: an edge of the road this far above the ground gets a skirt
MARK = 0.15  # metres: half the width of a painted line
MARK_LIFT = 0.02  # metres above the asphalt
GRADE = 0.08  # the steepest a ramp onto a bridge or down into a tunnel climbs, 8 %
OVER = 0.03  # metres a ground street stands over the drawn ground at the least, everywhere on its surface
SUNK = 0.3  # metres under the ground past which a street is in a cut (structures.BELOW), not to be raised
PROBE = 2.0  # metres between the points of a street's triangles where the ground under it is measured
CARRY = 0.3  # metres: a ramp still this far from a street's own height at its far end goes on through the junction
ACROSS = 0.7  # the least cosine between a street and the one a ramp goes straight on to through a junction
APART = 3.0  # metres: two levels' ribbons that overlap closer than this in height are no crossing, and are cut apart
SIDE = 0.16  # the grade of a ramp on a street that turns off at a junction the ramp passes: twice as steep
BRIDGE_CLEAR = 6.0  # metres from a street up to the bridge crossing it: 5 m of headroom and the deck
BRIDGE_STEP = 2.0  # metres between the samples of a bridge's profile, so no street under it is missed
TUNNEL_DEPTH = 8.0  # metres of a tunnel under the ground: its headroom, roof and cover
END_FREE = 8.0  # metres at each end of a bridge where the ground street it leaves from lies under it
SEA_DEPTH = 14.0  # metres under sea level where a tunnel runs beneath the sea
AT_END = 1.0  # metres: a ground street leaves from a bridge's or tunnel's end when its own end is this near
VALLEY = 10.0  # metres the ground drops under the line between a bridge's ends for the bridge to hold them up
DASH, DASH_GAP = 3.0, 6.0  # metres of paint and of gap along a dashed centre line, and of each painted piece
PAINTED = {c for k in ("motorway", "trunk", "primary", "secondary", "tertiary") for c in (k, f"{k}_link")}

# What contravia.structures adds per cell: triangles and their paint, walls with heights, cuts in the ground, roofs.
STRUCTURE_KINDS = ("struct_tris", "struct_paint", "struct_walls", "street_cuts", "street_roofs")

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


def left_normals(p, s):
    """The unit normal to the left, (-tz, tx), of the segment of polyline p at each arclength s: the normal
    offset() puts a ribbon's end on, so a point across an end is told apart exactly."""
    d = unit(np.diff(p, axis=0))
    i = np.clip(np.searchsorted(arclen(p), s, side="right") - 1, 0, len(d) - 1)
    return np.stack([-d[i, 1], d[i, 0]], axis=1)


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
        self.layer = layer  # above 0 on a bridge, below 0 in a tunnel (contravia.graph.layer)


def load(a, widths):
    """The junctions and the undirected edges of the road graph in the arrays a, each as wide as its lanes
    (contravia.graph.road_width)."""
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
        if "g_width" in a:  # from the lanes tag (contravia.graph.road_width)
            w = float(a["g_width"][e])
        else:
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
    ends = defaultdict(list)  # node -> [(outward direction, width)]
    for e in edges:
        ends[e.f].append((unit(e.p[1] - e.p[0]), e.w))
        ends[e.t].append((unit(e.p[-2] - e.p[-1]), e.w))

    def across(node, d):
        """How wide the junction at node is along direction d: the widest street there that d crosses."""
        return max((w for u, w in ends[node] if abs(float(u @ d)) < 0.7), default=0.0)

    for e in edges:
        d = unit(e.p[-1] - e.p[0])
        if length(e.p) < 0.5 * (across(e.f, d) + across(e.t, d)):
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
                    pts.insert(0, q[::-1][:-1])
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
    """fn over items, results in order, in this thread.

    It ran on a thread pool once (this Python has no GIL), but GEOS keeps a reference count on its shared geometry
    factory that is not safe across threads: geometries freed in two threads at once crashed the process
    (GeometryFactory::dropRef). `contravia prepare --jobs` builds provinces in separate processes instead.
    `chunk` is kept for the callers' sake and does nothing."""
    return [fn(x) for x in items]


def end_widths(chains, ends):
    """The width each end eases into at its junction: the mean of its own and that of the street it continues
    straight into, if any."""
    def heading(x):
        c, which = x
        q = chains[c].p if which == 0 else chains[c].p[::-1]
        s = arclen(q)
        k = min(int(np.searchsorted(s, 10.0)), len(q) - 1)
        return unit(q[k] - q[0])

    keys = [x for es in ends.values() for x in es]
    dirs = dict(zip(keys, pmap(heading, keys, chunk=512)))

    def widths(es):
        out = []
        for x in es:
            w = chains[x[0]].w
            cos, y = min(((float(dirs[x] @ dirs[y]), y) for y in es if y != x), default=(1.0, None))
            out.append((x, (w + chains[y[0]].w) / 2 if cos < -0.8 else w))
        return out

    return dict(x for r in pmap(widths, list(ends.values()), chunk=256) for x in r)


def trims(chains, cache):
    """How far each end is cut back on its smooth curve, past every neighbour's ribbon, and the pairs of
    clusters whose edge is too short for its two trims even without their corner room: two junctions that
    only lack room for their corner arcs stay apart, joined by a short ribbon, so a side street does not
    bend into a merged junction."""
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
             if trim[(c, 0)] + trim[(c, 1)] - 2 * CORNER > lens[c] - 1.0 and lens[c] < MERGE_SHORT and e.f != e.t]
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
            # cut square at both ends: a bend's disc near an end would reach past it, over the next piece
            ends = LineString(q).buffer(float(w.max()), cap_style="flat")
            rib = largest(shapely.intersection(shapely.union_all(segs + bends), ends))
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
    grounds, and the stretch within WALL_GAP of the wall is left out, as contravia.landmarks leaves out the
    old ribbons that cross it. The rest of the road stays, ending at a new node.

    Each level is built on its own, the ground first: a bridge does not meet the street it passes over, so
    neither joins, trims nor pairs with the other. Where a bridge or tunnel leaves the ground, its end is moved
    onto the end of the ground street there, so the two pieces meet. Every chain keeps its layer; a junction of
    a bridge or tunnel is named (layer, cluster) in rnode, a ground junction by its cluster alone."""
    J, edges = load(a, widths)
    if walls:
        J, edges = _cut_at_walls(J, edges, walls)
    _flatten_layers(edges)
    levels = defaultdict(list)
    for e in edges:
        levels[e.layer].append(e)
    ground_nodes = {n for e in levels.get(0, ()) for n in (e.f, e.t)}
    portals = frozenset(n for L, es in levels.items() if L for e in es for n in (e.f, e.t)) & ground_nodes
    out = [[], [], [], [], [], []]
    ends, ground = np.zeros((0, 2)), None
    for layer in sorted(levels, key=lambda x: (x != 0, abs(x), x)):
        Jl = J if layer == 0 else _onto_ground(J, levels[layer], ground_nodes, ends)
        ribbons, owner, axes, rings, rnode, chains = _level_pieces(Jl, levels[layer], portals)
        if layer and ground is not None:
            ribbons, owner, axes = _off_ground(ribbons, owner, axes, ground)
            rings = _reach_ground(rings, axes, [Jl[n] for n in {n for e in levels[layer] for n in (e.f, e.t)}
                                                & ground_nodes], ground)
        for c in chains:
            c.layer = layer
        if layer == 0 and chains:
            ends = np.array([p for c in chains for p in (c.p[0], c.p[-1])])
            ground = (built_tree(list(rings)), list(rings), list(ribbons), list(axes))
        n = len(out[5])
        out[0] += list(ribbons)
        out[1] += [o + n for o in owner]
        out[2] += list(axes)
        out[3] += list(rings)
        out[4] += [r if layer == 0 else (layer, r) for r in rnode]
        out[5] += chains
    out[0] = _split_twins(out[0], out[2], [k for k, o in enumerate(out[1]) if out[5][o].layer < 0])
    for L in sorted({c.layer for c in out[5] if c.layer > 0}):  # and a bridge's two carriageways, level by level
        out[0] = _split_twins(out[0], out[2], [k for k, o in enumerate(out[1]) if out[5][o].layer == L])
    return tuple(out)


def _flatten_layers(edges):
    """OSM's layer orders what crosses at one spot; it is not a depth. A bridge or tunnel on a layer beyond 1 (the
    Bay tunnel's bores, -1 and -3) that crosses no bridge or tunnel of another layer goes to layer 1 (or -1), so
    it is built and settled with the rest of its structure."""
    odd = [e for e in edges if abs(e.layer) > 1]
    if not odd:
        return
    lifted = [e for e in edges if e.layer]
    lines = [LineString(e.p) for e in lifted]
    tree = STRtree(lines)
    for e in odd:
        line = LineString(e.p)
        hits = [lifted[i] for i in tree.query(line, predicate="crosses")]
        if not any(o.layer != e.layer and o.layer * e.layer > 0 for o in hits):
            e.layer = 1 if e.layer > 0 else -1


def _split_twins(ribbons, axes, tubes):
    """The ribbons with each two tunnels side by side (a tunnel's two carriageways) cut apart halfway between
    their centrelines, where contravia.structures puts the edge of their shared box: overlapping, a wheel near
    the middle would stand on the other carriageway's road, which is higher or lower near the mouths. The two
    carriageways of a bridge are cut apart the same way, and structures builds them as one deck."""
    ribbons = list(ribbons)
    for i, k in enumerate(tubes):
        for k2 in tubes[i + 1:]:
            a_, b_ = ribbons[k], ribbons[k2]
            if a_.intersection(b_).area < 1.0:
                continue
            a = densify(axes[k], 2.0)
            cells = shapely.voronoi_polygons(shapely.multipoints(np.vstack([a, densify(axes[k2], 2.0)])),
                                             extend_to=a_.union(b_).envelope).geoms
            mine = shapely.points(a)
            side = shapely.union_all([c for c in cells if shapely.intersects(c, mine).any()])
            ribbons[k] = largest(a_.difference(b_.difference(side))) or a_
            ribbons[k2] = largest(b_.difference(a_.intersection(side))) or b_
    return ribbons


def _apart(ribbons, axes, layers, prof, moved=()):
    """The ribbons with every bridge or tunnel cut apart from a ribbon of another level it overlaps without
    passing APART over or under it, halfway between their centrelines; and so every ground ribbon a ramp took
    off the ground (moved) from a ground ribbon it overlaps at another height.

    A bridge crosses a street BRIDGE_CLEAR above it. A ramp that leaves a junction beside a bridge's end, the
    open approach of one carriageway beside the other's tunnel, or two approaches side by side at different
    heights lie partly on the same plan a little higher or lower: a wheel there would jump from one to the
    other. Each keeps its side only when that leaves both in one piece."""
    out = list(ribbons)
    lifted = [k for k, L in enumerate(layers) if L] + sorted(moved)
    if not lifted:
        return out
    moved = set(moved)
    tree, lines = built_tree(ribbons), {}

    def y(k, pts):
        if k not in lines:
            lines[k] = LineString(axes[k])
        return np.interp(shapely.line_locate_point(lines[k], pts), *prof[k])

    for k in lifted:
        for j in tree.query(ribbons[k], predicate="intersects"):
            j = int(j)
            if j == k or (layers[j] == layers[k] and (layers[k] or (j in moved and j < k))):
                continue
            if layers[j] and (layers[k] == 0 or j < k):  # each pair once, from its bridge or tunnel
                continue
            both = out[k].intersection(out[j])
            if both.area < 1.0:
                continue
            pts = shapely.points(shapely.get_coordinates(both.segmentize(4.0)))
            gap = np.abs(y(k, pts) - y(j, pts))
            if gap.min() >= APART or (layers[k] == layers[j] and gap.max() < 0.15):
                continue
            a = densify(axes[k], 2.0)
            cells = shapely.voronoi_polygons(shapely.multipoints(np.vstack([a, densify(axes[j], 2.0)])),
                                             extend_to=out[k].union(out[j]).envelope).geoms
            mine = shapely.points(a)
            side = shapely.union_all([c for c in cells if shapely.intersects(c, mine).any()])
            cut = [out[k].difference(out[j].difference(side)), out[j].difference(out[k].intersection(side))]
            kept = [largest(c) for c in cut]
            # two ribbons along the same line (a ford beside its bridge) do not part into two sides: left as they are
            if all(p is not None and p.area > c.area - 1.0 for p, c in zip(kept, cut)):
                out[k], out[j] = kept
    return out


def _reach_ground(rings, axes, portals, ground):
    """A bridge's or tunnel's junctions with a strip added from each ground street that starts at one of them.

    A bridge whose last metres are too short to stand apart is merged into the junction of its own level beside
    it. No ribbon of the level then ends at the ground street that goes on from there (at one of the points
    portals), and the junction's piece stops short of it: a gap with no street. The junction gets the street's
    width from the street's end straight on until it is inside the junction. ground is as in _off_ground."""
    _, _, gribs, gaxes = ground
    if not len(rings) or not gaxes or not portals:
        return rings
    rings = list(rings)
    ends = np.array([q[i] for q in axes for i in (0, -1)]).reshape(-1, 2)
    starts = built_tree(shapely.points(np.array([q[i] for q in gaxes for i in (0, -1)])))
    tree = built_tree(rings)
    for p in portals:
        if len(ends) and np.hypot(*(ends - p).T).min() < 2.0:  # a ribbon of the level ends here
            continue
        P = shapely.Point(p)
        i = tree.query_nearest(P, max_distance=20.0)
        if not len(i):
            continue
        i = int(i[0])
        for g in starts.query(P, predicate="dwithin", distance=0.05):
            q, rib = gaxes[int(g) // 2], gribs[int(g) // 2]
            u = unit(q[0] - q[1] if int(g) % 2 == 0 else q[-1] - q[-2])  # on past the street's end
            half = rib.area / max(float(arclen(q)[-1]), 1e-9) / 2
            strip = LineString([p, p + u * (rings[i].distance(P) + half)]).buffer(half, cap_style="flat")
            rings[i] = largest(shapely.union_all([rings[i], strip.difference(rib)])) or rings[i]
    return rings


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


def _off_ground(ribbons, owner, axes, ground):
    """A tunnel's or a bridge's pieces fitted to the ground pieces they start from.

    _onto_ground puts each end at the end of a ground street, which is the centre of the junction when one is
    there: left as it is, the tunnel or bridge would run on into that junction, two surfaces at different
    heights in one place (a junction is flat or a plane, a bridge climbs from its first metre), with the walls
    of one across the other. Each end in a junction is cut back along its centreline to the junction's edge,
    and the ribbon loses what still lies in it. An end where a ground street starts gets the sliver between
    the two squared ends, which meet at a slight angle: through that crack a wheel would fall to the ground
    under a bridge, or see only a street crossing above a tunnel. ground is (an STRtree of the ground
    junctions, the junctions, the ground ribbons and their centrelines)."""
    tree, rings, gribs, gaxes = ground
    starts = built_tree(shapely.points(np.array([q[i] for q in gaxes for i in (0, -1)]))) if gaxes else None
    out_r, out_a = [], []
    for rib, q in zip(ribbons, axes):
        line = LineString(q)
        L, trim, held, meet = line.length, [0.0, 0.0], [], []
        for end, p in ((0, q[0]), (1, q[-1])):
            P = shapely.Point(p)
            for g in tree.query(P, predicate="within"):
                cross = shapely.get_coordinates(shapely.intersection(line, rings[g].exterior))
                if not len(cross):
                    continue
                s = shapely.line_locate_point(line, shapely.points(cross))
                trim[end] = max(trim[end], float(s.min() if end == 0 else L - s.max()))
                held.append(rings[g])
            if not trim[end] and starts is not None:
                meet += [(P, gribs[int(i) // 2]) for i in starts.query(P, predicate="dwithin", distance=0.05)]
        if held and sum(trim) < L - 2.0:
            q = cut(q, trim[0], L - trim[1])
            rib = largest(rib.difference(shapely.union_all(held))) or rib
        for P, g in meet:
            ball = P.buffer(rib.area / max(L, 1e-9) / 2 + 1.0)  # past the half width
            wedge = shapely.union_all([rib.intersection(ball), g.intersection(ball)]).convex_hull.difference(g)
            rib = largest(rib.union(wedge)) or rib
        out_r.append(rib)
        out_a.append(q)
    return out_r, owner, out_a


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


def _follow(terrain, p, half):
    """The road's centre heights and cross slopes at centre points p (N, 2) with half widths half (N,).

    Across, the road is the straight line between the ground under its kerbs, LIFT over it, raised where the
    ground under the centre stands higher than that line (a crown), so the ground never shows through; its
    slope is metres up per metre toward the left, the side of (-tz, tx) for the tangent (tx, tz)."""
    t = tangents(p)
    side = np.stack([-t[:, 1], t[:, 0]], axis=1) * np.asarray(half)[..., None]
    left, centre, right = (terrain.height(q[:, 0], q[:, 1]) for q in (p + side, p, p - side))
    mean = (left + right) / 2
    return mean + np.maximum(centre - mean, 0) + LIFT, (left - right) / (2 * np.asarray(half, float))


class Plane:
    """A junction's surface: y = y0 + gx (x - x0) + gz (z - z0)."""

    __slots__ = ("gx", "gz", "x0", "y0", "z0")

    def __init__(self, x0, z0, y0, gx=0.0, gz=0.0):
        self.x0, self.z0, self.y0, self.gx, self.gz = float(x0), float(z0), float(y0), float(gx), float(gz)

    def __call__(self, xz):
        xz = np.asarray(xz, float).reshape(-1, 2)
        return self.y0 + self.gx * (xz[:, 0] - self.x0) + self.gz * (xz[:, 1] - self.z0)


def _junction_plane(poly, terrain):
    """The plane that fits the ground under junction poly best (least squares over points FIT metres apart on
    its edge and inside it), raised to stand LIFT over the highest of those points above it."""
    pts = shapely.get_coordinates(poly.segmentize(FIT))
    x0, z0, x1, z1 = poly.bounds
    gx, gz = np.meshgrid(np.arange(x0, x1, FIT), np.arange(z0, z1, FIT))
    grid = np.stack([gx.ravel(), gz.ravel()], axis=1)
    pts = np.vstack([pts, grid[shapely.contains_xy(poly, grid[:, 0], grid[:, 1])]])
    g = terrain.height(pts[:, 0], pts[:, 1]).astype(float)
    c = pts.mean(axis=0)
    d = pts - c
    if len(pts) >= 3 and np.linalg.matrix_rank(d) == 2:
        a = np.c_[np.ones(len(d)), d]
        # the normal equations written out: numpy.linalg.lstsq calls LAPACK, slower than this 3x3 solve
        y0, sx, sz = np.linalg.solve(a.T @ a, a.T @ g)
    else:
        y0, sx, sz = float(g.mean()), 0.0, 0.0
    plane = Plane(c[0], c[1], y0, sx, sz)
    plane.y0 += float((g - plane(pts)).max()) + LIFT
    return plane


class _Raised:
    """How much the ground streets are raised so the drawn ground never stands over them: metres per vertex of
    the ground's grid (rules.ground: N vertices per cell side), straight between them over the grid's two
    triangles per square, as the ground itself is drawn. A street's heights follow the ground at its kerbs and
    its centre, a sample every SAMPLE metres; between them the grid's creases can stand over the asphalt. Every
    ground triangle that does lifts its three vertices here by what it stands over plus OVER, so all of it
    comes under, and the streets stay one surface since the lift is a height over the plane."""

    def __init__(self):
        self.keys, self.vals = np.zeros(0, np.int64), np.zeros(0)

    @staticmethod
    def _grid(xz):
        from contravia.rules.ground import N

        f = np.asarray(xz, float).reshape(-1, 2) / (CELL / (N - 1))
        c = np.floor(f).astype(np.int64)
        return c, f - c

    @staticmethod
    def _key(c, r):
        return (c << 32) | (r & 0xFFFFFFFF)

    def __call__(self, xz) -> np.ndarray:
        """The lift at points xz (P, 2)."""
        xz = np.asarray(xz, float).reshape(-1, 2)
        if not len(self.keys):
            return np.zeros(len(xz))
        c, t = self._grid(xz)

        def at(dc, dr):
            k = self._key(c[:, 0] + dc, c[:, 1] + dr)
            n = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
            return np.where(self.keys[n] == k, self.vals[n], 0.0)

        v00, v01, v10, v11 = at(0, 0), at(1, 0), at(0, 1), at(1, 1)
        tx, tz = t[:, 0], t[:, 1]
        return np.where(tx + tz <= 1, v00 + tx * (v01 - v00) + tz * (v10 - v00),
                        v11 + (1 - tx) * (v10 - v11) + (1 - tz) * (v01 - v11))

    def add(self, xz, need) -> None:
        """Lift the ground triangle under each point xz (P, 2) by need (P,) metres more, at the least."""
        if not len(xz):
            return
        c, t = self._grid(xz)
        lower = t.sum(axis=1) <= 1
        corners = np.where(lower[:, None, None], [[0, 0], [1, 0], [0, 1]], [[1, 1], [0, 1], [1, 0]])
        k = self._key(c[:, None, 0] + corners[:, :, 0], c[:, None, 1] + corners[:, :, 1]).ravel()
        keys, back = np.unique(np.concatenate([self.keys, k]), return_inverse=True)
        vals = np.zeros(len(keys))
        np.maximum.at(vals, back[: len(self.keys)], self.vals)
        np.maximum.at(vals, back[len(self.keys):], np.repeat(self(xz) + need, 3))
        self.keys, self.vals = keys, vals


class _Drawn:
    """The ground as the cells draw it (contravia.rules.ground.drawn), with a Terrain's height()."""

    def __init__(self, terrain):
        self.terrain = terrain

    def height(self, x, z, near=None):
        from contravia.rules.ground import drawn

        return drawn(self.terrain, x, z)


def heights(ribbons, owner, axes, rings, rnode, chains, terrain, moved=None, sunk=None):
    """A plane per ground junction, a height per bridge or tunnel junction, and a height profile and a cross
    slope per ribbon.

    The ground here is the ground as the cells draw it, which the streets must clear (_Drawn).
    A ground junction lies on the plane that fits the ground under it, raised so that ground stands LIFT under
    it (_junction_plane). A ribbon follows the ground: at each point of its centreline, LIFT over the line
    across between the ground under its kerbs, tilted with it (_follow). Within EASE metres of a junction (half
    the ribbon when it is shorter) the ribbon takes on what that junction's plane differs from it by at its end,
    in height and in cross slope, so it meets the plane exactly there without following it away from the
    junction, where the ground may curve away from the plane. Bridges and tunnels then take their own profiles (structure_profile), level across, and the ground
    streets they leave from ramp up or down to meet them (_ramp_to), level across too.
    Returns (junction heights: the plane's height at its fitted centre, or the flat height, [(arclengths,
    heights)] along each centreline, [cross slopes at those arclengths], {junction: Plane}). A point of ribbon
    k at arclength s and distance u to the left of its centreline stands at heights + u * cross slope. A set
    `moved` gets the ground ribbons a ramp took off the ground, and `sunk` those of them it took under it."""
    layer = [chains[owner[k]].layer for k in range(len(axes))]
    ground_rings = [(r, n) for r, n in zip(rings, rnode) if not isinstance(n, tuple)]
    seen = _Drawn(terrain)
    planes = dict(zip([n for _, n in ground_rings], pmap(lambda rn: _junction_plane(rn[0], seen),
                                                          ground_rings, chunk=64)))
    hj = {n: planes[n].y0 if n in planes else 0.0 for n in rnode}
    ring_of = {n: r for r, n in ground_rings}

    def base(k):
        q = densify(axes[k], SAMPLE)
        # the arclength along the centreline itself, where mesh() looks a point up, not along the chords
        s = np.linspace(0.0, arclen(axes[k])[-1], len(q))
        e = chains[owner[k]]
        if layer[k]:
            return s, _level(terrain, q, np.full(len(q), e.w / 2)), np.zeros(len(q))
        h, tilt = _follow(seen, q, np.full(len(q), e.w / 2))
        span = min(EASE, s[-1] / 2) or 1.0
        left = left_normals(axes[k], s)
        lift, turn = np.zeros(len(q)), np.zeros(len(q))
        for node, w, i in ((e.f, _ease(s / span), 0), (e.t, _ease((s[-1] - s) / span), -1)):
            if node in planes and ring_of[node].distance(shapely.Point(q[i])) < 1.0:  # an end at its junction:
                # what the plane differs by there, faded out along the ribbon
                pl = planes[node]
                lift += (float(pl(q[i])[0]) - h[i]) * w
                turn += (pl.gx * left[i, 0] + pl.gz * left[i, 1] - tilt[i]) * w
        return s, h + lift, tilt + turn

    done = pmap(base, range(len(axes)), chunk=64)
    out = [(s, h) for s, h, _ in done]
    tilts = [t for _, _, t in done]
    if not any(layer):
        return hj, out, tilts, planes
    ground = [k for k in range(len(axes)) if not layer[k]]
    gring = [i for i, n in enumerate(rnode) if not isinstance(n, tuple)]
    near = _Ends(ground, axes, [rings[i] for i in gring],
                 [rnode[i] for i in gring], [(chains[owner[k]].f, chains[owner[k]].t) for k in ground])
    near.portals = built_tree(shapely.points(np.array([axes[k][i] for k in range(len(axes)) if layer[k]
                                                       for i in (0, -1)])))
    for L in sorted({x for x in layer if x}, key=lambda x: (abs(x), x)):
        # what this level passes over or under: the ground and the levels nearer to it, already final
        below = [k for k in range(len(axes)) if (layer[k] < L if L > 0 else 0 >= layer[k] > L)]
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
                    n = rnode[gring[g_ - len(below)]]
                    v = float(planes[n](xz[p_])[0]) if n in planes else hj[n]
                y[p_] = v if np.isnan(y[p_]) else max(y[p_], v)
            return y

        ks = [k for k in range(len(axes)) if layer[k] == L]
        for k in ks:
            room = (np.inf, np.inf) if L > 0 else (near.free(axes[k][0]), near.free(axes[k][-1]))
            w = chains[owner[k]].w
            held = _hold_ends(axes[k], w, terrain, near, hj, out, tilts, planes) if L > 0 else (None, None)
            out[k] = structure_profile(axes[k], w, L, terrain, y_lower, room, held)
            tilts[k] = np.zeros(len(out[k][0]))
        _share_twins(ks, axes, out, up=L > 0)

        def ground_at(point, up=L > 0):
            """The height of the ground street or junction a bridge's or tunnel's end at point leaves from, and
            that junction: (None, None) off the ground."""
            nodes, streets = near.at(point)
            ys = [float(planes[n](point)[0]) if n in planes else hj[n] for n in nodes]
            ys += [float(out[k][1][0 if at_start else -1]) for k, at_start in streets  # those ending at the point
                   if not nodes or np.hypot(*(near.axis[k][0 if at_start else -1] - point)) <= AT_END]
            return ((max if up else min)(ys) if ys else None), (nodes[0] if nodes else None)

        leaving = {}  # a junction of this level -> the ends of the ground streets that start at its edge
        for i, n in enumerate(rnode):
            if isinstance(n, tuple) and n[0] == L and near.ends is not None:
                for g in near.ends.query(rings[i], predicate="dwithin", distance=0.3):
                    k2, at_start = near.k[int(g) // 2], int(g) % 2 == 0
                    if near.node_of.get((k2, at_start)) not in ring_of:  # no junction of its own there
                        leaving.setdefault(n[1], []).append((k2, at_start))
        was = None
        for _ in range(4):  # a ramp can lift the junction another end leaves from: agree again until none moves
            ends = _agree(ks, L > 0, [(chains[owner[k]].f, chains[owner[k]].t) for k in ks], axes, out, ground_at,
                          {n: [float(out[k2][1][0 if st else -1]) for k2, st in v] for n, v in leaving.items()})
            now = np.array([out[k][1][i] for k in ks for i in (0, -1)])
            if was is not None and np.abs(now - was).max() < 0.01:
                break
            was = now
            for k in ks:
                h = out[k][1]
                for end, y in ((axes[k][0], h[0]), (axes[k][-1], h[-1])):
                    _ramp_to(end, float(y), L > 0, near, hj, out, tilts, planes)
            for n, v in leaving.items():
                for k2, at_start in v:
                    if n in ends:
                        _ramp_to(near.axis[k2][0 if at_start else -1], ends[n], L > 0, near, hj, out, tilts, planes)
        for n in rnode:  # a junction on a bridge or in a tunnel: at the height its streets end at
            if isinstance(n, tuple) and n[0] == L and n[1] in ends:
                hj[n] = ends[n[1]]
    if moved is not None:
        moved.update(k for k in ground if np.abs(out[k][1] - done[k][1]).max() > 0.05)
    if sunk is not None:
        sunk.update(k for k in ground if (out[k][1] - done[k][1]).min() < -0.05)
    return hj, out, tilts, planes


def _hold_ends(axis, width, terrain, near, hj, out, tilts, planes):
    """The heights a bridge over a valley holds its two ends at: (None, None) for a bridge whose ground never
    drops VALLEY under the line between its ends, and None for an end nothing on the ground leaves from.

    A relief too coarse for a gorge lays the gorge's walls under the bridge's ends, and the street that
    reaches an end comes down that wall. Such an end stands as high as its street can come down to at GRADE
    from its far end, and the street takes that slope where the ground falls faster: a fill, level across
    where it left the ground (out, tilts). Where the end leaves from a junction on the ground, every street
    of that junction comes down so and the end stands at the highest of them; heights then lays the junction
    flat there and brings its other streets to it (_ramp_to). A street shorter than MERGE_SHORT has no room
    to come down in: the junction at its far end is held up that way instead, and the street runs level
    from it to the bridge."""
    q = densify(axis, BRIDGE_STEP)
    s = arclen(q)
    g = _level(terrain, q, np.full(len(q), width / 2))
    if (g[0] + (g[-1] - g[0]) * s / max(s[-1], 1e-9) - g).max() < VALLEY:
        return None, None

    def come_down(k, at_start):
        """Street k comes down to its end (its start, at_start) no steeper than GRADE: that end's height."""
        sk, h = out[k]
        f = h.copy()
        for i in range(len(sk) - 2, -1, -1) if at_start else range(1, len(sk)):
            j = i + 1 if at_start else i - 1  # the sample farther from the bridge
            f[i] = max(h[i], f[j] - GRADE * abs(sk[j] - sk[i]))
        out[k] = (sk, f)
        tilts[k] = tilts[k] * (1 - np.clip((f - h) / 0.3, 0, 1))
        return float(f[0 if at_start else -1])

    held = []
    for point in (axis[0], axis[-1]):
        nodes, streets = near.at(point)
        tops = []
        for k, at_start in [x for n in nodes for x in near.meeting.get(n, ())] if nodes else streets:
            far = near.node_of.get((k, not at_start))
            beyond = [x for x in near.meeting.get(far, ()) if x[0] != k] if far in near.nodes else []
            if nodes or out[k][0][-1] >= MERGE_SHORT or not beyond:
                tops.append(come_down(k, at_start))
                continue
            top = max(come_down(k2, start2) for k2, start2 in beyond)
            if top > float(out[k][1].max()):
                out[k] = (out[k][0], np.full(len(out[k][0]), top))
                tilts[k] = np.zeros_like(tilts[k])
                _ramp_to(near.axis[k][-1 if at_start else 0], top, True, near, hj, out, tilts, planes)
            tops.append(float(out[k][1][0 if at_start else -1]))
        held.append(max(tops) if tops else None)
    return tuple(held)


def structure_profile(axis, width, layer, terrain, y_lower, room=(np.inf, np.inf), held=(None, None)):
    """(arclengths, heights) of a bridge (layer > 0) or a tunnel (layer < 0) along its centreline.

    A bridge never dips below the straight line between its ends nor below the ground, and stands
    BRIDGE_CLEAR above every street it crosses, climbing to it at GRADE; between two places it has to stand
    high over, its deck runs straight instead of coming down and up again (_arch). A tunnel never rises above the
    straight line between its ends, and runs TUNNEL_DEPTH under the ground or SEA_DEPTH under the sea,
    going down to it at GRADE; its ends sit as deep as the ground there allows, and the streets it leaves
    from go down to them.

    room is how many metres of street each end of a tunnel has to ramp down on before a junction (_Ends.free):
    a mouth sits no deeper than that street can reach at GRADE, and the tunnel goes on down inside, so the
    ramp does not sink a junction and the streets round it. What a tunnel has to pass under still decides:
    BRIDGE_CLEAR under a street crossing over it, SEA_DEPTH under the sea. held is the height each end of a
    bridge stands at when that is over the ground there (_hold_ends), None for the ground's."""
    from contravia.rules.ground import SEA

    q = densify(axis, BRIDGE_STEP if layer > 0 else SAMPLE)
    s = arclen(q)
    g = _level(terrain, q, np.full(len(q), width / 2))
    y0, y1 = (g[i] if y is None else max(float(g[i]), y) for i, y in ((0, held[0]), (-1, held[1])))
    lin = y0 + (y1 - y0) * s / max(s[-1], 1e-9)
    d = np.abs(s[:, None] - s[None, :])
    if layer > 0:
        under = y_lower(q)
        hit = ~np.isnan(under) & (s > END_FREE) & (s < s[-1] - END_FREE)  # not the street it leaves from
        h = np.maximum(lin, g)
        if hit.any():
            h = np.maximum(h, (under[hit][None, :] + BRIDGE_CLEAR - GRADE * d[:, hit]).max(axis=1))
        return s, _arch(s, h)
    raw = terrain.height(q[:, 0], q[:, 1])
    need = np.where(raw < SEA, -SEA_DEPTH, raw - TUNNEL_DEPTH)
    h = (need[None, :] + GRADE * d).min(axis=1)
    if np.isfinite(room).any():
        over = y_lower(q)
        must = np.where(raw < SEA, -SEA_DEPTH, np.inf)
        hit = ~np.isnan(over) & (s > END_FREE) & (s < s[-1] - END_FREE)  # not the street it leaves from
        must[hit] = np.minimum(must[hit], over[hit] - BRIDGE_CLEAR)
        reach = np.maximum(g[0] - GRADE * (room[0] + s), g[-1] - GRADE * (room[1] + s[-1] - s))
        h = np.minimum(np.maximum(h, reach), (must[None, :] + GRADE * d).min(axis=1))
    return s, np.minimum(lin, h)


def _arch(s, h):
    """Heights h at arclengths s raised to their upper hull: straight between any two points, never below."""
    hull = []
    for i in range(len(s)):
        while len(hull) >= 2 and ((s[hull[-1]] - s[hull[-2]]) * (h[i] - h[hull[-2]])
                                  - (h[hull[-1]] - h[hull[-2]]) * (s[i] - s[hull[-2]])) >= 0:
            hull.pop()
        hull.append(i)
    return np.interp(s, s[hull], h[hull])


def _share_twins(tubes, axes, out, up=False):
    """Two tunnels side by side, closer than structures.TWIN (the two carriageways of one tunnel, which
    structures builds as one box), run at one depth: each takes the lower of the two where they are side by side,
    then climbs from it no steeper than GRADE, so the box has one floor and one roof. With up, they are the two
    carriageways of a bridge: each takes the higher of the two and comes down from it no steeper than GRADE, so
    the deck is one."""
    from contravia.structures import TWIN

    if len(tubes) < 2:
        return
    sign = -1.0 if up else 1.0  # a bridge's heights are shared upside down
    lines = {k: LineString(axes[k]) for k in tubes}
    first = {k: out[k] for k in tubes}
    for k in tubes:
        s, h = first[k]
        h = sign * h
        pts = shapely.points(np.c_[[lines[k].interpolate(v).coords[0] for v in s]])
        low = h.copy()
        for j in tubes:
            if j == k:
                continue
            near = shapely.distance(pts, lines[j]) < TWIN
            if near.any():
                s2, h2 = first[j]
                other = sign * np.interp(shapely.line_locate_point(lines[j], pts[near]), s2, h2)
                low[near] = np.minimum(low[near], other)
        if (low < h).any():
            out[k] = (s, sign * np.minimum(h, (low[None, :] + GRADE * np.abs(s[:, None] - s[None, :])).min(axis=1)))


class _Ends:
    """The ground streets' ends and junctions, found by point: where bridges and tunnels leave the ground."""

    def __init__(self, ground, axes, rings, nodes, ends):
        self.k = ground
        self.axes = [axes[k] for k in ground]
        self.reach = np.full(len(ground), AT_END)
        pts = np.array([q[i] for q in self.axes for i in (0, -1)]).reshape(-1, 2)
        self.ends = built_tree(shapely.points(pts)) if len(pts) else None
        self.rings, self.nodes = rings, nodes
        self.rtree = built_tree(rings) if rings else None
        self.node_of = {}  # (ribbon, its start) -> the junction there
        self.meeting = {}  # junction -> [(ribbon, whether its start is there)]
        for k, (f, t) in zip(ground, ends):
            for node, at_start in ((f, True), (t, False)):
                self.node_of[k, at_start] = node
                self.meeting.setdefault(node, []).append((k, at_start))
        self.axis = dict(zip(ground, self.axes))
        self.portals = None  # an STRtree of the ends of every bridge and tunnel, set by heights

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

    def free(self, point):
        """How many metres of ground street lead away from point before a junction where another street joins:
        the room a ramp from a tunnel's mouth there has. inf when nothing on the ground starts at point."""
        nodes, streets = self.at(point)
        if nodes:
            return 0.0 if any(len(self.meeting.get(n, ())) > 1 for n in nodes) else np.inf
        if not streets:
            return np.inf
        best = 0.0
        for k, at_start in streets:
            got, seen = 0.0, set()
            while k not in seen:
                seen.add(k)
                got += float(arclen(self.axis[k])[-1])
                node = self.node_of.get((k, not at_start))
                on = [x for x in self.meeting.get(node, ()) if x[0] != k]
                far = shapely.Point(self.axis[k][-1 if at_start else 0])
                if not on and self.portals is not None and far.distance(shapely.Point(point)) > AT_END and len(
                        self.portals.query(far, predicate="dwithin", distance=AT_END)):
                    return np.inf  # it leads into another tunnel or onto a bridge: no junction to spare
                if len(on) != 1 or got > TUNNEL_DEPTH / GRADE:
                    break
                k, at_start = on[0]
            best = max(best, got)
        return best

    def straight_on(self, k, into_end, node):
        """The street that goes on most straight from ribbon k through the junction node at k's end into_end
        (its start when True), with whether its start is there; None when none turns less than ACROSS."""
        q = self.axis[k]
        u = unit(q[0] - q[1] if into_end else q[-1] - q[-2])
        best, out = ACROSS, None
        for k2, at_start in self.meeting.get(node, ()):
            if k2 == k:
                continue
            q2 = self.axis[k2]
            c = float(np.dot(u, unit(q2[1] - q2[0] if at_start else q2[-2] - q2[-1])))
            if c > best:
                best, out = c, (k2, at_start)
        return out


def _agree(ks, up, nodes, axes, out, ground_at, also=None):
    """The ribbons ks of one level brought to one height at each junction they share and to the ground's where an
    end leaves it: {node: height}.

    Each profile is solved on its own, so two bridges meeting at a junction on a viaduct arrive at different
    heights, and a bridge's end can come out under the junction it leaves from. A junction takes the highest of
    its ends (the lowest in a tunnel) and of the ground there (ground_at(point), None off the ground); every
    ribbon then climbs to its two junctions no steeper than GRADE, which can lift its other end, until nothing
    moves. Ground lower than an end is left to _ramp_to. also is {node: heights of ground streets that start at
    that junction's own edge}, which count as its ground."""
    pick, sign = (max, 1.0) if up else (min, -1.0)
    at, same = {}, {}  # same: a node of this level -> the ground junction its end leaves from, shared by others
    grounds = [[ground_at(axes[k][i]) for i in (0, -1)] for k in ks]
    for (f, t), g in zip(nodes, grounds):
        for node, (_, junction) in zip((f, t), g):
            if junction is not None:
                same[node] = ("ground", junction)
    nodes = [(same.get(f, f), same.get(t, t)) for f, t in nodes]
    for k, (f, t), g in zip(ks, nodes, grounds):
        for node, i, (y, _) in ((f, 0, g[0]), (t, -1, g[1])):
            ys = [float(out[k][1][i])] + ([at[node]] if node in at else [])
            at[node] = pick(ys if y is None else ys + [y])
    for node, ys in (also or {}).items():
        node = same.get(node, node)
        if node in at and ys:
            at[node] = pick(ys + [at[node]])
    for _ in range(4 * len(ks) + 4):
        moved = False
        for k, (f, t) in zip(ks, nodes):
            s, h = out[k]
            reach = np.stack([at[f] - sign * GRADE * s, at[t] - sign * GRADE * (s[-1] - s)])
            new = np.maximum(h, reach.max(axis=0)) if up else np.minimum(h, reach.min(axis=0))
            out[k] = (s, new)
            for node, i in ((f, 0), (t, -1)):
                if sign * (new[i] - at[node]) > 1e-6:
                    at[node], moved = float(new[i]), True
        if not moved:
            break
    return {**at, **{node: at[g] for node, g in same.items()}}


def _ramp_to(point, y, up, near, hj, out, tilts, planes):
    """The ground streets that end at a bridge's or tunnel's end `point` climb (up) or go down to its height y
    at GRADE, easing back to their own profile before their far end; a ground junction there lower than y (higher,
    for a tunnel) takes y.

    A street too short to get back to its own height carries the ramp through the junction at its far end:
    that junction takes the height the ramp reaches there, the street going most straight on from it carries
    the ramp on at GRADE and the streets that turn off there at SIDE, each through its own far junction in turn
    when it is too short as well. A junction keeps the highest ramp that reaches it (the lowest, down to a
    tunnel), and a street whose end stands past a junction's new height eases to it. A street too short that
    ends at another bridge or tunnel keeps the ramp's height there, and that structure's end comes to it. Every junction it sets is
    flat, and a street it changes comes level across where it changed it (tilts), as the bridge or tunnel is."""
    sign = 1.0 if up else -1.0
    todo = []

    def lift(node, y, at, came=None):
        """Junction node, standing at height `at` where the ramp reaches it, takes y if that is past it."""
        if sign * (y - at) < 0.02:
            return
        hj[node] = y
        planes.pop(node, None)
        on = None if came is None else near.straight_on(came[0], came[1], node)
        for k2, start2 in near.meeting.get(node, ()):
            if came is None or k2 != came[0]:
                todo.append((k2, start2, node, came is None or (on is not None and k2 == on[0])))

    nodes, streets = near.at(point)
    for node in nodes:
        lift(node, y, float(planes[node](point)[0]) if node in planes else hj[node])
    todo += [(k, at_start, None, True) for k, at_start in streets  # those ending at the point itself
             if not nodes or np.hypot(*(near.axis[k][0 if at_start else -1] - point)) <= AT_END]
    while todo:
        k, at_start, node, straight = todo.pop()
        y0 = y if node is None else hj[node]  # the junction's height now: a later ramp may have lifted it further
        s, h = out[k]
        dist = s if at_start else s[-1] - s
        span = min(EASE, s[-1] / 2) or 1.0
        off = y0 - float(h[0] if at_start else h[-1])
        if sign * off < 0:  # the junction came to rest short of this street's end
            w = _ease(dist / span)
            out[k] = (s, h + off * w)
            tilts[k] = tilts[k] * (1 - w)
            continue
        want = y0 - sign * (GRADE if straight else SIDE) * dist
        gap = np.maximum(want - h, 0) if up else np.minimum(want - h, 0)
        far = near.node_of.get((k, not at_start))
        left = float(gap[-1] if at_start else gap[0])
        if abs(left) >= CARRY and far not in hj and near.portals is not None:
            end = shapely.Point(near.axis[k][-1 if at_start else 0])
            if end.distance(shapely.Point(point)) > AT_END and len(near.portals.query(end, predicate="dwithin",
                                                                                      distance=AT_END)):
                out[k] = (s, h + gap)  # it ends at another bridge or tunnel, whose end takes this height (heights)
                tilts[k] = np.zeros_like(tilts[k])
                continue
        if abs(left) < CARRY or far not in hj:
            fade = np.clip((s[-1] - dist) / span, 0, 1)
            out[k] = (s, h + gap * fade)
            tilts[k] = tilts[k] * (1 - _ease(dist / span))
            continue
        out[k] = (s, h + gap)
        tilts[k] = np.zeros_like(tilts[k])
        lift(far, float(want[-1] if at_start else want[0]),
             float(h[-1] if at_start else h[0]) if far in planes else hj[far], (k, not at_start))


def _on(tris, xz, fallback):
    """The height at points xz (N, 2) of the triangles (T, 3, 3) they fall in; `fallback` where none."""
    if not len(tris):
        return np.array(fallback, float)
    a = tris[:, 0, [0, 2]]
    u, v = tris[:, 1, [0, 2]] - a, tris[:, 2, [0, 2]] - a
    det = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
    ok = np.abs(det) > 1e-9
    a, u, v, det, y = a[ok], u[ok], v[ok], det[ok], tris[ok][:, :, 1]
    # Only the pairs of a point and a triangle whose box (with a margin over the tolerance below) holds it: a
    # long highway has thousands of points and triangles, and every pair would take gigabytes.
    xy = tris[ok][:, :, [0, 2]]
    lo, hi = xy.min(axis=1) - 0.5, xy.max(axis=1) + 0.5
    pi, ti = STRtree(shapely.box(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1])).query(shapely.points(xz))
    d = xz[pi] - a[ti]
    b1 = (d[:, 0] * v[ti, 1] - d[:, 1] * v[ti, 0]) / det[ti]
    b2 = (u[ti, 0] * d[:, 1] - u[ti, 1] * d[:, 0]) / det[ti]
    inside = (b1 > -1e-4) & (b2 > -1e-4) & (b1 + b2 < 1 + 1e-4)
    pi, ti, b1, b2 = pi[inside], ti[inside], b1[inside], b2[inside]
    first = np.lexsort((ti, pi))  # a point on several triangles takes the first, as it always has
    first = first[np.r_[True, pi[first][1:] != pi[first][:-1]]] if len(first) else first
    p, t, w1, w2 = pi[first], ti[first], b1[first], b2[first]
    out = np.array(fallback, float)
    out[p] = (1 - w1 - w2) * y[t, 0] + w1 * y[t, 1] + w2 * y[t, 2]
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
    SAMPLE metres apart along a ribbon. A point of a junction takes its plane's height (or its flat height on a
    bridge, in a tunnel or where a ramp set it); a point of a ribbon takes the height of its profile at the point of
    the centreline beside it plus its distance to the left times the cross slope there. A skirt edge is a stretch
    of a piece's boundary no other piece touches, where the road stands more than CLEAR above the ground, and
    not on a bridge or in a tunnel, which contravia.structures dresses instead. A dict `keep` gets the junction
    heights, the ribbons' profiles, the junctions' planes and the ribbons' cross slopes as "hj", "prof", "planes"
    and "tilts", and as "ribbons" the ribbons as meshed: cut apart where two levels nearly touched (_apart)."""
    moved, sunk = set(), set()
    hj, prof, tilts, planes = heights(ribbons, owner, axes, rings, rnode, chains, terrain, moved, sunk)
    ribbons = _apart(ribbons, axes, [chains[o].layer for o in owner], prof, moved)
    if keep is not None:
        keep["hj"], keep["prof"], keep["planes"], keep["tilts"], keep["ribbons"] = hj, prof, planes, tilts, ribbons
    pieces_ = list(ribbons) + list(rings)
    nr = len(ribbons)
    lines = [LineString(q) for q in axes]

    raised = _Raised()  # filled once the pieces are cut (_clear_ground)

    def level(k):
        """Whether piece k is on a bridge or in a tunnel, which the ground is not drawn against."""
        return chains[owner[k]].layer if k < nr else isinstance(rnode[k - nr], tuple)

    def y_of(k, xz):
        return y_laid(k, xz) + (0.0 if level(k) else raised(xz))

    def y_laid(k, xz):
        if k >= nr:
            n = rnode[k - nr]
            return planes[n](xz) if n in planes else np.full(len(xz), hj[n])
        s, h = prof[k]
        at = shapely.line_locate_point(lines[k], shapely.points(xz))
        q = axes[k]
        foot = np.stack([np.interp(at, arclen(q), q[:, 0]), np.interp(at, arclen(q), q[:, 1])], axis=1)
        u = np.einsum("ij,ij->i", xz - foot, left_normals(q, at))
        return np.interp(at, s, h) + u * np.interp(at, s, tilts[k])

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
                    if k < nr:  # a ribbon cut along its length by a cell's edge keeps its profile on that edge
                        g = g.segmentize(SAMPLE)
                    t = shapely.get_coordinates(shapely.constrained_delaunay_triangles(g)).reshape(-1, 4, 2)[:, :3]
                    tris.append(t)
                    keys.append(np.tile((i, j), (len(t), 1)))
        if not tris:
            return np.zeros((0, 3, 3)), np.zeros((0, 2), int), poly
        t = np.concatenate(tris)
        y = y_of(k, t.reshape(-1, 2)).reshape(-1, 3)
        return np.stack([t[..., 0], y, t[..., 1]], axis=-1), np.concatenate(keys), poly

    cut = pmap(cut_cells, range(len(pieces_)), chunk=64)
    # the pieces that belong over the ground: not a ribbon a ramp took under it, nor a junction a ramp set lower
    def over(k):
        if k < nr:
            return k not in sunk
        n = rnode[k - nr]
        c = pieces_[k].representative_point()
        return n in planes or hj[n] >= _Drawn(terrain).height(np.array([c.x]), np.array([c.y]))[0]

    laid = [k for k in range(len(pieces_)) if not level(k) and over(k)]
    raised = _clear_ground(cut, [k for k in range(len(pieces_)) if not level(k)], laid, terrain)
    if keep is not None:
        keep["raised"] = raised
    tree = built_tree(pieces_)

    def skirt(k):
        if level(k):
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


def _clear_ground(cut: list, flat: list, laid: list, terrain, rounds: int = 3) -> _Raised:
    """Raise the triangles of the ground pieces (cut[k][0] for k in flat, in place) wherever the drawn ground
    would stand over those laid on it, and return the lift (_Raised). The ground is measured only under the
    pieces in laid: a ramp goes under the ground on purpose, and takes the lift of the streets round it only so
    that it still meets them. A point more than SUNK under the ground is in a cut, where the ground is left
    out, and stays. A lift is straight between the ground grid's vertices and a street's triangle between its
    own corners, so a triangle across a crease can be left a little short: the pass runs again on what it has
    raised, rounds times at most."""
    from contravia.mesh import lattice
    from contravia.rules.ground import drawn

    raised = _Raised()
    todo = [k for k in laid if len(cut[k][0])]
    laid = {k: cut[k][0][:, :, 1].copy() for k in flat if len(cut[k][0])}

    def short(k):
        pts, _ = lattice(cut[k][0], PROBE)
        need = drawn(terrain, pts[:, 0], pts[:, 2]) - pts[:, 1] + OVER
        on = (need > 0) & (need <= SUNK + OVER)
        return pts[on][:, [0, 2]], need[on]

    for _ in range(rounds):
        found = [(k, *r) for k, r in zip(todo, pmap(short, todo, chunk=64)) if len(r[1])]
        if not found:
            break
        raised.add(np.concatenate([f[1] for f in found]), np.concatenate([f[2] for f in found]))
        for k in laid:
            t = cut[k][0]
            t[:, :, 1] = laid[k] + raised(t[:, :, [0, 2]].reshape(-1, 2)).reshape(-1, 3)
        todo = [f[0] for f in found]
    return raised


def add_to(city, widths) -> dict:
    """Build the streets of the city's road graph and store them in it, cut per cell; then the height of every
    graph point, so traffic drives on the same surface. Returns counts for the report."""
    from contravia.landmarks import Landmarks
    from contravia.surface import Surface

    from contravia import structures

    for k in ("street_tris", "street_edges", "street_marks") + STRUCTURE_KINDS:
        city.a.pop(k, None)
        city.a.pop(f"{k}_offsets", None)
    ribbons, owner, axes, rings, rnode, chains = build(city.a, widths, Landmarks(city).wall_rings)
    keep = {}
    tris, tkeys, edges, ekeys, lines, mkeys = mesh(ribbons, owner, axes, rings, rnode, chains, city.terrain, keep)
    city.add_kind("street_tris", tris, tkeys)
    city.add_kind("street_edges", edges, ekeys)
    city.add_kind("street_marks", lines, mkeys)
    ribbons = keep["ribbons"]  # cut apart where two levels nearly touched (_apart)
    ground = [r for k, r in enumerate(ribbons) if not chains[owner[k]].layer]
    ground += [r for r, n in zip(rings, rnode) if not isinstance(n, tuple)]
    st, paint, skeys, walls, wkeys, cuts, ckeys, roofs, rkeys, holes = structures.build(
        ribbons, owner, axes, chains, keep["prof"], city.terrain, built_tree(ground) if ground else None,
        rings, rnode, {n: y for n, y in keep["hj"].items() if n not in keep["planes"]}, keep["tilts"])
    from contravia import medians

    mt, mp, mk, mw, mwk = medians.build(ribbons, owner, axes, chains, keep["prof"], city.a)
    st, paint, skeys = np.concatenate([st, mt]), np.concatenate([paint, mp.astype(paint.dtype)]), np.concatenate([skeys, mk])
    walls, wkeys = np.concatenate([walls, mw.astype(walls.dtype)]), np.concatenate([wkeys, mwk])
    city.add_kind("struct_tris", st, skeys)
    city.add_kind("struct_paint", paint, skeys)
    city.add_kind("struct_walls", walls, wkeys)
    city.add_kind("street_cuts", cuts, ckeys)
    city.add_kind("street_roofs", roofs, rkeys)
    city.a["sea_holes"] = holes
    poly = city.a["g_poly"].astype(float)
    city.a["g_poly_y"] = Surface(city, city.terrain).height(poly[:, 0], poly[:, 1]).astype(np.float32)
    return {"streets": len(chains), "bridges_and_tunnels": sum(1 for c in chains if c.layer),
            "structure_triangles": len(st), "structure_walls": len(walls), "cut_triangles": len(cuts),
            "median_triangles": len(mt), "median_walls": len(mw),
            "sea_hole_triangles": len(holes),
            "junction_pieces": len(rings), "street_triangles": len(tris),
            "street_skirt_edges": len(edges), "street_mark_triangles": len(lines)}


def main(path) -> dict:
    """Rebuild the streets of a prepared data file in place and print the counts."""
    import json
    import time

    from contravia.citydata import CityData
    from contravia.prepare import WIDTHS

    t0 = time.perf_counter()
    city = CityData.load(path)
    report = add_to(city, WIDTHS)
    city.save(path)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(report, indent=2))
    return report
