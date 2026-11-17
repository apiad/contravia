"""Small road graphs built by hand for the traffic tests: a straight road and a four-way junction."""

import numpy as np

from contravia.graph import CLASSES, Graph


def make_graph(junctions, edges, cls="residential", control=None, lanes=1, kmh=40.0, uturns=False) -> Graph:
    """A Graph over these junction positions and (from, to) edges, every turn allowed except U-turns.

    control: {junction index: 1 signal, 2 stop, 3 give_way} for the edges ending there.
    """
    control = control or {}
    j = np.asarray(junctions, np.float64)
    E = len(edges)
    a = {"g_from": np.array([e[0] for e in edges], np.int32), "g_to": np.array([e[1] for e in edges], np.int32)}
    a["g_poly"] = np.concatenate([j[[f, t]] for f, t in edges]).astype(np.float32)
    a["g_poly_start"] = np.arange(0, 2 * E + 1, 2, dtype=np.int64)
    a["g_len"] = np.hypot(*(j[a["g_to"]] - j[a["g_from"]]).T).astype(np.float32)
    a["g_lanes"] = np.full(E, lanes, np.int8)
    a["g_vfree"] = np.full(E, kmh / 3.6, np.float32)
    a["g_class"] = np.full(E, CLASSES.index(cls), np.int8)
    a["g_control"] = np.array([control.get(t, 0) for _, t in edges], np.int8)
    a["g_junction"] = j.astype(np.float32)
    a["g_junction_node"] = np.arange(len(j), dtype=np.int64)
    a["g_way"] = np.array([min(f, t) * 1000 + max(f, t) for f, t in edges], np.int64)
    a["g_mid"] = ((j[a["g_from"]] + j[a["g_to"]]) / 2).astype(np.float32)
    a["g_built"] = np.ones(E, np.float32)
    a["g_muni"] = np.full(E, -1, np.int16)
    a["g_muni_names"] = np.array([], dtype=str)
    t_in, t_out = [], []
    for e, (f, t) in enumerate(edges):
        for o, (f2, t2) in enumerate(edges):
            if f2 == t and (uturns or t2 != f):
                t_in.append(e)
                t_out.append(o)
    a["g_turn_in"], a["g_turn_out"] = np.array(t_in, np.int32), np.array(t_out, np.int32)
    a["g_turn_w"] = np.ones(len(t_in), np.float32)
    return Graph(a)


def straight(length=400.0) -> Graph:
    """Two edges east, 0 → 1 → 2, with a dead end turned around at each end so cars can keep going."""
    return make_graph([(0, 0), (length / 2, 0), (length, 0)], [(0, 1), (1, 2), (2, 1), (1, 0)], uturns=True)


def four_way(arm=150.0, control=None) -> Graph:
    """A cross: junction 0 at the origin, arms to the east (1), south (2), west (3) and north (4), both ways."""
    j = [(0, 0), (arm, 0), (0, arm), (-arm, 0), (0, -arm)]
    edges = []
    for k in (1, 2, 3, 4):
        edges += [(k, 0), (0, k)]
    return make_graph(j, edges, control=control)
