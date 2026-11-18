"""Where the car is, said as a habanero would say it: the street, between which streets (or at which corner),
and the barrio and municipio. The graph carries the parts per edge (contravia.graph.addresses); this module
finds the edge under the car and puts the words together."""

import math
import re

import numpy as np

CORNER = 15.0  # metres from a crossing within which the car is at the corner
ALIGNED = 0.7  # |cos| of the angle between the car and an edge that counts as driving along it
CROSSWISE = 10.0  # metres added to an edge's distance when the car is not driving along it


def locate(graph, x: float, z: float, yaw: float, reach: float = 150.0):
    """(edge, metres along it, whether the car faces the edge's direction) of the edge the car drives on: the
    nearest by distance to its line, an edge the car crosses counting CROSSWISE metres farther. None if no
    edge is near."""
    a = graph.a
    near = graph.edges_near(x, z, reach)
    if not len(near):
        return None
    poly, starts = a["g_poly"], a["g_poly_start"]
    heading = np.array([math.cos(yaw), math.sin(yaw)])
    best = None
    for e in near.tolist():
        p = poly[starts[e]:starts[e + 1]].astype(float)
        d = np.diff(p, axis=0)
        seg = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        t = np.clip(((x - p[:-1, 0]) * d[:, 0] + (z - p[:-1, 1]) * d[:, 1]) / seg**2, 0, 1)
        q = p[:-1] + d * t[:, None]
        dist = np.hypot(q[:, 0] - x, q[:, 1] - z)
        k = int(np.argmin(dist))
        cos = float(d[k] @ heading / seg[k])
        score = dist[k] + (0.0 if abs(cos) >= ALIGNED else CROSSWISE)
        if best is None or score < best[0]:
            along = float(seg[:k].sum() + seg[k] * t[k])
            best = (score, e, along, cos >= 0)
    return best[1:]


def short(name: str) -> str:
    """A street's name as said after "entre": "Calle 23" is "23", "Avenida de los Presidentes (Calle G)" is
    "Avenida de los Presidentes"."""
    return re.sub(r"\s*\(.*\)$", "", name).removeprefix("Calle ")


def _and(a: str, b: str) -> str:
    """"a y b", with "e" before a word that starts with the sound i."""
    return f"{a} {'e' if b.lower().startswith(('i', 'hi')) and not b.lower().startswith(('hie', 'hia')) else 'y'} {b}"


def describe(a: dict, edge: int, along: float, forward: bool) -> tuple[str, str, str]:
    """(street, the line under it, "barrio · municipio") for a car `along` metres down edge, facing its way or
    not. The line is "entre L y M", "esquina a L" within CORNER metres of a crossing, "desde L" or "hasta M"
    when only one is known, or empty."""
    names = a["g_names"]
    n = int(a["g_name"][edge])
    street = re.sub(r"\s*\(.*\)$", "", str(names[n])) if n >= 0 else ""
    behind, ahead = (int(v) for v in a["g_between"][edge])
    length = float(a["g_len"][edge])
    if not forward:
        behind, ahead, along = ahead, behind, length - along
    b = short(str(names[behind])) if behind >= 0 else ""
    f = short(str(names[ahead])) if ahead >= 0 else ""
    if not street:
        line = ""
    elif f and length - along < CORNER:
        line = f"esquina a {f}"
    elif b and along < CORNER:
        line = f"esquina a {b}"
    elif b and f and b != f:
        line = f"entre {_and(b, f)}"
    elif b:
        line = f"desde {b}"
    elif f:
        line = f"hasta {f}"
    else:
        line = ""
    parts = []
    k = int(a["g_barrio"][edge]) if "g_barrio" in a else -1
    if k >= 0:
        parts.append(str(a["g_barrio_names"][k]))
    m = int(a["g_muni"][edge]) if "g_muni" in a else -1
    if 0 <= m < len(a.get("g_muni_names", ())):
        parts.append(str(a["g_muni_names"][m]))
    return street, line, " · ".join(dict.fromkeys(parts))
