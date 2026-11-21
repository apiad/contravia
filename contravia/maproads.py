"""The map's roads as lines over the relief, a fixed width in screen pixels (docs/superpowers/specs/2026-11-20-map-v2-
design.html, section 06): manifoldx's engine.lines, depth-tested with a bias, so a road is never cut by the coarse
ground under it and a sierra in front still hides it.

Two kinds, each a dark casing under a light fill. The main roads are the root summary's (the island's widest, about
4 800 km), read once. The streets are the roads of the far views drawn now, from STREET_LEVELS, gathered again on a
thread when the drawn nodes change. Widths and alpha follow the camera's height through each layer's look, so a
zoom uploads nothing; below FADE_TO the cells' own streets take over and the lines are gone. Only the map draws them.
"""

import math
import threading

import numpy as np

STREET_LEVELS = (2, 8)  # the far views whose roads are the streets layer
LIFT = 1.0  # metres over the ground under each end
FADE_FROM, FADE_TO = 2_000.0, 1_200.0  # metres of camera height: the lines fade out between these
STREETS_FROM, STREETS_TO = 150_000.0, 80_000.0  # and the streets fade in between these, coming down
MAJOR = 12.0  # metres: a road this wide or wider is drawn as a main road
CASING = 1.6  # screen pixels a casing is wider than its fill


def _linear(hex_colour: str, alpha: float = 1.0) -> tuple:
    """An sRGB hex colour as linear RGBA, as engine.lines takes it."""
    c = [int(hex_colour[k:k + 2], 16) / 255 for k in (1, 3, 5)]
    return (*(v ** 2.2 for v in c), alpha)


FILL = {"main": _linear("#f2c14e"), "streets": _linear("#f4efe4")}  # mustard for main roads, paper for streets
DARK = _linear("#0b0b0b", 0.85)
WIDTH = {"main": 2.0, "streets": 1.1}  # fill pixels at scale 1
KEYS = ("streets-casing", "streets", "main-casing", "main")  # drawing order: streets under main roads


def scale(h: float) -> float:
    """Widths' factor with the camera h metres up: 0.7 from 1 000 km, 1 at 200 km, 1.5 at 20 km, 2.2 at 2 km."""
    stops = ((3.3, 2.2), (4.3, 1.5), (5.3, 1.0), (6.0, 0.7))
    return float(np.interp(math.log10(max(h, 1.0)), [s for s, _ in stops], [v for _, v in stops]))


def alphas(h: float) -> tuple[float, float]:
    """(main roads, streets) alpha with the camera h metres up."""
    down = min(1.0, max(0.0, (h - FADE_TO) / (FADE_FROM - FADE_TO)))
    streets = min(1.0, max(0.0, (STREETS_FROM - h) / (STREETS_FROM - STREETS_TO)))
    return down, down * streets


def segments(roads: np.ndarray, terrain, lift: float = LIFT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(a, b, width) of summary roads (rows ax, az, bx, bz, width) in world metres, each end on the ground."""
    from contravia.summary import _heights

    r = np.asarray(roads, float).reshape(-1, 5)
    ya = _heights(r[:, 0:2], terrain, True) + lift
    yb = _heights(r[:, 2:4], terrain, True) + lift
    return np.c_[r[:, 0], ya, r[:, 1]], np.c_[r[:, 2], yb, r[:, 3]], r[:, 4]


class MapRoads:
    """The map's road layers in engine.lines, kept in step with the camera and the drawn far views."""

    def __init__(self, engine, world, root: tuple[int, int, int]):
        self.engine, self.world, self.root = engine, world, root
        self.lock = threading.Lock()
        self.ready: dict[str, tuple] = {}  # kind -> (a, b) world metres, built on a thread, not set yet
        self.world_ab: dict[str, tuple] = {}  # kind -> (a, b) set now, to set again when the origin moves
        self.origin = None
        self.nodes = None  # the drawn nodes the streets were gathered for
        self.busy = False
        self.shown = False
        for k in KEYS:  # set now, empty, so they draw in this order whichever is ready first
            engine.lines.set(k, np.zeros((0, 3)), np.zeros((0, 3)), DARK, 1.0)
            engine.lines.look(k, alpha=0.0)
        threading.Thread(target=self._main, daemon=True, name="maproads-main").start()

    def _main(self) -> None:
        s = self.world.summary(*self.root)
        roads = np.asarray(s.get("roads", np.zeros((0, 5))), float).reshape(-1, 5)
        a, b, _ = segments(roads, self.world.terrain)
        with self.lock:
            self.ready["main"] = (a, b)

    def _streets(self, nodes) -> None:
        parts = []
        for L, a, b in nodes:
            r = self.world.summary(L, a, b).get("roads")
            if r is not None and len(r):
                parts.append(np.asarray(r, float).reshape(-1, 5))
        roads = np.concatenate(parts) if parts else np.zeros((0, 5))
        roads = roads[roads[:, 4] < MAJOR]  # the main roads have their own layer
        sa, sb, _ = segments(roads, self.world.terrain)
        with self.lock:
            self.ready["streets"] = (sa, sb)
            self.busy = False

    def update(self, on: bool, h: float, drawn) -> None:
        """Show the layers when on (the map), with widths and alpha for a camera h metres up; gather the
        streets again when the drawn far views in STREET_LEVELS changed."""
        lines = self.engine.lines
        if not on:
            if self.shown:
                for k in KEYS:
                    lines.look(k, alpha=0.0)
                self.shown = False
            return
        nodes = frozenset(n for n in drawn if STREET_LEVELS[0] <= n[0] <= STREET_LEVELS[1])
        if nodes != self.nodes and not self.busy and alphas(h)[1] > 0:
            self.nodes, self.busy = nodes, True
            threading.Thread(target=self._streets, args=(sorted(nodes),), daemon=True, name="maproads-streets").start()
        origin = tuple(self.engine.origin)
        with self.lock:
            fresh, self.ready = self.ready, {}
        if origin != self.origin:  # the render frame moved: every layer again
            fresh = {**self.world_ab, **fresh}
            self.origin = origin
        o = np.array([origin[0], 0.0, origin[1]])
        for kind, (a, b) in fresh.items():
            self.world_ab[kind] = (a, b)
            ra, rb = a - o, b - o
            lines.set(f"{kind}-casing", ra, rb, DARK, WIDTH[kind] + CASING)
            lines.set(kind, ra, rb, FILL[kind], WIDTH[kind])
        k = scale(h)
        main, streets = alphas(h)
        for kind, alpha in (("streets", streets), ("main", main)):
            lines.look(f"{kind}-casing", scale=k, alpha=alpha)
            lines.look(kind, scale=k, alpha=alpha)
        self.shown = True
