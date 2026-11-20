"""What lies under the map's crosshair: the province, the municipio and the street, for contravia/nearby.py.

The province comes from the world's cells (World.province_at). The municipio and the street come from the road
graph of that province, as the HUD's address does while driving (contravia/address.py), but the map may be over
any province, not only the one the car drives in: so only the arrays an address needs are read from the
province's file the first time the crosshair reaches it (a tenth of a second for La Habana), and kept.

The map must not wait for any of it (province_at alone takes 30 ms over the sea), so the map screen asks with
`ask` and picks up the answer with `take`: one worker thread answers the newest question and drops the older
ones, and stops after IDLE seconds without one.
"""

import threading
import traceback

import numpy as np

from contravia import nearby

KEYS = ("g_from", "g_mid", "g_poly", "g_poly_start", "g_name", "g_names", "g_between", "g_len", "g_barrio",
        "g_barrio_names", "g_muni", "g_muni_names")
MUNI_REACH = 3_000.0  # metres: past this from any street the crosshair is in no municipio
STREET_REACH = 150.0  # metres: the street under the crosshair is the one within this, as address.locate finds it
IDLE = 5.0  # seconds the worker waits for a question before it stops


class Where:
    """The lookups of one world, with each province's address arrays read once."""

    def __init__(self, world):
        self.world = world
        self._graphs: dict[int, object] = {}  # province -> its address Graph, or None when it has no graph
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._wanted = None  # the newest question: (x, z, h, places)
        self._answer = None  # the newest answer not taken yet: a nearby.Here
        self._worker: threading.Thread | None = None

    def _graph(self, k: int):
        """Province k's address graph, read now the first time; None when it has none."""
        if k not in self._graphs:
            from contravia.graph import Graph

            path = self.world.paths[k] if hasattr(self.world, "paths") else None
            try:
                if path is None:  # a world made in memory: its arrays are already there
                    a = self.world.province(k).a
                    arrays = {key: a[key] for key in KEYS if key in a}
                else:
                    with np.load(path) as f:
                        arrays = {key: f[key] for key in KEYS if key in f.files}
                self._graphs[k] = Graph(arrays) if "g_muni" in arrays and len(arrays["g_mid"]) else None
            except (OSError, KeyError, ValueError):
                self._graphs[k] = None
        return self._graphs[k]

    def at(self, x: float, z: float, detail: int) -> nearby.Spot:
        """What lies at (x, z): the province; with detail 1 also its municipio and barrio; with detail 2 also
        the street under it and the line that places it ("entre L y M")."""
        k = self.world.province_at(x, z)
        if k is None:
            return nearby.Spot()
        province = str(self.world.names[k])
        g = self._graph(k) if detail else None
        if g is None:
            return nearby.Spot(province=province)
        a = g.a
        d = np.hypot(a["g_mid"][:, 0] - x, a["g_mid"][:, 1] - z)
        e = int(np.argmin(d))
        if d[e] > MUNI_REACH:
            return nearby.Spot(province=province)
        m = int(a["g_muni"][e])
        municipio = str(a["g_muni_names"][m]) if 0 <= m < len(a["g_muni_names"]) else ""
        barrio = _barrio(a, e)
        if detail < 2 or "g_name" not in a:
            return nearby.Spot(province=province, municipio=municipio, barrio=barrio)
        from contravia.address import describe, locate

        found = locate(g, x, z, 0.0, STREET_REACH)
        if found is not None and a["g_name"][found[0]] < 0:  # a nameless service road: the nearest named street
            named = np.where(a["g_name"] >= 0, d, np.inf)
            n = int(np.argmin(named))
            found = (n, float(a["g_len"][n]) / 2, True) if named[n] <= STREET_REACH else None
        if found is None:
            return nearby.Spot(province=province, municipio=municipio, barrio=barrio)
        street, line, _ = describe(a, *found)
        return nearby.Spot(province=province, municipio=municipio, street=street, line=line,
                           barrio=_barrio(a, found[0]) or barrio)

    def here(self, places, x: float, z: float, h: float) -> nearby.Here:
        """The panel for the crosshair at (x, z) with the camera h metres up, looked up now."""
        name = nearby.band(h).name
        detail = {"isla": 0, "calle": 2}.get(name, 1)
        return nearby.choose(places, self.at(x, z, detail), x, z, h)

    def ask(self, places, x: float, z: float, h: float) -> None:
        """Ask for the panel at (x, z, h) without waiting; `take` gives it once the worker has it."""
        with self._lock:
            self._wanted = (places, x, z, h)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, daemon=True, name="whereis")
                self._worker.start()
        self._wake.set()

    def take(self) -> nearby.Here | None:
        """The newest answer, once; None when there is no new one."""
        with self._lock:
            answer, self._answer = self._answer, None
        return answer

    def _work(self) -> None:
        while True:
            self._wake.wait(IDLE)
            with self._lock:
                self._wake.clear()
                wanted, self._wanted = self._wanted, None
                if wanted is None:
                    self._worker = None
                    return
            try:
                answer = self.here(*wanted)
            except (OSError, KeyError, IndexError, ValueError):  # a lookup that fails leaves the panel as it was, and says why in the log
                traceback.print_exc()
                continue
            with self._lock:
                self._answer = answer


def _barrio(a: dict, e: int) -> str:
    b = int(a["g_barrio"][e]) if "g_barrio" in a else -1
    return str(a["g_barrio_names"][b]) if 0 <= b < len(a["g_barrio_names"]) else ""
