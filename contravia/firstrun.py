"""The first run: build the whole island from the shipped province sources (contravia.source) into the player's
home, province by province (La Habana first) and four steps each, then the island's index and summaries
(contravia.world.index). Each finished step leaves a checkpoint and each finished province its file, so a build
stopped half way goes on from the last step done.

Build.run does the work, from any thread; a screen reads its progress (step, fractions, ETA, megabytes) while
it runs. The fraction within a step is estimated from that step's usual time, scaled by how fast the steps
already done went on this machine."""

import json
import multiprocessing
import os
import pickle
import queue
import shutil
import sys
import time
from pathlib import Path

from contravia.citydata import CityData

# (key, what the screen shows, seconds the step takes for La Habana on the reference machine: 12 cores, no GIL);
# another province's steps take these times scaled by the size of its source.
STEPS = (("map", "Leer el mapa", 1.0), ("buildings", "Edificios", 14.0), ("streets", "Calles y puentes", 48.0),
         ("walks", "Aceras", 25.0))
ISLAND = ("island", "Resumen de la isla", 70.0)  # the index, the summaries and the GPS route graph, at the end
DONE_LABEL = "Listo"
CITY = "cuba"  # the folder of built provinces in the home
FORMAT = 7  # the built city's format: a city of another format is built again (2: street names and barrios, 3: widths from lanes, 4: medians, 5: no OSM tree points, 6: parks, 7: the whole island, by province)


def home() -> Path:
    """Where the game keeps the built city: CONTRAVIA_HOME, else the platform's per-user data folder."""
    if os.environ.get("CONTRAVIA_HOME"):
        return Path(os.environ["CONTRAVIA_HOME"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Contravia"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "contravia"


def built(where: Path | None = None) -> Path | None:
    """The built island's folder, if the first run has finished with this FORMAT."""
    where = where or home()
    city, stamp = where / CITY, where / "build.json"
    if not (city / "index.npz").exists() or not stamp.exists():
        return None
    try:
        return city if json.loads(stamp.read_text()).get("format") == FORMAT else None
    except ValueError:
        return None


def sources(folder) -> list[tuple[str, str, Path]]:
    """(name, slug, path) of each province source in folder, in the order the island is built: La Habana first."""
    from contravia.geo import PROVINCES

    return [(name, slug, Path(folder) / f"{slug}.src") for name, slug in PROVINCES
            if (Path(folder) / f"{slug}.src").exists()]


PROVINCE_GB = 4.0  # one province's build at its peak, with room: La Habana's walks take about 3 GB, the largest


def free_gb() -> float:
    """The memory the system can give now, in GB: MemAvailable on Linux, the available physical memory on
    Windows; 8 elsewhere."""
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("pf_total", ctypes.c_ulonglong),
                        ("pf_avail", ctypes.c_ulonglong), ("v_total", ctypes.c_ulonglong),
                        ("v_avail", ctypes.c_ulonglong), ("v_ext", ctypes.c_ulonglong)]

        st = Status(ctypes.sizeof(Status))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.avail / 2**30
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 8.0


def workers() -> int:
    """How many provinces to build at once: one per four cores, as many as the free memory holds, at least one."""
    return max(1, min((os.cpu_count() or 1) // 4, int(free_gb() // PROVINCE_GB)))


class Build:
    """The build of the island from its province sources into where, several provinces at once (workers()),
    each in its own process unless isolate is off. Read province, step, label, step_fraction, overall, eta and
    mb while run() goes; done is set at the end and error holds what stopped it. The province shown is the
    first one still building."""

    def __init__(self, source, where: Path | None = None, isolate: bool = True, parallel: int | None = None):
        self.source, self.where, self.isolate = Path(source), Path(where or home()), isolate
        self.provinces = sources(self.source)
        if not self.provinces:
            raise FileNotFoundError(f"{self.source}: no province source (<slug>.src)")
        self.parallel = 1 if not isolate else (parallel or workers())
        sizes = [p.stat().st_size for _, _, p in self.provinces]
        ref = sizes[0] if self.provinces[0][0] == "La Habana" else max(sizes)
        self.factor = [s / ref for s in sizes]
        self.out, self.work = self.where / CITY, self.where / "build"
        self.p, self.step, self.done, self.error, self.report = 0, 0, False, None, {}
        self.updating = self.where.joinpath("build.json").exists() and not built(self.where)
        self._t0 = time.perf_counter()
        self._running: dict[int, tuple[int, float]] = {}  # province: (its step, when that step began)
        self._scale = 1.0  # actual over reference seconds, from the steps finished in this run
        self._spent = self._ref_spent = self._ref_done = 0.0
        self._total = sum(s[2] for s in STEPS) * sum(self.factor) + ISLAND[2]

    # progress
    @property
    def province(self) -> str:
        """The province being built; the island's name during the summary."""
        return self.provinces[self.p][0] if self.p < len(self.provinces) else "Cuba"

    @property
    def label(self) -> str:
        if self.done:
            return DONE_LABEL
        return ISLAND[1] if self.p >= len(self.provinces) else STEPS[self.step][1]

    def _ref(self, p: int, k: int) -> float:
        """Step k of province p's reference seconds; the summary's when p is past the provinces."""
        return ISLAND[2] if p >= len(self.provinces) else STEPS[k][2] * self.factor[p]

    def _fraction(self, p: int) -> float:
        k, start = self._running.get(p, (self.step, time.perf_counter()))
        return min(0.99, (time.perf_counter() - start) / (self._ref(p, k) * self._scale))

    def _under_way(self) -> float:
        """Reference seconds of the running steps, by how far each has gone."""
        return sum(self._ref(p, k) * self._fraction(p) for p, (k, _) in list(self._running.items()))

    @property
    def step_fraction(self) -> float:
        if self.done:
            return 1.0
        return self._fraction(self.p)

    @property
    def overall(self) -> float:
        if self.done:
            return 1.0
        return min(0.99, (self._ref_done + self._under_way()) / self._total)

    @property
    def eta(self) -> float:
        """Seconds left, by the reference times scaled to this machine and shared among the provinces at once."""
        if self.done:
            return 0.0
        # max: read from another thread between a step's end and the next one's start, the sum dips below 0
        left = max(0.0, self._total - self._ref_done - self._under_way()) * self._scale
        return left / max(1, min(self.parallel, len(self._running) or 1))

    @property
    def mb(self) -> float:
        """Megabytes written to the home so far."""
        if not self.where.exists():
            return 0.0
        return sum(f.stat().st_size for f in self.where.rglob("*") if f.is_file()) / 1e6

    # work
    def run(self, raises: bool = True) -> None:
        """Build; on a failure keep it in error (the screen shows it) and raise it unless raises is False.
        The checkpoints stay for the next try."""
        try:
            self._run()
        except Exception as e:
            self.error = str(e) if isinstance(e, ProvinceError) else f"{type(e).__name__}: {e}"
            if raises:
                raise

    def _show(self) -> None:
        """Show the first province still building."""
        if self._running:
            self.p = min(self._running)
            self.step = self._running[self.p][0]

    def _event(self, p: int, kind: str, k: int) -> None:
        now = time.perf_counter()
        if kind == "begin":
            self._running[p] = (k, now)
        elif kind == "finish":
            ref = self._ref(p, k)
            self._spent += now - self._running[p][1]
            self._ref_spent += ref
            self._ref_done += ref
            self._scale = self._spent / self._ref_spent
            self._running[p] = (min(k + 1, len(STEPS) - 1), now)
        else:  # skip
            self._ref_done += self._ref(p, k)
            self._running[p] = (min(k + 1, len(STEPS) - 1), now)
        self._show()

    def _run(self) -> None:
        from contravia.world import index

        self.out.mkdir(parents=True, exist_ok=True)
        stamp = self.where / "build.json"
        if stamp.exists() and not built(self.where):  # a city of another format: start the island over
            shutil.rmtree(self.out, ignore_errors=True)
            stamp.unlink()
            self.out.mkdir(parents=True)
        todo = []
        for p, (_, slug, _) in enumerate(self.provinces):
            if (self.out / f"{slug}.npz").exists():  # built in an earlier run: its checkpoint is the file itself
                self._ref_done += sum(self._ref(p, k) for k in range(len(STEPS)))
            else:
                todo.append(p)
        reports = self._in_process(todo) if not self.isolate else self._in_children(todo)
        self.p, self._running = len(self.provinces), {}
        start = time.perf_counter()
        (self.out / "index.npz").unlink(missing_ok=True)
        from contravia import causeway, coast, cover, regions, shore, water

        for name in (coast.NAME, cover.NAME, shore.NAME, regions.NAME, causeway.NAME, water.NAME):  # the coast and cover the index cuts per node; the shore
            if (self.source / name).exists():
                shutil.copyfile(self.source / name, self.out / name)
        island = index(self.out)
        from contravia.gps import RouteGraph
        from contravia.world import World

        RouteGraph.for_world(World.load(self.out))  # the GPS's routing graph, beside the index
        self._ref_spent += ISLAND[2]
        self._spent += time.perf_counter() - start
        self._ref_done += ISLAND[2]
        names = [n for n, _, _ in self.provinces]
        self.report = {"island": island, "provinces": {names[p]: reports[p] for p in sorted(reports)}}
        stamp.write_text(json.dumps({"format": FORMAT, **self.report}, indent=2, default=str))
        shutil.rmtree(self.work, ignore_errors=True)
        self.done = True

    def _args(self, p: int) -> tuple[Path, Path, Path]:
        _, slug, src = self.provinces[p]
        return src, self.work / slug, self.out / f"{slug}.npz"

    def _in_process(self, todo: list[int]) -> dict:
        reports = {}
        for p in todo:
            self.p, self.step = p, 0
            reports[p] = province(*self._args(p), lambda kind, k, p=p: self._event(p, kind, k))
            self._running.pop(p, None)
        return reports

    def _in_children(self, todo: list[int]) -> dict:
        """The provinces in todo, each in a child process, self.parallel at once: the memory a province takes
        goes back to the system when it ends, so the peak is that of the largest few, not the island's sum."""
        ctx = multiprocessing.get_context("spawn")
        events = ctx.Queue()
        children, reports, waiting = {}, {}, list(todo)
        try:
            while waiting or children:
                while waiting and len(children) < self.parallel:
                    p = waiting.pop(0)
                    children[p] = ctx.Process(target=_child, args=(p, *self._args(p), events), daemon=True)
                    children[p].start()
                    self._running[p] = (0, time.perf_counter())
                    self._show()
                try:
                    p, kind, value = events.get(timeout=0.5)
                except queue.Empty:
                    for p, child in list(children.items()):
                        if not child.is_alive() and events.empty():
                            raise ProvinceError(f"{self.provinces[p][0]}: the build stopped "
                                                f"(exit code {child.exitcode})") from None
                    continue
                if kind == "report":
                    reports[p] = value
                    children.pop(p).join()
                    self._running.pop(p, None)
                    self._show()
                elif kind == "error":
                    self.p = p
                    raise ProvinceError(value)
                else:
                    self._event(p, kind, value)
        finally:  # on a failure the others stop; their checkpoints stay for the next try
            for child in children.values():
                child.terminate()
                child.join()
        return reports


class ProvinceError(Exception):
    """A province's build failed in its child process; the message is the child's error."""


def province(src: Path, work: Path, target: Path, step) -> dict:
    """One province's four steps, each leaving a checkpoint in work; returns its report. step(kind, k) is told
    when step k begins, finishes or is skipped."""
    from contravia import prepare, source

    work.mkdir(parents=True, exist_ok=True)
    base, streets = work / "buildings.npz", work / "streets.npz"
    info_p = work / "info.pkl"
    city = info = None
    if streets.exists() or base.exists():  # resume: the map and the base city are behind us
        step("skip", 0), step("skip", 1)
        info = pickle.loads(info_p.read_bytes())
    else:
        step("begin", 0)
        raw, terrain = source.load(src)
        step("finish", 0)
        step("begin", 1)
        city, info = prepare.base_city(raw, terrain)
        del raw
        info_p.write_bytes(pickle.dumps(info))  # before the checkpoint, which marks the step done
        _save(city, base)
        step("finish", 1)
    report = {}
    if streets.exists():
        step("skip", 2)
        report = pickle.loads((work / "streets.pkl").read_bytes())
        city = CityData.load(streets)
    else:
        step("begin", 2)
        city = city or CityData.load(base)
        report = prepare.add_streets(city) if info["edges"] else {}
        _save(city, streets)
        (work / "streets.pkl").write_bytes(pickle.dumps(report))
        base.unlink(missing_ok=True)
        step("finish", 2)
    step("begin", 3)
    report.update(prepare.add_walks(city) if info["edges"] else {})
    out = prepare.report(city, info, report)
    _save(city, target)
    step("finish", 3)
    shutil.rmtree(work, ignore_errors=True)
    return out


def _save(city: CityData, path: Path) -> None:
    """Save through a temporary name, so a file that exists is whole."""
    tmp = path.with_name(path.stem + ".part.npz")
    city.save(tmp)
    os.replace(tmp, path)


def _child(p: int, src: Path, work: Path, target: Path, events) -> None:
    """Province p built in a child process, its steps, report or error sent through events."""
    try:
        events.put((p, "report", province(src, work, target, lambda kind, k: events.put((p, kind, k)))))
    except Exception as e:  # noqa: BLE001 - any failure goes to the parent, which shows it
        events.put((p, "error", f"{type(e).__name__}: {e}"))


def main(source, where=None) -> dict:
    """Build the island from its province sources, printing the progress once a second."""
    import threading

    b = Build(source, where)
    t = threading.Thread(target=b.run, daemon=True)
    t.start()
    while t.is_alive():
        t.join(1.0)
        print(f"{b.province:<20} {b.step + 1}/{len(STEPS)} {b.label:<18} {b.step_fraction:4.0%}  total {b.overall:4.0%}  "
              f"eta {b.eta:5.0f} s  {b.mb:6.1f} MB", flush=True)
    if b.error:
        raise SystemExit(b.error)
    print(f"{b.out}: {time.perf_counter() - b._t0:.0f} s")
    return b.report
