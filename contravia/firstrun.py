"""The first run: build the whole island from the shipped province sources (contravia.source) into the player's
home, province by province (La Habana first) and four steps each, then the island's index and summaries
(contravia.world.index). Each finished step leaves a checkpoint and each finished province its file, so a build
stopped half way goes on from the last step done.

Build.run does the work, from any thread; a screen reads its progress (step, fractions, ETA, megabytes) while
it runs. The fraction within a step is estimated from that step's usual time, scaled by how fast the steps
already done went on this machine."""

import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

from contravia.citydata import CityData

# (key, what the screen shows, seconds the step takes for La Habana on the reference machine: 12 cores, no GIL);
# another province's steps take these times scaled by the size of its source.
STEPS = (("map", "Leer el mapa", 1.0), ("buildings", "Edificios", 14.0), ("streets", "Calles y puentes", 48.0),
         ("walks", "Aceras", 25.0))
ISLAND = ("island", "Resumen de la isla", 60.0)  # the index and the summaries over every province, at the end
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


class Build:
    """The build of the island from its province sources into where. Read province, step, label, step_fraction,
    overall, eta and mb while run() goes; done is set at the end and error holds what stopped it."""

    def __init__(self, source, where: Path | None = None):
        self.source, self.where = Path(source), Path(where or home())
        self.provinces = sources(self.source)
        if not self.provinces:
            raise FileNotFoundError(f"{self.source}: no province source (<slug>.src)")
        sizes = [p.stat().st_size for _, _, p in self.provinces]
        ref = sizes[0] if self.provinces[0][0] == "La Habana" else max(sizes)
        self.factor = [s / ref for s in sizes]
        self.out, self.work = self.where / CITY, self.where / "build"
        self.p, self.step, self.done, self.error, self.report = 0, 0, False, None, {}
        self.updating = self.where.joinpath("build.json").exists() and not built(self.where)
        self._t0 = self._start = time.perf_counter()
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

    def _ref(self) -> float:
        """The current step's reference seconds."""
        return ISLAND[2] if self.p >= len(self.provinces) else STEPS[self.step][2] * self.factor[self.p]

    @property
    def step_fraction(self) -> float:
        if self.done:
            return 1.0
        return min(0.99, (time.perf_counter() - self._start) / (self._ref() * self._scale))

    @property
    def overall(self) -> float:
        if self.done:
            return 1.0
        return min(0.99, (self._ref_done + self._ref() * self.step_fraction) / self._total)

    @property
    def eta(self) -> float:
        """Seconds left, by the reference times scaled to this machine."""
        if self.done:
            return 0.0
        # max: read from another thread between a step's end and the next one's start, the sum dips below 0
        return max(0.0, (self._total - self._ref_done - self._ref() * self.step_fraction) * self._scale)

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
            self.error = f"{type(e).__name__}: {e}"
            if raises:
                raise

    def _begin(self, k: int) -> None:
        self.step, self._start = k, time.perf_counter()

    def _finish(self, k: int) -> None:
        ref = self._ref()
        self._spent += time.perf_counter() - self._start
        self._ref_spent += ref
        self._ref_done += ref
        self._scale = self._spent / self._ref_spent

    def _skip(self, k: int) -> None:
        self._ref_done += STEPS[k][2] * self.factor[self.p]
        self.step = k + 1

    def _run(self) -> None:
        from contravia.world import index

        self.out.mkdir(parents=True, exist_ok=True)
        stamp = self.where / "build.json"
        if stamp.exists() and not built(self.where):  # a city of another format: start the island over
            shutil.rmtree(self.out, ignore_errors=True)
            stamp.unlink()
            self.out.mkdir(parents=True)
        reports = {}
        for p, (name, slug, src) in enumerate(self.provinces):
            self.p = p
            target = self.out / f"{slug}.npz"
            if target.exists():  # built in an earlier run: its checkpoint is the file itself
                for k in range(len(STEPS)):
                    self._skip(k)
                continue
            reports[name] = self._province(src, self.work / slug, target)
        self.p, self._start = len(self.provinces), time.perf_counter()
        (self.out / "index.npz").unlink(missing_ok=True)
        island = index(self.out)
        self._ref_spent += ISLAND[2]
        self._spent += time.perf_counter() - self._start
        self._ref_done += ISLAND[2]
        self.report = {"island": island, "provinces": reports}
        stamp.write_text(json.dumps({"format": FORMAT, **self.report}, indent=2, default=str))
        shutil.rmtree(self.work, ignore_errors=True)
        self.done = True

    def _province(self, src: Path, work: Path, target: Path) -> dict:
        """One province's four steps, each leaving a checkpoint in work; returns its report."""
        from contravia import prepare, source

        work.mkdir(parents=True, exist_ok=True)
        base, streets = work / "buildings.npz", work / "streets.npz"
        info_p = work / "info.pkl"
        city = info = None
        self.step = 0
        if streets.exists() or base.exists():  # resume: the map and the base city are behind us
            self._skip(0), self._skip(1)
            info = pickle.loads(info_p.read_bytes())
        else:
            self._begin(0)
            raw, terrain = source.load(src)
            self._finish(0)
            self._begin(1)
            city, info = prepare.base_city(raw, terrain)
            del raw
            info_p.write_bytes(pickle.dumps(info))  # before the checkpoint, which marks the step done
            self._save(city, base)
            self._finish(1)
        report = {}
        if streets.exists():
            self._skip(2)
            report = pickle.loads((work / "streets.pkl").read_bytes())
            city = CityData.load(streets)
        else:
            self._begin(2)
            city = city or CityData.load(base)
            report = prepare.add_streets(city) if info["edges"] else {}
            self._save(city, streets)
            (work / "streets.pkl").write_bytes(pickle.dumps(report))
            base.unlink(missing_ok=True)
            self._finish(2)
        self._begin(3)
        report.update(prepare.add_walks(city) if info["edges"] else {})
        out = prepare.report(city, info, report)
        self._save(city, target)
        self._finish(3)
        shutil.rmtree(work, ignore_errors=True)
        return out

    @staticmethod
    def _save(city: CityData, path: Path) -> None:
        """Save through a temporary name, so a file that exists is whole."""
        tmp = path.with_name(path.stem + ".part.npz")
        city.save(tmp)
        os.replace(tmp, path)


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
