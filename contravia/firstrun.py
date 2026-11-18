"""The first run: build the city from the shipped source (contravia.source) into the player's home, a step at
a time. Each finished step leaves a checkpoint, so a build stopped half way goes on from the last step done.

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

# (key, what the screen shows, seconds the step takes on the reference machine: 12 cores, no GIL)
STEPS = (("map", "Leer el mapa", 1.0), ("buildings", "Edificios", 14.0), ("streets", "Calles y puentes", 48.0),
         ("walks", "Aceras", 25.0))
DONE_LABEL = "Listo"
CITY = "havana.npz"
FORMAT = 5  # the built city's format: a city of another format is built again (2: street names and barrios, 3: widths from lanes, 4: medians, 5: no OSM tree points)


def home() -> Path:
    """Where the game keeps the built city: CONTRAVIA_HOME, else the platform's per-user data folder."""
    if os.environ.get("CONTRAVIA_HOME"):
        return Path(os.environ["CONTRAVIA_HOME"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Contravia"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "contravia"


def built(where: Path | None = None) -> Path | None:
    """The built city's file, if the first run has finished with this FORMAT."""
    where = where or home()
    city, stamp = where / CITY, where / "build.json"
    if not city.exists() or not stamp.exists():
        return None
    try:
        return city if json.loads(stamp.read_text()).get("format") == FORMAT else None
    except ValueError:
        return None


class Build:
    """The build of the city from source into where. Read step, label, step_fraction, overall, eta and mb while
    run() goes; done is set at the end and error holds what stopped it."""

    def __init__(self, source, where: Path | None = None):
        self.source, self.where = Path(source), Path(where or home())
        self.work = self.where / "build"
        self.step, self.done, self.error, self.report = 0, False, None, {}
        self.updating = (self.where / CITY).exists()  # an older city is there: this build replaces it
        self._t0 = self._start = time.perf_counter()
        self._scale = 1.0  # actual over reference seconds, from the steps finished in this run
        self._spent = self._ref_spent = 0.0

    # progress
    @property
    def label(self) -> str:
        return DONE_LABEL if self.done else STEPS[self.step][1]

    @property
    def step_fraction(self) -> float:
        if self.done:
            return 1.0
        expected = STEPS[self.step][2] * self._scale
        return min(0.99, (time.perf_counter() - self._start) / expected)

    @property
    def overall(self) -> float:
        if self.done:
            return 1.0
        total = sum(s[2] for s in STEPS)
        before = sum(s[2] for s in STEPS[:self.step])
        return min(0.99, (before + STEPS[self.step][2] * self.step_fraction) / total)

    @property
    def eta(self) -> float:
        """Seconds left, by the reference times scaled to this machine."""
        if self.done:
            return 0.0
        left = STEPS[self.step][2] * (1 - self.step_fraction) + sum(s[2] for s in STEPS[self.step + 1:])
        return left * self._scale

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
        self._spent += time.perf_counter() - self._start
        self._ref_spent += STEPS[k][2]
        self._scale = self._spent / self._ref_spent

    def _skip(self, k: int) -> None:
        self.step = k + 1

    def _run(self) -> None:
        from contravia import prepare, source

        self.work.mkdir(parents=True, exist_ok=True)
        base, streets = self.work / "buildings.npz", self.work / "streets.npz"
        info_p = self.work / "info.pkl"
        city = info = None
        if streets.exists() or base.exists():  # resume: the map and the base city are behind us
            self._skip(0), self._skip(1)
            info = pickle.loads(info_p.read_bytes())
        else:
            self._begin(0)
            raw, terrain = source.load(self.source)
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
            report = pickle.loads((self.work / "streets.pkl").read_bytes())
            city = CityData.load(streets)
        else:
            self._begin(2)
            city = city or CityData.load(base)
            report = prepare.add_streets(city) if info["edges"] else {}
            self._save(city, streets)
            (self.work / "streets.pkl").write_bytes(pickle.dumps(report))
            base.unlink(missing_ok=True)
            self._finish(2)
        self._begin(3)
        report.update(prepare.add_walks(city) if info["edges"] else {})
        self.report = prepare.report(city, info, report)
        self._save(city, self.where / CITY)
        (self.where / "build.json").write_text(json.dumps({"format": FORMAT, **self.report}, indent=2))
        self._finish(3)
        shutil.rmtree(self.work, ignore_errors=True)
        self.done = True

    @staticmethod
    def _save(city: CityData, path: Path) -> None:
        """Save through a temporary name, so a file that exists is whole."""
        tmp = path.with_name(path.stem + ".part.npz")
        city.save(tmp)
        os.replace(tmp, path)


def main(source, where=None) -> dict:
    """Build the city from source, printing the progress once a second."""
    import threading

    b = Build(source, where)
    t = threading.Thread(target=b.run, daemon=True)
    t.start()
    while t.is_alive():
        t.join(1.0)
        print(f"{b.step + 1}/{len(STEPS)} {b.label:<18} {b.step_fraction:4.0%}  total {b.overall:4.0%}  "
              f"eta {b.eta:5.0f} s  {b.mb:6.1f} MB", flush=True)
    if b.error:
        raise SystemExit(b.error)
    print(f"{b.where / CITY}: {time.perf_counter() - b._t0:.0f} s")
    return b.report
