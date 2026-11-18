"""The Garaje outside its screens: the player's saved designs, and building a vehicle without stopping the frame.

Designs are kept one JSON file each in the `garaje` folder of the user's configuration folder (settings.config_dir),
newest first in Mis diseños. They never drive in the city's traffic.

A car takes seconds to build, longer than a frame, so the Builder builds on one worker thread and the screens ask
it each frame whether the vehicle they want is ready; until it is, the game keeps showing the one before. Built
vehicles are cached by their design, so going back to a preset or undoing a change costs nothing.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from contravia.rules.designs import Design, build, fix
from contravia.settings import config_dir

CACHED = 12  # built vehicles kept


def designs_dir() -> Path:
    return config_dir() / "garaje"


def slug(name: str) -> str:
    """A file name for a design's name: lower case ASCII, words joined by hyphens."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "diseno"


def load_designs() -> list[Design]:
    """The saved designs, newest first. A file out of today's ranges comes back fixed; an unreadable one is
    skipped."""
    out = []
    folder = designs_dir()
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append(fix(Design.from_json(json.loads(path.read_text()))))
        except (OSError, ValueError, KeyError):
            continue
    return out


def save_design(design: Design) -> Path:
    """Write a design under its name's slug, replacing a design of the same name."""
    folder = designs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug(design.name)}.json"
    path.write_text(json.dumps({**design.to_json(), "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2,
                               ensure_ascii=False))
    return path


def delete_design(design: Design) -> None:
    (designs_dir() / f"{slug(design.name)}.json").unlink(missing_ok=True)


def new_name(base: str, taken: list[str]) -> str:
    """"Mi <base> <n>" with the first n no saved design uses."""
    n = 1
    while f"Mi {base.lower()} {n}" in taken:
        n += 1
    return f"Mi {base.lower()} {n}"


def key(design: Design) -> str:
    """What identifies a built vehicle: the parameters only, since colours are a palette, not geometry."""
    d = design.to_json()
    return json.dumps({"base": d["base"], "changes": d["changes"],
                       "livery": d["livery"], "second": d["second"] if d["livery"] else None}, sort_keys=True)


class Builder:
    """Builds vehicles on one worker thread, keeping the last CACHED by their design."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="garaje")
        self.cache: dict[str, object] = {}
        self.jobs: dict[str, Future] = {}

    def want(self, design: Design):
        """The design's Vehicle if it is built, else None after starting (or continuing) its build. A build that
        fails raises here, once."""
        k = key(design)
        if k in self.cache:
            self.cache[k] = self.cache.pop(k)  # most recent last
            return self.cache[k]
        job = self.jobs.get(k)
        if job is None:
            self.jobs[k] = self.pool.submit(build, design)
            return None
        if not job.done():
            return None
        del self.jobs[k]
        vehicle = job.result()
        self.cache[k] = vehicle
        while len(self.cache) > CACHED:
            self.cache.pop(next(iter(self.cache)))
        return vehicle

    def now(self, design: Design):
        """The design's Vehicle, built here and now if it has to be."""
        k = key(design)
        if k not in self.cache:
            job = self.jobs.pop(k, None)
            self.cache[k] = job.result() if job else build(design)
        return self.cache[k]

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
