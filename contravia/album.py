"""The player's collection: which places they have photographed, when, and where the photo is.

It lives in the configuration folder beside the options and the Garaje (settings.config_dir): `coleccion.json`
maps a place's id to the real date the photo was taken, the game's day and time, its weather and the PNG's name,
and the PNGs are in `fotos/`. Ids no longer in the list of places are ignored, so a new list never breaks an old
collection; a file that cannot be read counts as empty and is left alone until the next photo is added.
"""

import json
import os
from datetime import date
from pathlib import Path

from contravia.settings import config_dir

FILE = "coleccion.json"
PHOTOS = "fotos"


class Album:
    """The photographed places, for a list of places given as {id: province}."""

    def __init__(self, provinces: dict[str, str], entries: dict[str, dict]):
        self.provinces = provinces
        self.entries = {k: v for k, v in entries.items() if k in provinces}
        self._others = {k: v for k, v in entries.items() if k not in provinces}  # kept in the file, not shown

    @classmethod
    def load(cls, provinces: dict[str, str]) -> Album:
        """The collection kept in the configuration folder, or an empty one."""
        try:
            entries = json.loads((config_dir() / FILE).read_text(encoding="utf-8"))
            if not isinstance(entries, dict):
                entries = {}
        except (OSError, ValueError):
            entries = {}
        return cls(provinces, entries)

    def has(self, pid: str) -> bool:
        return pid in self.entries

    def entry(self, pid: str) -> dict:
        return self.entries[pid]

    def count(self, province: str) -> int:
        """How many places of a province are in the collection."""
        return sum(self.provinces[k] == province for k in self.entries)

    def photo_path(self, pid: str) -> Path:
        """Where the photo of a place is, or goes."""
        return config_dir() / PHOTOS / f"{pid}.png"

    def add(self, pid: str, game_time: str, weather: str) -> None:
        """Put a place in the collection, with today's date and the game's time and weather, and save it."""
        self.entries[pid] = {"date": date.today().isoformat(), "game_time": game_time, "weather": weather,
                             "photo": f"{pid}.png"}
        path = config_dir() / FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._others | self.entries, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
