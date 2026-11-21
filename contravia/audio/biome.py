"""The biome under the car: what the sound and the music of a place are, from what the game knows.

A region of contravia/regions.toml may name the biome (or only the music) of a place. Otherwise
the sun decides night, the sea within 300 m the coast, a built-up square in a dense municipio the
city, a less built one the barrio, and the rest is countryside. A new biome takes over only
after the car has spent HOLD seconds in it, so a short trip through a park changes nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contravia.rules.sound import BIOMES

HOLD = 10.0  # seconds in a new biome before it takes over
NIGHT_SUN = -6.0  # degrees: below this the biome is night
SEA_NEAR = 300.0  # metres
CITY_BUILT, BARRIO_BUILT = 0.5, 0.15  # shares of built-up land
BUILT_UP = 50  # ESA WorldCover's class for built-up land


@dataclass
class Place:
    """What decides the biome at a point: built-up share, metres to the sea (or more than 300),
    the municipio's zone (walks.zone_of: 0 old centre, 1 Vedado-like, 2 other), the sun's
    altitude in degrees, and a region's biome and music, if any."""

    built: float
    sea_m: float
    zone: int
    sun_alt: float
    biome: str | None = None
    music: str | None = None


def resolve(p: Place) -> str:
    """The biome of a place, by the rules in this module's docstring."""
    if p.biome:
        return p.biome
    if p.sun_alt < NIGHT_SUN:
        return "night"
    if p.sea_m <= SEA_NEAR:
        return "coast"
    if p.built >= CITY_BUILT and p.zone in (0, 1):
        return "city"
    if p.built >= BARRIO_BUILT:
        return "barrio"
    return "countryside"


class Biome:
    """The biome heard, which follows the biome of the place after HOLD seconds."""

    def __init__(self, hold: float = HOLD):
        self.hold, self.current, self.candidate, self.since = hold, None, None, 0.0

    def update(self, now: float, candidate: str) -> str:
        if self.current is None:
            self.current = candidate
        if candidate == self.current:
            self.candidate = None
        elif candidate != self.candidate:
            self.candidate, self.since = candidate, now
        elif now - self.since >= self.hold:
            self.current, self.candidate = candidate, None
        return self.current


class Sampler:
    """Reads a Place for a point of the island: WorldCover and the coast from the island's folder,
    loaded once; without them the place reads as unbuilt and inland, and says so once."""

    def __init__(self, folder):
        self.folder, self._cover, self._land, self._warned = folder, None, None, False

    def _load(self) -> None:
        if self._cover is not None or self.folder is None:
            return
        from contravia import coast, cover

        try:
            self._cover = cover.load(self.folder / "cover.src")
        except (OSError, ValueError, KeyError) as e:
            self._cover = False
            self._warn(f"no cover for the sound's biome: {e}")
        try:
            import shapely

            land = coast.load(self.folder / coast.NAME)
            shapely.prepare(land)
            self._land = land
        except (OSError, ValueError, KeyError) as e:
            self._land = False
            self._warn(f"no coast for the sound's biome: {e}")

    def _warn(self, why: str) -> None:
        if not self._warned:
            self._warned = True
            print(f"contravia.audio: {why}")

    def built(self, x: float, z: float) -> float:
        self._load()
        if not self._cover:
            return 0.0
        g = np.arange(-2, 3) * 50.0
        xs, zs = np.meshgrid(x + g, z + g)
        return float(np.mean(np.asarray(self._cover.sample(xs.ravel(), zs.ravel())) == BUILT_UP))

    def sea_m(self, x: float, z: float) -> float:
        self._load()
        if not self._land:
            return 1000.0
        import shapely

        a = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        off = ~shapely.contains_xy(self._land, x + SEA_NEAR * np.cos(a), z + SEA_NEAR * np.sin(a))
        return SEA_NEAR if off.any() else 1000.0


def sample(game, sampler: Sampler) -> Place:
    """The Place where the game's car is now."""
    from contravia import regions
    from contravia.walks import zone_of

    car = game.car
    sun = getattr(game, "sun_now", None)
    area = getattr(game, "area", None) or None
    try:
        params = regions.current().at(car.x, car.z, area)
        biome, music = params.audio_biome, params.audio_music
    except (OSError, ValueError):  # no regions file, or one that does not load: the place decides alone
        biome = music = None
    return Place(sampler.built(car.x, car.z), sampler.sea_m(car.x, car.z), zone_of(area),
                 sun.alt if sun is not None else 45.0, biome, music)


__all__ = ["BIOMES", "Biome", "Place", "Sampler", "resolve", "sample"]
