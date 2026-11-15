"""Havana's weather, hour by hour, from the chain havana climate learnt from thirty years of ERA5.

The weather has five states: clear, partly cloudy, overcast, rain and storm. Each month and hour of the day
has its own table of how likely each state is to follow each other, so June's afternoons fill with storms
and February stays clear. A day's weather is drawn once, when first asked for, from random numbers seeded by
its date: the same date always has the same weather. The run starts three days earlier, so a day continues
the weather of the night before.

Within an hour the cloud and rain slide from one state to the next over ten minutes, and roads stay wet for
about an hour after the rain stops. SkyConfig holds the settings a menu can change while the game runs.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from havana.climate import SKY_DATA

STATES = ("clear", "partly", "overcast", "rain", "storm")
LEAD_DAYS = 3
BLEND_MIN = 10.0
DRY_HOURS = 1.0


@dataclass(frozen=True)
class SkyConfig:
    """What a player or an agent may change about the sky; every field can change while the game runs.

    rain_factor and cloud_factor bend each month's chain toward rain or cloud (1 = Havana's climate). force
    holds a state until force_until. star_limit pins the faintest star shown (None: from city glow, moon,
    twilight and cloud). city_glow scales the city's light at night: 0 is a blackout.
    """

    seed: int = 0
    rain_factor: tuple = (1.0,) * 12
    cloud_factor: tuple = (1.0,) * 12
    force: str | None = None
    force_until: datetime | None = None
    star_limit: float | None = None
    city_glow: float = 1.0
    extra: dict = field(default_factory=dict, compare=False, repr=False)

    def updated(self, **changes) -> "SkyConfig":
        """A copy with `changes`; ValueError names a key that is unknown or a value out of range."""
        names = {f.name for f in dataclasses.fields(self)} - {"extra"}
        out = {}
        for k, v in changes.items():
            if k not in names:
                raise ValueError(f"unknown sky setting: {k}")
            if k in ("rain_factor", "cloud_factor"):
                v = (float(v),) * 12 if isinstance(v, (int, float)) else tuple(float(x) for x in v)
                if len(v) != 12 or min(v) < 0:
                    raise ValueError(f"{k} must be one number or 12, none negative")
            elif k == "force" and v is not None and v not in STATES:
                raise ValueError(f"force must be one of {', '.join(STATES)}")
            elif k == "force_until" and isinstance(v, str):
                v = datetime.fromisoformat(v)
            elif k in ("star_limit", "city_glow") and v is not None:
                v = float(v)
                if k == "city_glow" and v < 0:
                    raise ValueError("city_glow must not be negative")
            elif k == "seed":
                v = int(v)
            out[k] = v
        return dataclasses.replace(self, **out)


def load_sky_config(path: str) -> SkyConfig:
    """Settings from a JSON file holding only the keys to change."""
    return SkyConfig().updated(**json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class Now:
    """The weather at one moment: state name, cloud cover 0-1, rain intensity 0-1 and road wetness 0-1."""

    state: str
    cloud: float
    rain: float
    wet: float


def load_table(path: Path | None = None) -> dict:
    """The chain built by havana climate."""
    return json.loads((path or SKY_DATA / "climate.json").read_text())


class Weather:
    """The weather chain, drawn day by day and cached."""

    def __init__(self, table: dict | None = None, config: SkyConfig | None = None):
        self.table = table or load_table()
        self.T = np.array(self.table["transitions"])  # (12, 24, 5, 5)
        self.start = np.array(self.table["start"])
        self.cloud_mean = np.array(self.table["cloud"])
        self.rain_mean = np.array(self.table["rain"])
        self.config = config or SkyConfig()
        self._days: dict[date, np.ndarray] = {}

    def configure(self, config: SkyConfig) -> None:
        """New settings; days drawn with the old ones are drawn again."""
        if (config.seed, config.rain_factor, config.cloud_factor) != (
                self.config.seed, self.config.rain_factor, self.config.cloud_factor):
            self._days.clear()
        self.config = config

    def _matrix(self, month: int, hour: int) -> np.ndarray:
        m = self.T[month, hour].copy()
        rf, cf = self.config.rain_factor[month], self.config.cloud_factor[month]
        if rf != 1.0 or cf != 1.0:
            m *= np.array([1.0, cf, cf, rf * cf, rf * cf])
            m /= m.sum(axis=1, keepdims=True)
        return m

    def _uniforms(self, d: date) -> np.ndarray:
        """The 24 random numbers of date d: every run through d uses the same ones."""
        return np.random.default_rng([d.toordinal(), self.config.seed]).random(24)

    def day(self, d: date) -> np.ndarray:
        """The 24 hourly states of date d (indices into STATES).

        The run starts LEAD_DAYS earlier from the month's usual state at midnight. Every run that passes
        through a day uses that day's own random numbers, by inverse sampling, so two runs starting on
        different days fall onto the same states within hours: the run for d agrees with the run for d - 1
        on the hours they share, and the weather has no seam at midnight.
        """
        if d not in self._days:
            first = d - timedelta(days=LEAD_DAYS)
            p = np.cumsum(self.start[first.month - 1])
            s = int(np.searchsorted(p / p[-1], self._uniforms(first - timedelta(days=1))[23]))
            out = np.zeros(24, int)
            for k in range(LEAD_DAYS + 1):
                dd = first + timedelta(days=k)
                u = self._uniforms(dd)
                for h in range(24):
                    if k == LEAD_DAYS:
                        out[h] = s
                    c = np.cumsum(self._matrix(dd.month - 1, h)[s])
                    s = min(4, int(np.searchsorted(c / c[-1], u[h])))
            self._days[d] = out
            if len(self._days) > 16:
                self._days.pop(next(iter(self._days)))
        return self._days[d]

    def state_at(self, t: datetime) -> int:
        """The state at the hour holding t, or the forced one."""
        c = self.config
        if c.force and (c.force_until is None or t < c.force_until):
            return STATES.index(c.force)
        return int(self.day(t.date())[t.hour])

    def at(self, t: datetime) -> Now:
        """The weather at t, blended over the first minutes of an hour from the previous hour's state."""
        s = self.state_at(t)
        prev = self.state_at(t - timedelta(hours=1))
        m = t.month - 1
        a = min(1.0, (t.minute + t.second / 60.0) / BLEND_MIN)
        cloud = (1 - a) * self.cloud_mean[m, prev] + a * self.cloud_mean[m, s]
        rain = (1 - a) * self.rain_mean[m, prev] + a * self.rain_mean[m, s]
        cloud = min(1.0, cloud * self.config.cloud_factor[m] ** 0.5)
        # Roads dry over DRY_HOURS after the last rain hour.
        wet = 0.0
        for k in range(4):
            if self.state_at(t - timedelta(hours=k)) >= 3:
                since = 0.0 if k == 0 else k - 1 + (t.minute / 60.0)
                wet = max(0.0, 1.0 - since / DRY_HOURS)
                break
        return Now(STATES[s], float(cloud), float(rain), float(wet))
