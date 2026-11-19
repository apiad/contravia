"""How much traffic Havana has, by hour and day: every number is a setting.

The defaults are an estimate from published figures (spec section 07): 1 814 207 people (ONEI 2023), 2.3 trips
per person per day and the modal split from the city's sustainable mobility plan (PMUS 2022). People per
vehicle, the share of shared taxis, buses and freight are assumptions. The hour curves have the peaks reported
for Havana (7–9 and 16–18); the rest of their shape, and the day totals, are assumptions too.

The demand gives how many vehicles are moving; the mix splits them over the fleet's ten kinds, and the family
shares split a kind over its families (fleet spec section 05). Havana has one row of the mix until the streets carry
place classes. A near vehicle's look (kind, family, variant and colourway) is drawn from those shares by a seeded
hash, so the same street at the same minute draws the same vehicles.

Change any of them with TrafficConfig.updated(...) while the game runs, and the flow layer follows.
"""

import dataclasses
import functools
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from contravia.rules.fleet import FAMILIES, KINDS, variant

MODES = ("car", "taxi", "bus", "moto", "truck")  # what the demand estimate counts trips of
DAY_TYPES = ("weekday", "friday", "saturday", "sunday")

# Percent of a day's trips starting in each hour, 00–01 first. Normalised when a config is made.
HOURS = {
    "weekday": (0.8, 0.5, 0.4, 0.3, 0.5, 1.5, 4.0, 7.5, 8.5, 6.5, 5.5, 5.5,
                5.8, 5.8, 5.5, 6.0, 7.5, 8.0, 6.5, 4.5, 3.5, 2.6, 1.8, 1.0),
    "friday": (0.8, 0.5, 0.4, 0.3, 0.5, 1.5, 4.0, 7.3, 8.2, 6.3, 5.3, 5.3,
               5.6, 5.6, 5.3, 5.8, 7.2, 7.6, 6.4, 4.8, 3.9, 3.1, 2.4, 1.5),
    "saturday": (0.9, 0.6, 0.5, 0.4, 0.4, 0.8, 2.0, 3.8, 5.2, 6.5, 7.6, 7.9,
                 7.8, 7.0, 6.2, 6.0, 6.0, 6.1, 5.8, 4.9, 3.9, 3.1, 2.4, 1.5),
    "sunday": (0.9, 0.7, 0.6, 0.4, 0.3, 0.5, 1.2, 2.2, 3.5, 5.0, 6.5, 7.2,
               7.5, 7.4, 7.0, 6.8, 6.8, 6.7, 6.2, 5.4, 4.3, 3.4, 2.5, 1.5),
}
DAYS = {"weekday": 1.00, "friday": 1.05, "saturday": 0.80, "sunday": 0.60}
MODAL_SPLIT = {"walk": 0.462, "public": 0.436, "car": 0.060, "moto": 0.032, "bike": 0.011}
OCCUPANCY = {"car": 1.5, "moto": 1.2, "taxi": 4.0}
ATTRACT = {"Plaza de la Revolución": 1.5, "La Lisa": 1.3}
# Percent of the vehicles on Havana's streets by kind: the spec's `havana` row.
MIX = {"car": 26, "taxi": 12, "bus": 6, "truck": 8, "moto": 30, "tricycle": 7, "bicitaxi": 3, "bike": 6,
       "police": 1.5, "fire": 0.5}
# Percent of a kind by family in Havana: the spec's "Inside a kind", an era's share split evenly over its families.
FAMILY_SHARE = {
    "almendron": 25, "ruso": 20, "moscovita": 20, "moderno": 17.5, "huevito": 17.5,  # 1950s 25 · 80s 40 · modern 35
    "compartido": 75, "estatal": 25,
    "metroguagua": 80, "guagua_vieja": 20, "interprovincial": 0,
    "furgon": 40, "camion_80": 30, "cisterna": 15, "volquete": 15, "pasajeros": 0,
    "moto_electrica": 85, "moto_dos_tiempos": 15,
    "triciclo_carga": 50, "triciclo_pasajeros": 50,
    "bicitaxi": 100, "bicicleta": 100, "policia": 100, "bomberos": 100,
}


def _normalised(values: dict) -> dict:
    total = sum(values.values())
    return {k: v / total for k, v in values.items()} if total > 0 else dict(values)


@dataclass(frozen=True)
class TrafficConfig:
    """The demand settings. Make a changed copy with updated(); never mutate one."""

    population: float = 1_814_207
    trips_per_person: float = 2.3
    modal_split: dict = field(default_factory=lambda: dict(MODAL_SPLIT))
    occupancy: dict = field(default_factory=lambda: dict(OCCUPANCY))
    shared_taxi_share: float = 0.25
    bus_trips: float = 5000
    freight_share: float = 0.25
    trip_minutes: float = 15
    hours: dict = field(default_factory=lambda: {k: tuple(v) for k, v in HOURS.items()})
    days: dict = field(default_factory=lambda: dict(DAYS))
    attract: dict = field(default_factory=lambda: dict(ATTRACT))
    mix: dict = field(default_factory=lambda: dict(MIX))
    families: dict = field(default_factory=lambda: dict(FAMILY_SHARE))
    level: float = 1.0

    def __post_init__(self):
        for name in ("population", "trips_per_person", "shared_taxi_share", "bus_trips", "freight_share",
                     "trip_minutes", "level"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a number ≥ 0, got {value!r}")
        if self.trip_minutes == 0:
            raise ValueError("trip_minutes must be more than 0")
        for name, known in (("mix", KINDS), ("families", FAMILIES)):
            unknown = sorted(set(getattr(self, name)) - set(known))
            if unknown:
                raise ValueError(f"{name} has no {', '.join(unknown)}; it takes {', '.join(known)}")
        if sum(self.mix.values()) <= 0:
            raise ValueError("mix needs a share above 0")
        for name in ("modal_split", "occupancy", "days", "attract", "mix", "families"):
            if any(not isinstance(v, (int, float)) or v < 0 for v in getattr(self, name).values()):
                raise ValueError(f"{name} values must be numbers ≥ 0")
        if any(self.occupancy.get(k, 0) <= 0 for k in OCCUPANCY):
            raise ValueError(f"occupancy needs {', '.join(OCCUPANCY)} above 0")
        for day in DAY_TYPES:
            if day not in self.hours or day not in self.days:
                raise ValueError(f"hours and days need every day type: {', '.join(DAY_TYPES)}")
        hours = {}
        for day, curve in self.hours.items():
            curve = tuple(curve)
            if len(curve) != 24 or any(not isinstance(v, (int, float)) or v < 0 for v in curve):
                raise ValueError(f"hours[{day!r}] must be 24 numbers ≥ 0")
            total = sum(curve)
            hours[day] = tuple(v / total for v in curve) if total > 0 else curve
        object.__setattr__(self, "hours", hours)
        object.__setattr__(self, "modal_split", _normalised(self.modal_split))

    def updated(self, **changes) -> TrafficConfig:
        """A copy with these settings changed. Dict settings merge key by key; an unknown name is an error."""
        names = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(set(changes) - names)
        if unknown:
            raise ValueError(f"unknown traffic setting: {', '.join(unknown)}")
        merged = {}
        for name, value in changes.items():
            old = getattr(self, name)
            merged[name] = {**old, **value} if isinstance(old, dict) and isinstance(value, dict) else value
        return dataclasses.replace(self, **merged)


def load_config(path) -> TrafficConfig:
    """The defaults with the settings a JSON file names."""
    with open(path) as f:
        return TrafficConfig().updated(**json.load(f))


def as_dict(cfg: TrafficConfig) -> dict:
    """The settings as plain data, for engine.state and JSON."""
    return dataclasses.asdict(cfg)


def day_type(when: datetime) -> str:
    """Monday to Thursday are one type; Friday, Saturday and Sunday each have their own."""
    return {4: "friday", 5: "saturday", 6: "sunday"}.get(when.weekday(), "weekday")


def _base_trips(cfg: TrafficConfig) -> dict:
    """Vehicle trips in a day whose total factor is 1, by mode of the demand estimate."""
    people = cfg.population * cfg.trips_per_person
    split, occ = cfg.modal_split, cfg.occupancy
    trips = {
        "car": people * split.get("car", 0) / occ["car"],
        "taxi": people * split.get("public", 0) * cfg.shared_taxi_share / occ["taxi"],
        "bus": cfg.bus_trips,
        "moto": people * split.get("moto", 0) / occ["moto"],
    }
    trips["truck"] = cfg.freight_share * sum(trips.values())
    return {k: trips[k] for k in MODES}


def vehicle_trips(cfg: TrafficConfig, day: str) -> dict:
    """Vehicle trips in a day of this type, by mode of the demand estimate."""
    return {k: v * cfg.days[day] for k, v in _base_trips(cfg).items()}


def _hour_value(cfg: TrafficConfig, day_start: datetime, hour: int) -> float:
    """The day-weighted share of hour `hour` counted from midnight of `day_start` (hour 24 is the next day's 0)."""
    when = day_start + timedelta(hours=hour)
    d = day_type(when)
    return cfg.days[d] * cfg.hours[d][when.hour]


def hour_share(cfg: TrafficConfig, when: datetime) -> float:
    """Trips starting per hour at `when`, as a share of a day whose total factor is 1 (the day's own factor is included).

    The value of each hour sits at its midpoint and is interpolated linearly between midpoints, across
    midnight into the next day's curve, so demand never jumps on the hour.
    """
    midnight = when.replace(hour=0, minute=0, second=0, microsecond=0)
    t = (when - midnight).total_seconds() / 3600 - 0.5
    h = int(t // 1)
    f = t - h
    return (1 - f) * _hour_value(cfg, midnight, h) + f * _hour_value(cfg, midnight, h + 1)


def moving(cfg: TrafficConfig, when: datetime) -> dict:
    """Vehicles on the streets at `when`, by kind: trips starting per hour times the hours a trip lasts, split by the mix."""
    total = sum(_base_trips(cfg).values()) * hour_share(cfg, when) * cfg.trip_minutes / 60 * cfg.level
    mix = sum(cfg.mix.values())
    return {k: total * cfg.mix.get(k, 0) / mix for k in KINDS}


@functools.cache
def looks() -> tuple[tuple[str, str, int, int], ...]:
    """Every look the traffic can draw, as (kind, family, variant, colourway), kind after kind in KINDS order."""
    out = []
    for kind in KINDS:
        for family in (f for f in FAMILIES.values() if f.kind == kind):
            for k in range(family.variants):
                out += [(kind, family.name, k, j) for j in range(len(variant(kind, family.name, k).colourways))]
    return tuple(out)


def look_kinds() -> np.ndarray:
    """The KINDS index of every look."""
    return np.array([KINDS.index(look[0]) for look in looks()], np.int64)


def look_shares(cfg: TrafficConfig) -> np.ndarray:
    """The probability of each look: its kind's share of the mix, its family's share of the kind, then even."""
    return _shares(tuple(sorted(cfg.mix.items())), tuple(sorted(cfg.families.items())))


@functools.cache
def _shares(mix_items: tuple, family_items: tuple) -> np.ndarray:
    every, mix_of, share = looks(), dict(mix_items), dict(family_items)
    mix = sum(mix_of.values())
    per_kind = {kind: sum(share.get(f.name, 0) for f in FAMILIES.values() if f.kind == kind) for kind in KINDS}
    counts = {}
    for kind, family, _k, _j in every:
        counts[family] = counts.get(family, 0) + 1
    p = np.zeros(len(every))
    for i, (kind, family, _k, _j) in enumerate(every):
        if per_kind[kind] > 0:
            p[i] = mix_of.get(kind, 0) / mix * share.get(family, 0) / per_kind[kind] / counts[family]
    return p / p.sum()


def _hash(*parts) -> int:
    """A stable 64-bit seed from numbers, the same on every machine."""
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def pick(cfg: TrafficConfig, edge: int, when: datetime, n: int) -> np.ndarray:
    """The looks of `n` vehicles entering `edge` at `when`'s minute: a seeded hash against the look shares."""
    u = np.random.default_rng(_hash(edge, when.strftime("%Y-%m-%d %H:%M"))).random(n)
    return np.minimum(np.searchsorted(np.cumsum(look_shares(cfg)), u, side="right"), len(looks()) - 1).astype(np.int64)
