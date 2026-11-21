"""The traffic heard around the car.

The traffic layer moves cars without engines, so the four loudest at the listener get a cheap
engine each, fed from their speed and acceleration through a virtual gearbox; the rest of the
cars near the car are one bed of noise that thickens with their number. A car that has waited
behind a stopped player honks once per wait, and a guagua that stops hisses its air brakes.
"""

from __future__ import annotations

import numpy as np
from euterpe.sound import (
    ADSR,
    BandPass,
    BrownNoise,
    HighPass,
    Live,
    LowPass,
    PinkNoise,
    Share,
    Unit,
    Wavetable,
    WhiteNoise,
)

from contravia.audio.car import HORNS, Quiet, Timed, _horn_kind
from contravia.rules.sound import of

SLOTS = 4
HEAVY = ("bus", "truck", "fire")
WEIGHT = {"bus": 2.0, "truck": 2.0, "fire": 2.0, "moto": 1.3}
WAIT_HONK = 4.0  # seconds a car waits behind the player before it honks
BED_NEAR = 80.0  # metres: the cars counted into the bed
ENGINE = {"car": (800.0, 6000.0, 25.0), "heavy": (550.0, 2600.0, 12.0), "moto": (1400.0, 9000.0, 20.0)}  # idle, red, km/h a gear


def loudest(xyz: np.ndarray, speed: np.ndarray, kinds: np.ndarray, listener: np.ndarray, k: int = SLOTS) -> np.ndarray:
    """The indices of the k cars heard loudest: near, fast and heavy first."""
    if not len(xyz):
        return np.zeros(0, np.int64)
    d = np.linalg.norm(xyz - listener, axis=1)
    w = np.array([WEIGHT.get(str(kd), 1.0) for kd in kinds])
    score = np.minimum(1.0, 5.0 / np.maximum(d, 1e-3)) * (0.3 + np.abs(speed) / 20.0) * w
    return np.argsort(-score, kind="stable")[:k]


def _engine_class(kind: str) -> str:
    return "heavy" if kind in HEAVY else "moto" if kind == "moto" else "car"


def virtual_rpm(speed: float, acc: float, kind: str) -> float:
    """An rpm for a car with no engine: a gear every 25 km/h (12 for heavy vehicles), idle to 70 % of
    the redline within a gear, and 15 % of the redline more when it accelerates."""
    idle, red, gear_kmh = ENGINE[_engine_class(kind)]
    kmh = abs(speed) * 3.6
    within = (kmh % gear_kmh) / gear_kmh
    return float(idle + within * (0.7 * red - idle) + 0.15 * red * min(max(acc / 2.0, 0.0), 1.0))


def _pulse(size: int = 512) -> np.ndarray:
    x = np.exp(-np.arange(size) / size * 6) * np.sin(np.pi * np.minimum(np.arange(size) / size * 3, 1))
    return x - x.mean()


def cheap_engine(slot: int, ctl) -> Unit:
    """One traffic car's engine: a pulse at the firing frequency through a low-pass, read from the
    controls traffic.N.rpm, .cyl, .bright and .on, so the slot never rebuilds when its car changes."""
    p = f"traffic.{slot}."
    fire = Live(lambda: ctl(p + "rpm") * max(ctl(p + "cyl"), 1.0) / 120.0, start=0.0)
    on = Share(Live(lambda: 1.0 if ctl(p + "on") > 0.5 else 0.0, start=0.0))
    # A fixed low-pass: a moving cutoff would redesign the filter many times a block, for four cars.
    return Quiet(LowPass(Wavetable(_pulse(), fire), 1200.0) * (on.tap() * 0.5), lambda: ctl(p + "on"))


def bed(ctl) -> Unit:
    """The cars beyond the four: a low roar that grows with how many are within BED_NEAR metres."""
    lv = Share(Live(lambda: min(max(ctl("traffic.count"), 0.0) / 20.0, 1.0), start=0.0))
    roar = LowPass(PinkNoise(seed=71), 500.0) * 0.12 + LowPass(BrownNoise(seed=72), 150.0) * 0.3
    return Quiet(roar * lv.tap(), lambda: ctl("traffic.count"))


class Honks:
    """Which waiting cars honk: one that has waited WAIT_HONK seconds near a stopped player, once per wait."""

    def __init__(self):
        self.done = np.zeros(0, bool)

    def due(self, wait: np.ndarray, near_player: np.ndarray) -> np.ndarray:
        n = len(wait)
        if len(self.done) != n:
            self.done = np.resize(self.done, n) if n else np.zeros(0, bool)
        self.done &= wait > 0
        now = (wait > WAIT_HONK) & near_player & ~self.done
        self.done |= now
        return np.flatnonzero(now)


def honk(family: str, seed: int) -> Timed:
    """A traffic car's horn, half a second, in its family's voice."""
    pitches, wave = HORNS[_horn_kind(family)]
    tone = None
    for f in pitches:
        tone = wave(f) if tone is None else tone + wave(f)
    gate = 0.25 + 0.3 * np.random.default_rng(seed).random()
    return Timed(BandPass(tone, pitches[0] * 2.0, 0.8) * ADSR(0.01, 0.05, 0.8, 0.05, gate=gate) * 0.3, gate + 0.1)


def brakes(seed: int) -> Timed:
    """A guagua's air brakes letting go: a hiss that fades."""
    hiss = HighPass(WhiteNoise(seed=seed), 3000.0) * ADSR(0.01, 0.6, 0.0, 0.0, gate=0.02, curve=0.5)
    return Timed(hiss * 0.4, 0.8)


def slot_numbers(family: str) -> tuple[float, float]:
    """A slot's cylinders and brightness for a car of this family."""
    s = of(family)
    return float(s.cylinders or 1), float(s.bright)
