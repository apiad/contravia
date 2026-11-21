"""Rain, thunder and the sea.

Rain is a graph read from weather.rain (0 to 1), car.wet and car.speed: a field of drops and the
hiss of tyres on a wet road. Thunder is drawn in the game process by Thunder, which decides when
a storm strikes and how far, and fires a one-shot when the sound would arrive. A car that drives
into the sea makes a splash, then the world is muffled and bubbles rise while it sinks.
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
    WhiteNoise,
)

from contravia.audio.car import Quiet, Timed

SOUND = 343.0  # m/s
MEAN_GAP = 25.0  # seconds between strikes in a storm, on average
NEAR, FAR = 300.0, 6000.0  # metres: how close and how far a strike may be


class Drops(Unit):
    """Raindrops: sparse clicks at a rate that follows the rain, each a short burst. Its spectrum is
    broad, a crackle over the hiss."""

    def __init__(self, level: Unit, seed: int):
        self.level, self.seed = level, seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        super().reset()
        self.rng = np.random.default_rng(self.seed)

    def block(self, n: int) -> np.ndarray:
        lv = self.level.block(n)
        hits = self.rng.random(n) < 0.004 * lv
        return (hits * self.rng.uniform(-1, 1, n)).astype(np.float32)


def rain(ctl) -> Unit:
    """Rain around the car: a hiss and drops that grow with the rain, and spray from wet tyres."""
    lv = Share(Live(lambda: min(max(ctl("weather.rain"), 0.0), 1.0), start=0.0))
    hiss = HighPass(PinkNoise(seed=51), 1200.0) * (lv.tap() * 0.12)
    drops = BandPass(Drops(lv.tap(), seed=52), 3000.0, 0.6) * 0.5
    spray = Live(lambda: min(abs(ctl("car.speed")) / 25.0, 1.0) * min(max(ctl("car.wet"), 0.0), 1.0) * 0.08)
    wet = lambda: max(ctl("weather.rain"), ctl("car.wet") * min(abs(ctl("car.speed")), 1.0))
    return Quiet(hiss + drops + HighPass(WhiteNoise(seed=53), 2500.0) * spray, wet)


class Thunder:
    """When a storm strikes and how far: one strike every MEAN_GAP seconds on average, between NEAR
    and FAR metres, heard distance / 343 seconds after it strikes. Seeded."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.next: float | None = None
        self.pending: list[tuple[float, float]] = []  # (heard at, distance)
        self.struck: list[float] = []  # when each strike happened, for tests and the sky

    def due(self, now: float, state: str) -> list[float]:
        """The distances of the strikes heard by `now`, in order."""
        if state == "storm":
            if self.next is None:
                self.next = now + self.rng.exponential(MEAN_GAP)
            while self.next <= now:
                d = float(self.rng.uniform(NEAR, FAR))
                self.pending.append((self.next + d / SOUND, d))
                self.struck.append(self.next)
                self.next += self.rng.exponential(MEAN_GAP)
        else:
            self.next = None
        out = [d for t, d in self.pending if t <= now]
        self.pending = [(t, d) for t, d in self.pending if t > now]
        return out


def thunder(distance: float, seed: int) -> Timed:
    """A strike heard from `distance` metres: a crack that is shorter and duller far away, then a
    rumble that rolls for seconds."""
    near = float(np.clip(1 - (distance - NEAR) / (FAR - NEAR), 0.0, 1.0))
    crack_cut = 600.0 + 5000.0 * near
    crack = LowPass(WhiteNoise(seed=seed), crack_cut) * ADSR(0.002, 0.15 + 0.2 * near, 0.0, 0.0, gate=0.01, curve=0.6)
    length = 3.0 + 3.0 * (1 - near)
    rumble = LowPass(BrownNoise(seed=seed + 1), 160.0) * ADSR(0.3, length, 0.0, 0.0, gate=0.31, curve=0.4)
    level = 0.3 + 0.6 * near
    return Timed((crack * (0.8 * near + 0.05) + rumble * 2.5) * level, float(min(8.0, length + 0.6)))


def splash(seed: int) -> Timed:
    """The car hitting the water: a burst of noise falling into a low gurgle."""
    burst = BandPass(WhiteNoise(seed=seed), 900.0, 0.5) * ADSR(0.005, 0.5, 0.0, 0.0, gate=0.02, curve=0.6)
    body = LowPass(BrownNoise(seed=seed + 1), 300.0) * ADSR(0.05, 1.0, 0.0, 0.0, gate=0.1, curve=0.5)
    return Timed(burst * 0.8 + body * 1.5, 1.6)


def bubbles(ctl) -> Unit:
    """Bubbles while car.under is 1: drops of a low resonance, heard through the muffled master."""
    under = Share(Live(lambda: 1.0 if ctl("car.under") > 0.5 else 0.0, start=0.0))
    return Quiet(BandPass(Drops(under.tap(), seed=61), 450.0, 4.0) * (under.tap() * 2.0), lambda: ctl("car.under"))
