"""The ambience of each biome: a bed that plays while the biome is heard, and one-shots scattered
around the listener (birds, a rooster at dawn, a dog, a tres phrase from a window at night).

Each bed reads its level from the control amb.<biome>, which the game moves over a few seconds
when the biome changes, so the beds cross instead of cutting.
"""

from __future__ import annotations

import numpy as np
from euterpe.sound import (
    ADSR,
    FM,
    BandPass,
    BrownNoise,
    HighPass,
    Live,
    LowPass,
    PinkNoise,
    Share,
    Sine,
    Unit,
)

from contravia.audio.car import Quiet, Timed

ONESHOTS = {  # biome -> (name, mean seconds between, first hour, last hour)
    "city": (("dog", 40.0, 0, 24),),
    "barrio": (("dog", 25.0, 0, 24), ("rooster", 30.0, 5.5, 7.0)),
    "countryside": (("bird", 6.0, 5.5, 19.0), ("rooster", 20.0, 5.5, 7.5), ("cow", 50.0, 6, 19), ("dog", 60.0, 0, 24)),
    "coast": (("gull", 15.0, 6, 19),),
    "night": (("dog", 45.0, 0, 24), ("tres", 70.0, 0, 24)),
}


def _swell(rate: float, depth: float) -> Unit:
    return Sine(rate) * depth + (1 - depth)


def bed(biome: str, ctl) -> Unit:
    """The biome's continuous sound, scaled by the control amb.<biome>."""
    lv = Share(Live(lambda: min(max(ctl(f"amb.{biome}"), 0.0), 1.0), start=0.0))
    if biome == "city":
        x = BandPass(PinkNoise(seed=81), 450.0, 0.8) * _swell(0.13, 0.3) * 0.5 \
            + BandPass(PinkNoise(seed=82), 1300.0, 1.2) * _swell(0.31, 0.5) * 0.15
    elif biome == "barrio":
        x = BandPass(PinkNoise(seed=83), 600.0, 1.0) * _swell(0.21, 0.6) * 0.3
    elif biome == "countryside":
        x = HighPass(PinkNoise(seed=84), 1800.0) * _swell(0.09, 0.6) * 0.12
    elif biome == "coast":
        waves = LowPass(BrownNoise(seed=85), 700.0) * (_swell(1 / 8, 0.8) * _swell(1 / 13, 0.3)) * 0.8
        x = waves + HighPass(PinkNoise(seed=86), 3000.0) * _swell(1 / 8, 0.9) * 0.05
    else:  # night: crickets in pulses, and frogs far off
        chirp = Sine(4600.0) * (Sine(28.0) * 0.5 + 0.5) * (Sine(0.9) * 0.5 + 0.5)
        frogs = Sine(310.0) * (Sine(6.0) * 0.5 + 0.5) * _swell(0.17, 1.0)
        x = chirp * 0.06 + frogs * 0.04
    return Quiet(x * lv.tap(), lambda: ctl(f"amb.{biome}"))


class Scatter:
    """When a one-shot of the ambience plays, and from which bearing: each kind of the biome at
    its own mean interval, within its hours. Seeded."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.next: dict[str, float] = {}

    def due(self, now: float, biome: str, hour: float) -> list[tuple[str, float]]:
        out = []
        for name, gap, h0, h1 in ONESHOTS.get(biome, ()):
            if not h0 <= hour < h1:
                self.next.pop(name, None)
                continue
            t = self.next.setdefault(name, now + float(self.rng.exponential(gap)))
            if now >= t:
                out.append((name, float(self.rng.uniform(0, 2 * np.pi))))
                self.next[name] = now + float(self.rng.exponential(gap))
        return out


def oneshot(kind: str, seed: int) -> Timed:
    """One sound of the ambience by kind: bird, rooster, dog, tres, gull or cow."""
    name = kind
    rng = np.random.default_rng(seed)
    if name == "bird":
        f = 2500.0 + 1500.0 * rng.random()
        u = FM(f, f * 1.5, 2.0) * ADSR(0.005, 0.12, 0.0, 0.0, gate=0.02) * 0.2
        return Timed(u, 0.3)
    if name == "rooster":
        glide = Sine(5.0) * 40.0 + 520.0
        u = BandPass(FM(glide, 260.0, 1.5), 1100.0, 0.8) * ADSR(0.05, 0.3, 0.7, 0.4, gate=1.1) * 0.25
        return Timed(u, 1.7)
    if name == "dog":
        bark = BandPass(PinkNoise(seed=seed) + Sine(420.0) * 0.6, 700.0, 1.5)
        return Timed(bark * ADSR(0.005, 0.12, 0.0, 0.0, gate=0.03, curve=0.5) * 0.6, 0.35)
    if name == "gull":
        cry = FM(Sine(3.0) * 200.0 + 1400.0, 700.0, 1.0) * ADSR(0.02, 0.3, 0.0, 0.0, gate=0.25) * 0.15
        return Timed(cry, 0.7)
    if name == "cow":
        moo = LowPass(FM(Sine(0.8) * 15.0 + 140.0, 140.0, 1.2), 900.0) * ADSR(0.2, 0.4, 0.8, 0.5, gate=1.4) * 0.3
        return Timed(moo, 2.2)
    if name == "tres":
        from euterpe.instruments import Tres

        root = 196.0 * 2 ** (int(rng.integers(0, 5)) / 12)
        notes = [(0.0, root, 0.3), (0.25, root * 1.26, 0.3), (0.5, root * 1.5, 0.3), (0.75, root * 2, 0.5)]
        return Timed(LowPass(Tres(seed=seed).play(notes), 1800.0) * 0.5, 2.0)
    raise ValueError(f"no ambience one-shot named {name!r}")
