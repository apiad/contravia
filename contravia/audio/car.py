"""The car's other sounds: tyres rolling and squealing, wind, the horn, and one-shots for a crash,
a bump and a bicycle's bell.

The continuous ones are Euterpe graphs read from controls the game writes every frame: car.speed,
car.slip (how far the worst wheel is past its tyre's peak slip, 0 below it), car.wet, car.surface
(0 street, 1 pavement, 2 grass or earth, 3 sand, 4 rock), car.horn (1 while H is held) and
car.cabin (1 with the camera in the cabin). The one-shots are built from an event's values.
"""

from __future__ import annotations

import numpy as np
from euterpe.sound import (
    ADSR,
    BandPass,
    HighPass,
    Live,
    LowPass,
    Modal,
    PinkNoise,
    Saw,
    Share,
    Sine,
    Square,
    Unit,
    WhiteNoise,
)

CRASH_MIN = 300.0  # N·s of contact impulse a crash needs to be heard
GLASS_MIN = 3000.0  # N·s from which a crash breaks glass

# Per surface (car.surface): the noise's low-pass in Hz and its level at 30 m/s.
SURFACES = ((500.0, 0.05), (900.0, 0.07), (1800.0, 0.16), (5000.0, 0.10), (2500.0, 0.22))


def _live(ctl, name: str, f=lambda v: v) -> Live:
    return Live(lambda: f(ctl(name)), start=f(ctl(name)))


class Timed(Unit):
    """A one-shot that lasts `seconds`: its source's samples, and a duration the mixer can read."""

    def __init__(self, src: Unit, seconds: float):
        self.src, self.seconds = src, seconds
        self.channels = src.channels

    @property
    def duration(self) -> float:
        return self.seconds

    def block(self, n: int) -> np.ndarray:
        return self.src.block(n)


class Quiet(Unit):
    """Its source, or silence without asking the source while level() stays at 0: a horn nobody
    presses costs nothing. Its spectrum is its source's."""

    def __init__(self, src: Unit, level):
        self.src, self.level, self.was = src, level, 1.0
        self.channels = src.channels

    def block(self, n: int) -> np.ndarray:
        lv = self.level()
        if lv <= 0.0 and self.was <= 0.0:
            return np.zeros((n, 2) if self.channels == 2 else n, np.float32)
        self.was = lv
        return self.src.block(n)


def roll(ctl) -> Unit:
    """The tyres on the ground: noise darker on asphalt, brighter and louder off it, rising with speed."""
    speed = Share(_live(ctl, "car.speed", lambda v: min(abs(v) / 30.0, 1.5)))
    out = None
    for k, (cut, level) in enumerate(SURFACES):
        on = _live(ctl, "car.surface", lambda v, k=k: 1.0 if round(v) == k else 0.0)
        part = Quiet(LowPass(PinkNoise(seed=11 + k), cut) * (speed.tap() * on * level),
                     lambda k=k: float(round(ctl("car.surface")) == k and abs(ctl("car.speed")) > 0.1))
        out = part if out is None else out + part
    return out


def squeal(ctl) -> Unit:
    """A tyre past its grip: a wobbling band of noise near 1 kHz, louder the further past, duller when wet."""
    slip = _live(ctl, "car.slip", lambda v: min(max(v, 0.0) * 2.0, 1.0))
    dry = _live(ctl, "car.wet", lambda v: 1.0 - 0.6 * min(max(v, 0.0), 1.0))
    centre = Sine(5.3) * 150.0 + 1050.0
    return Quiet(BandPass(WhiteNoise(seed=21), centre, 6.0) * (slip * dry * 0.6), lambda: ctl("car.slip"))


def wind(ctl) -> Unit:
    """The air past the car: band noise that rises in pitch and level with speed, muffled in the cabin."""
    # Two fixed bands crossed by speed: the rise in pitch without redesigning a filter every block.
    def level(): return (min(abs(ctl("car.speed")), 50.0) / 40.0) ** 2 * (1 - 0.7 * ctl("car.cabin"))
    def high(): return min(abs(ctl("car.speed")) / 35.0, 1.0)
    low_band = BandPass(PinkNoise(seed=31), 350.0, 0.7) * Live(lambda: level() * (1 - high()))
    high_band = BandPass(PinkNoise(seed=32), 1000.0, 0.7) * Live(lambda: level() * high())
    return Quiet((low_band + high_band) * 0.25, level)


HORNS = {  # family group -> (pitches in Hz, wave)
    "two": ((350.0, 440.0), Saw),
    "one": ((420.0,), Saw),
    "air": ((220.0, 277.0), Saw),
    "thin": ((600.0,), Square),
}


def _horn_kind(family: str) -> str:
    if family in ("almendron", "compartido"):
        return "two"
    if family in ("moto_dos_tiempos", "moto_electrica", "triciclo_carga", "triciclo_pasajeros"):
        return "thin"
    if family in ("guagua_vieja", "interprovincial", "pasajeros", "metroguagua", "camion_80", "furgon",
                  "cisterna", "volquete", "bomberos"):
        return "air"
    return "one"


def horn(family: str, ctl) -> Unit:
    """The horn while car.horn is 1: two saws for an almendrón, one for a Lada, an air horn for the
    guaguas and the trucks, a thin square for the motos."""
    pitches, wave = HORNS[_horn_kind(family)]
    on = Share(_live(ctl, "car.horn", lambda v: 1.0 if v > 0.5 else 0.0))
    tone = None
    for f in pitches:
        part = wave(f)
        tone = part if tone is None else tone + part
    if _horn_kind(family) == "air":
        tone = tone + WhiteNoise(seed=41) * 0.2
    return Quiet(BandPass(tone, pitches[0] * 2.0, 0.8) * (on.tap() * 0.35), lambda: ctl("car.horn"))


def crash(impulse: float, seed: int) -> Timed:
    """A hit: a low thump, inharmonic metal and, past GLASS_MIN, glass; louder and longer with impulse."""
    k = float(np.clip(impulse / CRASH_MIN, 1.0, 20.0))
    loud = min(1.0, 0.25 + 0.25 * np.log2(k))
    rng = np.random.default_rng(seed)
    length = 0.4 + 0.15 * np.log2(k)
    thump = Sine(48.0 + 8 * rng.random()) * ADSR(0.001, 0.25, 0.0, 0.0, gate=0.02, curve=0.7)
    base = 180.0 + 120.0 * rng.random()
    metal = Modal([base * r for r in (1.0, 2.31, 3.47, 5.12, 7.03)], [length * d for d in (0.9, 0.6, 0.45, 0.3, 0.2)],
                  [1.0, 0.7, 0.5, 0.35, 0.2], excite=WhiteNoise(seed=seed) * ADSR(0.0005, 0.03, 0.0, 0.0, gate=0.005))
    out = thump * 0.9 + metal * 0.35
    if impulse >= GLASS_MIN:
        shards = HighPass(WhiteNoise(seed=seed + 1), 5000.0) * ADSR(0.001, 0.6, 0.0, 0.0, gate=0.01, curve=0.6)
        out = out + shards * 0.5
    return Timed(out * loud, float(min(2.5, length + 0.5)))


def bump(depth: float, seed: int) -> Timed:
    """A kerb or a pothole: a short thud, deeper with how far the suspension moved."""
    k = float(np.clip(depth / 0.02, 1.0, 5.0))
    thud = LowPass(WhiteNoise(seed=seed), 180.0) * ADSR(0.001, 0.08, 0.0, 0.0, gate=0.005, curve=0.6)
    return Timed(thud * (0.6 * k) + Sine(70.0) * ADSR(0.001, 0.1, 0.0, 0.0, gate=0.005) * (0.2 * k), 0.3)


def bell(seed: int) -> Timed:
    """A bicycle's bell: two bright partials that ring."""
    strike = WhiteNoise(seed=seed) * ADSR(0.0005, 0.004, 0.0, 0.0, gate=0.001)
    return Timed(Modal([2400.0, 2400.0 * 2.76, 2400.0 * 5.4], [1.2, 0.7, 0.3], [1.0, 0.5, 0.2], excite=strike) * 0.5, 1.4)
