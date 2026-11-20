"""The engine: a Euterpe graph that follows the rpm and the throttle the physics gives it.

One wavetable holds a whole four-stroke cycle, as many decaying pulses as cylinders, each a
little different, played once per cycle: the firing frequency follows the rpm without
crossfades. The pulses ring two resonances of the exhaust and pass a low-pass that opens with
load and rpm; intake noise rises with the throttle; a lift-off at high rpm pops, a diesel
clicks at each firing and a turbo whines. The numbers of each family are in rules/sound.py.
"""

from __future__ import annotations

import numpy as np
from euterpe.sound import BandPass, Live, LowPass, Share, Sine, Tanh, Unit, Wavetable, WhiteNoise

from contravia.rules.sound import FamilySound


def cycle(cylinders: int, uneven: float, seed: int, size: int = 2048) -> np.ndarray:
    """One cycle of the engine: a decaying pulse per cylinder, each up to `uneven` weaker, seeded."""
    amp = 1 - uneven * np.random.default_rng(seed).random(cylinders)
    ph = np.arange(size) / size * cylinders
    k = ph.astype(int)
    fr = ph - k
    x = amp[k] * np.exp(-fr * 6) * np.sin(np.pi * np.minimum(fr * 3, 1))
    return x - x.mean()


def _live(ctl, name: str) -> Share:
    return Share(Live(lambda: ctl(name), start=ctl(name)))


def pulses(snd: FamilySound, ctl, seed: int, rpm: Share | None = None, load: Share | None = None) -> Unit:
    """The firing pulses, times the load: one cycle every 60 x strokes / 2 seconds per 1000 rpm."""
    rpm = rpm or _live(ctl, "car.rpm")
    load = load or Share(Live(lambda: ctl("car.throttle"), start=ctl("car.throttle")) * 0.75 + 0.25)
    return Wavetable(cycle(snd.cylinders, snd.uneven, seed), rpm.tap() * (1 / (30 * snd.strokes))) * load.tap()


class Pops(Unit):
    """The exhaust's pops on a lift-off: sparse clicks while the throttle is shut above 45 % of the
    redline, each a short burst of noise. Its spectrum is broad, darkened by the low-pass after it."""

    def __init__(self, rpm: Unit, throttle: Unit, red: float, seed: int):
        self.rpm, self.throttle, self.red, self.seed = rpm, throttle, red, seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        super().reset()
        self.rng = np.random.default_rng(self.seed)

    def block(self, n: int) -> np.ndarray:
        r, t = self.rpm.block(n), self.throttle.block(n)
        on = (t < 0.05) & (r > 0.45 * self.red)
        hits = (self.rng.random(n) < 0.0006) & on
        return (hits * self.rng.standard_normal(n) * 8).astype(np.float32)


def engine(snd: FamilySound, idle: float, red: float, ctl, seed: int) -> Unit:
    """The whole engine of one vehicle, read from the controls car.rpm, car.throttle and car.speed."""
    if snd.strokes == 0:
        return _no_pulses(snd, ctl, seed)
    rpm = _live(ctl, "car.rpm")
    thr = _live(ctl, "car.throttle")
    load = Share(thr.tap() * 0.75 + 0.25)
    p = Share(pulses(snd, ctl, seed, rpm, load))
    body = BandPass(p.tap(), snd.res[0], 2.5) + BandPass(p.tap(), snd.res[1], 2.5) + p.tap() * 0.4
    if snd.knock:  # a click with each firing: noise gated by the pulses
        body = body + BandPass(WhiteNoise(seed=seed + 1) * p.tap(), 4000.0, 1.2) * snd.knock
    open_ = (thr.tap() * 0.5 + 0.5) * (rpm.tap() * (0.8 / red) + 0.6) * snd.bright
    tone = LowPass(body, open_)
    intake = BandPass(WhiteNoise(seed=seed + 2), 2200.0, 1.0) * (thr.tap() * rpm.tap() * (0.06 / red))
    out = tone + intake
    if snd.crackle:
        out = out + LowPass(Pops(rpm.tap(), thr.tap(), red, seed + 3), 3000.0) * snd.crackle
    if snd.turbo:
        out = out + Sine(rpm.tap() * (30 / (30 * snd.strokes))) * (load.tap() * snd.turbo * 0.05)
    return Tanh(out * 3) * (1 / 3)


def _no_pulses(snd: FamilySound, ctl, seed: int) -> Unit:
    """An electric whine that rises with speed, or a chain's clicks at the pedalling cadence."""
    speed = _live(ctl, "car.speed")
    if snd.kind == "chain":
        click = np.zeros(512)
        click[:12] = np.hanning(24)[12:] * np.array([1, -1] * 6)
        return BandPass(Wavetable(click, speed.tap() * 0.5 + 0.01), 3000.0, 1.0) * 0.4
    whine = Sine(speed.tap() * 18.0 + 40.0) * 0.15 + Sine(speed.tap() * 36.0 + 80.0) * 0.05
    return BandPass(whine + WhiteNoise(seed=seed) * 0.002, 900.0, 0.5) * 2.0


def player_engine(family: str, idle: float, red: float, seed: int, ctl) -> Unit:
    """The source the sound process builds for the player's car: a module-level function, so the
    game can hand it over as functools.partial(player_engine, family, idle, red, seed)."""
    from contravia.rules.sound import of

    return engine(of(family), idle, red, ctl, seed) * 0.8
