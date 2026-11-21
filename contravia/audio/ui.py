"""The menus' sounds and the camera's: short synthesized one-shots on the ui bus."""

from __future__ import annotations

from euterpe.sound import ADSR, BandPass, HighPass, Sine, Triangle, WhiteNoise

from contravia.audio.car import Timed

KINDS = ("move", "choose", "back", "shutter", "card", "level")


def ui(kind: str, seed: int = 0) -> Timed:
    """One sound by kind: move (a soft tick), choose (two rising notes), back (two falling), shutter
    (a camera's click and whirr), card (a tres chord for a new place) and level (a two-note blip)."""
    if kind == "move":
        return Timed(Triangle(1400.0) * ADSR(0.001, 0.03, 0.0, 0.0, gate=0.005) * 0.15, 0.06)
    if kind in ("choose", "back", "level"):
        a, b = {"choose": (660.0, 990.0), "back": (660.0, 440.0), "level": (880.0, 1320.0)}[kind]
        env = ADSR(0.002, 0.06, 0.0, 0.0, gate=0.01)
        second = Sine(b) * ADSR(0.002, 0.09, 0.0, 0.0, gate=0.06)
        return Timed((Sine(a) * env + second) * 0.15, 0.2)
    if kind == "shutter":
        click = HighPass(WhiteNoise(seed=seed), 2000.0) * ADSR(0.0005, 0.02, 0.0, 0.0, gate=0.002)
        whirr = BandPass(WhiteNoise(seed=seed + 1), 900.0, 2.0) * ADSR(0.03, 0.15, 0.0, 0.0, gate=0.12)
        return Timed(click * 0.6 + whirr * 0.25, 0.35)
    if kind == "card":
        from euterpe.instruments import Tres

        notes = [(0.0, 392.0, 0.8), (0.02, 493.9, 0.8), (0.04, 587.3, 0.8), (0.06, 784.0, 0.8)]
        return Timed(Tres(seed=seed).play(notes) * 0.5, 1.4)
    raise ValueError(f"no ui sound {kind!r} (known: {', '.join(KINDS)})")
