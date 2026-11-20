"""Music composed while you drive: a style per biome writes one bar at a time from a seed, and
compose() turns a few bars into a Euterpe unit that the music process renders ahead.

The five styles came from the spike approved by ear (docs/superpowers/plans/2026-11-20-audio-spike/
music.py) and keep its numbers. Each style may thin its percussion (intensity below 1, for the
menu) and has sections: the son and the changüí open with the tres alone and break to the
percussion now and then; the guajira and the bolero rest their melody in an interlude.
"""

from __future__ import annotations

import zlib

import numpy as np
from euterpe.sound import FDNReverb, Pan, Unit

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 11]  # harmonic minor, for the dominant
PERCUSSION = ("bongo", "maracas", "guiro", "claves", "palitos", "cajon", "tumba")


def hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def triad(key: int, scale: list[int], degree: int, seventh: bool = False) -> list[int]:
    """Midi notes of the chord on a scale degree (0-based), root position, octave of key."""
    idx = [degree, degree + 2, degree + 4] + ([degree + 6] if seventh else [])
    return [key + scale[i % 7] + 12 * (i // 7) for i in idx]


def seed_of(style: str, date) -> int:
    """The seed of a style on a game date: the same day plays the same music."""
    return zlib.crc32(f"{style}:{date.isoformat()}".encode())


class Style:
    """A biome's music: tempo, key, a progression and a function that writes one bar."""

    bpm, beats, key, scale, prog = 100, 4, 55, MAJOR, [(0, 0), (3, 0), (4, 1), (3, 0)]
    reverb = 0.18
    level = 0.6  # the style's gain, so every style peaks near 0.45

    def __init__(self, seed: int, intensity: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.seed, self.intensity = seed, intensity
        self._thin = np.random.default_rng(seed + 1)
        self.I = self.instruments()

    @property
    def bar_s(self) -> float:
        return self.beats * 60 / self.bpm

    def chord(self, k: int) -> list[int]:
        deg, sev = self.prog[k % len(self.prog)]
        return triad(self.key, self.scale, deg, bool(sev))

    def instruments(self) -> dict:
        raise NotImplementedError

    def bar(self, k: int, t0: float, ev: dict) -> None:
        raise NotImplementedError

    intro = 0  # bars of tres alone at the start
    breaks = False  # whether the last bar of some 8-bar spans is percussion alone
    melody: tuple[str, ...] = ()  # the instruments an interlude rests

    def section(self, k: int) -> str:
        """What bar k is: "intro", "break", "interlude" or "main", chosen by the seed per 8 bars."""
        if k < self.intro:
            return "intro"
        draw = np.random.default_rng((self.seed, k // 8)).random()
        if self.breaks and k % 8 == 7 and draw < 0.5:
            return "break"
        if self.melody and k % 16 in (14, 15) and draw < 0.6:
            return "interlude"
        return "main"

    def compose(self, k0: int, nbars: int) -> tuple[Unit, float]:
        """Bars k0 to k0 + nbars as one unit whose time 0 is bar k0's start, and their length."""
        ev = {name: [] for name in self.I}
        for k in range(k0, k0 + nbars):
            before = {name: len(v) for name, v in ev.items()}
            self.bar(k, (k - k0) * self.bar_s, ev)
            part = self.section(k)
            for name in ev:
                rest = (part == "intro" and name != "tres") or (part == "break" and name not in PERCUSSION) \
                    or (part == "interlude" and name in self.melody)
                if rest:
                    del ev[name][before[name]:]
        if self.intensity < 1:  # thin the percussion, seeded, keeping the first beat of each bar
            for name in PERCUSSION:
                if name in ev:
                    ev[name] = [e for e in ev[name] if self._thin.random() < self.intensity or _on_bar(e, self.bar_s)]
        parts = []
        if not any(ev.values()):  # a bar with nothing left: one silent note keeps the unit whole
            ev[next(iter(ev))].append((0.0, 220.0, 0.01, 0.0))
        for name, (inst, pan, gain) in self.I.items():
            if ev[name]:
                parts.append(Pan(inst.play(ev[name], seed=self.seed * 1000 + k0) * gain, pan))
        mix = parts[0]
        for p in parts[1:]:
            mix = mix + p
        return FDNReverb(mix * self.level, decay=1.6, size=0.6, mix=self.reverb, seed=self.seed % 100000), nbars * self.bar_s


def _on_bar(e, bar_s: float) -> bool:
    t = e[0] if isinstance(e, tuple) else e.time
    return abs(t / bar_s - round(t / bar_s)) < 1e-6
