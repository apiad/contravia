"""Countryside: guajira, ported from the approved spike with its numbers."""

from __future__ import annotations

import numpy as np
from euterpe.instruments import BassGuitar, Claves, Guiro, Guitar, Laud

from contravia.audio.music.base import Style, hz


class Guajira(Style):
    """Campo: guajira / punto. Laúd melody, guitar in sesquiáltera (6/8 against 3/4), güiro, claves."""

    melody = ("laud",)
    level = 0.9
    bpm, beats, key, prog = 180, 6, 50, [(0, 0), (0, 0), (3, 0), (4, 1)]
    reverb = 0.22

    def instruments(self):
        s = self.seed
        return {"laud": (Laud(seed=s, level=0.55), 0.3, 1), "guitar": (Guitar(seed=s), -0.25, 0.9),
                "guiro": (Guiro(seed=s, level=0.3), -0.45, 1), "claves": (Claves(seed=s, level=0.3), 0.5, 1),
                "bass": (BassGuitar(seed=s, level=0.8), 0, 1)}

    def bar(self, k, t0, ev):
        e = self.bar_s / 6
        ch = self.chord(k)
        accents = [0, 3] if k % 2 == 0 else [0, 2, 4]
        for i in accents:
            for j, n in enumerate(ch + [ch[0] + 12]):
                ev["guitar"].append((t0 + i * e + j * 0.012, hz(n), 1.6 * e, 0.6 if i == 0 else 0.45))
        ev["bass"].append((t0, hz(ch[0] - 12), 2.5 * e, 0.8))
        for i in range(6):
            ev["guiro"] += self.I["guiro"][0].hit(t0 + i * e, 0.7 if i in accents else 0.35)
        for i in accents:
            ev["claves"] += self.I["claves"][0].hit(t0 + i * e, 0.6)
        # laúd: a walk over the scale that lands on chord tones on the accents
        pos = getattr(self, "pos", ch[0] + 24)
        scale = [self.key + 12 * o + d for o in range(1, 4) for d in self.scale]
        for i in range(6):
            if self.rng.random() < (0.9 if i in accents else 0.55):
                if i in accents:
                    pos = min((n + 24 for n in ch), key=lambda n: abs(n - pos))
                else:
                    j = min(range(len(scale)), key=lambda q: abs(scale[q] - pos))
                    pos = scale[int(np.clip(j + self.rng.choice([-2, -1, 1, 1, 2]), 0, len(scale) - 1))]
                pos = int(np.clip(pos, 64, 86))
                ev["laud"].append((t0 + i * e, hz(pos), e * 1.1, 0.75))
        self.pos = pos
