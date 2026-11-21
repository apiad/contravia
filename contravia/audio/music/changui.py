"""Coast: changüí, ported from the approved spike with its numbers."""

from __future__ import annotations

from euterpe.instruments import Bongo, Guiro, Maracas, Marimbula, Tres

from contravia.audio.music.base import Style, hz


class Changui(Style):
    """Costa / Oriente: changüí. Tres with anticipated montuno, marímbula bass, bongó, güiro, maracas."""

    intro, breaks = 2, True
    bpm, key, prog = 118, 60, [(0, 0), (4, 1), (4, 1), (0, 0)]

    def instruments(self):
        s = self.seed
        return {"tres": (Tres(seed=s), 0.3, 1), "marimbula": (Marimbula(seed=s, level=0.8), 0, 1),
                "bongo": (Bongo(seed=s, level=0.6), -0.3, 1), "guiro": (Guiro(seed=s, level=0.28), -0.5, 1),
                "maracas": (Maracas(seed=s, level=0.3), 0.5, 1)}

    def bar(self, k, t0, ev):
        e = self.bar_s / 8
        ch, nxt = self.chord(k), self.chord(k + 1)
        for i in range(8):
            ev["guiro"] += self.I["guiro"][0].hit(t0 + i * e, 0.7 if i % 2 == 0 else 0.3)
            ev["maracas"] += self.I["maracas"][0].hit(t0 + i * e, 0.5, hand="izq" if i % 2 else "der")
            if self.rng.random() < (0.9 if i in (1, 3, 5, 7) else 0.35):
                ev["bongo"] += self.I["bongo"][0].hit(t0 + i * e + 0.01 * self.rng.random(),
                                                      0.5 + 0.4 * self.rng.random(),
                                                      drum="macho" if self.rng.random() < 0.6 else "hembra")
        ev["marimbula"] += self.I["marimbula"][0].hit(t0 + 3 * e, 0.9, pitch=hz(nxt[0] - 24))
        ev["marimbula"] += self.I["marimbula"][0].hit(t0 + 7 * e, 0.8, pitch=hz(nxt[2] - 24))
        tones = [n + 12 for n in ch]
        for i in (1, 2, 3, 5, 6, 7):
            if self.rng.random() < 0.85:
                n = tones[(i * 2 + k) % 3] - 12 * (self.rng.random() < 0.3)
                ev["tres"].append((t0 + i * e, hz(n), e * 0.8, 0.7))
                ev["tres"].append((t0 + i * e + 0.006, hz(n + 12), e * 0.8, 0.5))
