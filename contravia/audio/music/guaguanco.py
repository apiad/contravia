"""Barrio: rumba guaguancó, ported from the approved spike with its numbers."""

from __future__ import annotations

from euterpe.instruments import Cajon, Claves, Muted, Tumbadora

from contravia.audio.music.base import Style


class Guaguanco(Style):
    """Barrio: rumba guaguancó. Three tumbadoras, rumba clave 3-2, palitos on the cajón."""

    level = 0.5
    bpm, key = 106, 55
    reverb = 0.15

    def instruments(self):
        s = self.seed
        return {"tumba": (Tumbadora(seed=s), 0.0, 1), "claves": (Claves(seed=s, level=0.4), 0.35, 1),
                "palitos": (Claves(tune=1700, seed=s + 1, level=0.18), -0.45, 1),
                "cajon": (Cajon(seed=s, level=0.55), -0.15, 1)}

    def bar(self, k, t0, ev):
        s = self.bar_s / 16
        T = self.I["tumba"][0]
        for i in ([0, 6, 11] if k % 2 == 0 else [4, 8]):
            ev["claves"] += self.I["claves"][0].hit(t0 + i * s, 0.85)
        for i in (0, 2, 4, 6, 7, 10, 12, 14) if k % 2 == 0 else (0, 3, 4, 6, 8, 10, 12, 14):
            ev["palitos"] += self.I["palitos"][0].hit(t0 + i * s, 0.6)
        ev["tumba"] += T.hit(t0 + 6 * s, 0.75, drum="tres-dos") + T.hit(t0 + 7 * s, 0.6, drum="tres-dos")
        ev["tumba"] += T.hit(t0 + 12 * s, 0.85, drum="tumba") + T.hit(t0 + 14 * s, 0.75, drum="tumba")
        ev["tumba"] += T.hit(t0 + 3 * s, 0.4, drum="tres-dos", articulation=Muted())
        ev["cajon"] += self.I["cajon"][0].hit(t0, 0.6) + self.I["cajon"][0].hit(t0 + 8 * s, 0.4)
        for i in range(16):  # the quinto improvises, busier every fourth bar
            if self.rng.random() < (0.45 if k % 4 == 3 else 0.12):
                ev["tumba"] += T.hit(t0 + i * s, 0.5 + 0.45 * self.rng.random(), drum="quinto")
