"""City: son montuno, ported from the approved spike with its numbers."""

from __future__ import annotations

from euterpe.instruments import BassGuitar, Bongo, Claves, Maracas, Muted, Tres

from contravia.audio.music.base import Style, hz


class Son(Style):
    """Ciudad: son montuno. Tres montuno, tumbao bass, son clave 2-3, bongó martillo, maracas."""

    intro, breaks = 2, True
    bpm, key, prog = 100, 55, [(0, 0), (3, 0), (4, 1), (3, 0)]

    def instruments(self):
        s = self.seed
        return {"tres": (Tres(seed=s), 0.25, 1.0), "bass": (BassGuitar(seed=s), 0.0, 0.9),
                "claves": (Claves(seed=s, level=0.35), 0.4, 1), "bongo": (Bongo(seed=s), -0.3, 0.8),
                "maracas": (Maracas(seed=s, level=0.35), -0.5, 1)}

    def bar(self, k, t0, ev):
        e = self.bar_s / 8
        ch, nxt = self.chord(k), self.chord(k + 1)
        for i in ([2, 4] if k % 2 == 0 else [0, 3, 6]):
            ev["claves"] += self.I["claves"][0].hit(t0 + i * e, 0.8)
        for i in range(8):
            drum = "hembra" if i in (3, 7) else "macho"
            v = 0.75 if i in (3, 7) else 0.45 + 0.1 * self.rng.random()
            art = None if i in (3, 7) else Muted()
            ev["bongo"] += self.I["bongo"][0].hit(t0 + i * e, v, drum=drum, articulation=art)
            ev["maracas"] += self.I["maracas"][0].hit(t0 + i * e, 0.6 if i % 2 == 0 else 0.4,
                                                      hand="izq" if i % 2 else "der")
        ev["bass"] += [(t0 + 3 * e, hz(nxt[0] - 24), 1.4 * e, 0.85), (t0 + 6 * e, hz(nxt[2] - 24), 2 * e, 0.8)]
        mask = [1, 0, 1, 1, 0, 1, 1, 0] if k % 2 == 0 else [0, 1, 1, 0, 1, 1, 0, 1]
        tones = [n + 12 for n in ch] + [ch[0] + 24]
        for i, on in enumerate(mask):
            if on:
                a = tones[(i + k) % len(tones)]
                ev["tres"].append((t0 + i * e, hz(a), e * 0.9, 0.7))
                if self.rng.random() < 0.35:
                    ev["tres"].append((t0 + i * e, hz(a + 12 if a + 12 <= 84 else a - 12), e * 0.9, 0.55))
