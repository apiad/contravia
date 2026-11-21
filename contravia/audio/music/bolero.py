"""Night: bolero, ported from the approved spike with its numbers."""

from __future__ import annotations

from euterpe.instruments import BassGuitar, Bongo, Flute, Guitar, Maracas, Muted, Nylon

from contravia.audio.music.base import MINOR, Style, hz


class Bolero(Style):
    """Noche: bolero. Nylon guitar arpeggios, soft bongó and maracas, a flute singing long notes."""

    melody = ("flute",)
    level = 1.6
    bpm, key, scale, prog = 72, 57, MINOR, [(0, 0), (3, 0), (4, 1), (0, 0)]
    reverb = 0.32

    def instruments(self):
        s = self.seed
        return {"guitar": (Guitar(string=Nylon(), seed=s), -0.2, 1.2), "bass": (BassGuitar(seed=s, level=0.7), 0, 1),
                "bongo": (Bongo(seed=s, level=0.35), -0.4, 1), "maracas": (Maracas(seed=s, level=0.25), 0.45, 1),
                "flute": (Flute(seed=s, level=0.35), 0.25, 1)}

    def bar(self, k, t0, ev):
        e = self.bar_s / 8
        ch = self.chord(k)
        ev["bass"].append((t0, hz(ch[0] - 12), 3.5 * e, 0.7))
        ev["bass"].append((t0 + 4 * e, hz(ch[2] - 24), 3.5 * e, 0.6))
        arp = [ch[0], ch[1] + 12, ch[2], ch[1] + 12, ch[0] + 12, ch[1] + 12, ch[2], ch[1] + 12]
        for i, n in enumerate(arp):
            ev["guitar"].append((t0 + i * e, hz(n), 2 * e, 0.55 if i % 2 else 0.65))
        for i in (0, 1, 3, 4, 5, 7):
            ev["bongo"] += self.I["bongo"][0].hit(t0 + i * e, 0.45 if i in (0, 4) else 0.3,
                                                  drum="hembra" if i == 7 else "macho",
                                                  articulation=None if i == 7 else Muted())
        for i in (0, 2, 3, 4, 6, 7):
            ev["maracas"] += self.I["maracas"][0].hit(t0 + i * e, 0.4, hand="izq" if i % 2 else "der")
        if k % 8 >= 2:  # the flute enters after two bars, two or three notes a bar
            pos = getattr(self, "pos", ch[2] + 12)
            starts = [0, 3, 6] if self.rng.random() < 0.5 else [0, 4]
            for j, st in enumerate(starts):
                cand = [n + 12 * o for n in ch for o in (1, 2)] if j == 0 else \
                    [self.key + 12 * o + d for o in (1, 2) for d in self.scale]
                cand = [c for c in cand if 72 <= c <= 88]
                pos = min(cand, key=lambda c: abs(c - pos) + 3 * self.rng.random())
                nxt = starts[j + 1] if j + 1 < len(starts) else 8
                ev["flute"].append((t0 + st * e, hz(pos), (nxt - st) * e * 0.95, 0.6))
            self.pos = pos
