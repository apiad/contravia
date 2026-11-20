"""Throwaway spike: procedural Cuban background music per biome with Euterpe.

Each style composes one bar at a time from a seed (as the game would, endlessly), and the
renderer turns 4-bar chunks into audio with a 2.5 s tail, overlap-added, the way a worker
would prepare music ahead of the player. Writes one OGG per biome and a crossfade demo.
"""

import sys
import time
import warnings

import numpy as np
import soundfile as sf

warnings.simplefilter("ignore")
from euterpe.instruments import (BassGuitar, Bongo, Cajon, Claves, Flute, Guiro, Guitar, Laud, Maracas,  # noqa: E402
                                 Marimbula, Muted, Nylon, Tres, Tumbadora)
from euterpe.sound import FDNReverb, Pan  # noqa: E402
from euterpe.sound.core import RATE, render  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
TAIL = 2.5


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 11]  # harmonic minor, for the dominant


def triad(key, scale, degree, seventh=False):
    """Midi notes of the chord on a scale degree (0-based), root position, octave of key."""
    idx = [degree, degree + 2, degree + 4] + ([degree + 6] if seventh else [])
    return [key + scale[i % 7] + 12 * (i // 7) for i in idx]


class Style:
    """A biome's music: tempo, key, a progression and a function that writes one bar."""

    bpm, beats, key, scale, prog = 100, 4, 55, MAJOR, [(0, 0), (3, 0), (4, 1), (3, 0)]
    reverb = 0.18

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    @property
    def bar_s(self):
        return self.beats * 60 / self.bpm

    def chord(self, k):
        deg, sev = self.prog[k % len(self.prog)]
        return triad(self.key, self.scale, deg, bool(sev))

    def instruments(self):
        raise NotImplementedError

    def bar(self, k, t0, ev):
        raise NotImplementedError


class Son(Style):
    """Ciudad: son montuno. Tres montuno, tumbao bass, son clave 2-3, bongó martillo, maracas."""

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


class Guajira(Style):
    """Campo: guajira / punto. Laúd melody, guitar in sesquiáltera (6/8 against 3/4), güiro, claves."""

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


class Changui(Style):
    """Costa / Oriente: changüí. Tres with anticipated montuno, marímbula bass, bongó, güiro, maracas."""

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


class Guaguanco(Style):
    """Barrio: rumba guaguancó. Three tumbadoras, rumba clave 3-2, palitos on the cajón."""

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


class Bolero(Style):
    """Noche: bolero. Nylon guitar arpeggios, soft bongó and maracas, a flute singing long notes."""

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


def chunk(style, k0, nbars):
    """Audio of bars k0..k0+nbars of a style, with its tail: what a worker would render ahead."""
    style.I = style.instruments()
    ev = {name: [] for name in style.I}
    for k in range(k0, k0 + nbars):
        style.bar(k, (k - k0) * style.bar_s, ev)
    seconds = nbars * style.bar_s + TAIL
    parts = []
    for name, (inst, pan, gain) in style.I.items():
        if ev[name]:
            parts.append(Pan(inst.play(ev[name], seed=style.seed * 1000 + k0) * gain, pan))
    mix = parts[0]
    for p in parts[1:]:
        mix = mix + p
    mix = FDNReverb(mix, decay=1.6, size=0.6, mix=style.reverb, seed=style.seed)
    return render(mix, seconds)


def stream(style, bars, every=4):
    """Bars of a style rendered every 4 bars and overlap-added, timing each chunk."""
    total = int(round(bars * style.bar_s * RATE + TAIL * RATE))
    out = np.zeros((total, 2), np.float32)
    worst = 0.0
    for k0 in range(0, bars, every):
        t = time.perf_counter()
        x = chunk(style, k0, every)
        worst = max(worst, (time.perf_counter() - t) / (every * style.bar_s))
        a = int(round(k0 * style.bar_s * RATE))
        out[a:a + len(x)] += x[: total - a]
    return out, worst


def save(path, x):
    """Write an OGG in one-second blocks: libsndfile's vorbis writer crashes on one large write."""
    with sf.SoundFile(path, "w", RATE, 2) as f:
        for i in range(0, len(x), RATE):
            f.write(x[i:i + RATE])


def finish(x):
    peak = np.abs(x).max()
    return (x / peak * 0.8 if peak > 0 else x).astype(np.float32)


if __name__ == "__main__":
    styles = {"ciudad-son": Son, "campo-guajira": Guajira, "costa-changui": Changui,
              "barrio-guaguanco": Guaguanco, "noche-bolero": Bolero}
    rendered = {}
    for name, cls in styles.items():
        st = cls(seed=7)
        bars = int(np.ceil(32 / st.bar_s / 4) * 4)
        x, worst = stream(st, bars)
        rendered[name] = (x, st)
        save(f"{OUT}/{name}.ogg", finish(x))
        print(f"{name}: {bars} bars, {len(x) / RATE:.0f} s, worst chunk {worst:.2f} of real time")
    # crossfade demo: the car leaves the city for the countryside; the son fades over 3 s at a bar line
    a, sa = rendered["ciudad-son"]
    b, _ = rendered["campo-guajira"]
    cut = int(round(8 * sa.bar_s * RATE))
    fade = int(3 * RATE)
    ramp = np.linspace(0, np.pi / 2, fade)[:, None]
    head = a[:cut + fade].copy()
    head[cut:] *= np.cos(ramp)
    tail = b[: len(b)].copy()
    tail[:fade] *= np.sin(ramp)
    y = np.zeros((cut + len(tail), 2), np.float32)
    y[: len(head)] += head
    y[cut:] += tail
    save(f"{OUT}/transicion-ciudad-a-campo.ogg", finish(y))
    print(f"transition: {len(y) / RATE:.0f} s, cut at {cut / RATE:.1f} s")
