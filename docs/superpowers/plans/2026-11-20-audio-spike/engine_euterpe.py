"""Throwaway: the same engine as engine.py, as a Euterpe unit graph, to time it."""
import sys, time
import numpy as np, soundfile as sf
from euterpe.sound import Wavetable, BandPass, LowPass, Tanh, WhiteNoise
from euterpe.sound.core import Unit, RATE, render
import engine as E

class Feed(Unit):
    """Stand-in for a live control: a 120 Hz series from the game, held to 48 kHz by linear steps."""
    def __init__(self, values):
        self.values = np.asarray(values, float); self.i = 0
    def reset(self): self.i = 0
    def inputs(self): return []
    def block(self, n):
        t = (self.i + np.arange(n)) / RATE * E.PHYS
        self.i += n
        return np.interp(t, np.arange(len(self.values)), self.values).astype(np.float32)

def cycle_table(cyl, uneven, seed=0, size=2048):
    """One four-stroke cycle: cyl decaying pulses, each cylinder a little different."""
    rng = np.random.default_rng(seed); amp = 1 - uneven * rng.random(cyl)
    ph = np.arange(size) / size * cyl; k = ph.astype(int); fr = ph - k
    x = amp[k] * np.exp(-fr * 6) * np.sin(np.pi * np.minimum(fr * 3, 1))
    return x - x.mean()

def engine(f, rpm, thr):
    """Euterpe pulls each unit from one place only, so every use of a control or of the pulses is its own copy."""
    R, T = (lambda: Feed(rpm)), (lambda: Feed(thr))
    table = cycle_table(f["cyl"], f["uneven"])
    P = lambda: Wavetable(table, R() * (1 / 120)) * (T() * 0.75 + 0.25)
    body = BandPass(P(), f["res"][0], 2.5) + BandPass(P(), f["res"][1], 2.5) + P() * 0.4
    tone = LowPass(body, (T() * 0.5 + 0.5) * (R() * (0.8 / f["red"]) + 0.6) * f["bright"])
    intake = BandPass(WhiteNoise(seed=1), 2200, 1.0) * (T() * R() * (0.06 / f["red"]))
    return Tanh((tone + intake) * 3)


if __name__ == "__main__":
    out = sys.argv[1]
    for name, f in E.FAMILIES.items():
        rpm, thr = E.drive(f)
        u = engine(f, rpm, thr); secs = len(rpm) / E.PHYS
        for block in (1024, 256):
            t0 = time.perf_counter(); x = render(u, secs, block=block); dt = time.perf_counter() - t0
            print(f"{name} block {block}: {secs:.0f} s in {dt:.2f} s (x{secs/dt:.0f} real time)")
        x = x / np.abs(x).max() * 0.8
        with sf.SoundFile(f"{out}/motor-euterpe-{name}.ogg", "w", RATE, 1) as fh:
            for i in range(0, len(x), RATE): fh.write(x[i:i+RATE])
