"""Throwaway spike: an engine sound synthesized from rpm and throttle, for three families.

A plain longitudinal model drives the car (idle, two revs in neutral, a pull through the
gears, a lift-off, braking to idle) and gives rpm and throttle at 120 Hz, as manifoldx's
vehicles would. The sound is built per sample from them: each cylinder fires a pulse once
every two turns, the pulses go through the exhaust's resonances and a low-pass that opens
with load, intake noise rises with throttle, and a lift-off at high rpm crackles.
"""

import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import butter, iirpeak, lfilter, sosfilt

RATE, PHYS = 48000, 120
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

FAMILIES = {
    # cylinders, idle, redline, gear ratios, final drive, exhaust resonances (Hz), brightness, unevenness, mass
    "v8-americano-1956": dict(cyl=8, idle=600, red=4600, ratios=[2.5, 1.5, 1.0], final=3.4, res=(85, 260),
                              bright=900, uneven=0.35, mass=1800, power=150e3, crackle=0.5),
    "lada-2107": dict(cyl=4, idle=850, red=6200, ratios=[3.67, 2.1, 1.36, 1.0], final=4.1, res=(170, 540),
                      bright=2200, uneven=0.08, mass=1030, power=55e3, crackle=0.2),
    "camion-diesel": dict(cyl=6, idle=550, red=2600, ratios=[6.0, 3.5, 2.1, 1.4, 1.0], final=5.0, res=(70, 210),
                          bright=1400, uneven=0.12, mass=9000, power=140e3, crackle=0.0, knock=0.6),
}


def drive(f):
    """rpm and throttle at 120 Hz for a scripted drive of about 30 s."""
    r, dt = 0.33, 1 / PHYS
    v, gear, rpm, out_rpm, out_thr = 0.0, -1, f["idle"], [], []
    t = 0.0
    while t < 30:
        if t < 2:
            thr, clutch = 0.0, False
        elif t < 4.5:  # two blips in neutral
            thr, clutch = (0.8 if (2.3 < t < 2.8 or 3.4 < t < 3.9) else 0.0), False
        elif t < 19:
            thr, clutch = 1.0, True
        elif t < 23:
            thr, clutch = 0.0, True  # lift off, engine braking
        else:
            thr, clutch = 0.0, False
        if not clutch:
            gear = -1
            target = f["idle"] + thr * (f["red"] * 0.75 - f["idle"])
            rpm += (target - rpm) * (8 if thr > 0 else 3) * dt
            v = max(0.0, v - 3.0 * dt)
        else:
            if gear < 0:
                gear = 0
            ratio = f["ratios"][gear] * f["final"]
            wheel_rpm = v / (2 * np.pi * r) * 60 * ratio
            rpm = max(wheel_rpm, f["idle"] * 1.3 if thr > 0 else f["idle"])
            if thr > 0 and rpm > f["red"] * 0.92 and gear < len(f["ratios"]) - 1:
                gear += 1
            power = f["power"] * thr * min(1.0, rpm / (f["red"] * 0.7))
            force = power / max(v, 2.0) - 0.4 * v * v - (0 if thr > 0 else 600 + 0.15 * rpm)
            v = max(0.0, v + force / f["mass"] * dt)
            if thr == 0 and t > 21 and gear > 0 and rpm < f["idle"] * 1.6:
                gear -= 1
        out_rpm.append(rpm)
        out_thr.append(thr)
        t += dt
    return np.array(out_rpm), np.array(out_thr)


def synth(f, rpm, thr, seed=0):
    """Samples at 48 kHz from rpm and throttle at 120 Hz."""
    rng = np.random.default_rng(seed)
    n = len(rpm) * RATE // PHYS
    tt = np.arange(n) / RATE
    tp = np.arange(len(rpm)) / PHYS
    rpm_s = np.interp(tt, tp, rpm)
    thr_s = np.interp(tt, tp, np.convolve(thr, np.ones(6) / 6, "same"))  # pedal smoothed over 50 ms
    cyc = np.cumsum(rpm_s / 120 / RATE)  # four-stroke cycles: one firing per cylinder every two turns
    pos = cyc * f["cyl"]
    k = np.floor(pos).astype(int)
    frac = pos - k
    amp_cyl = 1 - f["uneven"] * rng.random(f["cyl"])  # each cylinder fires a little differently
    load = 0.25 + 0.75 * thr_s
    amp = amp_cyl[k % f["cyl"]] * load * (1 + 0.04 * rng.standard_normal(len(np.unique(k)))[k - k.min()])
    pulse = amp * np.exp(-frac * 6.0) * np.sin(np.pi * np.minimum(frac * 3, 1))
    pulse -= pulse.mean()
    # exhaust: two resonant peaks and a low-pass that opens with rpm and throttle, by blocks of 256
    y = np.zeros(n)
    for fr in f["res"]:
        b, a = iirpeak(fr, 2.5, RATE)
        y += lfilter(b, a, pulse)
    y += 0.4 * pulse
    out = np.zeros(n)
    zi = None
    for i in range(0, n, 256):
        cut = f["bright"] * (0.5 + 0.5 * thr_s[i]) * (0.6 + 0.8 * rpm_s[i] / f["red"])
        sos = butter(2, min(cut, 16000), fs=RATE, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        out[i:i + 256], zi = sosfilt(sos, y[i:i + 256], zi=zi)
    # intake: band noise that rises with throttle and rpm
    noise = rng.standard_normal(n)
    sos = butter(2, [1200, 4000], btype="band", fs=RATE, output="sos")
    out += sosfilt(sos, noise) * 0.06 * thr_s * (rpm_s / f["red"])
    # diesel knock: a click at each firing
    if f.get("knock"):
        click = (frac < 0.04) * rng.standard_normal(n)
        sos = butter(2, [2500, 6000], btype="band", fs=RATE, output="sos")
        out += sosfilt(sos, click) * f["knock"] * 0.5
    # crackle on overrun: high rpm, no throttle
    if f["crackle"]:
        over = (thr_s < 0.05) & (rpm_s > f["red"] * 0.45)
        pops = (rng.random(n) < 0.0006) * over * rng.standard_normal(n) * 8
        out += lfilter(*butter(1, 3000, fs=RATE), pops) * f["crackle"]
    out = np.tanh(out * 3) / 3
    return out / np.abs(out).max() * 0.8


if __name__ == "__main__":
    for name, f in FAMILIES.items():
        rpm, thr = drive(f)
        t = time.perf_counter()
        x = synth(f, rpm, thr)
        dt = time.perf_counter() - t
        with sf.SoundFile(f"{OUT}/motor-{name}.ogg", "w", RATE, 1) as fh:
            for i in range(0, len(x), RATE):
                fh.write(x[i:i + RATE].astype(np.float32))
        print(f"{name}: rpm {rpm.min():.0f}-{rpm.max():.0f}, {len(x) / RATE:.0f} s in {dt:.2f} s "
              f"(x{len(x) / RATE / dt:.0f} real time)")
