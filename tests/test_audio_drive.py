"""A recorded drive of the whole sound holds its budget: no underruns, a limited master, music
under the engine, and each process within its share of a core."""

import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest


def read(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2) / 32767


def drive(tmp_path, monkeypatch, backend, seconds):
    """A fake game cruising past twelve traffic cars for `seconds`; returns the sound's usage."""
    from test_sound import fake_game

    from contravia.audio import Sound
    from contravia.rules.traffic import looks
    from contravia.settings import Options

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    s = Sound.start(SimpleNamespace(), Options.load(), backend=backend,
                    path=tmp_path / "drive.wav" if backend == "file" else None)
    g = fake_game(2500.0)
    g.engine.input = SimpleNamespace(is_pressed=lambda k: False)
    car_look = next(i for i, lk in enumerate(looks()) if lk[0] == "car")
    n = 12
    s.attach(g)
    t0 = time.time()
    while time.time() - t0 < seconds:  # the engine between 2000 and 3500 rpm
        t = time.time() - t0
        g.car.rpm = 2750 + 750 * np.sin(t / 3)
        xyz = np.c_[np.arange(n) * 15.0 - 60 + (t * 8) % 15, np.zeros(n), np.full(n, 4.0)]
        g.traffic_now = (xyz, np.zeros(n), np.full(n, 9.0), np.full(n, car_look), np.zeros(n))
        s.frame(g)
        time.sleep(1 / 60)
    use = s.usage()
    s.close()
    return use


@pytest.mark.slow
def test_a_recorded_drive_is_limited_with_the_music_under_the_engine(tmp_path, monkeypatch):
    import pyloudnorm

    drive(tmp_path, monkeypatch, "file", 40)
    master, engine, music = (read(tmp_path / f) for f in ("drive.wav", "drive.engine.wav", "drive.music.wav"))
    meter = pyloudnorm.Meter(48000)
    tail = slice(-20 * 48000, None)  # the last 20 s, with the music playing
    gap = meter.integrated_loudness(engine[tail]) - meter.integrated_loudness(music[tail])
    assert np.abs(master).max() <= 10 ** (-1 / 20) + 2e-3
    assert 10 <= gap <= 18, gap


@pytest.mark.slow
def test_the_sound_fits_its_budget(tmp_path, monkeypatch):
    """Needs an idle host: the shares are of one core, and contention inflates them."""
    use = drive(tmp_path, monkeypatch, "null", 40)
    assert use["audio_underruns"] == 0 and use["audio_cpu"] < 0.6 and use.get("music_cpu", 0) < 0.35, use
