"""The game's sound: one process kept across reloads, volumes from the options, quiet outside the drive."""

import time
from types import SimpleNamespace

import pytest

from contravia.audio import Sound
from contravia.settings import Options


@pytest.fixture
def options(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    return Options.load()


def test_sound_is_kept_across_a_rebuild(options):
    st = SimpleNamespace()
    a = Sound.start(st, options, backend="null")
    b = Sound.start(st, options, backend="null")
    try:
        assert a is b and st.sound is a
        t = time.time()
        while not a.audio.state()["alive"] and time.time() - t < 20:
            time.sleep(0.05)
        assert a.audio.state()["alive"]
    finally:
        a.close()


def test_a_kept_sound_missing_controls_is_replaced(options):
    st = SimpleNamespace()
    a = Sound.start(st, options, backend="null")
    a.names = ("only.this",)  # as a process started by older code would have
    b = Sound.start(st, options, backend="null")
    try:
        assert b is not a and not a.audio.alive
    finally:
        b.close()


def test_volume_rows_set_the_bus_gains(options):
    options["vol"].pick(5)
    options["vol_music"].pick(10)
    options["vol_engine"].pick(7)
    g = Sound.gains(options)
    assert g["music"] == 0.5 and abs(g["engine"] - 0.35) < 1e-9 and g["ui"] == 0.5


def test_default_volumes(options):
    assert [options[k].value for k in ("vol", "vol_engine", "vol_effects", "vol_ambience", "vol_music")] == [8, 7, 7, 6, 5]


def test_engine_and_effects_are_silent_outside_the_drive():
    base = {"engine": .7, "effects": .7, "music": .5, "ambience": .6, "ui": .8}
    for mode in ("map", "garage", "menu"):
        g = Sound.mode_gains(mode, base)
        assert g["engine"] == g["effects"] == 0 and g["music"] == .5 and g["ui"] == .8
    assert Sound.mode_gains("drive", base) == base


def test_the_pause_lowers_the_music_by_10_db():
    g = Sound.mode_gains("paused", {"engine": .7, "effects": .7, "music": .5, "ambience": .6, "ui": .8})
    assert g["engine"] == 0 and abs(g["music"] - .5 * 10 ** (-10 / 20)) < 1e-9


def test_the_volume_rows_are_saved_and_loaded(options):
    options["vol_music"].pick(2)
    options.save()
    assert Options.load()["vol_music"].value == 2



def fake_game(rpm):
    import numpy as np

    from contravia.car import CarState, Controls

    cam = SimpleNamespace(position=np.array([-6.0, 2.0, 0.0]), target=np.array([0.0, 1.0, 0.0]))
    return SimpleNamespace(mode="drive", family="ruso", driven=None, sinking=None,
                           engine=SimpleNamespace(camera=cam, state=SimpleNamespace()),
                           car=CarState(x=0.0, z=0.0, yaw=0.0, speed=10.0, rpm=rpm),
                           controls_now=Controls(0.6, 0.0, 0.0, False))


def test_a_recorded_engine_follows_the_rpm(options, tmp_path):
    import wave

    import numpy as np

    out = tmp_path / "drive.wav"
    s = Sound.start(SimpleNamespace(), options, backend="file", path=out)
    g = fake_game(2400.0)
    s.attach(g)
    t = time.time()
    while time.time() - t < 3.0:
        s.frame(g)
        time.sleep(1 / 60)
    s.close()
    with wave.open(str(out)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2)[:, 0] / 32767
    x = x[-48000:]
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / 48000)
    band = (f > 40) & (f < 400)
    firing = 2400 * 4 / 120  # a four-cylinder four-stroke at 2400 rpm: 80 Hz
    peaks = f[band][np.argsort(spec[band])[-3:]]
    assert np.abs(x).max() > 0.01 and any(abs(p / firing - round(p / firing)) < 0.05 for p in peaks)
