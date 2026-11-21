"""Rain, thunder and the sea."""

from types import SimpleNamespace

import numpy as np
from euterpe.sound import render

from contravia.audio.weather import Thunder, bubbles, rain, splash, thunder


def fixed(values):
    return lambda name: values.get(name, 0.0)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def strikes(t, state, seconds=600.0, step=0.1):
    out, now = [], 0.0
    while now < seconds:
        out += [(now, d) for d in t.due(now, state)]
        now += step
    return out


def test_thunder_only_in_storms():
    assert strikes(Thunder(1), "rain") == [] and len(strikes(Thunder(1), "storm")) > 5


def test_thunder_is_seeded():
    assert strikes(Thunder(3), "storm") == strikes(Thunder(3), "storm")


def test_thunder_is_heard_distance_over_343_seconds_late():
    t = Thunder(5)
    heard = strikes(t, "storm", 300.0, 0.01)
    assert heard and all(abs(when - (struck + d / 343.0)) < 0.02 for (when, d), struck in zip(heard, t.struck))
    assert all(300 <= d <= 6000 for _, d in heard)


def test_rain_follows_intensity():
    light = rms(render(rain(fixed({"weather.rain": 0.2})), 1.0))
    heavy = rms(render(rain(fixed({"weather.rain": 1.0})), 1.0))
    assert rms(render(rain(fixed({})), 1.0)) < 1e-4 and heavy > 2 * light > 0


def test_one_shots_end():
    for u in (thunder(800.0, 1), thunder(5000.0, 1), splash(1)):
        assert 0.5 < u.duration <= 8.0 and rms(render(u, u.duration)) > 1e-4


def test_far_thunder_is_duller_than_near():
    def highs(x):
        w = np.abs(np.fft.rfft(x))
        f = np.fft.rfftfreq(len(x), 1 / 48000)
        return w[f > 1500].sum() / w.sum()

    assert highs(render(thunder(400.0, 2), 2.0)) > 2 * highs(render(thunder(5000.0, 2), 2.0))


def test_bubbles_only_under_water():
    assert rms(render(bubbles(fixed({"car.under": 0.0})), 0.5)) < 1e-4
    assert rms(render(bubbles(fixed({"car.under": 1.0})), 0.5)) > 1e-3


def test_sinking_muffles_and_coming_ashore_releases():
    from test_sound import Recorder, fake_game

    from contravia.audio import CONTROLS, Sound

    s = Sound(Recorder(), CONTROLS)
    g = fake_game(1000.0)
    g.sinking = 0.0
    s.frame(g)
    g.sinking = None
    s.frame(g)
    calls = [(c[0], c[1]) for c in s.audio.calls if c[0] in ("lowpass", "event")]
    assert ("lowpass", (400.0,)) in calls and ("lowpass", (20000.0,)) in calls
    assert any(c[0] == "event" and c[1][0] == "splash" for c in calls)


def test_a_storm_fires_thunder_events():
    from test_sound import Recorder, fake_game

    from contravia.audio import CONTROLS, Sound

    s = Sound(Recorder(), CONTROLS)
    g = fake_game(1000.0)
    g.weather_now = SimpleNamespace(state="storm", rain=1.0, wet=1.0, cloud=1.0)
    for k in range(6000):  # ten minutes of frames, the clock stepped by hand
        s.frame(g, now=k * 0.1)
    assert any(c[0] == "event" and c[1][0] == "thunder" for c in s.audio.calls)
