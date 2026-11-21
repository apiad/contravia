"""Cuban music by biome, composed bar by bar from a seed and rendered ahead in chunks."""

import datetime
import time

import numpy as np
import pytest
from euterpe.sound import render
from euterpe.stream import chunks

from contravia.audio.music import STYLES, seed_of
from contravia.audio.music.player import crossfade
from contravia.rules.sound import BY_BIOME
from contravia.rules.sound import STYLES as NAMES

DAY = datetime.date(2026, 11, 20)


def events(style, bars=8):
    ev = {name: [] for name in style.I}
    for k in range(bars):
        style.bar(k, k * style.bar_s, ev)
    return ev


def test_the_names_match():
    assert tuple(STYLES) == NAMES and set(BY_BIOME.values()) == set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_a_style_is_seeded(name):
    def numbers(ev):
        return {n: [(e[0], float(e[1]), e[2]) if isinstance(e, tuple) else (e.time, float(e.freq), e.duration)
                    for e in v] for n, v in ev.items()}

    a, b = events(STYLES[name](seed_of(name, DAY))), events(STYLES[name](seed_of(name, DAY)))
    assert numbers(a) == numbers(b)
    assert seed_of(name, DAY) != seed_of(name, DAY + datetime.timedelta(days=1))


@pytest.mark.parametrize("name", NAMES)
def test_events_stay_in_their_bar(name):
    st = STYLES[name](1)
    for k in range(8):
        ev = {n: [] for n in st.I}
        st.bar(k, 0.0, ev)
        times = [e[0] if isinstance(e, tuple) else e.time for v in ev.values() for e in v]
        assert times and min(times) >= -0.02 and max(times) < st.bar_s + 0.02


def test_the_crossfade_keeps_the_power():
    g_old, g_new = crossfade(4 * 48000)
    assert abs(g_old[0] - 1) < 1e-6 and g_new[0] == 0 and g_old[-1] < 1e-6 and abs(g_new[-1] - 1) < 1e-6
    np.testing.assert_allclose(g_old**2 + g_new**2, 1.0, atol=0.01)
    assert abs(np.argmin(np.abs(g_old - g_new)) / 48000 - 1.5) < 0.01  # they cross halfway through 3 s


@pytest.mark.parametrize("name", NAMES)
def test_a_chunk_rings_into_its_tail(name):
    """What sounds at a chunk's end carries on into its tail, which the next chunk is added over:
    the seam keeps the ringing notes and the reverb instead of cutting them."""
    st = STYLES[name](seed_of(name, DAY))
    c = next(chunks(lambda k: st.compose(4 + 2 * k, 2), tail=2.5))
    length = 2 * st.bar_s
    end = round(length * 48000)
    tail = c.samples[end:end + 4800]
    assert np.sqrt(np.mean(tail**2)) > 1e-3

def test_the_intro_is_the_tres_alone():
    st = STYLES["son"](3)
    assert st.section(0) == "intro" and st.section(1) == "intro" and st.section(2) == "main"


def test_menu_intensity_thins_the_percussion():
    full, thin = STYLES["son"](5), STYLES["son"](5, intensity=0.5)
    a, b = render(full.compose(4, 2)[0], 4.0), render(thin.compose(4, 2)[0], 4.0)
    assert np.sqrt(np.mean(b**2)) < np.sqrt(np.mean(a**2))


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_each_style_renders_within_budget(name):
    st = STYLES[name](seed_of(name, DAY))
    t = time.perf_counter()
    unit, length = st.compose(4, 4)
    render(unit, length + 2.5)
    assert (time.perf_counter() - t) / length < 0.35


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_each_style_passes_the_audio_lint(name):
    from euterpe.lint import lint_audio

    st = STYLES[name](seed_of(name, DAY))
    unit, length = st.compose(0, 32)
    x = render(unit, length + 2.5)
    assert not lint_audio(x).errors, lint_audio(x).to_text()


def test_the_music_process_fills_the_stream(tmp_path, monkeypatch):
    """The game's sound with a file backend: the music bus's stem carries music within seconds."""
    import wave
    from types import SimpleNamespace

    from test_sound import fake_game

    from contravia.audio import Sound
    from contravia.settings import Options

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    s = Sound.start(SimpleNamespace(), Options.load(), backend="file", path=tmp_path / "drive.wav")
    g = fake_game(1000.0)
    s.attach(g)
    t = time.time()
    while time.time() - t < 60 and s.audio.state()["levels"]["music"] < -50:
        s.frame(g)
        time.sleep(1 / 30)
    level = s.audio.state()["levels"]["music"]
    time.sleep(1.0)
    s.close()
    assert level > -50
    with wave.open(str(tmp_path / "drive.music.wav")) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    assert np.abs(x).max() > 300
