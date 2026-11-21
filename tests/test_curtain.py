import numpy as np
import pytest

from contravia.clock import Clock, parse_at
from contravia.curtain import FADE, TRUST, Curtain, Waiting, ready
from contravia.settings import FreeDrive, Options


def test_ready_only_when_nothing_waits():
    assert ready(Waiting())
    for field in ("nodes", "cells", "detail", "trees", "grass", "tombs", "traffic"):
        assert not ready(Waiting(**{field: 1})), field


def test_the_start_runs_on_the_second_step_and_counts_are_trusted_after_it():
    ran = []
    c = Curtain(0.0, then=lambda: ran.append(True))
    assert c.step(0.0, Waiting()) == 1.0 and not ran  # the black frame is drawn before the start
    assert c.step(0.1, Waiting()) == 1.0 and ran == [True]
    for k in range(TRUST - 1):  # counts so soon after the start may be the menu's place's: not trusted
        assert c.step(0.2 + k * 0.1, Waiting()) == 1.0
    assert c.lift_at is None


def test_it_stays_while_something_waits_then_fades_out():
    c = Curtain(0.0)
    t = 0.0
    for _ in range(50):
        t += 0.1
        assert c.step(t, Waiting(nodes=2, trees=1)) == 1.0
    c.step(t + 0.1, Waiting())
    assert c.why == "ready" and c.held == pytest.approx(t + 0.1)
    assert c.step(t + 0.1 + FADE / 2, Waiting()) == pytest.approx(0.5)
    assert c.step(t + 0.11 + FADE, Waiting()) == 0.0 and not c.up


def test_it_lifts_at_the_cap_whatever_still_waits():
    c = Curtain(0.0, cap=20.0)
    for k in range(1, 300):
        c.step(k * 0.1, Waiting(cells=1) if k > 3 else None)
        if c.lift_at is not None:
            break
    assert c.why == "cap" and c.held == pytest.approx(20.0)


def _menu(city_npz, tmp_path, monkeypatch):
    import manifoldx as mx

    from contravia.game import Game
    from contravia.menu import Menu
    from contravia.world import World

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    engine = mx.Engine("curtain", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 10:00")))
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 17:30"), Options.load())
    engine.render_frame(supersample=1)
    return engine, game, menu


def _close(game):
    game.streamer.close()
    game.detail.close()


def _black(img) -> bool:
    return bool(np.asarray(img)[..., :3].max(axis=-1)[:4, :].max() < 8)  # the top rows: no logo there


def test_a_free_drive_from_the_menu_starts_under_the_curtain_and_lifts_once_ready(city_npz, tmp_path, monkeypatch):
    engine, game, menu = _menu(city_npz, tmp_path, monkeypatch)
    waits = [Waiting(nodes=4, trees=2)]
    game.waiting_round = lambda: waits[0]
    menu.press("Enter")  # Modo libre
    menu.press("Enter")  # Arrancar
    assert game.curtain is not None and game.curtain.up
    for k in range(20):  # the counts say something waits: black on every frame, the car started under it
        img = engine.render_frame(supersample=1, dt=0.05)
        assert _black(img), k
    assert game.mode == "drive" and game.clock.now().strftime("%a %H:%M") == "Tue 17:30"
    assert game.curtain.lift_at is None
    menu.press("ArrowDown")  # keys other than Escape do nothing under it
    menu.press("Enter")
    assert menu.current is None and game.mode == "drive"
    waits[0] = Waiting()
    for _ in range(2):  # the stream system counts after the drive steps the curtain: one frame late
        engine.render_frame(supersample=1, dt=0.05)
    assert game.curtain.why == "ready"
    for _ in range(200):
        if game.curtain is None:
            break
        engine.render_frame(supersample=1, dt=0.05)
    assert game.curtain is None and engine.state.curtain["why"] == "ready"
    assert not _black(engine.render_frame(supersample=1, dt=0.05))  # the city shows
    assert game.streamer.refine_ms == 2.0  # the frame-time budgets are back
    _close(game)


def test_the_real_counts_lift_it_once_the_start_is_drawn_with_traffic(city_npz, tmp_path, monkeypatch):
    import manifoldx as mx

    from contravia.game import Game
    from contravia.menu import Menu
    from contravia.world import World

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    engine = mx.Engine("curtain", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 08:30")), with_traffic=True)
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 08:30", traffic=True), Options.load())
    engine.render_frame(supersample=1)
    menu.press("Enter")
    menu.press("Enter")
    for _ in range(600):
        if game.curtain is None:
            break
        engine.render_frame(supersample=1, dt=0.05)
    assert engine.state.curtain["why"] == "ready"
    assert game.waiting() == 0 and engine.state.traffic["cars"] > 0
    _close(game)
    game.runner.stop()


def test_escape_under_the_curtain_goes_back_to_the_main_menu(city_npz, tmp_path, monkeypatch):
    engine, game, menu = _menu(city_npz, tmp_path, monkeypatch)
    game.waiting_round = lambda: Waiting(nodes=1)
    menu.press("ArrowDown")
    menu.press("Enter")  # Mapa de Cuba
    assert menu.current == "map" and game.curtain is not None
    for _ in range(3):
        engine.render_frame(supersample=1, dt=0.05)
    engine.input._just_pressed_keys = frozenset({"Escape"})  # as the window reports it
    menu.update(None, 0.05)
    assert game.curtain is None and menu.current == "main" and game.mode == "menu" and game.map is None
    _close(game)


def test_the_curtain_lifts_at_its_cap_and_never_shows_within_a_mode(city_npz, tmp_path, monkeypatch):
    engine, game, menu = _menu(city_npz, tmp_path, monkeypatch)
    game.waiting_round = lambda: Waiting(cells=3)  # never ready
    menu.press("Enter")
    menu.press("Enter")
    game.curtain.cap = 0.3
    for _ in range(300):
        if game.curtain is None:
            break
        engine.render_frame(supersample=1, dt=0.05)
    assert game.curtain is None and engine.state.curtain["why"] == "cap"
    menu.press("Escape")  # the pause, then the map: a flight, no curtain
    menu.press("ArrowDown")
    menu.press("Enter")
    assert menu.current == "map" and game.flight is not None and game.curtain is None
    _close(game)
