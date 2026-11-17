import os
import subprocess
import sys
import time

import numpy as np
import pytest

from contravia.clock import Clock, parse_at
from contravia.settings import FreeDrive, Options


def test_free_drive_rows_follow_the_flags():
    fd = FreeDrive.from_flags(when="Sun 08:00", force="rain", traffic=True, warp=10.0)
    assert fd["hour"].text == "08:00" and fd["day"].text == "Domingo"
    assert fd["weather"].text == "Lluvia" and fd["traffic"].value is True and fd["warp"].text == "×10"
    start = fd.start()
    assert start["when"] == "sun 08:00" and start["force"] == "rain" and start["warp"] == 10.0
    assert np.hypot(*start["at"]) < 1.0  # the Capitolio is the map's origin
    parse_at(start["when"])  # the clock reads it


def test_a_start_point_from_the_command_line_is_the_first_place():
    fd = FreeDrive.from_flags(at=(120.0, -40.0))
    assert fd["place"].text == "Punto de partida" and fd.start()["at"] == (120.0, -40.0)
    fd["place"].step(-1)
    assert fd.start()["at"] is None  # "Al azar": a random road


def test_options_are_saved_and_read_back(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    opts = Options.load()
    assert opts["units"].value == "kmh"
    opts["units"].step(1)
    opts["view"].step(1)
    opts.save()
    again = Options.load()
    assert again["units"].value == "mph" and again["view"].text == opts["view"].text


def test_clock_pauses_and_resumes_where_it_was():
    c = Clock(at=parse_at("2026-02-10 12:00"), warp=3600.0)
    c.pause()
    t = c.now()
    time.sleep(0.05)
    assert c.now() == t
    c.resume()
    time.sleep(0.05)
    assert c.now() > t


def test_menu_goes_from_main_to_driving_to_pause_and_back(city_npz, tmp_path, monkeypatch):
    import manifoldx as mx

    from contravia.game import Game
    from contravia.menu import Menu
    from contravia.world import World

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    engine = mx.Engine("menu", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 10:00")))
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 17:30"), Options.load())
    engine.render_frame(supersample=1)
    assert menu.current == "main" and game.mode == "menu"
    menu.press("ArrowUp")  # Historia: coming soon, stays here
    menu.press("Enter")
    assert menu.current == "main" and "Próximamente" in menu.screens["main"].note.text
    menu.press("ArrowDown")
    menu.press("Enter")  # Modo libre
    assert menu.current == "free"
    menu.press("Enter")  # Arrancar
    assert menu.current is None and game.mode == "drive"
    assert game.clock.now().strftime("%a %H:%M") == "Tue 17:30"
    engine.render_frame(supersample=1)
    menu.press("Escape")
    assert menu.current == "pause" and game.mode == "paused" and game.clock.paused
    for _ in range(4):
        menu.press("ArrowDown")  # Menú principal
    menu.press("Enter")
    assert menu.current == "main" and game.mode == "menu"
    game.streamer.close()
    game.detail.close()


@pytest.mark.parametrize("screen, top", [("main", 250 + 72), ("pause", 140)])  # the chosen option's row
def test_a_menu_screen_draws_its_yellow_stripe(city_npz, tmp_path, screen, top):
    from PIL import Image

    out = tmp_path / f"{screen}.png"
    env = {**os.environ, "CONTRAVIA_CONFIG": str(tmp_path)}
    subprocess.run([sys.executable, "-m", "contravia.cli", "play", "--data", str(city_npz), "--screen", screen,
                    "--shot", str(out), "--frames", "10"], check=True, timeout=600, env=env)
    img = np.asarray(Image.open(out).convert("RGB"), int)
    box = img[top + 8: top + 60, 90: 380]  # the shot is 1280 x 720, the GUI's own pixels
    yellow = (box[..., 0] > 230) & (box[..., 1] > 190) & (box[..., 1] < 225) & (box[..., 2] < 40)
    assert yellow.mean() > 0.4  # the chosen option's stripe, #ffd400, under its dark letters
