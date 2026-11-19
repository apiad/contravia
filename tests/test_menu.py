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


def test_realism_is_an_option_from_easy_to_real(tmp_path, monkeypatch):
    from contravia.settings import Options

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    o = Options.load()
    row = o["realism"]
    assert row.label == "Realismo" and row.value == 30 and o.locked_assists == ()
    assert [v for _, v in row.options] == list(range(0, 101, 10))
    assert "Fácil" in row.options[0][0] and "Realista" in row.options[-1][0]
    row.pick(70)
    o.locked_assists = ("abs",)
    o.save()
    again = Options.load()
    assert again["realism"].value == 70 and again.locked_assists == ("abs",)


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
    for _ in range(5):
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


def test_the_map_opens_from_the_pause_and_the_main_menu_and_drives_at_its_crosshair(city_npz, tmp_path, monkeypatch):
    import manifoldx as mx

    from contravia import mapview
    from contravia.game import FAR, Game
    from contravia.menu import Menu
    from contravia.world import World

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    engine = mx.Engine("map", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 10:00")))
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 10:00"), Options.load())
    engine.render_frame(supersample=1)
    menu.press("ArrowDown")  # from Modo libre to Mapa de Cuba
    menu.press("Enter")
    assert menu.current == "map" and game.mode == "map" and game.map is not None
    menu.press("Escape")
    assert menu.current == "main" and game.mode == "menu" and game.map is None

    menu.press("ArrowUp")  # Modo libre
    menu.press("Enter")
    menu.press("Enter")  # Arrancar
    menu.press("Escape")
    menu.press("ArrowDown")  # Mapa
    menu.press("Enter")
    assert menu.current == "map" and game.clock.paused
    game.map = mapview.zoom(game.map, 20.0)
    engine.render_frame(supersample=1)
    assert engine.camera.far > FAR and game.eye[1] == pytest.approx(game.map.h)
    menu.press("Escape")
    assert menu.current == "pause"

    menu.press("Enter")  # the map again: the pause keeps Mapa chosen
    x, z = game.car.x, game.car.z
    road = game.walls.random_road(np.random.default_rng(3))[:2]
    game.map = mapview.MapCam(float(road[0]), float(road[1]), 800.0, 0.0)
    menu.press("Enter")  # Manejar aquí
    assert menu.current is None and game.mode == "drive" and not game.clock.paused and game.map is None
    assert np.hypot(game.car.x - road[0], game.car.z - road[1]) < 30.0
    assert (game.car.x, game.car.z) != (x, z)
    game.streamer.close()
    game.detail.close()


def _place_at(game, pid, dx, dz, side=8.0, h=10.0):
    """A collectible place dx, dz metres from the car, as places.json writes it."""
    from contravia import collect, geo

    lat, lon = geo.to_latlon([[game.car.x + dx, game.car.z + dz]])[0]
    r = {"id": pid, "title": f"Lugar {pid}", "type": "monumento", "province": "La Habana", "municipio": "Plaza",
         "text": "Un texto.", "photo_target": "", "lat": lat, "lon": lon,
         "box": {"w": side, "d": side, "h": h, "yaw_deg": 0.0},
         "scores": {"h": 1, "c": 1, "v": 1, "total": 3, "tier": "débil"}}
    return collect.Places([r]).by_id(pid)


def test_f_aims_at_the_spot_space_says_what_is_wrong_and_escape_drives_on(city_npz):
    import manifoldx as mx

    from contravia import collect
    from contravia.game import Game
    from contravia.menu import Menu
    from contravia.world import World

    engine = mx.Engine("aim", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 10:00")))
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 10:00"), Options.load(), screen=None)
    try:
        engine.render_frame(supersample=1)
        game.spot = None
        menu.press("f")
        assert menu.current is None and game.mode == "drive"  # nothing to aim at
        game.spot = _place_at(game, "lejos", 250.0, 0.0)
        game.walls.near = lambda *a, **k: np.zeros((0, 2, 2))
        menu.press("f")
        assert menu.current == "aim" and game.mode == "aim"
        assert game.aim["zoom"] == collect.ZOOM_MAX  # too small to fill the frame even zoomed in
        engine.render_frame(supersample=1)
        assert engine.camera.fov == pytest.approx(collect.fov_for(collect.ZOOM_MAX))
        menu.press("Space")
        assert menu.screens["aim"].note.text == "Muy lejos"
        menu.press("Escape")
        assert menu.current is None and game.mode == "drive" and game.aim is None
        engine.render_frame(supersample=1)
        assert engine.camera.fov == pytest.approx(collect.FOV_DEG)
    finally:
        game._shutdown()


def test_a_good_shot_saves_the_photo_fills_the_album_and_shows_the_card(city_npz):
    from concurrent.futures import Future

    import manifoldx as mx
    from PIL import Image

    from contravia.game import PHOTO_SIZE, Game
    from contravia.menu import Menu
    from contravia.world import World

    engine = mx.Engine("shot", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 10:00")))
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 10:00"), Options.load(), screen=None)
    read = []

    def read_texture(texture):
        read.append(texture)
        f = Future()
        f.set_result(np.full((PHOTO_SIZE[1], PHOTO_SIZE[0], 3), 200, np.uint8))
        return f

    engine.read_texture = read_texture
    try:
        engine.render_frame(supersample=1)
        from contravia import album, collect

        game.spot = _place_at(game, "cerca", 40.0, 0.0, side=12.0, h=14.0)
        game.places = collect.Places([])
        game.places._all = [game.spot]
        game.album = album.Album.load({"cerca": "La Habana"})
        game.walls.near = lambda *a, **k: np.zeros((0, 2, 2))
        menu.press("f")
        assert game.shoot().verdict == "good"  # the starting zoom frames it
        menu.press("Space")
        assert menu.current == "card" and read
        assert menu.card["count"] == "NUEVO · 1 de 1 en La Habana" and "mar 10:00" in menu.card["meta"]
        assert game.album.has("cerca") and game.spot is None
        engine.render_frame(supersample=1)  # the next frame writes the PNG
        png = game.album.photo_path("cerca")
        assert Image.open(png).size == PHOTO_SIZE
        menu.press("Enter")
        assert menu.current is None and game.mode == "drive"
    finally:
        game._shutdown()
