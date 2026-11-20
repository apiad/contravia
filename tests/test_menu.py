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


def test_driving_help_is_an_option_from_autopilot_to_real(tmp_path, monkeypatch):
    from contravia.settings import Options

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    o = Options.load()
    row = o["help"]
    assert row.label == "Ayuda al manejo" and row.value == 3 and o.locked_assists == ()
    assert [v for _, v in row.options] == list(range(6))
    assert row.options[0][0] == "Casi piloto" and row.options[-1][0] == "Real"
    row.pick(5)
    o.locked_assists = ("abs",)
    o.save()
    again = Options.load()
    assert again["help"].value == 5 and again.locked_assists == ("abs",)


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
    for _ in range(6):
        menu.press("ArrowDown")  # Menú principal
    menu.press("Enter")
    assert menu.current == "main" and game.mode == "menu"
    game.streamer.close()
    game.detail.close()


@pytest.mark.parametrize("screen, top", [("main", 214 + 64), ("pause", 140)])  # the chosen option's row
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

    from contravia import flight, mapview
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
    engine.render_frame(supersample=1)
    menu.press("Escape")
    menu.press("ArrowDown")  # Mapa
    eye = game.eye.copy()
    menu.press("Enter")
    assert menu.current == "map" and game.clock.paused and game.flight is not None
    assert game.map.h == flight.UP_H and (game.map.x, game.map.z) == (game.car.x, game.car.z)
    engine.render_frame(supersample=1, dt=0.05)
    assert np.linalg.norm(game.eye - eye) < 5.0  # the flight up starts on the chase camera
    _fly(engine, game)
    assert np.allclose(game.eye, mapview.pose(game.map)[0])
    game.map = mapview.zoom(game.map, 20.0)
    engine.render_frame(supersample=1)
    assert engine.camera.far > FAR and game.eye[1] == pytest.approx(game.map.h)
    menu.press("Escape")
    assert menu.current == "pause"

    menu.press("Enter")  # the map again: the pause keeps Mapa chosen
    _fly(engine, game)
    x, z = game.car.x, game.car.z
    road = game.walls.random_road(np.random.default_rng(3))[:2]
    game.map = mapview.MapCam(float(road[0]), float(road[1]), 800.0, 0.0)
    engine.render_frame(supersample=1)
    menu.press("Enter")  # Manejar aquí
    assert menu.current is None and game.flight is not None and game.map is None
    engine.render_frame(supersample=1, dt=0.05)
    assert (game.car.x, game.car.z) == (x, z) and game.eye[1] > 100.0  # the car waits for the camera to land
    _fly(engine, game)
    assert game.mode == "drive" and not game.clock.paused
    assert np.hypot(game.car.x - road[0], game.car.z - road[1]) < 30.0
    landed = game.eye.copy()
    engine.render_frame(supersample=1, dt=0.05)
    assert np.linalg.norm(game.eye - landed) < 1.0  # the chase camera takes over with no jump
    game.streamer.close()
    game.detail.close()


def test_the_map_screen_opens_as_the_shot_path_opens_it(city_npz, tmp_path, monkeypatch):
    import math

    import manifoldx as mx

    from contravia import mapview
    from contravia.game import Game
    from contravia.menu import Menu
    from contravia.world import World

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    engine = mx.Engine("map", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), at=(0.0, 0.0), clock=Clock(at=parse_at("Tue 10:00")))
    game.map = mapview.zoom(mapview.MapCam(0.0, 0.0, 1.0, math.radians(30.0)), 2500.0)  # --height 2500 --yaw 30
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 10:00"), Options.load(), screen=None)
    engine.on("startup")(lambda _=None: (menu.pause(), menu.show("map")))  # as main() opens --screen map
    for _ in range(3):
        engine.render_frame(supersample=1, dt=0.05)
    assert menu.current == "map" and game.flight is None and game.map.h == pytest.approx(2500.0)
    game.streamer.close()
    game.detail.close()


def _fly(engine, game, frames=200):
    """Render frames until the camera's flight lands."""
    for _ in range(frames):
        if game.flight is None:
            return
        engine.render_frame(supersample=1, dt=0.05)
    raise AssertionError("the flight never landed")


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


def test_the_map_names_the_place_under_its_crosshair_offers_its_card_and_joins_a_municipio(
        city_npz, tmp_path, monkeypatch):
    import manifoldx as mx

    from contravia import mapview, placemarks, ui
    from contravia.clock import Clock, parse_at
    from contravia.game import Game
    from contravia.menu import Menu
    from contravia.world import World

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    engine = mx.Engine("map", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 10:00")))
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 10:00"), Options.load())
    engine.render_frame(supersample=1)
    capitolio = game.places.by_id("L0333")
    game.album.photo_path(capitolio.id).parent.mkdir(parents=True, exist_ok=True)
    game.album.photo_path(capitolio.id).write_bytes(_png())
    game.album.add(capitolio.id, "10:00", "Despejado")
    menu.press("ArrowDown")  # Mapa de Cuba
    menu.press("Enter")
    screen = menu.screens["map"]
    game.map = mapview.MapCam(capitolio.x, capitolio.z, 600.0, 0.0)
    screen.tick(0.0)
    assert screen.note.text == capitolio.title and screen.under.got

    menu.press("Enter")  # the choice
    assert screen.choice is not None and screen.choice_panel in engine.gui._panels
    menu.press("Enter")  # Ver ficha
    assert menu.current == "card" and menu.card["count"].startswith("1 de ")
    assert "base64" in menu.card["texture"] and "Despejado" in menu.card["meta"]
    menu.press("Escape")
    assert menu.current == "map" and game.map.x == capitolio.x and screen.back_to == "main"

    other = next(p for p in game.places if not game.album.has(p.id))
    menu.view_card(other)
    assert menu.card["texture"] == ui.no_photo(480, 270)
    menu.press("Enter")
    assert menu.current == "map"

    game.map = mapview.MapCam(capitolio.x, capitolio.z, placemarks.CITY_H * 2, 0.0)
    group = next(m for m in placemarks.on_map(game.places, game.album.has, game.map, screen.width, ui.DESIGN_H)
                 if capitolio.id in m.ids)
    game.map = mapview.MapCam(*group.at, placemarks.CITY_H * 2, 0.0)
    screen.tick(0.0)
    assert screen.under is not None and len(screen.under.ids) > 1 and " lugares" in screen.note.text
    menu.press("Enter")  # comes down over the group
    assert game.map.h < placemarks.CITY_H and menu.current == "map" and screen.choice is None


def _png():
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    raw = b"".join(b"\x00" + b"\x80\x40\x20" * 4 for _ in range(3))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 3, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_the_album_counts_by_province_opens_a_province_page_and_its_cards(city_npz, tmp_path, monkeypatch):
    import manifoldx as mx

    from contravia.clock import Clock, parse_at
    from contravia.game import Game
    from contravia.menu import PAGE_COLS, PAGE_ROWS, Menu
    from contravia.world import World

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    engine = mx.Engine("album", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 10:00")))
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 10:00"), Options.load())
    engine.render_frame(supersample=1)
    capitolio = game.places.by_id("L0333")
    game.album.photo_path(capitolio.id).parent.mkdir(parents=True, exist_ok=True)
    game.album.photo_path(capitolio.id).write_bytes(_png())
    game.album.add(capitolio.id, "10:00", "Despejado")
    for _ in range(3):
        menu.press("ArrowDown")  # Historia is "próximamente", so the list starts on Modo libre: down to Álbum
    menu.press("Enter")
    screen = menu.screens["album"]
    assert menu.current == "album" and screen.back_to == "main"
    assert screen.rows[0][0] == "La Habana" and screen.rows[0][1] == 1  # the largest province first
    assert sum(r[2] for r in screen.rows) == len(game.places)

    menu.press("Enter")
    page = menu.screens["province"]
    assert menu.current == "province" and page.places[0].id == capitolio.id  # the photographed first
    menu.press("Enter")
    assert menu.current == "card" and menu.card["back"] == "province"
    menu.press("Escape")
    assert menu.current == "province"
    for _ in range(PAGE_ROWS):
        menu.press("ArrowDown")  # off the first page
    assert page.page == 1 and page.index == PAGE_COLS * PAGE_ROWS
    menu.press("Escape")
    menu.press("Escape")
    assert menu.current == "main"


def test_the_maps_panel_names_what_lies_near_the_crosshair_at_each_height(city_npz, tmp_path, monkeypatch):
    import manifoldx as mx

    from contravia import mapview
    from contravia.game import Game
    from contravia.menu import Menu
    from contravia.nearby import band
    from contravia.world import World

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    engine = mx.Engine("map", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 10:00")))
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 10:00"), Options.load())
    engine.render_frame(supersample=1)
    menu.press("ArrowDown")  # Mapa de Cuba
    menu.press("Enter")
    screen = menu.screens["map"]
    capitolio = game.places.by_id("L0333")

    def settle(h):
        game.map = mapview.MapCam(capitolio.x, capitolio.z, h, 0.0)
        end = time.monotonic() + 10
        while time.monotonic() < end:
            screen.tick(0.0)
            if screen.shown and screen.shown[0].band == band(h).name:  # the worker answered for this height
                return [row[2].text for row in screen.here_rows if row[0] in screen.here_panel.children]
            time.sleep(0.01)
        raise AssertionError(f"no panel at {h} m")

    close = settle(300.0)  # the fixture city's streets are "Calle 0", "Calle 1"...
    assert screen.here_panel in screen.panels and screen.here_title.text.startswith("Calle ")
    assert close[0] == capitolio.title and screen.here_header.text == "AQUÍ"
    high = settle(500_000.0)
    assert screen.here_title.text == "La Habana" and len(high) == 25 and screen.here_header.text == "LO MÁS NOTABLE" and capitolio.title in high


def test_l_in_drive_mode_turns_the_driving_help_and_saves_it(city_npz):
    import manifoldx as mx

    from contravia.driving_help import LEVELS
    from contravia.game import Game
    from contravia.menu import Menu
    from contravia.world import World

    engine = mx.Engine("help", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 10:00")))
    options = Options.load()
    options.locked_assists = ("abs",)
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 10:00"), options, screen=None)
    try:
        engine.render_frame(supersample=1)
        assert game.mode == "drive" and game.help_level == 3
        np.testing.assert_allclose(game.strengths(), [1, 1, 0, 1])
        menu.press("l")
        assert menu.current is None and game.help_level == 2
        menu.press("l")
        menu.press("l")
        menu.press("l")
        assert game.help_level == 5
        np.testing.assert_allclose(game.strengths(), [0, 0, 0, 0])
        menu.press("l")
        assert game.help_level == 4
        np.testing.assert_allclose(game.strengths(), [0.0, 0.6, 0.0, 0.6])  # abs stays locked off
        assert Options.load()["help"].value == 4
        assert game.hud.note_text.text == f"AYUDA 4 · {LEVELS[4][0].upper()}"
        menu.show("options")
        menu.press("l")
        assert game.help_level == 4  # a menu on the screen does not read L
    finally:
        game._shutdown()
