import subprocess
import sys
import time

import numpy as np
from PIL import Image


def test_play_renders_a_shot_with_streets(city_npz, tmp_path):
    out = tmp_path / "shot.png"
    clear = tmp_path / "clear.json"
    clear.write_text('{"force": "clear"}')  # today's real weather could bring the fog in
    subprocess.run([sys.executable, "-m", "contravia.cli", "play", "--data", str(city_npz), "--time", "2026-02-10 12:00",
                    "--sky", str(clear), "--shot", str(out), "--frames", "60"], check=True, timeout=600)
    import numpy as np

    img = np.asarray(Image.open(out).convert("RGB"), float)
    lower = img[img.shape[0] // 2:]
    sky = np.abs(lower - img[0, 0]).max(axis=2) < 6     # the top-left corner is always sky
    assert sky.mean() < 0.2        # the streamed cells cover the ground below the horizon


def test_teleport_moves_the_car_to_a_road(city):
    from contravia.walls import Walls

    x, z, _ = Walls(city).nearest_road(9000, -2000)
    assert abs(x) < 1000 and abs(z) < 1000   # the fixture has roads only near the origin


def test_teleport_far_keeps_precision_and_sea_reports(city, city_npz, island_dir):
    import os

    import manifoldx as mx
    import numpy as np

    from contravia.game import Game
    from contravia.world import World

    old = os.environ["HAVANA_DATA"]
    os.environ["HAVANA_DATA"] = str(island_dir)
    try:
        engine = mx.Engine("far", width=64, height=48)
        game = Game(engine, World.load(island_dir))
        game.teleport(900_300.0, 0.0)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:  # the builders are other processes: wait for the first node
            engine.render_frame(supersample=1)
            if any(h is not None for h in game.streamer._drawn.values()):
                break
        assert abs(game.car.x - 900_000.0) < 100
        ox, oz = engine.origin
        assert np.hypot(game.car.x - ox, game.car.z - oz) < 2000
        # The streamed nodes around the car draw within 20 km of the origin (the landmarks, at the Capitolio,
        # are rightly 900 km away and past the far plane).
        drawn = [h for h in game.streamer._drawn.values() if h is not None]
        assert drawn
        rel = np.array([np.asarray(h.transform.pos, float) for h in drawn]) - (ox, 0.0, oz)
        assert np.abs(rel).max() < 20_000
        before = (game.car.x, game.car.z)
        game.teleport(450_000.0, 300_000.0)
        engine.render_frame(supersample=1)
        assert engine.state.teleport["error"] and (game.car.x, game.car.z) == before
        game.streamer.close()
        if game.layer:
            game.runner.stop()
        assert game.city.focused == "Artemisa" or game.city.focus(game.car.x, game.car.z) is False
    finally:
        os.environ["HAVANA_DATA"] = old


def test_a_reload_drives_on_from_the_same_place_and_stops_the_old_game(city):
    import manifoldx as mx

    from contravia.game import Game
    from contravia.world import World

    engine = mx.Engine("old", width=64, height=48)
    game = Game(engine, World(city))
    x, z, _ = game.walls.nearest_road(300.0, 200.0)
    game.teleport(x, z)
    for _ in range(3):
        engine.render_frame(supersample=1)
    pose = list(engine.state.car["pose"])
    game._shutdown()
    assert not game.layer or not game.runner._thread.is_alive()

    engine2 = mx.Engine("new", width=64, height=48)  # under contravia dev both engines share one state
    engine2.state.update(engine.state)
    game2 = Game(engine2, World(city))
    try:
        assert [game2.car.x, game2.car.z, game2.car.yaw] == pose[:3]
        assert abs(game2.car.x) > 50  # not back at the start
    finally:
        game2._shutdown()


def test_slow_frames_raise_tau_and_fast_ones_lower_it(city):
    import manifoldx as mx

    from contravia import game as g
    from contravia.world import World

    engine = mx.Engine("tune", width=64, height=48)
    game = g.Game(engine, World(city))
    for ms, check in ((40.0, lambda t: t > g.TAU[0]), (5.0, lambda t: t == g.TAU[0])):
        for _ in range(g.TUNE_EVERY * 20):
            game.frame_ms = ms
            game._tune()
        assert check(game.streamer.tau)
    for _ in range(g.TUNE_EVERY * 40):
        game.frame_ms = 100.0
        game._tune()
    assert game.streamer.tau == g.TAU[1]
    game.streamer.close()


def test_a_game_can_start_in_another_province(island_dir):
    import manifoldx as mx

    from contravia.game import Game
    from contravia.world import World

    engine = mx.Engine("start-far", width=64, height=48)
    game = Game(engine, World.load(island_dir), at=(900_000.0, 0.0))
    assert game.city.focused == "Artemisa" and abs(game.car.x - 900_000.0) < 100
    game.streamer.close()
    game.detail.close()


def test_the_spot_is_the_nearest_place_in_reach_and_sight_without_a_photo(city):
    import manifoldx as mx

    from contravia import collect, geo
    from contravia.game import Game
    from contravia.world import World

    engine = mx.Engine("spot", width=64, height=48)
    game = Game(engine, World(city))
    try:
        x, z, _ = game.walls.nearest_road(300.0, 200.0)
        game.teleport(x, z)

        def place(pid, dx, dz):
            lat, lon = geo.to_latlon([[game.car.x + dx, game.car.z + dz]])[0]
            return {"id": pid, "title": pid, "type": "monumento", "province": "La Habana", "municipio": "Plaza",
                    "text": "", "photo_target": "", "lat": lat, "lon": lon,
                    "box": {"w": 8.0, "d": 8.0, "h": 10.0, "yaw_deg": 0.0},
                    "scores": {"h": 1, "c": 1, "v": 1, "total": 3, "tier": "débil"}}

        game.places = collect.Places([place("near", 0.0, 0.0), place("far", 400.0, 0.0)])
        game.walls.near = lambda *a, **k: np.zeros((0, 2, 2))  # an open field: nothing in the way
        game._find_spot(1e9)
        assert game.spot.id == "near"
        game.album.add("near", "lun 10:00", "despejado")
        game._find_spot(2e9)
        assert game.spot is None  # the other one is out of reach
    finally:
        game._shutdown()


def test_streamed_nodes_fade_in_instead_of_popping(city):
    import time

    import manifoldx as mx

    from contravia.game import Game

    engine = mx.Engine("fade", width=64, height=48)
    game = Game(engine, city)
    s = game.streamer
    assert s.crossfade  # the game turns the fade on
    s.crossfade = 30.0  # long enough that even a slow CI frame lands halfway
    seen = set()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 60 and not any(0.0 < v < 1.0 for v in seen):
        engine.render_frame(supersample=1)
        fades = engine.store._components.get("Fade")
        seen |= {round(float(np.ravel(fades[h.index])[0]), 2) for h, _, out in s._fading.values() if not out}
    assert any(0.0 < v < 1.0 for v in seen)  # caught halfway in, not popped
    s.crossfade = 0.01
    t0 = time.monotonic()
    while time.monotonic() - t0 < 60 and s._fading:
        engine.render_frame(supersample=1)
    fades = engine.store._components["Fade"]
    drawn = [h for h in s._drawn.values() if h is not None]
    assert drawn and all(np.ravel(fades[h.index])[0] == 1.0 for h in drawn)  # and whole once faded in
    s.close()
    if game.layer:
        game.runner.stop()


def test_dev_reloads_when_the_region_rules_or_a_side_file_change(tmp_path, monkeypatch):
    import os

    import pytest
    from manifoldx import hot

    from contravia import game

    class Built(Exception):
        pass

    def stop(data):
        raise Built  # the engine and its watch list are made; the world is not needed

    monkeypatch.setattr(hot, "_watched", [])  # as manifoldx run sets it while it loads the app
    monkeypatch.setattr(game.World, "load", stop)
    with pytest.raises(Built):
        game.main(str(tmp_path))
    rules = os.path.join(os.path.dirname(game.__file__), "regions.toml")
    assert os.path.isfile(rules)
    assert [os.fspath(p) for p in hot._watched] == [rules, os.path.join(str(tmp_path), "*.src")]
