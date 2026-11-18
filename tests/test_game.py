import subprocess
import sys
import time

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
