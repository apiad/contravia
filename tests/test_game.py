import subprocess
import sys

from PIL import Image


def test_play_renders_a_shot_with_streets(city_npz, tmp_path):
    out = tmp_path / "shot.png"
    clear = tmp_path / "clear.json"
    clear.write_text('{"force": "clear"}')  # today's real weather could bring the fog in
    subprocess.run([sys.executable, "-m", "caiman.cli", "play", "--data", str(city_npz), "--time", "2026-02-10 12:00",
                    "--sky", str(clear), "--shot", str(out), "--frames", "60"], check=True, timeout=600)
    import numpy as np

    img = np.asarray(Image.open(out).convert("RGB"), float)
    lower = img[img.shape[0] // 2:]
    sky = np.abs(lower - img[0, 0]).max(axis=2) < 6     # the top-left corner is always sky
    assert sky.mean() < 0.2        # the streamed cells cover the ground below the horizon


def test_teleport_moves_the_car_to_a_road(city):
    from caiman.walls import Walls

    x, z, _ = Walls(city).nearest_road(9000, -2000)
    assert abs(x) < 1000 and abs(z) < 1000   # the fixture has roads only near the origin


def test_teleport_far_keeps_precision_and_sea_reports(city, city_npz, island_dir):
    import os

    import manifoldx as mx
    import numpy as np

    from caiman.game import Game
    from caiman.tiles import Tiles
    from caiman.world import World

    os.environ["HAVANA_TILES"] = str(island_dir)
    try:
        engine = mx.Engine("far", width=64, height=48)
        game = Game(engine, World(city, Tiles(island_dir)))
        game.teleport(900_300.0, 0.0)
        for _ in range(5):
            engine.render_frame(supersample=1)
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
    finally:
        del os.environ["HAVANA_TILES"]
