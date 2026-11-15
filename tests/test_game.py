import subprocess
import sys

from PIL import Image


def test_play_renders_a_shot_with_streets(city_npz, tmp_path):
    out = tmp_path / "shot.png"
    subprocess.run([sys.executable, "-m", "havana.cli", "play", "--data", str(city_npz),
                    "--shot", str(out), "--frames", "60"], check=True, timeout=600)
    import numpy as np

    img = np.asarray(Image.open(out).convert("RGB"), float)
    lower = img[img.shape[0] // 2:]
    sky = np.abs(lower - img[0, 0]).max(axis=2) < 6     # the top-left corner is always sky
    assert sky.mean() < 0.2        # the streamed cells cover the ground below the horizon


def test_teleport_moves_the_car_to_a_road(city):
    from havana.walls import Walls

    x, z, _ = Walls(city).nearest_road(9000, -2000)
    assert abs(x) < 1000 and abs(z) < 1000   # the fixture has roads only near the origin
