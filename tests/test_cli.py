import subprocess
import sys


def test_help_lists_the_three_commands():
    out = subprocess.run([sys.executable, "-m", "havana.cli", "--help"],
                         capture_output=True, text=True, check=True).stdout
    for cmd in ("prepare", "play", "bench"):
        assert cmd in out


def test_flowmap_draws_busy_streets_brighter(city_npz, tmp_path):
    import numpy as np
    from PIL import Image

    from havana.flowmap import main as flowmap

    out = tmp_path / "flow.png"
    totals = flowmap(str(city_npz), str(out), "Tue 08:00", size=400)
    img = np.asarray(Image.open(out).convert("RGB"), int)
    assert totals["vehicles"] > 0
    background = img[-1, -1]
    assert (np.abs(img - background).sum(axis=2) > 30).mean() > 0.01  # streets drawn


def test_play_and_flowmap_take_a_time(monkeypatch):
    from havana import cli

    seen = {}
    monkeypatch.setattr("havana.flowmap.main", lambda *a, **k: seen.update(a=a, k=k))
    cli.main(["flowmap", "x.png", "--time", "Sun 08:00", "--box", "0,0,10,10"])
    assert seen["a"][2] == "Sun 08:00" and seen["k"]["box"] == (0.0, 0.0, 10.0, 10.0)
