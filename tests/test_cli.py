import subprocess
import sys


def test_help_lists_the_three_commands():
    out = subprocess.run([sys.executable, "-m", "contravia.cli", "--help"],
                         capture_output=True, text=True, check=True).stdout
    for cmd in ("prepare", "play", "bench"):
        assert cmd in out


def test_flowmap_draws_busy_streets_brighter(city_npz, tmp_path):
    import numpy as np
    from PIL import Image

    from contravia.flowmap import main as flowmap

    out = tmp_path / "flow.png"
    totals = flowmap(str(city_npz), str(out), "Tue 08:00", size=400)
    img = np.asarray(Image.open(out).convert("RGB"), int)
    assert totals["vehicles"] > 0
    background = img[-1, -1]
    assert (np.abs(img - background).sum(axis=2) > 30).mean() > 0.01  # streets drawn


def test_play_and_flowmap_take_a_time(monkeypatch):
    from contravia import cli

    seen = {}
    monkeypatch.setattr("contravia.flowmap.main", lambda *a, **k: seen.update(a=a, k=k))
    cli.main(["flowmap", "x.png", "--time", "Sun 08:00", "--box", "0,0,10,10"])
    assert seen["a"][2] == "Sun 08:00" and seen["k"]["box"] == (0.0, 0.0, 10.0, 10.0)


def test_route_prints_the_steps_and_a_summary(island_dir, capsys):
    import re

    import numpy as np
    import pytest

    from contravia import cli
    from contravia.geo import to_latlon
    from contravia.world import World

    g = World.load(island_dir).province(0).graph
    mid = g.a["g_mid"].astype(float)
    a, b = mid[0], mid[np.argmax(np.hypot(*(mid - mid[0]).T))]
    (la, lo), (lb, lob) = to_latlon(np.array([a, b]))
    cli.main(["route", "--data", str(island_dir), "--from", f"{la},{lo}", "--to", f"{lb},{lob}"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert re.match(r"^[\d,]+ km · \d+ min · route found in \d+ ms$", lines[-1])
    assert len(lines) >= 2
    sea = to_latlon(np.array([a + (0.0, -10000.0)]))[0]
    with pytest.raises(SystemExit) as stop:
        cli.main(["route", "--data", str(island_dir), "--from", f"{la},{lo}", "--to", f"{sea[0]},{sea[1]}"])
    assert stop.value.code == 1
