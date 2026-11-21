import numpy as np
import pytest

from contravia.bench import autopilot, road_path, summary
from manifoldx.vehicles import FlatGround, strengths_for

from contravia.car import CarState, Driven
from contravia.rules import fleet, handling


def test_autopilot_follows_a_square():
    path = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    d, i = Driven(handling.spec_for(fleet.moderno(), "moderno"), CarState(0, 0, 0, 0), FlatGround()), 0
    for _ in range(60 * 40):
        c, i = autopilot(d.state, path, i)
        d.step(c, 1 / 60, np.zeros((0, 2, 2), np.float32), strengths_for(0))
    assert i == len(path) - 1


def test_summary_percentiles():
    r = summary(list(range(1, 101)), [0.5, 1.5])
    assert r["p50_ms"] == 50.5 and r["max_ms"] == 100 and r["worst_cell_wait_s"] == 1.5


def test_road_path_follows_connected_segments(city):
    path = road_path(city, [(0.0, 0.0), (190.0, 200.0)])   # opposite corners of the street grid
    ends = {(round(float(x), 1), round(float(z), 1)) for r in city.a["roads"] for x, z in (r[0:2], r[2:4])}
    assert len(path) > 4
    assert all((round(float(x), 1), round(float(z), 1)) in ends for x, z in path)
    assert np.hypot(*np.diff(path, axis=0).T).max() <= 50.0 + 1e-3


def test_road_path_refuses_stops_with_no_road_between(city):
    with pytest.raises(ValueError):
        road_path(city, [(0.0, 0.0), (-80.0, -100.0)])   # the grid and the separate primary road


def test_scene_file_and_flight_path():
    import json

    from contravia.bench import SCENES, flight_path
    from contravia.geo import to_local

    scenes = json.loads(SCENES.read_text())
    assert {"santiago", "hill", "flight"} <= set(scenes)
    f = scenes["flight"]
    path = flight_path(f)
    step = np.hypot(*np.diff(path, axis=0).T)
    assert np.allclose(step, f["speed"]) and len(path) == f["seconds"] + 1
    toward = to_local(*f["toward"]) - path[0]
    heading = path[-1] - path[0]
    assert np.dot(toward, heading) / (np.hypot(*toward) * np.hypot(*heading)) > 0.9999  # straight at Santa Clara


def test_pipelined_table_puts_both_runs_under_each_time():
    from bench.pipelined import table

    run = {"frame_ms": 21.5, "update_ms": 20.1, "render_ms": 1.2, "overlap_ms": 0.0,
           "slowest": [(9.9, "Game._frame")]}
    text = table({"Tue 08:00": {"serial": run, "pipelined": run}}, ["Tue 08:00"])
    assert "| Tue 08:00 | serial | 21.5 | 20.1 | 1.2 | 0.0 | Game._frame 9.9 |" in text
    assert "| Tue 08:00 | pipelined |" in text


def test_play_takes_pipelining_off(monkeypatch):
    import contravia.cli
    import contravia.game

    seen = {}
    monkeypatch.setattr(contravia.game, "main", lambda *a, **k: seen.update(k))
    contravia.cli.main(["play", "--data", "data/cuba"])
    assert seen["pipelined"] is True
    contravia.cli.main(["play", "--data", "data/cuba", "--no-pipeline"])
    assert seen["pipelined"] is False


def test_showcase_scenes_and_their_camera_paths():
    from contravia import showcase

    scenes = showcase.load()
    assert set(scenes) == {"malecon", "obispo", "loma"}
    for s in scenes.values():
        assert len(s["when"].split()) == 2  # a fixed date: the light must not change with the day of the run
        assert len(s["at"]) == 2 and len(s["path"]) >= 2 and all(len(p) == 6 for p in s["path"])
    s = {"path": [[0, 10, 0, 100, 0, 0], [100, 20, 0, 200, 0, 0]]}
    assert showcase.path_at(s, 0.0, 30.0) == ((0.0, 10.0, 0.0), (100.0, 0.0, 0.0))
    assert showcase.path_at(s, 15.0, 30.0) == ((50.0, 15.0, 0.0), (150.0, 0.0, 0.0))
    assert showcase.path_at(s, 99.0, 30.0) == ((100.0, 20.0, 0.0), (200.0, 0.0, 0.0))


def test_showcase_runs_each_scene_and_saves_a_shot(city, tmp_path):
    import manifoldx as mx

    from contravia import showcase
    from contravia.game import Game
    from contravia.world import World

    engine = mx.Engine("showcase", width=64, height=48)
    game = Game(engine, World(city))
    try:
        out = showcase.run(engine, game, showcase.load(), seconds=0.2, wait=5.0, shots=tmp_path, tag="t")
    finally:
        game._shutdown()
    assert set(out) == {"malecon", "obispo", "loma"}
    for name, r in out.items():
        assert r["frames"] >= 1 and r["p50_ms"] > 0 and "gpu_ms" in r
        assert (tmp_path / f"t-{name}.jpg").exists() and r["shot"].endswith(f"t-{name}.jpg")
        assert r["clock"].startswith(showcase.load()[name]["when"].replace(" ", "T"))  # set again after the wait


def test_compare_writes_one_page_with_every_run(tmp_path):
    import json

    from PIL import Image

    from contravia import showcase

    for k in (1, 2):
        Image.new("RGB", (32, 18), (40 * k, 80, 120)).save(tmp_path / f"r{k}.jpg")
    lines = [{"showcase": True, "preset": p, "knobs": {}, "host": "h", "gpu": "G", "size": [64, 48],
              "gpu_memory_mb": None,
              "scenes": {"malecon": {"p50_ms": ms, "p95_ms": ms + 1, "gpu_ms": {"main": 1.5}, "frames": 9,
                                     "seconds_to_drawn": 0.1, "shot": str(tmp_path / f"r{k}.jpg")}}}
             for k, (p, ms) in enumerate((("bajo", 31.4), ("ultra", 12.7)), 1)]
    history = tmp_path / "history.jsonl"
    history.write_text("\n".join(json.dumps(x) for x in [{"p50_ms": 1}, *lines]) + "\n")
    page = showcase.compare(history, tmp_path / "page.html").read_text()
    assert "bajo" in page and "ultra" in page and "31.4" in page and "12.7" in page
    assert page.count("data:image/jpeg;base64,") == 2 and "http" not in page.split("<body")[1]


def test_bench_takes_the_showcase_flags(monkeypatch):
    import contravia.cli
    import contravia.showcase

    seen = {}
    monkeypatch.setattr(contravia.showcase, "main", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(contravia.showcase, "compare", lambda *a, **k: seen.update(compare=a))
    contravia.cli.main(["bench", "--showcase", "--preset", "bajo", "--knob", "far=7000", "--knob", "tau=24,72",
                        "--size", "800x600"])
    assert seen["preset"] == "bajo" and seen["knobs"] == {"far": 7000.0, "tau": [24.0, 72.0]}
    assert seen["size"] == (800, 600)
    contravia.cli.main(["bench", "--compare"])
    assert "compare" in seen
