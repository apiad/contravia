import numpy as np
import pytest

from havana.bench import autopilot, road_path, summary
from havana.car import CarState, step


def test_autopilot_follows_a_square():
    path = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    s, i = CarState(0, 0, 0, 0), 0
    for _ in range(60 * 40):
        c, i = autopilot(s, path, i)
        s = step(s, c, 1 / 60, np.zeros((0, 2, 2), np.float32))
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

    from havana.bench import SCENES, flight_path
    from havana.geo import to_local

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
    import havana.cli
    import havana.game

    seen = {}
    monkeypatch.setattr(havana.game, "main", lambda *a, **k: seen.update(k))
    havana.cli.main(["play"])
    assert seen["pipelined"] is True
    havana.cli.main(["play", "--no-pipeline"])
    assert seen["pipelined"] is False
