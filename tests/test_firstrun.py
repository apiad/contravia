import threading

import numpy as np
import pytest

from contravia import firstrun, source
from contravia.citydata import CityData
from contravia.prepare import build_city, read_osm
from contravia.terrain import Terrain

FIX = "tests/fixtures/mini.osm"


@pytest.fixture
def src(tmp_path):
    path = tmp_path / "havana.src"
    source.write(read_osm(FIX), Terrain.flat(), path)
    return path


def same(a: CityData, b: CityData) -> list:
    return [k for k in a.a if k not in b.a or not np.array_equal(a.a[k], b.a[k], equal_nan=a.a[k].dtype.kind == "f")]


def test_source_round_trip_keeps_the_extract_and_the_terrain(src):
    raw, terrain = source.load(src)
    assert raw.roads == read_osm(FIX).roads
    assert np.array_equal(terrain.heights, Terrain.flat().heights)


def test_build_from_source_gives_the_city_build_city_gives(src, tmp_path):
    b = firstrun.Build(src, tmp_path / "home")
    b.run()
    assert b.done and b.overall == 1.0 and b.label == firstrun.DONE_LABEL
    assert firstrun.built(tmp_path / "home") and not (tmp_path / "home" / "build").exists()
    ref, _ = build_city(*source.load(src))
    assert same(ref, CityData.load(tmp_path / "home" / firstrun.CITY)) == []


def test_a_stopped_build_goes_on_from_its_last_step(src, tmp_path, monkeypatch):
    from contravia import prepare

    def boom(city):
        raise RuntimeError("stopped")

    monkeypatch.setattr(prepare, "add_walks", boom)
    b = firstrun.Build(src, tmp_path / "home")
    with pytest.raises(RuntimeError):
        b.run()
    assert b.error and b.step == 3 and (tmp_path / "home" / "build" / "streets.npz").exists()
    monkeypatch.undo()
    calls = []
    monkeypatch.setattr(prepare, "base_city", lambda *a: calls.append(1))
    again = firstrun.Build(src, tmp_path / "home")
    again.run()
    assert again.done and calls == []
    monkeypatch.undo()
    ref, _ = build_city(*source.load(src))
    assert same(ref, CityData.load(tmp_path / "home" / firstrun.CITY)) == []


def test_progress_is_readable_while_it_runs(src, tmp_path):
    b = firstrun.Build(src, tmp_path / "home")
    t = threading.Thread(target=b.run)
    t.start()
    seen = []
    while t.is_alive():
        seen.append((b.step, b.label, b.overall, b.eta, b.mb))
        t.join(0.01)
    assert all(0.0 <= o <= 1.0 and e >= 0 for _, _, o, e, _ in seen)
    assert b.mb > 0 and b.eta == 0.0


def test_home_follows_contravia_home(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRAVIA_HOME", str(tmp_path))
    assert firstrun.home() == tmp_path and firstrun.built() is None


def test_the_shipped_source_is_the_version_the_game_reads():
    raw, _ = source.load("data/havana.src")
    assert raw.roads
