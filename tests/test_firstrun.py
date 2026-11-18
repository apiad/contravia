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
    """A folder of province sources: La Habana from the fixture, and Artemisa as the same extract again."""
    folder = tmp_path / "cuba-src"
    source.write(read_osm(FIX), Terrain.flat(), folder / "la-habana.src")
    source.write(read_osm(FIX), Terrain.flat(), folder / "artemisa.src")
    return folder


def same(a: CityData, b: CityData) -> list:
    return [k for k in a.a if k not in b.a or not np.array_equal(a.a[k], b.a[k], equal_nan=a.a[k].dtype.kind == "f")]


def test_source_round_trip_keeps_the_extract_and_the_terrain(src):
    raw, terrain = source.load(src / "la-habana.src")
    assert raw.roads == read_osm(FIX).roads
    assert np.array_equal(terrain.heights, Terrain.flat().heights)


def test_build_from_source_gives_the_city_build_city_gives(src, tmp_path):
    b = firstrun.Build(src, tmp_path / "home", parallel=2)  # both provinces at once, each in its own process
    b.run()
    assert b.done and b.overall == 1.0 and b.label == firstrun.DONE_LABEL
    island = firstrun.built(tmp_path / "home")
    assert island and not (tmp_path / "home" / "build").exists() and (island / "index.npz").exists()
    ref, _ = build_city(*source.load(src / "la-habana.src"))
    for slug in ("la-habana", "artemisa"):
        assert same(ref, CityData.load(island / f"{slug}.npz")) == []


def test_a_stopped_build_goes_on_from_its_last_step(src, tmp_path, monkeypatch):
    from contravia import prepare

    def boom(city):
        raise RuntimeError("stopped")

    monkeypatch.setattr(prepare, "add_walks", boom)
    b = firstrun.Build(src, tmp_path / "home", isolate=False)  # the patches live in this process
    with pytest.raises(RuntimeError):
        b.run()
    assert b.error and b.step == 3 and (tmp_path / "home" / "build" / "la-habana" / "streets.npz").exists()
    monkeypatch.undo()
    calls, real = [], prepare.base_city
    monkeypatch.setattr(prepare, "base_city", lambda *a: calls.append(1) or real(*a))
    again = firstrun.Build(src, tmp_path / "home", isolate=False)
    again.run()
    assert again.done and calls == [1]  # La Habana went on from its streets; only Artemisa was read
    monkeypatch.undo()
    ref, _ = build_city(*source.load(src / "la-habana.src"))
    assert same(ref, CityData.load(tmp_path / "home" / firstrun.CITY / "la-habana.npz")) == []


def test_a_province_that_fails_in_its_process_stops_the_build_with_its_error(src, tmp_path):
    (src / "artemisa.src").write_bytes(b"not a source")
    b = firstrun.Build(src, tmp_path / "home", parallel=1)
    b.run(raises=False)
    assert not b.done and b.p == 1 and b.error and not b.error.startswith("ProvinceError")
    assert (tmp_path / "home" / firstrun.CITY / "la-habana.npz").exists()  # the province before it is kept


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


def test_workers_fit_the_cores_and_the_memory(monkeypatch):
    monkeypatch.setattr(firstrun.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(firstrun, "free_gb", lambda: 9.0)
    assert firstrun.workers() == 2
    monkeypatch.setattr(firstrun, "free_gb", lambda: 1.0)
    assert firstrun.workers() == 1
    monkeypatch.undo()
    assert 0 < firstrun.free_gb() < 10_000


def test_home_follows_contravia_home(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRAVIA_HOME", str(tmp_path))
    assert firstrun.home() == tmp_path and firstrun.built() is None


def test_the_shipped_source_is_the_version_the_game_reads():
    raw, _ = source.load("data/havana.src")
    assert raw.roads
