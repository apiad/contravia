"""What lies under the map's crosshair, read from the province's road graph (contravia/whereis.py)."""

import time

import numpy as np

from contravia import collect, whereis
from contravia.world import World


def _world(city_npz):
    """The fixture city, whose edges all lie in Habana Vieja (mini.osm has no municipio boundary)."""
    world = World.load(str(city_npz))
    a = world.province(0).a
    a["g_muni"] = np.zeros(len(a["g_from"]), np.int16)
    a["g_muni_names"] = np.array(["Habana Vieja"])
    return world


def _street_point(world):
    """The middle of a named edge of the fixture city's graph."""
    a = world.graph.a
    e = int(np.flatnonzero(a["g_name"] >= 0)[0])
    return float(a["g_mid"][e, 0]), float(a["g_mid"][e, 1]), str(a["g_names"][a["g_name"][e]])


def test_each_detail_adds_the_municipio_then_the_street(city_npz):
    world = _world(city_npz)
    x, z, name = _street_point(world)
    where = whereis.Where(world)
    assert where.at(x, z, 0) == whereis.nearby.Spot(province="La Habana")
    assert where.at(x, z, 1).municipio == "Habana Vieja" and where.at(x, z, 1).street == ""
    spot = where.at(x, z, 2)
    assert spot.municipio == "Habana Vieja" and spot.street and name.startswith(spot.street)


def test_far_from_any_street_it_names_no_municipio(city_npz, monkeypatch):
    monkeypatch.setattr(whereis, "MUNI_REACH", 1_000.0)  # the fixture city is a few blocks
    world = _world(city_npz)
    x, z, _ = _street_point(world)
    spot = whereis.Where(world).at(x + 1_700, z, 2)  # still in the province's cells, which reach 2 km
    assert spot.province == "La Habana" and spot.municipio == "" and spot.street == ""


def test_a_nameless_road_under_the_crosshair_gives_the_nearest_named_street(city_npz):
    world = _world(city_npz)
    a = world.province(0).a
    x, z, name = _street_point(world)
    e = int(np.argmin(np.hypot(a["g_mid"][:, 0] - x, a["g_mid"][:, 1] - z)))
    keep = int(a["g_name"][e])
    a["g_name"][a["g_name"] == keep] = -1  # the street under the crosshair loses its name
    spot = whereis.Where(world).at(x, z, 2)
    assert spot.street and spot.street != name


def test_the_arrays_are_read_from_the_provinces_file(city_npz, tmp_path, monkeypatch):
    world = _world(city_npz)
    world.province(0).save(tmp_path / "with-muni.npz")
    monkeypatch.setattr(world, "paths", [tmp_path / "with-muni.npz"])
    x, z, _ = _street_point(world)
    assert whereis.Where(world).at(x, z, 2).municipio == "Habana Vieja"


def test_the_worker_answers_the_newest_question_once(city_npz):
    world = _world(city_npz)
    x, z, _ = _street_point(world)
    where = whereis.Where(world)
    places = collect.Places([])
    where.ask(places, x, z, 500_000.0)
    where.ask(places, x, z, 400.0)
    end = time.monotonic() + 10
    got = None
    while time.monotonic() < end:
        got = where.take() or got
        if got is not None and got.band == "calle":
            break
        time.sleep(0.01)
    assert got is not None and got.band == "calle" and got.title
    assert where.take() is None
