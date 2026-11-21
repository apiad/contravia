import os

import pytest

from contravia.citydata import CityData
from contravia.prepare import build_city, read_osm


_KNOBS = (("forest", "LOD_AT"), ("rules.foliage", "NEAR"), ("graveyard", "LOD_AT"), ("grass", "NEAR"), ("grass", "FAR"),
          ("grass", "BLADES"), ("rain", "POOL"), ("atmosphere", "FOG_CLEAR"))


@pytest.fixture(autouse=True)
def _alto():
    """A test that applies a quality preset leaves the module-wide knobs (quality.py) as they were: Alto."""
    import importlib

    mods = {m: importlib.import_module(f"contravia.{m}") for m, _ in _KNOBS}
    saved = [(mods[m], k, getattr(mods[m], k)) for m, k in _KNOBS]
    yield
    for mod, k, v in saved:
        setattr(mod, k, v)


@pytest.fixture(autouse=True)
def _config(tmp_path_factory, monkeypatch):
    """Every test keeps options, designs and the photo collection in a folder of its own, never the user's."""
    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path_factory.mktemp("config")))


@pytest.fixture(autouse=True)
def _regions(monkeypatch):
    """Every test reads the shipped regions.toml without the regions.src of the data on this machine; a test that
    wants other regions sets contravia.regions._current itself."""
    from contravia import regions

    if not hasattr(_regions, "shipped"):
        _regions.shipped = regions.Regions.load()
    monkeypatch.setattr(regions, "_current", _regions.shipped)


@pytest.fixture(scope="session")
def city_npz(tmp_path_factory):
    """The mini.osm fixture prepared into an npz; HAVANA_DATA points at it for cells.build."""
    path = tmp_path_factory.mktemp("data") / "havana.npz"
    city, _ = build_city(read_osm("tests/fixtures/mini.osm"))
    city.save(path)
    os.environ["HAVANA_DATA"] = str(path)
    return path


@pytest.fixture(scope="session")
def city(city_npz):
    return CityData.load(city_npz)


@pytest.fixture(scope="session")
def island_dir(tmp_path_factory, city_npz, city):
    """The two-province island of tests/test_world.py, for the game tests."""
    from tests.test_world import island

    return island.__wrapped__(tmp_path_factory, city_npz, city)[0]
