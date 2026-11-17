import os

import pytest

from caiman.citydata import CityData
from caiman.prepare import build_city, read_osm


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
def island_dir(tmp_path_factory, city):
    """The island's tiles of tests/test_world.py, for the game tests."""
    from tests.test_world import island

    return island.__wrapped__(tmp_path_factory, city)
