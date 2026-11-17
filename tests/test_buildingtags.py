import numpy as np

from caiman.buildingtags import HEIGHT, KIND, KINDS, LEVELS, MUNI, USES, encode, kind, number
from caiman.citydata import CityData
from caiman.geo import to_latlon


def test_numbers_and_kinds_from_tag_values():
    assert number("4") == 4 and number("12.5 m") == 12.5 and number("3;4") == 3
    assert np.isnan(number(None)) and np.isnan(number("tall"))
    assert KINDS[kind("cathedral")] == "church" and KINDS[kind(None)] == "yes" and KINDS[kind("igloo")] == "other"


def test_prepare_keeps_the_tags_of_each_building(city):
    a, ids = city.a["building_attrs"], city.a["building_ids"]
    assert a.shape == (len(ids), 5)
    tagged = a[a[:, KIND] == KINDS.index("apartments")]
    assert len(tagged) == 1 and tagged[0, LEVELS] == 5 and tagged[0, USES] == 4 and np.isnan(tagged[0, HEIGHT])


def test_municipio_of_each_building():
    ring = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], np.float32)
    city = CityData.from_features(np.zeros((0, 5)), [ring, ring + 1000], np.array([1, 2], np.int64), [],
                                  np.zeros(0, np.int64), np.zeros((0, 2)), None)
    box = to_latlon(np.array([[-50.0, -50.0], [100.0, -50.0], [100.0, 100.0], [-50.0, 100.0]]))
    out = encode(city, {1: {"building": "house", "height": "7 m"}}, [("Centro Habana", [tuple(p) for p in box])])
    a = out["building_attrs"][np.argsort(city.a["building_ids"])]
    assert list(out["building_muni_names"]) == ["Centro Habana"]
    assert a[0, MUNI] == 0 and a[0, HEIGHT] == 7 and KINDS[int(a[0, KIND])] == "house"
    assert a[1, MUNI] == -1 and KINDS[int(a[1, KIND])] == "yes"
