"""The Lambert projection for Cuba matches pyproj and inverts itself."""

import numpy as np

from havana.geo import to_latlon, to_local

# (lat, lon) -> (x, z) metres, from pyproj 3 once:
# +proj=lcc +lat_1=20.5 +lat_2=22.5 +lat_0=23.13528 +lon_0=-82.35972 +ellps=WGS84, z = -northing.
REFERENCE = {
    "capitolio": ((23.13528, -82.35972), (0.0000, 0.0000)),
    "nacional": ((23.1442, -82.3812), (-2200.4077, -988.2589)),
    "santa_clara": ((22.4069, -79.9647), (246586.8784, 78780.4930)),
    "santiago": ((20.0213, -75.8294), (683210.6201, 330512.7954)),
    "san_antonio": ((21.8686, -84.9525), (-267934.6018, 138049.4361)),
    "maisi": ((20.2436, -74.1453), (857965.0424, 297624.8762)),
}


def test_matches_pyproj_within_a_millimetre():
    ll = np.array([v[0] for v in REFERENCE.values()])
    want = np.array([v[1] for v in REFERENCE.values()])
    assert np.abs(to_local(ll[:, 0], ll[:, 1]) - want).max() < 1e-3


def test_round_trip():
    ll = np.array([v[0] for v in REFERENCE.values()])
    back = to_latlon(to_local(ll[:, 0], ll[:, 1]))
    assert np.abs(back - ll).max() < 1e-9
