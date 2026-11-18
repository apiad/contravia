"""Where things are: OSM latitude and longitude turned into local metres, and metres into cells.

The game works in metres around the Capitolio: x grows to the east, z to the south and y up, the way
manifoldx's camera sees a map from above. One projection covers the whole island: a Lambert conformal
conic on the WGS84 ellipsoid with standard parallels 20.5 and 22.5 degrees north (Snyder, Map
Projections: A Working Manual, equations 15-1 to 15-11). Its scale stays within 0.03 % of true from
Cabo de San Antonio to Maisí, and the game moves manifoldx's origin with the car, so float32 keeps
millimetres anywhere on it.
"""

import numpy as np

ORIGIN = (23.13528, -82.35972)  # the Capitolio, as (lat, lon)
CELL = 200.0  # the side of a streamed cell, in metres
HAVANA_BOX = (22.90, -82.60, 23.25, -82.00)  # (south, west, north, east) degrees around the province
CUBA_BOX = (19.6, -85.1, 23.4, -73.9)  # (south, west, north, east) degrees around the island
# The island's provinces as OSM names them (boundary=administrative, admin_level=4), with the slug of each one's
# file; the game builds them in this order, La Habana first.
PROVINCES = (("La Habana", "la-habana"), ("Artemisa", "artemisa"), ("Mayabeque", "mayabeque"),
             ("Pinar del Río", "pinar-del-rio"), ("Matanzas", "matanzas"), ("Cienfuegos", "cienfuegos"),
             ("Villa Clara", "villa-clara"), ("Sancti Spíritus", "sancti-spiritus"),
             ("Ciego de Ávila", "ciego-de-avila"), ("Camagüey", "camaguey"), ("Las Tunas", "las-tunas"),
             ("Holguín", "holguin"), ("Granma", "granma"), ("Santiago de Cuba", "santiago-de-cuba"),
             ("Guantánamo", "guantanamo"), ("Isla de la Juventud", "isla-de-la-juventud"))

_A = 6_378_137.0  # WGS84 semi-major axis, metres
_E = np.sqrt(2 / 298.257223563 - (1 / 298.257223563) ** 2)  # WGS84 eccentricity
_PARALLELS = np.radians([20.5, 22.5])


def _m(phi):
    return np.cos(phi) / np.sqrt(1 - (_E * np.sin(phi)) ** 2)


def _t(phi):
    es = _E * np.sin(phi)
    return np.tan(np.pi / 4 - phi / 2) / ((1 - es) / (1 + es)) ** (_E / 2)


_N = (np.log(_m(_PARALLELS[0])) - np.log(_m(_PARALLELS[1]))) / (
    np.log(_t(_PARALLELS[0])) - np.log(_t(_PARALLELS[1])))
_AF = _A * _m(_PARALLELS[0]) / (_N * _t(_PARALLELS[0]) ** _N)
_RHO0 = _AF * _t(np.radians(ORIGIN[0])) ** _N


def to_local(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Project latitudes and longitudes to an (N, 2) float64 array of (x, z) metres."""
    rho = _AF * _t(np.radians(np.asarray(lat, float))) ** _N
    theta = _N * np.radians(np.asarray(lon, float) - ORIGIN[1])
    return np.stack([rho * np.sin(theta), -(_RHO0 - rho * np.cos(theta))], axis=-1)


def to_latlon(xz: np.ndarray) -> np.ndarray:
    """The inverse of to_local: an (N, 2) array of (x, z) metres back to (lat, lon) degrees."""
    xz = np.asarray(xz, float)
    x, north = xz[..., 0], -xz[..., 1]
    rho = np.hypot(x, _RHO0 - north)
    t = (rho / _AF) ** (1 / _N)
    lon = np.degrees(np.arctan2(x, _RHO0 - north) / _N) + ORIGIN[1]
    phi = np.pi / 2 - 2 * np.arctan(t)
    for _ in range(8):  # the latitude converges to well under a nanodegree in a few rounds
        es = _E * np.sin(phi)
        phi = np.pi / 2 - 2 * np.arctan(t * ((1 - es) / (1 + es)) ** (_E / 2))
    return np.stack([np.degrees(phi), lon], axis=-1)


def cell_of(xz: np.ndarray) -> np.ndarray:
    """The (i, j) cell holding each point of an (N, 2) array; cell (i, j) starts at (i*CELL, j*CELL)."""
    return np.floor(np.asarray(xz, float) / CELL).astype(np.int32)
