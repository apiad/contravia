"""Where things are: OSM latitude and longitude turned into local metres, and metres into cells.

The game works in metres around the Capitolio: x grows to the east, z to the south and y up, the way
manifoldx's camera sees a map from above. Within the 20 km the province reaches, a flat projection is
off by far less than a road's width, and float32 keeps millimetres, so no floating origin is needed.
"""

import numpy as np

ORIGIN = (23.13528, -82.35972)  # the Capitolio, as (lat, lon)
CELL = 200.0  # the side of a streamed cell, in metres
HAVANA_BOX = (22.90, -82.60, 23.25, -82.00)  # (south, west, north, east) degrees around the province

_M_PER_DEG_LAT = 110_574.0
_M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(ORIGIN[0]))


def to_local(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Project latitudes and longitudes to an (N, 2) float64 array of (x, z) metres."""
    x = (np.asarray(lon, float) - ORIGIN[1]) * _M_PER_DEG_LON
    z = -(np.asarray(lat, float) - ORIGIN[0]) * _M_PER_DEG_LAT
    return np.stack([x, z], axis=-1)


def cell_of(xz: np.ndarray) -> np.ndarray:
    """The (i, j) cell holding each point of an (N, 2) array; cell (i, j) starts at (i*CELL, j*CELL)."""
    return np.floor(np.asarray(xz, float) / CELL).astype(np.int32)
