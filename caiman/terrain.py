"""The ground's height anywhere in the province, from Copernicus GLO-30 cleaned of roofs and tree tops.

GLO-30 measures the surface: in a dense city its "ground" is the roofs. `clean` takes, in every window of
5 x 5 samples (about 150 m), the lowest height, then the highest of those lowest heights, which drops
anything narrower than the window and keeps hills; a light blur then smooths the steps. `prepare` does this
once and stores the grid in the city data. Everything that places geometry asks `Terrain.height`, so the
ground, the roads, the buildings and the car all stand on the same surface.
"""

import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from caiman.geo import HAVANA_BOX, ORIGIN, to_local

# Metres per degree of the projection at the Capitolio. The height grid stays a regular grid in (x, z):
# across the province the projected degree grid bends by about one GLO-30 sample, which every piece of
# geometry shares because all of it stands on Terrain.height.
_STEP = to_local(np.array([ORIGIN[0], ORIGIN[0], ORIGIN[0] + 0.01]),
                 np.array([ORIGIN[1], ORIGIN[1] + 0.01, ORIGIN[1]]))
_M_PER_DEG_LON = float(_STEP[1, 0] - _STEP[0, 0]) * 100
_M_PER_DEG_LAT = float(_STEP[0, 1] - _STEP[2, 1]) * 100


FADE = 1000.0  # metres over which the relief falls to flat at the grid's edge, so leaving it shows no cliff


class Terrain:
    """A height grid: sample [r, c] stands at x = origin[0] + c * step[0], z = origin[1] + r * step[1].

    Heights fade with a smoothstep to 0 over the last `fade` metres inside the grid and are 0 outside it,
    where the rest of the island is flat.
    """

    def __init__(self, heights: np.ndarray, origin, step, fade: float = FADE):
        self.heights = np.asarray(heights, np.float32)
        self.origin = (float(origin[0]), float(origin[1]))
        self.step = (float(step[0]), float(step[1]))
        self.fade = float(fade)

    @classmethod
    def flat(cls) -> "Terrain":
        """Height 0 everywhere: the terrain of data prepared without heights."""
        south, west, north, east = HAVANA_BOX
        nw, se = to_local(np.array([north, south]), np.array([west, east]))
        return cls(np.zeros((2, 2), np.float32), nw, se - nw)

    def height(self, x, z) -> np.ndarray:
        """The height at (x, z), interpolated between the four nearest samples; edge heights outside."""
        h = self.heights
        rows, cols = h.shape
        fx = np.clip((np.asarray(x, float) - self.origin[0]) / self.step[0], 0, cols - 1)
        fz = np.clip((np.asarray(z, float) - self.origin[1]) / self.step[1], 0, rows - 1)
        c0 = np.minimum(np.floor(fx).astype(int), cols - 2)
        r0 = np.minimum(np.floor(fz).astype(int), rows - 2)
        tx, tz = fx - c0, fz - r0
        top = h[r0, c0] * (1 - tx) + h[r0, c0 + 1] * tx
        bottom = h[r0 + 1, c0] * (1 - tx) + h[r0 + 1, c0 + 1] * tx
        out = top * (1 - tz) + bottom * tz
        if self.fade <= 0:
            return out
        x0, z0 = self.origin
        x1, z1 = x0 + (cols - 1) * self.step[0], z0 + (rows - 1) * self.step[1]
        x, z = np.asarray(x, float), np.asarray(z, float)
        edge = np.minimum(np.minimum(x - min(x0, x1), max(x0, x1) - x), np.minimum(z - min(z0, z1), max(z0, z1) - z))
        s = np.clip(edge / self.fade, 0.0, 1.0)
        return out * (s * s * (3 - 2 * s))


def seat(ring, terrain: Terrain) -> float:
    """The lowest ground under a footprint, sampled at its corners and its centroid."""
    ring = np.asarray(ring, float).reshape(-1, 2)
    pts = np.vstack([ring, ring.mean(axis=0)])
    return float(terrain.height(pts[:, 0], pts[:, 1]).min())


def _filter(grid: np.ndarray, size: int, reduce) -> np.ndarray:
    """Apply reduce (np.min or np.max) over a size x size window, one axis at a time, edges repeated."""
    k = size // 2
    out = grid
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (k, k)
        out = reduce(sliding_window_view(np.pad(out, pad, mode="edge"), size, axis=axis), axis=-1)
    return out


def _blur(grid: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur of sigma samples, one axis at a time, edges repeated."""
    r = math.ceil(3 * sigma)
    t = np.arange(-r, r + 1)
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    out = grid
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        out = sliding_window_view(np.pad(out, pad, mode="edge"), 2 * r + 1, axis=axis) @ kernel
    return out


def clean(grid: np.ndarray, size: int = 5, sigma: float = 1.5) -> np.ndarray:
    """A surface model turned into ground: grey opening, then a blur, then nothing below 0."""
    opened = _filter(_filter(np.asarray(grid, np.float32), size, np.min), size, np.max)
    return np.maximum(_blur(opened, sigma), 0.0).astype(np.float32)


def _tile_name(lat: int, lon: int) -> str:
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM.tif"


def read_tiles(folder, box=HAVANA_BOX) -> Terrain:
    """The raw GLO-30 heights over box (south, west, north, east), from the one-degree tiles in folder.

    A tile named for (lat, lon) covers lat..lat+1 and lon..lon+1 degrees, row 0 at its north edge and
    column 0 at its west edge. Rows and columns per degree are read from the tile's own shape.
    """
    import tifffile

    south, west, north, east = box
    lats = range(math.ceil(north) - 1, math.floor(south) - 1, -1)  # north to south; an edge on a whole
    lons = range(math.floor(west), math.ceil(east))                 # degree needs no tile beyond it
    rows = []
    for lat in lats:
        row = []
        for lon in lons:
            path = Path(folder) / _tile_name(lat, lon)
            if not path.exists():
                raise FileNotFoundError(f"missing GLO-30 tile {path.name} in {folder}")
            row.append(tifffile.imread(path).astype(np.float32))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    per_lat, per_lon = grid.shape[0] // len(lats), grid.shape[1] // len(lons)
    top, left = lats[0] + 1, lons[0]
    r0, r1 = math.floor((top - north) * per_lat), math.ceil((top - south) * per_lat) + 1
    c0, c1 = math.floor((west - left) * per_lon), math.ceil((east - left) * per_lon) + 1
    origin = to_local(np.array([top - r0 / per_lat]), np.array([left + c0 / per_lon]))[0]
    return Terrain(grid[r0:r1, c0:c1], origin, (_M_PER_DEG_LON / per_lon, _M_PER_DEG_LAT / per_lat))
