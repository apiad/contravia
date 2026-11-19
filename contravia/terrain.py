"""The ground's height anywhere in the province, from Copernicus GLO-30 cleaned of roofs and tree tops.

GLO-30 measures the surface: in a dense city its "ground" is the roofs. `clean` takes, in every window of
5 x 5 samples (about 150 m), the lowest height, then the highest of those lowest heights, which drops
anything narrower than the window and keeps hills; a light blur then smooths the steps. `prepare` does this
once and stores the grid in the city data. Everything that places geometry asks `Terrain.height`, so the
ground, the roads, the buildings and the car all stand on the same surface.
"""

import math
import os
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from contravia.geo import HAVANA_BOX, ORIGIN, to_local

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
    def flat(cls) -> Terrain:
        """Height 0 everywhere: the terrain of data prepared without heights."""
        south, west, north, east = HAVANA_BOX
        nw, se = to_local(np.array([north, south]), np.array([west, east]))
        return cls(np.zeros((2, 2), np.float32), nw, se - nw)

    def height(self, x, z, near=None) -> np.ndarray:
        """The height at (x, z), interpolated between the four nearest samples; edge heights outside. near is
        ignored: the ground has one height per point (see contravia.surface)."""
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


GLO30_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
TILE_SAMPLES = 3600  # a GLO-30 tile's rows and columns per degree below 50 degrees of latitude


def fetch_tile(folder, lat: int, lon: int) -> Path | None:
    """The GLO-30 tile for (lat, lon) in folder, downloaded first if missing; None where there is only sea (the
    bucket has no tile there, remembered with an empty .none file so it is not asked again)."""
    import urllib.error
    import urllib.request

    name = _tile_name(lat, lon)
    path, none = Path(folder) / name, Path(folder) / (name + ".none")
    if path.exists():
        return path
    if none.exists():
        return None
    Path(folder).mkdir(parents=True, exist_ok=True)
    import http.client
    import time

    for attempt in range(5):  # the bucket is slow from some networks: a read that times out is tried again
        try:
            with urllib.request.urlopen(GLO30_URL.format(name=name.removesuffix(".tif")), timeout=300) as r:
                part = path.with_suffix(f".{os.getpid()}.part")  # parallel provinces may fetch the same tile
                part.write_bytes(r.read())
                part.rename(path)
            return path
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):
                raise
            none.touch()
            return None
        except (TimeoutError, urllib.error.URLError, ConnectionError, http.client.IncompleteRead):
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
    return path


def read_terrain(folder, box, step: float = 30.0, stride: int = 1, fetch: bool = True) -> Terrain:
    """The cleaned ground over box (south, west, north, east) on a regular grid of local metres.

    Each sample is placed through the island's projection and read from the GLO-30 tile under it, so the grid
    stays true far from the Capitolio, where meridians turn by degrees against the local axes. The heights are
    cleaned (clean) at `step` metres, then every `stride`-th sample is kept: a coarser grid for big provinces.
    Missing tiles are downloaded into folder when fetch is True; sea without a tile reads 0.
    """
    import tifffile

    from contravia.geo import to_latlon

    south, west, north, east = box
    t = np.linspace(0.0, 1.0, 33)
    edge_lat = np.concatenate([np.full_like(t, south), np.full_like(t, north), south + t * (north - south),
                               south + t * (north - south)])
    edge_lon = np.concatenate([west + t * (east - west), west + t * (east - west), np.full_like(t, west),
                               np.full_like(t, east)])
    xz = to_local(edge_lat, edge_lon)
    x0, z0 = math.floor(xz[:, 0].min() / step) * step, math.floor(xz[:, 1].min() / step) * step
    cols = math.ceil((xz[:, 0].max() - x0) / step) + 1
    rows = math.ceil((xz[:, 1].max() - z0) / step) + 1
    tiles: dict = {}

    def tile(lat: int, lon: int):
        if (lat, lon) not in tiles:
            path = fetch_tile(folder, lat, lon) if fetch else Path(folder) / _tile_name(lat, lon)
            tiles[(lat, lon)] = tifffile.imread(path).astype(np.float32) if path and path.exists() else None
        return tiles[(lat, lon)]

    out = np.zeros((rows, cols), np.float32)
    xs = x0 + np.arange(cols) * step
    for r in range(0, rows, 256):  # a band of rows at a time: the lat/lon of every sample at once is gigabytes
        zs = z0 + np.arange(r, min(rows, r + 256)) * step
        gx, gz = np.meshgrid(xs, zs)
        ll = to_latlon(np.stack([gx.ravel(), gz.ravel()], axis=1))
        lat, lon = ll[:, 0], ll[:, 1]
        band = np.zeros(len(lat), np.float32)
        tl, tn = np.floor(lat).astype(int), np.floor(lon).astype(int)
        for la, lo in set(zip(tl.tolist(), tn.tolist(), strict=True)):
            grid = tile(la, lo)
            if grid is None:
                continue
            k = (tl == la) & (tn == lo)
            n = grid.shape[0]
            fr = np.clip((la + 1 - lat[k]) * n - 0.5, 0, n - 1.001)
            fc = np.clip((lon[k] - lo) * grid.shape[1] - 0.5, 0, grid.shape[1] - 1.001)
            r0, c0 = fr.astype(int), fc.astype(int)
            ar, ac = fr - r0, fc - c0
            band[k] = ((grid[r0, c0] * (1 - ac) + grid[r0, c0 + 1] * ac) * (1 - ar)
                       + (grid[r0 + 1, c0] * (1 - ac) + grid[r0 + 1, c0 + 1] * ac) * ar)
        out[r: r + len(zs)] = band.reshape(len(zs), cols)
    ground = clean(out)[::stride, ::stride]
    return Terrain(ground, (x0, z0), (step * stride, step * stride))
