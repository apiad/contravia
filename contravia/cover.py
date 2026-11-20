"""What covers the land, from ESA WorldCover 2021: one class per 100 m square of the island, for the map's colours.

WorldCover (10 m, CC BY 4.0) classes every square of land as trees, shrubs, grass, crops, built-up, bare, water,
wetland or mangrove, the same way across the whole island, where OSM's landuse is mapped in patches
(docs/superpowers/specs/2026-11-19-map-sea-coast-design.html). `build` downloads the 3 x 3 degree tiles over the
coast (a few tens of MB each), reads each at its overview nearest STEP and puts one class per STEP square of the
local grid, then deletes the tile. `save` and `load` keep the grid in cover.src, beside the coast. `shares` gives a
summary the share of each class in each of its squares; the index keeps them, and `below` hands them down to the
nodes made inside a leaf while playing. The colours are rules (contravia/rules/cover.py).
"""

import math
import pickle
from compression import zstd
from pathlib import Path

import numpy as np

from contravia.geo import CELL, to_latlon, to_local

VERSION = 1
NAME = "cover.src"
STEP = 100.0  # metres a side of a cover square
CLASSES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)  # WorldCover's codes; 0 is no data
SAMPLES = 4  # points a side sampled in each of a summary's squares
URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{name}_Map.tif"
CREDIT = ("© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by "
          "ESA WorldCover consortium")


def tile_name(lat: float, lon: float) -> str:
    """The WorldCover tile holding (lat, lon), named by its south-west corner: N21W084."""
    s, w = 3 * math.floor(lat / 3), 3 * math.floor(lon / 3)
    return f"{'N' if s >= 0 else 'S'}{abs(s):02d}{'E' if w >= 0 else 'W'}{abs(w):03d}"


class Cover:
    """A grid of WorldCover codes: square [r, c] covers x from origin[0] + c * STEP, z from origin[1] + r * STEP."""

    def __init__(self, codes: np.ndarray, origin):
        self.codes = np.asarray(codes, np.uint8)
        self.origin = (float(origin[0]), float(origin[1]))

    def sample(self, x, z) -> np.ndarray:
        """The code at points (x, z); 0 outside the grid."""
        c = np.floor((np.asarray(x, float) - self.origin[0]) / STEP).astype(np.int64)
        r = np.floor((np.asarray(z, float) - self.origin[1]) / STEP).astype(np.int64)
        rows, cols = self.codes.shape
        inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        out = np.zeros(np.shape(c), np.uint8)
        out[inside] = self.codes[r[inside], c[inside]]
        return out


def build(land, cache, fetch=None) -> tuple[Cover, dict]:
    """(cover, report) over the box of land (a shapely geometry in local metres), from the tiles in cache,
    downloaded there first if missing and deleted once read. fetch(url, path) downloads; urllib by default."""
    import tifffile

    x0, z0, x1, z1 = land.bounds
    x0, z0 = math.floor(x0 / STEP) * STEP, math.floor(z0 / STEP) * STEP
    cols, rows = math.ceil((x1 - x0) / STEP), math.ceil((z1 - z0) / STEP)
    codes = np.zeros((rows, cols), np.uint8)
    corners = to_latlon(np.array([[x0, z0], [x0 + cols * STEP, z0], [x0, z0 + rows * STEP],
                                  [x0 + cols * STEP, z0 + rows * STEP]]))
    lat0, lat1 = math.floor(corners[:, 0].min() / 3) * 3, math.floor(corners[:, 0].max() / 3) * 3
    lon0, lon1 = math.floor(corners[:, 1].min() / 3) * 3, math.floor(corners[:, 1].max() / 3) * 3
    report = {"tiles": 0, "missing": 0}
    for lat in range(lat0, lat1 + 1, 3):
        for lon in range(lon0, lon1 + 1, 3):
            name = tile_name(lat, lon)
            path = Path(cache) / f"{name}.tif"
            if not path.exists() and not _download(URL.format(name=name), path, fetch):
                report["missing"] += 1  # open sea: WorldCover has no tile there
                continue
            with tifffile.TiffFile(path) as tif:
                levels = tif.series[0].levels
                k = max(i for i in range(len(levels)) if levels[0].shape[0] / levels[i].shape[0] * 10.0 <= STEP)
                img = levels[k].asarray()
            path.unlink()
            _paint(codes, (x0, z0), img, lat, lon)
            report["tiles"] += 1
    report["land_squares"] = int((codes > 0).sum())
    return Cover(codes, (x0, z0)), report


def _paint(codes, origin, img, lat, lon) -> None:
    """Put the tile img (3 x 3 degrees from (lat, lon) at its south-west corner, north up) into codes, row by
    row of the local grid over the tile's box."""
    box = to_local(np.array([lat, lat, lat + 3, lat + 3], float), np.array([lon, lon + 3, lon, lon + 3], float))
    rows, cols = codes.shape
    c0 = max(0, int((box[:, 0].min() - origin[0]) // STEP))
    c1 = min(cols, int((box[:, 0].max() - origin[0]) // STEP) + 1)
    r0 = max(0, int((box[:, 1].min() - origin[1]) // STEP))
    r1 = min(rows, int((box[:, 1].max() - origin[1]) // STEP) + 1)
    if c0 >= c1 or r0 >= r1:
        return
    n = img.shape[0]
    xs = origin[0] + (np.arange(c0, c1) + 0.5) * STEP
    for r in range(r0, r1):
        z = origin[1] + (r + 0.5) * STEP
        ll = to_latlon(np.c_[xs, np.full(len(xs), z)])
        pr = np.floor((lat + 3 - ll[:, 0]) / 3 * n).astype(np.int64)
        pc = np.floor((ll[:, 1] - lon) / 3 * n).astype(np.int64)
        ok = (pr >= 0) & (pr < n) & (pc >= 0) & (pc < n)
        codes[r, c0:c1][ok] = img[pr[ok], pc[ok]]


def _download(url: str, path: Path, fetch=None) -> bool:
    """Download url to path; False when the server has no such file."""
    import urllib.error
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")  # a cut download never passes for a tile
    try:
        if fetch is not None:
            fetch(url, part)
        else:
            urllib.request.urlretrieve(url, part)
    except urllib.error.HTTPError as e:
        part.unlink(missing_ok=True)
        if e.code in (403, 404):
            return False
        raise
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(path)
    return True


def save(cover: Cover, path) -> int:
    """Write cover to path; returns its size in bytes."""
    body = {"version": VERSION, "codes": cover.codes, "origin": cover.origin}
    data = zstd.compress(pickle.dumps(body, protocol=pickle.HIGHEST_PROTOCOL), level=19)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return len(data)


def load(path) -> Cover:
    """The cover saved at path."""
    body = pickle.loads(zstd.decompress(Path(path).read_bytes()))
    if body.get("version") != VERSION:
        raise ValueError(f"{path}: cover version {body.get('version')}, this game reads {VERSION}")
    return Cover(body["codes"], body["origin"])


def shares(cover: Cover, L: int, a: int, b: int, origin, grid: int = 16) -> np.ndarray:
    """(grid, grid, len(CLASSES)) float16: the share of each class among SAMPLES x SAMPLES points in each square
    of node (L, a, b); a square with no data has all shares 0."""
    side = (1 << L) * CELL
    x0, z0 = (origin[0] + (a << L)) * CELL, (origin[1] + (b << L)) * CELL
    t = (np.arange(grid * SAMPLES) + 0.5) * side / (grid * SAMPLES)
    x, z = np.meshgrid(x0 + t, z0 + t)
    codes = cover.sample(x, z)
    known = np.isin(codes, CLASSES)
    k = np.searchsorted(CLASSES, codes)  # a known code's index in CLASSES
    onehot = np.zeros((*x.shape, len(CLASSES)), np.float32)
    rr, cc = np.nonzero(known)
    onehot[rr, cc, k[known]] = 1.0
    return onehot.reshape(grid, SAMPLES, grid, SAMPLES, -1).mean(axis=(1, 3)).astype(np.float16)


def below(up: np.ndarray, up_key, key, grid: int = 16) -> np.ndarray:
    """The shares of node key, inside node up_key, read from up's squares (each child square takes the share
    of the square of up it lies in)."""
    (U, ua, ub), (L, a, b) = up_key, key
    f = 1 << (U - L)  # how many of key's sides fit in up's
    rows = ((b - ub * f) * grid + np.arange(grid)) // f
    cols = ((a - ua * f) * grid + np.arange(grid)) // f
    return np.asarray(up)[np.ix_(rows, cols)]
