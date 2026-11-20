"""Where the satellite sees buildings: ESA WorldCover 2021's built-up class on a STEP-metre grid over a province.

OSM maps houses unevenly: whole towns (Guanabacoa, Cotorro) have streets and almost no buildings, and the blocks
OSM left empty would become parks (contravia.parks). WorldCover classes every 10 m square of land from Sentinel
imagery, the same way across the island, so a block it sees built over keeps the filler houses OSM's emptiness
would otherwise lose. `read` samples the tiles' STEP-metre overview over a province's box, through the island's
projection, into a grid of local metres; `prepare` and `source` keep it in the province's Raw, so the first run
builds the same island. `share` gives the part of each polygon built over.

The tiles (a few tens of MB each, contravia.cover.URL) are downloaded into a cache folder when missing and kept
there, like the relief's. Its licence (CC BY 4.0) asks for contravia.cover.CREDIT wherever the game names its data.
"""

import math
from pathlib import Path

import numpy as np

from contravia.cover import URL, _download, tile_name
from contravia.geo import to_latlon, to_local

STEP = 20.0  # metres a side of a grid square: against WorldCover's 10 m it keeps 96 % of the parks' verdicts
BUILT = 50  # WorldCover's code for built-up


class BuiltUp:
    """A grid of built-up squares: [r, c] covers x from origin[0] + c * STEP, z from origin[1] + r * STEP."""

    step = STEP  # metres a side of a square; a grid pickled before it had its own step reads this one

    def __init__(self, bits: np.ndarray, shape, origin, step: float = STEP):
        self.bits = np.asarray(bits, np.uint8)  # np.packbits of the (rows, cols) mask, row after row
        self.shape = (int(shape[0]), int(shape[1]))
        self.origin = (float(origin[0]), float(origin[1]))
        self.step = float(step)

    @classmethod
    def of(cls, mask: np.ndarray, origin, step: float = STEP) -> "BuiltUp":
        return cls(np.packbits(mask, axis=None), mask.shape, origin, step)

    @classmethod
    def of_cover(cls, cover) -> "BuiltUp":
        """The built-up squares of a contravia.cover grid (WorldCover at its 100 m), for a province built without
        its own 20 m grid at hand."""
        from contravia.cover import STEP as COVER_STEP

        return cls.of(cover.codes == BUILT, cover.origin, COVER_STEP)

    def mask(self) -> np.ndarray:
        rows, cols = self.shape
        return np.unpackbits(self.bits, count=rows * cols).reshape(rows, cols).astype(bool)

    def share(self, polys) -> np.ndarray:
        """The part of each shapely polygon whose grid squares are built-up (by their centres); 0 where none
        of its centres falls inside it or the grid."""
        import shapely

        m = self.mask()
        rows, cols = self.shape
        out = np.zeros(len(polys))
        for k, poly in enumerate(polys):
            x0, z0, x1, z1 = poly.bounds
            step = self.step
            c0, c1 = max(0, int((x0 - self.origin[0]) // step)), min(cols, int((x1 - self.origin[0]) // step) + 1)
            r0, r1 = max(0, int((z0 - self.origin[1]) // step)), min(rows, int((z1 - self.origin[1]) // step) + 1)
            if c0 >= c1 or r0 >= r1:
                continue
            cx = self.origin[0] + (np.arange(c0, c1) + 0.5) * step
            cz = self.origin[1] + (np.arange(r0, r1) + 0.5) * step
            gx, gz = np.meshgrid(cx, cz)
            inside = shapely.contains_xy(poly, gx, gz)
            if inside.any():
                out[k] = m[r0:r1, c0:c1][inside].mean()
        return out


def read(cache, box) -> BuiltUp:
    """The built-up grid over box (south, west, north, east), from the WorldCover tiles in cache, downloaded
    there first if missing. Open sea, which has no tile, reads as not built."""
    import tifffile

    south, west, north, east = box
    t = np.linspace(0.0, 1.0, 33)  # the box's edges, projected: the local grid must hold all of it
    lat = np.concatenate([np.full_like(t, south), np.full_like(t, north), south + t * (north - south),
                          south + t * (north - south)])
    lon = np.concatenate([west + t * (east - west), west + t * (east - west), np.full_like(t, west),
                          np.full_like(t, east)])
    xz = to_local(lat, lon)
    x0, z0 = math.floor(xz[:, 0].min() / STEP) * STEP, math.floor(xz[:, 1].min() / STEP) * STEP
    cols = math.ceil((xz[:, 0].max() - x0) / STEP)
    rows = math.ceil((xz[:, 1].max() - z0) / STEP)
    mask = np.zeros((rows, cols), bool)
    xs = x0 + (np.arange(cols) + 0.5) * STEP
    for s in range(3 * math.floor(south / 3), 3 * math.floor(north / 3) + 1, 3):
        for w in range(3 * math.floor(west / 3), 3 * math.floor(east / 3) + 1, 3):
            name = tile_name(s, w)
            path = Path(cache) / f"{name}.tif"
            if not path.exists() and not _download(URL.format(name=name), path):
                continue  # open sea
            with tifffile.TiffFile(path) as tif:
                levels = tif.series[0].levels
                k = max(i for i in range(len(levels)) if levels[0].shape[0] / levels[i].shape[0] * 10.0 <= STEP)
                img = levels[k].asarray()
            n = img.shape[0]
            for r in range(rows):
                ll = to_latlon(np.c_[xs, np.full(cols, z0 + (r + 0.5) * STEP)])
                pr = np.floor((s + 3 - ll[:, 0]) / 3 * n).astype(np.int64)
                pc = np.floor((ll[:, 1] - w) / 3 * n).astype(np.int64)
                ok = (pr >= 0) & (pr < n) & (pc >= 0) & (pc < n)
                mask[r, ok] |= img[pr[ok], pc[ok]] == BUILT
    return BuiltUp.of(mask, (x0, z0))
