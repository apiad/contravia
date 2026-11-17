"""Value noise over the world's x, z: smooth, repeatable numbers in [0, 1] for colours and scattering.

Each point of a square lattice gets a number hashed from its integer coordinates and a seed, and a point
between lattice points blends its four corners with a smoothstep. The same place always gets the same
number, on any machine and in any cell, so neighbouring cells agree along their edges. fbm adds a few
octaves, each twice as fine and half as strong, for patches with ragged edges.
"""

import numpy as np

_M = np.uint64(0xFFFFFFFF)


def _hash(i, j, seed: int) -> np.ndarray:
    """A number in [0, 1) for each pair of integer lattice coordinates."""
    with np.errstate(over="ignore"):
        h = (i.astype(np.int64).astype(np.uint64) * np.uint64(374761393)
             + j.astype(np.int64).astype(np.uint64) * np.uint64(668265263)
             + np.uint64(seed) * np.uint64(2246822519)) & _M
        h = ((h ^ (h >> np.uint64(13))) * np.uint64(1274126177)) & _M
        h = h ^ (h >> np.uint64(16))
    return h.astype(np.float64) / 4294967296.0


def value(x, z, scale: float, seed: int = 0) -> np.ndarray:
    """Smooth noise in [0, 1] at points (x, z), with features about `scale` metres across."""
    gx, gz = np.asarray(x, float) / scale, np.asarray(z, float) / scale
    i, j = np.floor(gx), np.floor(gz)
    fx, fz = gx - i, gz - j
    fx, fz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
    a, b = _hash(i, j, seed), _hash(i + 1, j, seed)
    c, d = _hash(i, j + 1, seed), _hash(i + 1, j + 1, seed)
    return (a + (b - a) * fx) * (1 - fz) + (c + (d - c) * fx) * fz


def fbm(x, z, scale: float, seed: int = 0, octaves: int = 3) -> np.ndarray:
    """Octaves of value noise, each twice as fine and half as strong, normalised to [0, 1]."""
    total, weight, w = 0.0, 0.0, 1.0
    for k in range(octaves):
        total = total + w * value(x, z, scale / 2**k, seed + 101 * k)
        weight += w
        w /= 2
    return total / weight
