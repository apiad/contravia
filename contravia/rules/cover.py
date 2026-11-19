"""The map's colours for what covers the land: a flat atlas palette, a few tones, not a photo.

Each WorldCover class (contravia.cover.CLASSES) has one tone; a summary's square mixes them by their shares in it.
Trees take the far green of the game's date (rules.trees.far_green), so the woods turn with the seasons as the
trees up close do. `tint` mixes in an atlas's height tints from the city stage up. `shade` darkens the ground on
slopes turned away from a fixed light in the north-west, measured on the slope as exaggerated at that level, more
at the levels drawn from high up, so the mountains read from the whole island even under a noon sun.
"""

import numpy as np

from contravia.rules.ground import GROUND
from contravia.rules.trees import far_green

TONES = {
    20: (0.47, 0.53, 0.31),  # shrubs
    30: (0.56, 0.62, 0.37),  # grass
    40: (0.70, 0.67, 0.42),  # crops: cane and fields, ochre
    50: (0.64, 0.58, 0.53),  # built-up, like the roofs
    60: (0.80, 0.74, 0.58),  # bare ground and sand
    70: (0.90, 0.90, 0.90),  # snow (none in Cuba)
    80: (0.14, 0.33, 0.43),  # water: left out of the mix (see colours)
    90: (0.39, 0.50, 0.38),  # herbaceous wetland: the ciénagas
    95: (0.22, 0.38, 0.28),  # mangrove
    100: (0.50, 0.56, 0.40),  # moss
}
LIGHT = np.array([-0.5, 0.7071, -0.5])  # toward the light: north-west (-x, -z), 45 degrees up
SHADE_FROM, SHADE_TO, SHADE_MAX = 3, 9, 0.35  # levels where the shading starts and reaches its strength
TINTS = ((0.0, (0.42, 0.58, 0.32)), (150.0, (0.64, 0.68, 0.40)), (400.0, (0.74, 0.62, 0.38)),
         (800.0, (0.58, 0.43, 0.29)), (1500.0, (0.60, 0.53, 0.48)))  # (metres, tone): an atlas's height tints
TINT_FROM, TINT_TO, TINT_MAX = 5, 9, 0.55  # levels where the tints start (the city stage) and reach their share


def palette(when) -> np.ndarray:
    """(len(CLASSES), 3) tones in the order of contravia.cover.CLASSES for the date `when`."""
    from contravia.cover import CLASSES

    return np.array([far_green(when) if code == 10 else TONES[code] for code in CLASSES], float)


def colours(shares, when) -> np.ndarray:
    """(..., 3) colours of squares with these class shares; a square with no data is GROUND. Water is left out:
    a square on the coast is half sea, which the ocean already draws, and would tint its land blue."""
    from contravia.cover import CLASSES

    s = np.asarray(shares, float).copy()
    s[..., CLASSES.index(80)] = 0.0
    total = s.sum(axis=-1, keepdims=True)
    mixed = s @ palette(when) / np.maximum(total, 1e-9)
    return np.where(total > 0, mixed, np.asarray(GROUND, float))


def tint(colour, heights, L: int) -> np.ndarray:
    """colour mixed with the height tint of each point (green low, ochre and brown high), by a share growing
    from 0 at level TINT_FROM to TINT_MAX at TINT_TO."""
    w = TINT_MAX * np.clip((L - TINT_FROM) / (TINT_TO - TINT_FROM), 0.0, 1.0)
    if w == 0:
        return np.asarray(colour, float)
    h = np.asarray(heights, float)
    stops = np.array([m for m, _ in TINTS])
    tones = np.array([t for _, t in TINTS])
    by_height = np.stack([np.interp(h, stops, tones[:, k]) for k in range(3)], axis=1)
    return np.asarray(colour, float) * (1 - w) + by_height * w


def exaggeration(L: int) -> float:
    """How much the map stretches heights when a level-L node is on screen: 1 to level 6, then doubling each
    level to 20, near what contravia.mapview.exaggeration gives for the heights those levels are drawn from."""
    return float(min(20.0, max(1.0, 2.0 ** (L - 6))))


def shade(colour, normals, L: int) -> np.ndarray:
    """colour darkened by up to SHADE_MAX where the normals turn further from LIGHT than flat ground does, from
    level SHADE_FROM up; flat ground and slopes toward the light keep their colour."""
    s = SHADE_MAX * np.clip((L - SHADE_FROM) / (SHADE_TO - SHADE_FROM), 0.0, 1.0)
    light = LIGHT / np.linalg.norm(LIGHT)
    lit = np.clip(np.asarray(normals, float) @ light / light[1], 0.0, 1.0)
    return np.asarray(colour, float) * (1 - s * (1 - lit))[:, None]
