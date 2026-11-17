"""The Morro: the castle's OSM footprints as low stone walls, and a lighthouse on its north-west point."""

import numpy as np

from contravia.landmarks.shapes import Piece, circle, prism, pyramid
from contravia.terrain import seat

NAME = "Morro"
CASTLE, ANNEX = 23918434, 23918435
STONE, WHITE, DARK = (0.78, 0.70, 0.52), (0.92, 0.90, 0.85), (0.15, 0.15, 0.15)


def place(city, mb) -> Piece | None:
    """Add the castle and its lighthouse; None when the data has no castle footprint."""
    ring = city.building(CASTLE)
    if ring is None:
        return None
    t = city.terrain
    low, top = seat(ring, t), float(t.height(ring[:, 0], ring[:, 1]).max()) + 14
    prism(mb, ring, low, top, STONE)
    piece = Piece(replaces={CASTLE}, walls=[ring])
    annex = city.building(ANNEX)
    if annex is not None:
        prism(mb, annex, seat(annex, t), float(t.height(annex[:, 0], annex[:, 1]).max()) + 10, STONE)
        piece.replaces.add(ANNEX)
        piece.walls.append(annex)
    tip = ring[np.argmin(ring[:, 0] + ring[:, 1])]  # north-west is -x and -z
    c = tip + (ring.mean(axis=0) - tip) / np.linalg.norm(ring.mean(axis=0) - tip) * 8
    prism(mb, circle(c, 3.5, 16), top, top + 30, WHITE)
    prism(mb, circle(c, 2.5, 12), top + 30, top + 34, DARK)
    pyramid(mb, circle(c, 2.8, 12), top + 34, top + 37, DARK)
    return piece
