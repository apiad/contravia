"""The Plaza de la Revolución: the memorial's star tower on its base, Martí's statue, the paved square,
and the two ministries with the murals that face it.

The memorial, its base and the communications ministry are OSM footprints. The interior ministry has no
building in OSM, so it stands at the position Nominatim gives for it, as a box of its rough size.
"""

import numpy as np

from havana.geo import to_local
from havana.landmarks.shapes import Piece, circle, flat, panel, prism, rect

NAME = "Plaza de la Revolución"
TOWER, BASE, MINCOM = 1203463943, 23876175, 23876190
MININT = (23.1262275, -82.3858803)  # lat, lon of the interior ministry
WHITE, GREY, PAVE, DARK = (0.90, 0.90, 0.88), (0.80, 0.80, 0.78), (0.55, 0.55, 0.53), (0.12, 0.12, 0.12)
MINT_COLOR, MINCOM_COLOR = (0.78, 0.76, 0.70), (0.72, 0.74, 0.76)


def place(city, mb) -> Piece | None:
    """Add the square and its buildings; None when the data has no memorial."""
    tower, base = city.building(TOWER), city.building(BASE)
    if tower is None or base is None:
        return None
    t = tower.mean(axis=0)
    prism(mb, base, 0, 9, WHITE)
    prism(mb, tower, 9, 109, GREY)
    statue = (t[0], base[:, 1].min() - 15)  # in front of the memorial, on the square's side
    prism(mb, circle(statue, 3, 10), 0, 18, WHITE)

    m = to_local(np.array([MININT[0]]), np.array([MININT[1]]))[0]
    minint = rect(m, (1.0, 0.0), 55, 12)
    prism(mb, minint, 0, 32, MINT_COLOR)
    panel(mb, (m[0] - 12, m[1] + 12.3), (m[0] + 12, m[1] + 12.3), 6, 30, (0, 1), DARK)  # Che, facing south

    square = np.array([(t[0] - 130, m[1] + 30), (t[0] + 130, m[1] + 30),
                       (t[0] + 130, base[:, 1].min() - 5), (t[0] - 130, base[:, 1].min() - 5)])
    flat(mb, square, 0.02, PAVE)
    piece = Piece(replaces={TOWER, BASE}, clear=[square, rect(m, (1.0, 0.0), 60, 17)],
                  walls=[base, minint, circle(statue, 3, 10)])

    mincom = city.building(MINCOM)
    if mincom is not None:
        prism(mb, mincom, 0, 28, MINCOM_COLOR)
        # Camilo, on the wall whose midpoint is nearest the square's centre
        mids = (mincom + np.roll(mincom, -1, axis=0)) / 2
        k = int(np.argmin(np.hypot(*(mids - square.mean(axis=0)).T)))
        p0, p1 = mincom[k], mincom[(k + 1) % len(mincom)]
        out = square.mean(axis=0) - mids[k]
        out = out / np.linalg.norm(out)
        inset = (p1 - p0) * 0.15
        panel(mb, p0 + inset + out * 0.3, p1 - inset + out * 0.3, 5, 25, out, DARK)
        piece.replaces.add(MINCOM)
        piece.walls.append(mincom)
    return piece
