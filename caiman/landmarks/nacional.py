"""The Hotel Nacional: its OSM footprint raised 30 m, with twin towers on the side facing the sea."""

from caiman.landmarks.shapes import Piece, frame, prism, pyramid, rect
from caiman.terrain import seat

NAME = "Hotel Nacional"
OSM_ID = 23866797
CREAM, TILE = (0.93, 0.87, 0.72), (0.62, 0.30, 0.20)


def place(city, mb) -> Piece | None:
    """Add the hotel on its footprint; None when the data has no such footprint."""
    ring = city.building(OSM_ID)
    if ring is None:
        return None
    y = seat(ring, city.terrain)
    prism(mb, ring, y, y + 30, CREAM)
    c, u, v, a, b = frame(ring)
    if v[1] > 0:  # the towers stand on the north side, toward -z
        v = -v
    for s in (-1, 1):
        tower = rect(c + v * (b - 8) + u * s * a * 0.3, u, 5, 5)
        prism(mb, tower, y + 30, y + 44, CREAM)
        pyramid(mb, tower, y + 44, y + 50, TILE)
    return Piece(replaces={OSM_ID}, walls=[ring])
