"""The FOCSA: its OSM footprint raised to 121 m, the tallest building in the city."""

from havana.landmarks.shapes import Piece, prism

NAME = "FOCSA"
OSM_ID = 23866811
WHITE = (0.85, 0.85, 0.82)


def place(city, mb) -> Piece | None:
    """Add the tower on its footprint; None when the data has no such footprint."""
    ring = city.building(OSM_ID)
    if ring is None:
        return None
    prism(mb, ring, 0, 121, WHITE)
    return Piece(replaces={OSM_ID}, walls=[ring])
