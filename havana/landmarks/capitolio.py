"""The Capitolio: its OSM footprint raised 22 m, then a drum, a stepped dome, a lantern and a spire to 92 m."""

from havana.landmarks.shapes import Piece, circle, dome, prism, pyramid

NAME = "Capitolio"
OSM_ID = 12639209
STONE, WHITE, GOLD = (0.86, 0.82, 0.70), (0.90, 0.90, 0.88), (0.75, 0.68, 0.40)


def place(city, mb) -> Piece | None:
    """Add the Capitolio on its footprint; None when the data has no such footprint."""
    ring = city.building(OSM_ID)
    if ring is None:
        return None
    c = ring.mean(axis=0)
    prism(mb, ring, 0, 22, STONE)
    prism(mb, circle(c, 18), 22, 26, STONE)
    prism(mb, circle(c, 15), 26, 44, STONE, roof=False)
    dome(mb, c, 15, 44, 16, WHITE)
    prism(mb, circle(c, 3, 12), 60, 76, WHITE)
    pyramid(mb, circle(c, 2, 8), 76, 92, GOLD)
    return Piece(replaces={OSM_ID}, walls=[ring])
