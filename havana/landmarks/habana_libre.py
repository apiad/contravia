"""The Habana Libre: a 12 m podium on its OSM footprint and a slab tower to 126 m along its long side."""

from havana.landmarks.shapes import Piece, frame, prism, rect

NAME = "Habana Libre"
OSM_ID = 289803271
PODIUM, GLASS, CAP = (0.88, 0.88, 0.85), (0.62, 0.72, 0.78), (0.92, 0.92, 0.90)


def place(city, mb) -> Piece | None:
    """Add the hotel on its footprint; None when the data has no such footprint."""
    ring = city.building(OSM_ID)
    if ring is None:
        return None
    prism(mb, ring, 0, 12, PODIUM)
    c, u, _, a, _ = frame(ring)
    slab = rect(c, u, min(38, a - 5), 11)
    prism(mb, slab, 12, 124, GLASS)
    prism(mb, slab, 124, 126, CAP)
    return Piece(replaces={OSM_ID}, walls=[ring])
