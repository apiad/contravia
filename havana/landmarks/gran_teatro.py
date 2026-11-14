"""The Gran Teatro: its OSM footprint raised 24 m, with a turret and spire at each corner."""

from havana.landmarks.shapes import Piece, frame, prism, pyramid, rect

NAME = "Gran Teatro"
OSM_ID = 19355360
BODY, ROOF = (0.93, 0.92, 0.86), (0.55, 0.55, 0.52)


def place(city, mb) -> Piece | None:
    """Add the theatre on its footprint; None when the data has no such footprint."""
    ring = city.building(OSM_ID)
    if ring is None:
        return None
    prism(mb, ring, 0, 24, BODY)
    c, u, v, a, b = frame(ring)
    for su in (-1, 1):
        for sv in (-1, 1):
            corner = c + u * su * (a - 5) + v * sv * (b - 5)
            turret = rect(corner, u, 4, 4)
            prism(mb, turret, 24, 34, BODY)
            pyramid(mb, turret, 34, 42, ROOF)
    return Piece(replaces={OSM_ID}, walls=[ring])
