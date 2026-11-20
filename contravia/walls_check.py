"""contravia walls-check: count what the car would hit inside a road's drivable width, by where it comes from.

The car hits OSM footprints, filler boxes and the landmarks' walls (contravia/walls.py), and none of them may
stand in a road the car can drive (contravia/clearance.py). This walks every cell of Havana's data, takes the
boxes the cells draw and the landmark walls, and counts those a drivable road enters: boxes by footprint
and filler, landmark walls by segment. Every count must be 0.
"""

from collections import Counter
from dataclasses import replace

import numpy as np

from contravia.citydata import CityData
from contravia.clearance import boxes, entering, roads_around, streets_around
from contravia.landmarks import Landmarks


class _Kept:
    """The city as the car drives it: each cell after the landmarks' filter, so blocked roads are gone."""

    def __init__(self, city, landmarks):
        self.city, self.landmarks = city, landmarks

    def cell(self, i, j):
        return self.landmarks.keep(i, j, self.city.cell(i, j))


def count(city) -> Counter:
    """Boxes and landmark wall segments that a drivable road enters, by source, over every cell."""
    landmarks = Landmarks(city)
    kept = _Kept(city, landmarks)
    out = Counter({"footprint": 0, "filler": 0, "landmark": 0})
    no = np.zeros(0, np.int64)
    for i, j in city.cells().tolist():
        c = kept.cell(i, j)
        roads = roads_around(kept, i, j)
        near, streets = roads_around(city, i, j), streets_around(city, i, j)
        drawn = {"footprint": boxes(replace(c, fillers=[], filler_seeds=no), near, streets),
                 "filler": boxes(replace(c, buildings=[], building_ids=no, building_attrs=None), near, streets)}
        for source, items in drawn.items():
            out[source] += sum(bool(entering(ring, roads).any()) for ring, *_ in items)
        out["landmark"] += sum(bool(entering(seg, roads).any()) for seg in landmarks.walls(i, j))
    return out


def main(data: str) -> int:
    """Print the counts; the exit status is 1 when any is above 0."""
    counts = count(CityData.load(data))
    for source, n in counts.items():
        print(f"{source:9s} {n:6d} inside a road")
    return int(any(counts.values()))
