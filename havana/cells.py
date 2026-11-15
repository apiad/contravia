"""build(i, j): the mesh of one streamed cell, made from the city data by the rules.

manifoldx's Streamer calls build in a worker process for each cell that comes near the car. The city data
is loaded once per process, from the file named by HAVANA_DATA, and every rule adds its parts to one
MeshBuilder, so a cell is one mesh and one draw call. Buildings, filler and trees a landmark replaces or
clears are left out; the landmarks themselves are drawn once by the game, not per cell.
"""

import os

from havana.citydata import CityData
from havana.geo import CELL
from havana.landmarks import Landmarks
from havana.mesh import MeshBuilder
from havana.rules import buildings, roads, trees

GROUND = (0.32, 0.36, 0.30)
SKIPPED_ROOFS = 0  # footprints drawn as walls only in this process, because their roof could not be cut

_city: CityData | None = None
_city_path: str | None = None
_landmarks: Landmarks | None = None


def city() -> CityData:
    """The city data of this process, loaded on first use from HAVANA_DATA."""
    global _city, _city_path, _landmarks
    path = os.environ.get("HAVANA_DATA", "data/havana.npz")
    if _city is None or path != _city_path:
        _city, _city_path = CityData.load(path), path
        _landmarks = Landmarks(_city)
    return _city


def landmarks() -> Landmarks:
    """The landmarks placed on this process's city data."""
    city()
    return _landmarks


def build(i: int, j: int):
    """The MeshData of cell (i, j): its ground, roads, buildings, filler boxes and trees."""
    global SKIPPED_ROOFS
    c = landmarks().keep(i, j, city().cell(i, j))
    x0, z0 = i * CELL, j * CELL
    mb = MeshBuilder()
    mb.add([(x0, 0, z0), (x0 + CELL, 0, z0), (x0 + CELL, 0, z0 + CELL), (x0, 0, z0 + CELL)],
           [(0, 1, 0)] * 4, GROUND, [(0, 1, 2), (0, 2, 3)])
    roads.ribbon(mb, c.roads)
    for ring, osm_id in zip(c.buildings, c.building_ids, strict=True):
        if not buildings.box(mb, ring, int(osm_id)):
            SKIPPED_ROOFS += 1
    for block, seed in zip(c.fillers, c.filler_seeds, strict=True):
        buildings.filler(mb, block, int(seed))
    trees.trees(mb, c.trees)
    return mb.to_meshdata(x0, z0)
