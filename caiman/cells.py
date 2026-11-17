"""build_node(L, a, b): the mesh of one streamed quadtree node; build(i, j): the mesh of one cell.

manifoldx's QuadStreamer calls build_node in a worker process for each node it picks. A level-0 node is
one cell, built in full by the rules; a coarser node is its summary's boxes, ground and roads
(caiman/summary.py). The world is loaded once per process from the files named by HAVANA_DATA and,
for the rest of the island, HAVANA_TILES, and every rule adds its parts to one
MeshBuilder, so a cell is one mesh and one draw call. Buildings, filler and trees a landmark replaces or
clears are left out; the landmarks themselves are drawn once by the game, not per cell.
"""

import os

from caiman.clearance import boxes, roads_around
from caiman.geo import CELL
from caiman.landmarks import Landmarks
from caiman.mesh import MeshBuilder
from caiman.rules import buildings, ground, roads, trees
from caiman.terrain import seat
from caiman.world import World

SKIPPED_ROOFS = 0  # footprints drawn as walls only in this process, because their roof could not be cut

_city: World | None = None
_city_path: tuple | None = None
_landmarks: Landmarks | None = None


def city() -> World:
    """The world of this process, loaded on first use from HAVANA_DATA and HAVANA_TILES (if set)."""
    global _city, _city_path, _landmarks
    path = (os.environ.get("HAVANA_DATA", "data/havana.npz"), os.environ.get("HAVANA_TILES") or None)
    if _city is None or path != _city_path:
        _city, _city_path = World.load(*path), path
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
    terrain = city().terrain
    ground.ground(mb, i, j, terrain)
    if c.street_tris is None:
        roads.ribbon(mb, c.roads, terrain)
    else:
        roads.streets(mb, c.street_tris, c.street_edges, terrain, c.street_marks)
    for ring, style_id in boxes(c, roads_around(city(), i, j)):
        if not buildings.box(mb, ring, style_id, base=seat(ring, terrain)):
            SKIPPED_ROOFS += 1
    trees.trees(mb, c.trees, terrain.height(c.trees[:, 0], c.trees[:, 1]))
    if not (len(c.buildings) or len(c.fillers) or len(c.trees)):  # land OSM says nothing about
        spots, size, tone = trees.wild(i, j, terrain, roads_around(city(), i, j))
        trees.trees(mb, spots, terrain.height(spots[:, 0], spots[:, 1]), size, tone)
    return mb.to_meshdata(x0, z0)


def build_node(L: int, a: int, b: int):
    """The MeshData of quadtree node (L, a, b), relative to its corner: a full cell at level 0, else its summary."""
    from caiman.summary import summary_mesh

    w = city()
    i, j = w.tree.origin[0] + (a << L), w.tree.origin[1] + (b << L)
    if L == 0:
        return build(i, j)
    return summary_mesh(w.summary(L, a, b), L, a, b, w.tree.origin, w.terrain)
