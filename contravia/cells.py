"""build_node(L, a, b): the mesh of one streamed quadtree node; build(i, j): the mesh of one cell.

manifoldx's QuadStreamer calls build_node in a worker process for each node it picks. A level-0 node is
one cell, built in full by the rules; a coarser node is its summary's boxes, ground and roads
(contravia/summary.py). The world is loaded once per process from HAVANA_DATA (the island's folder of
provinces, contravia/world.py), and every rule adds its parts to one MeshBuilder, so a cell is one mesh and one draw call. Buildings, filler and trees a landmark replaces or
clears are left out; the landmarks themselves are drawn once by the game, not per cell.
"""

import os

from contravia.clearance import boxes, roads_around, streets_around
from contravia.geo import CELL
from contravia.landmarks import Landmarks, cached
from contravia.mesh import MeshBuilder
from contravia.rules import buildings, detail, facades, ground, roads
from contravia.terrain import seat
from contravia.world import World

SKIPPED_ROOFS = 0  # footprints drawn as walls only in this process, because their roof could not be cut

_city: World | None = None
_city_path: tuple | None = None
_landmarks: Landmarks | None = None


def city() -> World:
    """The world of this process, loaded on first use from HAVANA_DATA: the island's folder or one province."""
    global _city, _city_path, _landmarks
    path = (os.environ.get("HAVANA_DATA", "data/cuba"),)
    if _city is None or path != _city_path:
        _city, _city_path = World.load(*path), path
        _landmarks = cached(_city)
    return _city


def landmarks() -> Landmarks:
    """The landmarks placed on this process's city data."""
    city()
    return _landmarks


def warm() -> None:
    """What a builder loads once, run when it starts while the game is still loading: the world, its landmarks and
    the parks lookup of the focused province, which its first cell would otherwise wait for."""
    from contravia import parks

    w = city()
    landmarks()
    a = w.province(w.names.index(w.focused)).a
    if "park_kind" in a and len(a["park_kind"]):
        parks._lookup(a)


def build(i: int, j: int):
    """The MeshData of cell (i, j): its ground, roads, buildings and filler boxes. Its trees are not in it: the
    game draws them as instances (forest.py)."""
    global SKIPPED_ROOFS
    c = landmarks().keep(i, j, city().cell(i, j))
    x0, z0 = i * CELL, j * CELL
    mb = MeshBuilder()
    terrain = city().terrain
    ground.ground(mb, i, j, terrain, c.street_cuts)
    if c.street_tris is None:
        roads.ribbon(mb, c.roads, terrain)
    else:
        roads.streets(mb, c.street_tris, c.street_edges, terrain, c.street_marks)
        if c.struct_tris is not None:
            roads.structures(mb, c.struct_tris, c.struct_paint)
        if c.walk_tris is not None:
            roads.walks(mb, c.walk_tris, c.walk_paint)
        if c.park_tris is not None:
            roads.structures(mb, c.park_tris, c.park_paint)
    near = roads_around(city(), i, j)
    for ring, style_id, plan in boxes(c, near, streets_around(city(), i, j)):
        portals = facades.portal_walls(ring, near) if plan.portal else None
        if not buildings.box(mb, ring, style_id, base=seat(ring, terrain), plan=plan, portals=portals):
            SKIPPED_ROOFS += 1
    return mb.to_meshdata(x0, z0)


def build_detail(i: int, j: int):
    """The detail ring's MeshData of cell (i, j): cornices, balconies, sills, coping and roof tanks of the
    buildings the cell's massing draws (rules/detail.py), or None for a cell with no buildings."""
    c = landmarks().keep(i, j, city().cell(i, j))
    terrain = city().terrain
    near = roads_around(city(), i, j)
    mb = MeshBuilder()
    for ring, style_id, plan in boxes(c, near, streets_around(city(), i, j)):
        portals = facades.portal_walls(ring, near) if plan.portal else None
        detail.dress(mb, ring, style_id, seat(ring, terrain), plan, near, portals)
    if not mb._pos:
        return None
    return mb.to_meshdata(i * CELL, j * CELL)


def build_node(L: int, a: int, b: int):
    """The MeshData of quadtree node (L, a, b), relative to its corner: a full cell at level 0, else its summary."""
    from contravia.summary import summary_mesh

    w = city()
    i, j = w.tree.origin[0] + (a << L), w.tree.origin[1] + (b << L)
    if L == 0:
        return build(i, j)
    return summary_mesh(w.summary(L, a, b), L, a, b, w.tree.origin, w.terrain)
