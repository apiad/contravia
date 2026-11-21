"""contravia.hlod: level-1 summaries carry an image of their cells, painted on the CPU."""

import numpy as np
from manifoldx.resources import SURFACE_BAKED
from manifoldx.streaming import MeshData

from contravia import hlod
from contravia.geo import CELL


def quad(x0, z0, x1, z1, y, colour, up=True):
    """Two triangles over the rectangle at height y, facing up (or sideways)."""
    p = np.array([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], np.float32)
    n = np.tile((0, 1, 0) if up else (1, 0, 0), (4, 1)).astype(np.float32)
    return MeshData(p, n, np.tile(np.float32(colour), (4, 1)), np.array([0, 2, 1, 0, 3, 2], np.uint32))


def test_paint_draws_up_facing_triangles_lowest_first_x_along_the_width():
    side = 100.0
    ground = quad(0, 0, 100, 100, 0.0, (0.0, 1.0, 0.0))
    roof = quad(0, 0, 50, 100, 10.0, (1.0, 0.0, 0.0))  # the west half, higher: covers the ground
    wall = quad(50, 0, 100, 100, 20.0, (0.0, 0.0, 1.0), up=False)  # never painted
    img = hlod.paint([(roof, 0.0, 0.0), (ground, 0.0, 0.0), (wall, 0.0, 0.0)], side, n=16)
    assert img.shape == (16, 16, 3)
    assert tuple(img[8, 3]) == (255, 0, 0) and tuple(img[8, 12]) == (0, 255, 0)


def test_paint_puts_offset_meshes_where_they_stand_and_z_down_the_height():
    cell = quad(0, 0, 50, 50, 0.0, (1.0, 1.0, 1.0))
    img = hlod.paint([(cell, 0.0, 50.0)], 100.0, n=16)  # the south-west quarter: x < 50, z > 50
    assert img[12, 3].min() == 255 and img[3, 3].max() == 0 and img[12, 12].max() == 0


def test_base_goes_under_everything_whatever_its_height():
    summary = quad(0, 0, 100, 100, 30.0, (0.0, 0.0, 1.0))  # a tall merged block
    street = quad(0, 0, 100, 100, 0.0, (0.5, 0.5, 0.5))
    img = hlod.paint([(summary, 0.0, 0.0), (street, 0.0, 0.0)], 100.0, n=8, base=True)
    assert tuple(img[4, 4]) == (128, 128, 128)


def test_baked_marks_the_summarys_up_facing_vertices_and_keeps_its_walls():
    roof = quad(0, 0, 400, 400, 5.0, (0.2, 0.2, 0.2))
    wall = quad(0, 0, 400, 400, 5.0, (0.9, 0.9, 0.9), up=False)
    summary = MeshData(np.vstack([roof.positions, wall.positions]), np.vstack([roof.normals, wall.normals]),
                       np.vstack([roof.colors, wall.colors]), np.r_[roof.indices, wall.indices + 4])
    cell = quad(0, 0, CELL, CELL, 0.0, (1.0, 0.0, 0.0))
    out = hlod.baked(summary, [(cell, CELL, 0.0)], 2 * CELL, n=32)
    assert list(out.surfaces) == [SURFACE_BAKED] * 4 + [0] * 4
    assert out.albedo.shape == (32, 32, 3)
    assert tuple(out.albedo[8, 24]) == (255, 0, 0)  # the north-east cell over the summary
    assert tuple(out.albedo[24, 8]) == (51, 51, 51)  # where no cell draws, the summary shows
    assert np.array_equal(out.positions, summary.positions) and np.array_equal(out.colors, summary.colors)


def test_node_cells_builds_the_existing_cells_with_their_offsets():
    built = []

    def build(L, a, b):
        built.append((L, a, b))
        return quad(0, 0, 1, 1, 0.0, (1, 1, 1))

    cells = hlod.node_cells(build, lambda L, a, b: (a, b) != (7, 7), 1, 3, 3)
    assert built == [(0, 6, 6), (0, 6, 7), (0, 7, 6)]
    assert [(dx, dz) for _, dx, dz in cells] == [(0.0, 0.0), (0.0, CELL), (CELL, 0.0)]
