import numpy as np
import pytest

from contravia.mesh import MeshBuilder, triangulate

L = np.array([[0, 0], [20, 0], [20, 10], [10, 10], [10, 20], [0, 20]], float)


def area(ring, tris):
    a, b, c = ring[tris[:, 0]], ring[tris[:, 1]], ring[tris[:, 2]]
    ab, ac = b - a, c - a
    return 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]).sum()


def test_ear_clipping_covers_an_l_shape_exactly():
    assert np.isclose(area(L, triangulate(L)), 300.0)


def test_clockwise_ring_triangulates_too():
    assert np.isclose(area(L[::-1], triangulate(L[::-1])), 300.0)


def test_collinear_points_do_not_break_it():
    ring = np.array([[0, 0], [5, 0], [10, 0], [10, 10], [0, 10]], float)
    assert np.isclose(area(ring, triangulate(ring)), 100.0)


def test_a_ring_that_cannot_be_clipped_raises():
    with pytest.raises(ValueError):
        triangulate(np.array([[0, 0], [1, 1], [2, 2]], float))


def test_meshdata_offsets_by_the_cell_corner():
    mb = MeshBuilder()
    mb.add(np.array([[200, 0, 400], [210, 0, 400], [200, 0, 410]], float), np.tile([0, 1, 0], (3, 1)),
           (1, 0, 0), np.array([[0, 1, 2]]))
    md = mb.to_meshdata(200, 400)
    assert md.positions[:, [0, 2]].tolist() == [[0, 0], [10, 0], [0, 10]]
    assert md.indices.dtype == np.uint32 and md.positions.dtype == np.float32


def test_a_small_ring_far_from_the_origin_keeps_its_area():
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([8e5 + 0.01 * np.cos(t), 3e5 + 0.01 * np.sin(t)], axis=1)  # a lantern cap in Guantánamo
    assert np.isclose(area(ring, triangulate(ring)), area(ring - ring[0], triangulate(ring - ring[0])))
