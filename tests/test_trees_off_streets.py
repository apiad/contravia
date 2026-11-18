"""Trees never stand on the street a cell draws, nor within CLEAR metres of it."""

import numpy as np

from contravia.rules.trees import CLEAR, off_streets


def test_a_tree_on_or_beside_a_junction_is_dropped():
    # a 20 m square of asphalt made of two triangles, at y = 0
    sq = np.array([[[0, 0, 0], [20, 0, 0], [20, 0, 20]], [[0, 0, 0], [20, 0, 20], [0, 0, 20]]], float)
    pts = np.array([[10, 10], [20 + CLEAR / 2, 10], [20 + CLEAR + 0.5, 10], [-5, -5]], float)
    assert off_streets(pts, sq).tolist() == [False, False, True, True]
    assert off_streets(pts, None).all()
