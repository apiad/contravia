"""Trees never stand on the street a cell draws, nor within CLEAR metres of it."""

import numpy as np
import shapely

from contravia.rules.trees import CLEAR, off_streets


def test_a_tree_on_or_beside_a_junction_is_dropped():
    # a 20 m square of asphalt made of two triangles, at y = 0
    sq = np.array([[[0, 0, 0], [20, 0, 0], [20, 0, 20]], [[0, 0, 0], [20, 0, 20], [0, 0, 20]]], float)
    pts = np.array([[10, 10], [20 + CLEAR / 2, 10], [20 + CLEAR + 0.5, 10], [-5, -5]], float)
    assert off_streets(pts, sq).tolist() == [False, False, True, True]
    assert off_streets(pts, None).all()


def test_a_trunk_wider_than_the_clearance_is_kept_off_the_street_by_its_own_radius():
    from contravia.rules import trees
    from contravia.rules.species import SPECIES, TRUNK_R

    wide = max(range(len(SPECIES)), key=lambda k: TRUNK_R[SPECIES[k]])
    r = TRUNK_R[SPECIES[wide]] * 1.5
    street = np.array([[[0.0, -50.0], [0.0, 50.0], [-8.0, 50.0]], [[0.0, -50.0], [-8.0, 50.0], [-8.0, -50.0]]])
    xz = np.array([[r - 0.1, 0.0], [r + trees.TRUNK_CLEAR + 0.1, 20.0]])       # the street's edge is x = 0
    n = len(xz)
    t = trees.Trees(xz, np.zeros(n), np.full(n, wide, np.int8), np.full(n, 1.5), np.zeros(n), np.zeros(n, int),
                    np.full(n, 0.5))
    kept = trees.off_trunks(t, street)
    assert len(kept) == 1 and kept.xz[0, 1] == 20.0
    assert not shapely.intersects(shapely.polygons(street), shapely.linestrings(trees.trunks(kept))[:, None]).any()
