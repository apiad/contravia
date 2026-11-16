"""The adaptive quadtree: every cell in one leaf, leaves within the split, negative cells floored."""

import numpy as np

from havana.quadtree import Tree


def _random_weights(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    cells = {(int(i), int(j)) for i, j in rng.integers(-40, 60, size=(n, 2))}
    return {c: float(rng.integers(1, 400)) for c in cells}


def test_every_cell_in_one_leaf_and_leaves_within_split():
    weights = _random_weights()
    tree = Tree.build(weights)
    for c in weights:
        assert tree.leaf_of(*c) is not None
    held = {}
    for c, w in weights.items():
        held[tree.leaf_of(*c)] = held.get(tree.leaf_of(*c), 0.0) + w
    assert held == tree.leaves  # each cell counted in exactly one leaf
    assert all(w <= 2000 or L == 0 for (L, _, _), w in tree.leaves.items())
    assert Tree.build(weights).leaves == tree.leaves


def test_negative_cells_and_existence():
    tree = Tree.build({(-5, -3): 3000.0, (7, 2): 1.0, (8, 2): 1.0}, split=2000)
    leaf = tree.leaf_of(-5, -3)
    assert leaf[0] == 0  # the heavy cell splits down to level 0
    assert tree.leaf_of(7, 2) is not None
    L, a, b = tree.leaf_of(7, 2)
    assert tree.exists(L + 1, a >> 1, b >> 1)  # an ancestor exists
    if L > 0:
        assert tree.exists(0, *(np.array([7, 2]) - tree.origin))  # the weighted cell inside the leaf
        assert not tree.exists(0, *(np.array([7, 3]) - tree.origin))  # an empty cell inside it does not
