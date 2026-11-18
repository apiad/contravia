

def test_green_grows_the_same_trees_each_time_one_per_150_square_metres():
    import numpy as np

    from contravia.rules.trees import TREE_EVERY, in_green

    tri = np.array([[[0.0, 0.0], [200.0, 0.0], [0.0, 200.0]]])  # 20 000 m²
    a, b = in_green(tri, [42]), in_green(tri, [42])
    assert np.array_equal(a, b) and abs(len(a) - 20_000 / TREE_EVERY) <= 1
    assert (a[:, 0] + a[:, 1] <= 200.0 + 1e-6).all() and (a >= 0).all()
    assert not np.array_equal(a, in_green(tri, [43]))
