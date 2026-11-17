"""Rain drops stay in their box and their number follows the rain."""

import numpy as np

from contravia.rain import BOX, POOL, fall, shown


def test_drops_stay_in_the_box():
    p = np.random.default_rng(1).uniform(-BOX / 2, BOX / 2, (500, 3))
    for _ in range(200):
        p = fall(p, 0.05)
    assert np.all(np.abs(p) <= BOX / 2)


def test_count_follows_intensity():
    assert shown(0.0) == 0 and shown(1.0) == POOL and shown(0.5) == POOL // 2 and shown(3) == POOL
