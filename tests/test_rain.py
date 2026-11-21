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


def test_configure_respawns_the_pool_at_the_presets_size():
    import manifoldx as mx

    from contravia import rain
    from contravia.quality import PRESETS

    engine = mx.Engine("rain-q", width=64, height=64)
    r = rain.Rain(engine)
    alive = int(engine.store._alive.sum())
    r.configure(PRESETS["bajo"])
    engine.commands.execute(engine.store)  # destroys apply at the frame's end, which then clears them
    engine.commands.clear()
    assert rain.shown(1.0) == 1000 and len(r.rel) == 1000
    assert int(engine.store._alive.sum()) == alive - 2000
    r.configure(PRESETS["alto"])
    engine.commands.execute(engine.store)
    engine.commands.clear()
    assert rain.shown(1.0) == 3000 and int(engine.store._alive.sum()) == alive
