from datetime import date

import numpy as np

from contravia.forest import FADE, HYSTERESIS, LINGER, LOD_AT, Forest, Stand, levels, linear
from contravia.rules import trees
from contravia.rules.species import SPECIES, season


def _trees(xz, species="laurel", y=None):
    xz = np.asarray(xz, float)
    n = len(xz)
    return trees.Trees(xz, np.zeros(n) if y is None else np.asarray(y, float), np.full(n, SPECIES.index(species), np.int8),
                       np.ones(n), np.zeros(n), np.zeros(n, int), np.full(n, 0.5))


def test_levels_follow_distance_and_hold_near_a_threshold():
    d = np.array([10.0, LOD_AT[0] + 1, LOD_AT[1] + 1, 900.0])
    assert list(levels(d)) == [0, 1, 2, 2]
    old = np.array([0, 0, 1, 1], np.int8)
    held = levels(np.array([LOD_AT[0] + HYSTERESIS / 2, LOD_AT[0] + 2 * HYSTERESIS,
                            LOD_AT[1] - HYSTERESIS / 2, LOD_AT[1] + 2 * HYSTERESIS]), old)
    assert list(held) == [0, 1, 1, 2]


def test_a_stand_puts_each_tree_on_its_spot_and_ground_and_tints_it_for_the_day():
    t = _trees([[10.0, 0.0], [50.0, 20.0]], "flamboyan", y=[1.0, 6.0])
    s = Stand(t, date(2026, 6, 15))
    assert np.allclose(s.pos, [[10, 1, 0], [50, 6, 20]])
    look = season("flamboyan", date(2026, 6, 15))
    tone = 0.85 + 0.3 * 0.5
    assert np.allclose(s.tint[0, 1], linear(np.asarray(look.leaf) * tone))
    assert np.allclose(s.tint[0, 2], linear(np.asarray(look.accent) * tone))
    assert np.allclose(s.tint[:, 0], 1.0)  # the wood keeps its own colours
    assert np.allclose(np.linalg.norm(s.rot, axis=1), 1.0)


def test_the_trees_the_car_hits_are_the_ones_drawn():
    t = _trees([[10.0, 0.0], [50.0, 20.0], [-30.0, 7.0]])
    s = Stand(t, date(2026, 6, 15))
    walls = trees.trunks(t).reshape(len(t), 4, 2, 2)
    assert np.allclose(walls[:, :, 0].mean(axis=1), s.pos[:, [0, 2]])


def test_trees_are_instances_that_change_level_of_detail_with_the_camera(monkeypatch):
    import manifoldx as mx

    monkeypatch.setattr(trees, "today", lambda: date(2026, 6, 15))
    engine = mx.Engine("t", width=64, height=64)
    forest = Forest(engine, city=None, landmarks=None)
    monkeypatch.setattr(forest, "grow", lambda i, j: _trees([[5.0, 5.0], [400.0, 5.0], [800.0, 5.0]]))
    for _ in range(200):
        forest.update({(0, 0)}, (0.0, 5.0, 0.0))
        if forest.stands:
            break
        forest._pending[(0, 0)].result()
    assert list(forest.stands[(0, 0)].lod) == [0, 2, 2]
    near = int(engine.store._alive.sum())
    forest.place((400.0, 5.0, 0.0))  # the camera drives to the second tree
    assert list(forest.stands[(0, 0)].lod) == [2, 0, 2]
    stand = forest.stands[(0, 0)]
    shared = forest._meshes[("laurel", int(stand.variant[1]), "full", 0, "wood")]
    assert all(k[1] == 0 for k in forest._meshes if k[3] > 0)  # far trees share variant 0
    assert engine.store._components["Mesh"][stand.ents[1, 0], 0] == shared._geometry_id  # one mesh, many instances
    assert forest.counts() == {"trees": 3, "lod0": 1, "lod1": 0, "lod2": 2}
    forest.update(set(), (0.0, 5.0, 0.0))  # the streamer drops the cell: its trees linger a while
    assert forest.stands and near > 0
    for _ in range(LINGER):
        forest.update(set(), (0.0, 5.0, 0.0))
    assert not forest.stands
    monkeypatch.setattr(forest, "grow", None)  # it opens again from the rows it kept, grown once
    forest.update({(0, 0)}, (0.0, 5.0, 0.0))
    assert list(forest.stands[(0, 0)].lod) == [0, 2, 2] and not forest._pending
    forest.close()


def test_warm_grows_every_mesh_a_species_may_be_drawn_with():
    from contravia.forest import warm
    from contravia.rules.treegrammar import meshes

    meshes.cache_clear()
    warm([SPECIES.index("coco")])
    before = meshes.cache_info()
    for lod, crown in ((0, "full"), (1, "sparse"), (2, "bare")):
        meshes("coco", lod, crown, 0)
    meshes("coco", 0, "full", 1)
    assert meshes.cache_info().misses == before.misses  # each was grown already


def _flush(engine):
    """Run the commands the frame would, destroys among them."""
    engine.commands.execute(engine.store)
    engine.commands.clear()


def test_trees_fade_in_change_level_over_each_other_and_fade_out_before_they_go(monkeypatch):
    import manifoldx as mx

    monkeypatch.setattr(trees, "today", lambda: date(2026, 6, 15))
    engine = mx.Engine("t", width=64, height=64)
    forest = Forest(engine, city=None, landmarks=None)
    now = [0.0]
    forest.clock = lambda: now[0]
    monkeypatch.setattr(forest, "grow", lambda i, j: _trees([[5.0, 5.0], [400.0, 5.0]]))
    for _ in range(200):  # the grow thread may finish within the first update or after it
        forest.update({(0, 0)}, (0.0, 5.0, 0.0))  # the cell's trees spawn, drawing nothing yet
        if forest.stands:
            break
        forest._pending[(0, 0)].result()
    stand = forest.stands[(0, 0)]
    first = stand.ents[stand.ents >= 0]
    fade = engine.store._components["Fade"]
    assert np.allclose(fade[first], 0.0)
    now[0] = FADE / 2
    forest._step()
    assert np.allclose(fade[first], 0.5)
    now[0] = FADE
    forest._step()
    assert np.allclose(fade[first], 1.0) and not forest._fading

    old = stand.ents[1][stand.ents[1] >= 0]
    forest.place((400.0, 5.0, 0.0))  # the second tree comes near: its far level gives way to its near one
    new = stand.ents[1][stand.ents[1] >= 0]
    assert list(stand.lod) == [2, 0] and engine.store._alive[old].all()
    now[0] = FADE * 1.25
    forest._step()
    assert np.allclose(fade[old], -0.75) and np.allclose(fade[new], 0.25)  # complementary: every pixel once
    now[0] = FADE * 2
    forest._step()
    _flush(engine)
    assert not engine.store._alive[old].any() and np.allclose(fade[new], 1.0)

    gone = stand.ents[stand.ents >= 0]
    for _ in range(LINGER + 1):
        forest.update(set(), (400.0, 5.0, 0.0))  # the cell closes: its trees fade out, then go
    assert not forest.stands and engine.store._alive[gone].all()
    assert np.allclose(fade[gone], -1.0)
    now[0] = FADE * 2.5
    forest._step()
    assert np.allclose(fade[gone], -0.5)
    now[0] = FADE * 3.5
    forest._step()
    _flush(engine)
    assert not engine.store._alive[gone].any() and not forest._fading
    forest.close()


def test_a_tree_leaving_while_it_fades_in_fades_out_from_what_shows(monkeypatch):
    import manifoldx as mx

    monkeypatch.setattr(trees, "today", lambda: date(2026, 6, 15))
    engine = mx.Engine("t", width=64, height=64)
    forest = Forest(engine, city=None, landmarks=None)
    now = [0.0]
    forest.clock = lambda: now[0]
    forest.stands[(0, 0)] = stand = Stand(_trees([[5.0, 5.0]]), date(2026, 6, 15))
    forest.place((0.0, 5.0, 0.0))
    ents = stand.ents[stand.ents >= 0]
    fade = engine.store._components["Fade"]
    now[0] = FADE * 0.25
    forest._step()
    forest._drop(stand, np.arange(1))
    forest._step()
    assert np.allclose(fade[ents], -0.25)  # a quarter showed, a quarter still shows
    now[0] = FADE * 0.5
    forest._step()
    _flush(engine)
    assert not engine.store._alive[ents].any() and not forest._fading
    forest.close()
