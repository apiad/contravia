from datetime import date

import numpy as np
import pytest

from contravia.forest import FADE, LOD_AT, Forest, Stand
from contravia.rules import foliage, trees
from contravia.rules.species import SPECIES
from contravia.rules.treegrammar import FORMS, _crown, skeleton


def _trees(xz, species="laurel"):
    xz = np.asarray(xz, float)
    n = len(xz)
    return trees.Trees(xz, np.zeros(n), np.full(n, SPECIES.index(species), np.int8),
                       np.ones(n), np.zeros(n), np.zeros(n, int), np.full(n, 0.5))


def _forest(monkeypatch, day=date(2026, 6, 15)):
    import manifoldx as mx

    monkeypatch.setattr(trees, "today", lambda: day)
    engine = mx.Engine("t", width=64, height=64)
    forest = Forest(engine, city=None, landmarks=None)
    now = [0.0]
    forest.clock = lambda: now[0]
    return engine, forest, now


def _flush(engine):
    engine.commands.execute(engine.store)
    engine.commands.clear()


def test_near_holds_a_tree_on_either_side_of_the_radius():
    d = np.array([foliage.NEAR - 1, foliage.NEAR + 1, foliage.NEAR + foliage.HOLD + 1, foliage.NEAR - foliage.HOLD - 1])
    assert list(foliage.near(d)) == [True, False, False, True]
    assert list(foliage.near(d, np.array([False, True, True, False]))) == [False, True, False, True]
    assert foliage.NEAR + foliage.HOLD < LOD_AT[0]  # a tree carrying the layer is always at LOD 0


@pytest.mark.parametrize("species", SPECIES)
def test_every_species_has_a_layer_of_thin_leaves_facing_where_it_is_seen_from(species):
    sp = foliage.spots(species, 0, "full")
    assert len(sp.key) > 0
    for key in dict.fromkeys(sp.key):
        g = foliage.geometry(key)
        tri = g["indices"].reshape(-1, 3)
        if key[0] == "frond":  # a palm's leaflets are seen from under and over the frond
            assert len(tri) % 2 == 0 and np.array_equal(tri[len(tri) // 2:], tri[: len(tri) // 2][:, ::-1])
        else:  # a clump's leaves face out of their cluster
            p = g["positions"][tri]
            n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            assert (np.einsum("ij,ij->i", n, p.mean(1)) > 0).mean() > 0.95
        assert np.allclose(np.linalg.norm(g["normals"], axis=1), 1.0)
        assert (g["colors"] > 0).all() and (g["colors"] <= 1).all()
    assert foliage.triangles(species, 0, "bare") == 0


def test_a_clump_covers_its_cluster_and_a_plume_follows_its_frond():
    pos, rot = np.array([[100.0, 2.0, -50.0]]), np.array([[0.0, np.sin(0.4), 0.0, np.cos(0.4)]])
    scale = np.full((1, 3), 1.3)
    sp = foliage.spots("laurel", 1, "full")
    (_tree, _accent, p, q, z), = foliage.instances(sp, pos, rot, scale, np.array([0.2])).values()
    blobs = _crown(skeleton("laurel", 1), 0, "full")
    want = pos + foliage.rotate(np.repeat(rot, len(blobs), 0), 1.3 * np.array([c for c, _, _ in blobs]))
    assert np.allclose(p, want) and np.allclose(z, 1.3 * np.array([r for _, r, _ in blobs]))
    # a frond's plume runs from the frond's foot to its tip as treegrammar grows it
    form, palm = FORMS["coco"], FORMS["coco"].palm
    made = foliage.instances(foliage.spots("coco", 0, "full"), pos, rot, scale, np.array([0.2]))
    (_, _, p, q, z) = made[("frond", "coco", True)]
    yaw = 2 * np.pi * 0 / palm.fronds + 0.3
    tip = np.array((form.lean, form.height, 0.0)) + palm.length * np.array([np.cos(yaw), 0, np.sin(yaw)]) \
        + (0.0, palm.rise - palm.droop, 0.0)
    chord = np.hypot(palm.length, palm.rise - palm.droop)
    assert np.allclose(p[0] + foliage.rotate(q[:1], z[:1] * (0.0, chord, 0.0)), pos + foliage.rotate(rot, 1.3 * tip))


def test_a_frond_sways_more_toward_its_tip():
    g = foliage.frond("palma", True)
    y = g["positions"][:, 1]
    assert y.min() > -0.2  # the shader sways by the height over the origin, which runs along the frond
    chord = np.hypot(FORMS["palma"].palm.length, FORMS["palma"].palm.rise - FORMS["palma"].palm.droop)
    assert y.max() > 0.9 * chord


def test_the_layer_grows_only_on_near_trees_at_lod_0_and_holds_at_the_radius(monkeypatch):
    _engine, forest, _now = _forest(monkeypatch)
    forest.stands[(0, 0)] = stand = Stand(_trees([[5.0, 0.0], [45.0, 0.0], [120.0, 0.0]]), date(2026, 6, 15))
    forest.place((0.0, 5.0, 0.0))
    assert list(stand.lod) == [0, 0, 1] and list(stand.near) == [True, False, False]
    clusters = len(foliage.spots("laurel", int(stand.variant[0]), "full").key)
    assert sum(len(a) for a in stand.leaves[0]) == clusters and not stand.leaves[1] and not stand.leaves[2]
    assert forest.counts()["near"] == 1 and forest.counts()["leaves"] == clusters
    forest.place((-foliage.NEAR - foliage.HOLD / 2 + 5.0, 5.0, 0.0))  # the first tree just past the radius: held
    assert stand.near[0]
    forest.place((-foliage.NEAR - foliage.HOLD - 1 + 5.0, 5.0, 0.0))
    assert not stand.near[0] and not stand.leaves[0] and stand.lod[0] == 0
    forest.place((45.0 - foliage.NEAR + foliage.HOLD / 2, 5.0, 0.0))  # the second tree nears, but not enough
    assert not stand.near[1]
    forest.place((45.0, 5.0, 0.0))
    assert list(stand.near) == [False, True, False]
    forest.close()


def test_the_layer_is_the_same_for_the_same_tree(monkeypatch):
    spots = [[5.0, 0.0], [9.0, 3.0]]
    seen = []
    for _ in range(2):
        engine, forest, _now = _forest(monkeypatch)
        forest.stands[(0, 0)] = stand = Stand(_trees(spots), date(2026, 6, 15))
        forest.place((0.0, 5.0, 0.0))
        tr = engine.store._components["Transform"]
        seen.append([tr[np.concatenate(stand.leaves[k])].copy() for k in range(2)])
        forest.close()
    assert all(np.array_equal(a, b) for a, b in zip(*seen, strict=True))
    assert not np.allclose(seen[0][0][:, 3:7], seen[0][1][:, 3:7])  # two trees turn their clumps apart


def test_the_layer_takes_the_seasons_tint(monkeypatch):
    engine, forest, _now = _forest(monkeypatch)
    forest.stands[(0, 0)] = stand = Stand(_trees([[5.0, 0.0]], "flamboyan"), date(2026, 6, 15))
    forest.place((0.0, 5.0, 0.0))
    tint = engine.store._components["Tint"]
    sp = foliage.spots("flamboyan", int(stand.variant[0]), "full")
    ents = np.concatenate(stand.leaves[0])
    want = np.where((sp.role == 3)[:, None], stand.tint[0, 2], stand.tint[0, 1])  # flowers take the accent
    assert (sp.role == 3).any() and np.allclose(tint[ents], want)
    forest.close()
    # in the dry season the ceiba stands bare: no leaves to lay over it
    engine, forest, _now = _forest(monkeypatch, date(2026, 2, 10))
    forest.stands[(0, 0)] = stand = Stand(_trees([[5.0, 0.0]], "ceiba"), date(2026, 2, 10))
    forest.place((0.0, 5.0, 0.0))
    assert stand.lod[0] == 0 and stand.near[0] and not stand.leaves[0]
    forest.close()


def test_the_layer_fades_in_and_out_with_its_tree(monkeypatch):
    engine, forest, now = _forest(monkeypatch)
    forest.stands[(0, 0)] = stand = Stand(_trees([[5.0, 0.0]], "palma"), date(2026, 6, 15))
    forest.place((0.0, 5.0, 0.0))
    fade = engine.store._components["Fade"]
    leaves, parts = np.concatenate(stand.leaves[0]), stand.ents[0][stand.ents[0] >= 0]
    assert np.allclose(fade[leaves], 0.0)
    now[0] = FADE / 2
    forest._step()
    assert np.allclose(fade[leaves], 0.5) and np.allclose(fade[parts], 0.5)
    now[0] = FADE
    forest._step()
    forest.place((LOD_AT[0] + 20.0, 5.0, 0.0))  # the tree drops to LOD 1: its layer goes with its old level
    assert not stand.near[0] and engine.store._alive[leaves].all()
    now[0] = FADE * 1.5
    forest._step()
    assert np.allclose(fade[leaves], -0.5) and np.allclose(fade[parts], -0.5)
    now[0] = FADE * 2.5
    forest._step()
    _flush(engine)
    assert not engine.store._alive[leaves].any()
    forest.close()


@pytest.mark.parametrize("species", [s for s in SPECIES if FORMS[s].palm is None])
def test_a_clumps_leaves_poke_out_of_its_blob(species):
    tips = foliage.clump(species)["positions"][3::4]  # each leaf's fourth point is its tip
    assert (np.linalg.norm(tips, axis=1) > 1.0).mean() > 0.9  # the blob's skin lies within the unit sphere
