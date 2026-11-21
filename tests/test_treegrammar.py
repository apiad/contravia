from datetime import date

import numpy as np
import pytest

from contravia.rules.species import CROWNS, LEAF, SPECIES, season
from contravia.rules.treegrammar import LODS, VARIANTS, build, meshes, skeleton, triangles


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("lod", range(LODS))
def test_a_mesh_stands_on_the_origin_with_unit_normals(species, lod):
    t = build(species, lod).done()
    assert np.allclose(np.linalg.norm(t.normals, axis=1), 1.0)
    assert t.positions[:, 1].min() >= -0.01
    base = t.positions[t.positions[:, 1] < 0.05]
    assert np.linalg.norm(base[:, [0, 2]].mean(axis=0)) < 0.6  # the trunk (and its buttresses) round the origin


@pytest.mark.parametrize("species", SPECIES)
def test_each_level_of_detail_draws_less_and_the_far_one_is_as_cheap_as_the_old_trees(species):
    n = [triangles(species, lod) for lod in range(LODS)]
    assert n[0] > n[1] > n[2]
    assert n[2] <= 80  # the old one-detail trees drew 62 to 82
    assert n[0] >= 4 * 80 or species in ("palma", "coco")  # the near tree has much more detail
    for lod in range(LODS):
        full, sparse, bare = (triangles(species, lod, c) for c in CROWNS)
        assert full >= sparse >= bare


def test_the_grammar_is_seeded_and_variants_differ():
    a, b = skeleton.__wrapped__("laurel", 0), skeleton.__wrapped__("laurel", 0)
    assert all(np.allclose(x[1], y[1]) for x, y in zip(a.wood, b.wood, strict=True))
    if VARIANTS > 1:
        c = skeleton.__wrapped__("laurel", 1)
        assert not np.allclose(a.clusters[0][0], c.clusters[0][0])


def test_branches_grow_in_levels_from_the_trunk_and_carry_the_leaves():
    sk = skeleton("laurel")
    levels = [w[4] for w in sk.wood]
    assert levels.count(0) == 1 and levels.count(1) == 5 and levels.count(2) == 15
    tips = np.array([w[1] for w in sk.wood if w[4] == 2])
    for c, *_ in sk.clusters:  # every cluster hangs near a twig's tip
        assert np.linalg.norm(tips - c, axis=1).min() < 3.0


def test_crowns_are_darker_underneath_and_leaves_are_grey_for_the_tint():
    t = build("laurel", 0).done()
    leaf = t.role == LEAF
    assert t.shade[leaf & (t.normals[:, 1] < -0.5)].mean() < t.shade[leaf & (t.normals[:, 1] > 0.5)].mean()
    g = meshes("laurel", 0)["leaf"]["colors"]
    assert np.allclose(g[:, 0], g[:, 1]) and np.allclose(g[:, 1], g[:, 2])


def test_accent_parts_exist_where_a_season_paints_them():
    for s, day in (("flamboyan", date(2026, 6, 15)), ("almendro", date(2026, 1, 15))):
        for lod in range(LODS - (s == "almendro")):  # the far almendro is one green crown
            assert "accent" in meshes(s, lod, season(s, day).crown)


def test_a_triangle_faces_along_the_normalised_cross_product_of_its_edges():
    from contravia.rules.species import _Mesh

    rng = np.random.default_rng(3)
    m = _Mesh()
    for a, b, c in rng.normal(0, 5, (300, 3, 3)):
        m.tri(a, b, c, LEAF)
    for k in range(300):
        a, b, c = m.pos[3 * k: 3 * k + 3]
        n = np.cross(b - a, c - a)
        assert np.array_equal(m.nrm[3 * k], n / np.linalg.norm(n))
