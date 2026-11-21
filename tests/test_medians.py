"""Medians: Carlos III's kerb with lamps, Boyeros' wall, the boulevards' paving."""

import numpy as np

from contravia import medians
from contravia.graph import build_graph
from contravia.prepare import WIDTHS
from contravia.streets import LIFT, arclen, build
from contravia.structures import BARRIER, KERB_TOP, LAMPS, POLE


def _avenue(name, z0=0.0):
    """Two three-lane one-way carriageways 10.5 m apart along x at z0, named `name`, crossed at x = 0 and 200."""
    def way(i, hw, refs, pts, **tags):
        return (i, hw, refs, pts, tags)

    xs = (-200.0, 0.0, 200.0, 400.0)
    ways = [way(1, "primary", [100, 101, 102, 103], [(z0 - 5.25, x) for x in xs], lanes="3", oneway="yes", name=name),
            way(2, "primary", [203, 202, 201, 200], [(z0 + 5.25, x) for x in xs[::-1]], lanes="3", oneway="yes",
                name=name)]
    for n, x in ((1, 0.0), (2, 200.0)):
        ways.append(way(10 + n, "residential", [900 + n, 100 + n, 200 + n, 950 + n],
                        [(z0 - 100, x), (z0 - 5.25, x), (z0 + 5.25, x), (z0 + 100, x)], name=f"Calle {n}"))
    ident = lambda lat, lon: np.stack([np.asarray(lon, float), np.asarray(lat, float)], axis=1)  # noqa: E731
    a = build_graph(ways, {}, [], [], [], ident, WIDTHS, lambda p: np.ones(len(p), bool))
    ribbons, owner, axes, _, _, chains = build(a, WIDTHS)
    return a, ribbons, owner, axes, chains


def _flat(axes):
    """Each ribbon's centreline profile on flat ground."""
    return [(arclen(q), np.full(len(q), LIFT)) for q in axes]


def test_carlos_iii_gets_a_kerb_with_lamps_on_its_centreline():
    a, ribbons, owner, axes, chains = _avenue("Avenida Salvador Allende (Carlos III)")
    tris, paint, keys, walls, _ = medians.build(ribbons, owner, axes, chains, _flat(axes), a)
    kerb = tris[paint == KERB_TOP]
    mid = kerb[(kerb[:, :, 0] > 30).all(axis=1) & (kerb[:, :, 0] < 170).all(axis=1)]  # away from the junctions
    assert len(mid) and np.abs(mid[:, :, 2]).max() < medians.KERB_W / 2 + 1.0         # on the centreline (it drifts under 1 m)
    assert (paint == POLE).any() and (paint == LAMPS).any()
    assert tris[paint == POLE][:, :, 1].max() > medians.LAMP_H                     # tall lamps
    assert not len(walls) and len(keys) == len(tris)


def test_boyeros_south_of_the_rotonda_gets_a_wall_the_car_hits():
    z = medians.ROTONDA_Z + 1000
    a, ribbons, owner, axes, chains = _avenue("Avenida Rancho Boyeros", z0=z)
    tris, paint, _, walls, _ = medians.build(ribbons, owner, axes, chains, _flat(axes), a)
    assert (paint == BARRIER).any() and len(walls)
    mid = walls[(walls[:, [0, 2]] > 30).all(axis=1) & (walls[:, [0, 2]] < 170).all(axis=1)]
    assert len(mid) and (np.abs(mid[:, [1, 3]] - z) < medians.WALL_W / 2 + 1.0).all()
    assert (walls[:, 5] - walls[:, 4] >= medians.WALL_H - 1e-5).all()


def test_another_avenue_gets_no_median():
    a, ribbons, owner, axes, chains = _avenue("Avenida Cualquiera")
    tris, *_ = medians.build(ribbons, owner, axes, chains, _flat(axes), a)
    assert not len(tris)


def test_the_ground_between_a_boulevards_carriageways_is_paved_but_a_city_block_is_not():
    import shapely

    a, *_ = _avenue("Paseo")
    blocks = np.array([shapely.box(20, -5.25, 180, 5.25), shapely.box(20, 20, 120, 120)], dtype=object)
    assert medians.boulevard(a, blocks).tolist() == [True, False]
