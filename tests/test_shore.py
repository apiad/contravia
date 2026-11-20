"""The shore: what OSM's tags make of it, the distances and types its grid measures, the profiles and colours of
each type, and a coastal cell's field over a made-up coast (contravia/shore.py, contravia/rules/shore.py)."""

import numpy as np
import pytest
import shapely

from contravia import shore
from contravia.geo import CELL
from contravia.rules import shore as R
from contravia.rules.ground import SEA, drawn
from contravia.terrain import Terrain


def slope(rise=0.02, coast=1100.0):
    """Ground rising `rise` per metre east of x = coast, under the sea west of it."""
    x = np.arange(0.0, 3001.0, 10.0)
    h = np.tile((x - coast) * rise + SEA, (301, 1)).astype(np.float32)
    return Terrain(h, (0.0, 0.0), (10.0, 10.0), fade=0.0)


@pytest.mark.parametrize("tags,kind", [({"natural": "beach"}, "beach"), ({"natural": "sand"}, "beach"),
                                       ({"natural": "cliff"}, "cliff"),
                                       ({"natural": "wetland", "wetland": "mangrove"}, "mangrove"),
                                       ({"natural": "wetland"}, "mangrove"), ({"natural": "wood"}, None), ({}, None)])
def test_kind_of_tags(tags, kind):
    assert shore.kind_of(tags) == kind


def test_save_and_load_keep_the_features(tmp_path):
    features = {"beach": shapely.box(0, 0, 10, 10), "cliff": shapely.LineString([(0, 0), (5, 5)]),
                "mangrove": shapely.Polygon()}
    shore.save(features, tmp_path / shore.NAME)
    back = shore.load(tmp_path / shore.NAME)
    assert back["beach"].equals(features["beach"]) and back["cliff"].equals(features["cliff"])
    assert back["mangrove"].is_empty


def test_distance_is_exact_up_to_its_cap():
    rng = np.random.default_rng(3)
    mask = rng.random((50, 60)) < 0.01
    rr, cc = np.nonzero(mask)
    g = np.indices(mask.shape)
    brute = np.sqrt(((g[0][..., None] - rr) ** 2 + (g[1][..., None] - cc) ** 2).min(-1)) * R.STEP
    got = R.distance(mask)
    near = brute <= R.REACH + R.STEP
    assert np.allclose(got[near], brute[near]) and np.isinf(got[~near]).all()


def test_signed_distance_is_positive_on_land_and_crosses_zero_at_the_water():
    sea = np.zeros((40, 40), bool)
    sea[:, :20] = True
    d = R.signed(sea)
    assert (d[:, :18] < 0).all() and (d[:, 22:] > 0).all()
    assert abs(d[20, 19] + d[20, 20]) < 1e-9  # the waterline lies half way between the last sea and first land node
    assert d.max() <= R.REACH and d.min() >= -R.REACH


def test_kind_takes_mapped_cliffs_then_beaches_then_mangrove_then_steep_ground_else_rock():
    f = np.zeros(5, bool)
    t = np.array([True, False, False, False, False])
    assert R.kind(t, t, t, t)[0] == R.CLIFF
    assert R.kind(f, np.roll(t, 1), np.roll(t, 1), np.roll(t, 1))[1] == R.BEACH
    assert R.kind(f, f, np.roll(t, 2), np.roll(t, 2))[2] == R.MANGROVE
    assert R.kind(f, f, f, np.roll(t, 3))[3] == R.CLIFF
    assert R.kind(f, f, f, f)[4] == R.ROCK


def test_shares_sum_to_one_and_blend_between_types():
    types = np.full((40, 40), R.ROCK, np.int8)
    types[:, 20:] = R.BEACH
    w = R.shares(types)
    assert np.allclose(w.sum(axis=-1), 1.0)
    assert w[5, 2, R.ROCK] == 1.0 and w[5, 37, R.BEACH] == 1.0
    assert 0 < w[5, 20, R.BEACH] < 1


def test_profiles_meet_the_water_and_return_to_the_ground_inland():
    d = np.array([-5.0, 0.0, 10.0, 40.0])
    h = np.full(4, 3.0)
    top = np.full(4, 9.0)
    p = R.profiles(d, h, top, np.full(4, 0.5))
    assert np.allclose(p[0, [R.BEACH, R.ROCK, R.MANGROVE]], R.SHALLOW_Y)  # the shallows' band, at sea
    assert p[1, R.BEACH] == pytest.approx(R.SHALLOW_Y)  # the sand meets the water at its level
    assert p[2, R.BEACH] == pytest.approx(R.SHALLOW_Y + 10 * R.BEACH_SLOPE)  # and rises gently from it
    assert p[1, R.CLIFF] == pytest.approx(9.0)  # a cliff stands at its top at the water's edge
    assert R.profiles(np.array([-2.0]), h[:1], top[:1], np.full(1, 0.5))[0, R.CLIFF] < R.WATER_Y  # its foot under water
    assert np.allclose(p[3], 3.0)  # every type is the ground again inland


def test_shallows_fade_into_the_sea_colour():
    d = np.array([-100.0])
    c = R.colours(d, np.array([[0.3, 0.4, 0.2]]), np.array([0.5]))
    assert np.allclose(c[0], R.SEA_COLOUR)


def test_keep_leaves_the_ground_near_anything_built():
    k = R.keep(np.array([0.0, R.KEEP_NEAR, R.KEEP_FAR, 100.0]))
    assert k[0] == 0 and k[1] == 0 and k[2] == 1 and k[3] == 1


def test_coastal_only_near_the_sea():
    t = slope()
    assert R.coastal(t, int(1100 // CELL), 3)
    assert not R.coastal(t, 10, 3)


def test_field_draws_the_relief_away_from_the_shore_and_the_shallows_at_sea():
    s = shore.Shore(slope())
    f = s.field(5, 3)  # x 1000 to 1200: the water's edge at x = 1100
    t = np.arange(R.NODES) * R.STEP
    x, z = np.meshgrid(f.x0 + t, f.z0 + t)
    far = f.d > 40
    assert np.allclose(f.y[far], drawn(s.terrain, x[far], z[far]), atol=1e-4)
    assert (f.w[..., R.ROCK] > 0.99).all()  # nothing mapped and gentle ground: rocky shore
    assert np.allclose(f.y[(f.d < 0) & ~f.out], R.SHALLOW_Y)
    assert f.out[:, 0].all() and not f.out[:, -1].any()  # past the shallows the sea plane shows
    tris = f.triangles()
    assert len(tris) and not f.out.ravel()[tris].all(axis=1).any()


def test_fields_of_neighbours_agree_along_their_edge():
    s = shore.Shore(slope(), {"beach": shapely.box(1000, 0, 1300, 3000)})
    a, b = s.field(5, 3), s.field(5, 4)
    assert np.allclose(a.y[-1], b.y[0]) and np.allclose(a.rgb[-1], b.rgb[0]) and np.allclose(a.n[-1], b.n[0])
    assert (a.w[np.abs(a.d) < 30, R.BEACH] > 0.99).all()


def test_a_field_matches_the_ten_metre_ground_where_its_neighbour_is_not_coastal():
    from contravia.mesh import MeshBuilder
    from contravia.rules import ground

    s = shore.Shore(slope())
    f = s.field(5, 3)
    mb = MeshBuilder()
    ground.ground(mb, 6, 3, s.terrain)  # x 1200 to 1400, the sea out of reach
    assert not s.coastal(6, 3)
    edge = np.asarray(mb._pos[0])[np.isclose(np.asarray(mb._pos[0])[:, 0], 1200.0)]
    assert np.allclose(np.interp(f.z0 + np.arange(R.NODES) * R.STEP, edge[:, 2], edge[:, 1]), f.y[:, -1], atol=1e-4)


def test_a_mapped_cliff_stands_over_the_water():
    s = shore.Shore(slope(), {"cliff": shapely.LineString([(1100, 0), (1100, 3000)])})
    f = s.field(5, 3)
    edge = (f.d > 0) & (f.d < 3)
    assert (f.y[edge] >= R.SHALLOW_Y + R.CLIFF_MIN - 0.1).all()


def test_mangroves_stand_in_the_shallows_and_rocks_on_the_rock_band_the_same_every_time():
    s = shore.Shore(slope(), {"mangrove": shapely.box(1000, 0, 1300, 3000)})
    f = s.field(5, 3)
    xz, y = R.mangroves(f)
    assert len(xz) > 50 and (y >= 0.1 - 1e-9).all()
    d = f.sample(f.d, xz[:, 0], xz[:, 1])
    assert (d < 0).any() and (d < 10).all()
    assert np.array_equal(xz, R.mangroves(shore.Shore(slope(), {"mangrove": shapely.box(1000, 0, 1300, 3000)})
                                          .field(5, 3))[0])
    rocks, _ = R.rocks(shore.Shore(slope()).field(5, 3))
    assert len(rocks) > 20 and len(R.mangroves(shore.Shore(slope()).field(5, 3))[0]) == 0


def test_the_shore_height_is_the_field_near_the_sea_and_the_terrain_away():
    s = shore.Shore(slope())
    f = s.field(5, 3)
    x, z = np.array([1100.0, 1700.0]), np.array([700.0, 700.0])
    h = s.height(x, z)
    assert h[0] == pytest.approx(float(f.height(x[:1], z[:1])[0]))
    assert h[1] == pytest.approx(float(s.terrain.height(x[1:], z[1:])[0]))


def test_the_ground_grid_of_a_coastal_cell_is_the_field():
    from contravia.mesh import MeshBuilder
    from contravia.rules import ground

    s = shore.Shore(slope())
    mb = MeshBuilder()
    ground.ground(mb, 5, 3, s.terrain, shore=s)
    assert len(mb._pos[0]) == R.NODES * R.NODES


def test_no_tree_mix_draws_the_mangle():
    from contravia.rules import trees
    from contravia.rules.species import SPECIES

    seeds = trees.point_seeds(np.random.default_rng(0).random((5000, 2)) * 1000)
    for zone in trees.MIX:
        assert SPECIES.index("mangle") not in trees.pick(np.full(len(seeds), zone), seeds)


def test_of_is_none_for_a_city_that_is_not_an_island():
    class Flat:
        folder = None

    assert shore.of(Flat()) is None
