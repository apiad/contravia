from types import SimpleNamespace

import numpy as np
import shapely

from contravia.grass import FAR, MOVE, NEAR, ROAD_CLEAR, STEP, Grass, Lawn, patch, spots
from contravia.rules.buildings import filler_rings


def _square(x0, z0, x1, z1, y=None):
    """Two triangles over the rectangle, (2, 3, 2), or (2, 3, 3) at height y."""
    t = np.array([[(x0, z0), (x1, z0), (x1, z1)], [(x0, z0), (x1, z1), (x0, z1)]], float)
    return t if y is None else np.insert(t, 1, y, axis=2)


def _cell(**kw):
    """A cell holding nothing but what kw gives it."""
    empty = {k: None for k in ("green_tris", "street_tris", "walk_tris", "park_tris", "struct_tris", "street_cuts")}
    empty |= {"roads": np.zeros((0, 5)), "buildings": [], "fillers": [], "filler_seeds": np.zeros(0, np.int64)}
    return SimpleNamespace(**(empty | kw))


LOT_BLOCK = np.array([(120, 210), (195, 210), (195, 280), (120, 280)], float)  # a block of filler houses in (0, 1)


class _City:
    """A city whose ground rises 0.1 m a metre eastward, from the sea at x 0. Cell (0, 0) holds 100 x 100 m of green,
    a street across it at z 50 and a park path at x 20. One province's arrays hold a laid park from x 220 to 420
    (across cells (1, 0) and (2, 0)), its kerb paved along x 220 to 223. Cell (0, 1) is a bare lot with a house at
    x 20 to 40, z 220 to 240, a block of filler houses, and an 8 m road along z 300."""

    def __init__(self, parks=True):
        self.terrain = SimpleNamespace(height=lambda x, z: 0.1 * np.asarray(x, float), heights=np.array([0.0, 20.0]))
        self.green = _cell(green_tris=_square(0, 0, 100, 100), street_tris=_square(0, 46, 100, 54, 0.0),
                           park_tris=_square(18, 0, 22, 100, 0.0))
        self.empty = _cell()
        self.kerb = _cell(walk_tris=_square(220, 20, 223, 100, 0.15))
        self.lot = _cell(buildings=[np.array([(20, 220), (40, 220), (40, 240), (20, 240)], float)],
                         fillers=[LOT_BLOCK], filler_seeds=np.array([11]), roads=np.array([(0, 300, 200, 300, 8.0)]))
        ring = np.array([(220, 20), (420, 20), (420, 100), (220, 100), (220, 20)], float)
        self.a = {"park_verts": ring, "park_starts": np.array([0, len(ring)]), "park_kind": np.array([3]),
                  "park_seed": np.array([7])} if parks else {}

    def cell(self, i, j):
        return {(0, 0): self.green, (1, 0): self.kerb, (0, 1): self.lot}.get((i, j), self.empty)

    def provinces_of(self, i, j):
        return [0]

    def province(self, k):
        return SimpleNamespace(a=self.a)


def test_a_patch_is_a_square_of_blades_drawn_on_both_sides():
    g = patch(10, seed=1)
    assert len(g["positions"]) == 10 * 5 and len(g["indices"]) == 2 * 10 * 3 * 3
    assert np.abs(g["positions"][:, [0, 2]]).max() <= STEP / 2 + 0.2
    assert np.allclose(g["normals"], [0, 1, 0])
    assert np.array_equal(g["positions"], patch(10, seed=1)["positions"])


def test_patches_stand_on_the_green_off_the_streets_and_park_motifs():
    city = _City()
    xz, bare = spots(city, 0, 0)
    p = xz[~bare]
    assert len(p) > 1000
    assert ((p >= 0) & (p <= 100)).all()
    assert not ((xz[:, 0] <= 100) & (xz[:, 1] > 45) & (xz[:, 1] < 55)).any()  # the street, and a metre round it
    assert not ((xz[:, 0] > 17) & (xz[:, 0] < 23) & (xz[:, 1] <= 100)).any()  # the park's path
    assert (xz[:, 0] >= 3).all()  # the sea, where the ground lies under SEA
    assert np.allclose(xz % STEP, 0) and len(np.unique(xz, axis=0)) == len(xz)
    assert ((xz >= 0) & (xz < 200)).all()  # in the cell, its far edges left to the next cells
    assert all(np.array_equal(u, v) for u, v in zip((xz, bare), spots(city, 0, 0), strict=True))  # same cell, same grass


def test_every_bare_lot_grows_tufts_off_houses_roads_and_landmarks():
    """Open land that is neither OSM green nor a laid park grows grass too, in tufts, and never on the houses, the
    filler houses, a road with its margin or a landmark's ground."""
    city = _City()
    xz, bare = spots(city, 0, 1)
    assert len(xz) > 2000 and bare.all()
    x, z = xz[:, 0], xz[:, 1]
    assert not ((x > 19) & (x < 41) & (z > 219) & (z < 241)).any()  # the house, and a metre round it
    for r in filler_rings(LOT_BLOCK, 11):
        assert not shapely.contains_xy(shapely.Polygon(r).buffer(0.9), x, z).any()  # each filler house
    assert not (np.abs(z - 300) < 4 + ROAD_CLEAR).any()  # the road's 8 m and its margin
    open_ = (x >= 50) & (x < 110) & (z >= 330) & (z < 390)  # 900 grid spots with nothing on them
    assert 0.25 < open_.sum() / 900 < 0.85  # tufts, not a lawn
    plaza = SimpleNamespace(clear=[np.array([(60, 320), (100, 320), (100, 360), (60, 360)], float)], wall_rings=[])
    q = spots(city, 0, 1, plaza)[0]
    assert len(q) < len(xz) and not ((q[:, 0] > 59) & (q[:, 0] < 101) & (q[:, 1] > 319) & (q[:, 1] < 361)).any()


def test_the_lawns_of_laid_parks_grow_grass_too():
    """A park contravia.parks lays out has no OSM green under it: its lawn is the park less its kerb and paths."""
    city = _City()
    xz, bare = spots(city, 1, 0)
    p = xz[~bare]
    assert len(p) > 1000
    assert ((p[:, 0] >= 220) & (p[:, 0] < 400) & (p[:, 1] >= 20) & (p[:, 1] <= 100)).all()  # in the park, in the cell
    assert not ((xz[:, 0] > 219) & (xz[:, 0] < 224) & (xz[:, 1] > 19) & (xz[:, 1] < 101)).any()  # off the kerb
    xz2, bare2 = spots(city, 2, 0)
    q = xz2[~bare2]
    assert len(q) and ((q[:, 0] >= 400) & (q[:, 0] <= 420)).all()  # the rest of the park, in the next cell
    assert np.array_equal(xz, spots(city, 1, 0)[0])
    assert spots(_City(parks=False), 1, 0)[1].all()  # without the park, the same ground is a bare lot


def test_grass_is_instances_that_change_level_as_the_camera_moves():
    import manifoldx as mx

    engine = mx.Engine("t", width=64, height=64)
    grass = Grass(engine, _City())
    before = int(engine.store._alive.sum())
    for _ in range(200):
        grass.update({(0, 0)}, (10.0, 2.0, 10.0), (10.0, 10.0))
        if grass.lawns:
            break
        grass._pending[(0, 0)].result()
    lawn = grass.lawns[(0, 0)]
    d = np.hypot(lawn.xz[:, 0] - 10, lawn.xz[:, 1] - 10)
    assert (lawn.level[d < NEAR] == 0).all() and (lawn.level[(d >= NEAR) & (d < FAR)] == 1).all()
    assert (lawn.level[d >= FAR] == -1).all() and (lawn.ents[d >= FAR] == -1).all()
    assert np.allclose(lawn.y, 0.1 * lawn.xz[:, 0] + 0.02)  # on the ground
    counts = grass.counts()
    assert int(engine.store._alive.sum()) - before == counts["dense"] + counts["sparse"]
    assert engine.benders[0][:2] == (10.0, 10.0)
    kept = lawn.ents.copy()
    grass.update({(0, 0)}, (10.0 + MOVE / 2, 2.0, 10.0))  # a short move checks nothing
    assert np.array_equal(lawn.ents, kept)
    grass.update({(0, 0)}, (20.0, 2.0, 10.0))  # the camera drives on: only patches that change level respawn
    d = np.hypot(lawn.xz[:, 0] - 20, lawn.xz[:, 1] - 10)
    assert (lawn.level[d < NEAR] == 0).all() and (lawn.level[d >= FAR] == -1).all()
    same = (lawn.level == 0) & (np.hypot(lawn.xz[:, 0] - 10, lawn.xz[:, 1] - 10) < NEAR)
    assert same.any() and np.array_equal(lawn.ents[same], kept[same])
    grass.update(set(), (90.0, 2.0, 90.0))  # the streamer drops the cell
    engine.commands.execute(engine.store)  # destroys apply at the frame's end
    assert not grass.lawns and int(engine.store._alive.sum()) == before
    grass.close()


def test_a_bare_lot_grows_shorter_drier_grass_than_a_lawn():
    xz = np.array([(2.0 * k, 0.0) for k in range(50)])
    bare = np.arange(50) % 2 == 1
    flat = SimpleNamespace(height=lambda x, z: np.zeros(len(x)))
    lawn = Lawn(xz, flat, bare)
    assert lawn.height[bare].max() < lawn.height[~bare].min()
    assert (lawn.tint[bare, 2] < 0.7 * lawn.tint[bare, 0]).all()  # straw: red over blue
    assert np.array_equal(Lawn(xz, flat).height[~bare], lawn.height[~bare])  # a lawn is as it was
