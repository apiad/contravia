from types import SimpleNamespace

import numpy as np

from contravia.grass import FAR, MOVE, NEAR, STEP, Grass, patch, spots


def _square(x0, z0, x1, z1, y=None):
    """Two triangles over the rectangle, (2, 3, 2), or (2, 3, 3) at height y."""
    t = np.array([[(x0, z0), (x1, z0), (x1, z1)], [(x0, z0), (x1, z1), (x0, z1)]], float)
    return t if y is None else np.insert(t, 1, y, axis=2)


class _City:
    """A city of one cell (0, 0) holding 100 x 100 m of green, a street across it at z 50 and a park path at x 20,
    and one province whose arrays hold a laid park from x 220 to 420 (across cells (1, 0) and (2, 0)), its kerb
    paved along x 220 to 223."""

    def __init__(self, parks=True):
        self.terrain = SimpleNamespace(height=lambda x, z: 0.1 * np.asarray(x, float))
        self.green = SimpleNamespace(green_tris=_square(0, 0, 100, 100), street_tris=_square(0, 46, 100, 54, 0.0),
                                     walk_tris=None, park_tris=_square(18, 0, 22, 100, 0.0),
                                     roads=np.zeros((0, 5)))
        self.empty = SimpleNamespace(green_tris=None, street_tris=None, walk_tris=None, park_tris=None,
                                     roads=np.zeros((0, 5)))
        self.kerb = SimpleNamespace(green_tris=None, street_tris=None, walk_tris=_square(220, 20, 223, 100, 0.15),
                                    park_tris=None, roads=np.zeros((0, 5)))
        ring = np.array([(220, 20), (420, 20), (420, 100), (220, 100), (220, 20)], float)
        self.a = {"park_verts": ring, "park_starts": np.array([0, len(ring)]), "park_kind": np.array([3]),
                  "park_seed": np.array([7])} if parks else {}

    def cell(self, i, j):
        return self.green if (i, j) == (0, 0) else self.kerb if (i, j) == (1, 0) else self.empty

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
    p = spots(city, 0, 0)
    assert len(p) > 1000
    assert ((p >= 0) & (p <= 100)).all()
    assert not ((p[:, 1] > 45) & (p[:, 1] < 55)).any()  # the street, and a metre round it
    assert not ((p[:, 0] > 17) & (p[:, 0] < 23)).any()  # the park's path
    assert np.allclose(p % STEP, 0)
    assert np.array_equal(p, spots(city, 0, 0))  # same cell, same grass
    assert len(spots(city, 0, 1)) == 0


def test_the_lawns_of_laid_parks_grow_grass_too():
    """A park contravia.parks lays out has no OSM green under it: its lawn is the park less its kerb and paths."""
    city = _City()
    p = spots(city, 1, 0)
    assert len(p) > 1000
    assert ((p[:, 0] >= 220) & (p[:, 0] < 400) & (p[:, 1] >= 20) & (p[:, 1] <= 100)).all()  # in the park, in the cell
    assert not ((p[:, 0] > 219) & (p[:, 0] < 224)).any()  # off the kerb, and a metre round it
    q = spots(city, 2, 0)
    assert len(q) and ((q[:, 0] >= 400) & (q[:, 0] <= 420)).all()  # the rest of the park, in the next cell
    assert np.array_equal(p, spots(city, 1, 0))
    assert len(spots(_City(parks=False), 1, 0)) == 0


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
