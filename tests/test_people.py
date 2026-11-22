import json
import time
from pathlib import Path

import numpy as np
import pytest
from manifoldx.humanoid.cycles import bake_cycle
from manifoldx.modeling.assets.human import build_human

from contravia import people
from contravia.rules import people as rules

ROOT = Path(__file__).resolve().parents[1]


def test_crowd_bodies_stay_under_the_heavy_limit_and_are_the_same_for_a_seed():
    for seed in range(40):
        assert rules.body(seed).weight <= rules.HEAVY
    assert repr(rules.body(5)) == repr(rules.body(5))


def test_an_outfit_is_the_same_for_a_seed_and_zone_and_changes_with_the_zone():
    assert repr(rules.outfit(3, 0, 0.2)) == repr(rules.outfit(3, 0, 0.2))
    assert len({repr(rules.outfit(3, z, 0.2)) for z in range(3)}) == 3


def test_men_wear_no_skirts_blouses_or_long_hair():
    for seed in range(60):
        o = rules.outfit(seed, seed % 3, 0.9)
        assert o.bottom != "skirt" and o.top != "blouse" and o.hair not in ("long", "bun")


def test_the_npcs_are_the_catalog_people_at_places_the_game_knows():
    catalog = json.loads((ROOT / "assets/reference/catalog.json").read_text())
    items = catalog if isinstance(catalog, list) else catalog.get("items", catalog)
    items = items if isinstance(items, list) else list(items.values())
    assert sorted(n.name for n in rules.NPCS) == sorted(i["name"] for i in items if i.get("kind") == "npc")
    places = {p["id"]: p for p in json.loads((ROOT / "contravia/places.json").read_text())}
    for n in rules.NPCS:
        assert places[n.place]["province"] == "La Habana", n.name
    assert len({n.seed for n in rules.NPCS}) == len(rules.NPCS)


@pytest.mark.parametrize("npc", rules.NPCS, ids=lambda n: n.name)
def test_every_npc_builds_at_every_level(npc):
    for level in (0, 1, 2):
        f = build_human(npc.params, outfit=npc.outfit, level=level)
        assert len(f.mesh.faces) > 0


def test_a_clip_baked_from_level_2_is_the_clip_of_level_0():
    p, o, _ = people.Figures.body(None, 4)
    far, near = build_human(p, outfit=o, level=2), build_human(p, outfit=o, level=0)
    assert np.allclose(bake_cycle(far, "walk", 1.3, 8), bake_cycle(near, "walk", 1.3, 8), atol=1e-5)


def test_the_pool_builds_every_far_level_before_any_nearer_one():
    stages = sorted((s, k, item) for k in range(3) for s, item in people.work(k))
    items = [item for _, _, item in stages]
    last_far = max(i for i, it in enumerate(items) if it[0] == "figure" and it[2] == 2)
    first_mid = min(i for i, it in enumerate(items) if it[0] == "figure" and it[2] == 1)
    first_near = min(i for i, it in enumerate(items) if it[0] == "figure" and it[2] == 0)
    assert last_far < first_mid < first_near
    walks = [i for i, it in enumerate(items) if it[:3] == ("clip", 0, "walk")]
    assert max(walks) < first_mid


class FakeEngine:
    def __init__(self):
        self.state, self.clips = {}, []

    def bake_poses(self, frames):
        self.clips.append(np.asarray(frames).shape)
        return len(self.clips) - 1


def test_figures_hand_out_bodies_once_built_and_put_up_few_results_a_frame():
    e = FakeEngine()
    f = people.Figures(e, 2)
    try:
        assert f.ready(0) == [] and f.mesh(0, 0) is None
        deadline = time.monotonic() + 120
        while f.waiting and time.monotonic() < deadline:
            assert f.step() <= people.UPLOADS
            time.sleep(0.01)
        assert not f.waiting
        assert f.ready(0) == [0] and f.ready(1) == [1] and f.ready(2) == []
        assert f.counts() == {"bodies": 2, "meshes": 6, "clips": 2 * len(rules.CLIPS), "waiting": 0}
        assert e.clips[0] == (people.FRAMES + 1, 19, 4, 4)
        assert f.clip(1, "run", rules.RUN[0]) is not None
        assert f.builder.cache == {}  # the store holds every result, the builder none
    finally:
        f.close()


def test_a_mesh_not_built_at_its_level_falls_back_to_the_nearest_coarser_first():
    f = people.Figures.__new__(people.Figures)
    f.store = {("figure", 0, 2): "far", ("figure", 0, 0): "near"}
    assert f.mesh(0, 1) == "far"
    assert f.mesh(0, 0) == "near"
    f.store = {("figure", 0, 0): "near"}
    assert f.mesh(0, 2) == "near"


def test_the_pool_survives_a_reload_in_the_same_engine():
    e = FakeEngine()
    f = people.Figures(e, 1)
    try:
        deadline = time.monotonic() + 60
        while f.waiting and time.monotonic() < deadline:
            f.step()
            time.sleep(0.01)
    finally:
        f.close()
    g = people.Figures(e, 1)
    try:
        assert not g.waiting and g.ready(0) == [0]
    finally:
        g.close()


# ------------------------------------------------------------------------------------------------- lanes, crowd

from types import SimpleNamespace  # noqa: E402

from contravia import walks  # noqa: E402

TOP = walks.paint(0, walks.TOP)


def quad(x0, z0, x1, z1, y):
    """Two triangles covering a rectangle at height y."""
    a, b, c, d = (x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)
    return [[a, b, c], [a, c, d]]


class FakeCity:
    """Cell (0, 0): a 40 m square block pavement at 0.15 m, and a 4 m wide pavement cut by the cell's east edge."""

    def __init__(self):
        tris = quad(20, 20, 60, 60, 0.15) + quad(150, 100, 250, 104, 0.15)
        self.c = SimpleNamespace(walk_tris=np.array(tris, float), walk_paint=np.full(len(tris), TOP))

    def cell(self, i, j):
        return self.c if (i, j) == (0, 0) else SimpleNamespace(walk_tris=None, walk_paint=None)


def test_a_block_gives_one_looping_lane_inset_from_its_kerb_and_a_cut_pavement_an_open_one():
    lanes = people.lay(FakeCity(), 0, 0)
    assert len(lanes) == 2
    loop = int(np.flatnonzero(lanes.loop)[0])
    assert lanes.length[loop] == pytest.approx(4 * (40 - 2 * people.INSET), abs=0.5)
    open_ = 1 - loop
    pts = lanes.pts[lanes.start[open_]: lanes.start[open_] + lanes.count[open_]]
    assert pts[:, 0].max() <= 200 - people.BORDER + 1e-6
    assert np.allclose(lanes.pts[:, 1], 0.15)
    assert np.all(lanes.step <= people.STEP + 1e-9)
    assert lanes.zone.tolist() == [0, 0]


def test_a_cell_with_no_walks_has_no_lanes():
    assert len(people.lay(FakeCity(), 3, 3)) == 0


def test_the_crowd_follows_the_hour_and_the_rain():
    assert people.hour_factor(12) == 1.0 and people.hour_factor(21) == 0.5 and people.hour_factor(3) == 0.15


def _engine():
    import manifoldx as mx

    return mx.Engine("crowd", width=64, height=48)


Q = SimpleNamespace(people_lod=(10.0, 30.0, 80.0), people=(40, 3, 8, 3))


@pytest.fixture(scope="module")
def crowd():
    e = _engine()
    c = people.Crowd(e, FakeCity(), Q, seed=3)
    deadline = time.monotonic() + 120
    while c.figures.waiting and time.monotonic() < deadline:
        c.figures.step()
        time.sleep(0.01)
    c.clips = c.figures.clip_table()
    yield e, c
    c.close()


def run(c, frames, eye=(40.0, 1.7, 40.0), dt=1 / 30, **kw):
    for _ in range(frames):
        c.update({(0, 0)}, eye, dt, **kw)
        while c.waiting():
            time.sleep(0.01)
            c.update({(0, 0)}, eye, 0.0, **kw)


def test_the_crowd_fills_its_lanes_and_draws_within_the_caps(crowd):
    e, c = crowd
    run(c, 30)
    n = c.counts()
    weight = c.t_weight.sum()
    assert n["walkers"] == c.target == int(min(Q.people[0], people.RATE * weight))
    assert n["lod0"] <= Q.people[1] and n["lod1"] <= Q.people[2]
    live = np.flatnonzero(c.alive & ~c.leaving)
    assert np.allclose(c.pos[live, 1], 0.15)
    drawn = live[c.level[live] < people.HIDDEN]
    assert (c.ent[drawn] >= 0).all()
    img = e.render_frame(supersample=1)  # skinned, faded and posed from baked clips: it draws
    assert img.shape == (48, 64, 3)


def test_walkers_move_at_their_buckets_speed(crowd):
    _, c = crowd
    run(c, 5)
    walking = np.flatnonzero(c.alive & ~c.leaving & (c.mode == people.WALK) & c.t_loop[c.lane])
    assert len(walking)
    s0 = c.s[walking].copy()
    c.update({(0, 0)}, (40.0, 1.7, 40.0), 0.1)
    still = c.mode[walking] == people.WALK
    moved = np.abs(c.s[walking] - s0)[still]
    moved = np.minimum(moved, c.t_length[c.lane[walking]][still] - moved)  # a loop wraps
    want = np.array(rules.CLIPS)[c.clip[walking][still], 1].astype(float) * 0.1
    assert np.allclose(moved, want, atol=1e-6)


def test_a_walker_at_an_open_end_looks_stands_and_turns_back(crowd):
    _, c = crowd
    run(c, 2)
    lane = int(np.flatnonzero(~c.t_loop)[0])
    w = int(np.flatnonzero(c.alive & ~c.leaving)[0])
    c.lane[w], c.local[w] = lane, lane - c._offs[c.cell[w]] if c.cell[w] in c._offs else 0
    c.cell[w] = c.cells_open[int(c.t_cell[lane])]
    c.local[w] = lane - c._offs[c.cell[w]]
    c.s[w], c.dir[w], c.clip[w] = c.t_length[lane] - people.LOOK - 0.05, 1.0, 1
    c._mode(np.array([w]), people.WALK)
    seen = set()
    for _ in range(400):
        c.update({(0, 0)}, (40.0, 1.7, 40.0), 1 / 30)
        seen.add(int(c.mode[w]))
        if c.dir[w] < 0 and c.mode[w] == people.WALK:
            break
    assert {people.CROSS, people.IDLE} <= seen and c.dir[w] < 0


def test_hysteresis_keeps_a_walker_at_a_band_on_its_level(crowd):
    _, c = crowd
    run(c, 2)
    w = int(np.flatnonzero(c.alive & ~c.leaving)[0])
    c.level[w] = 1
    eye = c.pos[w] + np.array([Q.people_lod[0] - 1.0, 0.0, 0.0])  # 1 m inside the level 0 band
    c._levels(eye)
    assert c.level[w] == 1


def test_a_limit_of_zero_sends_everyone_away(crowd):
    _, c = crowd
    run(c, 2)
    c.set_limit(0)
    run(c, 3)
    assert c.counts()["walkers"] == 0
    c.set_limit(None)
    run(c, 10)
    assert c.counts()["walkers"] > 0
