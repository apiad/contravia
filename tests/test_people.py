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
