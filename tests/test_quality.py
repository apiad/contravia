"""The quality presets: Alto is today's game, the presets climb, and Personalizado only takes values of a ladder."""

import pytest

from contravia import atmosphere, forest, game, grass, graveyard, hud, rain
from contravia.quality import KNOBS, PRESETS, Quality, ladder
from contravia.rules import foliage


def test_alto_is_todays_game():
    q = PRESETS["alto"]
    assert q == Quality()
    assert q.tau == game.TAU and q.far == game.FAR and q.fog_clear == atmosphere.FOG_CLEAR
    assert q.tree_lod == forest.LOD_AT and q.foliage_near == foliage.NEAR and q.grave_lod == graveyard.LOD_AT
    assert q.grass == (grass.NEAR, grass.FAR, *grass.BLADES)
    assert q.shadow == (200.0, atmosphere.SHADOW_RES, 1, 3)  # cascades, since manifoldx has them
    assert q.rain_pool == rain.POOL and q.minimap == (hud.MINIMAP_SIZE, hud.MINIMAP_EVERY)
    assert q.target_ms == game.TARGET_MS and q.detail_radius == 1 and q.traffic_radius == 400.0


def test_presets_climb():
    b, m, a, u = (PRESETS[k] for k in ("bajo", "medio", "alto", "ultra"))
    assert b.tau[0] > m.tau[0] > a.tau[0] > u.tau[0]
    assert b.far < m.far < a.far
    assert b.foliage_near == 0 and not b.detail_noise and b.detail_radius == 0
    assert PRESETS["cine"].tune is False and PRESETS["cine"].tau == (5.0, 5.0)


def test_every_knob_has_a_label_and_a_ladder():
    assert set(KNOBS) == {f for f in Quality.__dataclass_fields__ if f not in ("tune", "target_ms")}
    assert ladder("rain_pool") == [1000, 2000, 3000, 6000, 10000]


def test_changed_refuses_unknown_knobs_and_values():
    with pytest.raises(ValueError, match="nope"):
        Quality().changed(nope=1)
    with pytest.raises(ValueError, match="rain_pool"):
        Quality().changed(rain_pool=1234)
    assert Quality().changed(rain_pool=1000).diff(Quality()) == {"rain_pool": 1000}


def test_play_takes_a_quality_for_the_run(monkeypatch):
    import contravia.cli
    import contravia.game

    seen = {}
    monkeypatch.setattr(contravia.game, "main", lambda *a, **k: seen.update(k))
    contravia.cli.main(["play", "--data", "data/cuba"])
    assert seen["quality"] is None
    contravia.cli.main(["play", "--data", "data/cuba", "--quality", "cine"])
    assert seen["quality"] == "cine"
    with pytest.raises(SystemExit):
        contravia.cli.main(["play", "--data", "data/cuba", "--quality", "maximo"])


def test_changed_keeps_the_ladders_own_values():
    q = Quality().changed(grass=[30.0, 70.0, 90.0, 24.0], minimap=[384.0, 2.0], detail_radius=2.0)
    assert q.grass == (30.0, 70.0, 90, 24) and isinstance(q.grass[2], int)
    assert q.minimap == (384, 2) and isinstance(q.minimap[0], int) and type(q.detail_radius) is int


def test_apply_builds_the_detail_ring_one_past_the_drawn_one():
    from types import SimpleNamespace as NS

    from contravia.quality import PRESETS, apply

    part = NS(configure=lambda q: None)
    game = NS(streamer=NS(tau=12.0, material=NS(detail=True)), engine=NS(camera=NS(far=0.0), state=NS()),
              detail=NS(ready_radius=2), layer=None, forest=part, graveyard=part, grass=part, rain=part,
              atmosphere=part, hud=NS(configure_quality=lambda q: None))
    apply(game, PRESETS["ultra"], "ultra")
    assert game.detail_radius == 2 and game.detail.ready_radius == 3
    apply(game, PRESETS["bajo"], "bajo")
    assert game.detail.ready_radius == 2


def test_the_presets_set_the_screen_effects():
    from types import SimpleNamespace as NS

    from contravia.quality import PRESETS, set_post

    calls = []
    engine = NS(set_post=lambda **k: calls.append(k))
    set_post(engine, PRESETS["bajo"])
    set_post(engine, PRESETS["alto"])
    set_post(engine, PRESETS["ultra"])
    bajo, alto, ultra = calls
    assert not bajo["fxaa"] and bajo["ssao"] is bajo["bloom"] is bajo["shafts"] is bajo["grade"] is None
    assert alto["fxaa"] and alto["ssao"].radius == 3.0 and alto["bloom"].intensity == 0.5 and alto["shafts"] is None
    assert ultra["shafts"].intensity == 0.6 and ultra["ssao"].radius == 4.0
    set_post(NS(), PRESETS["alto"])  # a manifoldx without set_post: nothing to do
