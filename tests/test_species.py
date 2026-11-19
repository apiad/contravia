from datetime import date

import numpy as np
import pytest

from contravia.rules.species import FLOWER, GREEN, SPECIES, season


def test_the_flamboyan_flowers_in_summer_turns_orange_late_in_the_year_and_thins_in_the_dry_season():
    assert season("flamboyan", date(2026, 6, 15)).accent == FLOWER
    nov = season("flamboyan", date(2026, 11, 15))
    assert nov.crown == "full" and nov.leaf[0] > nov.leaf[1] * 0.8  # warm, not green
    assert season("flamboyan", date(2026, 3, 15)).crown == "sparse"
    assert season("flamboyan", date(2026, 9, 15)).leaf == GREEN["flamboyan"]


def test_the_ceiba_is_bare_in_the_dry_season_and_the_almendro_reddens():
    assert season("ceiba", date(2026, 2, 15)).crown == "bare"
    assert season("ceiba", date(2026, 7, 15)).crown == "full"
    jan, jul = season("almendro", date(2026, 1, 15)), season("almendro", date(2026, 7, 15))
    assert jan.accent[0] > jul.accent[0] and jan.leaf[0] > jul.leaf[0]


def test_evergreens_stay_full_and_yellow_a_little_in_april():
    for s in ("laurel", "palma", "coco"):
        apr, jul = season(s, date(2026, 4, 10)), season(s, date(2026, 7, 10))
        assert apr.crown == jul.crown == "full"
        assert apr.leaf != jul.leaf and jul.leaf == GREEN[s]


def test_a_shift_moves_the_turn_by_days():
    assert season("ceiba", date(2026, 3, 25)).crown == "bare"
    assert season("ceiba", date(2026, 3, 25), shift_days=10).crown == "full"


class _Land:
    """Ground 5 m up everywhere, except sea where x < sea_x."""

    def __init__(self, sea_x=-1e9):
        self.sea_x = sea_x

    def height(self, x, z):
        x = np.asarray(x, float)
        return np.where(x < self.sea_x, -2.0, 5.0)


def _rect(x0, z0, x1, z1):
    return np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]], float)


def _parks(rings, kinds, seeds, trees):
    from contravia.parks import KINDS

    verts = np.vstack(rings)
    starts = np.cumsum([0] + [len(r) for r in rings])
    return {"park_verts": verts, "park_starts": starts, "park_kind": np.array([KINDS.index(k) for k in kinds]),
            "park_seed": np.array(seeds, np.int64), "trees": np.asarray(trees, np.float32)}


@pytest.mark.parametrize("zone", range(5))
def test_each_zone_draws_its_mix_within_three_points(zone):
    from contravia.rules.trees import MIX, pick

    rng = np.random.default_rng(zone)
    got = pick(np.full(10_000, zone), rng.integers(0, 2**63, 10_000).astype(np.uint64))
    for k, s in enumerate(SPECIES):
        want = MIX[zone].get(s, 0) / sum(MIX[zone].values())
        assert abs((got == k).mean() - want) <= 0.03, s


def test_the_same_spot_grows_the_same_tree():
    from contravia.rules.trees import OLD, grow

    xz = np.random.default_rng(1).uniform(0, 500, (200, 2))
    a, b = grow(xz, _Land(), OLD), grow(xz.copy(), _Land(), OLD)
    for f in ("species", "scale", "yaw", "shift"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    assert 0.8 <= a.scale.min() and a.scale.max() <= 1.25 and np.abs(a.shift).max() <= 14


def test_the_prado_is_laurel_and_a_paseo_row_is_one_species():
    from contravia.rules.trees import OUT, grow

    prado, paseo = _rect(0, 0, 20, 400), _rect(100, 0, 130, 400)
    a = _parks([prado, paseo], ["prado", "paseo"], [11, 12], np.zeros((0, 2)))
    t = grow(np.array([[10.0, z] for z in range(5, 400, 8)] + [[115.0, z] for z in range(5, 400, 8)]), _Land(), OUT, [a])
    n = len(range(5, 400, 8))
    assert (t.species[:n] == SPECIES.index("laurel")).all()
    assert len(set(t.species[n:].tolist())) == 1


def test_a_big_city_park_may_grow_one_ceiba_nearest_its_centre():
    from contravia.parks import CEIBA_CHANCE
    from contravia.rules.trees import OUT, grow, unit

    seed = next(s for s in range(1000) if unit(np.array([s], np.uint64), 7)[0] < CEIBA_CHANCE)
    pts = np.array([[60.0 + dx, 60.0 + dz] for dx in (-30, -2, 30) for dz in (-30, 3, 30)])
    a = _parks([_rect(0, 0, 120, 120)], ["city"], [seed], pts)
    t = grow(pts, _Land(), OUT, [a])
    ceiba = t.species == SPECIES.index("ceiba")
    assert ceiba.sum() == 1 and np.allclose(pts[ceiba][0], (58.0, 63.0))


def test_trees_near_the_sea_grow_coastal_species():
    from contravia.rules.trees import COAST, OUT, grow, pick

    xz = np.column_stack([np.full(50, 100.0), np.arange(50) * 9.0])
    t = grow(xz, _Land(sea_x=0.0), OUT)
    from contravia.rules.trees import point_seeds

    assert np.array_equal(t.species, pick(np.full(50, COAST), point_seeds(xz)))


def test_a_cell_takes_the_zone_of_its_commonest_municipio():
    from contravia.buildingtags import MUNI
    from contravia.citydata import Cell
    from contravia.rules.trees import OLD, OUT, VEDADO, cell_zone

    attrs = np.zeros((3, MUNI + 1))
    attrs[:, MUNI] = [1, 1, 0]
    assert cell_zone(Cell(building_attrs=attrs, muni_names=("Boyeros", "Playa"))) == VEDADO
    attrs[:, MUNI] = [0, 0, -1]
    assert cell_zone(Cell(building_attrs=attrs, muni_names=("Centro Habana",))) == OLD
    assert cell_zone(Cell()) == OUT


def test_ceibas_draw_fewer_triangles_in_the_dry_season_and_the_far_green_turns_with_them():
    from contravia.rules.treegrammar import triangles
    from contravia.rules.trees import far_green

    feb, jun = season("ceiba", date(2026, 2, 15)), season("ceiba", date(2026, 6, 15))
    for lod in range(3):
        assert triangles("ceiba", lod, feb.crown) < triangles("ceiba", lod, jun.crown)
    assert far_green(date(2026, 2, 15)) != far_green(date(2026, 6, 15))


def test_a_builder_process_draws_for_the_date_the_game_set(monkeypatch):
    import subprocess
    import sys
    from datetime import datetime

    from contravia.rules import trees

    monkeypatch.delenv("CONTRAVIA_DATE", raising=False)
    monkeypatch.setattr(trees, "_clock", None)
    trees.set_clock(lambda: datetime(2026, 6, 15, 10, 30))
    out = subprocess.run([sys.executable, "-c", "from contravia.rules.trees import today; print(today())"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "2026-06-15"
    monkeypatch.setattr(trees, "_clock", None)
