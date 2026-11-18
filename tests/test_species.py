from datetime import date

import numpy as np
import pytest

from contravia.rules.species import ACCENT, CROWNS, FLOWER, GREEN, LEAF, SPECIES, season, template


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("crown", CROWNS)
def test_a_template_is_small_closed_over_the_ground_and_stands_on_the_origin(species, crown):
    t = template(species, crown)
    assert 0 < len(t.triangles) <= 90
    assert np.allclose(np.linalg.norm(t.normals, axis=1), 1.0)
    assert t.positions[:, 1].min() >= -0.01
    base = t.positions[t.positions[:, 1] < 0.01]
    assert np.allclose(base[:, [0, 2]].mean(axis=0), 0.0, atol=0.05)
    assert len(t.role) == len(t.rgb) == len(t.shade) == len(t.positions)


def test_full_crowns_average_at_most_75_triangles_and_thinner_crowns_draw_less():
    assert np.mean([len(template(s).triangles) for s in SPECIES]) <= 75
    for s in SPECIES:
        n = [len(template(s, c).triangles) for c in CROWNS]
        assert n[0] >= n[1] >= n[2]


def test_crowns_are_darker_underneath():
    t = template("laurel")
    leaf = t.role == LEAF
    assert t.shade[leaf & (t.normals[:, 1] < -0.5)].mean() < t.shade[leaf & (t.normals[:, 1] > 0.5)].mean()


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


def test_accent_parts_exist_where_a_season_paints_them():
    for s in ("flamboyan", "almendro"):
        assert (template(s).role == ACCENT).any()
