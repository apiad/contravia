"""What the map's panel names under and near its crosshair at each height (contravia/nearby.py)."""

import pytest

from contravia import collect, nearby
from tests.test_collect import at


def place(x, z, pid, total):
    rec = at(x, z, pid)
    rec["scores"] = {**rec["scores"], "total": total}
    return rec


SPOT = nearby.Spot(province="La Habana", municipio="Centro Habana", street="Galiano",
                   line="entre Neptuno y Concordia", barrio="Cayo Hueso")


@pytest.mark.parametrize("h, name", [(1_200_000, "isla"), (100_000, "isla"), (99_000, "ciudad"),
                                     (13_000, "ciudad"), (5_000, "barrio"), (1_600, "barrio"), (150, "calle")])
def test_the_bands_follow_the_maps_stages(h, name):
    assert nearby.band(h).name == name


def test_each_band_names_more_detail_as_the_camera_comes_down():
    ps = collect.Places([place(0, 0, "a", 5.0)])
    assert nearby.choose(ps, SPOT, 0, 0, 500_000).title == "La Habana"
    assert nearby.choose(ps, SPOT, 0, 0, 500_000).line == ""
    city = nearby.choose(ps, SPOT, 0, 0, 50_000)
    assert (city.title, city.line) == ("La Habana", "Centro Habana")
    barrio = nearby.choose(ps, SPOT, 0, 0, 5_000)
    assert barrio.title == "Centro Habana" and barrio.line == "Cayo Hueso · La Habana"
    street = nearby.choose(ps, SPOT, 0, 0, 400)
    assert street.title == "Galiano" and street.line == "entre Neptuno y Concordia · Cayo Hueso · Centro Habana"


def test_at_sea_or_off_the_streets_it_says_what_it_knows():
    ps = collect.Places([])
    assert nearby.choose(ps, nearby.Spot(), 0, 0, 500_000).title == "Cuba"
    assert nearby.choose(ps, nearby.Spot(), 0, 0, 5_000).title == "En el mar"
    rural = nearby.choose(ps, nearby.Spot(province="Pinar del Río", municipio="Viñales"), 0, 0, 400)
    assert (rural.title, rural.line, rural.header, rural.items) == ("Viñales", "Pinar del Río", "", ())


def test_from_high_up_the_most_important_places_on_screen_come_first_and_at_most_25():
    recs = [place(k * 1000.0, 0, f"p{k:02d}", float(k % 15)) for k in range(40)]
    here = nearby.choose(collect.Places(recs), SPOT, 0, 0, 500_000)
    assert len(here.items) == 25 and here.header == "LO MÁS NOTABLE"
    scores = [int(i.id[1:]) % 15 for i in here.items]
    assert scores == sorted(scores, reverse=True) and scores[0] == 14
    assert all(i.distance == "" for i in here.items)


def test_close_up_the_nearest_places_come_first_whatever_their_importance():
    ps = collect.Places([place(200, 0, "great", 15.0), place(40, 0, "small", 2.0), place(5_000, 0, "far", 15.0)])
    here = nearby.choose(ps, SPOT, 0, 0, 300)
    assert [i.id for i in here.items] == ["small", "great"] and here.header == "AQUÍ"
    assert [i.distance for i in here.items] == ["40 m", "200 m"]


def test_at_the_barrio_stage_importance_weighs_against_distance():
    ps = collect.Places([place(100, 0, "small", 3.0), place(1_500, 0, "great", 15.0), place(9_000, 0, "out", 15.0)])
    here = nearby.choose(ps, SPOT, 0, 0, 4_000)  # reach 2 km
    assert [i.id for i in here.items] == ["great", "small"] and here.items[0].distance == "1,5 km"


@pytest.mark.parametrize("m, text", [(3, "10 m"), (347, "350 m"), (994, "990 m"), (1_234, "1,2 km"),
                                     (9_960, "10 km"), (42_400, "42 km")])
def test_distances_read_in_spanish(m, text):
    assert nearby.distance(m) == text


def test_the_panel_is_chosen_again_only_when_the_view_changed_enough():
    was = (0.0, 0.0, 10_000.0)
    assert nearby.stale(None, 0, 0, 10_000)
    assert not nearby.stale(was, 200, 0, 10_000)
    assert nearby.stale(was, 400, 0, 10_000)
    assert not nearby.stale(was, 0, 0, 10_400)
    assert nearby.stale(was, 0, 0, 10_600)
    assert nearby.stale((0.0, 0.0, 13_200.0), 0, 0, 12_900)  # a small zoom that crosses a band
