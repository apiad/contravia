"""The map's city emblems: which towns show at each height, a layout with no overlaps, and SVG that draws."""

import pytest

from contravia import emblems, mapview, towns

VIEWS = {  # the mock's four views: (x, z, height)
    "isla": (285123.0, 134278.0, 1_000_000.0),
    "oriente": (600268.0, 269654.0, 350_000.0),
    "centro": (284537.0, 101068.0, 150_000.0),
    "santiago": (684153.0, 329509.0, 40_000.0),
}


@pytest.fixture(scope="module")
def every():
    return towns.load()


def test_from_the_island_only_the_big_cities_show(every):
    names = {t.name for t in emblems.shown(every, 1_000_000.0)}
    assert {"La Habana", "Santiago de Cuba", "Camagüey", "Holguín", "Santa Clara"} <= names
    assert all(emblems.has_medallion(t) and t.pop >= 140_000 for t in emblems.shown(every, 1_000_000.0))


def test_lower_down_every_capital_and_more_towns_show(every):
    at350 = emblems.shown(every, 350_000.0)
    assert {t.name for t in every if t.capital} <= {t.name for t in at350}
    assert any(not emblems.has_medallion(t) for t in at350)
    assert len(emblems.shown(every, 150_000.0)) > len(at350) and len(emblems.shown(every, 40_000.0)) == len(every)


@pytest.mark.parametrize("view", VIEWS)
def test_layout_never_overlaps_and_stays_on_screen(every, view):
    x, z, h = VIEWS[view]
    out = emblems.layout(every, mapview.MapCam(x, z, h, 0.0), 1280, 720)
    assert out
    boxes = []
    for e in out:
        boxes.append((e.x - e.r, e.y - e.r, e.x + e.r, e.y + e.r))
        if e.label:
            lx, ly, w, h_ = e.label
            boxes.append((lx, ly, lx + w, ly + h_))
    for k, a in enumerate(boxes):
        assert 0 <= a[0] and a[2] <= 1280 and 0 <= a[1] and a[3] <= 720
        for b in boxes[k + 1:]:
            assert not (a[0] < b[2] - 1e-6 and a[2] > b[0] + 1e-6 and a[1] < b[3] - 1e-6 and a[3] > b[1] + 1e-6)
    assert all(e.label or e.medallion for e in out)  # a town that lost its name lost its dot


def test_close_to_the_ground_the_emblems_are_gone(every):
    assert emblems.layout(every, mapview.MapCam(684153.0, 329509.0, 5_000.0, 0.0), 1280, 720) == []
    assert emblems.opacity(11_000.0) == pytest.approx(0.5)


def test_a_medallions_colour_never_changes_and_habana_is_yellow():
    assert emblems.accent("Bayamo") == emblems.accent("Bayamo") and emblems.accent("La Habana") == "#ffd400"


def test_the_svgs_draw():
    from manifoldx import svg

    for f in (emblems.CITY_FONT, emblems.TOWN_FONT):
        svg.load_font(f)
    medal = svg.rasterize_svg(emblems.medallion_svg("La Habana"), 100, 100)
    assert medal[..., 3].mean() > 100
    label = svg.rasterize_svg(emblems.label_svg("Santiago de Cuba", True, 20.0, 200, 32), 200, 32)
    assert (label[..., 3] > 0).mean() > 0.1
