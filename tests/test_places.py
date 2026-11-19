"""The list of collectible places shipped with the game, checked as data."""

import json
from pathlib import Path

import pytest

from contravia import geo

PLACES = json.loads((Path(geo.__file__).parent / "places.json").read_text(encoding="utf-8"))
TIERS = {"débil", "medio", "fuerte", "imprescindible"}
PROVINCES = {name for name, _ in geo.PROVINCES}


def test_ids_are_unique():
    ids = [p["id"] for p in PLACES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("key", ["title", "type", "text", "photo_target", "municipio"])
def test_every_place_has_its_words(key):
    assert [p["id"] for p in PLACES if not p.get(key)] == []


def test_every_box_has_positive_sides_and_a_yaw_in_half_a_turn():
    for p in PLACES:
        b = p["box"]
        assert b["w"] > 0 and b["d"] > 0 and b["h"] > 0, p["id"]
        assert 0 <= b["yaw_deg"] < 180, p["id"]


def test_every_centre_is_in_cuba_and_names_a_province():
    south, west, north, east = geo.CUBA_BOX
    for p in PLACES:
        assert south < p["lat"] < north and west < p["lon"] < east, p["id"]
        assert p["province"] in PROVINCES, p["id"]


def test_scores_are_in_range():
    for p in PLACES:
        s = p["scores"]
        assert all(0 <= s[k] <= 5 for k in ("h", "c", "v")), p["id"]
        assert s["total"] == pytest.approx(s["h"] + s["c"] + s["v"], abs=0.11), p["id"]
        assert s["tier"] in TIERS, p["id"]
