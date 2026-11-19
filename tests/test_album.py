"""The player's collection of photographed places, kept in the configuration folder."""

import json

import pytest

from contravia import album


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    return tmp_path


KNOWN = {"L0333": "La Habana", "L0330": "La Habana", "M100": "Matanzas"}


def test_an_empty_collection_has_nothing(config):
    a = album.Album.load(KNOWN)
    assert not a.has("L0333")
    assert a.count("La Habana") == 0


def test_a_photo_added_is_read_back_by_another_load(config):
    album.Album.load(KNOWN).add("L0333", "mar 18:40", "despejado")
    b = album.Album.load(KNOWN)
    assert b.has("L0333") and not b.has("L0330")
    assert b.count("La Habana") == 1 and b.count("Matanzas") == 0
    e = b.entry("L0333")
    assert e["game_time"] == "mar 18:40" and e["weather"] == "despejado" and e["photo"] == "L0333.png"
    assert e["date"]  # the real date it was taken
    assert b.photo_path("L0333") == config / "fotos" / "L0333.png"


def test_an_id_no_longer_in_the_list_is_ignored(config):
    (config / "coleccion.json").write_text(json.dumps({"GONE": {"photo": "GONE.png"}, "M100": {"photo": "M100.png"}}))
    a = album.Album.load(KNOWN)
    assert not a.has("GONE") and a.has("M100")


def test_a_broken_file_reads_as_empty_and_is_kept_until_the_next_add(config):
    path = config / "coleccion.json"
    path.write_text("{not json")
    a = album.Album.load(KNOWN)
    assert a.count("La Habana") == 0
    assert path.read_text() == "{not json"
    a.add("L0330", "lun 10:00", "lluvia")
    assert json.loads(path.read_text())["L0330"]["weather"] == "lluvia"


def test_tally_counts_what_the_player_has_against_the_total_largest_first():
    from types import SimpleNamespace as P

    from contravia.album import tally

    ps = [P(id="a", province="Matanzas"), P(id="b", province="La Habana"), P(id="c", province="La Habana")]
    assert tally(ps, lambda pid: pid in ("a", "c"), lambda p: p.province) == [("La Habana", 1, 2), ("Matanzas", 1, 1)]


def test_grid_step_moves_inside_the_grid_and_stays_at_its_edges():
    from contravia.album import grid_step

    assert grid_step(0, "right", 4, 10) == 1 and grid_step(0, "down", 4, 10) == 4
    assert grid_step(3, "right", 4, 10) == 3  # the row's last cell
    assert grid_step(4, "left", 4, 10) == 4  # the row's first cell
    assert grid_step(0, "up", 4, 10) == 0 and grid_step(8, "down", 4, 10) == 8 and grid_step(9, "right", 4, 10) == 9
    assert grid_step(5, "down", 4, 10) == 9
