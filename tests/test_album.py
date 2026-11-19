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
