"""Biomes from the place and the hour, and their ambience."""

import numpy as np
import pytest
from euterpe.sound import render

from contravia.audio.ambience import Scatter, bed, oneshot
from contravia.audio.biome import BIOMES, Biome, Place, resolve


def fixed(values):
    return lambda name: values.get(name, 0.0)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


@pytest.mark.parametrize("place,want", [
    (Place(built=.8, sea_m=1000, zone=0, sun_alt=40), "city"),          # the Capitolio by day
    (Place(built=.8, sea_m=1000, zone=0, sun_alt=-20), "night"),        # the Capitolio at night
    (Place(built=.7, sea_m=300, zone=1, sun_alt=30), "coast"),          # the Malecón
    (Place(built=.02, sea_m=1000, zone=2, sun_alt=50), "countryside"),  # tobacco in Pinar del Río
    (Place(built=.3, sea_m=1000, zone=2, sun_alt=50), "barrio"),        # Guanabacoa
    (Place(built=.6, sea_m=1000, zone=2, sun_alt=50), "barrio"),        # built, but not a dense zone
    (Place(built=.02, sea_m=1000, zone=2, sun_alt=50, biome="coast"), "coast"),
    (Place(built=.02, sea_m=1000, zone=2, sun_alt=-30, biome="city"), "city"),
])
def test_resolve(place, want):
    assert resolve(place) == want


def test_a_biome_changes_only_after_ten_seconds():
    b = Biome()
    assert b.update(0, "city") == "city"
    assert b.update(5, "barrio") == "city" and b.update(14.5, "barrio") == "city"
    assert b.update(15.5, "barrio") == "barrio"  # ten seconds after it was first heard


def test_a_short_visit_does_not_change_it():
    b = Biome()
    b.update(0, "city")
    seen = {b.update(t, "barrio" if 20 <= t < 24 else "city") for t in np.arange(0, 60, 0.5)}
    assert seen == {"city"}


def test_regions_read_the_audio_group(tmp_path):
    from contravia.regions import Regions

    f = tmp_path / "r.toml"
    f.write_text('[[region]]\nname = "x"\nwhere = { municipio = "Playa" }\naudio.music = "changui"\n'
                          'audio.biome = "coast"\n')
    p = Regions.load(f).municipio("Playa")
    assert (p.audio_music, p.audio_biome) == ("changui", "coast")


@pytest.mark.parametrize("bad", ['audio.biome = "desert"', 'audio.music = "reggaeton"', 'audio.volume = 3'])
def test_regions_reject_a_bad_audio_key(bad, tmp_path):
    from contravia.regions import RegionError, Regions

    f = tmp_path / "r.toml"
    f.write_text(f'[[region]]\nname = "x"\nwhere = {{ municipio = "Playa" }}\n{bad}\n')
    with pytest.raises(RegionError):
        Regions.load(f)


def test_every_biome_has_a_bed():
    for b in BIOMES:
        x = render(bed(b, fixed({f"amb.{b}": 1.0})), 2.0)
        assert np.isfinite(x).all() and rms(x) > 1e-4, b


def test_a_bed_is_silent_at_level_zero():
    assert rms(render(bed("city", fixed({})), 0.5)) < 1e-5


def test_scatter_is_seeded_and_fits_the_place():
    sa, sb = Scatter(4), Scatter(4)
    a = [sa.due(t, "countryside", 6.0) for t in np.arange(0, 300, 0.5)]
    b = [sb.due(t, "countryside", 6.0) for t in np.arange(0, 300, 0.5)]
    names = {n for shots in a for n, _ in shots}
    assert a == b and "bird" in names
    sc = Scatter(4)
    city_noon = {n for t in np.arange(0, 300, 0.5) for n, _ in sc.due(t, "city", 12.0)}
    assert "rooster" not in city_noon and "bird" not in city_noon


def test_every_oneshot_ends():
    for name in ("bird", "rooster", "dog", "tres", "gull", "cow"):
        u = oneshot(name, 3)
        assert 0.1 < u.duration < 4.0 and rms(render(u, u.duration)) > 1e-4


def test_the_frame_crosses_the_beds_toward_the_biome(monkeypatch):

    from test_sound import Recorder, fake_game

    import contravia.audio.biome as B
    from contravia.audio import CONTROLS, Sound

    monkeypatch.setattr(B, "sample", lambda game, sampler: Place(built=.8, sea_m=1000, zone=0, sun_alt=40))
    s = Sound(Recorder(), CONTROLS)
    g = fake_game(1000.0)
    for k in range(10):
        s.frame(g, now=k * 0.5)
    v = s.audio.values
    assert s.biome == "city" and v["amb.city"] == 1.0 and v["amb.coast"] == 0.0
