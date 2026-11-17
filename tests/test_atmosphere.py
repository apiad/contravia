"""The look of the sky from the sun, the moon and the weather; and the shadow box's snapping."""

from datetime import datetime

import numpy as np
import pytest

from caiman import sky
from caiman.atmosphere import clear_sky_factor, light_basis, look, snap_target
from caiman.sky import Body
from caiman.weather import Now, SkyConfig

CLEAR, STORM = Now("clear", 0.05, 0.0, 0.0), Now("storm", 1.0, 1.0, 1.0)
NO_MOON = Body(-30.0, 0.0, 0.0, 0.0, lit=0.0)


def test_noon_is_white_and_strong_sunset_orange_night_dim():
    noon = look(Body(60.0, 180.0, 0, 0), NO_MOON, CLEAR, SkyConfig())
    dusk = look(Body(2.0, 250.0, 0, 0), NO_MOON, CLEAR, SkyConfig())
    night = look(Body(-40.0, 0.0, 0, 0), Body(40.0, 120.0, 0, 0, lit=1.0), CLEAR, SkyConfig())
    assert noon.light_intensity > 2.5 and noon.light_color[2] > 0.8
    assert dusk.light_color[2] < 0.5 < dusk.light_color[0] and dusk.horizon[0] > dusk.horizon[2]
    assert night.light_intensity < 0.4 and night.light_color[2] > night.light_color[0]
    assert sum(night.zenith) < 0.3 < sum(noon.zenith)
    assert noon.light_dir[1] < 0  # light travels down


def test_cloud_cuts_the_sun_by_kasten_czeplak():
    sun = Body(60.0, 180.0, 0, 0)
    a = look(sun, NO_MOON, CLEAR, SkyConfig()).light_intensity
    b = look(sun, NO_MOON, Now("overcast", 0.9, 0.0, 0.0), SkyConfig()).light_intensity
    assert b / a == pytest.approx(clear_sky_factor(0.9) / clear_sky_factor(0.05), rel=1e-6)


def test_shadows_only_under_a_high_sun_and_a_clear_enough_sky():
    cfg = SkyConfig()
    assert look(Body(30.0, 180, 0, 0), NO_MOON, CLEAR, cfg).shadows
    assert not look(Body(2.0, 250, 0, 0), NO_MOON, CLEAR, cfg).shadows
    assert not look(Body(30.0, 180, 0, 0), NO_MOON, STORM, cfg).shadows


def test_stars_need_a_dark_clear_sky_and_glow_hides_them():
    dark = Body(-40.0, 0, 0, 0)
    city = look(dark, NO_MOON, CLEAR, SkyConfig()).star_limit
    blackout = look(dark, NO_MOON, CLEAR, SkyConfig(city_glow=0.0)).star_limit
    assert blackout > 6.0 > city > 4.0
    assert look(Body(30.0, 180, 0, 0), NO_MOON, CLEAR, SkyConfig()).star_limit < -1
    assert look(dark, NO_MOON, STORM, SkyConfig()).star_limit < 0


def test_storm_brings_the_fog_in():
    sun = Body(30.0, 180, 0, 0)
    assert look(sun, NO_MOON, STORM, SkyConfig()).fog[1] < look(sun, NO_MOON, CLEAR, SkyConfig()).fog[1] / 2


def test_snapping_keeps_the_grid():
    d = -sky.direction(*[float(v) for v in (40.0, 120.0)])
    a = np.array(snap_target((10.0, 5.0, -3.0), d))
    b = np.array(snap_target((10.0 + 0.03, 5.0, -3.0 + 0.02), d))
    texel = 2 * 150.0 / 2048
    basis = light_basis(d)
    for p in (a, b):
        along = basis @ p
        assert np.allclose(along / texel, np.round(along / texel), atol=1e-6)


def test_real_evening_in_november():
    t = datetime(2026, 11, 15, 17, 40)
    lk = look(sky.sun(t), sky.moon(t), CLEAR, SkyConfig())
    assert lk.env in ("golden", "twilight") and not lk.shadows
