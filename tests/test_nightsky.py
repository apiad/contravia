"""The night sky's packing and where its stars land."""

from datetime import datetime

import numpy as np

from contravia import sky
from contravia.nightsky import KIND_MW, colour_bin, load_points, pack, unpack


def test_pack_and_unpack():
    k, m = unpack(pack([0, 7, 15, KIND_MW], [-1.46, 3.2, 6.5, 0.0]))
    assert list(k) == [0, 7, 15, KIND_MW] and np.allclose(m, [-1.46, 3.2, 6.5, 0.0], atol=1e-4)


def test_red_stars_get_higher_bins():
    assert colour_bin(1.5) > colour_bin(0.7) > colour_bin(-0.2)


def test_betelgeuse_stands_in_the_east_on_a_november_night():
    when = datetime(2026, 11, 15, 23, 0)
    v, values, _, _ = load_points(when)
    kind, mag = unpack(values)
    d = v @ sky.equatorial_to_local(when).T
    i = int(np.argmin(np.abs(mag - 0.5) + 10 * (kind >= KIND_MW) + 10 * (np.abs(d[:, 1] - np.sin(np.radians(33.85))) > 0.05)))
    assert d[i, 0] > 0.7 and d[i, 1] > 0.4  # east (+x) and about 34 degrees up


def test_the_milky_way_is_behind_the_stars_and_half_below_the_horizon():
    when = datetime(2026, 7, 15, 23, 0)
    v, values, _, domes = load_points(when)
    mw = unpack(values)[0] == KIND_MW
    assert domes[mw].min() > domes[~mw].max()
    up = (v[mw] @ sky.equatorial_to_local(when).T)[:, 1] > 0
    assert 0.2 < up.mean() < 0.9


def test_the_catalogue_exists_only_while_stars_show():
    from types import SimpleNamespace

    import manifoldx as mx

    from contravia.nightsky import NightSky

    engine = mx.Engine("night-test", width=64, height=48)
    when = datetime(2026, 11, 15, 23, 0)
    ns = NightSky(engine, when)
    moon, sun = sky.Body(alt=-20.0, az=90.0, ra=0.0, dec=0.0), sky.Body(alt=-40.0, az=270.0, ra=0.0, dec=0.0)

    def frame(night):
        ns.update(when, (0.0, 2.0, 0.0), SimpleNamespace(night=night, star_limit=6.0), moon, sun)
        engine.render_frame(supersample=1)
        return len(engine.store.get_component_view(["PointCloud"]))

    assert frame(0.0) == 7  # by day: the planets and the moon only
    assert frame(1.0) == len(ns.v) + 7  # night: the catalogue too
    frame(0.0)  # dawn: destroyed at this frame's flush
    assert frame(0.0) == 7
    assert frame(1.0) == len(ns.v) + 7  # and back the next night


def test_a_storm_hides_the_stars_before_the_dome_turns_again():
    from types import SimpleNamespace

    import manifoldx as mx

    from contravia.nightsky import NightSky

    engine = mx.Engine("night-test", width=64, height=48)
    when = datetime(2026, 11, 15, 23, 0)
    ns = NightSky(engine, when)
    moon, sun = sky.Body(alt=-20.0, az=90.0, ra=0.0, dec=0.0), sky.Body(alt=-40.0, az=270.0, ra=0.0, dec=0.0)
    eye = (0.0, 2.0, 0.0)
    assert ns.update(when, eye, SimpleNamespace(night=1.0, star_limit=6.0), moon, sun) > 0
    assert ns.update(when, eye, SimpleNamespace(night=1.0, star_limit=-5.0), moon, sun) == 0  # same second, same place
