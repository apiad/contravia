"""Parks: which blocks become parks, and how each one is laid out."""

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box

from contravia import parks


def _rect(w, h, turn=0.0, at=(1000.0, 2000.0)):
    """A w x h block centred at `at`, turned by `turn` radians, as a (4, 2) ring."""
    c, s = np.cos(turn), np.sin(turn)
    p = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return p @ np.array([[c, s], [-s, c]]) + at


def _kind(ring, built=0.0, buildings=0, green=0.0, city=True, boulevard=False, prado=False):
    return parks.classify([Polygon(ring)], [built], [buildings], [green], [city], [boulevard], [prado])[0]


def test_an_empty_city_block_is_a_city_park_with_a_centrepiece_on_its_plaza():
    ring = _rect(100, 80)
    assert _kind(ring) == parks.CITY
    park = parks.layout(ring, parks.CITY, 7)
    names = [m[0] for m in park.motifs]
    assert {"glorieta", "fountain", "bust"} & set(names)
    centre = next(m for m in park.motifs if m[0] in ("glorieta", "fountain", "bust"))
    assert np.hypot(centre[1] - 1000, centre[2] - 2000) < 1.0
    assert park.kerb is not None and park.kerb.area > 0 and len(park.paved) >= 3
    assert len(park.trees) and len(park.benches) and len(park.lamps)


def test_a_long_thin_block_is_a_paseo_and_the_prado_is_paved_edge_to_edge():
    ring = _rect(300, 35)
    assert _kind(ring) == parks.PASEO
    assert _kind(_rect(100, 35), boulevard=True) == parks.PASEO  # between a boulevard's carriageways
    paseo = parks.layout(ring, parks.PASEO, 3)
    assert paseo.paved and not paseo.lake
    prado = parks.layout(ring, parks.PRADO, 3)
    assert abs(prado.kerb.area - Polygon(ring).area) < 1.0 and len(prado.trees) and len(prado.benches)


def test_a_small_empty_block_outside_the_city_keeps_its_filler():
    assert _kind(_rect(60, 60), city=False) == parks.NONE
    assert _kind(_rect(120, 100), city=False) == parks.ORGANIC


def test_built_blocks_are_not_parks_unless_osm_calls_them_green():
    ring = _rect(100, 80)
    assert _kind(ring, buildings=4) == parks.NONE
    assert _kind(ring, built=0.08 * 8000, buildings=2) == parks.NONE
    assert _kind(ring, built=0.08 * 8000, buildings=6, green=6000) == parks.CITY
    assert _kind(_rect(30, 30)) == parks.NONE  # under 1 500 m²
    assert _kind(_rect(200, 12)) == parks.NONE  # too narrow


def test_the_same_block_gives_the_same_park():
    ring = _rect(240, 160)
    a, b = parks.layout(ring, parks.CITY, 42), parks.layout(ring, parks.CITY, 42)
    assert a.motifs == b.motifs
    assert np.array_equal(a.trees, b.trees) and np.array_equal(a.benches, b.benches)
    assert (a.lake is None) == (b.lake is None)


def _lakes(ring, kind, seeds=range(40)):
    return [(s, p) for s in seeds if (p := parks.layout(ring, kind, s)).lake is not None]


def test_a_lake_only_in_a_big_park_and_no_path_crosses_it():
    assert not _lakes(_rect(150, 120), parks.CITY, range(15))  # 18 000 m²: too small
    got = _lakes(_rect(260, 180), parks.CITY)
    assert len(got) >= 10
    for _, park in got:
        water = park.lake.water
        assert parks.LAKE_MIN * 0.3 <= water.area <= parks.LAKE_MAX * 1.6
        crossing = [g for g in park.paved if g.intersection(water).area > 1.0]
        assert not crossing
        assert Polygon(_rect(260, 180)).buffer(-parks.KERB_W + 0.01).contains(water)


def test_lakes_get_footbridges_piers_and_boats_on_the_water():
    got = _lakes(_rect(260, 180), parks.CITY) + _lakes(_rect(260, 180), parks.ORGANIC)
    assert any(p.lake.bridge is not None for _, p in got)
    assert any(p.lake.pier is not None for _, p in got)
    for _, park in got:
        lk = park.lake
        if len(lk.boats):
            assert shapely.contains_xy(lk.water.buffer(0.5), lk.boats[:, 0], lk.boats[:, 1]).all()
        if lk.bridge is not None:
            ends = shapely.points(np.asarray(lk.bridge))
            assert not shapely.contains(lk.water, ends).any()  # both ends on land


def test_nothing_stands_on_a_path_or_in_the_water():
    for kind, ring in ((parks.CITY, _rect(260, 180)), (parks.CITY, _rect(90, 70, turn=0.5)),
                       (parks.ORGANIC, _rect(260, 180)), (parks.PASEO, _rect(300, 35))):
        for seed in range(8):
            park = parks.layout(ring, kind, seed)
            paths = shapely.union_all(park.paved + park.dirt) if park.paved or park.dirt else Polygon()
            water = park.lake.water if park.lake else Polygon()
            if len(park.trees):
                pts = shapely.points(park.trees)
                assert (shapely.distance(paths, pts) > 0.5).all() or paths.is_empty
                assert not shapely.contains(water, pts).any()
            if len(park.benches):
                assert not shapely.contains(water, shapely.points(park.benches[:, :2])).any()
            for name, x, z, size, _ in park.motifs:
                if name in ("playground", "domino", "picnic", "ranchon"):
                    assert not paths.buffer(-0.01).contains(Point(x, z)), (kind, seed, name)
            assert Polygon(ring).buffer(0.5).contains(shapely.multipoints(park.trees)) if len(park.trees) else True


def test_a_turned_block_lays_out_in_its_own_frame():
    ring = _rect(140, 60, turn=0.7)
    park = parks.layout(ring, parks.CITY, 1)
    block = Polygon(ring)
    assert all(block.contains(Point(x, z)) for _, x, z, _, _ in park.motifs)
    longest = max(park.paved, key=lambda g: g.length)
    assert block.buffer(0.5).contains(longest)


def test_claros_are_open_lawn():
    park = parks.layout(_rect(200, 150), parks.CITY, 5)
    assert len(park.claros)
    trees = shapely.points(park.trees)
    for c in park.claros[:20]:
        assert shapely.distance(trees, Point(c)).min() > parks.CLARO_TREES - 1e-6
    assert box(*Polygon(_rect(200, 150)).bounds).contains(shapely.multipoints(park.claros))
