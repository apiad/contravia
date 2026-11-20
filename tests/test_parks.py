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


def _kind(ring, built=0.0, buildings=0, green=0.0, city=True, boulevard=False, prado=False, seen=None):
    return parks.classify([Polygon(ring)], [built], [buildings], [green], [city], [boulevard], [prado],
                          None if seen is None else [seen])[0]


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


def test_a_block_the_satellite_sees_built_is_houses_osm_lacks_not_a_park():
    ring = _rect(100, 80)
    assert _kind(ring, seen=0.4) == parks.CITY
    assert _kind(ring, seen=parks.SEEN_BUILT) == parks.NONE
    assert _kind(_rect(120, 100), city=False, seen=0.9) == parks.NONE
    assert _kind(ring, green=6000, seen=0.9) == parks.CITY  # OSM's green wins
    assert _kind(_rect(300, 35), prado=True, seen=0.9) == parks.PRADO


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


def test_a_block_that_splits_when_shrunk_still_lays_out():
    dumbbell = shapely.union_all([box(0, 0, 60, 60), box(58, 28, 82, 31), box(80, 0, 140, 60)])  # a 3 m neck
    ring = np.asarray(dumbbell.exterior.coords)[:-1]
    for kind in (parks.CITY, parks.ORGANIC, parks.PASEO, parks.PRADO):
        for seed in range(6):
            park = parks.layout(ring, kind, seed)
            assert len(park.trees) or kind == parks.PRADO


def _street_city(ring, kind, band):
    """A CityData in cell (0, 0) holding one park of the given kind on the block `ring` and a drawn street
    surface over the box `band` (x0, z0, x1, z1), as streets.py lays one wider than the block's edge."""
    from contravia.citydata import CityData
    from contravia.terrain import Terrain

    x0, z0, x1, z1 = band
    t = np.array([[(x0, 0, z0), (x1, 0, z0), (x1, 0, z1)], [(x0, 0, z0), (x1, 0, z1), (x0, 0, z1)]], np.float32)
    city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), Terrain.flat())
    city.add_kind("street_tris", t, np.zeros((2, 2), int))
    city.add_kind("street_edges", np.zeros((0, 2, 3)), np.zeros((0, 2), int))
    ring = np.asarray(ring, np.float32)
    city.a.update(park_verts=ring, park_starts=np.array([0, len(ring)]), park_kind=np.array([kind], np.uint8),
                  park_zone=np.array([1], np.uint8), park_seed=np.array([parks.seed_of(Polygon(ring))]))
    return city


def test_no_motif_stands_on_the_drawn_street():
    ring = np.array([(5, 5), (95, 5), (95, 95), (5, 95)], float)
    city = _street_city(ring, parks.CITY, (0, 0, 22, 200))  # the street covers the park's west strip
    report = parks.add_to(city)
    tris = np.asarray(city.a["park_tris"], float)
    walls = np.asarray(city.a["park_walls"], float)
    assert len(tris) and len(walls)
    assert tris[:, :, 0].min() >= 22 - 1e-6  # no triangle of any motif reaches onto the street
    assert walls[:, [0, 2]].min() >= 22 - 1e-6
    assert report["props_on_streets"] > 0 and report["benches"] > 0  # some were left out, the rest stand


def test_a_park_the_drawn_street_mostly_covers_is_not_laid_out():
    strip = np.array([(0, 0), (300, 0), (300, 20), (0, 20)], float)  # a paseo between an avenue's carriageways
    city = _street_city(strip, parks.PASEO, (-10, -5, 310, 15))  # streets.py joined them over 3/4 of it
    assert parks.drop_covered(city) == {"parks_under_streets": 1}
    assert not len(city.a["park_kind"]) and not len(parks.laid(city.a))
    kept = _street_city(strip, parks.PASEO, (-10, -5, 310, 2))  # a street along its edge leaves it a park
    assert parks.drop_covered(kept) == {"parks_under_streets": 0} and len(kept.a["park_kind"]) == 1


def _grid(built: bool, size=4_000.0, at=(0.0, 0.0)):
    """A 20 m built-up grid size metres a side from `at`, all built or all open."""
    from contravia.builtup import BuiltUp

    n = int(size / 20)
    return BuiltUp.of(np.full((n, n), built), at)


def _town(ring, kind, city, built):
    return parks.in_town(np.array([kind]), [Polygon(ring)], np.array([city]), _grid(built))[0]


def test_a_park_in_a_built_town_is_urban_and_one_in_the_country_stays_wild():
    big = _rect(120, 100)  # 12 000 m²: too big for a plaza
    assert _kind(big, city=False) == parks.ORGANIC
    assert _town(big, parks.ORGANIC, False, built=True) == parks.CITY
    assert _town(big, parks.ORGANIC, False, built=False) == parks.ORGANIC
    assert _town(big, parks.CITY, False, built=False) == parks.ORGANIC  # the rule decides again, both ways


def test_a_park_half_ringed_by_houses_is_urban_and_one_with_a_few_is_not():
    from contravia.builtup import BuiltUp

    ring = _rect(120, 100, at=(2000.0, 2000.0))  # its ring runs from z 1750 to 2250
    half, strip = np.zeros((200, 200), bool), np.zeros((200, 200), bool)
    half[:100] = True  # houses everywhere south of z 2000: half the ring
    strip[88:90] = True  # one row of houses along z 1760 to 1800
    for mask, urban in ((half, True), (strip, False)):
        grid = BuiltUp.of(mask, (0.0, 0.0))
        share = parks.surroundings([Polygon(ring)], grid)[0]
        got = parks.in_town(np.array([parks.ORGANIC]), [Polygon(ring)], np.array([False]), grid)[0]
        assert (got == parks.CITY) == urban and (share >= parks.URBAN_BUILT) == urban, share


def test_the_city_municipios_parks_keep_their_kind_whatever_the_satellite_sees():
    for kind in (parks.CITY, parks.PASEO, parks.PRADO):
        for built in (True, False):
            assert _town(_rect(60, 50), kind, True, built) == kind
    assert _town(_rect(300, 35), parks.PASEO, False, False) == parks.PASEO  # a paseo is a paseo anywhere


def test_a_small_square_park_in_a_dense_town_centre_is_a_plaza():
    assert _town(_rect(60, 52), parks.ORGANIC, False, built=True) == parks.PLAZA
    assert _town(_rect(110, 40), parks.ORGANIC, False, built=True) == parks.CITY  # long: a city park


def test_a_plaza_is_paved_edge_to_edge_with_trees_in_planters_benches_facing_its_centrepiece():
    ring = _rect(60, 52, turn=0.4)
    poly = Polygon(ring)
    for seed in range(12):
        park = parks.layout(ring, parks.PLAZA, seed)
        assert park.kerb is not None and abs(park.kerb.area - poly.area) < 1.0  # all of it paved
        assert not park.paved and not park.dirt and park.lake is None and not len(park.claros)
        centre = [m for m in park.motifs if m[0] in ("statue", "glorieta", "fountain")]
        assert len(centre) == 1 and np.hypot(centre[0][1] - 1000, centre[0][2] - 2000) < 1.0
        planters = np.array([m[1:3] for m in park.motifs if m[0] == "planter"])
        assert len(park.trees) >= 12 and np.allclose(np.sort(planters, axis=0), np.sort(park.trees, axis=0))
        assert all(poly.buffer(-parks.PLANTER).contains(Point(p)) for p in planters)
        to_c = np.array([1000.0, 2000.0]) - park.benches[:, :2]
        face = np.column_stack([np.cos(park.benches[:, 2]), np.sin(park.benches[:, 2])])
        assert len(park.benches) >= 8 and np.all((face * to_c).sum(axis=1) / np.hypot(*to_c.T) > 0.99)
        assert len(park.lamps) >= 8 and all(poly.contains(Point(p)) for p in park.lamps[:, :2])
        tall = shapely.union_all([Point(m[1], m[2]).buffer(m[3] + 1.0) for m in park.motifs if m[0] != "planter"])
        assert not any(tall.contains(Point(p)) for p in park.trees)  # no tree in the centrepiece


def test_the_same_plaza_every_time():
    ring = _rect(55, 50)
    a, b = parks.layout(ring, parks.PLAZA, 99), parks.layout(ring, parks.PLAZA, 99)
    assert a.motifs == b.motifs and np.array_equal(a.trees, b.trees) and np.array_equal(a.benches, b.benches)
    assert np.array_equal(a.lamps, b.lamps) and a.kerb.equals(b.kerb)


def test_retown_turns_a_town_park_into_a_plaza_with_its_trees_and_motifs():
    from contravia.builtup import BuiltUp

    ring = np.array([(40, 40), (100, 40), (100, 92), (40, 92)], float)
    city = _street_city(ring, parks.ORGANIC, (-50, -50, -40, -40))
    city.a["park_zone"] = np.array([2], np.uint8)  # outside the city's municipios
    old = parks.layout(ring, parks.ORGANIC, int(city.a["park_seed"][0])).trees
    city.add_kind("trees", old.astype(np.float32), np.floor(old / 200).astype(int))
    city.a.pop("tree_offsets"), city.a.update(tree_offsets=city.a.pop("trees_offsets"))
    report = parks.retown(city, BuiltUp.of(np.ones((40, 40), bool), (-200.0, -200.0)))
    assert report["parks_changed"] == 1 and report["parks_after"]["plaza"] == 1
    assert int(city.a["park_kind"][0]) == parks.PLAZA
    want = parks.layout(ring, parks.PLAZA, int(city.a["park_seed"][0])).trees
    got = np.asarray(city.a["trees"], float)
    assert report["trees_dropped"] == len(old) and len(got) == len(want)
    assert np.allclose(np.sort(got, axis=0), np.sort(want, axis=0), atol=1e-3)
    built = parks.add_to(city)
    assert built["motifs"] >= len(want) + 1  # every planter and the centrepiece
    again = parks.retown(city, BuiltUp.of(np.ones((40, 40), bool), (-200.0, -200.0)))
    assert again["parks_changed"] == 0  # a second run changes nothing
