import numpy as np
import pytest
import shapely

from contravia import cemetery
from contravia import regions as module
from contravia.regions import RegionError, Regions

# A cemetery OSM draws to the middle of the streets round it (x 20..180, z 20..120), with a service road in from
# its north street and a chapel inside; and a second one, with no lane at all, east of it.
YARD = shapely.box(20.0, 20.0, 180.0, 120.0)
BARE = shapely.box(300.0, 20.0, 420.0, 100.0)
STREETS = [[20.0, 20.0, 180.0, 20.0, 8.0], [180.0, 20.0, 180.0, 120.0, 8.0], [180.0, 120.0, 20.0, 120.0, 8.0],
           [20.0, 120.0, 20.0, 20.0, 8.0], [100.0, 20.0, 100.0, 90.0, 4.0]]
CHAPEL = np.array([[120.0, 60.0], [140.0, 60.0], [140.0, 75.0], [120.0, 75.0]])
SRC = {"tag:landuse=cemetery": shapely.MultiPolygon([YARD, BARE]),
       "lanes:main": shapely.MultiLineString([[(100.0, 20.0), (100.0, 90.0)]])}
FILE = """
[[region]]
name = "Cementerios"
where = { tag = "landuse=cemetery" }
priority = 10
[region.blocks]
fill = "cemetery"
parks = "drop"
[region.cemetery]
pantheons = 0.5
"""


@pytest.fixture
def city(monkeypatch):
    from contravia.citydata import CityData

    monkeypatch.setattr(module, "_current", Regions.parse(FILE, SRC))
    return CityData.from_features(STREETS, [CHAPEL], [9], [], [], np.zeros((0, 2)))


def rows(city, cells=((0, 0),)):
    return [cemetery.yard(city, i, j) for i, j in cells]


def test_the_region_values_load_and_merge():
    r = Regions.parse(FILE + '\n[[region]]\nname = "Colón"\nwhere = { tag = "landuse=cemetery" }\npriority = 30\n'
                      '[region.cemetery]\npalette = ["#efece4", "#bfbab0"]\n', SRC)
    p = r.assign(np.array([[50.0, 50.0]]), [None])[0]
    assert (p.fill, p.pantheons, p.stones) == ("cemetery", 0.5, ("#efece4", "#bfbab0"))
    assert r.lanes.keys() == {"lanes:main"} and len(r.cemeteries()[0]) == 2


@pytest.mark.parametrize("text, where", [
    ('[cemetery]\npantheons = 2', "cemetery.pantheons"),
    ('[cemetery]\npalette = ["white"]', "cemetery.palette"),
    ('[cemetery]\nangels = 0.1', "cemetery.angels"),
])
def test_a_bad_cemetery_group_stops_the_load(text, where):
    body = FILE.replace("[region.cemetery]\npantheons = 0.5", text.replace("[cemetery]", "[region.cemetery]"))
    with pytest.raises(RegionError, match=f"region 'Cementerios', {where}"):
        Regions.parse(body, SRC)


def test_a_municipio_cannot_fill_with_a_cemetery():
    with pytest.raises(RegionError, match=r"region 'A', blocks\.fill"):
        Regions.parse('[[region]]\nname = "A"\nwhere = { municipio = "X" }\n[region.blocks]\nfill = "cemetery"')


def test_rows_are_the_same_every_time(city, monkeypatch):
    first = rows(city)[0]
    monkeypatch.setattr(module, "_current", Regions.parse(FILE, SRC))  # new regions: everything laid out again
    again = rows(city)[0]
    assert len(first) > 500
    for f in ("xz", "theta", "kind", "tone", "wall", "pillars"):
        np.testing.assert_array_equal(getattr(first, f), getattr(again, f))


def test_every_kind_is_laid_and_pantheons_stand_along_the_main_lane(city):
    y = rows(city)[0]
    counts = np.bincount(y.kind, minlength=len(cemetery.KINDS))
    assert (counts > 0).all(), dict(zip(cemetery.KINDS, counts, strict=True))
    lane = shapely.LineString([(100.0, 20.0), (100.0, 90.0)])
    pantheons = y.xz[y.kind == cemetery.KINDS.index("pantheon")]
    assert (shapely.distance(lane, shapely.points(pantheons)) < 10.0).all()


def test_plots_keep_off_lanes_streets_and_buildings_and_stay_inside_the_wall(city):
    y = rows(city)[0]
    plots = shapely.polygons(cemetery.footprints(y))
    strips = shapely.buffer(shapely.linestrings(np.array(STREETS)[:, :4].reshape(-1, 2, 2)),
                            np.array(STREETS)[:, 4] / 2 + cemetery.GAP - 0.01, cap_style="flat")
    assert not shapely.intersects(plots[:, None], strips[None]).any()
    assert not shapely.intersects(plots, shapely.Polygon(CHAPEL).buffer(cemetery.GAP - 0.01)).any()
    assert shapely.contains(YARD, plots).all()
    assert not shapely.intersects(plots[:, None], plots[None]).sum() > len(plots)  # only each with itself


def test_a_cemetery_with_no_lanes_gets_a_grid(city):
    y = rows(city, [(1, 0), (2, 0)])
    xz = np.concatenate([r.xz for r in y])
    inside = shapely.contains_xy(BARE, xz[:, 0], xz[:, 1])
    assert inside.sum() > 300
    grid = cemetery.plan_of(cemetery.touching(*BARE.bounds)[0], city).lanes["grid"]
    gap = shapely.distance(grid, shapely.points(xz[inside]))
    assert (gap >= cemetery.LANE["grid"] / 2 + cemetery.LONG - 1e-6).all()


def test_the_wall_leaves_a_gate_with_pillars_for_the_road_in(city):
    y = rows(city)[0]
    assert np.hypot(*(y.wall[:, 1] - y.wall[:, 0]).T).sum() > 400.0 and len(y.pillars) == 2
    road = shapely.LineString([(100.0, 20.0), (100.0, 90.0)]).buffer(2.0)
    assert not shapely.intersects(shapely.linestrings(y.wall), road).any()
    assert (np.abs(y.pillars[:, 0] - 100.0) < 6.0).all()
    streets = shapely.buffer(shapely.linestrings(np.array(STREETS)[:4, :4].reshape(-1, 2, 2)), 4.0)
    assert not shapely.intersects(shapely.linestrings(y.wall)[:, None], streets[None]).any()  # behind the street


def test_every_raised_tomb_is_in_the_walls_and_no_slab_is(city):
    from contravia.walls import Walls

    y = rows(city)[0]
    w = Walls(city)
    hit = {tuple(np.round(s.ravel(), 3)) for s in np.asarray(w.near(60.0, 70.0, 0.0), float)}
    sight = {tuple(np.round(s.ravel(), 3)) for s in np.asarray(w.near(60.0, 70.0), float)}
    corners = cemetery.footprints(y).astype(np.float32).astype(float)
    edges = np.stack([corners, np.roll(corners, -1, axis=1)], 2)
    near = np.hypot(*(y.xz - (60.0, 70.0)).T) < 40.0
    raised = np.array(cemetery.RAISED)[y.kind]
    for k in np.flatnonzero(near):
        inside = [tuple(np.round(e.ravel(), 3)) in hit for e in edges[k]]
        assert all(inside) if raised[k] else not any(inside), cemetery.KINDS[y.kind[k]]
        seen = [tuple(np.round(e.ravel(), 3)) in sight for e in edges[k]]
        assert all(seen) == (cemetery.KINDS[y.kind[k]] == "pantheon")  # tombs never block a line of sight


def test_a_cell_lays_out_only_its_own_plots_and_its_neighbour_the_next(city):
    a, b = rows(city, [(0, 0), (0, -1)])
    assert (a.xz >= 0).all() and (a.xz < cemetery.CELL).all() and not len(b)


def test_grass_and_yard_trees_keep_off_the_tombs(city):
    from contravia.grass import open_ground

    y = rows(city)[0]
    assert not open_ground(city, 0, 0, y.xz).any()


def test_every_kind_has_a_mesh_at_each_level_and_a_far_plot_is_its_flat_top():
    from contravia.rules.tombs import mesh

    for k, kind in enumerate(cemetery.KINDS):
        tris = []
        for lod in (0, 1, 2):
            g = mesh(k, lod)
            assert len(g["indices"]) and len(g["positions"]) == len(g["colors"]) == len(g["normals"])
            tris.append(len(g["indices"]) // 3)
        assert tris[0] >= tris[1] >= tris[2]
        if kind in ("slab", "tomb", "cross"):
            g = mesh(k, 2)
            p, n = g["positions"][g["indices"]].reshape(-1, 3, 3), g["normals"][g["indices"]].reshape(-1, 3, 3)
            assert tris[2] == 2 and (n[..., 1] == 1).all()
            assert (np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 1] > 0).all()  # wound to face up


def test_far_summaries_draw_the_cemetery_pale_with_its_pantheons(city):
    from contravia.mesh import MeshBuilder

    y = rows(city)[0]
    mb = MeshBuilder()
    cemetery.far(mb, 0.0, 0.0, 400.0, lambda xz: np.zeros(len(xz)), 1, cemetery.joined([y]))
    pantheons = int((y.kind == cemetery.KINDS.index("pantheon")).sum())
    assert pantheons and mb._count > 20 * pantheons
