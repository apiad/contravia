import numpy as np
import pytest
import shapely

from contravia.buildingtags import KINDS
from contravia.regions import RegionError, Regions
from contravia.rules.facades import plan

SQUARE = shapely.box(0, 0, 1000, 1000)  # a polygon in the island's metres, as regions.src would hold it
SRC = {"tag:landuse=cemetery": SQUARE, "osm:way/1": shapely.box(400, 400, 600, 600)}

FILE = """
[[region]]
name = "Low"
where = { municipio = "Guanabacoa" }
priority = 50
[region.buildings]
types = { colonial = 1 }
storeys = [1, 1]
portal = true

[[region]]
name = "Square"
where = { tag = "landuse=cemetery" }
[region.buildings]
types = { eclectic = 0.5, vedado = 0.5 }

[[region]]
name = "Middle"
where = { osm = "way/1" }
priority = 1
[region.buildings]
storeys = [3, 3]

[[region]]
name = "Middle high"
where = { osm = "way/1" }
priority = 2
[region.buildings]
wear = [0.9, 0.9]
storeys = [4, 4]
"""


def regions():
    return Regions.parse(FILE, SRC)


def params(r, *names):
    return r.params(frozenset(k for k, x in enumerate(r.regions) if x.name in names))


def test_later_layers_win_field_by_field_whatever_the_priority():
    r = regions()
    p = params(r, "Low", "Square")
    assert [c.types for c in p.types] == [(("eclectic", 0.5), ("vedado", 0.5))]  # the tag layer over the municipio's
    assert p.storeys == (1, 1) and p.portal  # what the tag region leaves out stays the municipio's
    assert p.regions == ("Low", "Square")


def test_inside_a_layer_the_higher_priority_wins():
    p = params(regions(), "Low", "Square", "Middle", "Middle high")
    assert p.storeys == (4, 4) and p.wear == (0.9, 0.9)


def test_a_weights_table_is_replaced_whole():
    p = params(regions(), "Low", "Square")
    assert dict(p.types[0].types) == {"eclectic": 0.5, "vedado": 0.5}  # no colonial left from Low


def test_the_cache_hands_back_the_same_params_for_the_same_regions():
    r = regions()
    assert params(r, "Low", "Square") is params(r, "Square", "Low")
    assert r.municipio("Guanabacoa") is r.municipio("Guanabacoa") is params(r, "Low")
    assert r.municipio("Nowhere") is params(r)


def test_a_cell_inside_or_outside_every_polygon_takes_the_fast_path():
    r = regions()
    assert r.around(100, 100, 300, 300) == ([0], [])  # inside the square, away from the way
    assert r.around(2000, 2000, 2200, 2200) == ([], [])  # outside them all
    pts = np.array([[150.0, 150.0], [250.0, 250.0]])
    a, b = r.assign(pts, ["Guanabacoa", None])
    assert a.regions == ("Low", "Square") and b.regions == ("Square",)


def test_a_cell_an_edge_crosses_puts_each_item_on_its_side():
    r = regions()
    whole, cut = r.around(500, 300, 700, 500)  # the way's corner at (600, 400) lies inside
    assert whole == [0] and len(cut) == 2  # both regions of the way
    pts = np.array([[550.0, 450.0], [650.0, 450.0], [550.0, 350.0]])
    inside, east, south = r.assign(pts, [None, None, None])
    assert inside.storeys == (4, 4) and inside.regions == ("Square", "Middle", "Middle high")
    assert east is south and east.regions == ("Square",)


def test_buildings_take_the_types_and_storeys_of_their_place():
    r = regions()
    yes = np.array([KINDS.index("yes"), np.nan, np.nan, 0, -1], np.float32)
    p = plan(7, yes, params(r, "Low"), 120.0)
    assert p.type == "colonial" and p.storeys == 1
    q = plan(7, np.array([KINDS.index("yes"), 6, np.nan, 0, -1], np.float32), params(r, "Low"), 120.0)
    assert q.storeys == 6  # OSM's levels come first
    w = plan(7, np.array([KINDS.index("warehouse"), np.nan, np.nan, 0, -1], np.float32), params(r, "Low"), 120.0)
    assert w.type == "warehouse" and w.storeys == 1


def test_cases_pick_by_kind_and_footprint():
    r = Regions.parse("""
[[region]]
name = "Plaza"
where = { municipio = "Plaza de la Revolución" }
[region.buildings]
types = [{ above = 900, types = { tower = 1 }, storeys = [5, 12] }, { kind = "apartments", types = { prefab = 1 } },
         { below = 50, types = { suburban = 1 } }, { types = { vedado = 1 } }]
""")
    p = r.municipio("Plaza de la Revolución")
    assert [p.pick(k, a).types[0][0] for k, a in
            (("yes", 1000), ("apartments", 1000), ("apartments", 100), ("yes", 40), ("yes", 100))] == \
        ["tower", "tower", "prefab", "suburban", "vedado"]
    assert all(5 <= plan(k, None, p, 1000.0).storeys <= 12 for k in range(30))


@pytest.mark.parametrize("text, where", [
    ('[[region]]\nname = "A"\nwhere = { municipio = "X" }\nheight = 3', "region 'A', height"),
    ('[[region]]\nname = "A"\nwhere = { municipio = "X" }\n[region.buildings]\nroof = 1', "region 'A', buildings.roof"),
    ('[[region]]\nname = "A"\nwhere = { municipio = "X" }\n[region.buildings]\ntypes = { castle = 1 }',
     "region 'A', buildings.types"),
    ('[[region]]\nname = "A"\nwhere = { municipio = "X" }\n[region.buildings]\ntypes = [{ above = 3, types = { tower = 1 } }]',
     "region 'A', buildings.types"),
    ('[[region]]\nname = "A"\nwhere = { tag = "landuse=farmland" }', "region 'A', where.tag"),
    ('[[region]]\nname = "A"\nwhere = { polygon = [[23.1, -82.3], [23.1, -82.3], [23.1, -82.3]] }',
     "region 'A', where.polygon"),
    ('[[region]]\nname = "A"\nwhere = { street = "Obispo" }', "region 'A', where"),
    ('[[region]]\nname = "A"\nwhere = { municipio = "X" }\n[region.buildings]\nstoreys = [3, 2]',
     "region 'A', buildings.storeys"),
])
def test_a_bad_region_stops_the_load_with_its_name_and_key(text, where):
    with pytest.raises(RegionError, match=where.replace("[", r"\[").replace(".", r"\.")):
        Regions.parse(text, SRC)


def test_a_municipio_the_data_lacks_stops_the_check():
    r = Regions.parse('[[region]]\nname = "A"\nwhere = { municipio = "Habana Veja" }')
    with pytest.raises(RegionError, match="region 'A', where.municipio"):
        r.check({"La Habana Vieja"})


def test_a_polygon_in_lat_lon_lands_in_the_islands_metres():
    from contravia.geo import to_local

    r = Regions.parse('[[region]]\nname = "P"\nwhere = { polygon = [[23.12, -82.30], [23.13, -82.30], '
                      '[23.13, -82.29], [23.12, -82.29]] }\n[region.buildings]\nstoreys = [2, 2]')
    inside, outside = to_local(np.array([23.125, 23.14]), np.array([-82.295, -82.295]))
    assert r.assign(np.array([inside, outside]), [None, None])[0].storeys == (2, 2)
    assert r.assign(np.array([outside]), [None])[0].storeys is None


def test_the_shipped_file_loads_and_names_only_municipios_of_la_habana():
    from contravia.regions import current

    current().check({"Arroyo Naranjo", "Boyeros", "Centro Habana", "Cerro", "Cotorro", "Diez de Octubre",
                     "Guanabacoa", "Habana del Este", "La Habana Vieja", "La Lisa", "Marianao", "Playa",
                     "Plaza de la Revolución", "Regla", "San Miguel del Padrón"})


CEMETERY = """
[[region]]
name = "Cementerios"
where = { tag = "landuse=cemetery" }
priority = 10
[region.blocks]
fill = "none"
parks = "drop"

[[region]]
name = "Colón"
where = { osm = "way/1" }
priority = 30
"""


def test_a_region_without_regions_src_is_left_out_and_one_it_lacks_stops_the_load():
    assert [r.name for r in Regions.parse(CEMETERY).regions] == []
    with pytest.raises(RegionError, match=r"region 'Colón', where\.osm"):
        Regions.parse(CEMETERY, {"tag:landuse=cemetery": SQUARE})


def test_blocks_merge_like_any_field():
    r = Regions.parse(CEMETERY, SRC)
    inside, outside = r.assign(np.array([[500.0, 500.0], [1500.0, 500.0]]), [None, None])
    assert (inside.fill, inside.parks, inside.regions) == ("none", "drop", ("Cementerios", "Colón"))
    assert (outside.fill, outside.parks) == ("houses", "keep")


@pytest.mark.parametrize("text, where", [
    ('[[region]]\nname = "A"\nwhere = { municipio = "X" }\n[region.blocks]\nfill = "tombs"', "region 'A', blocks.fill"),
    ('[[region]]\nname = "A"\nwhere = { municipio = "X" }\n[region.blocks]\nroads = "none"', "region 'A', blocks.roads"),
    ('[[region]]\nname = "A"\nwhere = { municipio = "X" }\n[region.blocks]\nparks = "drop"', "region 'A', blocks.parks"),
])
def test_a_bad_blocks_group_stops_the_load(text, where):
    with pytest.raises(RegionError, match=where.replace(".", r"\.")):
        Regions.parse(text, SRC)


def test_a_tag_of_many_polygons_is_one_region_in_every_part():
    two = shapely.MultiPolygon([shapely.box(0, 0, 100, 100), shapely.box(5000, 0, 5100, 100)])
    r = Regions.parse(CEMETERY.split("[[region]]\nname = \"Colón\"")[0], {"tag:landuse=cemetery": two})
    a, b, c = r.assign(np.array([[50.0, 50.0], [5050.0, 50.0], [2500.0, 50.0]]), [None, None, None])
    assert a is b and a.fill == "none" and c.fill == "houses"


def test_filler_houses_stay_out_of_a_region_that_fills_with_none(monkeypatch):
    from contravia import regions as module
    from contravia.citydata import Cell
    from contravia.clearance import boxes

    monkeypatch.setattr(module, "_current", Regions.parse(CEMETERY, SRC))
    block = lambda x: np.array([[x, 100.0], [x + 60, 100.0], [x + 60, 160.0], [x, 160.0]])
    cell = Cell(fillers=[block(500.0), block(1100.0)], filler_seeds=np.array([3, 4]))
    rings = [r for r, *_ in boxes(cell, np.zeros((0, 5)), np.zeros((0, 3, 2)))]
    assert rings and all(r[:, 0].min() >= 1100 for r in rings)  # only the block outside the cemetery


def test_parks_a_region_drops_lose_their_motifs_walls_and_trees():
    from contravia.citydata import Cell

    r = Regions.parse(CEMETERY, SRC)
    tri = lambda x: [[x, 0.0, 10.0], [x + 2, 0.0, 10.0], [x, 0.0, 12.0]]
    cell = Cell(trees=np.array([[50.0, 50.0], [1500.0, 50.0]]), park_tris=np.array([tri(50.0), tri(1500.0)], np.float32),
                park_paint=np.array([1, 2], np.uint8),
                park_walls=np.array([[40, 10, 60, 10, 0, 1], [1490, 10, 1510, 10, 0, 1]], np.float32))
    kept = r.parks(cell)
    assert kept.trees.tolist() == [[1500.0, 50.0]] and kept.park_paint.tolist() == [2]
    assert kept.park_tris[0, 0, 0] == 1500 and kept.park_walls[:, 0].tolist() == [1490]
    assert Regions.parse(FILE, SRC).parks(cell) is cell  # no region drops parks: the cell as it was


def test_regions_src_holds_the_tags_and_ids_the_file_names(tmp_path):
    from contravia import regions as module

    osm = tmp_path / "tiny.osm"
    nodes = [(1, 23.10, -82.40), (2, 23.10, -82.39), (3, 23.11, -82.39), (4, 23.11, -82.40),
             (5, 23.20, -82.40), (6, 23.20, -82.39), (7, 23.21, -82.39)]
    osm.write_text("<?xml version='1.0' encoding='UTF-8'?>\n<osm version='0.6'>\n"
                   + "".join(f"<node id='{i}' version='1' lat='{la}' lon='{lo}'/>\n" for i, la, lo in nodes)
                   + "<way id='10' version='1'><nd ref='1'/><nd ref='2'/><nd ref='3'/><nd ref='4'/><nd ref='1'/>"
                   "<tag k='landuse' v='cemetery'/></way>\n"
                   "<way id='11' version='1'><nd ref='5'/><nd ref='6'/><nd ref='7'/><nd ref='5'/></way>\n"
                   "<relation id='20' version='1'><member type='way' ref='11' role='outer'/>"
                   "<tag k='type' v='multipolygon'/><tag k='landuse' v='grass'/></relation>\n</osm>\n")
    shapes, report = module.read(osm, ["osm:relation/20", "tag:landuse=cemetery"])
    assert set(shapes) == {"tag:landuse=cemetery", "osm:relation/20"} and report["osm:relation/20"]["found"] == 1
    assert 1000 * 1000 < shapes["tag:landuse=cemetery"].area < 1100 * 1200  # about 1 km by 1.1 km, in metres
    module.save(shapes, tmp_path / module.NAME)
    back = module.load_src(tmp_path / module.NAME)
    assert back.keys() == shapes.keys() and back["osm:relation/20"].equals(shapes["osm:relation/20"])


def test_a_dropped_park_grows_no_lawn(monkeypatch):
    from types import SimpleNamespace

    from contravia import grass
    from contravia import regions as module

    ring = np.array([(20, 20), (180, 20), (180, 180), (20, 180), (20, 20)], float)  # inside the cemetery's square
    a = {"park_verts": ring, "park_starts": np.array([0, len(ring)]), "park_kind": np.array([3]),
         "park_seed": np.array([7])}
    city = SimpleNamespace(cell=lambda i, j: SimpleNamespace(green_tris=None), provinces_of=lambda i, j: [0],
                           province=lambda p: SimpleNamespace(a=a))
    monkeypatch.setattr(module, "_current", Regions.parse(FILE, SRC))
    assert len(grass.lawns(city, 0, 0)) == 1
    monkeypatch.setattr(module, "_current", Regions.parse(CEMETERY, SRC))
    assert grass.lawns(city, 0, 0) == []
