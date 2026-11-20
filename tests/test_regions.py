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
