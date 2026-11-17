import numpy as np

from contravia.buildingtags import KINDS
from contravia.rules.facades import TYPES, plan


def attrs(kind="yes", levels=np.nan, height=np.nan, uses=0, muni=-1):
    return np.array([KINDS.index(kind), levels, height, uses, muni], np.float32)


def test_the_municipio_picks_the_type_of_an_untagged_building():
    assert plan(1, attrs(), "La Habana Vieja").type == "colonial"
    assert plan(1, attrs(), "Centro Habana").type == "eclectic"
    assert plan(1, attrs(), "Arroyo Naranjo").type == "suburban"
    assert plan(1, attrs(), "Habana del Este", area=600).type == "prefab"


def test_osm_kind_levels_and_height_come_first():
    assert plan(1, attrs("warehouse"), "Centro Habana").type == "warehouse"
    p = plan(1, attrs(levels=12), "La Habana Vieja")
    assert p.type == "tower" and p.storeys == 12
    q = plan(1, attrs(height=30.0), "Playa")
    assert q.type == "tower" and abs(q.roof - 30.0) < q.floor


def test_a_plan_is_seeded_and_within_its_type():
    for osm_id in range(50):
        p = plan(osm_id, attrs(), "Cerro")
        assert p == plan(osm_id, attrs(), "Cerro")
        t = TYPES[p.type]
        assert t.storeys[0] <= p.storeys <= t.storeys[1] and t.floor[0] <= p.floor <= t.floor[1]
        assert p.height == p.roof + t.parapet and all(0 <= c <= 1 for c in p.color)
    assert len({plan(k, attrs(), "Cerro").color for k in range(50)}) > 3


def test_portals_face_main_roads_only():
    from contravia.rules.facades import portal_walls

    ring = np.array([[0, 0], [20, 0], [20, 12], [0, 12]], float)  # its first edge runs along z = 0
    main = np.array([[-50, -9, 70, -9, 10.0]])  # a 10 m road whose edge is 4 m from the wall
    lane = np.array([[-50, -9, 70, -9, 6.0]])
    across = np.array([[10, -60, 10, -8, 10.0]])
    assert list(portal_walls(ring, main)) == [True, False, False, False]
    assert not portal_walls(ring, lane).any() and not portal_walls(ring, across).any()


def test_a_portal_stands_on_columns():
    from contravia.mesh import MeshBuilder
    from contravia.rules import buildings

    ring = np.array([[0, 0], [20, 0], [20, 12], [0, 12]], float)
    p = plan(3, attrs(levels=3), "Centro Habana")
    assert p.portal
    plain, portal = MeshBuilder(), MeshBuilder()
    buildings.box(plain, ring, 3, plan=p)
    buildings.box(portal, ring, 3, plan=p, portals=[True, False, False, False])
    a, b = plain.to_meshdata(0, 0), portal.to_meshdata(0, 0)
    assert len(b.positions) > len(a.positions) + 40  # columns and a ceiling
    low = b.positions[b.positions[:, 1] < p.ground - 0.01]
    assert low[:, 2].min() >= 0.0 and (np.abs(low[:, 2] - 3.5) < 1e-3).any()  # the shop fronts stand back
