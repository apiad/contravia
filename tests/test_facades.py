import numpy as np

from caiman.buildingtags import KINDS
from caiman.rules.facades import TYPES, plan


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
