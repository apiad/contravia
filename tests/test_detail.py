"""The detail ring: cornices sit on the facade's floor lines, balconies face the street, budgets hold."""

import numpy as np

from caiman.mesh import MeshBuilder
from caiman.rules import detail, facades

RING = np.array([(0, 0), (24, 0), (24, 14), (0, 14)], float)
ROAD = np.array([(-50, -5, 80, -5, 12.0)])  # a 12 m street along the ring's z = 0 side


def _plan(name, storeys=3):
    p = facades.plan(4242, area=336.0)
    t = facades.TYPES[name]
    return facades.Plan(name, storeys, t.ground[0], t.floor[0], t.bay[0], t.parapet, p.color, p.roof_color,
                        t.portal)


def _dress(p, roads=ROAD):
    mb = MeshBuilder()
    detail.dress(mb, RING, 4242, 2.0, p, roads)
    return mb.to_meshdata(0, 0)


def test_floor_cornices_sit_on_the_painted_floor_lines():
    """The facade paints floor k at v = k, which stands ground + (k - 1) * floor over the base: every
    eclectic floor cornice starts there."""
    for storeys in (2, 3, 5):
        p = _plan("eclectic", storeys)
        m = _dress(p, roads=np.zeros((0, 5)))  # no street: only cornices, coping and tanks
        lines = [2.0 + p.ground + (k - 1) * p.floor for k in range(1, storeys)]
        bottoms = {round(float(y), 3) for y in m.positions[:, 1]}
        for y in lines:
            assert round(y, 3) in bottoms
        assert np.allclose(detail.floor_lines(p, 2.0), lines)


def test_balconies_only_on_the_street_front():
    """Colonial balconies (iron railings) stand on the wall along the street, not on the back."""
    m = _dress(_plan("colonial"))
    iron = np.all(np.isclose(m.colors, detail.IRON), axis=1)
    assert iron.any()
    assert m.positions[iron, 2].max() < 0.0  # every railing stands in front of the z = 0 wall


def test_budget_per_building():
    """A dressed three-storey eclectic block on a street stays under 2500 triangles."""
    for name in facades.TYPES:
        m = _dress(_plan(name))
        assert len(m.indices) // 3 < 2500, name
