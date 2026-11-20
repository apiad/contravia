"""The batched boxes draw what the per-quad loops they replaced drew, vertex for vertex.

The reference functions below are the loops as they stood before batching (one MeshBuilder.add per wall band
and per summary box side); the tests hold the batched code to their output."""

import dataclasses

import numpy as np
import pytest

from contravia.buildingtags import KINDS
from contravia.citydata import Cell
from contravia.geo import CELL
from contravia.mesh import MeshBuilder, signed_area, triangulate
from contravia.rules import buildings, facades
from contravia.summary import BOX, GRID, _boxes, _facade, _get, summarize


def _old_box(mb, ring, osm_id, base=0.0, plan=None, portals=None) -> bool:
    ring = np.asarray(ring, float)
    p = plan or facades.plan(osm_id, area=abs(signed_area(ring)))
    color, low = p.color, tuple(c * buildings.GROUND_SHADE for c in p.color)
    ground, roof, top = base + min(p.ground, p.roof), base + p.roof, base + p.height
    kind, seed = facades.SHADER_TYPE[p.type], (abs(int(osm_id)) * 0.6180339887) % 1.0
    ccw = signed_area(ring) > 0
    for k in range(len(ring)):
        (x0, z0), (x1, z1) = ring[k], ring[(k + 1) % len(ring)]
        d = np.array([x1 - x0, z1 - z0])
        length = np.hypot(*d)
        if length < 1e-6:
            continue
        out = np.array([d[1], -d[0]]) / length
        if not ccw:
            out = -out
        n, inner = (out[0], 0.0, out[1]), (-out[0], 0.0, -out[1])
        bays = max(1, round(length / p.bay))
        if portals is not None and portals[k] and p.portal and roof > ground:
            buildings._portal(mb, np.array([x0, z0]), np.array([x1, z1]), out, base, ground, bays, color, kind,
                              seed)
        else:
            mb.add([(x0, base, z0), (x1, base, z1), (x1, ground, z1), (x0, ground, z0)], [n] * 4, color,
                   [(0, 1, 2), (0, 2, 3)], [(0, 0, kind, seed), (bays, 0, kind, seed), (bays, 1, kind, seed),
                                            (0, 1, kind, seed)])
        if roof > ground:
            up = p.storeys
            mb.add([(x0, ground, z0), (x1, ground, z1), (x1, roof, z1), (x0, roof, z0)], [n] * 4, color,
                   [(0, 1, 2), (0, 2, 3)], [(0, 1, kind, seed), (bays, 1, kind, seed), (bays, up, kind, seed),
                                            (0, up, kind, seed)])
        mb.add([(x0, roof, z0), (x1, roof, z1), (x1, top, z1), (x0, top, z0)], [n] * 4, color,
               [(0, 1, 2), (0, 2, 3)])
        mb.add([(x0, roof, z0), (x1, roof, z1), (x1, top, z1), (x0, top, z0)], [inner] * 4, low,
               [(0, 2, 1), (0, 3, 2)])
    try:
        tris = triangulate(ring)
    except ValueError:
        return False
    mb.add([(x, roof, z) for x, z in ring], [(0, 1, 0)] * len(ring), p.roof_color, tris)
    return True


def _old_boxes(mb, s, x0, z0, sq, node) -> None:
    built, height = s["built"], s["height"]
    color, storeys, types = _get(s, "color"), _get(s, "storeys"), _get(s, "types")
    rows, cols = np.nonzero(built > 0)
    cx, cz = x0 + (cols + 0.5) * sq, z0 + (rows + 0.5) * sq
    for r, c, x, z, y in zip(rows, cols, cx, cz, np.zeros(len(cx)), strict=True):
        w = min(np.sqrt(built[r, c]), 0.9 * sq) / 2
        top = y + height[r, c]
        tone = tuple(color[r, c]) if color[r, c].any() else BOX
        corners = [(x - w, z - w), (x + w, z - w), (x + w, z + w), (x - w, z + w)]
        facade = _facade(types[r, c], storeys[r, c], 2 * w, (*node, r, c))
        for k in range(4):
            (ax, az), (bx, bz) = corners[k], corners[(k + 1) % 4]
            n = (np.sign(bz - az), 0.0, -np.sign(bx - ax))
            mb.add([(ax, y, az), (bx, y, bz), (bx, top, bz), (ax, top, az)], [n] * 4, tone,
                   [(0, 1, 2), (0, 2, 3)], facade)
        roof = tuple(np.multiply(facades.ROOF, 0.95))
        mb.add([(cx_, top, cz_) for cx_, cz_ in corners], [(0, 1, 0)] * 4, roof, [(0, 1, 2), (0, 2, 3)])


def _same(old: MeshBuilder, new: MeshBuilder) -> None:
    a, b = old.to_meshdata(0, 0), new.to_meshdata(0, 0)
    for field in a._fields:
        x, y = getattr(a, field), getattr(b, field)
        assert (x is None) == (y is None), field
        if x is not None:
            assert np.array_equal(x, y), field


L_RING = np.array([[0, 0], [20, 0], [20, 10], [10, 10], [10, 20], [0, 20]], float)
PORTAL = facades.plan(3, np.array([KINDS.index("yes"), 3, np.nan, 0, -1], np.float32),
                      "Centro Habana")


@pytest.mark.parametrize("ring, plan, portals", [
    (L_RING, None, None),
    (L_RING[::-1], None, None),  # clockwise
    (np.array([[0, 0], [5, 0], [5, 0], [5, 8], [0, 8]], float), None, None),  # a repeated corner
    (L_RING[:4], dataclasses.replace(PORTAL, storeys=1), None),  # no floors over the ground floor
    (L_RING, PORTAL, [True, False, True, True, False, False]),
    (L_RING, PORTAL, [False, False, False, False, False, True]),
    (np.array([[0, 0], [1, 1], [2, 2]], float), None, None),  # the roof cannot be cut
])
def test_box_draws_what_the_per_wall_loop_drew(ring, plan, portals):
    old, new = MeshBuilder(), MeshBuilder()
    assert _old_box(old, ring, 4242, 1.5, plan, portals) == buildings.box(new, ring, 4242, 1.5, plan, portals)
    _same(old, new)


def test_summary_boxes_draw_what_the_per_box_loop_drew():
    rng = np.random.default_rng(7)
    rings, ids = [], rng.integers(1, 1 << 40, size=30)
    for x, z in rng.uniform(5, 190, size=(30, 2)):
        rings.append(np.array([(x, z), (x + 8, z), (x + 8, z + 10), (x, z + 10)], np.float32))
    cell = Cell(roads=np.zeros((0, 5), np.float32), buildings=rings, building_ids=ids,
                trees=np.zeros((0, 2), np.float32))
    s = summarize([(0, 0, cell)], 1, 0, 0, (0, 0))
    s["color"][np.nonzero(s["built"] > 0)[0][0], np.nonzero(s["built"] > 0)[1][0]] = 0  # one box in BOX grey
    sq = 2 * CELL / GRID
    old, new = MeshBuilder(), MeshBuilder()
    _old_boxes(old, s, 0.0, 0.0, sq, (1, 0, 0))
    _boxes(new, s, 0.0, 0.0, sq, None, (1, 0, 0))
    assert old._pos and len(new._pos) == 1
    _same(old, new)
