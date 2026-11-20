"""Pavements: a dense block rises kerb to kerb over its streets; elsewhere buildings get a platform; wheels stand on both."""

import numpy as np
import shapely
from shapely.geometry import LineString

from contravia import walks
from contravia.surface import Surface
from contravia.terrain import Terrain

STREET = 10.0


def _arrays(buildings):
    """Prepared arrays for one 100 m block between four 10 m streets at height 0, in cell (0, 0)."""
    lines = [((-20, 0), (120, 0)), ((-20, 100), (120, 100)), ((0, -20), (0, 120)), ((100, -20), (100, 120))]
    roads = np.array([(*a, *b, STREET) for a, b in lines], np.float32)
    street = shapely.union_all([LineString(line).buffer(STREET / 2, cap_style="flat") for line in lines])
    t = shapely.get_coordinates(shapely.constrained_delaunay_triangles(street)).reshape(-1, 4, 2)[:, :3]
    tris = np.stack([t[:, :, 0], np.zeros(t.shape[:2]), t[:, :, 1]], axis=2).astype(np.float32)
    verts = np.concatenate(buildings).astype(np.float32)
    starts = np.concatenate([[0], np.cumsum([len(b) for b in buildings])])
    return {"roads": roads, "street_tris": tris, "street_tris_offsets": np.array([0, len(tris)]),
            "cells": np.array([[0, 0]]), "building_verts": verts, "building_starts": starts}


def _box(x, z, s):
    return np.array([(x, z), (x + s, z), (x + s, z + s), (x, z + s)], float)


def _normals(t):
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def test_a_dense_block_rises_kerb_to_kerb():
    a = _arrays([_box(x, z, 38) for x in (8, 52) for z in (8, 52)])
    tris, paint, keys, report = walks.build(a, Terrain.flat())
    assert report["dense_blocks"] == 1 and report["aprons"] == 0
    top = tris[paint % walks.PARTS == walks.TOP]
    assert len(top) and np.allclose(top[:, :, 1], walks.KERB)
    assert (_normals(top)[:, 1] > 0.99).all()
    xz = top[:, :, [0, 2]].reshape(-1, 2)
    assert xz.min() >= STREET / 2 - 0.05 and xz.max() <= 100 - STREET / 2 + 0.05  # kerb to kerb, off the street
    kerb = tris[paint % walks.PARTS == walks.KERB_FACE]
    out = kerb.mean(axis=1)[:, [0, 2]] - 50  # the block's centre is (50, 50)
    n = _normals(kerb)[:, [0, 2]]
    assert (np.einsum("ij,ij->i", n, out) > 0).all()  # every kerb faces the street


def test_a_lone_building_gets_a_platform_and_a_path():
    a = _arrays([_box(15, 40, 10)])  # 10 m from the street along x = 0
    tris, paint, keys, report = walks.build(a, Terrain.flat())
    assert report["dense_blocks"] == 0 and report["aprons"] == 1
    top = tris[paint % walks.PARTS == walks.APRON_TOP]
    xz = top[:, :, [0, 2]].reshape(-1, 2)
    assert abs(xz[:, 0].min() - STREET / 2) < 0.1  # the path runs from the platform to the street's edge


def test_wheels_stand_on_a_platform(city):
    tris, paint = city.a["walk_tris"], city.a["walk_paint"]
    top = tris[paint % walks.PARTS == walks.APRON_TOP]
    assert len(top)
    c = top[0].mean(axis=0)
    s = Surface(city, city.terrain)
    assert abs(float(s.height(c[0], c[2])[0]) - c[1]) < 1e-3


def test_a_park_block_is_paved_by_its_layout_and_its_lawn_left_bare():
    from contravia import parks

    a = _arrays([_box(15, 40, 1)])  # one shed: the block is almost empty
    ring = _box(5, 5, 90)  # the block, inside the streets' kerbs
    a.update(park_verts=ring.astype(np.float32), park_starts=np.array([0, 4]), park_kind=np.array([parks.CITY], np.uint8),
             park_zone=np.array([1], np.uint8), park_seed=np.array([parks.seed_of(shapely.Polygon(ring))]))
    tris, paint, _, report = walks.build(a, Terrain.flat())
    assert report["paved_parks"] == 1 and report["dense_blocks"] == 0
    part = paint % walks.PARTS
    path = tris[part == walks.PATH_TOP]
    assert len(path) and np.allclose(path[:, :, 1], walks.PATH_LIFT, atol=1e-3)
    kerb = tris[part == walks.TOP]
    c = kerb[:, :, [0, 2]].mean(axis=1)
    inside = (c > 5 + parks.KERB_W + 0.5).all(axis=1) & (c < 95 - parks.KERB_W - 0.5).all(axis=1)
    assert len(kerb) and not inside.any()  # the raised pavement is only the ring round the park


def _surface(a, tris, paint, terrain):
    """A Surface over the arrays' streets and the walks just built, in cell (0, 0)."""
    from contravia.citydata import CityData

    city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), terrain)
    city.add_kind("street_tris", a["street_tris"], np.zeros((len(a["street_tris"]), 2), int))
    city.add_kind("street_edges", np.zeros((0, 2, 3)), np.zeros((0, 2), int))
    city.add_kind("walk_tris", tris, np.zeros((len(tris), 2), int))
    city.add_kind("walk_paint", paint, np.zeros((len(tris), 2), int))
    return Surface(city, terrain)


def test_a_raised_street_edge_slopes_down_to_the_ground_over_a_verge():
    a = _arrays([_box(40, 40, 10)])  # one small building: not a dense block
    a["street_tris"][:, :, 1] = 0.4  # the streets stand 0.4 m over flat ground
    tris, paint, _, report = walks.build(a, Terrain.flat())
    assert report["verge_triangles"] and (paint == walks.SHOULDER_CODE).any() and (paint == walks.SLOPE_CODE).any()
    s = _surface(a, tris, paint, Terrain.flat())
    x = np.arange(4.0, 5.0 + walks.SHOULDER + walks.RUN + 1.0, 0.05)  # off the street along x = 0, at z = 30
    y = s.height(x, np.full(len(x), 30.0))
    assert np.allclose(y[x < 5.0], 0.4, atol=1e-3)                                 # the street
    assert np.allclose(y[(x > 5.02) & (x < 5.0 + walks.SHOULDER - 0.02)], 0.4, atol=1e-3)   # the shoulder
    assert np.abs(y[-1]) < 1e-3                                                     # back on the ground
    assert np.abs(np.diff(y)).max() < 0.4 / walks.RUN * 0.05 + 1e-3                  # no step anywhere


def test_no_verge_leads_into_a_dense_block():
    a = _arrays([_box(x, z, 38) for x in (8, 52) for z in (8, 52)])
    a["street_tris"][:, :, 1] = 0.4
    tris, paint, _, report = walks.build(a, Terrain.flat())
    verge = tris[paint >= walks.SHOULDER_CODE][:, :, [0, 2]].mean(axis=1)
    inside = (verge > walks.SAMPLE / 2).all(axis=1) & (verge < 100 - walks.SAMPLE / 2).all(axis=1)
    assert report["dense_blocks"] == 1 and not inside.any()


def test_a_wheel_rolls_up_a_bevelled_kerb():
    a = _arrays([_box(x, z, 38) for x in (8, 52) for z in (8, 52)])
    tris, paint, _, _ = walks.build(a, Terrain.flat())
    s = _surface(a, tris, paint, Terrain.flat())
    x = np.arange(3.0, 7.0, 0.01)  # across the kerb of the street along x = 0, at z = 30
    y = s.height(x, np.full(len(x), 30.0))
    assert abs(y[0]) < 1e-3 and abs(y[-1] - walks.KERB) < 1e-3
    assert np.diff(y).min() > -1e-4 and np.diff(y).max() < 0.01  # a ramp no steeper than 1:1, not a step


def test_a_seam_between_a_junction_and_its_street_raises_no_strip():
    """The street surface is a junction's plate and its streets' ribbons, triangulated apart: where they meet, their
    edges can miss each other by a millimetre. That seam is not land, so no pavement rises across the street there."""
    a = _arrays([_box(x, z, 38) for x in (8, 52) for z in (8, 52)])
    plates = [shapely.box(x - STREET / 2, z - STREET / 2, x + STREET / 2, z + STREET / 2) for x in (0, 100) for z in (0, 100)]
    ribbons = shapely.difference(shapely.unary_union([LineString(((-20, z), (120, z))).buffer(STREET / 2, cap_style="flat")
                                                      for z in (0, 100)] +
                                                     [LineString(((x, -20), (x, 120))).buffer(STREET / 2, cap_style="flat")
                                                      for x in (0, 100)]),
                                 shapely.unary_union([p.buffer(0.002, join_style="mitre") for p in plates]))
    t = shapely.get_coordinates(shapely.constrained_delaunay_triangles(
        shapely.GeometryCollection([*plates, ribbons]))).reshape(-1, 4, 2)[:, :3]
    a["street_tris"] = np.stack([t[:, :, 0], np.zeros(t.shape[:2]), t[:, :, 1]], axis=2).astype(np.float32)
    a["street_tris_offsets"] = np.array([0, len(t)])
    tris, paint, _, report = walks.build(a, Terrain.flat())
    assert report["dense_blocks"] == 1
    s = _surface(a, tris, paint, Terrain.flat())
    z = np.arange(80.0, 120.0, 0.05)  # straight through the junction at (0, 100), in the lane at x = 2
    y = s.height(np.full(len(z), 2.0), z)
    assert np.abs(np.diff(y)).max() < 0.005 and np.abs(y).max() < 0.005
