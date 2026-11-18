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
