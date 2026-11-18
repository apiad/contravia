"""Props: the park motifs' meshes face out, sit where they belong and stop the car."""

import numpy as np
import shapely
from shapely.geometry import Point

from contravia import parks, props
from contravia.structures import PAINTS, STONE, WATER
from contravia.walls import Walls


def _normals(t):
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _solid_faces_out(build):
    """Build one motif round the origin and return the share of its faces whose normal points away from its axis
    or up: those that face in are the inside of a basin or the underside of a roof."""
    p = props.Props()
    build(p)
    t = np.array(p.tris)
    n = _normals(t)
    c = t.mean(axis=1)
    radial = np.einsum("ij,ij->i", n[:, [0, 2]], c[:, [0, 2]])
    return float(((radial > -1e-6) | (n[:, 1] > 0.5)).mean()), p


def test_boxes_and_prisms_face_out():
    for build in (lambda p: p.box((0, 0), 1.0, 0.5, 0, 2, 0.3, STONE),
                  lambda p: props.Props.prism(p, (0, 0), 2.0, 8, 0, 1, STONE),
                  lambda p: props.bust(p, (0, 0), 0.4, 0.0)):  # one-piece motifs: faces point away from their axis
        share, p = _solid_faces_out(build)
        assert share == 1.0
        assert all(k in PAINTS for k in p.paint)


def test_every_motif_builds_finite_triangles_and_known_paints():
    p = props.Props()
    props.glorieta(p, (0, 0), 4.0, 0.3, 0.0)
    props.fountain(p, (20, 0), 4.0, 0.0)
    props.bench(p, (40, 0), 1.0, 0.0)
    props.playground(p, (60, 0), 0.0, 0.0)
    props.ranchon(p, (90, 0), 0.5, 0.0)
    props.picnic(p, (100, 0), 0.5, 0.0)
    props.lamp(p, (110, 0), 0.0)
    props.boat(p, (120, 0), 0.0, 0.0, 2)
    t = np.array(p.tris)
    assert len(t) > 500 and np.isfinite(t).all()
    assert all(k in PAINTS for k in p.paint) and len(p.paint) == len(t)
    assert t[:, :, 1].max() == 6.5  # the glorieta's lantern


def test_tall_motifs_are_walls_and_low_ones_are_not():
    p = props.Props()
    props.bench(p, (0, 0), 0.0, 0.0)
    assert len(p.walls) == 4
    q = props.Props()
    q.box((0, 0), 1, 1, 0, 0.3, 0.0, STONE)  # under HIT: the car drives over it
    assert not q.walls


def test_a_lake_holds_its_water_under_its_rim():
    water = shapely.Point(0, 0).buffer(10, 16)
    p = props.Props()
    props.lake(p, water, 2.0)
    t, paint = np.array(p.tris), np.array(p.paint)
    w = t[paint == WATER]
    assert len(w) and np.allclose(w[:, :, 1], 2.0)
    assert shapely.contains_xy(water.buffer(0.01), w[:, :, 0].ravel(), w[:, :, 2].ravel()).all()
    assert np.isclose(t[paint == STONE][:, :, 1].max(), 2.4)
    walls = np.array(p.walls)
    mid = (walls[:, 0:2] + walls[:, 2:4]) / 2
    assert (np.hypot(*mid.T) > 10).all()  # the wall is the rim's outside: the car never reaches the water


def test_the_prepared_city_draws_its_parks_motifs_and_the_car_hits_them(city):
    assert "park_tris" in city.a and len(city.a["park_tris"])
    assert "park_walls" in city.a and len(city.a["park_walls"])
    laid = parks.laid(city.a)
    table = next((m for park, _ in laid for m in park.motifs if m[0] in ("picnic", "ranchon")), None)
    assert table is not None
    walls = Walls(city).near(table[1], table[2])
    segs = shapely.linestrings(np.asarray(walls, float).reshape(-1, 2, 2))
    assert (shapely.distance(segs, Point(table[1], table[2])) < 3.0).any()
