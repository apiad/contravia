"""contravia streets-check: a built network reads 0, and each kind of thing on a street is counted."""

import numpy as np
import pytest

from contravia import streets_check
from contravia.citydata import CityData
from contravia.mesh import lattice
from contravia.prepare import WIDTHS
from contravia.streets import add_to
from contravia.streets_check import Check, Levels, count
from contravia.terrain import Terrain
from tests.test_structures import _arrays


def _city(terrain=None, roads=None):
    """The bridge over a street and the tunnel of tests/test_structures.py, built as prepare builds them."""
    city = CityData.from_features(np.zeros((0, 5)) if roads is None else roads, [], np.zeros(0), [], np.zeros(0),
                                  np.zeros((0, 2)), terrain or Terrain.flat())
    city.a.update(_arrays())
    add_to(city, WIDTHS)
    return city


@pytest.fixture(scope="module")
def built():
    return _city()


def _kinds(city, cells=None) -> dict:
    return {k: n for k, n in count(city, cells)[0].items() if n}


class _Changed:
    """A city with other ground, or with more rows of one kind of feature in the cell `at`."""

    def __init__(self, city, kind=None, rows=None, at=(0, 0), terrain=None):
        self._city, self._kind, self._rows, self._at = city, kind, rows, tuple(at)
        self.terrain = terrain or city.terrain

    def __getattr__(self, name):
        return getattr(self._city, name)

    def cell(self, i, j):
        from dataclasses import replace

        c = self._city.cell(i, j)
        if self._kind is None or (i, j) != self._at:
            return c
        old = getattr(c, self._kind)
        rows = self._rows if old is None else np.concatenate([np.asarray(old, self._rows.dtype), self._rows])
        return replace(c, **{self._kind: rows})


def _with(city, kind: str, rows: np.ndarray, at=(0, 0)):
    return _Changed(city, kind, rows, at)


def test_lattice_points_lie_on_their_triangles_corners_included():
    tri = np.array([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 4.0]]])
    pts, own = lattice(tri, 1.0)
    assert len(pts) == 28 and (own == 0).all()               # its longest side, 5.7 m, is cut in 6
    assert (pts.min(axis=0) == 0).all() and (pts[:, 0] + pts[:, 2] <= 4 + 1e-9).all()


def test_levels_gives_every_street_under_a_point(built):
    every = np.concatenate([built.cell(i, j).street_tris for i in (-1, 0) for j in (-1, 0)])
    _, h = Levels(every).under(np.array([[0.5, 0.5]]))     # the bridge crosses the street here
    assert h.max() - h.min() > 5.0


def test_a_built_network_with_a_bridge_and_a_tunnel_reads_zero(built):
    assert _kinds(built) == {}


def test_ground_over_a_street_is_counted():
    hump = np.zeros((201, 201), np.float32)                  # wide: a relief fades to 0 near its grid's edge
    hump[100, 85:89] = 0.6                                   # raised after the streets were laid on flat ground
    city = _Changed(_city(), terrain=Terrain(hump, (-2000.0, -2000.0), (20.0, 20.0)))
    found = [f for f in count(city)[1] if f[0] == "ground"]
    assert found and all(-330 < f[1] < -210 and abs(f[2]) < 10 for f in found)
    assert max(f[3] for f in found) > 0.3


def test_a_raised_triangle_is_a_step_and_an_upright_one_is_steep(built):
    tris = np.array(built.cell(-1, -1).street_tris, float)
    k = int(np.argmin(np.hypot(tris[:, :, 0].mean(axis=1) + 150, tris[:, :, 2].mean(axis=1))))
    lifted = tris[k : k + 1].copy()
    lifted[:, :, 1] += 0.4
    assert _kinds(_with(built, "street_tris", lifted, (-1, -1))) == {"step": 2}  # it and the one under it
    wall = tris[k : k + 1].copy()
    wall[0, 0, 1] += 3.0
    kinds = _kinds(_with(built, "street_tris", wall, (-1, -1)))
    assert kinds["steep"] == 1


def test_a_wall_across_a_street_is_counted_by_its_source_and_one_along_the_kerb_is_not(built, monkeypatch):
    across = np.array([[[-150.0, -8.0], [-150.0, 8.0]]], np.float32)
    beside = np.array([[[-180.0, 8.0], [-120.0, 8.0]]], np.float32)

    def parts(city, landmarks, i, j):
        on = (i, j) == (-1, 0)
        return {"tree": across if on else across[:0], "building": beside if on else beside[:0]}

    monkeypatch.setattr(streets_check, "parts", parts)
    kinds, found = count(built)
    assert {k: n for k, n in kinds.items() if n} == {"tree": 1}
    assert 3.0 < found[0][3] < 12.0                          # the metres of it inside the street


def test_a_parapet_across_a_deck_is_counted_only_at_the_deck_s_height(built):
    top = float(np.asarray(built.cell(0, 0).street_tris)[:, :, 1].max())
    at_deck = np.array([[0.0, -3.0, 0.0, 3.0, top - 0.5, top + 1.1]], np.float32)
    assert _kinds(_with(built, "struct_walls", at_deck)) == {"parapet": 1}
    high = at_deck + np.array([0, 0, 0, 0, 5.0, 5.0], np.float32)
    assert _kinds(_with(built, "struct_walls", high)) == {}


def test_a_bench_on_the_asphalt_is_counted_and_a_lamp_arm_over_it_is_not(built):
    bench = np.array([[[-150.0, 0.5, -1.0], [-149.0, 0.5, -1.0], [-150.0, 0.5, 1.0]]], np.float32)
    assert _kinds(_with(built, "park_tris", bench, (-1, -1))) == {"motif": 1}
    assert _kinds(_with(built, "park_tris", bench + np.array([0, 5.0, 0], np.float32), (-1, -1))) == {}


def test_a_pavement_lying_on_the_street_is_counted_and_one_along_its_edge_is_not(built):
    tris = np.array(built.cell(-1, -1).street_tris, float)
    half = float(np.abs(tris[np.abs(tris[:, :, 0].mean(axis=1) + 150) < 20][:, :, 2]).max())
    lying = np.array([[[-150.0, 0.15, -1.0], [-146.0, 0.15, -1.0], [-150.0, 0.15, -half - 2]]], np.float32)
    assert _kinds(_with(built, "walk_tris", lying, (-1, -1))) == {"walk": 1}
    edge = np.array([[[-150.0, 0.15, -half - 0.1], [-146.0, 0.15, -half - 0.1], [-150.0, 0.15, -half - 2]]], np.float32)
    assert _kinds(_with(built, "walk_tris", edge, (-1, -1))) == {}


def test_only_the_cells_asked_for_are_checked(built):
    check = Check(built)
    assert check.cell(40, 40) == []


def _twin_bridge():
    """A bridge OSM maps as two one-way carriageways 7 m apart, over a street, each going on as its own street."""
    from tests.graphs import make_graph

    j = [(-300, 3.5), (-40, 3.5), (40, 3.5), (300, 3.5), (300, -3.5), (40, -3.5), (-40, -3.5), (-300, -3.5),
         (0, -300), (0, 300)]
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 8)]
    a = dict(make_graph(j, edges).a)
    a["g_layer"] = np.array([0, 1, 0, 0, 1, 0, 0, 0], np.int8)
    city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), Terrain.flat())
    city.a.update(a)
    add_to(city, WIDTHS)
    return city


def test_the_two_carriageways_of_a_bridge_are_one_deck_with_no_parapet_between_them():
    city = _twin_bridge()
    assert _kinds(city) == {}
    walls = np.concatenate([w for i in (-1, 0) for j in (-1, 0) if (w := city.cell(i, j).struct_walls) is not None])
    mid = (walls[:, 0:2] + walls[:, 2:4]) / 2
    on_deck = (np.abs(mid[:, 0]) < 30) & (walls[:, 5] > 5.0)       # parapets, not the piers under the deck
    assert on_deck.any() and (np.abs(mid[on_deck, 1]) > 5.0).all()  # only along the deck's two outer edges


def test_streets_laid_on_rough_ground_are_raised_clear_of_every_crease_of_it():
    from contravia.streets import build, mesh
    from tests.graphs import make_graph

    rng = np.random.default_rng(3)
    rough = Terrain(rng.uniform(20.0, 23.0, (61, 61)).astype(np.float32), (-900.0, -900.0), (30.0, 30.0))
    j = [(-300, 0), (0, 0), (300, 0), (0, -300), (0, 300), (250, 280)]
    pairs = [(0, 1), (1, 2), (3, 1), (1, 4), (4, 5), (5, 2)]
    a = dict(make_graph(j, [e for f, t in pairs for e in ((f, t), (t, f))]).a)
    keep = {}
    mesh(*build(a, WIDTHS), rough, keep)
    assert keep["raised"].vals.max() > 0.05                 # some crease stood over the streets as first laid
    city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), rough)
    city.a.update(a)
    add_to(city, WIDTHS)
    assert _kinds(city).get("ground", 0) == 0


def test_no_verge_is_laid_across_a_street_where_it_climbs_off_the_ground():
    import shapely

    from contravia import walks

    city = _city(roads=np.array([[-300.0, 0.0, 300.0, 0.0, 7.5]]))
    assert walks.add_to(city)["walk_triangles"] > 0          # the avenue climbs to its bridge: its edges get verges
    cells = [(i, j) for i in (-1, 0) for j in (-1, 0)]
    streets = np.concatenate([np.asarray(city.cell(*c).street_tris, float) for c in cells])
    high = streets[streets[:, :, 1].max(axis=1) >= walks.OFF_GROUND]   # the ramp from 3 m up and the deck
    deck = shapely.union_all(shapely.polygons(high[:, :, [0, 2]]))
    verges = np.concatenate([np.asarray(city.cell(*c).walk_tris, float) for c in cells])
    assert shapely.area(shapely.intersection(deck, shapely.polygons(verges[:, :, [0, 2]]))).sum() < 0.01
    assert _kinds(city).get("walk", 0) == 0


def test_an_avenues_median_wall_is_not_counted_and_the_same_wall_of_a_bridge_is(built):
    """Boyeros' wall between its carriageways (contravia/medians.py) is no part of the street's surface; the same
    triangles and walls in a bridge's concrete are a wall along the middle of the street."""
    from dataclasses import replace

    from contravia.structures import BARRIER, CONCRETE

    x0, x1, z0, z1, top = -180.0, -120.0, 0.5, 1.0, 1.0
    tris = np.array([[[x0, top, z0], [x1, top, z0], [x1, top, z1]], [[x0, top, z0], [x1, top, z1], [x0, top, z1]]],
                    np.float32)
    walls = np.array([[x0, z0, x1, z0, 0.0, top], [x0, z1, x1, z1, 0.0, top]], np.float32)

    class Walled:
        def __init__(self, paint):
            self.paint = paint

        def __getattr__(self, name):
            return getattr(built, name)

        def cell(self, i, j):
            c = built.cell(i, j)
            if (i, j) != (-1, 0):
                return c
            return replace(c, struct_tris=np.concatenate([np.asarray(c.struct_tris, np.float32), tris]),
                           struct_paint=np.concatenate([np.asarray(c.struct_paint, np.uint8), [self.paint] * 2]),
                           struct_walls=np.concatenate([np.asarray(c.struct_walls, np.float32), walls]))

    found = _kinds(Walled(CONCRETE))
    assert found.get("structure") and found.get("parapet")
    assert _kinds(Walled(BARRIER)) == {}


def _scan(city, kept):
    """streets_check._run over the city's cells as a worker runs it: (findings, cells checked again)."""
    streets_check._check, streets_check._kept = Check(city), kept
    found, again, modules = streets_check._run(sorted(map(tuple, np.asarray(city.cells()).reshape(-1, 2).tolist())))
    assert "contravia.streets_check" in modules and "contravia.streets" not in modules
    return found, again


def test_a_second_run_checks_again_only_the_cells_whose_data_changed(built):
    found, first = _scan(built, {})
    assert first and all(len(mark) == 16 for mark, _ in first.values())
    again, second = _scan(built, first)
    assert second == {} and again == found                    # nothing changed: every cell is kept
    bench = np.array([[[-150.0, 0.5, -1.0], [-149.0, 0.5, -1.0], [-150.0, 0.5, 1.0]]], np.float32)
    changed, third = _scan(_with(built, "park_tris", bench, (-1, -1)), first)
    assert sorted(f[0] for f in changed) == sorted([f[0] for f in found] + ["motif"])
    near = {c for c in first if max(abs(c[0] + 1), abs(c[1] + 1)) <= streets_check.AROUND}
    assert set(third) == near and len(near) < len(first)      # only the cells that read the changed one


def test_what_a_run_keeps_holds_while_the_island_and_the_code_do(tmp_path, monkeypatch):
    np.savez(tmp_path / "a.npz", heights=np.zeros((2, 2), np.float32), heights_origin=np.zeros(2), heights_step=np.ones(2))
    (tmp_path / "coast.src").write_bytes(b"coast")
    island = streets_check.island_stamp(tmp_path)
    cells = {(0, 0): (b"m" * 16, [("step", 1.0, 2.0, 0.3)])}
    streets_check.save_kept(tmp_path, island, ["contravia.streets_check", "contravia.walls"], cells)
    assert streets_check.load_kept(tmp_path, island) == cells
    (tmp_path / "coast.src").write_bytes(b"another coast")       # a file every cell reads changed
    assert streets_check.island_stamp(tmp_path) != island
    assert streets_check.load_kept(tmp_path, streets_check.island_stamp(tmp_path)) == {}
    np.savez(tmp_path / "a.npz", heights=np.ones((2, 2), np.float32), heights_origin=np.zeros(2), heights_step=np.ones(2))
    (tmp_path / "coast.src").write_bytes(b"coast")
    assert streets_check.island_stamp(tmp_path) != island           # the relief changed
    monkeypatch.setattr(streets_check, "code_stamp", lambda modules: "other code")
    assert streets_check.load_kept(tmp_path, island) == {}          # a module the run loaded changed


def test_a_quick_run_checks_again_only_the_cells_with_findings_and_those_that_changed(built, monkeypatch):
    bench = np.array([[[-150.0, 0.5, -1.0], [-149.0, 0.5, -1.0], [-150.0, 0.5, 1.0]]], np.float32)
    city = _with(built, "park_tris", bench, (-1, -1))
    found, first = _scan(city, {})
    dirty = {c for c, (_, f) in first.items() if f}
    assert dirty and len(dirty) < len(first) and [f[0] for f in found] == ["motif"]
    monkeypatch.setattr(streets_check, "_quick", True)
    again, second = _scan(city, first)
    assert set(second) == dirty and again == found            # the clean cells are taken as the last run left them
    fixed, third = _scan(built, first)                        # the bench gone: the cells that read it changed
    near = {c for c in first if max(abs(c[0] + 1), abs(c[1] + 1)) <= streets_check.AROUND}
    assert fixed == [] and set(third) == near


def test_a_quick_run_reads_the_newer_of_the_two_kept_files_whatever_code_wrote_it(tmp_path, monkeypatch):
    import os

    full = {(0, 0): (b"m" * 16, [("step", 1.0, 2.0, 0.3)])}
    quick = {(0, 0): (b"m" * 16, [])}
    assert streets_check.load_quick(tmp_path) == {}
    streets_check.save_kept(tmp_path, b"island", ["contravia.streets_check"], full)
    monkeypatch.setattr(streets_check, "code_stamp", lambda modules: "other code")
    assert streets_check.load_kept(tmp_path, b"island") == {} and streets_check.load_quick(tmp_path) == full
    streets_check.save_kept(tmp_path, b"island", ["contravia.streets_check"], quick, streets_check.QUICK)
    os.utime(tmp_path / streets_check.KEPT, (1, 1))
    assert streets_check.load_quick(tmp_path) == quick
    os.utime(tmp_path / streets_check.QUICK, (0, 0))
    assert streets_check.load_quick(tmp_path) == full          # a full run since: it is the one to start from
