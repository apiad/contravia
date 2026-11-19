from datetime import date

import numpy as np
import pytest
import shapely

from contravia import cover
from contravia.geo import CELL, to_local
from contravia.rules import cover as rules
from contravia.rules.ground import GROUND


def test_tiles_are_named_by_their_south_west_corner():
    assert cover.tile_name(23.13, -82.36) == "N21W084"
    assert cover.tile_name(20.0, -74.2) == "N18W075"


def test_paint_puts_a_tiles_classes_in_the_local_grid():
    img = np.full((300, 300), 40, np.uint8)  # 3 x 3 degrees of crops, from (21, -84)
    img[:150] = 10  # trees in the north half, rows go south
    xy = to_local(np.array([22.4, 22.6]), np.array([-82.5, -82.5]))  # either side of 22.5 north
    origin = np.floor(xy.min(axis=0) / cover.STEP) * cover.STEP - 1000
    codes = np.zeros((300, 300), np.uint8)
    cover._paint(codes, origin, img, 21, -84)
    grid = cover.Cover(codes, origin)
    south, north = grid.sample(xy[:, 0], xy[:, 1])
    assert (south, north) == (40, 10)
    assert grid.sample(np.array([origin[0] - 50.0]), np.array([origin[1]]))[0] == 0  # outside: no data


def _grid(codes, origin=(0.0, 0.0)):
    return cover.Cover(np.asarray(codes, np.uint8), origin)


def test_shares_count_the_classes_in_each_square_and_below_hands_them_down():
    codes = np.zeros((64, 64), np.uint8)
    codes[:, :32] = 10  # trees in the west half of the node, crops in the east
    codes[:, 32:] = 40
    L = 6  # 12 800 m a side: 128 squares of cover, of which the grid above covers 64
    grid = _grid(np.kron(codes, np.ones((2, 2), np.uint8)))
    s = cover.shares(grid, L, 0, 0, (0, 0))
    assert s.shape == (16, 16, len(cover.CLASSES))
    trees, crops = cover.CLASSES.index(10), cover.CLASSES.index(40)
    assert float(s[3, 2, trees]) == pytest.approx(1.0) and float(s[3, 12, crops]) == pytest.approx(1.0)
    kid = cover.below(s, (L, 0, 0), (L - 1, 1, 0))  # the north-east quarter: crops
    assert kid.shape == s.shape and float(kid[0, 0, crops]) == pytest.approx(1.0)
    assert np.array_equal(kid, s[np.ix_(np.arange(16) // 2, 8 + np.arange(16) // 2)])


def test_the_palette_mixes_by_share_and_no_data_is_ground():
    shares = np.zeros((2, len(cover.CLASSES)))
    shares[0, cover.CLASSES.index(40)] = 0.5
    shares[0, cover.CLASSES.index(60)] = 0.5
    c = rules.colours(shares, date(2026, 6, 1))
    assert np.allclose(c[0], (np.array(rules.TONES[40]) + rules.TONES[60]) / 2)
    assert np.allclose(c[1], GROUND)
    half_sea = np.zeros(len(cover.CLASSES))
    half_sea[cover.CLASSES.index(80)] = 0.5  # a square on the coast: half sea, half sand
    half_sea[cover.CLASSES.index(60)] = 0.5
    assert np.allclose(rules.colours(half_sea, date(2026, 6, 1)), rules.TONES[60])


def test_shade_darkens_slopes_away_from_the_light_from_high_levels_only():
    colour = np.ones((2, 3))
    toward = rules.LIGHT / np.linalg.norm(rules.LIGHT)
    away = toward * (-1, 1, -1)
    away /= np.linalg.norm(away)
    n = np.stack([toward, away])
    low = rules.shade(colour, n, rules.SHADE_FROM)
    high = rules.shade(colour, n, rules.SHADE_TO)
    assert np.allclose(low, 1.0)
    assert np.allclose(high[0], 1.0) and (high[1] < 0.9).all()


def test_a_summary_with_cover_draws_its_squares_in_the_palette():
    from contravia.summary import summarize, summary_mesh

    L, origin = 4, (0, 0)
    side = (1 << L) * CELL
    s = summarize([], L, 0, 0, origin)
    s["land"] = np.frombuffer(shapely.to_wkb(shapely.box(0, 0, side, side)), np.uint8)
    s["cover"] = np.zeros((16, 16, len(cover.CLASSES)), np.float16)
    s["cover"][..., cover.CLASSES.index(60)] = 1.0  # all sand
    m = summary_mesh(s, L, 0, 0, origin)
    up = np.asarray(m.normals)[:, 1] > 0.5
    assert np.allclose(np.asarray(m.colors)[up], rules.TONES[60], atol=1e-3)
