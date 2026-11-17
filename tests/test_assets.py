"""Every SVG in assets/ must draw through manifoldx's GUI rasterizer, which is how the game shows it."""

from pathlib import Path

import numpy as np
import pytest

from manifoldx.svg import rasterize_svg

ASSETS = Path(__file__).parent.parent / "assets"
SVGS = sorted(ASSETS.rglob("*.svg"))


def test_assets_exist():
    assert len(SVGS) >= 59, "run `uv run --with fonttools python assets/build.py`"


@pytest.mark.parametrize("path", SVGS, ids=lambda p: str(p.relative_to(ASSETS)))
def test_svg_draws(path):
    rgba = np.asarray(rasterize_svg(str(path), 64, 64))
    assert rgba.shape == (64, 64, 4)
    assert (rgba[..., 3] > 0).mean() > 0.05, "the drawing is (almost) empty"
