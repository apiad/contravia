"""The fleet's review sheet: one PNG per kind, a row per variant and four views of it, rendered offscreen.

A row is one variant of one family, painted in its first colourway: three-quarter, left flank, front and rear.
The camera is placed from the vehicle's own size, and each view is cropped to what it draws and fitted into its
tile, so a bicycle and a coach read at the same size on the page. The renders use manifoldx's studio environment:
with a sun alone the faces turned away from it read as holes, and chrome without an environment renders black.

`contravia fleet-sheet OUTDIR [--kind KIND]` writes `sheet_<kind>.png` for every kind, or for one.
"""

from __future__ import annotations

import math
import os

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial
from PIL import Image, ImageDraw, ImageFont

from contravia.rules import fleet

TILE = (320, 240)  # one view
LABEL = 170  # the name column before the four views
PAD = 14  # margin inside a tile
VIEWS = ("q34", "left", "front", "rear")
TITLE = 36

# Where the camera stands for each view, as a direction from the vehicle's middle, and the sun that lights it.
# The front is +X and the right flank +Z, so the left view stands at -Z.
CAMERAS = {
    "q34": (0.95, 0.50, -0.85),
    "left": (0.0, 0.35, -1.0),
    "front": (1.0, 0.30, 0.0),
    "rear": (-1.0, 0.30, 0.0),
}
SUNS = {
    "q34": (-0.4, -0.8, 0.45),
    "left": (-0.3, -0.75, 0.6),
    "front": (-0.6, -0.75, 0.25),
    "rear": (0.6, -0.75, 0.25),
}


def _font(size: int = 15):
    """The sheet's label font: Pillow's built-in bitmap face, at a size when the Pillow version takes one."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: the bitmap face has one size
        return ImageFont.load_default()


def _engine(name: str, width: int, height: int):
    """An offscreen engine with the sheet's sky, ground and studio light."""
    e = mx.Engine(name, width=width, height=height)
    e.background_color = (0.957, 0.937, 0.894)
    e.set_environment("studio")  # sky fill, so faces away from the sun are not black
    e.camera.fov = 22
    return e


def _distance(extent_x: float, extent_y: float, aspect: float) -> float:
    """How far the camera stands so a box of those half-extents fills most of the frame."""
    t = math.tan(math.radians(11.0))
    return 1.25 * max(extent_x / (t * aspect), extent_y / t)


def _crop(im: Image.Image, points: np.ndarray) -> Image.Image:
    """The part of a render the vehicle's projected points occupy, with a margin, never empty."""
    xs, ys = points[:, 0], points[:, 1]
    margin = max(12.0, 0.06 * max(float(np.ptp(xs)), float(np.ptp(ys))))
    x0, x1 = int(max(xs.min() - margin, 0)), int(min(xs.max() + margin, im.width))
    y0, y1 = int(max(ys.min() - margin, 0)), int(min(ys.max() + margin, im.height))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return im
    return im.crop((x0, y0, x1, y1))


def _fit(im: Image.Image) -> Image.Image:
    """A crop centred in a tile of the sheet's size, scaled to fit it."""
    tile = Image.new("RGB", TILE, (250, 248, 243))  # the studio sky
    k = min((TILE[0] - 2 * PAD) / im.width, (TILE[1] - 2 * PAD) / im.height)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    tile.paste(im, ((TILE[0] - im.width) // 2, (TILE[1] - im.height) // 2))
    return tile


def _views(vehicle, palette: dict, name: str) -> list[Image.Image]:
    """Four rendered tiles of one built vehicle, painted in one colourway."""
    mesh = vehicle.merged(palette=palette)
    points = np.asarray(mesh.positions, float)
    low, high = points.min(0), points.max(0)
    length, height, width = high[0] - low[0], high[1] - low[1], high[2] - low[2]
    target = np.array([0.0, 0.45 * height, 0.0])
    aspect = TILE[0] / TILE[1]
    engine = _engine(name, TILE[0] * 2, TILE[1] * 2)
    engine.spawn(Mesh(mesh.to_geometry()), Material(StandardMaterial("#ffffff", roughness=0.6, vertex_colors=True)),
                 Transform())
    tiles = []
    for view in VIEWS:
        if view == "left":  # level with the flank, so the profile reads without looking down on the roof
            at = target + np.array([0.0, 0.0, -_distance(length / 2, height / 2, aspect)])
        elif view == "front":
            at = target + np.array([_distance(width / 2, height / 2, aspect), 0.0, 0.0])
        elif view == "rear":
            at = target + np.array([-_distance(width / 2, height / 2, aspect), 0.0, 0.0])
        else:  # three-quarters, from a little above, as a person sees a car at the kerb
            direction = np.asarray(CAMERAS[view], float)
            at = target + direction / np.linalg.norm(direction) * _distance(0.35 * (length + width), height / 2, aspect)
        engine.camera.set_pose(position=tuple(at), target=tuple(target))
        engine.set_sun(DirectionalLight(color="#fff1d6", intensity=3.2, direction=SUNS[view]))
        im = Image.fromarray(engine.render_frame(supersample=2)).convert("RGB")
        seen, _depth = engine.camera.project(points, im.width, im.height)
        tiles.append(_fit(_crop(im, np.asarray(seen))))
    engine.close()
    return tiles


def sheet(kind: str, outdir: str) -> str:
    """Write one kind's sheet and return its path."""
    families = [family for family in fleet.FAMILIES.values() if family.kind == kind]
    if not families:
        raise KeyError(f"fleet-sheet: no family of kind {kind!r}; the kinds are {fleet.KINDS}")
    rows = [(family, k) for family in families for k in range(family.variants)]
    width = LABEL + TILE[0] * len(VIEWS)
    height = TITLE + TILE[1] * len(rows)
    page = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    font, small = _font(18), _font(13)
    draw.text((16, 9), f"fleet · {kind}", fill=(20, 22, 26), font=font)
    for row, (family, k) in enumerate(rows):
        top = TITLE + row * TILE[1]
        variant = fleet.variant(kind, family.name, k)
        tiles = _views(variant.build(), variant.colourways[0], f"{kind}-{family.name}-{k}")
        draw.text((16, top + TILE[1] // 2 - 16), family.name, fill=(20, 22, 26), font=font)
        draw.text((16, top + TILE[1] // 2 + 2), f"{variant.era} · {k}", fill=(96, 100, 104), font=small)
        for column, tile in enumerate(tiles):
            page.paste(tile, (LABEL + column * TILE[0], top))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"sheet_{kind}.png")
    page.save(path)
    return path


def main(outdir: str, kind: str | None = None) -> list[str]:
    """Render the review sheet of every kind, or of one, into OUTDIR."""
    paths = []
    for name in (kind,) if kind else fleet.KINDS:
        path = sheet(name, outdir)
        print(path)
        paths.append(path)
    return paths
