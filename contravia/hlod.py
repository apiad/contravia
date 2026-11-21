"""Level-1 summaries that look like their cells (spec docs/superpowers/specs/2026-11-20-hlod-design.html, option D).

A level-1 node stands for four cells. Its summary draws merged blocks on a coarse ground coloured by the
satellite's land cover, so when it gives way to its cells the parks, the asphalt and the roofs change colour at
once. Here the node's builder paints, on the CPU, an image of its four cells seen from above: every triangle
that faces up, in its vertex colour, lowest first so roofs cover the ground. The summary's own up-facing
triangles (ground, roads, roofs) are marked SURFACE_BAKED and manifoldx's QuadStreamer draws the image on
them, through a layer of a texture array per drawn node. The walls keep their colours and facades. Nothing is
stored: the image is made with the node, in the worker, in about 50 ms.
"""

import numpy as np

from contravia.geo import CELL

LEVEL = 1  # the level whose summaries carry an image of their cells
SIZE = 512  # pixels a side: 0.78 m a pixel over a level-1 node's 400 m, about a screen pixel where it gives way
LAYERS = 96  # level-1 nodes drawn with their image at once; past it a node draws its summary's own colours
UP = 0.7  # a triangle faces up when its normal's y is over this
SUPERSAMPLE = 2  # the image is painted this many times larger and averaged down, so edges are smooth


def paint(meshes, side: float, n: int = SIZE, base: bool = False) -> np.ndarray:
    """An (n, n, 3) uint8 sRGB image of `meshes` seen from above over a square of `side` metres: each mesh is
    (MeshData, dx, dz), its positions relative to (dx, 0, dz) from the square's corner. Up-facing triangles
    are painted flat in their vertices' mean colour, lowest first; with base, the first mesh goes under all
    the others whatever its heights. x runs along the image's width and z down its height."""
    from PIL import Image, ImageDraw

    big = n * SUPERSAMPLE
    tris, colours, heights = [], [], []
    for k, (m, dx, dz) in enumerate(meshes):
        if m is None or not len(m.indices):
            continue
        idx = np.asarray(m.indices).reshape(-1, 3)
        up = np.asarray(m.normals)[idx][:, :, 1].mean(axis=1) > UP
        pos = np.asarray(m.positions, np.float64)[idx[up]] + (dx, 0.0, dz)
        tris.append(pos[:, :, [0, 2]] * (big / side))
        colours.append(np.asarray(m.colors)[idx[up]].mean(axis=1))
        heights.append(pos[:, :, 1].mean(axis=1) - (1e9 if base and k == 0 else 0.0))
    img = Image.new("RGB", (big, big), (0, 0, 0))
    if tris:
        tris, heights = np.concatenate(tris), np.concatenate(heights)
        c8 = np.clip(np.round(np.concatenate(colours) * 255), 0, 255).astype(int)
        draw = ImageDraw.Draw(img)
        for k in np.argsort(heights, kind="stable"):
            draw.polygon([tuple(p) for p in tris[k]], fill=tuple(c8[k]))
    return np.asarray(img.resize((n, n), Image.Resampling.BOX), np.uint8)


def baked(summary, cells, side: float, n: int = SIZE):
    """The node's summary MeshData with its up-facing vertices marked SURFACE_BAKED and, as its albedo, an
    image of the summary under its cells (cells as paint takes them): where no cell draws, such as the
    summary's own shore, the summary shows through."""
    from manifoldx.resources import SURFACE_BAKED

    if summary is None or not len(summary.indices):
        return summary
    surfaces = np.zeros(len(summary.positions), np.int32) if summary.surfaces is None else summary.surfaces.copy()
    surfaces[np.asarray(summary.normals)[:, 1] > UP] = SURFACE_BAKED
    image = paint([(summary, 0.0, 0.0), *cells], side, n, base=True)
    return summary._replace(surfaces=surfaces, albedo=image)


def node_cells(build, exists, L: int, a: int, b: int):
    """The cells under node (L, a, b) built with build(L, a, b) at level 0, as paint takes them."""
    out = []
    for da in range(1 << L):
        for db in range(1 << L):
            ca, cb = (a << L) + da, (b << L) + db
            if exists(0, ca, cb):
                out.append((build(0, ca, cb), da * CELL, db * CELL))
    return out
