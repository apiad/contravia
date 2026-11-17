"""caiman tiles: read the whole island's OSM once and write it as quadtree tiles.

The features are those of `caiman prepare` (roads, buildings, filler blocks, trees), with no province
clipping and no traffic graph. City blocks come from the road network chunk by chunk, 64 x 64 cells
(12.8 km) at a time with the roads of a 300 m margin around it, keeping the blocks whose representative
point lies in the chunk: no union of the whole island's roads is ever built. Green areas get scattered
trees only inside Havana's box; elsewhere only trees tagged in OSM are kept, until the trees layer.

Every cell with a feature gets a weight (a building 1, a road segment a quarter, anything else 0), the
quadtree splits on those weights, write_tiles writes the leaves, and the summaries are made per leaf from
its cells and pooled from there up to the root.
"""

import json
import time
from pathlib import Path

import numpy as np

from caiman.citydata import _centroids
from caiman.geo import CELL, CUBA_BOX, HAVANA_BOX, cell_of
from caiman.prepare import MAX_BLOCK, TREE_EVERY, WIDTHS, _local, _split, choose_fillers, read_osm
from caiman.quadtree import SPLIT, Tree
from caiman.summary import pool, summarize
from caiman.tiles import Tiles, _name, write_tiles

CHUNK = 64  # cells per side of a chunk whose blocks are found together
MARGIN = 300.0  # metres of roads around a chunk, so blocks on its edge close


def _features(raw):
    """Project the raw features: road segments, valid building rings and ids, trees."""
    from shapely.geometry import Polygon

    roads = np.vstack([_split(_local(pts), WIDTHS[hw]) for hw, pts in raw.roads]) if raw.roads else np.zeros((0, 5))
    rings, ids, skipped = [], [], 0
    for osm_id, pts in raw.buildings:
        ring = _local(pts)
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if len(np.unique(ring, axis=0)) < 3 or not Polygon(ring).is_valid:
            skipped += 1
            continue
        rings.append(ring)
        ids.append(osm_id)
    trees = [_local(raw.trees)] if raw.trees else []
    s, w, n, e = HAVANA_BOX
    for osm_id, ring in raw.green:
        if not all(s <= lat <= n and w <= lon <= e for lat, lon in ring):
            continue
        poly = Polygon(_local(ring))
        if not poly.is_valid or poly.area < TREE_EVERY:
            continue
        import shapely

        k = int(poly.area // TREE_EVERY)
        rng = np.random.default_rng(abs(int(osm_id)))
        x0, z0, x1, z1 = poly.bounds
        got = []
        while sum(len(g) for g in got) < k:
            p = rng.uniform((x0, z0), (x1, z1), size=(2 * k, 2))
            got.append(p[shapely.contains_xy(poly, p[:, 0], p[:, 1])])
        trees.append(np.vstack(got)[:k])
    trees = np.vstack(trees) if trees else np.zeros((0, 2))
    return roads, rings, np.array(ids, np.int64), trees, skipped


def _fillers(roads, cents, raw):
    """Filler blocks, found chunk by chunk from the road segments."""
    import shapely
    from shapely.geometry import MultiLineString, Polygon
    from shapely.ops import polygonize, unary_union

    no_build = [Polygon(_local(r)) for r in raw.not_built if len(r) >= 4]
    no_build = [p if p.is_valid else p.buffer(0) for p in no_build]
    nb_tree = shapely.STRtree(no_build)
    mids = (roads[:, 0:2] + roads[:, 2:4]) / 2
    side = CHUNK * CELL
    chunks = np.unique(np.floor(mids / side).astype(np.int64), axis=0)
    fillers, seeds = [], []
    for ci, cj in chunks.tolist():
        x0, z0 = ci * side, cj * side
        near = ((mids[:, 0] > x0 - MARGIN) & (mids[:, 0] < x0 + side + MARGIN)
                & (mids[:, 1] > z0 - MARGIN) & (mids[:, 1] < z0 + side + MARGIN))
        segs = roads[near]
        lines = unary_union(MultiLineString([((a, b), (c, d)) for a, b, c, d, _w in segs.tolist()]))
        blocks = []
        for blk in polygonize(lines):
            if blk.area >= MAX_BLOCK:
                continue
            p = blk.representative_point()
            if x0 <= p.x < x0 + side and z0 <= p.y < z0 + side:
                blocks.append(blk)
        inside = (cents[:, 0] > x0 - MARGIN) & (cents[:, 0] < x0 + side + MARGIN) & \
                 (cents[:, 1] > z0 - MARGIN) & (cents[:, 1] < z0 + side + MARGIN)
        f, s = choose_fillers(blocks, cents[inside], no_build, nb_tree)
        fillers += f
        seeds += s
    return fillers, np.array(seeds, np.int64)


def _summaries(out: Path, tiles: Tiles) -> int:
    """Summarize every leaf from its cells, pool up to the root and save every node above level 0."""
    tree, cells = tiles.tree, tiles.cells()
    made = {}
    for (L, a, b) in tree.leaves:
        inside = tree.cells_in(L, a, b, cells)
        made[(L, a, b)] = summarize([(i, j, tiles.cell(i, j)) for i, j in inside.tolist()], L, a, b, tree.origin)
    inner = sorted(set(tree.nodes()) - set(tree.leaves))
    for L, a, b in sorted(inner, key=lambda n: n[0]):
        kids = [made.get((L - 1, 2 * a + da, 2 * b + db)) for da in (0, 1) for db in (0, 1)]
        made[(L, a, b)] = pool(kids, L, a, b, tree.origin)
    for node, arrays in made.items():
        if node[0] > 0:
            np.savez(out / "summary" / _name(*node), **arrays)
    return sum(1 for n in made if n[0] > 0)


def main(pbf, out, split: float = SPLIT) -> dict:
    """Read the island, write its tiles to `out`, and print and return the report."""
    t0 = time.perf_counter()
    out = Path(out)
    raw = read_osm(pbf, boundary="", box=CUBA_BOX, keep_ways=False)
    roads, rings, ids, trees, skipped = _features(raw)
    cents = _centroids(rings)
    fillers, seeds = _fillers(roads, cents, raw)
    weights: dict = {}
    for keys, w in ((cell_of(cents), 1.0), (cell_of((roads[:, 0:2] + roads[:, 2:4]) / 2), 0.25),
                    (cell_of(_centroids(fillers)), 0.0), (cell_of(trees), 0.0)):
        for c in map(tuple, keys.tolist()):
            weights[c] = weights.get(c, 0.0) + w
    tree = Tree.build(weights, split)
    report = write_tiles(out, tree, roads, rings, ids, fillers, seeds, trees, {})
    report["summaries"] = _summaries(out, Tiles(out))
    report["buildings_skipped"] = skipped
    by_level: dict = {}
    for L, _a, _b in tree.leaves:
        by_level[L] = by_level.get(L, 0) + 1
    report["leaves_by_level"] = dict(sorted(by_level.items()))
    report["nodes"] = len(tree.nodes())
    report["mb"] = round(sum(f.stat().st_size for f in out.rglob("*.npz")) / 1e6, 1)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    with np.load(out / "index.npz") as f:
        arrays = {k: f[k] for k in f.files}
    arrays["report"] = np.array(json.dumps(report))
    np.savez(out / "index.npz", **arrays)
    print(json.dumps(report, indent=2))
    return report
