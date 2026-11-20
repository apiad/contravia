"""The trees the game draws: one row per tree in each open cell, drawn as GPU instances by level of detail.

A cell's mesh holds no tree. When the city's streamer draws a cell in full (a level-0 node), the forest grows
its trees on a thread (rules.trees.cell_trees: position, species, size, turn, season shift and tone, the same
trees the car hits) and spawns every tree as instances of its species' meshes (rules.treegrammar), one entity
per part: wood, leaves, accent. Every tree of one species, variant, crown and level of detail shares a geometry
and the tinted material, so the engine draws each such group in one instanced call, and each instance's Tint
carries its leaves' colour for the game's date, times its tone. Every few frames the forest measures each tree's
distance to the camera and moves the trees that crossed a threshold to their new level of detail.

Trees do not pop: the material is faded, a tree that comes fades in over FADE seconds, one that goes fades out
before it is destroyed, and a tree that changes level keeps its old level that long, fading out over the
complementary pixels of the new one (manifoldx's Fade), as the city's blocks do.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter

import numpy as np

from contravia.clearance import roads_around
from contravia.grass import yard_trees
from contravia.rules import trees
from contravia.rules.species import CROWNS, SPECIES, season
from contravia.rules.treegrammar import LODS, PARTS, VARIANTS, meshes

LOD_AT = (60.0, 160.0)  # metres from the camera where a tree drops to LOD 1, then to LOD 2
HYSTERESIS = 8.0  # metres a tree goes past a threshold before it changes level, so it does not flicker
EVERY = 6  # frames between two checks of the trees' levels of detail
LINGER = 60  # frames a cell's trees stay after the streamer stops drawing it in full: it often comes back at once
KEEP = 512  # cells whose rows are kept after they close, so a cell that opens again needs no growing
FADE = 0.3  # seconds a tree takes to fade in or out, and an old level of detail to give way to the new one


def linear(rgb) -> np.ndarray:
    """sRGB colours in [0, 1] as the linear light the shader multiplies by (a Tint is linear)."""
    c = np.asarray(rgb, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def levels(d, old=None) -> np.ndarray:
    """The level of detail of trees d metres from the camera; given their old levels, a tree keeps its own
    unless it is more than HYSTERESIS metres past the threshold."""
    d = np.asarray(d, float)
    new = np.searchsorted(LOD_AT, d).astype(np.int8)
    if old is None:
        return new
    near, far = np.searchsorted(LOD_AT, d - HYSTERESIS), np.searchsorted(LOD_AT, d + HYSTERESIS)
    keep = (old >= near) & (old <= far)
    return np.where(keep, old, new).astype(np.int8)


def warm(species) -> None:
    """Grow every mesh a tree of each species may be drawn with (its levels of detail, crowns and near
    variants), so the main thread finds them made: a species' meshes take about a second to grow."""
    for s in species:
        for lod in range(LODS):
            for crown in CROWNS:
                for v in range(VARIANTS if lod == 0 else 1):
                    meshes(SPECIES[int(s)], lod, crown, v)


class Stand:
    """The trees of one cell as rows: (Trees) t, and per tree its variant, crown, transform, tints per part,
    level of detail and the entity of each part (-1 for none)."""

    def __init__(self, t, day):
        self.t = t
        n = len(t)
        seeds = trees.point_seeds(t.xz)
        self.variant = np.minimum((trees.unit(seeds, 8) * VARIANTS).astype(int), VARIANTS - 1)
        width, height = 0.92 + 0.16 * trees.unit(seeds, 6), 0.9 + 0.2 * trees.unit(seeds, 7)
        self.pos = np.stack([t.xz[:, 0], t.y, t.xz[:, 1]], axis=1) if n else np.zeros((0, 3))
        half = -t.yaw / 2  # the old templates turned by -yaw about y; the trees keep their turn
        self.rot = np.stack([np.zeros(n), np.sin(half), np.zeros(n), np.cos(half)], axis=1)
        self.scale = t.scale[:, None] * np.stack([width, height, width], axis=1)
        self.lod = np.full(n, -1, np.int8)
        # a ball round the trees, so place() can skip a stand whose trees all stay in one level
        self.centre = self.pos.mean(axis=0) + (0.0, 5.0, 0.0) if n else np.zeros(3)
        self.reach = float(np.linalg.norm(self.pos + (0.0, 5.0, 0.0) - self.centre, axis=1).max()) if n else 0.0
        self.settled = -1  # the level every tree has, or -1 while they differ
        self.ents = np.full((n, len(PARTS)), -1, np.int64)
        self.dress(day)

    def dress(self, day) -> None:
        """Each tree's crown and tints on day: its leaves and accent in its species' colours for the season,
        times its tone; its wood untinted."""
        t, n = self.t, len(self.t)
        self.crown = np.empty(n, object)
        self.tint = np.ones((n, len(PARTS), 3))
        tone = 0.85 + 0.3 * t.tone
        looks = {}
        for k in range(n):
            key = (int(t.species[k]), int(t.shift[k]))
            if key not in looks:
                looks[key] = season(SPECIES[key[0]], day, key[1])
            look = looks[key]
            self.crown[k] = look.crown
            self.tint[k, 1] = np.asarray(look.leaf) * tone[k]
            self.tint[k, 2] = np.asarray(look.accent) * tone[k]
        self.tint[:, 1:] = linear(np.clip(self.tint[:, 1:], 0, 1))


class Forest:
    """The drawn trees of the open cells, kept in step with the city's streamer."""

    def __init__(self, engine, city, landmarks, workers: int = 2):
        from manifoldx.resources import StandardMaterial

        self.engine, self.city, self.landmarks = engine, city, landmarks
        self.material = StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True, tinted=True, faded=True)
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="forest")
        self._pending: dict[tuple[int, int], Future] = {}
        self.stands: dict[tuple[int, int], Stand] = {}
        self._meshes = {}  # (species, variant, crown, lod, part) -> its Mesh component, registered once
        self._absent: dict[tuple[int, int], int] = {}  # drawn cell -> the frame it left the streamer's set
        self._rows: dict = {}  # closed cell -> its (Trees) rows, oldest first
        self.day = None
        self._frame = 0
        self.clock = perf_counter
        self._fading: list[list] = []  # [entities, started, out]: parts fading in, or out and then destroyed

    def grow(self, i: int, j: int):
        """The (Trees) rows of cell (i, j): those the cell's rules grow and the car hits. On this thread too, the
        meshes of their species, which the main thread would otherwise stop for at their first spawn."""
        c = self.landmarks.keep(i, j, self.city.cell(i, j))
        near = roads_around(self.city, i, j)
        t = trees.cell_trees(i, j, c, self.city.terrain, near, trees.arrays_of(self.city, i, j),
                             yard_trees(self.city, i, j, self.landmarks))
        warm(np.unique(t.species))
        return t

    def update(self, cells, eye) -> None:
        """Follow the set of cells drawn in full: grow the new ones' trees on a thread, spawn those grown,
        drop the cells no longer drawn, and every EVERY frames move trees to their level of detail at eye."""
        day = trees.today()
        fresh = day != self.day
        if fresh:
            self.day = day
            for stand in self.stands.values():
                self._drop(stand, np.arange(len(stand.t)))
                stand.dress(day)
        cells = set(cells)
        self._frame += 1
        for c in cells:
            self._absent.pop(c, None)
        for c in [c for c in self.stands if c not in cells]:
            if self._frame - self._absent.setdefault(c, self._frame) >= LINGER:
                del self._absent[c]
                stand = self.stands.pop(c)
                self._drop(stand, np.arange(len(stand.t)))
                self._rows[c] = stand.t
                if len(self._rows) > KEEP:
                    del self._rows[next(iter(self._rows))]
        for c in [c for c in self._pending if c not in cells]:
            self._pending.pop(c).cancel()
        for c in cells:
            if c in self.stands or c in self._pending:
                continue
            if c in self._rows:
                self.stands[c] = Stand(self._rows.pop(c), day)
                fresh = True
            else:
                self._pending[c] = self._pool.submit(self.grow, *c)
        for c, fut in list(self._pending.items()):
            if fut.done():
                del self._pending[c]
                self.stands[c] = Stand(fut.result(), day)
                fresh = True
        if fresh or self._frame % EVERY == 0:
            self.place(eye)
        self._step()

    def place(self, eye) -> None:
        """Give every tree the level of detail its distance from eye calls for, respawning those that change."""
        eye = np.asarray(eye, float)
        stands = [s for s in self.stands.values() if len(s.t)]
        if not stands:
            return
        # a stand whose ball lies in one level, its trees already there, has no tree to move
        d = np.linalg.norm(np.array([s.centre for s in stands]) - eye, axis=1)
        reach = np.array([s.reach for s in stands]) + HYSTERESIS
        same = levels(d - reach) == levels(d + reach)
        band = levels(d)
        for k, stand in enumerate(stands):
            if same[k] and stand.settled == band[k]:
                continue
            d = np.linalg.norm(stand.pos + (0.0, 5.0, 0.0) - eye, axis=1)
            new = levels(d, np.where(stand.lod < 0, levels(d), stand.lod))
            moved = np.flatnonzero(new != stand.lod)
            if len(moved):
                self._drop(stand, moved)
                stand.lod[moved] = new[moved]
                self._spawn(stand, moved)
            stand.settled = int(stand.lod[0]) if (stand.lod == stand.lod[0]).all() else -1

    def _spawn(self, stand: Stand, idx) -> None:
        """Spawn the parts of trees idx of stand at their level of detail, one spawn per shared mesh."""
        from manifoldx.components import Fade, Material, Mesh, Tint, Transform

        t = stand.t
        # only near trees come in variants: far ones share one mesh per species, which halves their draw calls
        variant = np.where(stand.lod[idx] == 0, stand.variant[idx], 0)
        key = np.stack([t.species[idx], variant, stand.lod[idx]], axis=1)
        crowns = stand.crown[idx]
        for crown in np.unique(crowns):
            for k in np.unique(key[crowns == crown], axis=0):
                group = idx[(crowns == crown) & (key == k).all(axis=1)]
                s, v, lod = SPECIES[int(k[0])], int(k[1]), int(k[2])
                for p, part in enumerate(PARTS):
                    geo = meshes(s, lod, crown, v).get(part)
                    if geo is None:
                        continue
                    mesh = self._meshes.get((s, v, crown, lod, part))
                    if mesh is None:
                        mesh = self._meshes[(s, v, crown, lod, part)] = Mesh(geo)
                    h = self.engine.spawn(mesh, Material(self.material),
                                          Transform(pos=stand.pos[group], rot=stand.rot[group], scale=stand.scale[group]),
                                          Tint(rgb=stand.tint[group, p]), Fade(value=0.0), n=len(group))
                    stand.ents[group, p] = h.indices
                    self._fade(np.asarray(h.indices), out=False)

    def _drop(self, stand: Stand, idx) -> None:
        """Fade out the entities of trees idx of stand, to be destroyed once gone."""
        stand.settled = -1
        e = stand.ents[idx]
        e = e[e >= 0]
        if len(e):
            self._fade(e, out=True)
        stand.ents[idx] = -1
        stand.lod[idx] = -1

    def _fade(self, ents, out: bool) -> None:
        """Fade ents in from nothing, or out over the pixels a level fading in leaves; a part still fading in
        fades out from what shows of it now."""
        now = self.clock()
        if out:
            for group in [g for g in self._fading if not g[2]]:
                mid = np.isin(group[0], ents)
                if mid.any():
                    self._fading.append([group[0][mid], min(2 * now - group[1] - FADE, now), True])
                    ents = ents[~np.isin(ents, group[0][mid])]
                    group[0] = group[0][~mid]
        if len(ents):
            self._fading.append([ents, now, out])

    def _step(self) -> None:
        """Set every fading part's Fade for now: t in, -(1 - t) out; destroy the parts faded out."""
        if not self._fading:
            return
        now, fade, keep = self.clock(), self.engine.store._components["Fade"], []
        for ents, started, out in self._fading:
            if not len(ents):
                continue
            t = min((now - started) / FADE, 1.0)
            if t < 1.0:
                fade[ents] = -(1.0 - t) if out else t
                keep.append([ents, started, out])
            elif out:
                self.engine.destroy(ents)
            else:
                fade[ents] = 1.0
        self._fading = keep

    def counts(self) -> dict:
        """Trees held and drawn per level of detail, for the stats panel and probes."""
        lod = np.concatenate([s.lod for s in self.stands.values()]) if self.stands else np.zeros(0, int)
        return {"trees": len(lod), **{f"lod{k}": int((lod == k).sum()) for k in range(3)}}

    def close(self) -> None:
        """Stop growing and destroy every drawn tree."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        for stand in self.stands.values():
            self._drop(stand, np.arange(len(stand.t)))
        for ents, _started, out in self._fading:
            if out and len(ents):
                self.engine.destroy(ents)
        self._fading.clear()
        self.stands.clear()
        self._pending.clear()
