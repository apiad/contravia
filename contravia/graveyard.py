"""The cemeteries' plots and pantheons the game draws: one row per item in each open cell, as GPU instances.

A cell's mesh holds only the cemetery's wall (cemetery.dress). When the city's streamer draws a cell in full, the
graveyard lays out its rows on a thread (cemetery.yard: the same plots and pantheons the car hits) and spawns
each as an instance of its kind's mesh at its level of detail (rules/tombs.py): every item of one kind and level
shares a geometry and the tinted material, so the engine draws each such group in one call, and each instance's
Tint is its stone. Every few frames the graveyard measures each item's distance to the camera and moves the
items that crossed a threshold to their new level; past the last one a plot is only its flat top.

Items do not pop: they fade in and out over FADE seconds, and an item that changes level keeps its old level
that long, fading out over the complementary pixels of the new one (manifoldx's Fade), as the forest's trees do.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter

import numpy as np

from contravia import cemetery
from contravia.rules.tombs import LODS, mesh

LOD_AT = (45.0, 140.0)  # metres from the camera where an item drops to level 1, then to level 2
HYSTERESIS = 6.0  # metres an item goes past a threshold before it changes level, so it does not flicker
EVERY = 6  # frames between two checks of the levels of detail
LINGER = 60  # frames a cell's items stay after the streamer stops drawing it in full
KEEP = 256  # cells whose rows are kept after they close, so a cell that opens again needs no laying out
FADE = 0.3  # seconds an item takes to fade in or out


def levels(d, old=None) -> np.ndarray:
    """The level of detail of items d metres from the camera; given their old levels, an item keeps its own
    unless it is more than HYSTERESIS metres past the threshold."""
    d = np.asarray(d, float)
    new = np.searchsorted(LOD_AT, d).astype(np.int8)
    if old is None:
        return new
    near, far = np.searchsorted(LOD_AT, d - HYSTERESIS), np.searchsorted(LOD_AT, d + HYSTERESIS)
    keep = (old >= near) & (old <= far)
    return np.where(keep, old, new).astype(np.int8)


class Plots:
    """One cell's items as rows: the Yard y, and per item its transform, level of detail and entity (-1 none)."""

    def __init__(self, y: cemetery.Yard):
        self.y = y
        n = len(y)
        self.pos = np.stack([y.xz[:, 0], y.y, y.xz[:, 1]], 1) if n else np.zeros((0, 3))
        half = y.theta / 2
        self.rot = np.stack([np.zeros(n), np.sin(half), np.zeros(n), np.cos(half)], 1)
        self.lod = np.full(n, -1, np.int8)
        self.ents = np.full(n, -1, np.int64)
        self.centre = self.pos.mean(axis=0) if n else np.zeros(3)
        self.reach = float(np.linalg.norm(self.pos - self.centre, axis=1).max()) if n else 0.0
        self.settled = -1  # the level every item has, or -1 while they differ


class Graveyard:
    """The drawn plots and pantheons of the open cells, kept in step with the city's streamer."""

    def __init__(self, engine, city, landmarks, workers: int = 1):
        from manifoldx.resources import StandardMaterial

        self.engine, self.city, self.landmarks = engine, city, landmarks
        self.material = StandardMaterial("#ffffff", roughness=0.85, vertex_colors=True, tinted=True, faded=True)
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="graveyard")
        self._pending: dict[tuple[int, int], Future] = {}
        self.plots: dict[tuple[int, int], Plots] = {}
        self._meshes = {}  # (kind, lod) -> its Mesh component, registered once
        self._absent: dict[tuple[int, int], int] = {}
        self._rows: dict = {}  # closed cell -> its Yard, oldest first
        self._frame = 0
        self.clock = perf_counter
        self._fading: list[list] = []  # [entities, started, out]
        self.timings: list[float] = []  # seconds each cell's rows took on the thread, for probes

    def lay(self, i: int, j: int) -> cemetery.Yard:
        """Cell (i, j)'s rows, timed."""
        t = perf_counter()
        y = cemetery.yard(self.city, i, j, self.landmarks)
        if len(y):
            self.timings.append(perf_counter() - t)
        return y

    def update(self, cells, eye) -> None:
        """Follow the set of cells drawn in full: lay out the new ones on a thread, spawn those laid out, drop
        the cells no longer drawn, and every EVERY frames move items to their level of detail at eye."""
        cells = set(cells)
        self._frame += 1
        fresh = False
        for c in cells:
            self._absent.pop(c, None)
        for c in [c for c in self.plots if c not in cells]:
            if self._frame - self._absent.setdefault(c, self._frame) >= LINGER:
                del self._absent[c]
                p = self.plots.pop(c)
                self._drop(p, np.arange(len(p.y)))
                self._rows[c] = p.y
                if len(self._rows) > KEEP:
                    del self._rows[next(iter(self._rows))]
        for c in [c for c in self._pending if c not in cells]:
            self._pending.pop(c).cancel()
        for c in cells:
            if c in self.plots or c in self._pending:
                continue
            if c in self._rows:
                self.plots[c] = Plots(self._rows.pop(c))
                fresh = True
            else:
                self._pending[c] = self._pool.submit(self.lay, *c)
        for c, fut in list(self._pending.items()):
            if fut.done():
                del self._pending[c]
                self.plots[c] = Plots(fut.result())
                fresh = True
        if fresh or self._frame % EVERY == 0:
            self.place(eye)
        self._step()

    def place(self, eye) -> None:
        """Give every item the level of detail its distance from eye calls for, respawning those that change."""
        eye = np.asarray(eye, float)
        held = [p for p in self.plots.values() if len(p.y)]
        if not held:
            return
        d = np.linalg.norm(np.array([p.centre for p in held]) - eye, axis=1)
        reach = np.array([p.reach for p in held]) + HYSTERESIS
        same = levels(d - reach) == levels(d + reach)
        band = levels(d)
        for k, p in enumerate(held):
            if same[k] and p.settled == band[k]:
                continue
            dist = np.linalg.norm(p.pos - eye, axis=1)
            new = levels(dist, np.where(p.lod < 0, levels(dist), p.lod))
            moved = np.flatnonzero(new != p.lod)
            if len(moved):
                self._drop(p, moved)
                p.lod[moved] = new[moved]
                self._spawn(p, moved)
            p.settled = int(p.lod[0]) if (p.lod == p.lod[0]).all() else -1

    def _spawn(self, p: Plots, idx) -> None:
        """Spawn items idx of p at their level of detail, one spawn per kind and level; a kind a level does not
        draw keeps no entity."""
        from manifoldx.components import Fade, Material, Mesh, Tint, Transform

        key = p.y.kind[idx].astype(int) * LODS + p.lod[idx]
        for k in np.unique(key):
            kind, lod = divmod(int(k), LODS)
            geo = mesh(kind, lod)
            if geo is None:
                continue
            group = idx[key == k]
            m = self._meshes.get((kind, lod))
            if m is None:
                m = self._meshes[(kind, lod)] = Mesh(geo)
            h = self.engine.spawn(m, Material(self.material),
                                  Transform(pos=p.pos[group], rot=p.rot[group], scale=np.ones((len(group), 3))),
                                  Tint(rgb=p.y.tone[group]), Fade(value=0.0), n=len(group))
            p.ents[group] = h.indices
            self._fade(np.asarray(h.indices), out=False)

    def _drop(self, p: Plots, idx) -> None:
        """Fade out the entities of items idx of p, to be destroyed once gone."""
        p.settled = -1
        e = p.ents[idx]
        e = e[e >= 0]
        if len(e):
            self._fade(e, out=True)
        p.ents[idx] = -1
        p.lod[idx] = -1

    def _fade(self, ents, out: bool) -> None:
        """Fade ents in from nothing, or out over the pixels a level fading in leaves; an item still fading in
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
        """Set every fading item's Fade for now: t in, -(1 - t) out; destroy the items faded out."""
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

    def waiting(self) -> int:
        """Cells whose rows are still being laid out."""
        return len(self._pending)

    def counts(self) -> dict:
        """Items held and drawn per level of detail, for probes."""
        lod = np.concatenate([p.lod for p in self.plots.values()]) if self.plots else np.zeros(0, int)
        drawn = np.concatenate([p.ents >= 0 for p in self.plots.values()]) if self.plots else np.zeros(0, bool)
        return {"items": len(lod), "drawn": int(drawn.sum()), **{f"lod{k}": int((lod == k).sum()) for k in range(LODS)}}

    def close(self) -> None:
        """Stop laying out and destroy every drawn item."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        for p in self.plots.values():
            self._drop(p, np.arange(len(p.y)))
        for ents, _started, out in self._fading:
            if out and len(ents):
                self.engine.destroy(ents)
        self._fading.clear()
        self.plots.clear()
        self._pending.clear()
