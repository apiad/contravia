"""People on the pavements: the pool of bodies a preset builds, and (in later steps) the crowd that walks them.

A crowd does not get one body per person: a body costs about 350 ms over its three levels and 280 ms of clips, so
300 unique people would take minutes. A preset builds a pool of bodies instead (Figures), each a
rules.people.body dressed for one zone, at levels 2, 1 and 0, with the clips of rules.people.CLIPS, and every walker
plays one of them. The work runs on a Builder thread of its own, in an order that lets the crowd show early: every
body's level 2 and walk clips first, then level 1, then level 0 and the other clips. The GPU uploads (a Mesh, a
baked clip) happen on the frame thread, a few per frame, as results arrive.

A clip is baked from the body's level 2 figure: the skeleton is the same at every level, so it plays on all three.
Results are kept across a save under contravia dev (contravia.kept) until rules/people.py or this file changes.
Spec: docs/superpowers/specs/2026-11-22-pedestrians-in-game-design.html.
"""

from __future__ import annotations

from collections import deque

from contravia import kept as keep
from contravia.garage import Builder
from contravia.rules import people as rules

SEED = 1000  # body i of the pool is rules.people.body(SEED + i)
ZONES = 3  # body i dresses for zone i % ZONES
LEVELS = (2, 1, 0)
UPLOADS = 2  # results put on the GPU per frame at most
FRAMES = 32  # poses per baked cycle


def work(i: int) -> list[tuple]:
    """Every job of body i, as (stage, item): stage orders the jobs of the whole pool."""
    walk = [("clip", i, "walk", v) for v in rules.WALK]
    rest = [("clip", i, k, v) for k, v in rules.CLIPS if k != "walk"]
    return ([(0, ("figure", i, 2))] + [(1, j) for j in walk] + [(2, ("figure", i, 1))]
            + [(3, ("figure", i, 0))] + [(4, j) for j in rest])


class Figures:
    """The pool of bodies of one preset: their meshes at three levels and their clips, built off the frame."""

    def __init__(self, engine, bodies: int):
        self.engine = engine
        stamp = (keep.source(rules.__file__), keep.source(__file__))
        self.store = keep.kept(engine.state, "people", stamp, dict)  # item -> what the frame thread made of it
        self._figs = {}  # body -> its level 2 Figure, written and read on the worker thread only
        self.builder = Builder(make=self._make, key_of=repr, cached=1 << 30, name="people")
        self.waiting: deque = deque()
        self.n = 0
        self.resize(bodies)

    def body(self, i: int):
        """Body i's HumanParams, Outfit and zone."""
        params = rules.body(SEED + i)
        zone = i % ZONES
        return params, rules.outfit(SEED + i, zone, params.build), zone

    def resize(self, bodies: int) -> None:
        """Hold a pool of this many bodies, queueing the jobs of the bodies not built yet. A smaller pool keeps
        what it built; its extra bodies are only not handed out."""
        new = range(self.n, bodies)
        self.n = bodies
        jobs = sorted((s, k, item) for k, i in enumerate(new) for s, item in work(i))
        for _, _, item in jobs:
            if item not in self.store:
                self.waiting.append(item)
                self.builder.want(item)

    def _make(self, item):
        """On the worker: a figure's geometry, or a clip's poses."""
        from manifoldx.humanoid.cycles import bake_cycle
        from manifoldx.modeling.assets.human import build_human

        if item[0] == "figure":
            _, i, level = item
            params, outfit, _ = self.body(i)
            f = build_human(params, outfit=outfit, level=level)
            if level == 2:
                self._figs[i] = f
            return f.mesh.to_geometry()
        _, i, kind, speed = item
        f = self._figs.get(i)
        if f is None:  # its level 2 job ran in an earlier life of this object
            params, outfit, _ = self.body(i)
            f = self._figs[i] = build_human(params, outfit=outfit, level=2)
        return bake_cycle(f, kind, speed, FRAMES)

    def step(self) -> int:
        """On the frame thread: put up to UPLOADS finished results on the GPU. The worker runs its jobs in the
        order they were queued, so only the head of the queue can be done. Returns how many were put up."""
        from manifoldx.components import Mesh

        done = 0
        while self.waiting and done < UPLOADS:
            item = self.waiting[0]
            got = self.builder.want(item)
            if got is None:
                break
            self.waiting.popleft()
            self.builder.cache.pop(repr(item), None)  # the store holds it now
            self.store[item] = Mesh(got) if item[0] == "figure" else self.engine.bake_poses(got)
            done += 1
        return done

    def mesh(self, i: int, level: int):
        """Body i's Mesh at level, or at the nearest level built, coarser first; None while none is."""
        for lv in sorted(LEVELS, key=lambda lv: (abs(lv - level), -lv)):
            m = self.store.get(("figure", i, lv))
            if m is not None:
                return m
        return None

    def clip(self, i: int, kind: str, speed: float):
        """The id of body i's baked clip of kind at speed, or None while it is not baked."""
        return self.store.get(("clip", i, kind, speed))

    def ready(self, zone: int) -> list[int]:
        """The bodies of zone a walker can play now: one level built and every walking clip baked."""
        return [i for i in range(zone, self.n, ZONES)
                if self.mesh(i, 2) is not None and all(self.clip(i, "walk", v) is not None for v in rules.WALK)]

    def counts(self) -> dict:
        built = [k for k in self.store if k[1] < self.n]
        return {"bodies": self.n, "meshes": sum(k[0] == "figure" for k in built),
                "clips": sum(k[0] == "clip" for k in built), "waiting": len(self.waiting)}

    def close(self) -> None:
        self.builder.close()
