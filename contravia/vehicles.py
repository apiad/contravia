"""How traffic is drawn: one mesh per look of the fleet, each a pool of entities moved as arrays.

A look is a variant of a family of `contravia/rules/fleet.py` in one of its colourways (`rules.traffic.looks`). Its
mesh is manifoldx's `Vehicle.merged`: the outside pieces merged into one vertex-coloured mesh. It is not decimated:
a grid decimation smears the normals and bleeds the wheels' and the chrome's colours into the body, and the engine's
frustum culling keeps the cost to the cars in view. Building every look takes minutes, so the meshes are kept on disk under the user's cache
folder, stamped with the code that builds them, and built again on several threads only when that code changes.

At start the game spawns a pool of each look, sized by its share of Havana's mix; every frame the cars' positions are
written into the pools as arrays. A car takes a slot of its own look and, when its look's pool is full, a free slot
of another look of the same kind. The slots no car uses wait far below the ground.
"""

import hashlib
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from contravia.kept import kept
from contravia.rules.traffic import TrafficConfig, look_kinds, look_shares, looks

PARKED = -1000.0  # the height of unused slots
NEAR = 400  # the most cars the near traffic is expected to hold at once
SLOT = 10_000  # a slot's number is look · SLOT + its index in the pool
KEYS = ("positions", "normals", "colors", "indices")


def pool_sizes(near: int = NEAR) -> dict[int, int]:
    """Slots per look: its share of `near` cars with half again to spare, and at least one."""
    return {i: math.ceil(1.5 * p * near) + 1 for i, p in enumerate(look_shares(TrafficConfig())) if p > 0}


def _shape(kind: str, family: str, k: int) -> list[dict]:
    """The meshes of every colourway of one variant, centred on the origin with the nose at +x."""
    from contravia.rules.fleet import variant

    v = variant(kind, family, k)
    built = v.build()
    out = []
    for palette in v.colourways:
        geo = built.merged(palette=palette).to_geometry()
        out.append({key: geo[key] for key in KEYS})
    return out


def _stamp() -> str:
    """What the meshes are built from: this file, the fleet's rules and manifoldx's version."""
    from importlib.metadata import version

    from contravia.rules import fleet

    h = hashlib.sha256(version("manifold-gfx").encode())
    for path in (__file__, fleet.__file__):
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def cache_dir() -> Path:
    """Where built meshes are kept: $XDG_CACHE_HOME/contravia, else ~/.cache/contravia."""
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "contravia"


def meshes() -> list[dict]:
    """The mesh of every look, in looks() order: read from the cache, or built on all cores and cached."""
    path = cache_dir() / f"traffic-{_stamp()}.npz"
    every = looks()
    if path.exists():
        with np.load(path) as f:
            return [{key: f[f"{i}/{key}"] for key in KEYS} for i in range(len(every))]
    variants = list(dict.fromkeys((kind, family, k) for kind, family, k, _j in every))
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        shapes = dict(zip(variants, pool.map(lambda v: _shape(*v), variants)))
    out = [shapes[(kind, family, k)][j] for kind, family, k, j in every]
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in path.parent.glob("traffic-*.npz"):
        old.unlink()
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{f"{i}/{key}": m[key] for i, m in enumerate(out) for key in KEYS})
    tmp.replace(path)
    return out


def slots(pool: dict) -> np.ndarray:
    """The slot numbers of a pool, look after look."""
    return np.concatenate([i * SLOT + np.arange(n) for i, n in sorted(pool.items())]).astype(np.int64)


def _match(car_group, row_group, row_order) -> tuple[np.ndarray, np.ndarray]:
    """Pair the r-th car of each group with the r-th row of that group, rows taken in row_order.

    Returns the paired cars' and rows' indices; cars beyond their group's rows are left out.
    """
    rows = np.lexsort((row_order, row_group))
    g_rows = row_group[rows]
    cars = np.argsort(car_group, kind="stable")
    g_cars = car_group[cars]
    rank = np.arange(len(cars)) - np.searchsorted(g_cars, g_cars)
    start = np.searchsorted(g_rows, g_cars)
    ok = rank < np.searchsorted(g_rows, g_cars, side="right") - start
    return cars[ok], rows[start[ok] + rank[ok]]


def slot_transforms(slot_k, xyz, heading, pitch, look) -> tuple[np.ndarray, np.ndarray, int]:
    """Positions (R, 3) and quaternions (R, 4, xyzw) for the R slots in `slot_k`, and how many cars had no slot.

    Car c of look l goes to the slot of look l with index equal to c's rank among the cars of that look; the cars a
    full pool leaves out take the free slots of the other looks of their kind, in slot order.
    heading is atan2(dz, dx) of travel; manifoldx turns +x toward -z for a positive yaw, so yaw = -heading.
    """
    slot_k = np.asarray(slot_k, np.int64)
    R = len(slot_k)
    pos = np.zeros((R, 3))
    pos[:, 1] = PARKED
    rot = np.zeros((R, 4))
    rot[:, 3] = 1.0
    look = np.asarray(look, np.int64)
    a, p = -np.asarray(heading, float) / 2, np.asarray(pitch, float) / 2
    q = np.stack([np.sin(a) * np.sin(p), np.sin(a) * np.cos(p), np.cos(a) * np.sin(p), np.cos(a) * np.cos(p)], 1)
    s_look = slot_k // SLOT
    cars, rows = _match(look, s_look, slot_k)
    left = np.ones(len(look), bool)
    left[cars] = False
    free = np.ones(R, bool)
    free[rows] = False
    kinds = look_kinds()
    more_cars, more_rows = _match(kinds[look[left]], kinds[s_look[free]], slot_k[free])
    cars = np.r_[cars, np.flatnonzero(left)[more_cars]]
    rows = np.r_[rows, np.flatnonzero(free)[more_rows]]
    pos[rows] = xyz[cars]
    rot[rows] = q[cars]
    return pos, rot, len(look) - len(cars)


class Fleet:
    """The pools of vehicle entities in an engine, and the system that moves them each frame."""

    def __init__(self, engine, pool: dict | None = None):
        import manifoldx as mx
        from manifoldx.components import Component, Material, Mesh, Transform
        from manifoldx.resources import StandardMaterial
        from manifoldx.types import Float

        class Slot(Component):
            k: Float  # look · SLOT + index in its pool

        self.Slot, self.pool = Slot, {**pool_sizes(), **(pool or {})}
        self.overflow = 0
        self._next = None  # (xyz, heading, pitch, look) to draw on the next frame
        every = kept(engine.state, "traffic looks", _stamp(), meshes)
        for i, n in sorted(self.pool.items()):
            if not n:
                continue
            pos = np.zeros((n, 3))
            pos[:, 1] = PARKED
            engine.spawn(Mesh(dict(every[i])), Material(StandardMaterial("#ffffff", roughness=0.6, vertex_colors=True)),
                         Transform(pos=pos),
                         Slot(k=(i * SLOT + np.arange(n)).astype(float)), n=n)

        def move(q: mx.Query[Transform, Slot], dt: float):
            if self._next is None:
                return
            k = np.rint(q[Slot].k.data[:, 0]).astype(np.int64)
            pos, rot, self.overflow = slot_transforms(k, *self._next)
            q[Transform].pos = pos
            q[Transform].rot = rot

        engine.system(move)

    def draw(self, xyz, heading, pitch, look) -> None:
        """Show these cars from the next frame on."""
        self._next = (np.asarray(xyz, float), np.asarray(heading, float), np.asarray(pitch, float),
                      np.asarray(look, np.int64))

