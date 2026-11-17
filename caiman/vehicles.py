"""How traffic is drawn: five kinds of vehicle, each one mesh, each a pool of entities moved as arrays.

The sedan, the old American car (the shared taxi) and the motorbike are manifoldx's parametric car with other
proportions; the bus and the truck, which it cannot make, are coloured boxes. Its outside pieces are merged into one vertex-coloured mesh and
simplified, so hundreds can be drawn. At start the game spawns a fixed pool of each kind; every frame the
cars' positions are written into the pool as arrays, and the slots no car uses wait far below the ground.
"""

import functools

import numpy as np

from caiman.rules.traffic import KINDS

PARKED = -1000.0  # the height of unused slots
POOL = {"car": 800, "taxi": 400, "moto": 400, "bus": 60, "truck": 200}
SLOT = 10_000  # a slot's number is kind · SLOT + its index in the pool
PARAMS = {
    "car": {},
    "taxi": {"length": 5.0, "width": 1.95, "height": 1.5, "wheelbase": 3.0, "front_overhang": 0.95},
    "moto": {"length": 2.0, "width": 0.75, "height": 1.2, "wheelbase": 1.35, "front_overhang": 0.35, "wheel_radius": 0.3,
                 "track": 0.1},
}
PAINT = {"car": "#8a2b22", "taxi": "#2f5d8c", "bus": "#c9a227", "moto": "#303236", "truck": "#d6d3c8"}
INSIDE = {"interior", "carpet", "dash", "seat", "liner"}  # pieces never seen from the street
GRID = 28  # decimation cells along the longest side


def _rgb(hex_colour: str) -> np.ndarray:
    h = hex_colour.lstrip("#")
    return np.array([int(h[k:k + 2], 16) / 255 for k in (0, 2, 4)])


def _rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


# Kinds the parametric car cannot make, as boxes: (size x, y, z), centre x, y, z, colour. Nose at +x.
BOXES = {
    "bus": [((12.0, 2.3, 2.5), (0.0, 1.55, 0.0), "#c9a227"),  # body
            ((11.0, 0.9, 2.52), (0.2, 2.0, 0.0), "#1d2226"),  # window band
            ((0.1, 1.1, 2.0), (6.0, 1.9, 0.0), "#1d2226"),  # windscreen
            ((1.0, 1.0, 2.6), (3.6, 0.5, 0.0), "#141414"), ((1.0, 1.0, 2.6), (-3.8, 0.5, 0.0), "#141414")],
    "truck": [((2.2, 2.4, 2.4), (2.9, 1.6, 0.0), "#7a2a20"),  # cab
              ((0.1, 0.9, 2.0), (4.0, 2.2, 0.0), "#1d2226"),  # windscreen
              ((5.6, 2.6, 2.5), (-1.2, 1.9, 0.0), "#d6d3c8"),  # cargo box
              ((1.0, 1.0, 2.5), (2.9, 0.5, 0.0), "#141414"), ((1.0, 1.0, 2.5), (-2.8, 0.5, 0.0), "#141414")],
}


def _boxes(kind: str) -> dict:
    from manifoldx.modeling.mesh import Mesh

    pos, faces, cols, norms, base = [], [], [], [], 0
    for size, centre, colour in BOXES[kind]:
        b = Mesh.box(*size).translate(centre).recompute_normals()
        pos.append(b.positions)
        norms.append(b.normals)
        faces.append(np.asarray(b.faces) + base)
        cols.append(np.tile(_rgb(colour), (len(b.positions), 1)))
        base += len(b.positions)
    return {"positions": np.vstack(pos).astype(np.float32), "normals": np.vstack(norms).astype(np.float32),
            "colors": np.vstack(cols).astype(np.float32), "indices": np.vstack(faces).astype(np.uint32).ravel()}


@functools.cache
def merged(kind: str) -> dict:
    """One mesh for a kind: positions, normals, colors and indices, centred on the origin, nose at +x."""
    from manifoldx.modeling.assets.car import PALETTE, CarParams, build_car

    if kind in BOXES:
        return _boxes(kind)
    from manifoldx.modeling.mesh import Mesh

    car = build_car(CarParams(**PARAMS[kind]))
    palette = {**PALETTE, "paint": (PAINT[kind], *PALETTE["paint"][1:])}
    pose = car.pose()
    pos, faces, cols, base = [], [], [], 0
    for part in car.parts:
        at, q = pose[part.name]
        for mat, mesh in part.pieces.items():
            if mat in INSIDE:
                continue
            p = _rotate(np.asarray(q, float), np.asarray(mesh.positions, float)) + at
            pos.append(p)
            faces.append(np.asarray(mesh.faces) + base)
            cols.append(np.tile(_rgb(palette[mat][0]), (len(p), 1)))
            base += len(p)
    whole = Mesh(np.vstack(pos), np.vstack(faces), colors=np.vstack(cols))
    span = np.ptp(whole.positions, axis=0).max()
    small = whole.decimate(grid=GRID).recompute_normals() if span > 0 else whole
    p = np.asarray(small.positions, np.float64)
    colours = _nearest_colours(p, whole.positions, np.vstack(cols))  # decimate keeps no colours
    p[:, [0, 2]] -= (p[:, [0, 2]].max(0) + p[:, [0, 2]].min(0)) / 2
    return {"positions": p.astype(np.float32), "normals": np.asarray(small.normals, np.float32),
            "colors": colours.astype(np.float32), "indices": np.asarray(small.faces, np.uint32).ravel()}


def _nearest_colours(points: np.ndarray, source: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """The colour of the nearest source vertex for each point, in chunks to bound memory."""
    out = np.empty((len(points), 3))
    for k in range(0, len(points), 512):
        d = ((points[k:k + 512, None, :] - source[None, :, :]) ** 2).sum(-1)
        out[k:k + 512] = colours[np.argmin(d, axis=1)]
    return out


def slots(pool: dict) -> np.ndarray:
    """The slot numbers of a pool, kind after kind in KINDS order."""
    return np.concatenate([KINDS.index(k) * SLOT + np.arange(pool.get(k, 0)) for k in KINDS]).astype(np.int64)


def slot_transforms(slot_k, xyz, heading, pitch, kind) -> tuple[np.ndarray, np.ndarray, int]:
    """Positions (R, 3) and quaternions (R, 4, xyzw) for the R slots in `slot_k`, and how many cars had no slot.

    Car c of kind k goes to the slot of kind k with index equal to c's rank among the cars of that kind.
    heading is atan2(dz, dx) of travel; manifoldx turns +x toward -z for a positive yaw, so yaw = -heading.
    """
    slot_k = np.asarray(slot_k, np.int64)
    R = len(slot_k)
    pos = np.zeros((R, 3))
    pos[:, 1] = PARKED
    rot = np.zeros((R, 4))
    rot[:, 3] = 1.0
    s_kind, s_idx = slot_k // SLOT, slot_k % SLOT
    kind = np.asarray(kind, np.int64)
    overflow = 0
    a, p = -np.asarray(heading, float) / 2, np.asarray(pitch, float) / 2
    q = np.stack([np.sin(a) * np.sin(p), np.sin(a) * np.cos(p), np.cos(a) * np.sin(p), np.cos(a) * np.cos(p)], 1)
    for k in range(len(KINDS)):
        cars = np.flatnonzero(kind == k)
        rows = np.flatnonzero(s_kind == k)
        rows = rows[np.argsort(s_idx[rows])]
        n = min(len(cars), len(rows))
        overflow += len(cars) - n
        pos[rows[:n]] = xyz[cars[:n]]
        rot[rows[:n]] = q[cars[:n]]
    return pos, rot, overflow


class Fleet:
    """The pools of vehicle entities in an engine, and the system that moves them each frame."""

    def __init__(self, engine, pool: dict | None = None):
        import manifoldx as mx
        from manifoldx.components import Component, Material, Mesh, Transform
        from manifoldx.resources import StandardMaterial
        from manifoldx.types import Float

        class Slot(Component):
            k: Float  # kind · SLOT + index in its pool

        self.Slot, self.pool = Slot, {**POOL, **(pool or {})}
        self.overflow = 0
        self._next = None  # (xyz, heading, pitch, kind) to draw on the next frame
        for kind in KINDS:
            n = self.pool[kind]
            if not n:
                continue
            m = merged(kind)
            pos = np.zeros((n, 3))
            pos[:, 1] = PARKED
            engine.spawn(Mesh({"positions": m["positions"], "normals": m["normals"], "colors": m["colors"],
                               "indices": m["indices"]}),
                         Material(StandardMaterial("#ffffff", roughness=0.6, vertex_colors=True)),
                         Transform(pos=pos), Slot(k=(KINDS.index(kind) * SLOT + np.arange(n)).astype(float)), n=n)

        def move(q: mx.Query[Transform, Slot], dt: float):
            if self._next is None:
                return
            k = np.rint(q[Slot].k.data[:, 0]).astype(np.int64)
            pos, rot, self.overflow = slot_transforms(k, *self._next)
            q[Transform].pos = pos
            q[Transform].rot = rot

        engine.system(move)

    def draw(self, xyz, heading, pitch, kind) -> None:
        """Show these cars from the next frame on."""
        self._next = (np.asarray(xyz, float), np.asarray(heading, float), np.asarray(pitch, float),
                      np.asarray(kind, np.int64))
