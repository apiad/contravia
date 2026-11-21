"""The foliage near the camera: small leaves and leaflets over a near tree's crown, drawn as instances that sway.

A tree at its nearest level of detail (rules/treegrammar.py) carries blobs for its leaf clusters and big
leaflets on its fronds: they read well from afar and flat up close. Within NEAR metres of the camera the forest
lays a detail layer over them. Over each leaf cluster of a broadleaf tree goes a clump: twigs on the cluster's
surface, each with a few small lanceolate leaves. Along each frond of a palm goes a plume of thin leaflets.
Each species has one clump geometry and a palm has one frond geometry per pose (rising or drooping), shared
by every tree as GPU instances. A tree's look per species is in rules/species.FOLIAGE. Leaves are grey
shades that the instance's tint colours with the day's leaf or accent colour, as the crown's blobs are.

The leaves sway in manifoldx's vertex shader (StandardMaterial's sway): a vertex moves with its height over its
instance's origin. A clump's origin is its cluster's centre, so the clump's top and its leaves' upturned tips
move most and its underside stays still. A frond's instance is turned so that its own up runs from the frond's
foot to its tip, so a leaflet sways more the further out along the frond it grows.

instances() gives the transforms of a whole group of trees' clumps or fronds at once. The forest calls it when
trees come near and never per frame.
"""

import functools
from dataclasses import dataclass

import numpy as np

from contravia.rules.species import ACCENT, FOLIAGE
from contravia.rules.treegrammar import FORMS, _crown, skeleton

NEAR = 25.0  # metres from the camera within which a tree at LOD 0 carries its detail layer
HOLD = 4.0  # metres a tree goes past NEAR before it gains or loses the layer, so it does not flicker
SWAY = (0.06, 1.2)  # the clumps' sway per unit of height and reach, in their unit-sphere frame
FROND_SWAY = (0.035, 3.0)  # the fronds' sway and reach, in metres along the frond


def near(d, old=None) -> np.ndarray:
    """Whether trees d metres from the camera carry the detail layer; given whether they did, a tree keeps its
    state unless it is more than HOLD metres past NEAR. A NEAR of 0 (quality Bajo) gives no tree the layer."""
    d = np.asarray(d, float)
    if NEAR <= 0:
        return np.zeros(d.shape, bool)
    if old is None:
        return d < NEAR
    return np.where(old, d < NEAR + HOLD, d < NEAR - HOLD)


def _geometry(pos, nrm, shade, both: bool = True) -> dict:
    """The geometry dict of quads (4 points each: foot, two sides, tip), each drawn as two triangles, on both
    sides or, without both, only on the side its points wind counter-clockwise round."""
    pos = np.asarray(pos, np.float32).reshape(-1, 3)
    k = np.arange(0, len(pos), 4, dtype=np.uint32)[:, None]
    idx = np.concatenate([k + (0, 1, 3), k + (0, 3, 2)])
    return {"positions": pos, "normals": np.asarray(nrm, np.float32).reshape(-1, 3),
            "colors": np.repeat(np.asarray(shade, np.float32).reshape(-1, 1), 3, axis=1),
            "indices": (np.concatenate([idx, idx[:, ::-1]]) if both else idx).ravel()}


def _leaf(foot, d, side, length, width):
    """The four points of a lanceolate leaf from foot along d: widest a third of the way, pointed at the tip."""
    mid = foot + d * length * 0.35
    return [foot, mid + side * width / 2, mid - side * width / 2, foot + d * length]


@functools.cache
def clump(species: str) -> dict:
    """The geometry of one leaf cluster's clump of species, in the frame of the cluster's blob: a unit sphere the
    instance stretches to the cluster's radii. Twigs sit evenly over the sphere (a Fibonacci lattice), each
    with FOLIAGE.per leaves fanning out of the blob and a little up; normals point out of the sphere, so the clump is lit
    as one round crown, and leaves are lighter at the tip and on top. A clump is always seen from outside its
    cluster, so each leaf is drawn on its outer side only, half the triangles of a leaf seen from both."""
    f, cr = FOLIAGE[species], FORMS[species].crown
    unit = 1.0 / float(np.mean(cr.size))  # a metre in the sphere's frame
    rng = np.random.default_rng([sum(map(ord, species)), 77])
    pos, nrm, shade = [], [], []
    for k in range(f.count):
        y = 1 - 2 * (k + 0.5) / f.count
        a = k * np.pi * (3 - 5 ** 0.5)
        out = np.array([np.sqrt(1 - y * y) * np.cos(a), y, np.sqrt(1 - y * y) * np.sin(a)])
        foot = out * rng.uniform(0.9, 1.0)  # on the blob's skin: its faces lie between 0.8 and 1
        u = np.cross(out, (0.0, 1.0, 0.0) if abs(out[1]) < 0.95 else (1.0, 0.0, 0.0))
        u /= np.linalg.norm(u)
        w = np.cross(out, u)
        turn = rng.uniform(0, 2 * np.pi)
        for j in range(f.per):
            az = turn + 2 * np.pi * j / f.per
            d = np.cos(az) * u + np.sin(az) * w + out * 1.3 + (0.0, 0.25, 0.0)  # poking out of the blob
            d /= np.linalg.norm(d)
            side = np.cross(d, out)
            side /= np.linalg.norm(side)
            size = rng.uniform(0.8, 1.2)
            leaf = _leaf(foot, d, side, f.length * unit * size, f.width * unit * size)
            if np.dot(np.cross(leaf[1] - leaf[0], leaf[3] - leaf[0]), out) < 0:
                leaf[1], leaf[2] = leaf[2], leaf[1]  # wind it to face out of the cluster
            pos += leaf
            lit = 0.82 + 0.18 * max(y, -0.5)
            shade += [lit * 0.8, lit * 0.9, lit * 0.9, lit]
            nrm += [out] * 4
    return _geometry(pos, nrm, shade, both=False)


@functools.cache
def _frond_frame(species: str, up: bool) -> np.ndarray:
    """The rotation (3, 3, columns the local x, y, z) from a frond's own frame to its tree's, for a frond turned
    to yaw 0: local y runs from the frond's foot to its tip, local z is the frond's side."""
    palm = FORMS[species].palm
    rise, droop = (palm.rise, palm.droop) if up else (palm.rise * 0.4, palm.droop * 1.25)
    ey = np.array([palm.length, rise - droop, 0.0])
    ey /= np.linalg.norm(ey)
    ez = np.array([0.0, 0.0, 1.0])
    return np.stack([np.cross(ey, ez), ey, ez], axis=1)


@functools.cache
def frond(species: str, up: bool) -> dict:
    """The geometry of a palm's frond plume in the frond's own frame (_frond_frame), its foot at the origin: the
    same spine as treegrammar's frond, with FOLIAGE.count pairs of thin leaflets, longest mid-frond, hanging
    FOLIAGE.hang degrees below the frond. The royal palm's leaflets stand in three planes, a plume; the coconut's
    hang in a V."""
    f, palm = FOLIAGE[species], FORMS[species].palm
    rise, droop = (palm.rise, palm.droop) if up else (palm.rise * 0.4, palm.droop * 1.25)
    b = _frond_frame(species, up)
    rng = np.random.default_rng([sum(map(ord, species)), int(up), 78])
    pos, nrm, shade = [], [], []
    upn = b.T @ np.array([0.0, 1.0, 0.0])
    for k in range(f.count):
        t = 0.04 + 0.96 * (k + 0.5) / f.count
        spine = np.array([palm.length * t, (3 * rise + droop) * t - 2 * (rise + droop) * t * t, 0.0])
        ahead = np.array([palm.length, 3 * rise + droop - 4 * (rise + droop) * t, 0.0])
        ahead /= np.linalg.norm(ahead)
        length = f.length * (0.3 + 0.7 * np.sin(np.pi * min(t * 1.15, 1.0))) * rng.uniform(0.9, 1.1)
        for s in (1, -1):
            hang = f.hang + (25.0 * ((k + (s > 0)) % 3 - 1) if f.per > 1 else 0.0)
            h = np.radians(hang + rng.uniform(-6, 6))
            d = 0.45 * ahead + np.array([0.0, -np.sin(h), s * np.cos(h)])
            d /= np.linalg.norm(d)
            side = np.cross(d, (0.0, 1.0, 0.0))
            side /= max(np.linalg.norm(side), 1e-6)
            pos += [b.T @ p for p in _leaf(spine, d, side, length, f.width)]
            shade += [0.78, 0.9, 0.9, 1.0]
            nrm += [upn] * 4
    return _geometry(pos, nrm, shade)


def _quat(m) -> np.ndarray:
    """The unit quaternion (x, y, z, w) of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return np.array([x, y, z, w])


def qmul(a, b) -> np.ndarray:
    """The products a * b of (N, 4) quaternions (x, y, z, w): rotate by b, then by a."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    x1, y1, z1, w1 = a.T
    x2, y2, z2, w2 = b.T
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=1)


def rotate(q, v) -> np.ndarray:
    """(N, 3) vectors v turned by (N, 4) quaternions q."""
    q, v = np.atleast_2d(q), np.atleast_2d(v)
    u, w = q[:, :3], q[:, 3:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _yaw(angle) -> np.ndarray:
    """(N, 4) quaternions turning by angle radians about y."""
    angle = np.atleast_1d(np.asarray(angle, float))
    z = np.zeros(len(angle))
    return np.stack([z, np.sin(angle / 2), z, np.cos(angle / 2)], axis=1)


@dataclass(frozen=True)
class Spots:
    """Where one tree's detail goes, in its tree's frame before its scale: per instance its mesh key, foot,
    turn, scale in its own frame, role, and whether its scale follows the tree's width alone (a frond, turned
    out of the tree's axes) rather than the tree's width and height (a clump)."""

    key: list
    foot: np.ndarray
    turn: np.ndarray
    size: np.ndarray
    role: np.ndarray
    turned: np.ndarray


@functools.cache
def spots(species: str, variant: int, crown: str) -> Spots:
    """The detail of a tree of species, variant and crown at LOD 0: a clump on each cluster treegrammar draws, or
    a plume on each frond, none for a bare crown."""
    form = FORMS[species]
    keys, foot, turn, size, role, turned = [], [], [], [], [], []
    if form.palm is not None:
        palm = form.palm
        top = np.array((form.lean, form.height, 0.0)) + ((0.0, 2.0, 0.0) if palm.crownshaft else 0.0)
        n = {"full": palm.fronds, "sparse": palm.fronds // 2, "bare": 0}[crown]
        for k in range(n):
            up = k % 2 == 0
            yaw = 2 * np.pi * k / max(n, 1) + 0.3
            keys.append(("frond", species, up))
            foot.append(top)
            # treegrammar turns a frond's direction to (cos yaw, 0, sin yaw), a turn of -yaw about y
            turn.append(qmul(_yaw(-yaw), _quat(_frond_frame(species, up)))[0])
            size.append(np.ones(3))
            role.append(2)
            turned.append(True)
    else:
        for c, radii, r in _crown(skeleton(species, variant), 0, crown):
            keys.append(("clump", species))
            foot.append(c)
            turn.append(np.array([0.0, 0.0, 0.0, 1.0]))  # each tree turns its clumps its own way: instances()
            size.append(radii)
            role.append(r)
            turned.append(False)
    return Spots(keys, np.reshape(foot, (-1, 3)), np.reshape(turn, (-1, 4)), np.reshape(size, (-1, 3)),
                 np.array(role, np.int8), np.array(turned, bool))


def instances(sp: Spots, pos, rot, scale, unit) -> dict:
    """The instances of trees sharing sp, standing at pos (G, 3), turned by rot (G, 4) and scaled by scale (G, 3)
    (a tree's width, height, width), with unit (G,) a number in [0, 1) from each tree's seed that turns its
    clumps: {mesh key: (tree (N,) an index into the G trees, accent (N,) whether it takes the accent's tint,
    pos (N, 3), rot (N, 4), scale (N, 3))}."""
    g, k = len(pos), len(sp.key)
    out = {}
    if not g or not k:
        return out
    tree = np.repeat(np.arange(g), k)
    spot = np.tile(np.arange(k), g)
    s = scale[tree]
    p = pos[tree] + rotate(rot[tree], s * sp.foot[spot])
    # a clump's own turn about y, golden-ratio steps from the tree's seed so neighbours differ
    spin = _yaw(2 * np.pi * ((unit[tree] + 0.618034 * spot) % 1.0))
    own = np.where(sp.turned[spot, None], sp.turn[spot], spin)
    q = qmul(rot[tree], own)
    z = np.where(sp.turned[spot, None], s[:, :1] * sp.size[spot], s * sp.size[spot])
    for key in dict.fromkeys(sp.key):
        m = np.array([kk == key for kk in sp.key])[spot]
        out[key] = (tree[m], sp.role[spot][m] == ACCENT, p[m], q[m], z[m])
    return out


def geometry(key) -> dict:
    """The geometry of a mesh key from spots(): ("clump", species) or ("frond", species, up)."""
    return clump(key[1]) if key[0] == "clump" else frond(key[1], key[2])


def triangles(species: str, variant: int = 0, crown: str = "full") -> int:
    """How many triangles a tree of species adds with its detail layer."""
    return sum(len(geometry(k)["indices"]) // 3 for k in spots(species, variant, crown).key)
