"""Driving help: six levels from a real car (5) to almost autopilot (0), and where the car is on the road.

Each level sets the strengths of manifoldx's four chassis assists (steering, traction, anti-lock, stability) and,
from level 2 down, the road assists that read the streets (spec
docs/superpowers/specs/2026-11-20-driving-help-design.html). Every road assist starts from the same answer, which
`Road.locate` gives once a frame: the edge the car drives along, how far along it, how far right of its centreline,
and the lane that falls in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manifoldx.vehicles.assists import NARROW, NARROW_AT

from contravia.traffic import BRAKE, JUNCTION, LANE_SPEED, _idm, lane_geometry, lane_offsets

# (name in the menu, what it does) by level, 0 the most help
LEVELS = (
    ("Casi piloto", "maneja por ti: sigue al de delante, para en rojo y dobla donde le toques el timón"),
    ("Freno de emergencia", "te mantiene en la senda y frena solo antes de chocar"),
    ("Mantener senda", "con el timón suelto se centra en su senda"),
    ("Dirección asistida", "no se va de lado, no bloquea las ruedas y te corrige el timón"),
    ("Chasis", "frenos antibloqueo, control de tracción y de estabilidad, como un carro moderno"),
    ("Real", "sin ninguna ayuda"),
)
DEFAULT = 3
CHASSIS = 0.6  # strength of the chassis assists at level 4
REACH = 12.0  # metres: farthest a car can be from an edge's centreline and still be on it
ALONG = np.cos(np.radians(60))  # the car's heading must be within 60° of the edge's direction
CELL, PIECE = 16.0, 8.0  # the index's cells and the pieces segments are cut into for it
AROUND = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])  # the 3 × 3 cells around a point


def chassis_strengths(level: int, locked=()) -> np.ndarray:
    """The four chassis assists' strengths, in manifoldx's ASSISTS order, at this level; locked ones are 0."""
    from manifoldx.vehicles.assists import ASSISTS

    k = np.ones(len(ASSISTS)) if level <= 3 else np.full(len(ASSISTS), CHASSIS if level == 4 else 0.0)
    if level == 4:
        k[ASSISTS.index("steer")] = 0.0
    for name in locked:
        if name in ASSISTS:
            k[ASSISTS.index(name)] = 0.0
    return k


def level_from_realism(realism: float) -> int:
    """The level an old saved realism (0 easy to 100 real) becomes."""
    return 5 if realism >= 90 else 4 if realism >= 60 else 3


def next_level(level: int) -> int:
    """The level the L key turns to: one more help a press, from Real (5) down to Casi piloto (0), then Real."""
    return (level - 1) % len(LEVELS)


@dataclass
class Place:
    """Where the car is on the road: its edge, s metres along it, `right` metres right of its centreline, in
    `lane` (0 the rightmost), `dist` metres from the edge's polyline."""

    edge: int
    s: float
    right: float
    lane: int
    dist: float


class Road:
    """The road graph's polylines, indexed on a grid so a point finds the segments near it in a few lookups."""

    def __init__(self, graph):
        a = graph.a
        self.graph = graph
        self.lanes = a["g_lanes"].astype(np.int64)
        self.vfree = a["g_vfree"].astype(np.float64)
        self.twoway, self.lane_w = lane_geometry(a)
        poly, starts = a["g_poly"].astype(np.float64), a["g_poly_start"].astype(np.int64)
        edge_of = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
        k = np.flatnonzero(np.arange(len(poly) - 1) + 1 < starts[edge_of[:-1] + 1])  # segments inside an edge
        self.p0, self.edge = poly[k], edge_of[k]
        d = poly[k + 1] - self.p0
        self.seg_len = np.maximum(np.hypot(*d.T), 1e-9)
        self.u = d / self.seg_len[:, None]
        seg = np.zeros(len(poly))
        seg[k + 1] = self.seg_len
        cum = np.cumsum(seg)
        self.s0 = cum[k] - cum[starts[self.edge]]  # distance along the edge where each segment starts
        # cut every segment into pieces of at most PIECE and file each piece by its midpoint's cell
        n = np.ceil(self.seg_len / PIECE).astype(np.int64)
        which = np.repeat(np.arange(len(k)), n)
        f = (np.arange(len(which)) - np.repeat(np.cumsum(n) - n, n) + 0.5) / n[which]
        mid = self.p0[which] + d[which] * f[:, None]
        key = self._key(np.floor(mid / CELL).astype(np.int64))
        order = np.argsort(key, kind="stable")
        self.keys, self.pieces = key[order], which[order]
        self.out: dict[int, list[int]] = {}  # edge -> edges it turns into
        for i, o in zip(a["g_turn_in"].tolist(), a["g_turn_out"].tolist()):
            self.out.setdefault(i, []).append(o)

    @staticmethod
    def _key(cell: np.ndarray) -> np.ndarray:
        return (cell[..., 0] + (1 << 20)) * (1 << 21) + (cell[..., 1] + (1 << 20))

    def _near(self, x: float, z: float) -> np.ndarray:
        """Segments with a piece in the 3 × 3 cells around (x, z): every segment within REACH of it, and some more."""
        cells = self._key(np.floor(np.array([x, z]) / CELL).astype(np.int64) + AROUND)
        lo, hi = np.searchsorted(self.keys, cells).tolist(), np.searchsorted(self.keys, cells, side="right").tolist()
        return np.concatenate([self.pieces[a:b] for a, b in zip(lo, hi)])  # a segment may repeat: harmless

    def locate(self, x: float, z: float, heading: float) -> Place | None:
        """Where a car at (x, z) driving along `heading` (atan2(dz, dx)) is on the road, or None off the road,
        the wrong way on a one-way street or crosswise to every street."""
        seg = self._near(x, z)
        if not len(seg):
            return None
        rel = np.array([x, z]) - self.p0[seg]
        t = np.clip((rel * self.u[seg]).sum(1), 0, self.seg_len[seg])
        foot = self.p0[seg] + self.u[seg] * t[:, None]
        dist = np.hypot(*(foot - (x, z)).T)
        dist = np.where(self.u[seg] @ np.array([np.cos(heading), np.sin(heading)]) >= ALONG, dist, np.inf)
        k = int(np.argmin(dist))
        if not np.isfinite(dist[k]):
            return None
        i, e = seg[k], int(self.edge[seg[k]])
        real = float(np.hypot(*(foot[k] - (x, z))))
        right = float(rel[k] @ np.array([-self.u[i, 1], self.u[i, 0]]))
        offs = lane_offsets(int(self.lanes[e]), float(self.lane_w[e]), bool(self.twoway[e]))
        lane = int(np.argmin(np.abs(offs - right)))
        if real > REACH or abs(offs[lane] - right) > self.lane_w[e] + 1.5:
            return None
        return Place(e, float(self.s0[i] + t[k]), right, lane, real)


ROAD_ASSISTS = ("lane", "brake", "pilot")
LOOK_TIME, LOOK_MIN = 0.9, 6.0  # the lane keeping aims max(LOOK_MIN, LOOK_TIME × speed) metres ahead
MIX = 0.15  # the driver's steer under which the lane keeping mixes in, all of it at 0
CHANGE, CHANGE_HOLD = 0.5, 0.3  # steer past CHANGE for CHANGE_HOLD seconds: the next lane on that side
MOVING = 2.0  # m/s: slower, the road assists stand down
LIT = 0.02  # a change of the steer past this lights the assist's lamp
A_SOFT, A_HARD = 3.0, 6.0  # m/s² it would take to stop short: the emergency brake starts at the first, full at the second
SWERVE = 0.6  # a steer past this is a swerve: the emergency brake lets the driver have it
MARGIN = 0.3  # metres each side of the car's width that the brake keeps clear
HOLD = 1.5  # metres: creeping this close to something in its path, the car is held
STOP_AT = 2.0  # metres short of something in its path where the emergency brake aims to stop
WANT_STEP = 3.0  # m/s per second the held accelerator or brake moves the pilot's wanted speed
TURN_AT = 40.0  # metres before a corner where a tap of the wheel picks the pilot's exit
STRAIGHT = np.radians(30)  # an exit within this of straight on is straight on
TURN_LATERAL, TURN_RADIUS = 3.0, 8.0  # m/s² and metres the pilot takes a turn with
GAS, BRAKING = 2.5, 8.0  # m/s² of the pilot's wanted acceleration that full throttle and full brake stand for


def road_assists(level: int, locked=()) -> set[str]:
    """The road assists on at this level: lane keeping from 2 down, the emergency brake from 1, the pilot at 0."""
    return {name for name, below in zip(ROAD_ASSISTS, (2, 1, 0)) if level <= below and name not in locked}


class DrivingHelp:
    """The road assists of the player's car: each frame they read where it is on the road and correct the driver's
    controls before the chassis assists and the physics. `lit` says which corrected in the last frame."""

    def __init__(self, road: Road, max_steer: float, wheelbase: float, length: float = 4.5, width: float = 1.8):
        self.road, self.max_steer, self.wheelbase = road, max_steer, wheelbase
        self.length, self.width = length, width
        self.yaw = None  # the last frame's, for the yaw rate
        self.wanted = None  # m/s the pilot drives at most; None until it first drives
        self.turn, self.turn_edge = 0, None  # the exit the driver picked at the end of that edge: -1 left, 1 right
        self.manual, self.pressed = False, False  # handed back by the handbrake; the accelerator pressed since
        self.exit = None  # (edge, signed turn in radians) the pilot takes next, for the HUD's arrow
        self.edge = self.lane = None  # the edge and the lane the car keeps to
        self.off = 0.0  # metres right of the centreline it aims at, easing toward its lane's
        self.held, self.held_lane, self.side = 0.0, 0, 0.0  # a long tap of the wheel: how long, from which lane
        self.lit = {name: False for name in ROAD_ASSISTS}

    def apply(self, c, car, dt: float, level: int, locked=(), steer_strength: float = 1.0, obstacles=None,
              signal=None):
        """The controls after the road assists of this level; steer_strength is the chassis steering assist's,
        which narrows the wheel with speed. obstacles: (W, 2, 2) segments and their velocities (W, 2), what the
        brake and the pilot must not hit. signal(edge, dist, v): whether the signal at the end of edge lets the
        car go on (None: no signals, as with the traffic off)."""
        from dataclasses import replace

        self.lit = {name: False for name in ROAD_ASSISTS}
        on = road_assists(level, locked)
        yaw_rate = 0.0 if self.yaw is None else ((car.yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi) / dt
        self.yaw = car.yaw
        p = self.road.locate(car.x, car.z, car.yaw) if on & {"lane", "pilot"} else None
        pilot = "pilot" in on and p is not None and not self._handed_back(c)
        if not pilot:
            self.wanted, self.exit = None, None
        steer = c
        if pilot:
            if p.edge != self.turn_edge:
                self.turn, self.turn_edge = 0, p.edge
            if self.road.graph.a["g_len"][p.edge] - p.s < TURN_AT and abs(c.steer) > CHANGE:
                self.turn = int(np.sign(c.steer))  # a tap near the corner picks the exit, it does not steer
                steer = replace(c, steer=0.0)
        out = self._keep_lane(steer, car, dt, p if "lane" in on else None, steer_strength)
        path = self._in_path(car, yaw_rate, *obstacles) if obstacles is not None else None
        if pilot:
            out = self._pilot(out, c, car, dt, p, path, signal)
        if "brake" in on and path is not None:
            out = self._brake(out, c, car, *path)
        return out

    def _handed_back(self, c) -> bool:
        """Whether the handbrake handed the car back to the driver: from the pull until the accelerator is pressed
        and released again."""
        if c.handbrake:
            self.manual, self.pressed = True, False
        elif self.manual and c.throttle > 0:
            self.pressed = True
        elif self.manual and self.pressed:
            self.manual = False
        return self.manual

    def _in_path(self, car, yaw_rate: float, segs: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The gap from the car's nose to each segment in its path (its width and MARGIN each side, bent by the
        yaw rate) and that segment's speed along the car's heading; segments behind the car's middle left out."""
        if not len(segs):
            return np.zeros(0), np.zeros(0)
        f = np.array([np.cos(car.yaw), np.sin(car.yaw)])
        r = np.array([-f[1], f[0]])
        rel = np.asarray(segs, float) - (car.x, car.z)
        ahead, side = rel @ f, rel @ r  # (W, 2) each: metres ahead of and right of the car's middle
        side = side - yaw_rate / max(car.speed, 1.0) * ahead ** 2 / 2  # right of the arc the car is turning on
        w = self.width / 2 + MARGIN
        d0, d1 = side[:, 0], side[:, 1]
        flat = np.abs(d1 - d0) < 1e-9
        dd = np.where(flat, 1e-9, d1 - d0)
        ta, tb = (-w - d0) / dd, (w - d0) / dd
        lo = np.where(flat, np.where(np.abs(d0) <= w, 0.0, 2.0), np.maximum(np.minimum(ta, tb), 0.0))
        hi = np.where(flat, np.where(np.abs(d0) <= w, 1.0, -1.0), np.minimum(np.maximum(ta, tb), 1.0))
        a0, a1 = ahead[:, 0], ahead[:, 1]
        gap = np.minimum(a0 + (a1 - a0) * lo, a0 + (a1 - a0) * hi) - self.length / 2
        keep = (lo <= hi) & (gap > -self.length / 2)
        return gap[keep], (np.asarray(vel, float) @ f)[keep]

    def _pilot(self, out, c, car, dt: float, p: Place, path, signal):
        """Throttle and brake by the intelligent driver model: behind the nearest thing in the car's path or a red
        light's stop line, toward the street's speed capped by the wanted speed and slowed for a coming turn."""
        from dataclasses import replace

        e, length = p.edge, float(self.road.graph.a["g_len"][p.edge])
        if self.wanted is None:
            self.wanted = float(self.road.vfree[e])
        self.wanted = float(np.clip(self.wanted + WANT_STEP * dt * ((c.throttle > 0) - (c.brake > 0)), 0.0, 40.0))
        v0 = min(float(self.road.vfree[e]), self.wanted)
        to_end = length - p.s
        nxt = self._exit(e)
        self.exit = nxt
        if nxt is not None and abs(nxt[1]) > STRAIGHT:  # slow so as to turn at TURN_LATERAL
            v0 = min(v0, float(np.sqrt(TURN_LATERAL * TURN_RADIUS + 2 * BRAKE * max(to_end - JUNCTION, 0.0))))
        gap, lead = np.inf, 0.0
        if path is not None and len(path[0]):
            k = int(np.argmin(path[0]))
            gap, lead = float(path[0][k]), float(path[1][k])
        if signal is not None and not signal(e, to_end, car.speed) and to_end - 1.0 < gap:
            gap, lead = max(to_end - 1.0, 0.0), 0.0  # a red light: a stopped car at the stop line
        if v0 <= 0.1:
            acc = -BRAKING
        else:
            acc = float(_idm(np.array([max(car.speed, 0.0)]), np.array([v0]), np.array([gap]), np.array([lead]))[0])
        self.lit["pilot"] = True
        throttle, brake = float(np.clip(acc / GAS, 0, 1)), float(np.clip(-acc / BRAKING, 0, 1))
        return replace(out, throttle=throttle, brake=max(brake, c.brake))

    def _exit(self, e: int):
        """(edge, signed turn) the car takes at the end of edge e, right positive: the one the driver picked, else
        the one closest to straight on, else the widest; None at a dead end."""
        g = self.road.graph
        exits = np.array(self.road.out.get(e, []), np.int64)
        if not len(exits):
            return None
        _, _, end = g.points([e], [float(g.a["g_len"][e])], [0.0])
        _, _, start = g.points(exits, np.zeros(len(exits)), np.zeros(len(exits)))
        turn = (start - end + np.pi) % (2 * np.pi) - np.pi
        ok = np.abs(turn) < np.radians(150)  # never a U-turn
        exits, turn = exits[ok], turn[ok]
        if not len(exits):
            return None
        side = turn * self.turn > STRAIGHT if self.turn and e == self.turn_edge else np.zeros(len(exits), bool)
        if side.any():  # the first exit on the picked side
            k = np.flatnonzero(side)[np.argmin(np.abs(turn[side]))]
        elif np.abs(turn).min() < STRAIGHT:
            k = int(np.argmin(np.abs(turn)))
        else:
            k = int(np.argmax(self.road.lanes[exits] * self.road.lane_w[exits]))
        return int(exits[k]), float(turn[k])

    def _brake(self, out, c, car, gap: np.ndarray, lead: np.ndarray):
        """Brake when stopping short of something in the car's path would take more than A_SOFT, and hold a car
        creeping within HOLD of one; never while the driver swerves or pulls the handbrake."""
        from dataclasses import replace

        closing = car.speed - lead
        hit = closing > 0
        if not hit.any() or c.handbrake or abs(c.steer) > SWERVE or car.speed <= 0:
            return out
        if car.speed < MOVING:  # creeping out of a jam: only a nose almost touching is held
            brake = 1.0 if gap[hit].min() < HOLD else 0.0
        else:
            need = float(np.max(closing[hit] ** 2 / (2 * np.maximum(gap[hit] - STOP_AT, 0.1))))  # m/s² to stop short
            brake = float(np.clip((need - A_SOFT) / (A_HARD - A_SOFT), 0.0, 1.0))
        if brake <= 0:
            return out
        self.lit["brake"] = True
        return replace(out, brake=max(out.brake, brake), throttle=0.0)

    def _keep_lane(self, c, car, dt: float, p: Place | None, steer_strength: float):
        """Steer toward the lane's centre ahead while the driver's wheel is near the middle."""
        from dataclasses import replace

        if p is None or car.speed < MOVING:
            self.edge = None
            return c
        if p.edge != self.edge:  # a new street: keep to the lane the car is in
            self.edge, self.lane, self.off = p.edge, p.lane, p.right
        if abs(c.steer) > CHANGE:
            if self.held == 0.0:
                self.held_lane, self.side = p.lane, np.sign(c.steer)
            self.held += dt
        if abs(c.steer) > MIX:  # the driver steers alone; the lane follows the car
            self.lane, self.off = p.lane, p.right
            return c
        if self.held >= CHANGE_HOLD:  # steering right (+) moves toward lane 0, the rightmost
            self.lane = int(np.clip(self.held_lane - self.side, 0, self.road.lanes[p.edge] - 1))
        self.held = 0.0
        e = p.edge
        target = lane_offsets(int(self.road.lanes[e]), float(self.road.lane_w[e]), bool(self.road.twoway[e]))
        self.off += float(np.clip(target[min(self.lane, len(target) - 1)] - self.off, -LANE_SPEED * dt, LANE_SPEED * dt))
        look = max(LOOK_MIN, LOOK_TIME * car.speed)
        tx, tz = self._ahead(e, p.s + look, self.off)
        alpha = (np.arctan2(tz - car.z, tx - car.x) - car.yaw + np.pi) % (2 * np.pi) - np.pi
        wheel = np.arctan(2 * np.sin(alpha) / look * self.wheelbase)  # pure pursuit: the arc through the point
        narrow = 1 - (1 - NARROW) * np.clip(car.speed / NARROW_AT, 0, 1)
        want = float(np.clip(wheel / (self.max_steer * (1 - steer_strength * (1 - narrow))), -1, 1))
        steer = c.steer + (want - c.steer) * (1 - abs(c.steer) / MIX)
        self.lit["lane"] = abs(steer - c.steer) > LIT
        return replace(c, steer=steer)

    def _ahead(self, e: int, s: float, off: float) -> tuple[float, float]:
        """The point s metres along edge e, off metres right of its centreline; past its end, on the exit the car
        takes (_exit), in the lane as far from the right."""
        g, length = self.road.graph, float(self.road.graph.a["g_len"][e])
        nxt = self._exit(e) if s > length else None
        if nxt is not None:
            lanes = lane_offsets(int(self.road.lanes[e]), float(self.road.lane_w[e]), bool(self.road.twoway[e]))
            lane = int(np.argmin(np.abs(lanes - off)))
            n = nxt[0]
            then = lane_offsets(int(self.road.lanes[n]), float(self.road.lane_w[n]), bool(self.road.twoway[n]))
            x, z, _ = g.points([n], [s - length], [then[min(lane, len(then) - 1)]])
            return float(x[0]), float(z[0])
        x, z, _ = g.points([e], [min(s, length)], [off])
        return float(x[0]), float(z[0])


def brake_reach(speed: float) -> float:
    """How far ahead the emergency brake must look at this speed: where stopping would take A_SOFT, and 10 m."""
    return speed ** 2 / (2 * A_SOFT) + STOP_AT + 10.0
