"""The game's sound: every sound synthesized while the game runs, played by manifoldx.audio.

The game writes controls once per frame (the car's rpm, the listener's pose, the weather) and
fires events (a crash, a horn); a sound process builds Euterpe graphs from the factories in
this package, mixes them into five buses and plays them. One process lives as long as the
interpreter: a reload under contravia dev keeps it and re-imports these modules in it.
"""

from __future__ import annotations

import functools
import math
import time
import zlib

import numpy as np
from manifoldx.audio import Audio, pose_names

BUSES = ("engine", "effects", "ambience", "music", "ui")
ROWS = {"engine": "vol_engine", "effects": "vol_effects", "ambience": "vol_ambience", "music": "vol_music"}
PAUSED_MUSIC = 10 ** (-10 / 20)  # the pause lowers the music by 10 dB
DRIVING = ("drive", "aim")  # modes in which the car and the world are heard
MODULES = ["contravia.rules.sound", "contravia.audio.engine", "contravia.audio.car", "contravia.audio.weather",
           "contravia.audio.traffic", "contravia.audio"]  # re-imported in the sound process on a reload

# Every control the game writes. A process started with fewer (by older code) is replaced.
CONTROLS = (*pose_names("car"), "car.rpm", "car.throttle", "car.speed", "car.slip", "car.wet", "car.surface",
            "car.horn", "car.cabin", "car.under", "weather.rain", "traffic.count",
            *(f"traffic.{i}.{k}" for i in range(4) for k in ("x", "y", "z", "vx", "vy", "vz", "rpm", "cyl", "bright", "on")))


class Sound:
    """The game's handle on its sound process."""

    def __init__(self, audio: Audio, names: tuple):
        self.audio, self.names = audio, names
        self._mode, self._gains, self._published = None, {b: 1.0 for b in BUSES}, 0.0
        self._frame, self._crash_at, self._bump_at, self._travel = 0, -100, -100, None
        self._surface_at, self._surface_code, self._horn_was = 0.0, 0.0, False
        from contravia.audio.weather import Thunder

        self._thunder, self._under = Thunder(seed=7), False
        from contravia.audio.traffic import Honks

        self._honks, self._traffic_v, self._traffic_slow = Honks(), np.zeros(0), np.zeros(0, bool)

    @classmethod
    def start(cls, state, options, backend: str = "device", path=None) -> Sound:
        """The sound kept in state from before a reload, or a new one. backend: device, null or
        file (with path, and a stem per bus)."""
        old = getattr(state, "sound", None)
        if old is not None and old.audio.alive and set(CONTROLS) <= set(old.names):
            old.audio.reload(MODULES)
            s = old
        else:
            if old is not None:
                old.close()
            audio = Audio(backend=backend, path=path, stems=backend == "file", controls=CONTROLS)
            s = cls(audio, CONTROLS)
            state.sound = s
        s.apply_options(options)
        return s

    def attach(self, game) -> None:
        """Register every source and voice of this game in the sound process: at the start, after a
        reload, and again when the player takes another vehicle."""
        from contravia.audio import car

        a = self.audio
        a.source("roll", car.roll, "effects")  # the camera follows a few metres behind: only the engine is placed
        a.source("squeal", car.squeal, "effects")
        a.source("wind", car.wind, "effects")
        a.voice("crash", car.crash, "effects", cap=4, placed=True)
        a.voice("bump", car.bump, "effects", cap=2, placed=True)
        a.voice("bell", car.bell, "effects", cap=1, placed=True)
        from contravia.audio import weather

        a.source("rain", weather.rain, "ambience")
        a.source("bubbles", weather.bubbles, "effects")
        a.voice("thunder", weather.thunder, "ambience", cap=3)
        a.voice("splash", weather.splash, "effects", cap=1, placed=True)
        from contravia.audio import traffic

        for i in range(traffic.SLOTS):
            a.source(f"traffic.{i}", functools.partial(traffic.cheap_engine, i), "effects", placed=(f"traffic.{i}", 5.0, 200.0))
        a.source("traffic.bed", traffic.bed, "ambience")
        a.voice("honk", traffic.honk, "effects", cap=3, placed=True)
        a.voice("brakes", traffic.brakes, "effects", cap=2, placed=True)
        self.vehicle(game)

    def vehicle(self, game) -> None:
        """The player's engine, from its family's sound and its physics' idle and peak-power rpm."""
        from contravia.audio.engine import player_engine

        e = game.driven.spec.engine if game.driven is not None else None
        idle, red = (e.idle_rpm, e.peak_power_rpm * 1.15) if e is not None else (800.0, 5500.0)
        seed = zlib.crc32(game.family.encode())
        self.audio.source("engine", functools.partial(player_engine, game.family, float(idle), float(red), seed),
                          "engine", placed=("car", 5.0, 300.0))
        from contravia.audio import car

        self.audio.source("horn", functools.partial(car.horn, game.family), "effects")
        self._bicycle = game.family in ("bicitaxi", "bicicleta")

    @staticmethod
    def gains(options) -> dict:
        """Each bus's gain from the volume rows: general / 10 times the bus's row / 10."""
        general = options["vol"].value / 10
        g = {bus: general * options[row].value / 10 for bus, row in ROWS.items()}
        g["ui"] = general
        return g

    @staticmethod
    def mode_gains(mode: str, gains: dict) -> dict:
        """Outside the drive the car and its effects go quiet; the pause also lowers the music."""
        if mode in DRIVING:
            return dict(gains)
        g = dict(gains, engine=0.0, effects=0.0)
        if mode == "paused":
            g["music"] = gains["music"] * PAUSED_MUSIC
        return g

    def apply_options(self, options) -> None:
        self._gains = self.gains(options)
        self._mode = None  # apply on the next frame

    def frame(self, game, now: float | None = None) -> None:
        """Once per frame: the controls, the listener and the mode's gains. now: seconds on a clock
        of the caller's (the wall clock by default), for the thunder."""
        a = self.audio
        now = time.monotonic() if now is None else now
        if game.mode != self._mode:
            self._mode = game.mode
            for bus, g in self.mode_gains(game.mode, self._gains).items():
                a.gain(bus, g)
        cam = game.engine.camera
        p, t = cam.position, cam.target
        a.set("listener.x", float(p[0]))
        a.set("listener.y", float(p[1]))
        a.set("listener.z", float(p[2]))
        a.set("listener.yaw", math.atan2(float(t[2] - p[2]), float(t[0] - p[0])))
        car = game.car
        a.set("car.x", car.x)
        a.set("car.y", car.y)
        a.set("car.z", car.z)
        a.set("car.vx", car.speed * math.cos(car.yaw))
        a.set("car.vz", car.speed * math.sin(car.yaw))
        a.set("car.rpm", car.rpm)
        a.set("car.speed", abs(car.speed))
        c = getattr(game, "controls_now", None)
        a.set("car.throttle", 0.0 if c is None or getattr(game, "sinking", None) is not None else c.throttle)
        a.set("car.slip", slip_past_peak(game))
        now_w = getattr(game, "weather_now", None)
        a.set("car.wet", now_w.wet if now_w is not None else 0.0)
        a.set("car.surface", self._surface(game))
        horn = game.mode == "drive" and game.engine.input.is_pressed("h") if hasattr(game.engine, "input") else False
        if horn and getattr(self, "_bicycle", False):
            if not self._horn_was:
                a.event("bell", seed=self._frame, x=car.x, y=car.y, z=car.z)
            horn_on = 0.0
        else:
            horn_on = 1.0 if horn else 0.0
        self._horn_was = horn
        a.set("car.horn", horn_on)
        a.set("car.cabin", 0.0)  # the camera follows the car from outside; there is no cabin view yet
        self._events(game)
        self._weather(game, now)
        self._traffic(game)
        a.tick()
        if now - self._published > 0.5:
            self._published = now
            game.engine.state.audio = a.state()

    def _surface(self, game) -> float:
        """0 on a street, 2 off it; checked four times a second."""
        now = time.monotonic()
        if now - self._surface_at > 0.25:
            self._surface_at = now
            help = getattr(game, "help", None)
            on = help is not None and help.road.locate(game.car.x, game.car.z, game.car.yaw) is not None
            self._surface_code = 0.0 if on or help is None else 2.0
        return self._surface_code

    def _events(self, game) -> None:
        """A crash from the frame's contact impulse, a bump from the suspension's travel."""
        self._frame += 1
        t = getattr(game.car, "telemetry", None) or {}
        car, a = game.car, self.audio
        impulse = float(t.get("impulse", 0.0))
        from contravia.audio.car import CRASH_MIN

        if impulse >= CRASH_MIN and self._frame - self._crash_at > 6:
            self._crash_at = self._frame
            a.event("crash", impulse=impulse, seed=self._frame, x=car.x, y=car.y, z=car.z)
            a.duck("music", 6.0, 1.5)
        travel = t.get("travel")
        if travel is not None:
            travel = list(travel)
            if self._travel is not None and len(travel) == len(self._travel):
                jump = max(abs(x - y) for x, y in zip(travel, self._travel))
                if jump > 0.02 and self._frame - self._bump_at > 10:
                    self._bump_at = self._frame
                    a.event("bump", depth=jump, seed=self._frame, x=car.x, y=car.y, z=car.z)
            self._travel = travel

    def _weather(self, game, now: float) -> None:
        """The rain's level, thunder in a storm, and the sea: a splash, then muffled while sinking."""
        a = self.audio
        w = getattr(game, "weather_now", None)
        a.set("weather.rain", w.rain if w is not None else 0.0)
        for d in self._thunder.due(now, w.state if w is not None else "clear"):
            a.event("thunder", distance=d, seed=int(now * 1000) & 0xFFFF)
        under = getattr(game, "sinking", None) is not None
        if under != self._under:
            self._under = under
            a.lowpass(400.0 if under else 20000.0)
            if under:
                a.event("splash", seed=self._frame, x=game.car.x, y=game.car.y, z=game.car.z)
        a.set("car.under", 1.0 if under else 0.0)

    def _traffic(self, game) -> None:
        """The four loudest traffic cars into their slots, the bed's count, horns and air brakes."""
        from contravia.audio import traffic
        from contravia.rules.traffic import looks

        a, t = self.audio, getattr(game, "traffic_now", None)
        if t is None or not len(t[0]):
            a.set("traffic.count", 0.0)
            for i in range(traffic.SLOTS):
                a.set(f"traffic.{i}.on", 0.0)
            return
        xyz, head, v, look, wait = t
        table = looks()
        kinds = np.array([table[k][0] for k in look])
        lis = np.array([a.controls.get("listener.x"), a.controls.get("listener.y"), a.controls.get("listener.z")])
        if len(self._traffic_v) != len(v):
            self._traffic_v = v
        acc = (v - self._traffic_v) * 60.0  # m/s² at about 60 frames a second
        self._traffic_v = v
        pick = traffic.loudest(xyz, v, kinds, lis)
        for i in range(traffic.SLOTS):
            p = f"traffic.{i}."
            if i >= len(pick):
                a.set(p + "on", 0.0)
                continue
            k = pick[i]
            cyl, bright = traffic.slot_numbers(table[look[k]][1])
            for name, val in (("x", xyz[k, 0]), ("y", xyz[k, 1]), ("z", xyz[k, 2]), ("vx", v[k] * math.cos(head[k])),
                              ("vy", 0.0), ("vz", v[k] * math.sin(head[k])),
                              ("rpm", traffic.virtual_rpm(v[k], acc[k], kinds[k])), ("cyl", cyl), ("bright", bright),
                              ("on", 1.0)):
                a.set(p + name, float(val))
        car = game.car
        d = np.hypot(xyz[:, 0] - car.x, xyz[:, 2] - car.z)
        a.set("traffic.count", float(max(0, int((d < traffic.BED_NEAR).sum()) - len(pick))))
        stopped = abs(car.speed) < 1.0
        for k in self._honks.due(wait, (d < 30.0) & stopped):
            a.event("honk", family=table[look[k]][1], seed=self._frame + int(k), x=float(xyz[k, 0]),
                    y=float(xyz[k, 1]), z=float(xyz[k, 2]))
        if len(self._traffic_slow) == len(v):
            for k in np.flatnonzero((kinds == "bus") & (v < 0.5) & ~self._traffic_slow):
                a.event("brakes", seed=self._frame + int(k), x=float(xyz[k, 0]), y=float(xyz[k, 1]), z=float(xyz[k, 2]))
        self._traffic_slow = v < 0.5

    def close(self) -> None:
        self.audio.close()


def slip_past_peak(game) -> float:
    """How far the worst wheel is past its tyre's peak slip (0 below the peak), from the telemetry."""
    t = getattr(game.car, "telemetry", None) or {}
    slip = t.get("slip")
    if not slip or game.driven is None:
        return 0.0
    p = game.driven.fleet.p
    worst = 0.0
    for w, (k, a) in enumerate(slip):
        worst = max(worst, abs(k) / float(p.kappa_peak[0, w]), abs(a) / float(p.alpha_peak[0, w]))
    return max(0.0, worst - 1.0)
