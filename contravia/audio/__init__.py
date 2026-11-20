"""The game's sound: every sound synthesized while the game runs, played by manifoldx.audio.

The game writes controls once per frame (the car's rpm, the listener's pose, the weather) and
fires events (a crash, a horn); a sound process builds Euterpe graphs from the factories in
this package, mixes them into five buses and plays them. One process lives as long as the
interpreter: a reload under contravia dev keeps it and re-imports these modules in it.
"""

from __future__ import annotations

import math
import time

from manifoldx.audio import Audio, pose_names

BUSES = ("engine", "effects", "ambience", "music", "ui")
ROWS = {"engine": "vol_engine", "effects": "vol_effects", "ambience": "vol_ambience", "music": "vol_music"}
PAUSED_MUSIC = 10 ** (-10 / 20)  # the pause lowers the music by 10 dB
DRIVING = ("drive", "aim")  # modes in which the car and the world are heard
MODULES = ["contravia.audio"]  # re-imported in the sound process on a reload

# Every control the game writes. A process started with fewer (by older code) is replaced.
CONTROLS = (*pose_names("car"),)


class Sound:
    """The game's handle on its sound process."""

    def __init__(self, audio: Audio, names: tuple):
        self.audio, self.names = audio, names
        self._mode, self._gains, self._published = None, {b: 1.0 for b in BUSES}, 0.0

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
        s.register()
        s.apply_options(options)
        return s

    def register(self) -> None:
        """Register every source and voice in the sound process (again, after a reload)."""

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

    def frame(self, game) -> None:
        """Once per frame: the controls, the listener and the mode's gains."""
        a = self.audio
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
        a.tick()
        now = time.monotonic()
        if now - self._published > 0.5:
            self._published = now
            game.engine.state.audio = a.state()

    def close(self) -> None:
        self.audio.close()
