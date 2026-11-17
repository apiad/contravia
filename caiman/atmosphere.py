"""What the sky and the weather look like: light, sky colour, fog, ambient light, shadows and the star limit.

`look()` is a pure function of the sun, the moon, the weather and the settings, so it is tested without a
renderer. `Atmosphere` hands a Look to the engine each frame.

The ambient light comes from an environment map, whose precompute costs seconds of CPU. So there is a small
palette of environments (clear day, overcast, golden hour, twilight, night), each computed once and cached on
disk by manifoldx, and the look picks one and scales its intensity smoothly; the sky's own colour is the
engine's background, which changes every frame for free.
"""

from dataclasses import dataclass

import numpy as np

from caiman.sky import Body, direction
from caiman.weather import Now, SkyConfig

SHADOW_EXTENT = 150.0  # half the side of the square around the player that casts shadows, in metres
SHADOW_RES = 2048
FOG_CLEAR = (2000.0, 12000.0)
FOG_STORM = (90.0, 220.0)

# Palette of ambient environments: (zenith, horizon, ground) as sRGB colours, as a painter picks them; the
# engine takes linear colour, so linear() converts them on the way out.
PALETTE = {
    "day": ((0.30, 0.52, 0.86), (0.58, 0.72, 0.90), (0.35, 0.33, 0.30)),
    "overcast": ((0.55, 0.57, 0.60), (0.66, 0.67, 0.68), (0.30, 0.30, 0.30)),
    "golden": ((0.30, 0.38, 0.62), (0.95, 0.62, 0.38), (0.30, 0.24, 0.20)),
    "twilight": ((0.08, 0.10, 0.25), (0.45, 0.30, 0.35), (0.08, 0.07, 0.08)),
    "night": ((0.02, 0.03, 0.07), (0.06, 0.06, 0.08), (0.03, 0.03, 0.03)),
}


@dataclass(frozen=True)
class Look:
    """Everything the engine needs from the sky for one frame."""

    light_dir: tuple  # the direction light travels (toward the ground), game frame
    light_color: tuple
    light_intensity: float
    zenith: tuple  # linear colour
    horizon: tuple  # linear colour
    fog: tuple  # start, end in metres
    env: str  # a PALETTE key
    env_intensity: float
    shadows: bool
    star_limit: float  # faintest magnitude shown; below about -2 nothing shows
    night: float  # 0 by day, 1 in full night: how much the night sky shows


def linear(c) -> tuple:
    """sRGB colour to the linear colour the engine's background, fog and environments take."""
    return tuple(float(((v + 0.055) / 1.055) ** 2.4) if v > 0.04045 else float(v / 12.92) for v in c)


def _mix(a, b, t):
    t = min(1.0, max(0.0, t))
    return tuple(float(x + (y - x) * t) for x, y in zip(a, b, strict=True))


def _smooth(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def clear_sky_factor(cloud: float) -> float:
    """The share of direct sunlight left under `cloud` cover (Kasten and Czeplak, 1980)."""
    return 1.0 - 0.75 * cloud ** 3.4


def look(sun: Body, moon: Body, now: Now, config: SkyConfig) -> Look:
    """The frame's light and sky from where the sun and moon are and from the weather."""
    h, c = sun.alt, now.cloud
    kc = clear_sky_factor(c)
    day = _smooth(-6.0, 8.0, h)  # 0 in full dark, 1 a little after sunrise
    night = 1.0 - _smooth(-18.0, -6.0, h)

    # Direct light: the sun above -2 degrees, else the moon if it is up, else a faint glow from the zenith.
    if h > -2.0:
        warm = _smooth(0.0, 25.0, h)
        color = _mix((1.0, 0.45, 0.22), (1.0, 0.95, 0.87), warm)
        intensity = 3.6 * _smooth(-2.0, 8.0, h) * (0.55 + 0.45 * min(1.0, h / 45.0)) * kc
        d = direction(h, sun.az)
    elif moon.alt > 0.0:
        color = (0.62, 0.72, 1.0)
        intensity = 0.25 * moon.lit * _smooth(0.0, 10.0, moon.alt) * kc
        d = direction(moon.alt, moon.az)
    else:
        color, intensity, d = (0.6, 0.65, 0.8), 0.0, np.array([0.0, 1.0, 0.0])
    light_dir = tuple(float(v) for v in -np.asarray(d))

    # The sky's colour: day blue, golden and red near sunrise and sunset, deep blue at night; grey with cloud.
    golden = _smooth(12.0, 2.0, h) * _smooth(-6.0, -1.0, h)
    zenith = _mix(PALETTE["night"][0], PALETTE["day"][0], day)
    horizon = _mix(PALETTE["night"][1], PALETTE["day"][1], day)
    horizon = _mix(horizon, (0.95, 0.62, 0.42), 0.6 * golden * (1 - c))
    zenith = _mix(zenith, PALETTE["twilight"][0], golden * 0.5)
    grey = _mix((0.05, 0.05, 0.06), (0.62, 0.63, 0.65), day)
    cover = min(1.0, c * 0.9 + 0.3 * now.rain)
    zenith, horizon = _mix(zenith, grey, cover), _mix(horizon, grey, cover * 0.9)
    glow = 0.035 * config.city_glow * night
    horizon = tuple(v + glow * w for v, w in zip(horizon, (1.0, 0.75, 0.5), strict=True))

    # Ambient: one environment of the palette, its intensity following the light.
    if day > 0.6:
        env = "overcast" if c > 0.7 else ("golden" if golden > 0.4 else "day")
    elif day > 0.05:
        env = "twilight" if c < 0.7 else "overcast"
    else:
        env = "night"
    env_intensity = 0.12 + 0.55 * day
    if env == "night":
        env_intensity = 0.06 + 0.10 * config.city_glow + (0.25 * moon.lit if moon.alt > 0 else 0.0)

    fog_mix = max(now.rain, 0.6 * max(0.0, c - 0.8) / 0.2)
    fog = _mix(FOG_CLEAR, FOG_STORM, fog_mix)

    # Stars: the darkest sky shows magnitude 6.5; city glow, moonlight, twilight and cloud take stars away.
    base = config.star_limit if config.star_limit is not None else 6.5 - 2.0 * min(1.0, config.city_glow)
    limit = base - (1.5 * moon.lit if moon.alt > 0 else 0.0) - 6.0 * (1.0 - night) - 6.0 * c ** 1.5
    shadows = h > 3.0 and c < 0.85 and now.rain < 0.3

    return Look(light_dir, color, float(intensity), linear(zenith), linear(horizon), fog, env, float(env_intensity), bool(shadows),
                float(limit), float(night))


def light_basis(light_dir) -> np.ndarray:
    """Two unit vectors across the light's direction: the axes of the shadow map's texels."""
    d = np.asarray(light_dir, float)
    d = d / np.linalg.norm(d)
    up = np.array([0.0, 0.0, 1.0]) if abs(d[1]) > 0.999 else np.array([0.0, 1.0, 0.0])  # as manifoldx's _look_at
    u = np.cross(up, d)
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(d, u)])


def snap_target(eye, light_dir, extent: float = SHADOW_EXTENT, res: int = SHADOW_RES) -> tuple:
    """The shadow box's centre near `eye`, moved onto whole texels across the light, so still shadows stay
    still as the player moves."""
    texel = 2.0 * extent / res
    basis = light_basis(light_dir)
    p = np.asarray(eye, float)
    along = basis @ p
    snapped = np.round(along / texel) * texel
    return tuple(float(v) for v in p + basis.T @ (snapped - along))


class Atmosphere:
    """Applies a Look to the engine: sun, background, fog, ambient light and the shadow box."""

    def __init__(self, engine):
        from manifoldx.ibl import EnvironmentMap

        self.engine = engine
        self.envs = {k: EnvironmentMap.from_sky(*(linear(c) for c in v)) for k, v in PALETTE.items()}
        self._tasks = {}
        self.env_key = None
        self._fallback = "day"
        engine.fog_enabled = True

    def _ready(self, key: str) -> bool:
        """Start the environment's precompute in the background; True once it is done."""
        env = self.envs[key]
        if env._computed:
            return True
        if key not in self._tasks:
            self._tasks[key] = self.engine.background(env._precompute)()
        return False

    def prepare(self, wait: bool = False) -> None:
        """Start every palette environment's precompute; with wait, block until all are done (offline renders)."""
        for key in self.envs:
            self._ready(key)
        if wait:
            for task in self._tasks.values():
                task.wait()

    def apply(self, lk: Look, eye) -> None:
        from manifoldx.resources import DirectionalLight

        e = self.engine
        hexcolor = "#" + "".join(f"{round(min(1.0, max(0.0, v)) * 255):02x}" for v in lk.light_color)
        e.set_sun(DirectionalLight(color=hexcolor, intensity=lk.light_intensity, direction=lk.light_dir))
        e.background_color = lk.horizon  # the night sky's sprites read the sky's colour from the fog's
        e.fog_enabled, e.fog_start, e.fog_end, e.fog_color = True, lk.fog[0], lk.fog[1], lk.horizon
        key = lk.env if self._ready(lk.env) else (self.env_key or self._fallback)
        if key != self.env_key and self.envs[key]._computed:
            e.set_environment(self.envs[key])
            self.env_key = key
        if self.env_key:
            self.envs[self.env_key].intensity = lk.env_intensity
        if lk.shadows:
            target = snap_target(eye, lk.light_dir)
            e.enable_shadows(target=target, extent=SHADOW_EXTENT, resolution=SHADOW_RES, near=0.01, far=600.0,
                             bias=0.002, pcf_radius=1, auto_fit=False)
        else:
            e._shadow_config = None  # manifoldx has no public switch to turn shadows off


def sun_and_moon_text(sun: Body, moon: Body) -> dict:
    """The sun's and moon's heights for engine.state, rounded."""
    return {"sun_alt": round(sun.alt, 1), "sun_az": round(sun.az, 1), "moon_alt": round(moon.alt, 1),
            "moon_lit": round(moon.lit, 2)}

