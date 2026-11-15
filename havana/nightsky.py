"""The night sky over Havana: the stars of the Yale catalogue, the planets, the moon and the Milky Way.

Every point of the sky is a camera-facing sprite on a dome centred on the camera, just inside its far plane,
so buildings hide what is behind them. The stars are fixed in the equatorial frame: once per frame one 3x3
rotation from the local sidereal time turns the whole catalogue (havana.sky.equatorial_to_local).

The sprites are drawn opaque and unlit, without fog. Their colour is the sky's colour plus their own light,
which looks the same as adding light over the sky as long as the sky behind them is one colour, as the
engine's background is: the shader reads it from the fog colour, which the atmosphere keeps equal to the
background. A per-sprite number packs what the shader needs: kind and colour in its integer part
over 16, magnitude in the rest. The material's uniform carries the sky's colour, how dark the night is and
the faintest magnitude shown, so a frame changes the whole sky with four numbers.
"""

from datetime import datetime

import numpy as np

from havana import sky
from havana.climate import SKY_DATA

DOME = 660.0  # metres from the camera: well inside the far plane (750), which clips by depth, and past the fog
MW_DOME = 675.0  # the Milky Way sits behind the stars
NEAR_DOME = 650.0  # planets and the moon in front of the stars
MOON_DEG = 0.26  # the moon's angular radius
PX_DEG = 0.085  # about one pixel at 720 lines and the game's field of view
KIND_MW, KIND_MOON, KIND_MOON_DARK = 16, 17, 18
PLANET_BV = {"mercury": 0.9, "venus": 0.8, "mars": 1.5, "jupiter": 0.8, "saturn": 1.0}

_SHADER = """
struct Globals {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    camera_pos: vec3<f32>, time: f32,
    viewport_size: vec2<f32>, origin_mod: vec2<f32>,
    ibl_intensity: f32, ibl_enabled: u32, _pad_ibl: vec2<f32>,
    light_view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>, _pad_sun0: f32,
    sun_color: vec3<f32>, sun_intensity: f32,
    shadow_enabled: u32, shadow_bias: f32, shadow_map_size: f32, shadow_pcf_radius: u32,
    spot_position: vec3<f32>, spot_range: f32,
    spot_direction: vec3<f32>, spot_cos_inner: f32,
    spot_color: vec3<f32>, spot_intensity: f32,
    spot_cos_outer: f32, shadow_caster: u32, fog_start: f32, fog_end: f32,
    fog_color: vec3<f32>, fog_enabled: u32,
};
struct MaterialUniform { night: f32, limit: f32, milky: f32, moon: f32 };
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> material: MaterialUniform;
@group(0) @binding(3) var<storage, read> scalar_values: array<f32>;
@group(0) @binding(4) var<storage, read> radii: array<f32>;
@group(0) @binding(5) var lut_texture: texture_1d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;
struct VSIn { @location(0) position: vec3<f32> };
struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) scalar: f32,
};
@vertex
fn vs_main(in: VSIn, @builtin(instance_index) iidx: u32) -> VSOut {
    let model = transforms[iidx];
    let world_center = (model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let view_center = (globals.view * vec4<f32>(world_center, 1.0)).xyz;
    let offset = vec2<f32>(in.position.x, in.position.y) * radii[iidx];
    var out: VSOut;
    out.clip_position = globals.proj * vec4<f32>(view_center.x + offset.x, view_center.y + offset.y, view_center.z, 1.0);
    out.quad_uv = in.position.xy;
    out.scalar = scalar_values[iidx];
    return out;
}
fn star_colour(bv: f32) -> vec3<f32> {
    // B-V from -0.3 (blue-white) to 1.6 (orange-red), a short ramp that reads like the eye's.
    let t = clamp((bv + 0.3) / 1.9, 0.0, 1.0);
    let blue = vec3<f32>(0.70, 0.80, 1.00);
    let white = vec3<f32>(1.00, 0.98, 0.95);
    let orange = vec3<f32>(1.00, 0.70, 0.45);
    if (t < 0.4) { return mix(blue, white, t / 0.4); }
    return mix(white, orange, (t - 0.4) / 0.6);
}
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) { discard; }
    let lut = textureSample(lut_texture, lut_sampler, 0.5);  // bound by the sprite path; unused
    let kind = floor(in.scalar / 16.0);
    let mag = in.scalar - kind * 16.0 - 4.0;
    let bg = globals.fog_color;  // the sky's colour: the background and the fog share it
    var light = vec3<f32>(0.0);
    if (kind == 17.0) {          // the moon's lit disc, seen by day too
        light = vec3<f32>(0.95, 0.93, 0.88) * material.moon;
    } else if (kind == 18.0) {   // the moon's dark part: the sky's own colour hides the lit disc
        return vec4<f32>(bg, 1.0) * (1.0 + 0.0 * lut.a);
    } else if (kind == 16.0) {   // the Milky Way: magnitude holds its brightness, 0 to 1 times 8
        let b = (mag + 4.0) / 8.0;
        light = vec3<f32>(0.80, 0.82, 0.92) * (0.05 + 0.25 * b) * material.milky;
    } else {                     // a star or a planet: kind is its colour bin
        if (mag > material.limit) { discard; }
        let b = clamp((material.limit - mag) / 3.0 + 0.12, 0.0, 1.0) * material.night;
        light = star_colour(kind / 15.0 * 1.9 - 0.3) * b * (1.0 - 0.4 * r2);
    }
    if (max(light.r, max(light.g, light.b)) < 0.004) { discard; }
    return vec4<f32>(bg + light, 1.0);
}
""".strip()


def pack(kind, mag) -> np.ndarray:
    """The per-sprite number: kind (0-15 a colour bin, or a KIND_*) times 16, plus magnitude + 4 (0-12)."""
    return (np.asarray(kind, float) * 16.0 + np.clip(np.asarray(mag, float), -4.0, 11.9) + 4.0).astype(np.float32)


def unpack(value) -> tuple[np.ndarray, np.ndarray]:
    """Kind and magnitude from pack()."""
    v = np.asarray(value, float)
    kind = np.floor(v / 16.0)
    return kind, v - kind * 16.0 - 4.0


def colour_bin(bv) -> np.ndarray:
    """B-V colour index to the shader's 16 bins."""
    return np.clip(np.round((np.asarray(bv, float) + 0.3) / 1.9 * 15.0), 0, 15)


def star_radius(mag) -> np.ndarray:
    """Sprite radius at the dome for a magnitude: a pixel or less for the faint ones, three for Sirius."""
    px = np.clip(2.6 - 0.32 * (np.asarray(mag, float) + 1.0), 0.75, 3.2)
    return DOME * np.radians(px * PX_DEG)


def load_points(when: datetime):
    """Stars (with proper motion and precession to `when`) and Milky Way points: unit vectors, packed numbers
    and radii, as one array each; stars first."""
    stars = np.load(SKY_DATA / "stars.npy")
    years = (when.year + when.timetuple().tm_yday / 365.25) - 2000.0
    ra, dec = sky.precess(stars["ra"] + stars["pm_ra"] * years, stars["dec"] + stars["pm_dec"] * years, when)
    mw = np.load(SKY_DATA / "milkyway.npy")
    mw_ra, mw_dec = sky.precess(mw["ra"], mw["dec"], when)
    v = np.vstack([sky.unit_vectors(ra, dec), sky.unit_vectors(mw_ra, mw_dec)])
    values = np.concatenate([pack(colour_bin(stars["bv"]), stars["mag"]),
                             pack(KIND_MW, mw["w"].astype(float) * 8.0 - 4.0)])
    radii = np.concatenate([star_radius(stars["mag"]), np.full(len(mw), MW_DOME * np.radians(0.11))])
    domes = np.concatenate([np.full(len(stars), DOME), np.full(len(mw), MW_DOME)])
    return v, values, radii, domes


class NightSky:
    """The sky's sprites in an engine and the system that turns them with the clock each frame."""

    def __init__(self, engine, when: datetime):
        import manifoldx as mx
        from manifoldx.components import Component, Material, Transform
        from manifoldx.sky import StarfieldMaterial
        from manifoldx.types import Float
        from manifoldx.viz import PointCloud, Radius, ScalarValue

        class NightSkyMaterial(StarfieldMaterial):
            """Unlit sky sprites over a one-colour sky; see the module's docstring."""

            def __init__(self):
                super().__init__(cmap="gray", vmin=0.0, vmax=1.0)
                self.u = np.zeros(4, np.float32)

            @classmethod
            def _compile(cls) -> str:
                return _SHADER

            @classmethod
            def uniform_type(cls):
                return {k: "f32" for k in ("night", "limit", "milky", "moon")}

            def get_data(self, n: int, registry=None) -> np.ndarray:
                return np.broadcast_to(self.u, (n, 4)).copy()

        class SkyPoint(Component):
            k: Float  # index into this object's arrays

        self.engine, self.SkyPoint = engine, SkyPoint
        self.material = NightSkyMaterial()
        self._load(when)
        self.moving_values = np.zeros(7, np.float32)  # five planets, the moon's lit disc and its dark part
        self.moving_radii = np.zeros(7)
        self._dirty = True  # values and radii to upload: on the first frame, a new day and a moving body
        n = len(self.v) + 7
        self.pos = np.zeros((n, 3))
        engine.spawn(PointCloud(), Material(self.material), Transform(pos=np.zeros((n, 3), np.float32)),
                     ScalarValue(value=np.zeros(n, np.float32)), Radius(radius=np.zeros(n, np.float32)),
                     SkyPoint(k=np.arange(n, dtype=float)), n=n)

        def place(q: mx.Query[Transform, ScalarValue, Radius, SkyPoint], dt: float):
            k = np.rint(q[SkyPoint].k.data[:, 0]).astype(np.int64)
            q[Transform].pos = self.pos[k].astype(np.float32)
            if not self._dirty:
                return
            self._dirty = False
            q[ScalarValue].value = np.concatenate([self.values, self.moving_values])[k][:, None]
            q[Radius].radius = np.concatenate([self.radii, self.moving_radii])[k].astype(np.float32)[:, None]

        engine.system(place)

    def _load(self, when: datetime) -> None:
        """The catalogue moved to `when`'s date by proper motion and precession."""
        self.v, self.values, self.radii, self.domes = load_points(when)
        self.day = when.date()

    def update(self, when: datetime, eye, look, moon: sky.Body, sun: sky.Body) -> int:
        """Turn the sky to `when` around `eye` and set what shows from the frame's look; returns the number of
        stars bright enough to show."""
        if when.date() != self.day:  # precession and proper motion move once a day
            self._load(when)
            self._dirty = True
        eye = np.asarray(eye, float)
        rot = sky.equatorial_to_local(when)
        nf = len(self.v)
        self.pos[:nf] = eye + (self.v @ rot.T) * self.domes[:, None]
        ps = sky.planets(when)
        for i, (name, p) in enumerate(ps.items()):
            self.pos[nf + i] = eye + sky.direction(p.alt, p.az) * NEAR_DOME
            self.moving_values[i] = pack(colour_bin(PLANET_BV[name]), p.mag)
            self.moving_radii[i] = star_radius(p.mag) * NEAR_DOME / DOME
        md = sky.direction(moon.alt, moon.az)
        sd = sky.direction(sun.alt, sun.az)
        away = -(sd - md * (sd @ md))  # across the moon's face, away from the sun
        away /= max(np.linalg.norm(away), 1e-9)
        r = NEAR_DOME * np.radians(MOON_DEG)
        self.pos[nf + 5] = eye + md * NEAR_DOME
        self.pos[nf + 6] = eye + md * (NEAR_DOME - 1.0) + away * (2.0 * r * moon.lit)
        self.moving_values[5], self.moving_values[6] = pack(KIND_MOON, 0.0), pack(KIND_MOON_DARK, 0.0)
        self.moving_radii[5] = r
        self.moving_radii[6] = r * 1.02 if moon.lit < 0.97 else 0.0
        key = (np.round(self.moving_values, 2).tobytes(), np.round(self.moving_radii, 2).tobytes())
        if key != getattr(self, "_moving_key", None):
            self._moving_key, self._dirty = key, True
        milky = float(np.clip((look.star_limit - 5.3) / 1.0, 0.0, 1.0)) * look.night
        moon_light = float(np.clip(1.0 - 0.6 * (1.0 - look.night), 0.25, 1.0)) if moon.alt > -1 else 0.0
        self.material.u[:] = (look.night, look.star_limit, milky, moon_light)
        kind, mag = unpack(self.values)
        up = self.pos[:nf, 1] > eye[1]
        return int(((kind < KIND_MW) & (mag <= look.star_limit) & up).sum())
