"""Rain around the camera: thin streaks falling in a box that follows the player.

A fixed pool of drops lives in a 60 m box centred on the camera. Each frame they fall and drift with the wind;
a drop that leaves the box comes back on the other side, so the box always holds the same rain. Rain
intensity (0-1) decides how many of the pool show. The streaks are sprites stretched along the fall, drawn
unlit in the colour of the light, like the night sky's sprites.
"""

import numpy as np

POOL = 3000
BOX = 60.0  # metres, the side of the box around the camera
FALL = 9.0  # metres per second
STREAK = 0.9  # half the streak's length, metres

_SHADER = """
struct Globals {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    camera_pos: vec3<f32>, _pad0: f32,
    viewport_size: vec2<f32>, _pad1: vec2<f32>,
};
struct MaterialUniform { r: f32, g: f32, b: f32, stretch: f32 };
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> material: MaterialUniform;
@group(0) @binding(3) var<storage, read> scalar_values: array<f32>;
@group(0) @binding(4) var<storage, read> radii: array<f32>;
@group(0) @binding(5) var lut_texture: texture_1d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;
struct VSIn { @location(0) position: vec3<f32> };
struct VSOut { @builtin(position) clip_position: vec4<f32>, @location(0) quad_uv: vec2<f32> };
@vertex
fn vs_main(in: VSIn, @builtin(instance_index) iidx: u32) -> VSOut {
    let world_center = (transforms[iidx] * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let view_center = (globals.view * vec4<f32>(world_center, 1.0)).xyz;
    let r = radii[iidx];
    let offset = vec2<f32>(in.position.x * r * 0.03, in.position.y * r * material.stretch);
    var out: VSOut;
    out.clip_position = globals.proj * vec4<f32>(view_center.xy + offset, view_center.z, 1.0);
    out.quad_uv = in.position.xy;
    return out;
}
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let lut = textureSample(lut_texture, lut_sampler, 0.5);  // bound by the sprite path; unused
    if (abs(in.quad_uv.x) > 0.8) { discard; }
    return vec4<f32>(material.r, material.g, material.b, 1.0 + 0.0 * lut.a);
}
""".strip()


def fall(pos: np.ndarray, dt: float, wind=(1.5, 0.0)) -> np.ndarray:
    """Drops moved by one step of falling and drifting; positions are relative to the box's centre."""
    p = pos + np.array([wind[0], -FALL, wind[1]]) * dt
    return (p + BOX / 2) % BOX - BOX / 2


def shown(intensity: float) -> int:
    """How many drops of the pool show at this rain intensity."""
    return round(POOL * min(1.0, max(0.0, intensity)))


class Rain:
    """The pool of drops in an engine and the system that moves them each frame."""

    def __init__(self, engine, seed: int = 0):
        import manifoldx as mx
        from manifoldx.components import Component, Material, Transform
        from manifoldx.sky import StarfieldMaterial
        from manifoldx.types import Float
        from manifoldx.viz import PointCloud, Radius, ScalarValue

        class RainMaterial(StarfieldMaterial):
            """Thin unlit streaks in one colour."""

            def __init__(self):
                super().__init__(cmap="gray", vmin=0.0, vmax=1.0)
                self.u = np.array([0.7, 0.72, 0.75, 1.0], np.float32)

            @classmethod
            def _compile(cls) -> str:
                return _SHADER

            @classmethod
            def uniform_type(cls):
                return {"r": "f32", "g": "f32", "b": "f32", "stretch": "f32"}

            def get_data(self, n: int, registry=None) -> np.ndarray:
                return np.broadcast_to(self.u, (n, 4)).copy()

        class Drop(Component):
            k: Float

        rng = np.random.default_rng(seed)
        self.rel = rng.uniform(-BOX / 2, BOX / 2, (POOL, 3))
        self.material = RainMaterial()
        self.center = np.zeros(3)
        self.count = 0
        engine.spawn(PointCloud(), Material(self.material), Transform(pos=np.zeros((POOL, 3), np.float32)),
                     ScalarValue(value=np.zeros(POOL, np.float32)), Radius(radius=np.full(POOL, STREAK, np.float32)),
                     Drop(k=np.arange(POOL, dtype=float)), n=POOL)

        def move(q: mx.Query[Transform, Drop], dt: float):
            k = np.rint(q[Drop].k.data[:, 0]).astype(np.int64)
            pos = self.center + self.rel[k]
            pos[k >= self.count] = (0.0, -1e4, 0.0)  # drops not showing go far below the ground
            q[Transform].pos = pos.astype(np.float32)

        engine.system(move)

    def update(self, dt: float, eye, intensity: float, light) -> None:
        """Move the drops, follow the camera, show as many as the rain asks, in the frame's light."""
        self.rel = fall(self.rel, dt)
        self.center = np.asarray(eye, float)
        self.count = shown(intensity)
        self.material.u[:3] = np.clip(np.asarray(light, np.float32), 0.05, 0.9)
