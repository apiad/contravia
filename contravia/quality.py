"""How good the game looks against how fast it runs: every knob that trades one for the other, in one place.

A Quality holds one value per knob. Its defaults are Alto, which is the game as it was before the presets: the
streamer's detail, the far plane, the trees, grass, rain, shadows and minimap as their modules' constants set
them. PRESETS gives five: Bajo for a laptop, Medio, Alto, Ultra for a high-end card, and Cine, which only the
capture tool uses and which ignores the frame time. A player who turns one knob alone plays "Personalizado":
a preset with some knobs changed, each to a value some preset uses (its ladder).

The effects manifoldx does not have yet (cascaded shadows, ambient occlusion, bloom, a reflecting sea and the
rest, spec docs/superpowers/specs/2026-11-21-graphics-quality-design.html) become fields here as they land.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class Quality:
    """One value per knob; the defaults are Alto, today's game."""

    tau: tuple[float, float] = (12.0, 48.0)  # the streamer's screen error in pixels: the finest and the coarsest
    far: float = 15000.0  # the camera's far plane, metres
    fog_clear: tuple[float, float] = (2000.0, 12000.0)  # where the clear-weather fog starts and ends, metres
    detail_radius: int = 1  # cells around the car's with cornices, balconies and roof tanks: 1 is 3 x 3
    detail_noise: bool = True  # wear, grain and tufts from noise in the shader on the cells' ground
    tree_lod: tuple[float, float] = (60.0, 160.0)  # metres where a tree drops to LOD 1, then to LOD 2
    foliage_near: float = 25.0  # metres within which a LOD 0 tree carries its small leaves; 0 turns them off
    grass: tuple[float, float, int, int] = (20.0, 45.0, 60, 16)  # dense to, sparse to (m); blades dense, sparse
    grave_lod: tuple[float, float] = (45.0, 140.0)  # metres where a cemetery's item drops to level 1, then 2
    traffic_radius: float = 400.0  # metres round the player within which the car layer drives cars
    shadow: tuple[float, int, int] = (150.0, 2048, 1)  # half side of the shadowed square (m), map size, PCF radius
    rain_pool: int = 3000  # rain streaks at the heaviest rain
    minimap: tuple[int, int] = (256, 4)  # the minimap's texture side in pixels, fewest frames between two renders
    target_ms: float = 20.0  # the frame time the streamer's tau is tuned toward
    tune: bool = True  # whether tau follows the frame time within its range

    def changed(self, **knobs) -> Quality:
        """This quality with some knobs turned, each to a value of its ladder; anything else is a ValueError."""
        names = {f.name for f in fields(self)}
        for k, v in knobs.items():
            if k not in names or k not in KNOBS:
                raise ValueError(f"unknown knob {k!r}")
            v = tuple(v) if isinstance(v, list) else v
            values = ladder(k)
            if v not in values:
                raise ValueError(f"{k} cannot be {v!r}: it takes {values}")
            knobs[k] = values[values.index(v)]  # the ladder's own value: 90 for 90.0, an int where it is one
        return replace(self, **knobs)

    def diff(self, base: Quality) -> dict:
        """The knobs whose value differs from base's, by name."""
        mine, theirs = asdict(self), asdict(base)
        return {k: v for k, v in mine.items() if v != theirs[k]}


PRESETS = {
    "bajo": Quality(tau=(24.0, 72.0), far=7000.0, fog_clear=(1500.0, 6000.0), detail_radius=0, detail_noise=False,
                    tree_lod=(35.0, 90.0), foliage_near=0.0, grass=(0.0, 25.0, 60, 10), grave_lod=(30.0, 90.0),
                    traffic_radius=200.0, shadow=(80.0, 1024, 0), rain_pool=1000, minimap=(192, 8),
                    target_ms=33.3),
    "medio": Quality(tau=(16.0, 64.0), far=11000.0, fog_clear=(2000.0, 9000.0), tree_lod=(45.0, 120.0),
                     foliage_near=15.0, grass=(12.0, 30.0, 40, 12), grave_lod=(40.0, 120.0), traffic_radius=300.0,
                     rain_pool=2000, target_ms=25.0),
    "alto": Quality(),
    "ultra": Quality(tau=(8.0, 32.0), detail_radius=2, tree_lod=(90.0, 240.0), foliage_near=40.0,
                     grass=(30.0, 70.0, 90, 24), grave_lod=(70.0, 200.0), rain_pool=6000, minimap=(384, 2),
                     target_ms=16.7),
    "cine": Quality(tau=(5.0, 5.0), detail_radius=2, tree_lod=(120.0, 320.0), foliage_near=60.0,
                    grass=(40.0, 90.0, 120, 32), grave_lod=(90.0, 260.0), rain_pool=10000, tune=False),
}
NAMES = {"bajo": "Bajo", "medio": "Medio", "alto": "Alto", "ultra": "Ultra", "cine": "Cine"}
PLAYER = ("bajo", "medio", "alto", "ultra")  # the presets the options offer; Cine is for the capture tool

# The knobs Personalizado turns, with their labels, in the order its screen lists them.
KNOBS = {
    "tau": "Detalle de la ciudad",
    "far": "Distancia",
    "fog_clear": "Niebla",
    "detail_radius": "Adornos de fachadas",
    "detail_noise": "Textura del suelo",
    "tree_lod": "Árboles",
    "foliage_near": "Hojas finas",
    "grass": "Hierba",
    "grave_lod": "Cementerios",
    "traffic_radius": "Tráfico cercano",
    "shadow": "Sombras",
    "rain_pool": "Gotas de lluvia",
    "minimap": "Minimapa",
}


def ladder(knob: str) -> list:
    """The values a knob takes over the presets, each once, in the presets' order from Bajo to Cine."""
    out = []
    for q in PRESETS.values():
        v = getattr(q, knob)
        if v not in out:
            out.append(v)
    return out


def parse(changes: dict, current: Quality) -> tuple[str, Quality]:
    """What `manifoldx emit quality` asks for: {"preset": "ultra"} is that preset, knobs with values are the
    current quality with them turned ("personalizado"). A bad name or value raises ValueError; nothing is applied."""
    changes = dict(changes or {})
    if "preset" in changes:
        name = changes.pop("preset")
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}: one of {', '.join(PRESETS)}")
        if changes:
            raise ValueError("give a preset or knobs, not both")
        return name, PRESETS[name]
    if not changes:
        raise ValueError("nothing to change: give preset=... or a knob")
    return "personalizado", current.changed(**changes)


def apply(game, q: Quality, name: str, pinned: bool = False) -> None:
    """Put a quality into a running game: the streamer's tau range, the far plane, the detail ring, the tuning
    target, the traffic's reach, the ground's noise for nodes built from now on, and every module's knobs.
    pinned keeps it over the saved options (a --quality flag or an emitted change) across a reload."""
    from manifoldx.resources import StandardMaterial

    game.quality, game.quality_name = q, name
    game.tau_range = tuple(q.tau)
    game.streamer.tau = min(max(game.streamer.tau, q.tau[0]), q.tau[1])
    game.far, game.detail_radius, game.target_ms, game.tune = q.far, q.detail_radius, q.target_ms, q.tune
    game.engine.camera.far = q.far
    if game.layer is not None:
        game.layer.radius = q.traffic_radius
    if getattr(game.streamer.material, "detail", None) != q.detail_noise:
        game.streamer.material = StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True, faded=True,
                                                  detail=q.detail_noise)
    for part in (game.forest, game.graveyard, game.grass, game.rain, game.atmosphere):
        part.configure(q)
    game.hud.configure_quality(q)
    game.engine.state.quality = {"preset": name, **asdict(q), "error": None, "pinned": pinned}
