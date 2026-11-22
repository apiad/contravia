"""How good the game looks against how fast it runs: every knob that trades one for the other, in one place.

A Quality holds one value per knob. Its defaults are Alto, which is the game as it was before the presets: the
streamer's detail, the far plane, the trees, grass, rain, shadows and minimap as their modules' constants set
them. PRESETS gives five: Bajo for a laptop, Medio, Alto, Ultra for a high-end card, and Cine, which only the
capture tool uses and which ignores the frame time. A player who turns one knob alone plays "Personalizado":
a preset with some knobs changed, each to a value some preset uses (its ladder).

The screen-space effects (manifoldx.post: ambient occlusion, bloom, light shafts, a colour grade and FXAA) are
knobs too; Alto turns on all but the shafts. Bajo and Medio keep one shadow map around the car; Alto and up split
the view into cascaded maps (manifoldx's enable_shadows(cascades=...)), sharp near the car and reaching 200 to
300 m. The effects manifoldx does not have yet (spec docs/superpowers/specs/2026-11-21-graphics-quality-design.html)
become fields here as they land.
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
    people_lod: tuple[float, float, float] = (12.0, 45.0, 120.0)  # metres: a person drops to level 1, 2, not drawn
    people: tuple[int, int, int, int] = (300, 16, 80, 64)  # walkers, at level 0 at most, at level 1, bodies built
    traffic_radius: float = 400.0  # metres round the player within which the car layer drives cars
    # How far shadows reach (m; with one map, the half side of its square), map size, PCF radius, and cascades: with
    # more than one, the view near the camera gets its own sharp maps and the farthest still reaches that far
    shadow: tuple[float, int, int, int] = (200.0, 2048, 1, 3)
    rain_pool: int = 3000  # rain streaks at the heaviest rain
    minimap: tuple[int, int] = (256, 4)  # the minimap's texture side in pixels, fewest frames between two renders
    target_ms: float = 20.0  # the frame time the streamer's tau is tuned toward
    tune: bool = True  # whether tau follows the frame time within its range
    fxaa: bool = True  # smooth the edges of the finished frame
    ssao: float = 3.0  # the reach of the darkening in corners and at the foot of walls, metres; 0 turns it off
    bloom: float = 0.5  # how much what is brightest glows past its edge; 0 turns it off
    shafts: float = 0.0  # how strong the rays from the sun are when it is in the frame; 0 turns them off
    grade: tuple[float, float, float] = (1.15, 1.1, 0.3)  # saturation, contrast, vignette of the finished frame

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
                    people_lod=(8.0, 25.0, 60.0), people=(80, 6, 24, 24),
                    traffic_radius=200.0, shadow=(80.0, 1024, 0, 1), rain_pool=1000, minimap=(192, 8),
                    target_ms=33.3, fxaa=False, ssao=0.0, bloom=0.0, grade=(1.0, 1.0, 0.0)),
    "medio": Quality(tau=(16.0, 64.0), far=11000.0, fog_clear=(2000.0, 9000.0), tree_lod=(45.0, 120.0),
                     foliage_near=15.0, grass=(12.0, 30.0, 40, 12), grave_lod=(40.0, 120.0),
                     people_lod=(10.0, 35.0, 90.0), people=(150, 10, 40, 40), traffic_radius=300.0,
                     shadow=(150.0, 2048, 1, 1), rain_pool=2000, target_ms=25.0, ssao=0.0, bloom=0.3,
                     grade=(1.08, 1.05, 0.2)),
    "alto": Quality(),
    "ultra": Quality(tau=(8.0, 32.0), detail_radius=2, tree_lod=(90.0, 240.0), foliage_near=40.0,
                     grass=(30.0, 70.0, 90, 24), grave_lod=(70.0, 200.0),
                     people_lod=(16.0, 60.0, 160.0), people=(500, 24, 120, 80), shadow=(250.0, 2048, 1, 4),
                     rain_pool=6000, minimap=(384, 2),
                     target_ms=16.7, ssao=4.0, bloom=0.6, shafts=0.6),
    "cine": Quality(tau=(5.0, 5.0), detail_radius=2, tree_lod=(120.0, 320.0), foliage_near=60.0,
                    grass=(40.0, 90.0, 120, 32), grave_lod=(90.0, 260.0),
                    people_lod=(20.0, 80.0, 200.0), people=(600, 40, 200, 96), shadow=(300.0, 4096, 1, 4),
                    rain_pool=10000, tune=False, ssao=4.0,
                    bloom=0.6, shafts=0.8),
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
    "people_lod": "Peatones (detalle)",
    "people": "Peatones",
    "traffic_radius": "Tráfico cercano",
    "shadow": "Sombras",
    "rain_pool": "Gotas de lluvia",
    "minimap": "Minimapa",
    "fxaa": "Suavizado de bordes",
    "ssao": "Sombras en las esquinas",
    "bloom": "Brillo",
    "shafts": "Rayos de sol",
    "grade": "Color",
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
    game.detail.ready_radius = max(2, q.detail_radius + 1)  # one ring past the drawn one is built ahead
    if game.layer is not None:
        game.layer.radius = q.traffic_radius
    if getattr(game.streamer.material, "detail", None) != q.detail_noise:
        game.streamer.material = StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True, faded=True,
                                                  detail=q.detail_noise)
    for part in (game.forest, game.graveyard, game.crowd, game.grass, game.rain, game.atmosphere):
        part.configure(q)
    set_post(game.engine, q)
    game.hud.configure_quality(q)
    game.engine.state.quality = {"preset": name, **asdict(q), "error": None, "pinned": pinned}


def set_post(engine, q: Quality) -> None:
    """The screen-space effects of a quality, on a manifoldx with them (manifoldx.post); a no-op on one without."""
    if not hasattr(engine, "set_post"):
        return
    from manifoldx.post import SSAO, Bloom, Grade, LightShafts

    engine.set_post(fxaa=q.fxaa, ssao=SSAO(radius=q.ssao) if q.ssao else None,
                    bloom=Bloom(intensity=q.bloom) if q.bloom else None,
                    shafts=LightShafts(intensity=q.shafts) if q.shafts else None,
                    grade=Grade(*q.grade) if q.grade != (1.0, 1.0, 0.0) else None)
