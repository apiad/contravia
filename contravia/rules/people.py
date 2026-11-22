"""Who walks the pavements: how each zone dresses, how fast people go, how many are out, and the eight NPCs.

manifoldx gives the body (`HumanParams`) and the clothes (`Outfit`); which bodies and clothes a Cuban street has is
the game's choice and lives here. A body of the crowd is `HumanParams.sample(seed)` dressed by `outfit(seed, zone)`,
where zone is contravia.walks' (0 the old centre, 1 Vedado and Playa, 2 the rest). The eight NPCs are the people of
kind `npc` in assets/reference/catalog.json, each with a fixed body and outfit and the place of contravia/places.json
where it stands. Spec: docs/superpowers/specs/2026-11-22-pedestrians-in-game-design.html.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from manifoldx.modeling.assets.human import HumanParams
from manifoldx.modeling.assets.human_dress import Outfit

WALK = (1.1, 1.3, 1.5)  # m/s: the walking speeds a clip is baked at; a walker moves at its bucket's exactly
RUN = (3.0, 3.5)  # m/s: the running speeds
CROSS = 1.2  # m/s: the walk that looks both ways before a street
CLIPS = tuple(("walk", v) for v in WALK) + tuple(("run", v) for v in RUN) + (("idle", 0.0), ("cross", CROSS))

DENSITY = (1.0, 0.6, 0.25)  # walkers per lane metre, relative, in zones 0, 1 and 2
SIDE = 0.5  # aprons and park paths carry this share of their zone's density
HEAVY = 0.8  # the crowd's bodies keep weight under this: heavier figures still show manifoldx's open defects

# Clothes by zone, sRGB. The old centre is bright, Vedado paler, the rest plainer and darker.
TOPS = (
    ("#f2ede0", "#e8613c", "#2fa7a0", "#f0c14b", "#7fc8a9", "#d94f70", "#5b8bd9", "#ffffff"),
    ("#f4f1e8", "#a9cbe0", "#e9d7b9", "#cfe3c6", "#f2c7b4", "#9aa6b8", "#ffffff", "#e8613c"),
    ("#d9d3c3", "#6f7d8c", "#8c5a3c", "#3e6b5a", "#b7a27a", "#5a5f73", "#a33b3b", "#f2ede0"),
)
BOTTOMS = (
    ("#2f3e56", "#3b4a5c", "#e9e2cf", "#1f2a3a", "#7a5c3e", "#f0c14b"),
    ("#2f3e56", "#d9cfb8", "#3b4a5c", "#e9e2cf", "#5a6a7a", "#1f2a3a"),
    ("#2f3e56", "#3b4a5c", "#4a3f35", "#1f2a3a", "#5c5a50", "#2b2522"),
)
HAIR = ("#1a120c", "#2a1d14", "#3b2a1c", "#5a4030", "#8a8178", "#c9c2b5")
SHOES = ("#2b2522", "#e8e2d0", "#5a4030", "#3b4a5c")
HATS = ("#d9c48a", "#e8613c", "#2f3e56", "#f2ede0")


def body(seed: int) -> HumanParams:
    """The crowd's body for seed: HumanParams.sample with the weight held under HEAVY."""
    p = HumanParams.sample(seed)
    return replace(p, weight=min(p.weight, HEAVY))


def outfit(seed: int, zone: int, build: float) -> Outfit:
    """What a body of the given build (0 female to 1 male) wears in zone, the same for the same seed."""
    g = np.random.default_rng((seed, zone, 7))

    def pick(xs, p=None):
        return xs[int(g.choice(len(xs), p=p))]

    woman = build < 0.5
    bottom = pick(("trousers", "shorts", "skirt"), (0.5, 0.2, 0.3) if woman else (0.7, 0.3, 0.0))
    hot = zone != 2 or g.random() < 0.5  # the centre and the coast dress lighter
    return Outfit(
        top=pick(("tshirt", "shirt", "blouse", "tank"), (0.35, 0.15, 0.4, 0.1) if woman else (0.5, 0.35, 0.0, 0.15)),
        sleeves=round(float(g.uniform(0.1, 0.4) if hot else g.uniform(0.25, 1.0)), 2),
        bottom=bottom,
        bottom_length=round(float(g.uniform(0.45, 0.7) if bottom != "trousers" else g.uniform(0.9, 1.0)), 2),
        looseness=round(float(g.uniform(0.008, 0.03)), 3),
        shoes=pick(("shoes", "sandals", "boots"), (0.6, 0.35, 0.05)),
        hair=pick(("short", "long", "bun", "curly", None), (0.1, 0.4, 0.25, 0.25, 0.0) if woman
                  else (0.65, 0.0, 0.0, 0.2, 0.15)),
        hat=pick((None, "cap", "brim"), (0.8, 0.12, 0.08)),
        top_color=pick(TOPS[zone]),
        bottom_color=pick(BOTTOMS[zone]),
        shoe_color=pick(SHOES),
        hair_color=pick(HAIR[4:] if g.random() < 0.15 else HAIR[:4]),
        hat_color=pick(HATS),
    )


@dataclass(frozen=True)
class Npc:
    """One of the eight reference people: its body, clothes and seed, the place of contravia/places.json where it
    stands, and what it does there."""

    name: str
    params: HumanParams
    outfit: Outfit
    seed: int
    place: str  # an id of contravia/places.json
    does: str = "idle"  # "idle" stands, "loop" walks a short loop round its place


# From each catalog description. What Outfit cannot draw is left out for now: sunglasses, moustaches, the jabas,
# the peanut tin, the camera, the backpacks, the newspaper, the phone, and the bicitaxi itself.
NPCS = (
    Npc("almendronero",
        HumanParams(stature=1.72, build=0.95, weight=1.15, age="adult", skin="tan"),
        Outfit(top="shirt", sleeves=0.3, bottom="trousers", shoes="shoes", hair="short", hat="brim",
               top_color="#f4f1e8", bottom_color="#2b2f38", hair_color="#8a8178", hat_color="#d9c48a"),
        seed=101, place="L0464"),  # Parque de la Fraternidad
    Npc("bicitaxista",
        HumanParams(stature=1.74, build=1.0, weight=0.85, muscle=1.15, age="adult", skin="brown"),
        Outfit(top="tank", sleeves=0.0, bottom="shorts", bottom_length=0.45, shoes="sandals", hair="short",
               hat="cap", top_color="#2fb5a8", bottom_color="#8fdcc8", hat_color="#d23a2f",
               stripe=(0.35, 0.6, "#f0c14b")),
        seed=102, place="M061"),  # Calle Obispo
    Npc("manisero",
        HumanParams(stature=1.66, build=1.0, weight=0.9, age="elder", skin="dark"),
        Outfit(top="shirt", sleeves=0.35, bottom="trousers", shoes="shoes", hair="short", hat="brim",
               top_color="#ffffff", bottom_color="#8fdcc8", hair_color="#c9c2b5", hat_color="#e0c98f"),
        seed=103, place="M004", does="loop"),  # Malecón
    Npc("senora-jaba",
        HumanParams(stature=1.58, build=0.0, weight=1.3, hips=1.12, age="elder", skin="medium"),
        Outfit(top="blouse", sleeves=0.2, bottom="skirt", bottom_length=0.7, looseness=0.03, shoes="sandals",
               hair="bun", top_color="#7fd3c8", bottom_color="#7fd3c8", shoe_color="#b8bcc0",
               hair_color="#8a8178"),
        seed=104, place="M069"),  # Bulevar de San Rafael, Centro Habana
    Npc("estudiante",  # a secondary-school pupil, built as the smallest adult body: there are no children
        HumanParams(stature=1.55, build=0.0, weight=0.85, shoulders=0.9, age="adult", skin="tan"),
        Outfit(top="shirt", sleeves=0.25, bottom="skirt", bottom_length=0.5, shoes="shoes", hair="long",
               top_color="#ffffff", bottom_color="#c9a227", shoe_color="#2b2522", hair_color="#1a120c"),
        seed=105, place="M039"),  # Universidad de La Habana
    Npc("jubilado",
        HumanParams(stature=1.68, build=1.0, weight=1.05, age="elder", skin="light"),
        Outfit(top="shirt", sleeves=0.3, bottom="trousers", shoes="shoes", hair="short", hat="cap",
               top_color="#8fdcd8", bottom_color="#d9cfb8", hair_color="#c9c2b5", hat_color="#c9b88f"),
        seed=106, place="L0382"),  # Línea y G, Vedado
    Npc("turista",
        HumanParams(stature=1.80, build=0.9, weight=1.1, age="adult", skin="light"),
        Outfit(top="tshirt", sleeves=0.25, bottom="shorts", bottom_length=0.5, shoes="shoes", hair="short",
               hat="brim", top_color="#f0c14b", bottom_color="#8fdcc8", shoe_color="#f2ede0",
               hair_color="#5a4030", hat_color="#f2ede0"),
        seed=107, place="M007", does="loop"),  # Plaza de Armas
    Npc("joven",
        HumanParams(stature=1.76, build=1.0, weight=0.95, muscle=1.1, age="adult", skin="brown"),
        Outfit(top="tshirt", sleeves=0.25, bottom="trousers", shoes="shoes", hair="short", hat="cap",
               top_color="#a8ecd8", bottom_color="#6f8fbf", shoe_color="#f2ede0", hat_color="#e8613c"),
        seed=108, place="L0381"),  # Monumento a las víctimas del Maine, Malecón at Línea
)
