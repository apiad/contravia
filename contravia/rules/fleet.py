"""The fleet: the kinds of vehicle the streets draw, their presets, families, eras, ranges and colours.

A kind is what the traffic counts (a car, a taxi, a bus, a truck, a moto, a tricycle, a bicitaxi, a bike, a police
car or a fire engine). A family is one shape inside a kind: the 1950s almendrón or the 1980s three-box saloon.
Every family names its manifoldx builder, the reference preset it starts from, the era whose colours it wears and
how many variants it draws. The presets are the single source of every reference parameter set: a family draws
from its preset and the Garaje's :func:`catalogue` builds the same object, so the almendrón exists once.

:func:`ranges` is the one table of ranges. It answers for a family and, when given, checks its era; the traffic
draws from it and the Garaje spans the union of a builder's families with it, so a range changed once changes
both. :func:`variant` returns one variant of one family: a parameter set drawn from the family's ranges, plus its
colourways as palettes. It is seeded from the kind, the family and the index with SHA-256, never with Python's
``hash``, so the same arguments give the same vehicle on every machine and in every run. A draw that manifoldx's
``validate`` rejects is redrawn from a derived seed a bounded number of times; a family that keeps failing has a
bug in its ranges, and the error says so.

The names are the ones people in Havana use for these cars; no family is named after a maker or a model. The car
ranges follow the eras table of the vehicles spec, and the ranges of buses, trucks and cycles sit around their
reference presets. Every range lies inside the range manifoldx publishes for the same parameter
(:meth:`manifoldx.modeling.assets.vehicle.ParamsRanges.ranges`), so every draw and every slider position can be
built.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from manifoldx.modeling.assets.bus import BusParams, build_bus
from manifoldx.modeling.assets.car import CarParams, build_car
from manifoldx.modeling.assets.cycle import (
    CycleParams,
    build_cycle,
)
from manifoldx.modeling.assets.cycle import (
    bicycle as _bicycle,
)
from manifoldx.modeling.assets.cycle import (
    cargo_tricycle as _cargo_tricycle,
)
from manifoldx.modeling.assets.cycle import (
    cycle_rickshaw as _cycle_rickshaw,
)
from manifoldx.modeling.assets.cycle import (
    electric_moto as _electric_moto,
)
from manifoldx.modeling.assets.cycle import (
    motorcycle as _motorcycle,
)
from manifoldx.modeling.assets.truck import (
    TruckParams,
    build_truck,
)
from manifoldx.modeling.assets.truck import (
    box_truck as _box_truck,
)
from manifoldx.modeling.assets.truck import (
    cargo_truck_1980s as _cargo_truck_1980s,
)
from manifoldx.modeling.assets.truck import (
    dump_truck as _dump_truck,
)
from manifoldx.modeling.assets.truck import (
    fire_engine as _fire_engine,
)
from manifoldx.modeling.assets.truck import (
    tanker as _tanker,
)

# The ten kinds of the fleet spec, in the order the sheets and the traffic read them.
KINDS = ("car", "taxi", "bus", "truck", "moto", "tricycle", "bicitaxi", "bike", "police", "fire")

# The eras whose ranges and colours a family draws from. "services" is a livery over another era's car.
ERAS = ("1950s", "80s", "modern", "services")

# The manifoldx builder and parameter class of each builder name.
BUILDERS = {"car": build_car, "bus": build_bus, "truck": build_truck, "cycle": build_cycle}
PARAMS = {"car": CarParams, "bus": BusParams, "truck": TruckParams, "cycle": CycleParams}

# How many times a rejected draw is retried before the family is declared broken.
ATTEMPTS = 12


# --- reference presets ---------------------------------------------------------------------------------------


def almendron() -> CarParams:
    """A 1950s American sedan: long and low, chrome bumpers, round lamps on the wings and fins at the tail."""
    return CarParams(
        length=5.10, width=1.92, height=1.52, wheelbase=3.00, front_overhang=0.92, track=1.50,
        wheel_radius=0.34, tyre_width=0.20, rim_ratio=0.70, arch_gap=0.05,
        nose_top=0.88, hood_front=1.02, nose_blunt=0.85, cowl_u=0.28, cowl_height=1.12,
        windscreen_u=0.35, roof_peak_u=0.53, roof_end_u=0.73, roof_end_drop=0.04,
        hatch_u=0.83, hatch_height=1.00, tail_top=0.94, belt_front=0.96, belt_rear=0.96,
        sill=0.21, bumper_bottom_front=0.30, bumper_bottom_rear=0.30,
        bumper="chrome_bar", bumper_height=0.42, bumper_proud=0.10, overriders=True,
        hood_crown=0.13, nose_half=0.92, tail_half=0.92, round_front=0.14, round_rear=0.14,
        round_power=2.2, nose_bulge=0.03, tail_bulge=0.03, tumblehome=0.18, shoulder=0.96,
        fender_crown=0.0, fender_crown_u=0.30, fin_height=0.0, fin_start_u=0.62, fin_width=0.26,
        windscreen_wrap=0.35, side_trim=((0.72, 0.16, 0.95, "chrome"),),
        a_pillar_foot=0.10, b_pillar_u=0.56, b_pillar_width=0.05, c_pillar_u=0.66,
        black_pillars=False, roof="paint", roof_paint=False, roof_pillars=True,
        headlamp="round_pod", headlamp_size=(0.20, 0.20), headlamp_center=None, headlamp_yaw=20.0,
        grille=None, grille_style="mesh", grille_span="full_width",
        taillamp="pill", taillamp_center=None, taillamp_size=(0.20, 0.16), taillamp_yaw=25.0,
        wheel="hubcap", whitewall=0.05, mirrors=True, handles=True, crease=32.0, sections=28, detail=1.0,
    )


def ruso() -> CarParams:
    """A 1980s Soviet three-box saloon: upright and boxy, with square lamps and dark bumpers."""
    return CarParams(
        length=3.95, width=1.62, height=1.42, wheelbase=2.45, front_overhang=0.75, track=1.32,
        wheel_radius=0.29, tyre_width=0.18, nose_top=0.80, hood_front=0.94, nose_blunt=0.92,
        cowl_u=0.27, cowl_height=1.02, windscreen_u=0.34, roof_peak_u=0.55, roof_end_u=0.74,
        roof_end_drop=0.03, hatch_u=0.84, hatch_height=0.96, tail_top=0.90,
        belt_front=0.88, belt_rear=0.88, sill=0.19, bumper_bottom_front=0.28, bumper_bottom_rear=0.28,
        bumper="rubber", bumper_height=0.40, bumper_proud=0.06, hood_crown=0.09,
        nose_half=0.92, tail_half=0.94, round_front=0.10, round_rear=0.10, tumblehome=0.12,
        shoulder=0.98, a_pillar_foot=0.15, b_pillar_u=0.56, b_pillar_width=0.06, c_pillar_u=0.68,
        black_pillars=True, roof="paint", roof_paint=False,
        headlamp="rect", headlamp_size=(0.26, 0.13), headlamp_center=None, headlamp_yaw=16.0,
        grille=None, grille_style="bar", grille_span="between_lamps",
        taillamp="block", taillamp_center=None, taillamp_size=(0.26, 0.15), taillamp_yaw=22.0,
        wheel="hubcap", door_strip=True, mirrors=True, crease=34.0,
    )


def moscovita() -> CarParams:
    """A 1970s Soviet saloon: round lamps with chrome rings, a chrome grille and a softer shape than the ruso."""
    return CarParams(
        length=4.20, width=1.60, height=1.45, wheelbase=2.50, front_overhang=0.80, track=1.30,
        wheel_radius=0.30, tyre_width=0.18, nose_top=0.82, hood_front=0.96, nose_blunt=0.90,
        cowl_u=0.27, cowl_height=1.04, windscreen_u=0.34, roof_peak_u=0.56, roof_end_u=0.75,
        roof_end_drop=0.03, hatch_u=0.84, hatch_height=0.98, tail_top=0.92,
        belt_front=0.90, belt_rear=0.90, sill=0.19, bumper_bottom_front=0.29, bumper_bottom_rear=0.29,
        bumper="chrome_bar", bumper_height=0.40, bumper_proud=0.07, hood_crown=0.10,
        nose_half=0.90, tail_half=0.92, round_front=0.10, round_rear=0.10, tumblehome=0.14,
        shoulder=0.98, a_pillar_foot=0.15, b_pillar_u=0.56, c_pillar_u=0.68,
        black_pillars=False, roof="paint", roof_paint=False,
        headlamp="round", headlamp_size=(0.17, 0.17), headlamp_center=None,
        grille=None, grille_style="mesh", grille_span="between_lamps",
        taillamp="pill", taillamp_center=None, taillamp_size=(0.20, 0.14),
        wheel="hubcap", mirrors=True, crease=32.0,
    )


def moderno() -> CarParams:
    """A modern three-box saloon or small crossover: almond lamps, body-colour bumpers and alloy wheels."""
    return CarParams(
        length=4.55, width=1.78, height=1.52, wheelbase=2.65, front_overhang=0.90, track=1.48,
        wheel_radius=0.32, tyre_width=0.20, nose_top=0.86, hood_front=1.00, nose_blunt=0.75,
        cowl_u=0.30, cowl_height=1.06, windscreen_u=0.38, roof_peak_u=0.57, roof_end_u=0.75,
        roof_end_drop=0.05, hatch_u=0.85, hatch_height=1.00, tail_top=0.94,
        belt_front=0.92, belt_rear=0.92, sill=0.19, bumper_bottom_front=0.30, bumper_bottom_rear=0.30,
        bumper="skin", bumper_height=0.42, bumper_proud=0.0, hood_crown=0.10,
        nose_half=0.88, tail_half=0.90, round_front=0.16, round_rear=0.16, tumblehome=0.22,
        shoulder=0.96, a_pillar_foot=0.12, b_pillar_u=0.56, c_pillar_u=0.68,
        black_pillars=True, roof="paint", roof_paint=False,
        headlamp="almond", headlamp_size=(0.30, 0.14), headlamp_center=None, headlamp_yaw=22.0,
        grille=None, grille_style="bar", grille_span="between_lamps",
        taillamp="block", taillamp_center=None, taillamp_size=(0.24, 0.15),
        wheel="spokes", spokes=6, mirrors=True, crease=36.0,
    )


def huevito() -> CarParams:
    """A modern small hatchback: short, tall and rounded, with a rear glass that falls almost to the tail."""
    return CarParams(
        length=3.80, width=1.66, height=1.55, wheelbase=2.42, front_overhang=0.70, track=1.40,
        wheel_radius=0.31, tyre_width=0.18, nose_top=0.88, hood_front=1.00, nose_blunt=0.70,
        cowl_u=0.30, cowl_height=1.08, windscreen_u=0.38, roof_peak_u=0.60, roof_end_u=0.78,
        roof_end_drop=0.05, hatch_u=0.93, hatch_height=1.08, tail_top=1.06,
        belt_front=0.94, belt_rear=0.94, sill=0.19, bumper_bottom_front=0.30, bumper_bottom_rear=0.30,
        bumper="skin", bumper_height=0.42, hood_crown=0.10,
        nose_half=0.92, tail_half=0.92, round_front=0.24, round_rear=0.20, tumblehome=0.24,
        shoulder=0.94, a_pillar_foot=0.12, b_pillar_u=0.58, c_pillar_u=0.74,
        black_pillars=True, roof="paint", roof_paint=False,
        headlamp="almond", headlamp_size=(0.28, 0.14), headlamp_center=None,
        grille=None, grille_style="bar", grille_span="between_lamps",
        taillamp="block", taillamp_center=None, taillamp_size=(0.20, 0.16),
        wheel="hubcap", mirrors=True, crease=36.0,
    )


# The paint area of a bus: a band under the windows, recoloured by each colourway (see `_bus_colourways`).
BUS_LIVERY = (("#9aa0a4", [(0.0, 0.26), (1.0, 0.26), (1.0, 0.88), (0.0, 0.88)]),)


def metroguagua() -> BusParams:
    """A 12 m low-floor city bus: three double doors on its right, a band of windows and a route sign."""
    return BusParams(wheel="hubcap", livery=BUS_LIVERY)


def guagua_vieja() -> BusParams:
    """An older, smaller city bus: a higher floor, two double doors and a rounder, taller face."""
    return BusParams(
        length=9.4, width=2.45, height=3.00, floor=0.72, front_overhang=2.0, wheelbase=4.9,
        wheel_radius=0.50, tyre_width=0.30, track=2.05,
        windscreen_bottom=1.08, windscreen_top=2.52, sign_bottom=2.60, sign_top=2.82,
        window_bottom=1.42, window_top=2.42, rear_window_bottom=1.38, rear_window_top=2.38,
        headlamp="round", taillamp="round", doors=((0.05, 1.0, 2), (0.52, 1.0, 2)),
        pillar_spacing=1.15, wheel="hubcap", livery=BUS_LIVERY,
    )


def interprovincial() -> BusParams:
    """An intercity coach: long and tall, a single front door and a higher band of windows."""
    return BusParams(
        length=12.2, width=2.55, height=3.45, floor=0.62, front_overhang=2.4, wheelbase=6.2,
        wheel_radius=0.52, tyre_width=0.30, track=2.10,
        windscreen_bottom=1.18, windscreen_top=2.85, sign_bottom=3.00, sign_top=3.20,
        window_bottom=1.50, window_top=2.85, rear_window_bottom=1.45, rear_window_top=2.80,
        headlamp="rect", taillamp="bar", doors=((0.05, 0.95, 1),),
        pillar_spacing=1.50, wheel="hubcap", livery=BUS_LIVERY,
    )


def camion_pasajeros() -> TruckParams:
    """A passenger truck: a long body over benches, with a canvas roof.

    manifoldx has no bench body with a canvas roof yet; today's closest body is the closed box, so this preset
    paints a box truck in the canvas and wood of a passenger truck. Replacing `body` when the engine gains one
    is all this family needs.
    """
    return replace(
        _box_truck(), length=7.6, height=3.35, wheelbase=4.2, front_overhang=1.20,
        cab_length=1.75, body_height=3.15, body_floor=1.05,
        cab_paint="#d9d5c4", body_paint="#cfc9b4",
    )


def triciclo_pasajeros() -> CycleParams:
    """An electric passenger tricycle: a delta of three wheels with a bench under a canopy.

    manifoldx has no closed cab for a passenger tricycle yet, so this preset is the bicitaxi's bench and canopy
    with a hub motor and a footboard. The canopy is what reads as a cab from the street.
    """
    return replace(
        _cycle_rickshaw(), power="hub", step_through=True, footboard=True,
        headlamp="rect", tail_lamp=True, frame_paint="#e8e4d4", box_paint="#7fa8c4",
    )


# The trucks and cycles the catalogue names after their trade: each family and its preset are one and the same
# reference parameter set, so the traffic's variant and the Garaje's preset start from the same numbers.


def camion_80() -> TruckParams:
    """An 1980s cargo truck: a flatbed with wooden sides and a round-cabbed lorry in front."""
    return _cargo_truck_1980s()


def furgon() -> TruckParams:
    """A box truck: a closed cargo box behind the cab, for the centre's deliveries."""
    return _box_truck()


def cisterna() -> TruckParams:
    """A tanker: a silver cylinder over the chassis, for fuel and water."""
    return _tanker()


def volquete() -> TruckParams:
    """A dump truck: a raised steel bin that tips its load at a building site."""
    return _dump_truck()


def bomberos() -> TruckParams:
    """A fire engine: a red locker body, a ladder on the roof and beacons on the cab."""
    return _fire_engine()


def patrulla() -> CarParams:
    """A 1980s saloon in the police's white and blue, with the blue band low on its flanks and a roof bar."""
    return replace(ruso(), livery=POLICE_AREAS, lightbar="bar",
                   lightbar_colours=(POLICE_BLUES[0], POLICE_FLASH))


def moto_electrica() -> CycleParams:
    """An electric moto: a hub motor, a step-through frame and a flat footboard."""
    return _electric_moto()


def moto_dos_tiempos() -> CycleParams:
    """A 1980s two-stroke motorcycle: a small tank, a saddle and a chrome exhaust."""
    return _motorcycle()


def triciclo_carga() -> CycleParams:
    """An electric cargo tricycle: two rear wheels under a yellow box and a low saddle."""
    return _cargo_tricycle()


def bicitaxi() -> CycleParams:
    """A pedal rickshaw: a bench and a canopy over two rear wheels, pedalled from the front."""
    return _cycle_rickshaw()


def bicicleta() -> CycleParams:
    """A city bicycle: a plain diamond frame, a saddle and a rear rack."""
    return _bicycle()


# --- the ranges each era draws from --------------------------------------------------------------------------

# The measurements a family edits but its generator fixes through a style option or its preset instead of drawing
# a number. They sit in the same era table as every other measurement, so the Garaje's slider and the generator
# read one number, and the draw skips them so a family's variants stay the same.
STYLE_ONLY = frozenset({"fin_height", "fender_crown", "whitewall", "corner_radius"})

# Cars: the eras table of the vehicles spec, then each family narrows the fields it has to. All the heights
# are absolute metres and stay in order for every point of the box, so a draw never needs a repair: the sill is
# below the belt, the belt below the roof, the bumper below the belt, and the stations along the body keep
# their order. Fins, fender crowns and whitewalls are the era's styling: the 1950s wear them, the decades after
# keep them small or leave them out, and a style option picks the exact value.
CAR_RANGES = {
    "1950s": {
        "length": (4.80, 5.40), "width": (1.85, 2.00), "height": (1.45, 1.60),
        "wheelbase": (2.90, 3.10), "front_overhang": (0.85, 1.00), "track": (1.45, 1.55),
        "wheel_radius": (0.32, 0.36), "tyre_width": (0.19, 0.22),
        "nose_top": (0.84, 0.94), "hood_front": (0.96, 1.06), "cowl_u": (0.26, 0.30),
        "cowl_height": (1.08, 1.16), "windscreen_u": (0.33, 0.37), "roof_peak_u": (0.50, 0.56),
        "roof_end_u": (0.70, 0.76), "roof_end_drop": (0.02, 0.08), "hatch_u": (0.80, 0.86),
        "hatch_height": (0.96, 1.06), "tail_top": (0.90, 1.00),
        "belt_front": (0.92, 1.00), "belt_rear": (0.92, 1.00),
        "sill": (0.18, 0.24), "bumper_bottom_front": (0.26, 0.34), "bumper_bottom_rear": (0.26, 0.34),
        "bumper_height": (0.38, 0.46), "bumper_proud": (0.06, 0.12),
        "hood_crown": (0.10, 0.16), "windscreen_wrap": (0.00, 0.60),
        "b_pillar_u": (0.52, 0.60), "c_pillar_u": (0.62, 0.68),
        "headlamp_yaw": (15.0, 30.0), "taillamp_yaw": (20.0, 35.0),
        "fin_height": (0.0, 0.30), "fender_crown": (0.0, 0.20), "whitewall": (0.0, 0.08),
    },
    "80s": {
        "length": (3.70, 4.40), "width": (1.55, 1.70), "height": (1.38, 1.48),
        "wheelbase": (2.40, 2.50), "front_overhang": (0.72, 0.92), "track": (1.28, 1.36),
        "wheel_radius": (0.27, 0.30), "tyre_width": (0.18, 0.20),
        "nose_top": (0.78, 0.88), "hood_front": (0.90, 1.00), "cowl_u": (0.26, 0.30),
        "cowl_height": (1.00, 1.08), "windscreen_u": (0.33, 0.37), "roof_peak_u": (0.52, 0.58),
        "roof_end_u": (0.72, 0.76), "roof_end_drop": (0.01, 0.04), "hatch_u": (0.82, 0.88),
        "hatch_height": (0.92, 1.00), "tail_top": (0.86, 0.94),
        "belt_front": (0.84, 0.92), "belt_rear": (0.84, 0.92),
        "sill": (0.16, 0.22), "bumper_bottom_front": (0.24, 0.32), "bumper_bottom_rear": (0.24, 0.32),
        "bumper_height": (0.36, 0.44), "bumper_proud": (0.04, 0.08),
        "hood_crown": (0.08, 0.12), "b_pillar_u": (0.52, 0.58), "c_pillar_u": (0.62, 0.68),
        "headlamp_yaw": (12.0, 24.0), "taillamp_yaw": (16.0, 28.0),
        "fin_height": (0.0, 0.04), "fender_crown": (0.0, 0.04), "whitewall": (0.0, 0.0),
    },
    "modern": {
        "length": (4.30, 4.70), "width": (1.70, 1.85), "height": (1.45, 1.58),
        "wheelbase": (2.55, 2.75), "front_overhang": (0.85, 0.95), "track": (1.42, 1.52),
        "wheel_radius": (0.30, 0.34), "tyre_width": (0.19, 0.22),
        "nose_top": (0.80, 0.92), "hood_front": (0.94, 1.04), "cowl_u": (0.28, 0.32),
        "cowl_height": (1.02, 1.10), "windscreen_u": (0.36, 0.40), "roof_peak_u": (0.54, 0.60),
        "roof_end_u": (0.72, 0.78), "roof_end_drop": (0.02, 0.08), "hatch_u": (0.82, 0.96),
        "hatch_height": (0.96, 1.10), "tail_top": (0.90, 1.08),
        "belt_front": (0.88, 0.96), "belt_rear": (0.88, 0.96),
        "sill": (0.17, 0.22), "bumper_bottom_front": (0.25, 0.33), "bumper_bottom_rear": (0.25, 0.33),
        "bumper_height": (0.38, 0.46), "bumper_proud": (0.0, 0.04),
        "hood_crown": (0.08, 0.12), "b_pillar_u": (0.52, 0.58), "c_pillar_u": (0.62, 0.70),
        "headlamp_yaw": (18.0, 34.0), "taillamp_yaw": (16.0, 30.0),
        "fin_height": (0.0, 0.0), "fender_crown": (0.0, 0.06), "whitewall": (0.0, 0.0),
    },
}

# Buses: three families around the bus presets.
BUS_RANGES = {
    "metroguagua": {
        "length": (10.5, 12.5), "width": (2.45, 2.60), "height": (3.00, 3.20), "floor": (0.30, 0.40),
        "front_overhang": (2.20, 2.80), "wheelbase": (5.50, 6.40), "wheel_radius": (0.48, 0.52),
        "tyre_width": (0.28, 0.32), "track": (2.00, 2.15),
        "windscreen_bottom": (0.80, 0.95), "windscreen_top": (2.65, 2.85),
        "sign_bottom": (2.78, 2.86), "sign_top": (2.98, 3.08),
        "window_bottom": (1.15, 1.30), "window_top": (2.50, 2.70),
        "rear_window_bottom": (1.15, 1.30), "rear_window_top": (2.45, 2.65),
        "door_bottom": (0.28, 0.34), "pillar_spacing": (1.20, 1.50), "corner_radius": (0.05, 0.60),
    },
    "guagua_vieja": {
        "length": (8.80, 10.00), "width": (2.40, 2.50), "height": (2.90, 3.10), "floor": (0.65, 0.80),
        "front_overhang": (1.80, 2.20), "wheelbase": (4.60, 5.20), "wheel_radius": (0.48, 0.52),
        "tyre_width": (0.28, 0.32), "track": (1.98, 2.10),
        "windscreen_bottom": (1.00, 1.14), "windscreen_top": (2.40, 2.56),
        "sign_bottom": (2.60, 2.70), "sign_top": (2.82, 2.94),
        "window_bottom": (1.34, 1.48), "window_top": (2.30, 2.46),
        "rear_window_bottom": (1.30, 1.44), "rear_window_top": (2.26, 2.42),
        "door_bottom": (0.60, 0.72), "pillar_spacing": (1.05, 1.30), "corner_radius": (0.05, 0.45),
    },
    "interprovincial": {
        "length": (11.5, 13.0), "width": (2.50, 2.60), "height": (3.30, 3.60), "floor": (0.55, 0.70),
        "front_overhang": (2.20, 2.60), "wheelbase": (5.80, 6.60), "wheel_radius": (0.50, 0.54),
        "tyre_width": (0.28, 0.32), "track": (2.05, 2.15),
        "windscreen_bottom": (1.10, 1.24), "windscreen_top": (2.72, 2.90),
        "sign_bottom": (2.84, 2.94), "sign_top": (3.10, 3.24),
        "window_bottom": (1.44, 1.56), "window_top": (2.70, 2.88),
        "rear_window_bottom": (1.40, 1.54), "rear_window_top": (2.66, 2.84),
        "door_bottom": (0.50, 0.62), "pillar_spacing": (1.35, 1.60), "corner_radius": (0.05, 0.55),
    },
}

# Trucks: one family per body, around the truck presets. `height` stays over the body it carries.
TRUCK_RANGES = {
    "camion_80": {
        "length": (6.80, 7.80), "width": (2.30, 2.50), "height": (2.60, 2.90),
        "wheelbase": (3.60, 4.10), "front_overhang": (1.10, 1.30), "track": (1.90, 2.05),
        "wheel_radius": (0.50, 0.54), "tyre_width": (0.26, 0.30),
        "cab_length": (1.45, 1.65), "cab_height": (2.50, 2.70),
        "bonnet_length": (1.25, 1.45), "bonnet_height": (1.28, 1.42), "nose_height": (1.02, 1.15),
        "deck_height": (1.05, 1.25), "side_height": (0.35, 0.50),
    },
    "furgon": {
        "length": (7.20, 8.40), "width": (2.40, 2.50), "height": (3.50, 3.70),
        "wheelbase": (4.00, 4.60), "front_overhang": (1.15, 1.35), "track": (1.98, 2.10),
        "wheel_radius": (0.50, 0.54), "tyre_width": (0.26, 0.30),
        "cab_length": (1.75, 1.90), "cab_height": (2.55, 2.70),
        "body_height": (3.20, 3.50), "body_floor": (1.00, 1.15),
    },
    "cisterna": {
        "length": (9.00, 10.00), "width": (2.40, 2.50), "height": (3.20, 3.40),
        "wheelbase": (4.30, 4.70), "front_overhang": (1.30, 1.50), "track": (2.00, 2.10),
        "wheel_radius": (0.50, 0.54), "tyre_width": (0.26, 0.30),
        "cab_length": (1.50, 1.70), "cab_height": (2.55, 2.70),
        "tank_radius": (0.95, 1.05), "tank_center": (1.95, 2.10),
    },
    "volquete": {
        "length": (7.80, 8.60), "width": (2.45, 2.55), "height": (2.90, 3.10),
        "wheelbase": (4.00, 4.40), "front_overhang": (1.25, 1.40), "track": (2.00, 2.10),
        "wheel_radius": (0.52, 0.56), "tyre_width": (0.28, 0.32),
        "cab_length": (1.80, 1.95), "cab_height": (2.65, 2.80),
        "bin_height": (1.25, 1.45), "body_floor": (1.10, 1.25),
    },
    "pasajeros": {
        "length": (7.00, 8.00), "width": (2.40, 2.50), "height": (3.25, 3.45),
        "wheelbase": (3.90, 4.30), "front_overhang": (1.15, 1.35), "track": (1.98, 2.10),
        "wheel_radius": (0.50, 0.54), "tyre_width": (0.26, 0.30),
        "cab_length": (1.70, 1.85), "cab_height": (2.55, 2.70),
        "body_height": (3.00, 3.25), "body_floor": (1.00, 1.15),
    },
}

# Cycles: each family around its preset. A cycle's wheelbase always clears its two wheels.
CYCLE_RANGES = {
    "moto_electrica": {
        "wheel_radius": (0.24, 0.26), "tyre_width": (0.07, 0.08), "wheelbase": (1.22, 1.34),
        "tube_radius": (0.022, 0.024), "handlebar_width": (0.60, 0.65), "seat_width": (0.24, 0.28),
    },
    "moto_dos_tiempos": {
        "wheel_radius": (0.29, 0.31), "tyre_width": (0.08, 0.09), "wheelbase": (1.30, 1.42),
        "tube_radius": (0.021, 0.023), "handlebar_width": (0.62, 0.66), "seat_width": (0.22, 0.26),
    },
    "triciclo_carga": {
        "wheel_radius": (0.30, 0.32), "tyre_width": (0.07, 0.08), "wheelbase": (1.85, 1.95),
        "rear_track": (0.90, 0.98), "tube_radius": (0.021, 0.023), "canopy_height": (1.90, 2.00),
    },
    "triciclo_pasajeros": {
        "wheel_radius": (0.32, 0.34), "tyre_width": (0.04, 0.05), "wheelbase": (1.70, 1.80),
        "rear_track": (0.88, 0.96), "canopy_height": (2.00, 2.14),
        "bench_length": (0.84, 0.92), "bench_width": (0.86, 0.94),
    },
    "bicitaxi": {
        "wheel_radius": (0.32, 0.34), "tyre_width": (0.04, 0.05), "wheelbase": (1.70, 1.80),
        "rear_track": (0.88, 0.96), "canopy_height": (2.00, 2.14),
        "bench_length": (0.84, 0.92), "bench_width": (0.86, 0.94),
    },
    "bicicleta": {
        "wheel_radius": (0.32, 0.35), "tyre_width": (0.04, 0.05), "wheelbase": (1.08, 1.18),
        "tube_radius": (0.018, 0.021), "handlebar_width": (0.54, 0.60), "seat_width": (0.18, 0.22),
    },
}


# --- the families --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """One family inside a kind: its era, builder, reference preset, variant count and parameter ranges.

    `bases` names the car families a livery paints over, and is empty for a family that has its own shape.
    `hatch` is the probability that a modern family draws a hatchback body instead of a three-box saloon; when it
    does, `hatch_ranges` replaces the rear stations.
    """

    name: str
    kind: str
    era: str
    builder: str
    preset: Callable[[], object]
    variants: int
    ranges: dict
    bases: tuple[str, ...] = ()
    livery: str | None = None
    hatch: float = 0.0
    hatch_ranges: dict = field(default_factory=dict)


def _car_family(kind: str, era: str, name: str, preset, variants: int, hatch: float = 0.0,
                hatch_ranges: dict | None = None, **narrow) -> Family:
    """A car family: its era's ranges, with the family's own narrower ones drawn over them.

    `hatch` and `hatch_ranges` describe the family's hatchback body and are not ranges themselves, so they are
    taken out before the rest becomes the draw table.
    """
    return Family(name=name, kind=kind, era=era, builder="car", preset=preset, variants=variants,
                  ranges={**CAR_RANGES[era], **narrow}, hatch=hatch, hatch_ranges=hatch_ranges or {})


def _livery_family(kind: str, name: str, bases: tuple[str, ...], variants: int, livery: str) -> Family:
    """A livery over car families: no shape of its own, and the union of its bases' ranges for the Garaje.

    Each variant still draws from one base family's ranges (see :func:`variant`), because a union of two eras'
    boxes holds pairs that do not build together.
    """
    members = [FAMILIES[base] for base in bases]
    every = sorted({key for member in members for key in member.ranges})
    merged = {}
    for key in every:
        low = min(member.ranges[key][0] for member in members if key in member.ranges)
        high = max(member.ranges[key][1] for member in members if key in member.ranges)
        merged[key] = (low, high)
    return Family(name=name, kind=kind, era="services", builder="car", preset=members[0].preset,
                  variants=variants, ranges=merged, bases=bases, livery=livery)


FAMILIES: dict[str, Family] = {}


def _add(*families: Family) -> None:
    """Add families to the table by name."""
    for family in families:
        FAMILIES[family.name] = family


# Cars: eight variants per era, split over the era's families.
_add(
    _car_family("car", "1950s", "almendron", almendron, 8,
                length=(4.90, 5.40), wheelbase=(2.90, 3.10), track=(1.46, 1.55)),
    _car_family("car", "80s", "ruso", ruso, 4,
                length=(3.70, 3.98), width=(1.55, 1.64), wheelbase=(2.40, 2.47), front_overhang=(0.72, 0.80)),
    _car_family("car", "80s", "moscovita", moscovita, 4,
                length=(4.02, 4.30), width=(1.56, 1.66), wheelbase=(2.46, 2.50), front_overhang=(0.78, 0.88)),
    _car_family("car", "modern", "moderno", moderno, 4,
                hatch=0.4, hatch_ranges={"hatch_u": (0.90, 0.96), "hatch_height": (1.04, 1.10),
                                         "tail_top": (1.00, 1.08)}),
    _car_family("car", "modern", "huevito", huevito, 4,
                length=(3.60, 4.00), width=(1.55, 1.70), height=(1.50, 1.65),
                wheelbase=(2.35, 2.50), front_overhang=(0.60, 0.75), track=(1.32, 1.42),
                roof_peak_u=(0.56, 0.62), roof_end_u=(0.74, 0.80), hatch_u=(0.90, 0.96),
                hatch_height=(1.02, 1.12), tail_top=(1.00, 1.10)),
)

# Taxis and the police: liveries over the car families, never a shape of their own.
_add(
    _livery_family("taxi", "compartido", ("almendron",), 4, "shared"),
    _livery_family("taxi", "estatal", ("ruso", "moderno"), 4, "state"),
    _livery_family("police", "policia", ("ruso", "moderno"), 2, "police"),
)

# Buses, trucks and cycles: each family with its own preset.
_add(
    Family(name="metroguagua", kind="bus", era="modern", builder="bus", preset=metroguagua, variants=2,
           ranges=BUS_RANGES["metroguagua"]),
    Family(name="guagua_vieja", kind="bus", era="80s", builder="bus", preset=guagua_vieja, variants=2,
           ranges=BUS_RANGES["guagua_vieja"]),
    Family(name="interprovincial", kind="bus", era="modern", builder="bus", preset=interprovincial, variants=2,
           ranges=BUS_RANGES["interprovincial"]),
    Family(name="camion_80", kind="truck", era="80s", builder="truck", preset=camion_80, variants=2,
           ranges=TRUCK_RANGES["camion_80"]),
    Family(name="furgon", kind="truck", era="80s", builder="truck", preset=furgon, variants=2,
           ranges=TRUCK_RANGES["furgon"]),
    Family(name="cisterna", kind="truck", era="80s", builder="truck", preset=cisterna, variants=2,
           ranges=TRUCK_RANGES["cisterna"]),
    Family(name="volquete", kind="truck", era="modern", builder="truck", preset=volquete, variants=2,
           ranges=TRUCK_RANGES["volquete"]),
    Family(name="pasajeros", kind="truck", era="modern", builder="truck", preset=camion_pasajeros, variants=2,
           ranges=TRUCK_RANGES["pasajeros"]),
    Family(name="moto_electrica", kind="moto", era="modern", builder="cycle", preset=moto_electrica, variants=3,
           ranges=CYCLE_RANGES["moto_electrica"]),
    Family(name="moto_dos_tiempos", kind="moto", era="80s", builder="cycle", preset=moto_dos_tiempos, variants=2,
           ranges=CYCLE_RANGES["moto_dos_tiempos"]),
    Family(name="triciclo_carga", kind="tricycle", era="modern", builder="cycle", preset=triciclo_carga,
           variants=1, ranges=CYCLE_RANGES["triciclo_carga"]),
    Family(name="triciclo_pasajeros", kind="tricycle", era="modern", builder="cycle",
           preset=triciclo_pasajeros, variants=1, ranges=CYCLE_RANGES["triciclo_pasajeros"]),
    Family(name="bicitaxi", kind="bicitaxi", era="modern", builder="cycle", preset=bicitaxi, variants=2,
           ranges=CYCLE_RANGES["bicitaxi"]),
    Family(name="bicicleta", kind="bike", era="modern", builder="cycle", preset=bicicleta, variants=2,
           ranges=CYCLE_RANGES["bicicleta"]),
    Family(name="bomberos", kind="fire", era="services", builder="truck", preset=bomberos, variants=1,
           ranges={
               "length": (8.20, 9.00), "width": (2.45, 2.55), "height": (3.15, 3.35),
               "wheelbase": (4.40, 4.80), "front_overhang": (1.40, 1.60), "track": (2.00, 2.10),
               "wheel_radius": (0.52, 0.56), "tyre_width": (0.28, 0.32),
               "cab_length": (1.90, 2.10), "cab_height": (2.65, 2.80),
               "body_height": (3.05, 3.25), "body_floor": (1.05, 1.20),
           }),
)


# --- style options -------------------------------------------------------------------------------------------

# Each option is (field, values, weights): the generator picks one value with those probabilities. Only fields
# that exist on the family's parameter class appear here, and every value sits inside `ranges` or the class's own
# choices, so a style choice never makes a draw unbuildable.
STYLES = {
    "1950s": (
        ("roof_paint", (False, True), (0.50, 0.50)),                   # a two-tone roof half the time
        ("roof_pillars", (False, True), (0.60, 0.40)),
        ("whitewall", (0.0, 0.045, 0.06), (0.40, 0.35, 0.25)),         # whitewalls six cars in ten
        ("bumper", ("chrome_bar", "blade"), (0.80, 0.20)),
        ("overriders", (True, False), (0.60, 0.40)),
        ("headlamp", ("round_pod", "round"), (0.85, 0.15)),
        ("grille_style", ("mesh",), (1.0,)),
        ("grille_span", ("full_width", "between_lamps"), (0.60, 0.40)),
        ("taillamp", ("pill",), (1.0,)),
        ("wheel", ("hubcap",), (1.0,)),
        ("windscreen_wrap", (0.20, 0.40, 0.60), (0.30, 0.40, 0.30)),
        ("side_trim", ((), ((0.72, 0.16, 0.95, "chrome"),)), (0.40, 0.60)),
    ),
    "80s": (
        ("roof_paint", (False, True), (0.85, 0.15)),
        ("whitewall", (0.0,), (1.0,)),
        ("fin_height", (0.0,), (1.0,)),
        ("windscreen_wrap", (0.0,), (1.0,)),
        ("bumper", ("rubber", "chrome_bar"), (0.60, 0.40)),
        ("headlamp", ("rect", "round"), (0.60, 0.40)),
        ("grille_style", ("bar", "mesh"), (0.70, 0.30)),
        ("grille_span", ("between_lamps", "full_width"), (0.80, 0.20)),
        ("taillamp", ("block", "pill"), (0.75, 0.25)),
        ("wheel", ("hubcap", "spokes"), (0.70, 0.30)),
        ("door_strip", (True, False), (0.40, 0.60)),
    ),
    "modern": (
        ("roof_paint", (False, True), (0.85, 0.15)),
        ("whitewall", (0.0,), (1.0,)),
        ("fin_height", (0.0,), (1.0,)),
        ("bumper", ("skin",), (1.0,)),
        ("headlamp", ("almond", "bar", "rect"), (0.50, 0.30, 0.20)),
        ("grille_style", ("bar", "mesh"), (0.60, 0.40)),
        ("grille_span", ("between_lamps", "full_width"), (0.70, 0.30)),
        ("taillamp", ("block", "pill"), (0.60, 0.40)),
        ("wheel", ("spokes", "hubcap"), (0.60, 0.40)),
        ("spoiler", (False, True), (0.80, 0.20)),
        ("cladding", (False, True), (0.75, 0.25)),
        ("drl_bar", (True, False), (0.40, 0.60)),
    ),
}


# --- colours -------------------------------------------------------------------------------------------------

# The body colours of each era, before they are faded, each with the name the Garaje shows it under: the vehicles
# spec's lists. This is the one colour table: the generator draws the faded hexes and the Garaje's swatches are the
# same entries, in this order, with these names.
ERA_COLOURS = {
    "1950s": (("Turquesa", "#3fbfb5"), ("Crema", "#efe6cf"), ("Coral", "#e8796b"),
              ("Menta", "#a8dcc4"), ("Rosado", "#e6a6b4"), ("Amarillo pálido", "#efe0a0")),
    "80s": (("Azul marino", "#243b62"), ("Beige", "#cbbd9a"), ("Blanco", "#e8e6e0"),
            ("Ocre", "#b8862f"), ("Azul claro", "#7fa8c4"), ("Rojo", "#a83228")),
    "modern": (("Plata", "#eceff0"), ("Gris", "#b8bcc0"), ("Gris oscuro", "#6f7377"),
               ("Negro", "#232529"), ("Azul", "#2f5d8c"), ("Rojo oscuro", "#a4241c")),
}

# How far each era's paint has faded toward the sun-bleached dust of Havana.
FADE = {"1950s": 0.30, "80s": 0.22, "modern": 0.0}

# The second colour of a car: the roof of a two-tone, and the stripe of a livery.
ERA_SECONDS = {
    "1950s": ("#efe6cf", "#f2ece0", "#e5dcc6"),
    "80s": ("#e8e6e0", "#22262b", "#d8d4c8"),
    "modern": ("#eceff0", "#2a2d31", "#b8bcc0"),
}

# The service colours: a taxi's amber sign, the police's blue over white, the fire engine's red.
AMBER = "#f0b23c"
POLICE_WHITE = "#f2f4f6"
POLICE_BLUES = ("#22418c", "#16295a", "#2f5d8c", "#1d3f86")
POLICE_FLASH = "#e01b1b"
FIRE_REDS = ("#c62828", "#b32222", "#d33a2c")

# A police car's paint areas: a blue band low on the flanks and blue front doors. The hex names the paint the
# livery replaces, so the palette's `livery0` and `livery1` materials take the blue.
POLICE_AREAS = (("#111111", [(0.0, 0.30), (1.0, 0.30), (1.0, 0.60), (0.0, 0.60)]),
                ("#111111", [(0.02, 0.30), (0.34, 0.30), (0.34, 0.84), (0.02, 0.84)]))


def _rgb(colour: str) -> np.ndarray:
    """An sRGB hex colour as three floats in 0..1."""
    return np.array([int(colour[k:k + 2], 16) for k in (1, 3, 5)], float) / 255.0


def _contrast(body: str, seconds: tuple[str, ...]) -> str:
    """The first second colour that is visibly unlike the body, so a two-tone roof or a stripe reads."""
    for second in seconds:
        if np.abs(_rgb(second) - _rgb(body)).sum() > 0.45:
            return second
    return seconds[0]


def _faded(colour: str, amount: float) -> str:
    """A colour mixed toward dust, as the paint of a car left in the sun for forty years."""
    rgb = np.array([int(colour[k:k + 2], 16) for k in (1, 3, 5)], float)
    dust = np.array([211.0, 205.0, 190.0])
    out = np.rint((1 - amount) * rgb + amount * dust).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*out)


def faded(colour: str, era: str) -> str:
    """A colour as the street paints it in `era`: mixed toward dust by that era's fade."""
    return _faded(colour, FADE[era])


# The faded lists the generator draws from, and the Garaje's swatches read.
COLOURS = {era: tuple(faded(colour, era) for _, colour in colours)
           for era, colours in ERA_COLOURS.items()}

# The colours a guagua's band is painted in: the greens, cyans and oranges of the metroguaguas, and the blues
# and reds of the coaches.
BUS_ACCENTS = ("#8ccf7a", "#31b6d8", "#f0a63a", "#2f5d8c", "#c62828", "#e8d44f")

# The canopy colours a bicitaxi is topped with, one per colourway.
CANOPIES = ("#efe4c6", "#5fc4b7", "#e8796b", "#7fa8c4")


# --- the Garaje's catalogue ----------------------------------------------------------------------------------


def _swatch(colour: str, roughness: float = 0.45, metallic: float = 0.0) -> tuple:
    """A palette entry: an sRGB colour, a roughness and a metallic, as the builders' palettes take them."""
    return (colour, roughness, metallic)


@dataclass(frozen=True)
class Preset:
    """One reference vehicle of the catalogue: its key, the name the game shows, its family, tab and era, its
    reference parameters and colours, and the facts its card shows.

    `drivable` is False for cycles until riders exist. `livery` is the Garaje's livery the preset wears, if any.
    """

    key: str
    name: str
    family: str
    tab: str
    era: str
    params: object
    palette: dict
    seats: int
    where: str
    livery: str = "Ninguno"
    drivable: bool = True


@functools.cache
def catalogue() -> dict[str, Preset]:
    """Every reference vehicle by key, in the order the Garaje's Salón lists it.

    Each preset takes its parameters from the same function its traffic family draws from, so a shape fixed once
    is fixed for the traffic and for the Garaje together. The palette is only the colours the vehicle is shown
    with; the geometry is the preset's.
    """
    presets = [
        Preset("almendron", "Almendrón", "car", "Carros", "Años 50", almendron(),
               {"paint": _swatch("#7fded6"), "roof": _swatch("#f6eed6", 0.6)}, 6,
               "Rutas de almendrones, todo el día"),
        Preset("ruso", "Ruso", "car", "Carros", "Años 70-80", ruso(),
               {"paint": _swatch("#2b4a86")}, 5, "Toda la ciudad"),
        Preset("moscovita", "Moscovita", "car", "Carros", "Años 70-80", moscovita(),
               {"paint": _swatch("#f4a593")}, 5, "Toda la ciudad"),
        Preset("metroguagua", "Metroguagua", "bus", "Guaguas", "Moderno", metroguagua(),
               {"paint": _swatch("#f3efe2"), "livery0": _swatch("#8ccf7a", 0.5), "sign": _swatch(AMBER, 0.5),
                "black": _swatch("#33373a", 0.5), "glass": _swatch("#c8d8dd", 0.05),
                "chrome": _swatch("#a8adb1", 0.3)}, 80, "Sus rutas y paradas"),
        Preset("camion", "Camión de los 80", "truck", "Camiones", "Años 70-80", camion_80(),
               {"paint": _swatch("#7fa8bd", 0.4, 0.1), "body": _swatch("#7fa8bd", 0.4, 0.1),
                "deck": _swatch("#8a5a33", 0.85)}, 2, "Avenidas y puerto"),
        Preset("caja", "Camión de caja", "truck", "Camiones", "Moderno", furgon(),
               {"paint": _swatch("#f1f2f3", 0.35), "body": _swatch("#b9bdc2")}, 2, "El centro, de noche"),
        Preset("pipa", "Pipa", "truck", "Camiones", "Años 70-80", cisterna(),
               {"paint": _swatch("#e7c33b", 0.35), "body": _swatch("#c9ccd0", 0.25, 0.9),
                "tank": _swatch("#c9ccd0", 0.25, 0.9)}, 2, "Avenidas, de noche"),
        Preset("volteo", "Volteo", "truck", "Camiones", "Años 70-80", volquete(),
               {"paint": _swatch("#d96a1f", 0.4), "body": _swatch("#d96a1f", 0.4)}, 2, "Obras y afueras"),
        Preset("patrulla", "Patrulla", "car", "Servicios", "Años 70-80", patrulla(),
               {"paint": _swatch(POLICE_WHITE, 0.35), "roof": _swatch(POLICE_BLUES[0], 0.5)}, 4,
               "Donde haya lío", livery="Policía"),
        Preset("bomberos", "Bomberos", "truck", "Servicios", "Años 70-80", bomberos(),
               {"paint": _swatch(FIRE_REDS[0], 0.35), "body": _swatch(FIRE_REDS[0], 0.35),
                "locker": _swatch("#b9bdc2", 0.45, 0.6), "ladder": _swatch("#c9ccd0", 0.3, 0.8),
                "livery0": _swatch("#f4f4f2", 0.35)}, 4, "Desde su estación"),
        Preset("bicitaxi", "Bicitaxi", "cycle", "Ciclos", "Moderno", bicitaxi(),
               {"paint": _swatch("#d6dbdc", 0.35, 0.1), "box": _swatch("#5fc4b7"), "canopy": _swatch("#efe4c6", 0.65),
                "spoke": _swatch("#d8dcdd", 0.25, 0.8), "chrome": _swatch("#d8dcdd", 0.2, 0.8),
                "saddle": _swatch("#2a2a2e", 0.8)}, 2, "Centro y Habana Vieja", drivable=False),
        Preset("triciclo", "Triciclo eléctrico", "cycle", "Ciclos", "Moderno", triciclo_carga(),
               {"paint": _swatch("#ffe94f", 0.5), "box": _swatch("#ffe94f", 0.5), "canopy": _swatch("#ffe94f", 0.5),
                "seat": _swatch("#8fe0cf", 0.5), "saddle": _swatch("#8fe0cf", 0.5)}, 6, "Rutas cortas",
               drivable=False),
        Preset("moto", "Moto", "cycle", "Ciclos", "Años 70-80", moto_dos_tiempos(),
               {"paint": _swatch("#b6f2e2", 0.4, 0.05), "saddle": _swatch("#17171a", 0.85),
                "engine": _swatch("#c9c9c4", 0.45, 0.2), "chrome": _swatch("#dfe2e3", 0.15, 0.9)}, 2,
               "Toda la ciudad", drivable=False),
        Preset("motoelectrica", "Moto eléctrica", "cycle", "Ciclos", "Moderno", moto_electrica(),
               {"paint": _swatch("#ff7a62"), "saddle": _swatch("#bfe8d8", 0.6), "box": _swatch("#bfe8d8", 0.6),
                "chrome": _swatch("#c3c7c9", 0.35, 0.6), "rim": _swatch("#8e9295", 0.5, 0.6),
                "spoke": _swatch("#a9adaf", 0.5, 0.6)}, 2, "Toda la ciudad", drivable=False),
        Preset("bicicleta", "Bicicleta", "cycle", "Ciclos", "Moderno", bicicleta(),
               {"paint": _swatch("#2f5fa8")}, 1, "Toda la ciudad", drivable=False),
    ]
    return {preset.key: preset for preset in presets}


# --- ranges and variants -------------------------------------------------------------------------------------


def ranges(family: str, era: str | None = None) -> dict[str, tuple[float, float]]:
    """The ranges one family draws from, per parameter, as ``(low, high)`` in the parameter's own unit.

    `era` is checked against the family's era when it is given, so a caller that spans a builder's families cannot
    silently ask one family for another era's box. A livery family reports the union of the eras it paints over,
    and a family's table carries the measurements its style options fix as well, so the Garaje's sliders and the
    generator read one number.
    """
    if family not in FAMILIES:
        raise KeyError(f"fleet: no family named {family!r}; the families are {sorted(FAMILIES)}")
    fam = FAMILIES[family]
    if era is not None and era != fam.era:
        raise KeyError(f"fleet: family {family!r} is from the {fam.era!r} era, not {era!r}")
    return dict(fam.ranges)


def _seed(*parts) -> int:
    """A stable 64-bit seed from strings and numbers.

    Python's ``hash`` is salted per process, so it would give a different fleet on every run; SHA-256 does not.
    """
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class Variant:
    """One buildable variation of a family: its parameters and the colourways it is painted in.

    A colourway is a palette of material name -> ``(sRGB colour, roughness, metallic)``, in the form the builders'
    ``Vehicle.merged`` and ``spawn_car`` take.
    """

    kind: str
    family: str
    era: str
    k: int
    builder: str
    params: object
    colourways: tuple[dict, ...]

    def build(self):
        """The built vehicle, through the family's manifoldx builder."""
        return BUILDERS[self.builder](self.params)


def _base(family: Family, k: int) -> Family:
    """The car family a livery variant paints over, or the family itself when it has its own shape."""
    if not family.bases:
        return family
    return FAMILIES[family.bases[k % len(family.bases)]]


def _draw(base: Family, family: Family, k: int, rng: np.random.Generator) -> object:
    """One parameter set of `base`, with `family`'s style options, in the order the class declares its fields.

    The numeric draws go in field-name order so the stream does not depend on how the ranges dict was written,
    and the style options come after them. The measurements in :data:`STYLE_ONLY` are not drawn: the preset or a
    style option fixes them, and they sit in the table only so the Garaje can offer their whole range.
    """
    values = {}
    for name in sorted(base.ranges):
        if name in STYLE_ONLY:
            continue
        low, high = base.ranges[name]
        values[name] = round(float(rng.uniform(low, high)), 3)
    for name, choices, weights in (STYLES.get(base.era, ()) if base.builder == "car" else ()):
        values[name] = choices[int(rng.choice(len(choices), p=weights))]
    if family.hatch and rng.random() < family.hatch:
        for name, (low, high) in family.hatch_ranges.items():
            values[name] = round(float(rng.uniform(low, high)), 3)
    params = replace(base.preset(), **values)
    return _paint(family, params, k)


def _paint(family: Family, params: object, k: int) -> object:
    """Give a taxi or a police car its livery and its roof lamp; every other family keeps its preset's paint."""
    if family.livery == "shared":
        # A shared taxi of the 1950s is an almendrón with an amber sign over the roof.
        return replace(params, lightbar="dome", lightbar_colours=(AMBER, AMBER))
    if family.livery == "state":
        # A state taxi wears a band of amber along the flanks and the same sign.
        return replace(
            params, lightbar="dome", lightbar_colours=(AMBER, AMBER),
            livery=(("#111111", [(0.0, 0.42), (1.0, 0.42), (1.0, 0.58), (0.0, 0.58)]),),
        )
    if family.livery == "police":
        # White paint, a blue band low on the flanks, blue front doors and a roof bar of two colours.
        return replace(params, lightbar="bar", lightbar_colours=(POLICE_BLUES[0], POLICE_FLASH),
                       livery=POLICE_AREAS)
    return params


def _car_colourways(base: Family, k: int, rng: np.random.Generator) -> tuple[dict, ...]:
    """Three colourways of a car: a body colour of its era and a second colour for the roof or the stripe."""
    colours = COLOURS[base.era]
    seconds = ERA_SECONDS[base.era]
    out = []
    for j in range(3):
        body = colours[(5 * k + j) % len(colours)]
        out.append({"paint": (body, 0.35, 0.05), "roof": (_contrast(body, seconds), 0.35, 0.05)})
    return tuple(out)


def _colourways(family: Family, base: Family, k: int, rng: np.random.Generator) -> tuple[dict, ...]:
    """The colourways of one variant, in the material names its builder paints with."""
    if family.livery == "police":
        out = []
        for j in range(3):
            blue = POLICE_BLUES[(k + j) % len(POLICE_BLUES)]
            out.append({"paint": (POLICE_WHITE, 0.30, 0.05), "roof": (POLICE_WHITE, 0.30, 0.05),
                        "livery0": (blue, 0.35, 0.0), "livery1": (blue, 0.35, 0.0)})
        return tuple(out)
    if family.livery == "state":
        out = []
        for j in range(3):
            body = COLOURS[base.era][(5 * k + j) % len(COLOURS[base.era])]
            out.append({"paint": (body, 0.35, 0.05),
                        "roof": (_contrast(body, ERA_SECONDS[base.era]), 0.35, 0.05),
                        "livery0": (AMBER, 0.35, 0.0)})
        return tuple(out)
    if family.builder == "car":
        return _car_colourways(base, k, rng)
    if family.builder == "bus":
        return _bus_colourways(base, k)
    if family.builder == "truck":
        return _truck_colourways(family, base, k)
    return _cycle_colourways(family, base, k)


def _bus_colourways(base: Family, k: int) -> tuple[dict, ...]:
    """Three cream buses with a band in a colour of their era."""
    bodies = ("#f0ece0", "#e6e8e4", "#e0e6e8")
    out = []
    for j in range(3):
        body = bodies[(k + j) % len(bodies)]
        out.append({"paint": (body, 0.40, 0.0), "livery0": (BUS_ACCENTS[(5 * k + j) % len(BUS_ACCENTS)], 0.40, 0.0)})
    return tuple(out)


def _truck_colourways(family: Family, base: Family, k: int) -> tuple[dict, ...]:
    """Three trucks: a cab in a colour of its era over a body of its trade's colour."""
    if family.name == "bomberos":
        return tuple({"paint": (red, 0.35, 0.0), "body": (red, 0.35, 0.0),
                      "livery0": ("#f4f4f2", 0.35, 0.0)} for red in FIRE_REDS)
    colours = COLOURS[base.era]
    bodies = {
        "camion_80": ("#8a5a33", "#7d5230", "#96703f"),   # a wooden flatbed
        "furgon": ("#b9bdc2", "#c6cacf", "#aeb2b7"),       # a grey cargo box
        "cisterna": ("#c9ccd0", "#d2d5d9", "#c1c4c8"),     # a silver tank
        "volquete": (None, None, None),                     # painted with the cab
        "pasajeros": ("#cfc9b4", "#d8d2be", "#c6c0ab"),    # a canvas-covered bed
    }[family.name]
    out = []
    for j in range(3):
        cab = colours[(5 * k + j) % len(colours)]
        body = cab if bodies[j] is None else bodies[j]
        out.append({"paint": (cab, 0.40, 0.05), "body": (body, 0.45, 0.0)})
    return tuple(out)


def _cycle_colourways(family: Family, base: Family, k: int) -> tuple[dict, ...]:
    """Three cycles: a painted frame, a box or bench and, for a bicitaxi, the four canopy colours."""
    colours = COLOURS[base.era]
    frames = ("#b9c0c4", "#d6dbdc", "#a9adaf")   # the silver of a bicycle frame
    if family.name == "bicitaxi":
        return tuple({"paint": (frames[j % len(frames)], 0.35, 0.10),
                      "box": ("#5fc4b7", 0.45, 0.0),
                      "canopy": (canopy, 0.65, 0.0)} for j, canopy in enumerate(CANOPIES))
    boxes = {
        "moto_electrica": ("#bfe8d8", "#e8d8bf", "#d8c4e8"),
        "moto_dos_tiempos": ("#1c1c1e", "#2a2a2c", "#3a3a3c"),
        "triciclo_carga": ("#ffe94f", "#ffd24f", "#e8e04a"),
        "triciclo_pasajeros": ("#7fa8c4", "#8fbccc", "#6f98b4"),
        "bicicleta": ("#6f7377", "#8fbccc", "#c6cacf"),
    }[family.name]
    canopies = {
        "moto_electrica": ("#bfe8d8", "#c4e8d8", "#b4e0d0"),
        "moto_dos_tiempos": ("#1c1c1e", "#242426", "#2c2c2e"),
        "triciclo_carga": ("#ffe94f", "#ffd24f", "#e8e04a"),
        "triciclo_pasajeros": ("#efe4c6", "#e6dcc0", "#f4ecd4"),
        "bicicleta": ("#6f7377", "#8fbccc", "#c6cacf"),
    }[family.name]
    out = []
    for j in range(3):
        frame = colours[(5 * k + j) % len(colours)]
        out.append({"paint": (frame, 0.40, 0.05), "box": (boxes[j], 0.45, 0.0),
                    "canopy": (canopies[j], 0.65, 0.0)})
    return tuple(out)


def variant(kind: str, family: str, k: int) -> Variant:
    """Variant `k` of `family`, a family of `kind`: its parameters and its colourways.

    The draw is seeded from the kind, the family, the base family it paints over and `k`, so the same arguments
    give the same vehicle on every machine. A draw that `validate` rejects is retried from a derived seed; a family
    whose draws keep failing is a bug in its ranges, so the error names the family instead of hiding it.
    """
    if family not in FAMILIES:
        raise KeyError(f"fleet: no family named {family!r}; the families are {sorted(FAMILIES)}")
    fam = FAMILIES[family]
    if fam.kind != kind:
        raise KeyError(f"fleet: family {family!r} is a {fam.kind!r}, not a {kind!r}")
    if not 0 <= k < fam.variants:
        raise IndexError(f"fleet: family {family!r} has variants 0..{fam.variants - 1}, not {k}")
    base = _base(fam, k)
    for attempt in range(ATTEMPTS):
        rng = np.random.default_rng(_seed(kind, family, base.name, k, attempt))
        params = _draw(base, fam, k, rng)
        try:
            params.validate()
        except ValueError:
            continue
        return Variant(kind=kind, family=family, era=base.era, k=k, builder=fam.builder,
                       params=params, colourways=_colourways(fam, base, k, rng))
    raise ValueError(
        f"fleet: {kind}/{family} variant {k} failed validate() in {ATTEMPTS} draws; "
        f"its ranges are inconsistent and must be fixed, not drawn around"
    )
