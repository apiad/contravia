"""How each family of the fleet drives: mass, engine, gearbox, suspension, tyres and brakes.

The numbers are the starting point of the vehicle-physics spec (section 09) and are tuned by driving. The body's
measurements come from the vehicle built for the family (CarParams, BusParams or TruckParams all carry length,
width, height, wheelbase, front_overhang, track and wheel_radius); the mass here is the preset's, scaled by the
built body's volume, so a longer Garaje design weighs more.
"""

from __future__ import annotations

from dataclasses import dataclass

from manifoldx.vehicles import VehicleSpec, car_spec

from contravia.rules import fleet


@dataclass(frozen=True)
class Handling:
    mass: float  # kg, at the family preset's size
    power_kw: float
    drive: str  # "rear", "front" or "all"
    gears: tuple[float, ...]
    final: float
    peak_rpm: float  # where the power peaks
    spring_hz: tuple[float, float]  # front and rear
    damping: float  # share of critical
    arb: tuple[float, float]  # N/m at the preset's mass
    tyre_mu: float
    brake_torque: float  # N·m over all wheels, at the preset's mass
    brake_front: float
    cg_height: float  # share of the body's height
    cd: float
    max_steer: float = 0.6
    travel: float = 0.2


FOUR = (3.7, 2.1, 1.4, 1.0)
HANDLING: dict[str, Handling] = {
    # soft and floaty, drum brakes, a big lazy engine
    "almendron": Handling(1600, 75, "rear", (2.5, 1.5, 1.0), 4.1, 4400, (1.1, 1.1), 0.25, (5000, 2000), 0.85,
                          3800, 0.55, 0.38, 0.45, travel=0.25),
    "ruso": Handling(1030, 52, "rear", FOUR, 4.1, 5600, (1.3, 1.3), 0.3, (6000, 2500), 0.9, 4000, 0.6, 0.38, 0.45),
    "moscovita": Handling(1080, 55, "rear", FOUR, 4.2, 5800, (1.2, 1.2), 0.3, (5500, 2500), 0.9, 4000, 0.6, 0.40,
                          0.45),
    "moderno": Handling(1250, 85, "front", (3.6, 2.1, 1.4, 1.0, 0.8), 3.9, 6000, (1.6, 1.6), 0.35, (9000, 4000), 1.0,
                        5500, 0.65, 0.38, 0.32),
    "huevito": Handling(620, 18, "rear", (3.25, 2.06, 1.3, 0.87), 4.9, 5000, (1.4, 1.4), 0.3, (3000, 1500), 0.9,
                        2200, 0.55, 0.38, 0.40),
    "patrulla": Handling(1250, 100, "front", (3.6, 2.1, 1.4, 1.0, 0.8), 3.9, 6000, (1.6, 1.6), 0.35, (9000, 4000),
                         1.0, 5500, 0.65, 0.38, 0.32),
    "guagua": Handling(11000, 180, "rear", (3.5, 2.0, 1.35, 1.0), 5.3, 2200, (1.0, 1.0), 0.35, (90000, 60000), 0.85,
                       40000, 0.55, 0.55, 0.70, travel=0.25),
    "camion": Handling(8000, 130, "rear", (6.0, 3.5, 2.1, 1.4, 1.0), 4.5, 2600, (1.2, 1.2), 0.35, (60000, 40000),
                       0.85, 30000, 0.55, 0.55, 0.80, travel=0.25),
}
BY_KIND = {"bus": "guagua", "truck": "camion", "fire": "camion", "police": "patrulla"}


def handling_of(family: str) -> Handling:
    """The handling of a fleet family: its own, its kind's, or for a livery its first base family's.
    Raises KeyError for families nobody drives (motos, tricycles, bicitaxis, bikes)."""
    if family in HANDLING:
        return HANDLING[family]
    f = fleet.FAMILIES.get(family)
    if f is None:
        raise KeyError(f"no fleet family {family!r}")
    if f.kind in BY_KIND:
        return HANDLING[BY_KIND[f.kind]]
    if f.bases:
        return HANDLING[f.bases[0]]
    raise KeyError(f"family {family!r} ({f.kind}) is not driven")


def _preset_of(family: str):
    if family in fleet.FAMILIES:
        return fleet.FAMILIES[family].preset()
    return {"patrulla": fleet.patrulla, "guagua": fleet.metroguagua, "camion": fleet.camion_80}[family]()


def spec_for(params, family: str) -> VehicleSpec:
    """The VehicleSpec of a vehicle built from params for this family."""
    h = handling_of(family)
    preset = _preset_of(family)
    scale = (params.length * params.width * params.height) / (preset.length * preset.width * preset.height)
    mass = h.mass * scale
    return car_spec(length=params.length, width=params.width, height=params.height, wheelbase=params.wheelbase,
                    front_overhang=params.front_overhang, track=params.track, wheel_radius=params.wheel_radius,
                    mass=mass, power_kw=h.power_kw, drive=h.drive, ratios=h.gears, final=h.final,
                    spring_hz=h.spring_hz, damping=h.damping, arb=(h.arb[0] * scale, h.arb[1] * scale),
                    tyre_mu=h.tyre_mu, brake_torque=h.brake_torque * scale, brake_front=h.brake_front,
                    cg_height=h.cg_height, cd=h.cd, max_steer=h.max_steer, travel=h.travel,
                    peak_power_rpm=h.peak_rpm)
