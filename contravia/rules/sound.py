"""How each fleet family sounds: the numbers the engine graph is built from (contravia/audio/engine.py).

cylinders and strokes set the firing frequency, rpm x cylinders / (60 x strokes / 2): a four-stroke
fires each cylinder every two turns, a two-stroke every turn; strokes 0 is a vehicle with no
pulses (electric or pedalled). res are the exhaust's two resonances in Hz, bright the low-pass's
cutoff in Hz at full load and redline, uneven how differently the cylinders fire (a V8's burble),
crackle the pops of a lift-off at high rpm, knock a diesel's injection click, turbo a whine with
load, and kind which of the non-combustion sounds plays: "electric" or "chain".
"""

from __future__ import annotations

from dataclasses import dataclass

BIOMES = ("city", "barrio", "countryside", "coast", "night")
STYLES = ("son", "guaguanco", "guajira", "changui", "bolero")
BY_BIOME = {"city": "son", "barrio": "guaguanco", "countryside": "guajira", "coast": "changui", "night": "bolero"}


@dataclass(frozen=True)
class FamilySound:
    cylinders: int
    strokes: int
    res: tuple[float, float] = (150.0, 480.0)
    bright: float = 1200.0
    uneven: float = 0.05
    crackle: float = 0.0
    knock: float = 0.0
    turbo: float = 0.0
    kind: str = ""


V8 = FamilySound(8, 4, (85.0, 260.0), 900.0, 0.35, crackle=0.5)  # the spike's numbers, approved by ear
LADA = FamilySound(4, 4, (170.0, 540.0), 2200.0, 0.08, crackle=0.2)
MODERN = FamilySound(4, 4, (150.0, 480.0), 1200.0, 0.05)
DIESEL = FamilySound(6, 4, (70.0, 210.0), 1400.0, 0.12, knock=0.6)

FAMILY_SOUND: dict[str, FamilySound] = {
    "almendron": V8,
    "compartido": V8,  # the shared taxis are almendrones
    "ruso": LADA,
    "moscovita": LADA,
    "moderno": MODERN,
    "estatal": MODERN,
    "policia": MODERN,
    "huevito": FamilySound(3, 4, (150.0, 480.0), 1200.0, 0.05),
    "guagua_vieja": DIESEL,
    "interprovincial": DIESEL,
    "pasajeros": DIESEL,
    "camion_80": DIESEL,
    "furgon": DIESEL,
    "cisterna": DIESEL,
    "volquete": DIESEL,
    "bomberos": DIESEL,
    "metroguagua": FamilySound(6, 4, (70.0, 210.0), 1400.0, 0.12, knock=0.4, turbo=0.3),
    "moto_dos_tiempos": FamilySound(1, 2, (300.0, 900.0), 3000.0, 0.1),
    "moto_electrica": FamilySound(0, 0, kind="electric"),
    "triciclo_carga": FamilySound(0, 0, kind="electric"),
    "triciclo_pasajeros": FamilySound(0, 0, kind="electric"),
    "bicitaxi": FamilySound(0, 0, kind="chain"),
    "bicicleta": FamilySound(0, 0, kind="chain"),
}


def of(family: str) -> FamilySound:
    """A family's sound; a family this table does not name sounds like a modern car."""
    return FAMILY_SOUND.get(family, MODERN)
