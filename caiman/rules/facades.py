"""The facade grammar: what a building is, from what OSM says about it and where it stands.

plan() turns a building's OSM id, its tags (buildingtags) and its municipio into a Plan: one of TYPES, the
storeys, the ground floor's and the other floors' heights, the bay width, the parapet, the colour and how
worn it is. Every level of detail reads the same plan, so a building keeps its height and colour from the
detail ring out to the far summaries. The draws are seeded by the OSM id: a building is the same on every
drive. Most of Havana's buildings say only building=yes, so the municipio decides most types.
"""

from dataclasses import dataclass

import numpy as np

from caiman.buildingtags import HEIGHT, KIND, KINDS, LEVELS, MUNI, USES


@dataclass(frozen=True)
class Type:
    storeys: tuple[int, int]  # drawn when OSM gives neither levels nor height
    ground: tuple[float, float]  # the ground floor's height, metres
    floor: tuple[float, float]  # every other floor's
    bay: tuple[float, float]  # the width of one window bay
    parapet: float  # metres of wall above the roof
    palette: tuple[str, ...]  # wall colours, sRGB
    wear: tuple[float, float]  # the share of the colour left after weather: 1 is new
    portal: bool = False  # a portal with columns along a historic street front


TYPES = {
    "colonial": Type((1, 3), (4.8, 5.6), (4.5, 5.0), (3.6, 4.4), 1.0,
                     ("#d9a441", "#5b8fb9", "#7fae7a", "#e39b7b", "#e8d39a", "#c98b6b"), (0.72, 0.9)),
    "eclectic": Type((2, 4), (4.4, 5.0), (3.9, 4.4), (3.4, 4.0), 1.1,
                     ("#e7c3a1", "#9dc3c1", "#e9d48a", "#c9a3c7", "#f0b9a0", "#b8cfa0"), (0.72, 0.9), portal=True),
    "vedado": Type((1, 3), (3.6, 4.0), (3.4, 3.8), (3.8, 4.6), 0.8,
                   ("#f0ece2", "#cfe0d2", "#f2d7b6", "#dfe6ef", "#f3e3a8"), (0.8, 0.95)),
    "tower": Type((8, 20), (4.0, 4.6), (3.0, 3.1), (3.0, 3.4), 1.2,
                  ("#e8e8e4", "#b9c6cc", "#d6e4ee", "#ece2cf"), (0.78, 0.92)),
    "prefab": Type((4, 5), (2.8, 2.8), (2.7, 2.7), (2.8, 3.2), 0.9,
                   ("#c9c4b8", "#e0d6b8", "#c7d3c9", "#d9c2b0", "#b9c4cf"), (0.7, 0.85)),
    "suburban": Type((1, 2), (3.0, 3.3), (2.9, 3.0), (3.0, 3.6), 0.5,
                     ("#7fc0d9", "#f2b84b", "#9ccf83", "#f08b8b", "#f4f0e6", "#c7a6e0", "#f6d36b"), (0.75, 0.95)),
    "warehouse": Type((1, 1), (6.0, 9.0), (6.0, 9.0), (5.0, 6.0), 0.6,
                      ("#a8a9a3", "#b8b19c", "#9fa6a8", "#c2bba6"), (0.75, 0.9)),
}
ROOF = (0.55, 0.54, 0.52)  # flat roofs: grey waterproofing, worn with the walls
TALL = {"church": 1.5, "school": 1.25, "hospital": 1.25, "public": 1.3}  # ground floors of these, taller


@dataclass(frozen=True)
class Plan:
    type: str
    storeys: int
    ground: float  # the ground floor's height
    floor: float  # every other floor's
    bay: float
    parapet: float
    color: tuple[float, float, float]
    roof_color: tuple[float, float, float]

    @property
    def roof(self) -> float:
        """The roof's height above the building's base."""
        return self.ground + (self.storeys - 1) * self.floor

    @property
    def height(self) -> float:
        """The top of the walls above the base, parapet included."""
        return self.roof + self.parapet


def _rgb(hex_: str) -> np.ndarray:
    return np.array([int(hex_[k:k + 2], 16) / 255 for k in (1, 3, 5)])


def choose(kind: str, muni: str | None, area: float, levels: float, height: float, rng) -> str:
    """The building type, from OSM's kind, levels and height first, then the municipio and the footprint."""
    if kind in ("industrial", "warehouse") or (kind in ("retail", "commercial") and area > 1500):
        return "warehouse"
    if levels >= 8 or height >= 25:
        return "tower"
    if muni == "La Habana Vieja":
        return "colonial"
    if muni in ("Centro Habana", "Cerro"):
        return "eclectic"
    if muni in ("Regla", "Guanabacoa"):
        return "colonial" if area < 500 else "eclectic"
    if muni == "Diez de Octubre":
        return "eclectic" if area > 150 and rng.random() < 0.5 else "suburban"
    if muni in ("Plaza de la Revolución", "Playa"):
        if area > 900:
            return "tower"
        return "prefab" if kind == "apartments" or area > 500 else "vedado"
    if kind == "apartments" or (muni == "Habana del Este" and area > 300):
        return "prefab"
    return "suburban"


def plan(osm_id: int, attrs=None, muni: str | None = None, area: float = 100.0) -> Plan:
    """The plan of one building: attrs is its building_attrs row (None for a filler box) and muni its
    municipio's name."""
    rng = np.random.default_rng(abs(int(osm_id)))
    kind = KINDS[int(attrs[KIND])] if attrs is not None else "yes"
    levels = float(attrs[LEVELS]) if attrs is not None and np.isfinite(attrs[LEVELS]) else float("nan")
    height = float(attrs[HEIGHT]) if attrs is not None and np.isfinite(attrs[HEIGHT]) else float("nan")
    known_levels, known_height = not np.isnan(levels), not np.isnan(height)
    name = choose(kind, muni, area, levels if known_levels else 0.0, height if known_height else 0.0, rng)
    t = TYPES[name]
    if name == "tower" and muni in ("Plaza de la Revolución", "Playa") and not (known_levels or known_height):
        storeys = int(rng.integers(5, 13))
    else:
        storeys = int(rng.integers(t.storeys[0], t.storeys[1] + 1))
    ground, floor = rng.uniform(*t.ground), rng.uniform(*t.floor)
    ground *= TALL.get(kind, 1.0)
    if known_levels:
        storeys = int(np.clip(round(levels), 1, 60))
    elif known_height:
        storeys = int(np.clip(round((height - ground) / floor) + 1, 1, 60))
    wear = rng.uniform(*t.wear)
    color = _rgb(t.palette[int(rng.integers(len(t.palette)))]) * wear
    if attrs is not None and int(attrs[USES]) & 1:  # historic: kept up a little better
        color = np.minimum(color / wear * min(1.0, wear + 0.1), 1.0)
    roof = np.array(ROOF) * (0.85 + 0.15 * wear)
    return Plan(name, storeys, float(ground), float(floor), float(rng.uniform(*t.bay)), t.parapet,
                tuple(float(c) for c in color), tuple(float(c) for c in roof))


def municipio(attrs, names) -> str | None:
    """The municipio's name in a building_attrs row, or None."""
    k = int(attrs[MUNI]) if attrs is not None and np.isfinite(attrs[MUNI]) else -1
    return str(names[k]) if 0 <= k < len(names) else None
