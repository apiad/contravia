"""The base layer of regions changes no building: every building and filler box in 49 cells of Centro Habana,
La Habana Vieja, Vedado and Guanabacoa gets the same Plan from contravia/regions.toml as from the municipio
rules facades.choose held in code before regions, copied below. It reads the real island in data/cuba."""

import os
from pathlib import Path

import numpy as np
import pytest

from contravia import regions
from contravia.buildingtags import HEIGHT, KIND, KINDS, LEVELS, USES
from contravia.geo import CELL, to_local
from contravia.mesh import signed_area
from contravia.rules.buildings import filler_rings
from contravia.rules.facades import ROOF, TALL, TYPES, Plan, _rgb, municipio, plan

DATA = Path(os.environ.get("HAVANA_DATA", "data/cuba"))
PORTAL_MUNIS = ("La Habana Vieja", "Centro Habana", "Cerro", "Diez de Octubre")
PLACES = {"Centro Habana": ((23.1375, -82.3700), 13), "La Habana Vieja": ((23.1375, -82.3525), 12),
          "Vedado": ((23.1350, -82.3900), 12), "Guanabacoa": ((23.1240, -82.2960), 12)}

def _old_choose(kind: str, muni: str | None, area: float, levels: float, height: float, rng) -> str:
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


def _old_plan(osm_id: int, attrs=None, muni: str | None = None, area: float = 100.0) -> Plan:
    """The plan of one building: attrs is its building_attrs row (None for a filler box) and muni its
    municipio's name."""
    rng = np.random.default_rng(abs(int(osm_id)))
    kind = KINDS[int(attrs[KIND])] if attrs is not None else "yes"
    levels = float(attrs[LEVELS]) if attrs is not None and np.isfinite(attrs[LEVELS]) else float("nan")
    height = float(attrs[HEIGHT]) if attrs is not None and np.isfinite(attrs[HEIGHT]) else float("nan")
    known_levels, known_height = not np.isnan(levels), not np.isnan(height)
    name = _old_choose(kind, muni, area, levels if known_levels else 0.0, height if known_height else 0.0, rng)
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
    portal = name in ("eclectic", "colonial") and muni in PORTAL_MUNIS and storeys >= 2
    return Plan(name, storeys, float(ground), float(floor), float(rng.uniform(*t.bay)), t.parapet,
                tuple(float(c) for c in color), tuple(float(c) for c in roof), portal)
def _cells():
    """The 49 cells: those nearest each place's centre."""
    out = []
    for (lat, lon), n in PLACES.values():
        x, z = to_local(np.array([lat]), np.array([lon]))[0]
        ci, cj = int(np.floor(x / CELL)), int(np.floor(z / CELL))
        around = sorted(((ci + di, cj + dj) for di in range(-3, 4) for dj in range(-3, 4)),
                        key=lambda c: (c[0] - ci) ** 2 + (c[1] - cj) ** 2)
        out += around[:n]
    return out


@pytest.mark.skipif(not (DATA / "la-habana.npz").exists(), reason="needs the island in data/cuba")
def test_the_base_layer_gives_every_building_and_filler_its_old_plan():
    from contravia.world import World

    world, rules = World.load(DATA), regions.Regions.load()
    cells = _cells()
    assert len(set(cells)) == 49
    count, types, munis = 0, set(), set()
    for i, j in cells:
        c = world.cell(i, j)
        attrs = c.building_attrs if c.building_attrs is not None else [None] * len(c.buildings)
        at_buildings, at_fillers = rules.cell(c)
        names = []
        for ring, osm_id, a, params in zip(c.buildings, c.building_ids, attrs, at_buildings, strict=True):
            area = abs(signed_area(np.asarray(ring, float)))
            muni = municipio(a, c.muni_names)
            names.append(muni)
            old = _old_plan(int(osm_id), a, muni, area)
            assert plan(int(osm_id), a, params, area) == old, (i, j, int(osm_id), muni)
            count, types, munis = count + 1, types | {old.type}, munis | {muni}
        zone = max(set(names), key=names.count) if names else None
        for block, seed, params in zip(c.fillers, c.filler_seeds, at_fillers, strict=True):
            for k, r in enumerate(filler_rings(block, int(seed))):
                sid, area = abs(int(seed)) * 131 + k, abs(signed_area(r))
                assert plan(sid, None, params, area) == _old_plan(sid, None, zone, area), (i, j, sid, zone)
                count += 1
    assert count > 2000 and len(types) >= 5
    assert {"Centro Habana", "La Habana Vieja", "Plaza de la Revolución", "Guanabacoa"} <= munis
