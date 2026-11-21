"""Cuba's cities and towns for the map's emblems (docs/superpowers/specs/2026-11-20-map-v2-design.html, section 05).

OSM's place nodes tagged city or town with a name, their population and where they stand, kept in towns.json next
to this module (about 20 KB, like places.json): `contravia towns <pbf>` writes it again. A town is a provincial
capital when CAPITALS names it.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PATH = Path(__file__).parent / "towns.json"
CAPITALS = frozenset({"La Habana", "Santiago de Cuba", "Camagüey", "Holguín", "Santa Clara", "Guantánamo", "Las Tunas",
                      "Bayamo", "Cienfuegos", "Pinar del Río", "Matanzas", "Ciego de Ávila", "Sancti Spíritus",
                      "Artemisa", "San José de las Lajas", "Nueva Gerona"})


@dataclass(frozen=True)
class Town:
    name: str
    place: str  # OSM's place: "city" or "town"
    pop: int  # OSM's population, 0 when it has none
    x: float  # world metres
    z: float

    @property
    def capital(self) -> bool:
        return self.name in CAPITALS


def _population(text: str) -> int:
    try:
        return int(text.replace(".", "").replace(",", "").replace(" ", ""))
    except ValueError:
        return 0


def read(pbf: str) -> list[dict]:
    """The named city and town nodes of an OSM extract, largest first, as JSON rows (lat and lon kept)."""
    import osmium

    rows = []

    class Places(osmium.SimpleHandler):
        def node(self, n):
            place, name = n.tags.get("place"), n.tags.get("name", "")
            if place in ("city", "town") and name:
                rows.append({"name": name, "place": place, "pop": _population(n.tags.get("population", "")),
                             "lat": round(n.location.lat, 6), "lon": round(n.location.lon, 6)})

    Places().apply_file(pbf)
    return sorted(rows, key=lambda r: (-r["pop"], r["name"]))


def save(rows: list[dict], path: Path = PATH) -> int:
    """Write the rows; returns the bytes written."""
    text = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text)
    return len(text.encode())


def load(path: Path = PATH) -> list[Town]:
    """The towns, largest first, at their world positions."""
    from contravia import geo

    rows = json.loads(Path(path).read_text())
    if not rows:
        return []
    xz = geo.to_local(np.array([r["lat"] for r in rows]), np.array([r["lon"] for r in rows]))
    return [Town(r["name"], r["place"], r["pop"], float(x), float(z)) for r, (x, z) in zip(rows, xz, strict=True)]
