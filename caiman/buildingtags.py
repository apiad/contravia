"""What OSM says about each building, kept beside its footprint for the facade grammar.

`caiman prepare` reads it with the rest, and `caiman building-tags` adds it to a data file prepared before,
leaving every other kind as it is. building_attrs is (N, 5) float32 in the order of building_ids: the
building kind (an index into KINDS), building:levels, height in metres, use bits (USE) and the municipio
(an index into building_muni_names, -1 outside them all). Levels and height are NaN where OSM says nothing.
"""

import re

import numpy as np
import shapely
from shapely.geometry import Polygon

from caiman.geo import to_local

TAGS = ("building", "building:levels", "height", "amenity", "shop", "historic")
KINDS = ("yes", "house", "residential", "apartments", "detached", "terrace", "commercial", "retail",
         "industrial", "warehouse", "church", "school", "hospital", "office", "hotel", "public", "garage",
         "shed", "roof", "ruins", "construction", "other")
ALIASES = {"cathedral": "church", "chapel": "church", "university": "school", "college": "school",
           "kindergarten": "school", "civic": "public", "government": "public", "semidetached_house": "house",
           "bungalow": "house", "hut": "shed", "garages": "garage", "supermarket": "retail", "kiosk": "retail",
           "factory": "industrial", "manufacture": "industrial", "clinic": "hospital"}
USE = {"historic": 1, "shop": 2, "amenity": 4}
KIND, LEVELS, HEIGHT, USES, MUNI = range(5)


def kept(tags) -> dict:
    """The tags of TAGS an OSM object carries, as a plain dict that outlives osmium's buffers."""
    return {k: tags.get(k) for k in TAGS if k in tags}


def number(value) -> float:
    """The first number in a tag value ("4", "12 m", "3;4"), or NaN."""
    m = re.search(r"\d+(?:\.\d+)?", value or "")
    return float(m.group()) if m else float("nan")


def kind(value) -> int:
    """KINDS index of a building tag's value; unknown values are "other"."""
    v = ALIASES.get(value or "yes", value or "yes")
    return KINDS.index(v) if v in KINDS else KINDS.index("other")


def encode(city, tags: dict, municipios) -> dict:
    """building_attrs and building_muni_names for the buildings of `city`, from {osm id: kept tags} and
    the municipios as (name, ring of (lat, lon))."""
    ids = np.asarray(city.a["building_ids"])
    attrs = np.full((len(ids), 5), np.nan, np.float32)
    attrs[:, KIND], attrs[:, USES] = 0, 0
    for k, osm_id in enumerate(ids.tolist()):
        t = tags.get(osm_id)
        if t:
            attrs[k] = (kind(t.get("building")), number(t.get("building:levels")), number(t.get("height")),
                        sum(bit for key, bit in USE.items() if key in t), np.nan)
    starts = np.asarray(city.a["building_starts"])
    verts = np.asarray(city.a["building_verts"], float)
    cents = np.add.reduceat(verts, starts[:-1], axis=0) / np.diff(starts)[:, None] if len(ids) else verts
    names = sorted({name for name, _ in municipios})
    muni = np.full(len(ids), -1, np.int16)
    for name, ring in municipios:
        poly = Polygon(to_local(np.array([p[0] for p in ring]), np.array([p[1] for p in ring])))
        if not poly.is_valid:
            poly = poly.buffer(0)
        hit = shapely.contains_xy(poly, cents[:, 0], cents[:, 1]) & (muni < 0)
        muni[hit] = names.index(name)
    attrs[:, MUNI] = muni
    return {"building_attrs": attrs, "building_muni_names": np.array(names, dtype=str)}


def main(pbf: str, data: str) -> dict:
    """Read the buildings' tags and the municipios from the extract and add them to the data file."""
    import json
    import os
    import time

    from caiman.citydata import CityData
    from caiman.prepare import read_osm

    t0 = time.perf_counter()
    raw = read_osm(pbf, keep_ways=False)
    city = CityData.load(data)
    city.a.update(encode(city, raw.building_tags, raw.municipios))
    tmp = f"{data}.tmp.npz"
    city.save(tmp)
    os.replace(tmp, data)
    a = city.a["building_attrs"]
    report = {"buildings": len(a), "tagged_kind": int((a[:, KIND] > 0).sum()),
              "levels": int(np.isfinite(a[:, LEVELS]).sum()), "height": int(np.isfinite(a[:, HEIGHT]).sum()),
              "in_a_municipio": int((a[:, MUNI] >= 0).sum()),
              "kinds": {KINDS[k]: int(n) for k, n in zip(*np.unique(a[:, KIND].astype(int), return_counts=True))},
              "seconds": round(time.perf_counter() - t0, 1)}
    print(json.dumps(report, indent=2))
    return report
