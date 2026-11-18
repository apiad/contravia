"""The compact source the game ships instead of the built island: per province, what read_osm keeps of the Cuba
extract and the cleaned terrain, in one zstd file (<slug>.src). The first run builds the city from it (contravia.firstrun).

The terrain is kept in decimetres as int16 (Pico Turquino, 1974 m, stays well under 3276 m), so the built city's heights are
those of the measured build to within 5 cm."""

import pickle
from compression import zstd
from pathlib import Path

import numpy as np

from contravia.prepare import Raw, read_osm
from contravia.terrain import Terrain

VERSION = 3  # 2: road names, places and consejos populares; 3: land never made a park
LEVEL = 19  # zstd level: packing is done once, unpacking stays fast at any level


def pack(pbf, dem, out, provinces=None) -> dict:
    """Write out/<slug>.src for each province (all by default): its part of the extract and its cleaned relief
    from the GLO-30 tiles in dem (downloaded there if missing). Returns {name: sizes in bytes}."""
    from contravia.geo import PROVINCES
    from contravia.prepare import TERRAIN_STRIDE, province_boxes
    from contravia.terrain import read_terrain

    boxes, sizes = province_boxes(pbf), {}
    for name, slug in PROVINCES:
        if provinces and name not in provinces and slug not in provinces:
            continue
        terrain = read_terrain(dem, boxes[name], stride=TERRAIN_STRIDE.get(name, 3))
        sizes[name] = write(read_osm(pbf, boundary=name, box=boxes[name]), terrain, Path(out) / f"{slug}.src")
        print(f"{name}: {sizes[name]['bytes'] / 1e6:.1f} MB", flush=True)
    return sizes


def write(raw: Raw, terrain: Terrain, out) -> dict:
    """Write raw and terrain as a source to out. Returns its sizes in bytes."""
    heights = np.round(terrain.heights * 10).astype(np.int16)
    body = {"version": VERSION, "raw": raw, "heights_dm": heights, "origin": terrain.origin, "step": terrain.step}
    data = zstd.compress(pickle.dumps(body, protocol=pickle.HIGHEST_PROTOCOL), level=LEVEL)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    return {"bytes": len(data), "terrain_bytes": len(zstd.compress(heights.tobytes(), level=LEVEL))}


def load(path) -> tuple[Raw, Terrain]:
    """The extract and the terrain of a source written by pack."""
    body = pickle.loads(zstd.decompress(Path(path).read_bytes()))
    if body.get("version") != VERSION:
        raise ValueError(f"{path}: source version {body.get('version')}, this game reads {VERSION}")
    terrain = Terrain(body["heights_dm"].astype(np.float32) / 10, body["origin"], body["step"])
    return body["raw"], terrain


def main(pbf, dem, out, provinces=None) -> dict:
    """Pack each province's source and print the island's size."""
    sizes = pack(pbf, dem, out, provinces)
    print(f"{out}: {sum(v['bytes'] for v in sizes.values()) / 1e6:.1f} MB in {len(sizes)} provinces")
    return sizes
