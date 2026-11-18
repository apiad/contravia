"""The compact source the game ships instead of the built city: what read_osm keeps of the Cuba extract and the
cleaned terrain, in one zstd file. The first run builds the city from it (contravia.firstrun).

The terrain is kept in decimetres as int16 (Havana stays well under 3276 m), so the built city's heights are
those of the measured build to within 5 cm."""

import pickle
from compression import zstd
from pathlib import Path

import numpy as np

from contravia.prepare import HAVANA_BOX, Raw, read_osm
from contravia.terrain import Terrain, clean, read_tiles

VERSION = 2  # 2: road names, places and consejos populares
LEVEL = 19  # zstd level: packing is done once, unpacking stays fast at any level


def pack(pbf, dem, out) -> dict:
    """Read the extract and the GLO-30 tiles in dem and write the source to out. Returns its sizes in bytes."""
    tiles = read_tiles(dem, HAVANA_BOX)
    return write(read_osm(pbf), Terrain(clean(tiles.heights), tiles.origin, tiles.step), out)


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


def main(pbf, dem, out) -> dict:
    """Pack the source and print its size."""
    sizes = pack(pbf, dem, out)
    print(f"{out}: {sizes['bytes'] / 1e6:.1f} MB (terrain {sizes['terrain_bytes'] / 1e6:.1f} MB)")
    return sizes
