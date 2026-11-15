"""havana climate: build the sky's small tables once, from public data, so the game runs offline.

Three files land in havana/sky_data/, kept in git:

- stars.npy: the Yale Bright Star Catalogue (BSC5, public domain), stars down to magnitude 6.5.
- milkyway.npy: points sampled from NASA's Deep Star Maps 2020 Milky Way image (SVS 4851), weighted by
  brightness, with their colour.
- climate.json: a weather chain for Havana, learnt from ERA5's hourly cloud cover and rain for 1991-2020
  (Open-Meteo's archive), with each month's rain days corrected toward the Casablanca gauge (WMO 78325,
  NOAA GHCN-Daily) where the gauge has enough months.

The downloads are kept in a cache folder, so a second run reads them from disk.
"""

import csv
import io
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

SKY_DATA = Path(__file__).parent / "sky_data"
SOURCES = {
    "era5.json": "https://archive-api.open-meteo.com/v1/archive?latitude=23.13&longitude=-82.38"
                 "&start_date=1991-01-01&end_date=2020-12-31&hourly=cloud_cover,precipitation"
                 "&models=era5&timezone=America%2FHavana",
    "ghcn.csv": "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/CUM00078325.csv",
    "BSC5.bin": "http://tdc-www.harvard.edu/catalogs/BSC5",
    "milkyway_4k.jpg": "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/milkyway_2020_4k_print.jpg",
}
STATES = ("clear", "partly", "overcast", "rain", "storm")
MIN_GAUGE_MONTHS = 4  # a month's gauge record corrects ERA5 only with this many complete months
MIN_ROW = 30  # transitions a (month, hour, state) row needs before it stops borrowing from nearby hours

# B-V colour index by the first letter of the spectral type; the binary BSC5 carries no photometric colour.
_BV = {"O": -0.30, "B": -0.15, "A": 0.10, "F": 0.40, "G": 0.70, "K": 1.15, "M": 1.55}


def fetch(cache: Path) -> dict[str, Path]:
    """Download what the cache lacks; return the path of each source."""
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in SOURCES.items():
        path = cache / name
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "havana-driver-climate/0.1"})
            with urllib.request.urlopen(req, timeout=600) as r:
                path.write_bytes(r.read())
        out[name] = path
    return out


STAR_DTYPE = np.dtype([("ra", "f4"), ("dec", "f4"), ("pm_ra", "f4"), ("pm_dec", "f4"), ("mag", "f4"),
                       ("bv", "f4")])


def read_bsc5(data: bytes, faintest: float = 6.5) -> np.ndarray:
    """Stars of the binary BSC5 down to `faintest`: J2000 RA and Dec and their proper motions per year in
    degrees, V magnitude, and a B-V colour from the spectral type."""
    _, _, starn, _, _, nmag, nbent = struct.unpack("<7i", data[:28])
    rec = np.dtype([("xno", "<f4"), ("sra", "<f8"), ("sdec", "<f8"), ("sp", "S2"), ("mag", "<i2"),
                    ("xrpm", "<f4"), ("xdpm", "<f4")])
    assert nbent == rec.itemsize and nmag == 1, "not the BSC5 binary format"
    s = np.frombuffer(data, rec, count=abs(starn), offset=28)
    s = s[(s["mag"] / 100.0 <= faintest) & ((s["sra"] != 0) | (s["sdec"] != 0))]
    out = np.zeros(len(s), STAR_DTYPE)
    out["ra"], out["dec"] = np.degrees(s["sra"]), np.degrees(s["sdec"])
    out["pm_ra"], out["pm_dec"] = np.degrees(s["xrpm"]), np.degrees(s["xdpm"])
    out["mag"] = s["mag"] / 100.0
    out["bv"] = [_BV.get(sp[:1].decode("ascii", "replace").upper(), 0.6) for sp in s["sp"]]
    return out


MW_DTYPE = np.dtype([("ra", "f4"), ("dec", "f4"), ("r", "f4"), ("g", "f4"), ("b", "f4"), ("w", "f4")])


def sample_milky_way(rgb: np.ndarray, n: int = 25_000, seed: int = 0) -> np.ndarray:
    """Points drawn from an equirectangular celestial image, as often as each pixel is brighter than the
    image's median. RA 0 is at the centre and grows to the left, as in NASA's Deep Star Maps.

    Each point keeps its pixel's colour (0-1) and a weight `w`, its brightness over the median.
    """
    rgb = np.asarray(rgb, np.float32) / 255.0
    h, w = rgb.shape[:2]
    lum = rgb.mean(axis=2)
    excess = np.clip(lum - np.median(lum), 0.0, None)
    p = (excess / excess.sum()).ravel()
    rng = np.random.default_rng(seed)
    idx = rng.choice(p.size, size=n, p=p)
    row, col = np.divmod(idx, w)
    u, v = (col + rng.random(n)) / w, (row + rng.random(n)) / h
    out = np.zeros(n, MW_DTYPE)
    out["ra"] = (180.0 - u * 360.0) % 360.0
    out["dec"] = 90.0 - v * 180.0
    out["r"], out["g"], out["b"] = rgb[row, col, 0], rgb[row, col, 1], rgb[row, col, 2]
    out["w"] = excess.ravel()[idx] / excess.max()
    return out


def gauge_rain_days(text: str) -> dict[int, float]:
    """Mean days with 1 mm or more per month at the gauge, over months with at least 26 days of data;
    only months with MIN_GAUGE_MONTHS such months are returned."""
    by_month: dict[str, list[float]] = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r["PRCP"].strip():
            by_month.setdefault(r["DATE"][:7], []).append(float(r["PRCP"]) / 10.0)
    days: dict[int, list[int]] = {}
    for ym, v in by_month.items():
        if len(v) >= 26:
            days.setdefault(int(ym[5:]), []).append(sum(x >= 1.0 for x in v))
    return {m: float(np.mean(v)) for m, v in days.items() if len(v) >= MIN_GAUGE_MONTHS}


def build_table(time: np.ndarray, cloud: np.ndarray, rain: np.ndarray, gauge: dict[int, float]) -> dict:
    """The weather chain's table from an hourly series (local times as datetime64[h], cloud %, rain mm).

    A rain hour has at least `threshold[month]` mm. ERA5 spreads rain thin, in many hours of drizzle, so the
    threshold rises until the days with a rain hour match the gauge's rain days where the gauge has a record,
    and ERA5's own days with 1 mm or more elsewhere. The storm
    state is the wettest tenth of a month's rain hours. The rest split by cloud: under 20 %, 20-60 %, over.
    """
    time = np.asarray(time, "datetime64[h]")
    month = time.astype("datetime64[M]").astype(int) % 12
    hour = (time - time.astype("datetime64[D]")).astype(int)
    day = time.astype("datetime64[D]")
    years = len(np.unique(time.astype("datetime64[Y]")))
    days, inv = np.unique(day, return_inverse=True)
    dmonth = days.astype("datetime64[M]").astype(int) % 12

    daily = np.bincount(inv, rain, len(days))
    threshold = np.full(12, 0.1)
    for m in range(12):
        target = gauge[m + 1] * years if m + 1 in gauge else ((daily >= 1.0) & (dmonth == m)).sum()
        for th in np.arange(0.1, 10.0, 0.05):  # an hour of th mm or more makes a rain day
            wet = np.bincount(inv, (rain >= th) & (month == m), len(days)) > 0
            if (wet & (dmonth == m)).sum() <= target:
                threshold[m] = th
                break

    state = np.where(cloud < 20, 0, np.where(cloud <= 60, 1, 2))
    is_rain = rain >= threshold[month]
    storm_mm = np.array([np.percentile(rain[is_rain & (month == m)], 90) if (is_rain & (month == m)).any()
                         else np.inf for m in range(12)])
    state = np.where(is_rain, np.where(rain >= storm_mm[month], 4, 3), state)

    counts = np.zeros((12, 24, 5, 5), np.int64)
    np.add.at(counts, (month[:-1], hour[:-1], state[:-1], state[1:]), 1)
    trans = counts.astype(float)
    for m in range(12):
        for h in range(24):
            for s in range(5):
                k = 1
                while trans[m, h, s].sum() < MIN_ROW and k < 12:
                    trans[m, h, s] += counts[m, (h - k) % 24, s] + counts[m, (h + k) % 24, s]
                    k += 1
                if trans[m, h, s].sum() == 0:
                    trans[m, h, s, s] = 1.0
    trans /= trans.sum(axis=3, keepdims=True)

    cloud_mean = np.zeros((12, 5))
    rain_mean = np.zeros((12, 5))
    start = np.zeros((12, 5))
    for m in range(12):
        for s in range(5):
            sel = (month == m) & (state == s)
            cloud_mean[m, s] = cloud[sel].mean() / 100.0 if sel.any() else [0.1, 0.4, 0.8, 0.9, 1.0][s]
            rain_mean[m, s] = min(1.0, rain[sel].mean() / max(storm_mm[m], 1e-6)) if s >= 3 and sel.any() else 0.0
        start[m] = np.bincount(state[(month == m) & (hour == 0)], minlength=5) / max(1, ((month == m) & (hour == 0)).sum())
    rain_days = [float(((np.bincount(inv, state >= 3, len(days)) > 0) & (dmonth == m)).sum() / years) for m in range(12)]
    peak = [int(np.argmax([(state[(month == m) & (hour == h)] >= 3).mean() for h in range(24)])) for m in range(12)]
    return {
        "states": list(STATES),
        "source": "ERA5 1991-2020 at 23.0 N 82.5 W (Open-Meteo archive); rain days toward GHCN-Daily CUM00078325",
        "threshold_mm": threshold.round(2).tolist(),
        "transitions": trans.round(4).tolist(),
        "start": start.round(4).tolist(),
        "cloud": cloud_mean.round(3).tolist(),
        "rain": rain_mean.round(3).tolist(),
        "check": {"rain_days": [round(v, 2) for v in rain_days],
                  "cloud": [round(float(cloud[month == m].mean()) / 100, 3) for m in range(12)],
                  "peak_hour": peak},
    }


def main(cache: str, out: str | None = None) -> None:
    """Fetch the sources into `cache` and write the three tables into `out` (havana/sky_data by default)."""
    from PIL import Image

    dest = Path(out) if out else SKY_DATA
    dest.mkdir(parents=True, exist_ok=True)
    src = fetch(Path(cache))
    stars = read_bsc5(src["BSC5.bin"].read_bytes())
    np.save(dest / "stars.npy", stars)
    mw = sample_milky_way(np.asarray(Image.open(src["milkyway_4k.jpg"]).convert("RGB")))
    np.save(dest / "milkyway.npy", mw)
    h = json.loads(src["era5.json"].read_text())["hourly"]
    table = build_table(np.array(h["time"], "datetime64[h]"), np.array(h["cloud_cover"], float),
                        np.array(h["precipitation"], float), gauge_rain_days(src["ghcn.csv"].read_text()))
    (dest / "climate.json").write_text(json.dumps(table, separators=(",", ":")))
    print(json.dumps({"stars": len(stars), "milky_way_points": len(mw), "rain_threshold_mm": table["threshold_mm"],
                      "rain_days": table["check"]["rain_days"], "peak_hour": table["check"]["peak_hour"]}))
