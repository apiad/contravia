"""havana flowmap: a picture of the flow layer over the province at a given time, without opening the game.

It warms the flow layer up to that time with the given traffic settings and draws every street coloured by
how many vehicles pass along it per hour, from dark (none) to red (600 or more). Two settings files compare
as two pictures.
"""

from datetime import datetime

import numpy as np

from havana.graph import CLASSES

COLOURS = np.array([[61, 74, 51], [157, 176, 126], [224, 168, 114], [217, 128, 111]], float)
FULL = 600.0  # vehicles per hour drawn in the last colour
WIDE = {"motorway": 3, "trunk": 3, "primary": 3, "secondary": 2, "tertiary": 2}


def colour(fraction: np.ndarray) -> np.ndarray:
    """RGB for each fraction 0 to 1 along the four colours, on a square-root scale so quiet streets show."""
    t = np.sqrt(np.clip(fraction, 0, 1)) * 0.999 * (len(COLOURS) - 1)
    i = t.astype(int)
    f = (t - i)[:, None]
    return (COLOURS[i] + (COLOURS[i + 1] - COLOURS[i]) * f).astype(int)


def per_hour(view, length: np.ndarray) -> np.ndarray:
    """Vehicles passing along each edge per hour: density times speed."""
    return view.n * view.speed / np.maximum(length, 1e-9) * 3600


def render(city, when: datetime, config, out: str, size: int = 2000, box=None) -> dict:
    """Warm the flow up to `when` with `config`, draw it into the PNG `out`, and return the totals."""
    from PIL import Image, ImageDraw

    from havana.clock import FlowRunner
    from havana.flow import Flow
    from havana.rules.traffic import moving

    class Fixed:
        def now(self):
            return when

    g = city.graph
    a = g.a
    runner = FlowRunner(Flow(g, config), Fixed())
    runner.warm_up(minutes=60)
    view = runner.view
    flow = per_hour(view, a["g_len"].astype(float))
    if box is None:
        lo, hi = np.percentile(a["g_mid"], 1, axis=0), np.percentile(a["g_mid"], 99, axis=0)
        box = (lo[0], lo[1], hi[0], hi[1])
    x0, z0, x1, z1 = box
    W, H = size, max(1, int(size * (z1 - z0) / (x1 - x0)))
    img = Image.new("RGB", (W, H), (17, 16, 14))
    draw = ImageDraw.Draw(img)
    rgb = colour(flow / FULL)
    poly, starts = a["g_poly"], a["g_poly_start"]
    for e in np.argsort(flow):  # busiest last, on top
        p = poly[starts[e]:starts[e + 1]]
        if p[:, 0].max() < x0 or p[:, 0].min() > x1 or p[:, 1].max() < z0 or p[:, 1].min() > z1:
            continue
        pts = [((x - x0) / (x1 - x0) * W, (z - z0) / (z1 - z0) * H) for x, z in p]
        draw.line(pts, fill=tuple(int(c) for c in rgb[e]), width=WIDE.get(CLASSES[a["g_class"][e]], 1))
    totals = {"time": when.strftime("%a %H:%M"), "vehicles": round(float(view.n.sum())),
              "moving_by_settings": round(sum(moving(config, when).values())),
              "busiest_per_hour": round(float(flow.max())) if len(flow) else 0}
    draw.text((12, 10), f"{totals['time']}   vehicles in the flow {totals['vehicles']:,}   "
                        f"colour: vehicles per hour, dark 0 to red {FULL:.0f}+", fill=(241, 237, 226))
    img.save(out)
    return totals


def main(data: str, out: str, when: str, traffic: str | None = None, size: int = 2000, box=None) -> dict:
    import json

    from havana.citydata import CityData
    from havana.clock import parse_at
    from havana.rules.traffic import TrafficConfig, load_config

    city = CityData.load(data)
    if not len(city.graph):
        raise SystemExit("no road graph in the data file: run havana prepare again")
    config = load_config(traffic) if traffic else TrafficConfig()
    totals = render(city, parse_at(when), config, out, size, box)
    print(json.dumps(totals))
    return totals
