"""caiman skymap: a year of the weather chain as one picture, to compare two sets of sky settings.

Each row is a day of the year and each column an hour; the colour is the weather state. Light blue is clear,
pale grey partly cloudy, grey overcast, blue rain and dark violet storm, so the rainy season shows as a band
of afternoon storms from May to October.
"""

from datetime import date, timedelta

import numpy as np

from caiman.weather import SkyConfig, Weather

COLOURS = np.array([(150, 200, 240), (205, 210, 215), (140, 145, 150), (60, 100, 200), (70, 30, 110)], np.uint8)


def year_states(year: int, config: SkyConfig | None = None) -> np.ndarray:
    """The (days, 24) states of every hour of `year`."""
    w = Weather(config=config)
    first, days = date(year, 1, 1), (date(year + 1, 1, 1) - date(year, 1, 1)).days
    return np.array([w.day(first + timedelta(days=k)) for k in range(days)])


def main(out: str, year: int = 2026, sky: str | None = None, scale: int = 4) -> None:
    from PIL import Image

    from caiman.weather import load_sky_config

    states = year_states(year, load_sky_config(sky) if sky else None)
    img = COLOURS[states].repeat(scale, axis=0).repeat(scale * 4, axis=1)
    Image.fromarray(img).save(out)
    rain = (states >= 3).any(axis=1)
    print({"days": len(states), "rain_days": int(rain.sum()), "storm_hours": int((states == 4).sum())})
