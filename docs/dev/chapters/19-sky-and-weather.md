# Sky and weather {#sky-and-weather}

The light over Havana at five in the afternoon in September is low and orange, and often by six the clouds have built up and it is raining. In February at the same hour the sky is clear and the sun sets before seven. A driving game that follows the clock has to get both right, at any date, without a network connection and without an ephemeris file.

The game keeps two things apart. `contravia/sky.py` says where the sun, the moon, the planets and the stars are, and it is a pure function of time and of Havana's coordinates. `contravia/weather.py` says what the sky is doing, from a small table learnt from thirty years of Havana weather and seeded by the date. `contravia/atmosphere.py` turns both into the frame's light, sky colour, fog, shadows and rain. Everything is computed, nothing is downloaded while the game runs.

## The clock

The game's clock gives a naive local time, and the sky reads that time as Havana's: a naive `datetime` is read in the `America/Havana` zone, summer time included, and an aware one is converted to UTC first. The sky is computed in UTC and then seen from Havana, latitude `23.136°` north and longitude `82.359°` west, so the clock can change without moving a star.

`--time "2026-07-18 23:00"` pins the clock to a full date; `--time "Sun 08:00"` takes that weekday of the current week and `--time "17:30"` today at that hour. `--warp 60` runs it sixty times faster, for tests and the timelapse. Traffic, weather and the sky read the same clock, which is what keeps the rush hour, the light and the rain of one moment together.

## The sun, the moon and the planets

`sky.sun` follows NOAA's solar position equations: the mean longitude, the equation of the centre and the obliquity, from the Julian day. It returns the sun's altitude above the horizon, its bearing from the north toward the east, its equatorial place and its magnitude. `sky.moon` uses the main terms of Meeus's lunar series, thirteen for longitude, six for latitude and four for distance, then subtracts the parallax, because the observer stands on the surface and not at the centre of the Earth. It also returns the lit fraction, from the phase angle.

The five planets are JPL's approximate Keplerian elements, valid from 1800 to 2050. Each planet's heliocentric position comes from its orbital elements and Kepler's equation, solved by Newton's method in eight steps; the geocentric vector gives the place and the phase, and the magnitude follows from the distances and the phase with the Astronomical Almanac's coefficients. A planet is a few arcminutes from a full ephemeris, which is far below what a player can see.

The game's frame has x east, z south and y up, so north is −z. `sky.direction` turns an altitude and a bearing into a unit vector in that frame, and `equatorial_to_local` builds the 3×3 rotation that turns the whole equatorial sky to it at one moment. That one rotation is what the night sky uses.

```python {run echo=false}
import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

from contravia.skymap import COLOURS, year_states
from contravia.weather import STATES

states = year_states(2026)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.imshow(states, aspect="auto", cmap=ListedColormap(COLOURS / 255), interpolation="nearest")
ax.set(xlabel="hour", ylabel="day of 2026")
ax.set_xticks(range(0, 24, 4))
months = [date(2026, m, 1).timetuple().tm_yday for m in range(1, 13)]
ax.set_yticks(months, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOURS[i] / 255) for i in range(5)],
          labels=STATES, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False, fontsize=9)
print("days with rain:", int((states >= 3).any(axis=1).sum()))
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## The stars

The night sky is the Yale Bright Star Catalogue, public domain, packed by `contravia climate` into `contravia/sky_data/stars.npy`: `8 404` stars down to magnitude `6.5`, each with right ascension, declination, proper motion and a colour index from its spectral type. Positions are moved from J2000 to the game's year by precession and by their proper motion, about a third of a degree by 2026, so the pole star stands where it does today. The Milky Way is `80 000` points sampled from NASA's Deep Star Maps 2020, weighted by brightness.

The stars are camera-facing sprites on a dome centred on the camera: `13 200` m for the stars, `13 500` m for the Milky Way behind them and `13 000` m for the planets and the moon in front. The dome sits inside the `15 000` m far plane of the default preset; Bajo and Medio pull the far plane in to `7 000` and `11 000` m. Buildings hide the stars behind them by depth. The catalogue's places are moved once a day; the whole sky turns with one 3×3 rotation from the local sidereal time, recomputed at most every `10` s of game time or `2` m of travel. The planets and the moon, seven sprites, follow the camera every frame and their places are recomputed every `10` s. The catalogue's entities exist only while stars show, because the engine pays for every entity on every frame, seen or not.

What shows is decided by a limiting magnitude. It starts at `6.5` minus twice the city glow, which is `4.5` in the default city and `6.5` in a blackout, then loses `1.5` for a lit moon above the horizon, up to `6` through twilight and up to `6` under cloud. Stars fade in from the brightest as the sun goes down, and a star's size, colour and brightness come from its magnitude and its colour index.

```python {run echo=false}
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from contravia import sky
from contravia.atmosphere import look
from contravia.climate import SKY_DATA
from contravia.weather import SkyConfig, Weather

when = datetime(2026, 11, 15, 23, 0)
stars = np.load(SKY_DATA / "stars.npy")
years = when.year + when.timetuple().tm_yday / 365.25 - 2000.0
ra, dec = sky.precess(stars["ra"] + stars["pm_ra"] * years, stars["dec"] + stars["pm_dec"] * years, when)
v = sky.unit_vectors(ra, dec) @ sky.equatorial_to_local(when).T
alt = np.degrees(np.arcsin(np.clip(v[:, 1], -1.0, 1.0)))
az = np.degrees(np.arctan2(v[:, 0], -v[:, 2])) % 360.0
cfg = SkyConfig(force="clear")  # a clear night, so the chart shows the catalogue
sun, moon, now = sky.sun(when), sky.moon(when), Weather(config=cfg).at(when)
limit = look(sun, moon, now, cfg).star_limit
up = (alt > 0) & (stars["mag"] <= limit)

fig, ax = plt.subplots(figsize=(9, 4.4))
ax.scatter(az[up], alt[up], s=np.clip(18 - 3.2 * stars["mag"][up], 0.4, 22), c="#f6f2e8", lw=0)
for name, body in sky.planets(when).items():
    if body.alt > 0:
        ax.plot(body.az, body.alt, "o", ms=5, color="#ffd400")
        ax.annotate(name, (body.az, body.alt), xytext=(4, 3), textcoords="offset points", color="#ffd400", fontsize=8)
ax.set(xlim=(0, 360), ylim=(0, 90), xlabel="bearing", ylabel="height over the horizon, degrees")
ax.set_xticks(range(0, 361, 45), ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
ax.set_facecolor("#0b0b0b")
ax.set_title(f"{when:%Y-%m-%d %H:%M}, a clear night: {up.sum()} stars down to magnitude {limit:.1f}", fontsize=10)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

`look(...).star_limit` is the rule the atmosphere uses; the block above calls it directly so the chart shows exactly the stars the game would. The moon is left out because it is below the horizon at that hour.

## The light

`atmosphere.look` is a pure function of the sun, the moon, the weather and the settings, so the look of a frame is tested without a renderer. The day factor runs from complete dark at `−6°` of sun to full day at `8°`; the night factor is the opposite, from `1` below `−18°` to `0` above `−6°`.

Direct light is the sun while it is above `−2°`, and its intensity is `3.6` times a fade from `−2°` to `8°`, a factor that grows to `1` at `45°` of height, and the clear-sky factor of Kasten and Czeplak, `1 − 0.75 · cloud^3.4`. Its colour moves from orange at the horizon to white by `25°` of height. Below `−2°`, if the moon is up, the light becomes the moon: pale blue at `0.25` of its lit fraction and its height. Below both, the direct light goes out.

The sky's colour mixes a day palette with a night one, adds a warm horizon near sunrise and sunset when the sky is clear, and mixes it all toward grey with cloud and rain. The ambient light is one of five precomputed environments, day, overcast, golden, twilight and night, chosen by the sun's height and the cloud, and its intensity follows the light. Fog goes from `2 000` m to `12 000` m in clear weather and from `90` m to `220` m in a storm, and its colour is the horizon's.

Shadows are drawn when the sun is above `3°`, the cloud is under `0.85` and the rain is under `0.3`. The shadow map is an orthographic box around the player, half a side of the quality preset's reach, from `80` m at Bajo to `300` m at Cine (`200` m at the default, Alto), at `2 048` texels by default, with the centre snapped to whole texels in the light's frame so still shadows stay still as the player moves. Ultra and Cine use four cascades fitted to the view instead. There are no moon shadows.

## The weather chain

The weather has five states: clear, partly cloudy, overcast, rain and storm. A table holds one `5 × 5` matrix of hour-to-hour transitions for each of the twelve months and each of the twenty-four hours, `12 × 24 × 25` numbers, learnt by counting transitions in ERA5's hourly cloud cover and rain for 1991 to 2020 at the grid point nearest Havana. A row of one month, hour and state with fewer than `30` counts borrows from the neighbouring hours. Rain is an hour with at least the month's threshold in millimetres; that threshold rises until the number of rain days matches the Casablanca gauge where it has at least four complete months, and ERA5's own days elsewhere. A storm is the wettest tenth of a month's rain hours. The other states split by cloud: under `20 %`, `20 %` to `60 %`, and over `60 %`.

A day's weather is drawn once, from the day's own random numbers: the generator is seeded with the date's ordinal and the sky setting's seed, so the same date always has the same weather. The chain starts `3` days earlier from the month's distribution at midnight and walks forward, one hour at a time, using each day's own numbers by inverse sampling. That is why two runs that start on different days agree on the days they share: the weather has no seam at midnight. Sixteen days are cached.

Within an hour the cloud and the rain slide from the previous hour's state to this one over the first `10` minutes, so a storm arrives instead of switching on. Road wetness is `1` while it rains and dries over `1` hour after the last rainy hour. `weather.at` returns the state's name, the cloud cover, the rain intensity and the wetness, and the HUD, the light, the rain and the sound all read that one answer.

`SkyConfig` is what a menu or an agent may change: the seed, a rain factor and a cloud factor per month, a forced state with an end time, a pinned star limit and the city glow. `rain_factor` and `cloud_factor` bend each month's chain toward rain or cloud, with `1` being Havana's climate. `manifoldx emit weather state=storm minutes=30` forces a state, and `manifoldx emit sky_config city_glow=0` changes the sky, both while the game runs.

## A day of sky

The strip below is one column every five minutes for the day `contravia timelapse` picks from the 2026 rainy season: a date scored for a storm after noon, a clear sunset and night, and a moon that sets early. Each column's colours are the zenith and the horizon that `atmosphere.look` gives, so the strip is the sky the game would draw, without the city in front of it. The yellow line is the sun's height, from `−90°` to `90°`.

```python {run echo=false}
import io
from datetime import datetime, time, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from contravia import sky
from contravia.atmosphere import look
from contravia.timelapse import pick_day
from contravia.weather import SkyConfig, Weather

day, seed = pick_day(2026)
cfg = SkyConfig(seed=seed)
weather = Weather(config=cfg)
start = datetime.combine(day, time(4, 30))
sun_alt, img = [], np.zeros((40, 24 * 12, 3))
for k in range(img.shape[1]):
    when = start + timedelta(minutes=5 * k)
    sun, moon, now = sky.sun(when), sky.moon(when), weather.at(when)
    lk = look(sun, moon, now, cfg)
    ramp = np.linspace(0, 1, img.shape[0])[:, None]
    img[:, k] = np.array(lk.horizon) + (np.array(lk.zenith) - np.array(lk.horizon)) * ramp
    sun_alt.append(sun.alt)

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.imshow(np.clip(img, 0, 1) ** (1 / 2.2), aspect="auto", extent=(0, 24, 0, 1), origin="lower")
ax2 = ax.twinx()
ax2.plot(np.arange(img.shape[1]) * 5 / 60, sun_alt, color="#ffd400", lw=1.2)
ax2.set_ylim(-90, 90)
ax.set(xlabel="hours from 04:30")
ax.set_yticks([])
ax2.set_ylabel("sun height, °")
ax.set_title(f"{day}: the sky's colours, one column every five minutes", fontsize=10)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## Rain and the road

Rain is a fixed pool of drops in a `60` m box that follows the camera: `3 000` drops, or the number the quality preset sets, falling at `9` m/s and drifting with the wind. A drop that leaves the box comes back on the other side, so the box always holds the same rain, and the rain's intensity decides how many of the pool show. The streaks are sprites stretched along the fall, unlit, in the colour of the light. Their number follows the intensity, and in a storm the fog closes to `220` m.

Rain does not change how the car drives. `car.SurfaceGround.grip` answers one grip for the whole ground, the same on a wet Malecón as on a dry one, and the tyre model never reads the weather. What the wetness does change is the sound: the tyre squeal is duller and the spray louder on a wet road, and the rain has its own hiss and drops. The spec proposed a wet road material; it is not in the code today, and the rain does not slow the traffic either.

## The commands

`contravia climate` builds the three files in `contravia/sky_data/`. It needs the network once, downloads the BSC5 catalogue, the NASA Milky Way image, the ERA5 series and the gauge record into `--cache` (by default `~/.cache/havana`), and writes `stars.npy`, `milkyway.npy` and `climate.json`. It prints how many stars and points it wrote, the rain thresholds and the rain days. The game never fetches anything.

```bash
contravia climate
contravia skymap weather-2026.png --year 2026
contravia timelapse capitolio.mp4 --day 2026-07-18 --start 04:30 --minutes 1440 --step 1
```

<!-- shot: contravia timelapse capitolio.mp4 | A day over the Capitolio from the street: sunrise, the afternoon storm, the sunset and the stars in one minute a frame -->

`contravia skymap` draws a year of the chain as a PNG, days down and hours across, which is how two sets of sky settings compare, the same way `contravia flowmap` compares two traffic settings. `contravia timelapse` renders a day over the Capitolio, one game minute per frame, with the clock pinned and the car parked out of view. `--sky my.json` on either takes a `SkyConfig` file.

## The numbers

| Constant | Value | What it decides |
|---|---|---|
| `sky.LAT`, `sky.LON` | 23.136° N, 82.359° W | Where the sky is seen from |
| `weather.LEAD_DAYS` | 3 | Days the chain runs before the day it draws |
| `weather.BLEND_MIN` | 10 min | How long the weather takes to cross an hour |
| `weather.DRY_HOURS` | 1 h | How long a road stays wet after the rain |
| `weather.STATES` | 5 | clear, partly, overcast, rain, storm |
| transition table | 12 × 24 × 5 × 5 | One matrix per month and hour |
| `atmosphere.FOG_CLEAR` | 2 000–12 000 m | Clear-weather fog |
| `atmosphere.FOG_STORM` | 90–220 m | Fog in a storm |
| `atmosphere.SHADOW_EXTENT` | 150 m | Half the shadow box's side, before the quality preset |
| `atmosphere.SHADOW_RES` | 2 048 | Texels across the shadow map |
| `atmosphere.clear_sky_factor` | 1 − 0.75·c^3.4 | The share of sunlight left under cloud |
| star limit | 6.5 − 2·city_glow, then less | The faintest star shown |
| `rain.POOL` | 3 000 | Drops at the heaviest rain, before the quality preset |
| `rain.BOX` | 60 m | The side of the box of drops round the camera |
| `rain.FALL`, `rain.STREAK` | 9 m/s, 0.9 m | How fast a drop falls and how long its streak is |
| `nightsky.DOME` | 13 200 m | The stars' distance from the camera |
| `nightsky.MOON_DEG` | 0.26° | The moon's angular radius |

The modules are `contravia/sky.py` for the bodies, `contravia/weather.py` for the chain, `contravia/atmosphere.py` for the light and the shadow box, `contravia/nightsky.py` for the stars, `contravia/rain.py` for the drops, `contravia/climate.py` for the tables, `contravia/skymap.py` and `contravia/timelapse.py` for the two pictures. The clock they follow is `contravia/clock.py`, the same one the [traffic](#traffic) reads.
