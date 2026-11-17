"""contravia timelapse: a day of the sky over the Capitolio, one game minute per frame, as a video.

The camera stands on a street east of the Capitolio and looks at its dome, so the sun rises behind the
camera, crosses over and sets behind the dome, and at night the western and southern sky turns above it.
The game's clock is pinned and moved by hand each frame; the player's car is parked out of view. Traffic
keeps running: the flow layer is stepped to each frame's time in coarse steps.

`pick_day` looks through a season for a date whose weather and moon make a good demonstration: an afternoon
storm, a clear sunset and night, and a moon that sets early enough to leave a dark sky for the Milky Way.
"""

import os
import subprocess
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from contravia import sky
from contravia.weather import SkyConfig, Weather

AIM = (0.0, 70.0, 0.0)  # a little above the Capitolio's dome, so the frame holds more sky
STAND = (150.0, 0.0)  # on the Prado, about 140 m east of the dome


def score_day(w: Weather, d: date) -> float:
    """How well date d shows everything: storm after noon, clear evening and night, an early-setting moon."""
    today, tomorrow = w.day(d), w.day(d + timedelta(days=1))
    storm = (today[13:18] >= 3).any()
    clear_sunset = (today[19:21] <= 1).all()
    clear_night = (np.r_[today[21:], tomorrow[:4]] <= 1).mean()
    clear_morning = (today[5:9] <= 1).mean()
    m_evening = sky.moon(datetime(d.year, d.month, d.day, 20, 30))
    m_late = sky.moon(datetime(d.year, d.month, d.day, 23, 59))
    moon = m_evening.alt > 10 and m_late.alt < 0 and 0.1 < m_evening.lit < 0.5
    return float(storm) * 2 + float(clear_sunset) * 2 + clear_night * 2 + clear_morning + float(moon) * 3


def pick_day(year: int = 2026, months=(5, 6, 7, 8), config: SkyConfig | None = None,
             seeds=range(40)) -> tuple[date, int]:
    """The best (date, weather seed) of the rainy season months by score_day; the seed is a sky setting, so
    a demonstration can look through other plausible weathers for the same days."""
    config = config or SkyConfig()
    days = [date(year, m, 1) + timedelta(days=k) for m in months for k in range(31) if (
        date(year, m, 1) + timedelta(days=k)).month == m]
    best, best_score = (days[0], config.seed), -1.0
    for seed in seeds:
        w = Weather(config=config.updated(seed=seed))
        for d in days:
            sc = score_day(w, d)
            if sc > best_score:
                best, best_score = (d, seed), sc
    return best


class _Pinned:
    """A clock the timelapse moves by hand."""

    def __init__(self, t: datetime):
        self.t = t

    def now(self) -> datetime:
        return self.t


def main(data: str, out: str, day: str | None = None, start: str = "04:30", minutes: int = 1440,
         step: float = 1.0, size=(1280, 720), sky_file: str | None = None, fps: int = 30) -> None:
    """Render `minutes` of game time from `day` at `start`, one frame per `step` minutes, into the video `out`."""
    import manifoldx as mx
    from PIL import Image

    from contravia.car import Controls
    from contravia.citydata import CityData
    from contravia.game import Game
    from contravia.weather import load_sky_config

    config = (load_sky_config(sky_file) if sky_file else SkyConfig()).updated(star_limit=6.5)
    if day:
        d = date.fromisoformat(day)
    else:
        d, seed = pick_day(config=config)
        config = config.updated(seed=seed)
    t0 = datetime.fromisoformat(f"{d.isoformat()} {start}")
    print({"day": d.isoformat(), "seed": config.seed, "start": t0.isoformat(), "frames": int(minutes / step)},
          flush=True)
    os.environ["HAVANA_DATA"] = os.path.abspath(data)
    city = CityData.load(data)
    clock = _Pinned(t0)
    engine = mx.Engine("contravia-timelapse", width=size[0], height=size[1])
    game = Game(engine, city, at=(-300.0, 300.0), clock=clock, sky=config)  # the car parks out of view
    if game.layer:
        game.runner.stop()
    game.autopilot = lambda car: Controls(0, 1, 0, True)
    x, z, _ = game.walls.nearest_road(*STAND)
    eye = np.array([x, float(np.asarray(game.surface.height(np.array([x]), np.array([z]))).ravel()[0]) + 1.7, z])

    def camera(q, dt):
        engine.camera.set_pose(position=tuple(eye), target=AIM)
        game.eye = eye

    engine.system(camera)
    game.streamer.focus(x, z)
    game.atmosphere.prepare(wait=True)
    n = 0
    while game.waiting() and n < 2000:
        game.streamer.focus(x, z)
        engine.render_frame(supersample=1)
        n += 1
    frames = Path(tempfile.mkdtemp(prefix="timelapse-", dir=os.path.dirname(os.path.abspath(out)) or "."))
    total = int(minutes / step)
    import time

    started = time.monotonic()
    for i in range(total):
        clock.t = t0 + timedelta(minutes=i * step)
        if game.layer:
            game.runner.warm_up(minutes=step, step=20.0)
        game.streamer.focus(x, z)
        img = engine.render_frame(supersample=1)
        Image.fromarray(img).save(frames / f"{i:05d}.png")
        if i % 60 == 0:
            print({"frame": i, "of": total, "s_per_frame": round((time.monotonic() - started) / (i + 1), 2),
                   **engine.state.sky}, flush=True)
    game.streamer.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "%05d.png"),
                    "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", out], check=True)
    for f in frames.iterdir():
        f.unlink()
    frames.rmdir()
    print({"video": out, "frames": total})
