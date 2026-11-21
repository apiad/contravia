"""The music process: it composes the style of the biome heard and keeps it rendered ahead.

It reads three controls the game writes: music.style (an index into STYLES), music.intensity
and music.date (the game day's ordinal). It keeps AHEAD bars rendered beyond what the sound
process has played, in chunks of CHUNK bars with TAIL seconds of tail. When the style, the day
or the intensity changes, the new music starts where the pushed music ends, which is always a
bar line of the old, and the two cross with equal power over FADE seconds.
"""

from __future__ import annotations

import datetime
import math
import multiprocessing as mp
import time

import numpy as np

CHUNK, AHEAD = 2, 2  # bars per chunk, and bars kept ahead of the player
TAIL, FADE = 2.5, 3.0  # seconds
RATE = 48_000


def crossfade(n: int, seconds: float = FADE) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power gains over n samples: the old music falls along a cosine for `seconds`, the new
    rises along a sine, so their powers always add to 1."""
    t = np.minimum(np.arange(n) / (seconds * RATE), 1.0)
    return np.cos(t * math.pi / 2).astype(np.float32), np.sin(t * math.pi / 2).astype(np.float32)


class Composer:
    """One style playing from a stream time: the next bar to compose and where it starts."""

    def __init__(self, name: str, date: datetime.date, intensity: float, start: float):
        from contravia.audio.music import STYLES, seed_of

        self.key = (name, date, round(intensity, 2))
        self.style = STYLES[name](seed_of(name, date), intensity=intensity)
        self.k, self.t = 0, start

    def next_chunk(self) -> tuple[float, np.ndarray]:
        """The next CHUNK bars, rendered with their tail, and their start in stream seconds."""
        from euterpe.sound import render

        unit, length = self.style.compose(self.k, CHUNK)
        start = self.t
        self.k += CHUNK
        self.t += length
        return start, render(unit, length + TAIL)


def run(writer, cspec) -> None:
    """The music process's loop; it ends with its parent."""
    from manifoldx.audio.controls import Controls

    from contravia.rules.sound import STYLES

    ctl = Controls.attach(cspec)
    parent = mp.parent_process()
    playing: Composer | None = None
    while parent is None or parent.is_alive():
        name = STYLES[int(ctl.get("music.style")) % len(STYLES)]
        day = int(ctl.get("music.date")) or datetime.date.today().toordinal()
        key = (name, datetime.date.fromordinal(day), round(ctl.get("music.intensity") or 1.0, 2))
        clock = writer.clock()
        if playing is None:
            playing = Composer(*key, start=clock + 0.1)
        elif key != playing.key:  # cross into the new music where the pushed music ends
            old, new = playing, Composer(*key, start=playing.t)
            start, x = old.next_chunk()
            g_old, _ = crossfade(len(x))
            writer.push(start, x * g_old[:, None])
            start, x = new.next_chunk()
            _, g_new = crossfade(len(x))
            writer.push(start, x * g_new[:, None])
            playing = new
        while playing.t < clock + AHEAD * playing.style.bar_s:
            writer.push(*playing.next_chunk())
        time.sleep(0.2)
    ctl.close()
