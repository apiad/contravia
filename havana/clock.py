"""The game's clock, and the thread that keeps the flow layer up to it.

Play runs on the machine's local clock at real speed: at 08:10 on a Monday you drive in Monday's 08:10
traffic. Tests and the dev channel may start the clock at a fixed time (--at "Sun 08:00") and run it faster
(--warp 60); play never does.

FlowRunner steps the flow layer one simulated second at a time until it reaches the clock, in a thread of its
own, and publishes a FlowView the car layer reads without a lock. Anything that changes the flow (settings,
closures, the car layer's counts) is submitted as a function and runs in that thread between two steps.
"""

import threading
import time
from collections import deque
from datetime import datetime, timedelta

from havana.flow import Flow

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def parse_at(text: str) -> datetime:
    """'Sun 08:00' as that weekday and time in the current week (Monday first)."""
    day, hm = text.strip().split()
    hour, minute = (int(v) for v in hm.split(":"))
    today = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    monday = today - timedelta(days=today.weekday())
    return monday + timedelta(days=DAYS.index(day[:3].lower()))


class Clock:
    """Local time now; or, with `at`, a clock that starts there and runs `warp` times faster than real time."""

    def __init__(self, at: datetime | None = None, warp: float = 1.0):
        self.at, self.warp = at, warp
        self._t0 = time.monotonic()

    def now(self) -> datetime:
        if self.at is None and self.warp == 1.0:
            return datetime.now()
        start = self.at or datetime.now()
        return start + timedelta(seconds=(time.monotonic() - self._t0) * self.warp)


class FlowRunner:
    """Keeps a Flow stepped up to a clock, in a thread, and shares its latest view."""

    MAX_STEPS = 600  # per tick; after a longer pause (a sleeping laptop) the flow jumps to the clock

    def __init__(self, flow: Flow, clock):
        self.flow, self.clock = flow, clock
        self.view = flow.view()
        self._calls: deque = deque()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.step_ms = 0.0

    def submit(self, fn) -> None:
        """Run fn(flow) in the flow's thread before its next step."""
        self._calls.append(fn)

    def _run_calls(self) -> None:
        while self._calls:
            self._calls.popleft()(self.flow)

    def warm_up(self, minutes: float = 30) -> None:
        """Step the flow through the `minutes` that end at the clock's time, now, in this thread."""
        end = self.clock.now()
        t = end - timedelta(minutes=minutes)
        while t < end:
            self.flow.step(1.0, t)
            t += timedelta(seconds=1)
        self.flow.when = end
        self.view = self.flow.view()

    def tick(self) -> int:
        """Run the submitted calls, then step one simulated second at a time up to the clock; return the steps."""
        self._run_calls()
        now = self.clock.now()
        t = self.flow.when or now
        steps = int((now - t).total_seconds())
        if steps > self.MAX_STEPS:
            t = now - timedelta(seconds=self.MAX_STEPS)
            steps = self.MAX_STEPS
        if steps <= 0:
            return 0
        t0 = time.perf_counter()
        for k in range(steps):
            self.flow.step(1.0, t + timedelta(seconds=k))
        self.step_ms = (time.perf_counter() - t0) * 1000 / steps
        self.flow.when = t + timedelta(seconds=steps)  # the part of a second left over waits for the next tick
        self.view = self.flow.view()
        return steps

    def start(self) -> None:
        """Tick in a daemon thread, five times a second, until stop()."""
        def loop():
            while not self._stop.is_set():
                self.tick()
                self._stop.wait(0.2)

        self._thread = threading.Thread(target=loop, name="flow", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
