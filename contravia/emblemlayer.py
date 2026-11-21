"""The map's emblems on screen: pools of GUI panels that contravia/emblems.py's layout fills each frame.

Each medallion, dot and name is a root panel holding one SVG image. The SVG of a medallion is drawn once at a fixed
size and scaled by the panel's width, so zooming never rasterizes it again each frame; a name's SVG holds its size
and changes only when its text or size does. An emblem that comes or goes fades over FADE_S, drawn where the camera
sees it now; all of them fade together with the camera's height (emblems.opacity).
"""

import math

from contravia import emblems, mapview

MEDALS, DOTS, NAMES = 40, 120, 160  # panels in each pool
FADE_S = 0.25  # seconds an emblem takes to appear or leave
OFF = (-1000, -1000)  # offset that puts an unused panel outside the window
CROSS_PX = 6.0  # design pixels past a medallion's rim the crosshair still picks it
CLEAR = {"bg": "#00000000", "padding": 0, "gap": 0, "border": 0}
_FONTS = []


def _fonts() -> None:
    """Make the brand's fonts known to SVG text, once."""
    if not _FONTS:
        from manifoldx import svg

        for f in (emblems.CITY_FONT, emblems.TOWN_FONT):
            svg.load_font(f)
        _FONTS.append(True)


class Pool:
    """Root panels of one image each, handed out per frame. Only the panels in use are in the GUI: the GUI walks
    every root panel each frame, so a parked one would still cost. `attach(panel)` and `detach(panel)` put one in
    and take it out (the screen keeps its list of panels and its own panels on top)."""

    def __init__(self, n: int, attach, detach):
        from manifoldx.gui import Image, Panel

        self.items = []
        for _ in range(n):
            image = Image(emblems.dot_svg(), style_overrides={"width": 1})
            self.items.append((Panel([image], anchor="top-left", offset=OFF, style_overrides=dict(CLEAR)), image))
        self.used, self.attached, self.grew = 0, 0, False
        self.attach, self.detach = attach, detach

    def put(self, svg: str, x: float, y: float, w: float, h: float, alpha: float) -> None:
        if self.used >= len(self.items):
            return
        panel, image = self.items[self.used]
        self.used += 1
        if self.used > self.attached:
            self.attach(panel)
            self.attached, self.grew = self.used, True
        if image.svg != svg:
            image.svg = svg
        size = {"width": round(w, 1), "height": round(h, 1)}
        style = {**size, "opacity": round(alpha, 2)}
        if image.style_overrides != style:
            image.style_overrides = style
        if panel.style_overrides != {**CLEAR, **size}:  # a root panel with no size fills the window
            panel.style_overrides = {**CLEAR, **size}
        offset = (round(x, 1), round(y, 1))
        if panel.offset != offset:
            panel.offset = offset

    def park(self) -> bool:
        """Take the panels not used this frame out of the GUI; True when the panels in it changed."""
        changed = self.grew or self.attached != self.used
        for panel, _ in self.items[self.used:self.attached]:
            self.detach(panel)
        self.attached, self.used, self.grew = min(self.attached, self.used), 0, False
        return changed


class EmblemLayer:
    """The map's medallions, dots and names, with their fades. `panels` go in the screen's panels, below the rest."""

    def __init__(self, towns, places, attach, detach):
        """attach(panel) and detach(panel) put a root panel in the GUI and take it out."""
        _fonts()
        self.towns = towns
        self.near = emblems.places_near(towns, places)
        self.medals, self.dots, self.names = (Pool(n, attach, detach) for n in (MEDALS, DOTS, NAMES))
        self.shown: dict[str, list] = {}  # name -> [emblem, alpha, wanted]
        self.placed_for = None
        self.placed: list[emblems.Emblem] = []

    def update(self, cam: mapview.MapCam, w: float, h: float, dt: float, avoid: tuple = ()) -> bool:
        """Lay the emblems out again when the camera moved, step the fades and fill the pools. True when panels
        went into the GUI or left it, so the screen can put its own panels back on top."""
        if self.placed_for != (cam, w, avoid):
            self.placed_for = (cam, w, avoid)
            self.placed = emblems.layout(self.towns, cam, w, h, self.near, avoid)
            now = {e.name: e for e in self.placed}
            for name, entry in self.shown.items():
                entry[2] = name in now
                if name in now:
                    entry[0] = now[name]
                else:
                    entry[0] = self._moved(entry[0], cam, w, h)
            for name, e in now.items():
                self.shown.setdefault(name, [e, 0.0, True])
        step = dt / FADE_S
        for name in list(self.shown):
            entry = self.shown[name]
            entry[1] = min(1.0, entry[1] + step) if entry[2] else max(0.0, entry[1] - step)
            if entry[1] <= 0.0 and not entry[2]:
                del self.shown[name]
        k = emblems.opacity(cam.h)
        for e, alpha, _ in sorted(self.shown.values(), key=lambda v: -v[0].r):
            a = alpha * k
            if a <= 0.0 or e is None:
                continue
            if e.medallion:
                chip = e.places if e.r >= emblems.CHIP_FROM[0] and cam.h >= emblems.CHIP_FROM[1] else 0
                self.medals.put(emblems.medallion_svg(e.name, chip), e.x - e.r, e.y - e.r, 2 * e.r, 2 * e.r * 1.06, a)
            else:
                self.dots.put(emblems.dot_svg(), e.x - e.r, e.y - e.r, 2 * e.r, 2 * e.r, a)
            if e.label:
                lx, ly, bw, bh = e.label
                self.names.put(emblems.label_svg(e.name, e.medallion, round(e.size, 1), bw, bh), lx, ly, bw, bh, a)
        parked = [pool.park() for pool in (self.medals, self.dots, self.names)]  # every pool parks, then any
        return any(parked)

    def _moved(self, e, cam, w, h):
        """An emblem on its way out, moved to where the camera sees its town now; None once it is behind."""
        from contravia import collect

        eye, target = mapview.pose(cam)
        nx, ny, depth = collect.project(eye, target, [[e.at[0], 0.0, e.at[1]]], 1.0, w / h)[0]
        if depth <= 0:
            return None
        x, y = (nx + 1) / 2 * w, (1 - ny) / 2 * h
        dx, dy = x - e.x, y - e.y
        label = None if e.label is None else (e.label[0] + dx, e.label[1] + dy, e.label[2], e.label[3])
        return emblems.Emblem(e.name, e.medallion, x, y, e.r, e.size, label, e.places, e.at)

    def clear(self) -> None:
        """Take every panel out of the GUI, as when the screen leaves."""
        for pool in (self.medals, self.dots, self.names):
            pool.park()
        self.shown, self.placed_for, self.placed = {}, None, []

    def under(self, w: float, h: float):
        """The medallion under the crosshair at the screen's centre, if any."""
        best = None
        for e in self.placed:
            d = math.hypot(e.x - w / 2, e.y - h / 2)
            if e.medallion and d <= e.r + CROSS_PX and (best is None or d < best[0]):
                best = (d, e)
        return best[1] if best else None

    def label(self, e) -> str:
        n = self.near.get(e.name, 0)
        return f"{e.name} · {n} lugares" if n else e.name
