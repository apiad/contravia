"""The loading curtain: a black screen with the logo that hides a mode started from the main menu until the world
around its start is drawn, then fades away.

A mode chosen in the main menu (the free drive, the map) starts somewhere the streamers have not loaded yet. The
curtain goes up at once, the mode starts under it on its next frame, and it stays while anything around the start
still waits to be drawn: the quadtree nodes the view needs, the level-0 cells and the detail ring round the car, the
trees and grass of the cells drawn in full, the first traffic. Then it fades out over FADE seconds; after CAP seconds
it fades out whatever still waits. `ready` and `Curtain` are pure, over the counts the game reads off its streamers
(`Waiting`); `Panel` draws it.
"""

from collections.abc import Callable
from dataclasses import astuple, dataclass

from contravia import ui

CAP = 20.0  # seconds after which the curtain lifts, whatever still waits
FADE = 0.6  # seconds the curtain takes to fade out
TRUST = 3  # frames after the mode starts before its counts are trusted: before, they may be the menu's place's
LOGO_W = 420  # UI pixels of the logo's width
CLEAR = {"padding": 0, "bg": "#00000000"}


@dataclass(frozen=True)
class Waiting:
    """What still waits to be drawn around the start, each as a count; all zero means ready."""

    nodes: int = 0  # quadtree nodes the view needs near the camera and does not draw yet
    cells: int = 0  # level-0 cells round the car that hold something and are not drawn
    detail: int = 0  # cells of the detail ring round the car not drawn
    trees: int = 0  # cells drawn in full whose trees are growing, and groups of trees fading in
    grass: int = 0  # cells drawn in full whose grass is being laid
    tombs: int = 0  # cells drawn in full whose cemetery plots are being laid
    traffic: int = 0  # streets asked of the flow and not handed over yet (0 when the mode has no traffic)


def ready(w: Waiting) -> bool:
    """Whether nothing waits: the world round the start is drawn as it will stay."""
    return not any(astuple(w))


class Curtain:
    """The curtain's state from the moment it goes up, stepped once a frame with the time and what waits.

    `then`, the mode's start, runs on the second step, once the black frame has been drawn: a start that takes
    a while (the traffic's warm-up) never freezes the menu on screen. The counts are trusted TRUST steps after
    it; the curtain starts to lift on the first step they say ready, or CAP seconds after it went up."""

    def __init__(self, now: float, then: Callable[[], None] | None = None, cap: float = CAP, fade: float = FADE):
        self.started, self.then, self.cap, self.fade = now, then, cap, fade
        self.steps = 0
        self.lift_at: float | None = None  # when it began to lift
        self.why = ""  # "ready" or "cap", once it lifts
        self.alpha = 1.0

    @property
    def up(self) -> bool:
        """Whether it still hides the game, wholly or fading; while up, the game ignores the player's input."""
        return self.alpha > 0.0

    @property
    def held(self) -> float | None:
        """Seconds it stayed wholly black, once it began to lift."""
        return None if self.lift_at is None else self.lift_at - self.started

    def step(self, now: float, waiting: Waiting | None) -> float:
        """One frame at time `now` (seconds), with what waits now (None: not counted yet); returns the alpha."""
        self.steps += 1
        if self.steps == 2 and self.then is not None:
            then, self.then = self.then, None
            then()
        if self.lift_at is None:
            trusted = self.steps >= 2 + TRUST and waiting is not None
            if trusted and ready(waiting):
                self.lift_at, self.why = now, "ready"
            elif now - self.started >= self.cap and self.steps > 2:
                self.lift_at, self.why = now, "cap"
        if self.lift_at is not None:
            self.alpha = max(0.0, 1.0 - (now - self.lift_at) / self.fade)
        return self.alpha


class Panel:
    """The curtain on screen: black over the whole window, the logo in the middle and «Cargando…» under it.
    It stays the last root panel, over the HUD and the menus, and fades as one."""

    def __init__(self, engine):
        from manifoldx.gui import Image, Text
        from manifoldx.gui import Panel as GuiPanel

        self.engine = engine
        self.logo = Image(ui.logo(), style_overrides={"width": LOGO_W})
        self.line = Text("CARGANDO…", style_overrides={"font": ui.BOLD, "font_size": 16, "fg": ui.PAPER})
        self.middle = [GuiPanel([self.logo], anchor="center", offset=(0, -30), style_overrides={**CLEAR}),
                       GuiPanel([self.line], anchor="center", offset=(0, 52), style_overrides={**CLEAR})]
        self.back = GuiPanel([], anchor="top-left", offset=(-40, -40),  # past the edges: no lit seam at any scale
                             style_overrides={"padding": 0, "bg": "#000000ff"})
        self.shown = False

    def show(self, alpha: float) -> None:
        """Draw it at `alpha` (0 takes it off), kept over every other panel."""
        g = self.engine.gui
        if alpha <= 0.0:
            if self.shown:
                for p in self.panels:
                    g.remove(p)
                self.shown = False
            return
        self.back.style_overrides = {**self.back.style_overrides, "bg": ui.alpha("#000000", alpha),
                                     "width": int(ui.DESIGN_H * self.engine.w / max(self.engine.h, 1)) + 80,
                                     "height": ui.DESIGN_H + 80}
        self.logo.style_overrides = {**self.logo.style_overrides, "opacity": alpha}
        self.line.style_overrides = {**self.line.style_overrides, "fg": ui.alpha(ui.PAPER, alpha)}
        if not self.shown or [id(p) for p in list(g)[-len(self.panels):]] != [id(p) for p in self.panels]:
            for p in self.panels if self.shown else ():
                g.remove(p)
            for p in self.panels:
                g.append(p)
            self.shown = True

    @property
    def panels(self) -> list:
        return [self.back, *self.middle]
