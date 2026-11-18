"""The menus, as docs/ux/contravia-ux.html draws them: the main menu, the free drive's setup, the options, the
pause and the photo mode.

A Menu owns the screen that shows and turns keys into its actions. Each screen is a set of root GUI panels
built for the window's size: shown, they go into engine.gui; hidden, they come out. The keys are the arrows (or
WASD), Enter and Escape; a click on a row chooses it. Story and the map of Cuba stand in the main menu as
"próximamente": choosing them says so and goes nowhere.

The game behind runs in one of four modes (game.mode): "menu" while a menu screen covers it (the car stands,
the clock stops, the streamer keeps loading the start's cells), "drive", "paused" and "photo". The menu tells
the game what to do through Game.start, Game.pause, Game.resume and Game.to_menu.
"""

import math
import textwrap
from typing import ClassVar

from contravia import ui
from contravia.hud import DAY_NAMES
from contravia.settings import WEATHER_ICON, FreeDrive, Options

CLEAR = {"padding": 0, "bg": "#00000000"}
SLIDE = 8.0  # seconds each poster panel stays behind the main menu
ZOOM = 0.08  # how much closer a panel gets over its SLIDE seconds
ROW_H = 72  # pixels between two options of a list menu
NOTE_SHOWN = 2.0  # seconds the "próximamente" note stays
KEYS = {"ArrowUp": "up", "w": "up", "ArrowDown": "down", "s": "down", "ArrowLeft": "left", "a": "left",
        "ArrowRight": "right", "d": "right", "Enter": "enter", "Escape": "back", "Backspace": "back"}


def _row_class():
    """A Panel the pointer can click: a click calls its `on_click`."""
    from manifoldx.gui import Panel

    class Row(Panel):
        _is_gui_interactive = True

        def __init__(self, children, on_click, **kw):
            super().__init__(children, **kw)
            self.on_click = on_click

        def _on_pointer_down(self, ev, engine):
            self.on_click()

    return Row


class Screen:
    """A menu screen: its root panels, built for the window's width, and what its keys do."""

    def __init__(self, menu):
        self.menu = menu
        self.panels = []

    def build(self, width: float) -> None:
        """Make the panels for a window `width` UI pixels wide (the height is always ui.DESIGN_H)."""

    def key(self, k: str) -> None:
        """React to one of "up", "down", "left", "right", "enter" or "back"."""

    def tick(self, dt: float) -> None:
        """Animate what moves; called every frame while the screen shows."""


def _text(text: str, font: str, size: int, fg: str):
    from manifoldx.gui import Text

    return Text(text, style_overrides={"font": font, "font_size": size, "fg": fg})


def _hints(width: float, keys: list[tuple[str, str]], right: str = ""):
    """The bar along the bottom: each key in a framed box with what it does, and a line on the right."""
    from manifoldx.gui import Panel

    items = []
    for key, what in keys:
        items.append(Panel([_text(key, ui.KEYS, 13, ui.PAPER)],
                           style_overrides={"padding": "2 6", "bg": "#00000000", "border": 1,
                                            "border_color": ui.PAPER}))
        items.append(Panel([_text(what, ui.BOLD, 14, ui.PAPER)], style_overrides={**CLEAR, "padding": "3 18 0 2"}))
    items.append(Panel([], style_overrides={**CLEAR, "flex": 1}))
    if right:
        items.append(Panel([_text(right, ui.BOLD, 13, ui.GREY)], style_overrides={**CLEAR, "padding": "3 0 0 0"}))
    return Panel(items, anchor="bottom-left", offset=(0, 0),
                 style_overrides={"direction": "h", "padding": "9 44", "gap": 6, "bg": ui.alpha(ui.INK, 0.92),
                                  "width": int(width), "height": 44})


class Backdrop:
    """The poster's panels behind a menu, one after another, each drawn a little closer over SLIDE seconds,
    with a dark column on the left where the menu is read. `names` fixes one panel or a list to go through."""

    def __init__(self, width: float, names: list[str] | None = None, darken: float = 0.85):
        from manifoldx.gui import Image, Panel

        self.width, self.names, self.t = width, names or ui.panel_names(), 0.0
        self.k = 0
        self.image = Image(ui.panel(self.names[0]), style_overrides={"width": int(width), "height": ui.DESIGN_H})
        self.panels = [
            Panel([self.image], anchor="top-left", style_overrides={**CLEAR, "width": int(width), "height": ui.DESIGN_H}),
            Panel([], anchor="top-left", style_overrides={"padding": 0, "bg": ui.alpha("#000000", darken),
                                                         "width": 500, "height": ui.DESIGN_H}),
        ]
        self._frame()

    def show(self, names: list[str]) -> None:
        """Go through these panels from now on, starting with the first."""
        if names != self.names:
            self.names, self.k, self.t = names, 0, 0.0
            self.image.svg = ui.panel(names[0])
            self._frame()

    def _frame(self) -> None:
        name = self.names[self.k]
        zoom = 1.0 + ZOOM * min(self.t / SLIDE, 1.0) if len(self.names) > 1 else 1.0
        uv = ui.cover_uv(ui.panel_aspect(name), self.width / ui.DESIGN_H, zoom)
        self.image.style_overrides = {**self.image.style_overrides, "uv": uv}

    def tick(self, dt: float) -> None:
        if len(self.names) < 2:
            return
        self.t += dt
        if self.t >= SLIDE:
            self.t, self.k = 0.0, (self.k + 1) % len(self.names)
            self.image.svg = ui.panel(self.names[self.k])
        self._frame()


class ListScreen(Screen):
    """A title and a column of options in large slanted letters, the chosen one on the yellow stripe; each
    option has an action and may carry a line under it. An option whose action is None is "próximamente"."""

    title: str = ""
    hints: ClassVar[list] = [("↑↓", "Elegir"), ("ENTER", "Aceptar"), ("ESC", "Volver")]
    top = 270  # where the first option stands
    logo = False

    def items(self) -> list[tuple[str, str, object]]:
        """(title, line under it, action or None) for each option."""
        return []

    def back(self) -> None:
        """What Escape does."""

    def build(self, width: float) -> None:
        from manifoldx.gui import Image, Panel

        Row = _row_class()
        self.width = width
        self.options = self.items()
        self.index = getattr(self, "index", next((k for k, o in enumerate(self.options) if o[2]), 0))
        self.panels = self.background(width)
        if self.logo:
            self.panels.append(Panel([Image(ui.logo(), style_overrides={"width": 560})], anchor="top-left",
                                     offset=(56, 56), style_overrides=CLEAR))
        elif self.title:
            self.panels.append(Panel([_text(self.title.upper(), ui.TITLE, 50, ui.PAPER)], anchor="top-left",
                                     offset=(60, 40), style_overrides=CLEAR))
        self.stripe = Panel([Image(ui.stripe(360, ROW_H), style_overrides={"width": 360, "height": ROW_H})],
                            anchor="top-left", offset=(58, self.top),
                            style_overrides={**CLEAR, "width": 360, "height": ROW_H})
        self.mark = Panel([Image(ui.bar(10, 5, ui.RED), style_overrides={"width": 10, "height": 5})],
                          anchor="top-left", offset=(42, self.top + 28), style_overrides={**CLEAR, "width": 10, "height": 5})
        self.panels += [self.stripe, self.mark]
        self.titles, self.lines = [], []
        rows = []
        for k, (title, line, action) in enumerate(self.options):
            t = _text(title.upper(), ui.TITLE, 34, ui.PAPER)
            sub = _text(line.upper(), ui.BOLD, 12, ui.PAPER) if line else None
            self.titles.append(t)
            self.lines.append(sub)
            rows.append(Row([t] + ([sub] if sub else []), (lambda k=k: self.click(k)),
                            style_overrides={**CLEAR, "padding": "6 0 0 20", "gap": 0, "width": 420, "height": ROW_H}))
        self.panels.append(Panel(rows, anchor="top-left", offset=(56, self.top),
                                 style_overrides={**CLEAR, "gap": 0}))
        self.note = _text("", ui.BOLD, 16, ui.YELLOW)
        self.note_t = 0.0
        self.panels.append(Panel([self.note], anchor="top-left", offset=(78, self.top + len(self.options) * ROW_H + 8),
                                 style_overrides={**CLEAR, "width": 600, "height": 26}))
        self.panels += self.extra(width)
        self.panels.append(_hints(width, self.hints, self.right_hint()))
        self.select(self.index)

    def background(self, width: float) -> list:
        return []

    def extra(self, width: float) -> list:
        """More panels drawn over the list, such as the pause's card."""
        return []

    def right_hint(self) -> str:
        return ""

    def select(self, k: int) -> None:
        """Put the stripe on option k and colour the options for it."""
        self.index = k % len(self.options)
        for j, (t, sub, (_, _, action)) in enumerate(zip(self.titles, self.lines, self.options, strict=True)):
            on = j == self.index
            fg = ui.INK if on else (ui.PAPER if action else ui.GREY)
            t.style_overrides = {**t.style_overrides, "fg": fg}
            if sub is not None:
                sub.style_overrides = {**sub.style_overrides, "fg": ui.INK if on else (ui.PAPER if action else ui.GREY)}
        y = self.top + self.index * ROW_H
        self.stripe.offset = (58, y)
        self.mark.offset = (42, y + ROW_H // 2 - 4)

    def click(self, k: int) -> None:
        self.select(k)
        self.key("enter")

    def key(self, k: str) -> None:
        if k in ("up", "down"):
            self.select(self.index + (1 if k == "down" else -1))
        elif k == "enter":
            action = self.options[self.index][2]
            if action is None:
                self.note.text = "Próximamente: llega en una versión que viene."
                self.note_t = NOTE_SHOWN
            else:
                action()
        elif k == "back":
            self.back()

    def tick(self, dt: float) -> None:
        if self.note_t > 0:
            self.note_t -= dt
            if self.note_t <= 0:
                self.note.text = ""


class MainScreen(ListScreen):
    logo = True
    top = 250

    def items(self):
        m = self.menu
        return [("Historia", "Cuentos del almendrón · próximamente", None),
                ("Modo libre", "Maneja donde quieras, cuando quieras", lambda: m.show("free")),
                ("Mapa de Cuba", "Explora la isla desde el aire", lambda: m.show("map")),
                ("Opciones", "", lambda: m.show("options")),
                ("Salir", "", m.quit)]

    def background(self, width):
        self.backdrop = Backdrop(width)
        return list(self.backdrop.panels)

    def right_hint(self):
        return "Demo técnica · manifoldx"

    def back(self):
        pass

    def tick(self, dt):
        super().tick(dt)
        self.backdrop.tick(dt)


class PauseScreen(ListScreen):
    title = "Pausa"
    top = 140
    hints: ClassVar[list] = [("↑↓", "Elegir"), ("ENTER", "Aceptar"), ("ESC", "Seguir")]

    def items(self):
        m = self.menu
        return [("Seguir", "", m.resume), ("Mapa", "", lambda: m.show("map")), ("Modo foto", "", lambda: m.show("photo")),
                ("Opciones", "", lambda: m.show("options")), ("Menú principal", "", m.to_menu)]

    def background(self, width):
        from manifoldx.gui import Panel

        return [Panel([], anchor="top-left", style_overrides={"bg": ui.alpha("#000000", 0.72), "width": int(width),
                                                               "height": ui.DESIGN_H})]

    def extra(self, width):
        """The card on the right: where the car is, the time, the weather and the kilometres driven."""
        from manifoldx.gui import Image, Panel

        g = self.menu.game
        when = g.clock.now()
        state = getattr(g, "weather_state", None)
        area = g.area or "La Habana"
        line = Panel([Image(ui.icon("reloj"), style_overrides={"width": 20, "height": 20}),
                      _text(f"{DAY_NAMES[when.weekday()].capitalize()} {when:%H:%M}", ui.BOLD, 17, ui.PAPER),
                      Image(ui.icon(WEATHER_ICON.get(state, "sol")), style_overrides={"width": 20, "height": 20}),
                      _text(f"{g.km:.1f} km".replace(".", ","), ui.BOLD, 17, ui.PAPER)],
                     style_overrides={**CLEAR, "direction": "h", "gap": 10, "padding": "8 0 0 0"})
        card = Panel([Panel([], style_overrides={"bg": ui.YELLOW, "height": 4, "padding": 0}),
                      Panel([_text(area.upper(), ui.TITLE, 30, ui.PAPER),
                             _text("La Habana" if g.area else "", ui.BOLD, 14, ui.PAPER), line],
                            style_overrides={**CLEAR, "padding": "12 16 16 16", "gap": 2})],
                     anchor="top-right", offset=(56, 150),
                     style_overrides={"padding": 0, "gap": 0, "bg": ui.alpha(ui.INK, 0.92), "width": 400})
        return [card]

    def back(self):
        self.menu.resume()


class RowsScreen(Screen):
    """A title, a column of Choice rows turned with left and right, an explanation of the chosen row, and
    whatever the subclass puts on the right."""

    title = ""
    hints: ClassVar[list] = [("↑↓", "Ajuste"), ("←→", "Cambiar"), ("ESC", "Volver")]

    def rows(self):
        return []

    def build(self, width: float) -> None:
        from manifoldx.gui import Image, Panel

        Row = _row_class()
        self.width = width
        self.index = getattr(self, "index", 0)
        self.panels = self.background(width)
        self.panels.append(Panel([_text(self.title.upper(), ui.TITLE, 50, ui.PAPER)], anchor="top-left",
                                 offset=(60, 40), style_overrides=CLEAR))
        self.values, self.frames = [], []
        rows = []
        for k, row in enumerate(self.rows()):
            value = _text(f"‹  {row.text.upper()}  ›", ui.TITLE, 19, ui.YELLOW)
            value_panel = Panel([value], style_overrides={**CLEAR, "padding": "10 14 0 0"})
            self.values.append((value, value_panel))
            mark = Image(ui.bar(4, 44, ui.RED), style_overrides={"width": 4, "height": 44, "opacity": 0.0})
            r = Row([mark, Panel([_text(row.label.upper(), ui.BOLD, 17, ui.PAPER)],
                                 style_overrides={**CLEAR, "padding": "11 0 0 14"}),
                     Panel([], style_overrides={**CLEAR, "flex": 1}),
                     value_panel],
                    (lambda k=k: self.click(k)),
                    style_overrides={"direction": "h", "padding": 0, "gap": 0, "bg": ui.alpha(ui.INK, 0.82),
                                     "border": 2, "border_color": "#00000000", "width": 490, "height": 48})
            r.mark = mark
            self.frames.append(r)
            rows.append(r)
        self.panels.append(Panel(rows, anchor="top-left", offset=(60, 130), style_overrides={**CLEAR, "gap": 6}))
        self.help = [_text("", ui.BOLD, 15, ui.PAPER) for _ in range(3)]
        self.panels.append(Panel(self.help, anchor="top-left", offset=(62, 130 + 54 * len(rows) + 12),
                                 style_overrides={**CLEAR, "gap": 0, "width": 490, "height": 66}))
        self.panels += self.extra(width)
        self.panels.append(_hints(width, self.hints))
        self.select(self.index)

    def background(self, width: float) -> list:
        return []

    def extra(self, width: float) -> list:
        return []

    def select(self, k: int) -> None:
        rows = self.rows()
        self.index = k % len(rows)
        for j, r in enumerate(self.frames):
            on = j == self.index
            r.style_overrides = {**r.style_overrides, "border_color": ui.YELLOW if on else "#00000000"}
            r.mark.style_overrides = {**r.mark.style_overrides, "opacity": 1.0 if on else 0.0}
        lines = textwrap.wrap(rows[self.index].help, 62)[:3]
        for t, line in zip(self.help, lines + [""] * 3, strict=False):
            t.text = line
        self.changed()

    def changed(self) -> None:
        """Called when a row is chosen or turned."""

    def click(self, k: int) -> None:
        if k == self.index:
            self.key("right")
        else:
            self.select(k)

    def key(self, k: str) -> None:
        rows = self.rows()
        if k in ("up", "down"):
            self.select(self.index + (1 if k == "down" else -1))
        elif k in ("left", "right"):
            row = rows[self.index]
            row.step(1 if k == "right" else -1)
            value, panel = self.values[self.index]
            value.text = f"‹  {row.text.upper()}  ›"
            ui.refit(panel)
            self.changed()
        elif k == "enter":
            self.enter()
        elif k == "back":
            self.back()

    def enter(self) -> None:
        self.key("right")

    def back(self) -> None:
        pass


class FreeScreen(RowsScreen):
    """The free drive's setup: the rows of settings.FreeDrive, the place's poster panel and ARRANCAR."""

    title = "Modo libre"
    hints: ClassVar[list] = [("↑↓", "Ajuste"), ("←→", "Cambiar"), ("ENTER", "Arrancar"), ("ESC", "Volver")]

    def rows(self):
        return self.menu.free.rows

    def background(self, width):
        self.backdrop = Backdrop(width, ["capitolio"], darken=0.9)
        return list(self.backdrop.panels)

    def extra(self, width):
        from manifoldx.gui import Image, Panel

        Row = _row_class()
        self.card_image = Image(ui.panel("capitolio"), style_overrides={"width": 452, "height": 300})
        self.card_name = _text("", ui.TITLE, 24, ui.INK)
        card = Panel([self.card_image], anchor="top-right", offset=(60, 130),
                     style_overrides={"padding": 4, "bg": ui.PAPER, "width": 460, "height": 308})
        caption = Panel([Panel([self.card_name], style_overrides={"padding": "4 14", "bg": ui.YELLOW})],
                        anchor="top-right", offset=(240, 400), style_overrides=CLEAR)
        start = Row([Image(ui.stripe(240, 62), style_overrides={"width": 240, "height": 62})], self.menu.start_free,
                    anchor="bottom-right", offset=(60, 90), style_overrides={**CLEAR, "width": 240, "height": 62})
        label = Panel([_text("ARRANCAR ›", ui.TITLE, 32, ui.INK)], anchor="bottom-right", offset=(92, 100),
                      style_overrides=CLEAR)
        self.card_panels = [card, caption]
        return [card, caption, start, label]

    def changed(self):
        if not hasattr(self, "card_name"):
            return
        name, district, poster = self.menu.free.place()
        self.card_name.text = f"{name} · {district}".upper()
        ui.refit(self.card_panels[1])
        if poster:
            self.card_image.svg = ui.panel(poster)
            self.card_image.style_overrides = {**self.card_image.style_overrides,
                                               "uv": ui.cover_uv(ui.panel_aspect(poster), 452 / 300)}
            self.backdrop.show([poster])
        else:
            self.card_image.svg = ui.logo()
            self.card_image.style_overrides = {**self.card_image.style_overrides, "uv": (0.0, 0.0, 1.0, 1.0)}

    def enter(self):
        self.menu.start_free()

    def back(self):
        self.menu.show("main")


class OptionsScreen(RowsScreen):
    """The options kept between runs; Escape saves them and goes back where the menu came from."""

    title = "Opciones"
    hints: ClassVar[list] = [("↑↓", "Ajuste"), ("←→", "Cambiar"), ("ESC", "Guardar y volver")]

    def rows(self):
        return self.menu.options.rows

    def background(self, width):
        if self.menu.game.mode == "paused":
            from manifoldx.gui import Panel

            return [Panel([], anchor="top-left", style_overrides={"bg": ui.alpha("#000000", 0.8), "width": int(width),
                                                                   "height": ui.DESIGN_H})]
        self.backdrop = Backdrop(width, ["noche"], darken=0.9)
        return list(self.backdrop.panels)

    def changed(self):
        if hasattr(self, "values"):
            self.menu.game.apply_options(self.menu.options)

    def back(self):
        self.menu.options.save()
        self.menu.show("pause" if self.menu.game.mode == "paused" else "main")


class PhotoScreen(Screen):
    """The photo mode: the HUD and the menus gone, the camera free around the car; arrows turn it, W and S
    bring it closer or farther, Escape goes back to the pause."""

    def build(self, width):
        self.panels = [_hints(width, [("←→↑↓", "Girar"), ("W S", "Acercar"), ("ESC", "Volver")])]

    def key(self, k):
        if k == "back":
            self.menu.game.photo = None
            self.menu.show("pause")

    def tick(self, dt):
        g, inp = self.menu.game, self.menu.engine.input
        yaw, pitch, dist = g.photo or (g.car.yaw + math.pi, 0.25, 9.0)
        yaw += (inp.is_pressed("ArrowRight") - inp.is_pressed("ArrowLeft")) * dt * 1.5
        pitch = min(1.4, max(0.02, pitch + (inp.is_pressed("ArrowUp") - inp.is_pressed("ArrowDown")) * dt))
        dist = min(60.0, max(4.0, dist * (1 + (inp.is_pressed("s") - inp.is_pressed("w")) * dt * 1.5)))
        g.photo = (yaw, pitch, dist)


class MapScreen(Screen):
    """The map: the world from the air (contravia/mapview.py). The wheel or W and S zoom, dragging or the arrows
    move, Q and E turn, ENTER drives on the road nearest the crosshair, Escape goes back where it came from."""

    PAN_PX = 600.0  # screen pixels a second the arrows move the ground
    TURN = 1.2  # radians a second for Q and E

    def build(self, width):
        from manifoldx.gui import Panel

        if self.menu.current_before != "map":  # a resized window rebuilds the map from the map
            self.back_to = "pause" if self.menu.current_before == "pause" else "main"
        cross = Panel([], anchor="center", style_overrides={**CLEAR, "width": 22, "height": 22, "border": 3,
                                                             "border_color": ui.YELLOW})
        self.note = _text("", ui.BOLD, 15, ui.PAPER)
        self.panels = [cross, Panel([self.note], anchor="top-left", offset=(44, 40), style_overrides=CLEAR),
                       _hints(width, [("RUEDA", "Zoom"), ("ARRASTRAR", "Mover"), ("Q E", "Girar"),
                                      ("ENTER", "Manejar aquí"), ("ESC", "Volver")])]
        self.menu.game.open_map()

    def key(self, k):
        g = self.menu.game
        if k == "back":
            g.map = None
            if self.back_to == "pause":
                self.menu.show("pause")
            else:
                self.menu.to_menu()
        elif k == "enter":
            error = g.drive_here()
            if error:
                self.note.text = "Aquí no hay calle cerca"
            else:
                self.menu.show(None)

    def tick(self, dt):
        from contravia import mapview

        g, inp = self.menu.game, self.menu.engine.input
        if g.map is None:
            return
        cam, vh = g.map, max(self.menu.engine.h, 1)
        wheel = inp.wheel_delta[1]
        held = inp.is_pressed("s") - inp.is_pressed("w")
        if wheel or held:
            cam = mapview.zoom(cam, 1.0015 ** wheel * (1 + held * dt * 1.5))
        if inp.is_mouse_pressed(1):
            dx, dy = inp.mouse_delta
            cam = mapview.pan(cam, dx, dy, vh)
        ax = inp.is_pressed("ArrowRight") - inp.is_pressed("ArrowLeft")
        ay = inp.is_pressed("ArrowDown") - inp.is_pressed("ArrowUp")
        if ax or ay:  # the arrows move the view, so the ground goes the other way
            cam = mapview.pan(cam, -ax * self.PAN_PX * dt, -ay * self.PAN_PX * dt, vh)
        turn = inp.is_pressed("e") - inp.is_pressed("q")
        if turn:
            cam = mapview.MapCam(cam.x, cam.z, cam.h, cam.yaw + turn * self.TURN * dt)
        g.map = cam


SCREENS = {"main": MainScreen, "free": FreeScreen, "options": OptionsScreen, "pause": PauseScreen,
           "photo": PhotoScreen, "map": MapScreen}
KEY_NAMES = {"ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "w", "s", "a", "d", "Enter", "Escape", "Backspace"}


class Menu:
    """The menus over a Game: which screen shows, the keys, and the moves between menus and driving."""

    def __init__(self, engine, game, free: FreeDrive, options: Options, screen: str | None = "main"):
        self.engine, self.game, self.free, self.options = engine, game, free, options
        self.screens, self.current, self.current_before, self._size = {}, None, None, None
        game.apply_options(options)
        self._first = screen
        if screen:
            game.to_menu()
        engine.on("startup")(self._startup)
        engine.system(self.update)

    def _startup(self, _=None) -> None:
        if self._first:
            self.show(self._first)

    def width(self) -> float:
        """The window's width in UI pixels."""
        return ui.DESIGN_H * self.engine.w / max(self.engine.h, 1)

    def show(self, name: str | None) -> None:
        """Take the current screen off and put `name` on; None leaves no menu."""
        g = self.engine.gui
        if self.current is not None:
            for p in self.screens[self.current].panels:
                g.remove(p)
        self.current_before, self.current = self.current, name
        if name is None:
            return
        screen = self.screens.get(name) or SCREENS[name](self)
        self.screens[name] = screen
        screen.build(self.width())
        self._size = (self.engine.w, self.engine.h)
        for p in screen.panels:
            g.append(p if p.anchor != "top-left" and not any(hasattr(c, "children") for c in p.children)
                     else ui.fit(p))
        if name == "photo":
            self.game.mode = "photo"
        elif name == "map":
            self.game.mode = "map"
        elif name == "pause":
            self.game.mode = "paused"

    def press(self, key: str) -> None:
        """One key, by its engine name ("ArrowUp", "Enter", "Escape"...), as if it were just pressed."""
        if self.current is None:
            if key == "Escape" and self.game.mode == "drive":
                self.pause()
            return
        if key in KEYS:
            self.screens[self.current].key(KEYS[key])

    def update(self, query, dt: float) -> None:
        """Read this frame's keys, follow a resized window and animate the screen."""
        inp = self.engine.input
        for k in KEY_NAMES:
            if inp.just_pressed(k):
                if self.current in ("photo", "map") and k in ("w", "s", "a", "d", "ArrowUp", "ArrowDown", "ArrowLeft",
                                                     "ArrowRight"):
                    continue  # the photo mode reads them as held keys
                if self.current is None and k != "Escape":
                    continue
                self.press(k)
                break
        if self.current is None:
            return
        if self._size != (self.engine.w, self.engine.h):
            self.show(self.current)
        self.screens[self.current].tick(min(dt, 0.1))

    def start_free(self) -> None:
        """Drive with the free drive's rows."""
        self.show(None)
        self.game.start(**self.free.start())

    def pause(self) -> None:
        self.game.pause()
        self.screens.pop("pause", None)  # its card shows the moment it opens
        self.show("pause")

    def resume(self) -> None:
        self.show(None)
        self.game.resume()

    def to_menu(self) -> None:
        self.game.to_menu()
        self.show("main")

    def quit(self) -> None:
        self.engine.quit()
