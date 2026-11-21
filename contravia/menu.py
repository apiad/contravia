"""The menus, as docs/ux/contravia-ux.html draws them: the main menu, the free drive's setup, the options, the
pause and the photo mode.

A Menu owns the screen that shows and turns keys into its actions. Each screen is a set of root GUI panels
built for the window's size: shown, they go into engine.gui; hidden, they come out. The keys are the arrows (or
WASD), Enter and Escape; a click on a row chooses it. Story stands in the main menu as "próximamente": choosing
it says so and goes nowhere. The Garaje's two screens, the Salón (browse) and the Taller (design), are drawn as
docs/superpowers/specs/2026-11-18-garaje-design.html proposes, over the vehicle turning in the street.

The game behind runs in one of these modes (game.mode): "menu" while a menu screen covers it (the car stands,
the clock stops, the streamer keeps loading the start's cells), "drive", "paused", "photo", "map" and "garage". The menu tells
the game what to do through Game.start, Game.pause, Game.resume and Game.to_menu.
A mode chosen in the main menu (the free drive, the map) starts under the loading curtain (contravia/curtain.py,
Game.raise_curtain), which lifts once the world round its start is drawn; under it only Escape works, back to the
main menu. A later move within the game (the pause, the map's flights) never shows it.
While driving, Escape pauses, F aims a camera at a place and L turns the driving help to its next level.
"""

import functools
import math
import textwrap
from typing import ClassVar

from contravia import album, collect, emblemlayer, garage, mapview, nearby, placemarks, towns, ui, whereis
from contravia.driving_help import next_level
from contravia.hud import DAY_NAMES
from contravia.rules import designs, fleet
from contravia.settings import WEATHER_ICON, FreeDrive, Options

CLEAR = {"padding": 0, "bg": "#00000000"}
SLIDE = 8.0  # seconds each poster panel stays behind the main menu
ZOOM = 0.08  # how much closer a panel gets over its SLIDE seconds
ROW_H = 72  # pixels between two options of a list menu
NOTE_SHOWN = 2.0  # seconds the "próximamente" note stays
KEYS = {"ArrowUp": "up", "w": "up", "ArrowDown": "down", "s": "down", "ArrowLeft": "left", "a": "left",
        "ArrowRight": "right", "d": "right", "Enter": "enter", "Escape": "back", "Backspace": "back",
        "q": "prev", "e": "next", "Tab": "design", "g": "save", "r": "random", "z": "undo", "Delete": "delete", "Space": "shoot", "f": "photo"}
TURN = 0.35  # radians a second the Garaje's camera goes round the vehicle


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
    row_h = ROW_H
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
        self.stripe = Panel([Image(ui.stripe(360, self.row_h), style_overrides={"width": 360, "height": self.row_h})],
                            anchor="top-left", offset=(58, self.top),
                            style_overrides={**CLEAR, "width": 360, "height": self.row_h})
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
                            style_overrides={**CLEAR, "padding": "6 0 0 20", "gap": 0, "width": 420, "height": self.row_h}))
        self.panels.append(Panel(rows, anchor="top-left", offset=(56, self.top),
                                 style_overrides={**CLEAR, "gap": 0}))
        self.note = _text("", ui.BOLD, 16, ui.YELLOW)
        self.note_t = 0.0
        self.panels.append(Panel([self.note], anchor="top-left", offset=(78, self.top + len(self.options) * self.row_h + 8),
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
        y = self.top + self.index * self.row_h
        self.stripe.offset = (58, y)
        self.mark.offset = (42, y + self.row_h // 2 - 4)

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
    top = 214
    row_h = 64  # seven options above the hints

    def items(self):
        m = self.menu
        return [("Historia", "Cuentos del almendrón · próximamente", None),
                ("Modo libre", "Maneja donde quieras, cuando quieras", lambda: m.show("free")),
                ("Mapa de Cuba", "Explora la isla desde el aire", m.open_map),
                ("Garaje", "Todos los vehículos de Cuba, a tu gusto", lambda: m.show("garage")),
                ("Álbum", "Tus fotos de los lugares de Cuba", lambda: m.show("album")),
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
        route = [("Quitar ruta", "", lambda: (m.game.gps_clear(), m.resume()))] if m.game.gps.to is not None else []
        return [("Seguir", "", m.resume), ("Mapa", "", lambda: m.show("map")), *route,
                ("Cambiar vehículo", "", lambda: m.show("garage")), ("Modo foto", "", lambda: m.show("photo")),
                ("Álbum", "", lambda: m.show("album")),
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


class AimScreen(Screen):
    """Aiming at a place for its photo: the camera over the car looking at it; the arrows turn it a little, W and S
    zoom, SPACE shoots and says what is wrong if the framing is not good, Escape drives on."""

    def build(self, width):
        from manifoldx.gui import Panel

        self.note = _text(" ", ui.BOLD, 18, ui.PAPER)
        place = self.menu.game.aim["place"].title if self.menu.game.aim else ""
        self.panels = [Panel([_text(place, ui.TITLE, 30, ui.PAPER), self.note], anchor="top-left", offset=(44, 40),
                             style_overrides={**CLEAR, "gap": 4}),
                       _hints(width, [("←→↑↓", "Girar"), ("W S", "Zoom"), ("ESPACIO", "Disparar"), ("ESC", "Volver")])]

    def key(self, k):
        g = self.menu.game
        if k == "back":
            g.resume()
            self.menu.show(None)
        elif k == "shoot" and g.aim:
            f = g.shoot()
            self.note.text = collect.WHY.get(f.verdict, "¡Buena foto!")
            if f.verdict == "good":
                self.menu.took(f)

    def tick(self, dt):
        g, inp = self.menu.game, self.menu.engine.input
        if not g.aim:
            return
        g.turn_aim((inp.is_pressed("ArrowRight") - inp.is_pressed("ArrowLeft")) * dt * 0.4,
                   (inp.is_pressed("ArrowUp") - inp.is_pressed("ArrowDown")) * dt * 0.3,
                   1 + (inp.is_pressed("w") - inp.is_pressed("s")) * dt * 0.8)


class CardScreen(Screen):
    """A place's card, after its photo: the photo, how many of its province the player has, the title, what it is,
    where and when, and its text. ENTER or Escape drives on."""

    def build(self, width):
        from manifoldx.gui import Image, Panel

        c = self.menu.card
        p = c["place"]
        lines = textwrap.wrap(p.text, 58)[:8]
        body = [_text(c["count"], ui.BOLD, 13, ui.INK)]
        chip = Panel(body, style_overrides={"padding": "2 8", "bg": ui.YELLOW})
        words = Panel([chip, _text(p.title, ui.TITLE, 30, ui.PAPER), _text(c["meta"], ui.BOLD, 14, ui.GREY)]
                      + [_text(line, ui.BOLD, 15, ui.PAPER) for line in lines],
                      style_overrides={**CLEAR, "gap": 6, "width": 470})
        photo = Image(c["texture"], style_overrides={"width": 480, "height": 270})
        self.panels = [Panel([photo, words], anchor="center",
                             style_overrides={"direction": "h", "gap": 22, "padding": 22, "bg": ui.alpha(ui.INK, 0.92)}),
                       _hints(width, [("ENTER", "Volver" if c.get("back") else "Seguir")])]

    def key(self, k):
        if k in ("enter", "back"):
            if self.menu.card.get("back"):
                self.menu.show(self.menu.card["back"])
                return
            self.menu.game.resume()
            self.menu.show(None)


MAP_MARKS = 80  # most place marks the map shows at once
MARK_PX = 26  # pixels across a place's icon on the map
OFF = (-1000, -1000)  # offset that puts an unused mark outside the window
HERE_ICON_PX = 18  # pixels across a place's icon in the panel of what lies near the crosshair
HERE_CHARS = 40  # longest place name the panel writes whole
CHOICES = ("Ver ficha", "Manejar aquí", "Ir con GPS", "Quitar ruta")  # the most the map's choice offers


class MapScreen(Screen):
    """The map: the world from the air (contravia/mapview.py). The wheel or W and S zoom, dragging or the arrows
    move, Q and E turn, ENTER drives on the road nearest the crosshair, Escape goes back where it came from.
    The collectible places show as icons (contravia/placemarks.py); with the crosshair on one the note names it
    and ENTER offers its card or driving there, and on a municipio's group ENTER comes down over it. The panel on
    the right says what lies under and near the crosshair, in more detail as the camera comes down
    (contravia/nearby.py), looked up off the frame (contravia/whereis.py)."""

    PAN_PX = 600.0  # screen pixels a second the arrows move the ground
    TURN = 1.2  # radians a second for Q and E
    ZOOM = 1.5  # the log of the height a second for W and S
    GRIP = 0.06  # seconds of drag the glide's speed is measured over

    def build(self, width):
        from manifoldx.gui import Image, Panel

        if self.menu.current_before not in ("map", "card"):  # a resized window or a card rebuilds it from the map
            self.back_to = "pause" if self.menu.current_before == "pause" else "main"
        self.marks, self.under, self.choice, self.picked, self.marks_for = [], None, None, 0, None
        self.target, self.drawn, self.glide = None, None, (0.0, 0.0)  # the camera the input moves; _move
        g = self.menu.game
        self.emblems = emblemlayer.EmblemLayer(_towns(), g.places, self._attach, self._detach)
        self.city = None  # the medallion under the crosshair
        for _ in range(MAP_MARKS):
            image = Image(placemarks.icon("lugar", False, MARK_PX), style_overrides={"width": MARK_PX})
            count = _text("", ui.BOLD, 13, ui.PAPER)
            self.marks.append((Panel([image, count], anchor="top-left", offset=OFF,
                                     style_overrides={**CLEAR, "direction": "h", "gap": 4}), image, count))
        cross = Panel([], anchor="center", style_overrides={**CLEAR, "width": 22, "height": 22, "border": 3,
                                                             "border_color": ui.YELLOW})
        self.note = _text("", ui.BOLD, 15, ui.PAPER)
        self.options = [_text(t, ui.TITLE, 26, ui.PAPER) for t in CHOICES]
        self.choice_panel = Panel(self.options, anchor="center", offset=(0, 90),
                                  style_overrides={"gap": 4, "padding": "10 18", "bg": ui.alpha(ui.INK, 0.92)})
        self._build_here()
        self.top = [cross, self.here_panel, Panel([self.note], anchor="top-left", offset=(44, 40), style_overrides=CLEAR),
                    _hints(width, [("RUEDA", "Zoom"), ("ARRASTRAR", "Mover"), ("Q E", "Girar"),
                                   ("ENTER", "Elegir"), ("ESC", "Volver")])]
        self.panels = [*(m for m, _, _ in self.marks), *self.top]
        self.width = width
        self.menu.game.open_map()

    def _attach(self, panel) -> None:
        """Put an emblem's panel in the GUI and in this screen's panels, so leaving the screen takes it out."""
        self.menu.engine.gui.append(panel)
        self.panels.append(panel)

    def _detach(self, panel) -> None:
        self.menu.engine.gui.remove(panel)
        self.panels.remove(panel)

    def _clear_of(self) -> tuple:
        """The screen's boxes the emblems keep clear of: the panel on the right and the hints at the bottom."""
        so = self.here_panel.style_overrides
        w, h = so.get("width") or 0, so.get("height") or 0
        right, top = self.here_panel.offset
        return ((self.width - right - w, top, self.width - right, top + h), (0, ui.DESIGN_H - 46, self.width, ui.DESIGN_H))

    def _on_top(self) -> None:
        """Put the crosshair, the panel, the note and the hints back over the emblems."""
        gui = self.menu.engine.gui
        for p in self.top:
            gui.remove(p)
            gui.append(p)

    def _build_here(self) -> None:
        """The panel of what lies under and near the crosshair: a title, a line, a header and a row per place."""
        from manifoldx.gui import Image, Panel

        g = self.menu.game
        if getattr(self, "where", None) is None or self.where.world is not g.city:
            self.where = whereis.Where(g.city)
        self.asked, self.shown = None, None
        self.here_title = _text("", ui.TITLE, 24, ui.PAPER)
        self.here_line = _text("", ui.BOLD, 14, ui.GREY)
        self.here_header = _text("", ui.BOLD, 12, ui.GREY)
        self.here_rows = []
        for _ in range(max(b.count for b in nearby.BANDS)):
            image = Image(placemarks.icon("lugar", False, HERE_ICON_PX), style_overrides={"width": HERE_ICON_PX})
            name, far = _text("", ui.BOLD, 14, ui.PAPER), _text("", ui.BOLD, 13, ui.GREY)
            self.here_rows.append((Panel([image, name, far], style_overrides={**CLEAR, "direction": "h", "gap": 8}),
                                   image, name, far))
        self.here_panel = Panel([self.here_title], anchor="top-right", offset=(40, 40),
                                style_overrides={"gap": 4, "padding": "14 18", "bg": ui.alpha(ui.INK, 0.92)})

    def _here(self) -> None:
        """Ask for the panel again when the crosshair or the height moved enough, and paint a new answer."""
        g = self.menu.game
        cam = g.map
        if nearby.stale(self.asked, cam.x, cam.z, cam.h):
            self.asked = (cam.x, cam.z, cam.h)
            self.where.ask(g.places, cam.x, cam.z, cam.h)
        here = self.where.take()
        if here is None:
            return
        got = tuple(g.album.has(i.id) for i in here.items)
        if (here, got) == self.shown:
            return
        self.shown = (here, got)
        self.here_title.text, self.here_line.text, self.here_header.text = here.title, here.line, here.header
        children = [self.here_title] + [t for t in (self.here_line, self.here_header) if t.text]
        for item, has, (row, image, name, far) in zip(here.items, got, self.here_rows, strict=False):
            svg = placemarks.icon(item.kind, has, HERE_ICON_PX)
            if image.svg != svg:
                image.svg = svg
            name.text = item.title if len(item.title) <= HERE_CHARS else item.title[:HERE_CHARS - 1] + "…"
            far.text = item.distance
            children.append(row)
        self.here_panel.children = children
        ui.refit(self.here_panel)

    def _rows(self, mark) -> list[tuple[str, object]]:
        """What the choice offers for the place `mark` (or, None, the point under the crosshair): its card and driving
        there; from the drive's map, guiding the car there with the GPS, and dropping the route when one is set."""
        g = self.menu.game
        rows = []
        if mark is not None:
            p = g.places.by_id(mark.ids[0])
            rows += [("Ver ficha", lambda: self.menu.view_card(p)), ("Manejar aquí", lambda: self._drive(g.drive_to(p)))]
            to = lambda: (*g.near_spot(p), p.title)  # noqa: E731
        else:
            rows += [("Manejar aquí", lambda: self._drive(g.drive_here()))]
            to = lambda: (g.map.x, g.map.z, None)  # noqa: E731
        if self.back_to == "pause":  # a car to guide
            rows.append(("Ir con GPS", lambda: self._gps(*to())))
            if g.gps.to is not None:
                rows.append(("Quitar ruta", lambda: (g.gps_clear(), self._open_choice(False))))
        return rows

    def _drive(self, error) -> None:
        if error:
            self.note.text = "Aquí no hay calle cerca"
        else:
            self.menu.show(None)

    def _gps(self, x: float, z: float, title) -> None:
        g = self.menu.game
        g.gps_to(x, z, title)
        g.back_to_car()
        self.menu.show(None)

    def _open_choice(self, on: bool, mark=None) -> None:
        """Show or hide the choice for the place `mark` under the crosshair, or for the point under it."""
        if on == (self.choice is not None):
            return
        self.choice, self.picked = (self._rows(mark) if on else None), 0
        if on:
            for t, (label, _) in zip(self.options, self.choice, strict=False):
                t.text = label
            self.choice_panel.children = self.options[:len(self.choice)]
            ui.refit(self.choice_panel)
        (self.menu.engine.gui.append if on else self.menu.engine.gui.remove)(self.choice_panel)
        self._paint_choice()

    def _paint_choice(self) -> None:
        for k, t in enumerate(self.options):
            t.style_overrides = {**t.style_overrides, "fg": ui.YELLOW if k == self.picked else ui.PAPER}

    def key(self, k):
        g = self.menu.game
        if g.flight is not None and k != "back":
            return  # the camera is still flying to the map
        if self.choice is not None:
            if k in ("up", "down"):
                self.picked = (self.picked + (1 if k == "down" else -1)) % len(self.choice)
                self._paint_choice()
            elif k == "back":
                self._open_choice(False)
            elif k == "enter":
                act = self.choice[self.picked][1]
                self._open_choice(False)
                act()
            return
        if k == "enter" and self.under is None and self.city is not None:  # a city: come down over it
            x, z = self.city.at
            self.target = mapview.MapCam(x, z, placemarks.CITY_H / 2, g.map.yaw)
            return
        if k == "enter" and self.under is not None:
            if len(self.under.ids) > 1:  # a group: come down over it, below the city stage
                x, z = self.under.at
                self.target = mapview.MapCam(x, z, placemarks.CITY_H / 2, g.map.yaw)  # the follow glides there
            else:
                self._open_choice(True, self.under)
            return
        if k == "back":
            g.map = None
            if self.back_to == "pause":
                self.menu.show("pause")
            else:
                self.menu.to_menu()
        elif k == "enter" and self.back_to == "pause":  # from the drive: drive there or guide the car there
            self._open_choice(True)
        elif k == "enter":
            self._drive(g.drive_here())

    def tick(self, dt):
        g = self.menu.game
        if g.map is None or g.flight is not None:
            return
        if self.choice is None:
            self._move(dt)
        if self.emblems.update(g.map, self.width, ui.DESIGN_H, dt, self._clear_of()):
            self._on_top()
        self._place_marks()
        self._here()

    def _place_marks(self) -> None:
        """Put the places' marks where the camera sees them and name the one under the crosshair."""
        g = self.menu.game
        if getattr(self, "marks_for", None) == (g.map, self.width):
            return  # the camera did not move: the marks stand
        self.marks_for = (g.map, self.width)
        # From the city stage up the emblems stand for the places (their chips); below, each place has its mark.
        marks = [] if g.map.h >= placemarks.CITY_H else placemarks.on_map(g.places, g.album.has, g.map, self.width,
                                                                        ui.DESIGN_H)
        for k, (panel, image, count) in enumerate(self.marks):
            if k >= len(marks):
                panel.offset = OFF
                continue
            m = marks[k]
            svg = placemarks.icon(m.kind, m.got, MARK_PX)
            if image.svg != svg:
                image.svg = svg
            number = str(len(m.ids)) if len(m.ids) > 1 else ""
            if count.text != number:
                count.text = number
            panel.offset = (round(m.x - MARK_PX / 2), round(m.y - MARK_PX / 2))
        was = self.under or self.city
        self.under = placemarks.under_cross(marks, self.width, ui.DESIGN_H)
        self.city = None if self.under else self.emblems.under(self.width, ui.DESIGN_H)
        if self.under is not None:
            self.note.text = self.under.label
        elif self.city is not None:
            self.note.text = self.emblems.label(self.city)
        elif was is not None:
            self.note.text = ""

    def _move(self, dt):
        """Move the target camera by the input and let the drawn one follow it (mapview.follow): the wheel zooms
        toward the pointer, a drag holds the ground under the pointer and glides on when released, the keys
        zoom, pan and turn. A camera moved elsewhere (a flight, ENTER on a group) becomes the target."""
        g, inp = self.menu.game, self.menu.engine.input
        cam = g.map
        w, h = inp.viewport_size
        w, h = max(w, 1), max(h, 1)
        if self.target is None or cam != self.drawn:
            self.target, self.glide = cam, (0.0, 0.0)
        t = self.target
        wheel = inp.wheel_delta[1]
        if wheel:
            t = mapview.zoom_at(t, math.exp(mapview.WHEEL_LOG * wheel), *inp.mouse_pos, w, h, seen=cam)
        held = inp.is_pressed("s") - inp.is_pressed("w")
        if held:
            t = mapview.zoom(t, math.exp(held * self.ZOOM * dt))
        if inp.is_mouse_pressed(1):
            dx, dy = inp.mouse_delta
            gx, gz = 0.0, 0.0
            if dx or dy:
                px, py = inp.mouse_pos
                moved = mapview.drag(cam, (px - dx, py - dy), (px, py), w, h)
                gx, gz = moved.x - cam.x, moved.z - cam.z
                cam = mapview.MapCam(moved.x, moved.z, cam.h, cam.yaw)  # the ground stays under the pointer
                t = mapview.MapCam(t.x + gx, t.z + gz, t.h, t.yaw)
            k = math.exp(-dt / self.GRIP)  # the glide's speed: the drag's last moments, nothing once held still
            self.glide = (self.glide[0] * k + gx / max(dt, 1e-3) * (1 - k), self.glide[1] * k + gz / max(dt, 1e-3) * (1 - k))
        elif self.glide != (0.0, 0.0):
            vx, vz = self.glide
            t = mapview.MapCam(t.x + vx * dt, t.z + vz * dt, t.h, t.yaw)
            k = math.exp(-dt / mapview.GLIDE_S)
            self.glide = (vx * k, vz * k) if math.hypot(vx, vz) * k > t.h * 0.01 else (0.0, 0.0)
        ax = inp.is_pressed("ArrowRight") - inp.is_pressed("ArrowLeft")
        ay = inp.is_pressed("ArrowDown") - inp.is_pressed("ArrowUp")
        if ax or ay:  # the arrows move the view, so the ground goes the other way
            t = mapview.pan(t, -ax * self.PAN_PX * dt, -ay * self.PAN_PX * dt, h)
        turn = inp.is_pressed("e") - inp.is_pressed("q")
        if turn:
            t = mapview.MapCam(t.x, t.z, t.h, t.yaw + turn * self.TURN * dt)
        self.target = t
        g.map = self.drawn = mapview.follow(cam, t, dt)


@functools.cache
def _towns():
    """The map's cities and towns (contravia/towns.py), read once."""
    return towns.load()


TILE_W, TILE_H, TILE_GAP = 205, 92, 12  # a province's tile in the album
PHOTO_W, PHOTO_H, PHOTO_GAP = 200, 112, 14  # a place's photo in a province's page
PAGE_COLS, PAGE_ROWS = 5, 3


def _tile(children, x, y, w, h):
    from manifoldx.gui import Panel

    return Panel(children, anchor="top-left", offset=(x, y),
                 style_overrides={"padding": "8 12", "gap": 4, "width": w, "height": h, "bg": ui.alpha(ui.INK, 0.92),
                                  "border": 3, "border_color": "#00000000"})


def _dim(width):
    from manifoldx.gui import Panel

    return Panel([], anchor="top-left", style_overrides={"bg": ui.alpha("#000000", 0.82), "width": int(width),
                                                          "height": ui.DESIGN_H})


class AlbumScreen(Screen):
    """The album: a tile per province with how many of its places the player has photographed, and on the right
    the total and the count by kind of place. The arrows choose a province, ENTER opens its page, Escape goes
    back to the pause or the main menu."""

    COLS = 4
    back_to = "main"

    def build(self, width):
        from manifoldx.gui import Image, Panel

        if self.menu.current_before in ("pause", "main"):
            self.back_to = self.menu.current_before
        g = self.menu.game
        self.rows = album.tally(g.places, g.album.has, lambda p: p.province)
        self.index = getattr(self, "index", 0)
        self.panels = [_dim(width), Panel([_text("ÁLBUM", ui.TITLE, 50, ui.PAPER)], anchor="top-left",
                                          offset=(60, 40), style_overrides=CLEAR)]
        self.tiles = []
        for k, (name, got, total) in enumerate(self.rows):
            x, y = 60 + (k % self.COLS) * (TILE_W + TILE_GAP), 130 + (k // self.COLS) * (TILE_H + TILE_GAP)
            bar = Image(ui.progress(TILE_W - 30, 6, got / total), style_overrides={"width": TILE_W - 30, "height": 6})
            tile = _tile([_text(name.upper(), ui.TITLE, 17, ui.PAPER),
                          _text(f"{got} / {total}", ui.BOLD, 16, ui.YELLOW if got else ui.GREY), bar],
                         x, y, TILE_W, TILE_H)
            self.tiles.append(tile)
        self.panels += self.tiles
        have = sum(r[1] for r in self.rows)
        kinds = album.tally(g.places, g.album.has, lambda p: placemarks.glyph(p.type))
        lines = [_text(f"{have} de {len(g.places)} lugares", ui.TITLE, 24, ui.PAPER),
                 _text("POR TIPO", ui.BOLD, 13, ui.GREY)]
        for kind, got, total in kinds:
            lines.append(Panel([Image(placemarks.icon(kind, bool(got), 20), style_overrides={"width": 20}),
                                _text(f"{KIND_NAMES[kind]}  {got} / {total}", ui.BOLD, 15, ui.PAPER)],
                               style_overrides={**CLEAR, "direction": "h", "gap": 8}))
        self.panels.append(Panel(lines, anchor="top-right", offset=(56, 130),
                                 style_overrides={"gap": 6, "padding": "14 18", "bg": ui.alpha(ui.INK, 0.92)}))
        self.panels.append(_hints(width, [("←↑↓→", "Elegir"), ("ENTER", "Ver"), ("ESC", "Volver")]))
        self.select(self.index)

    def select(self, k):
        self.index = k
        for j, tile in enumerate(self.tiles):
            tile.style_overrides = {**tile.style_overrides, "border_color": ui.YELLOW if j == k else "#00000000"}

    def key(self, k):
        if k in ("left", "right", "up", "down"):
            self.select(album.grid_step(self.index, k, self.COLS, len(self.rows)))
        elif k == "enter":
            self.menu.province = self.rows[self.index][0]
            self.menu.screens.pop("province", None)
            self.menu.show("province")
        elif k == "back":
            if self.back_to == "pause":
                self.menu.show("pause")
            else:
                self.menu.to_menu()


class ProvinceScreen(Screen):
    """A province's page of the album: its places in a grid, the photographed first, each with its photo or a
    grey gap with its kind and municipio, and its title under it. ENTER shows the card, Escape the album."""

    def build(self, width):
        from manifoldx.gui import Image, Panel

        g = self.menu.game
        self.places = sorted((p for p in g.places if p.province == self.menu.province),
                             key=lambda p: (not g.album.has(p.id), p.title))
        self.index = getattr(self, "index", 0)
        per = PAGE_COLS * PAGE_ROWS
        self.page = self.index // per
        got = sum(g.album.has(p.id) for p in self.places)
        pages = (len(self.places) + per - 1) // per
        self.panels = [_dim(width), Panel([_text(self.menu.province.upper(), ui.TITLE, 44, ui.PAPER),
                                           _text(f"{got} / {len(self.places)} · página {self.page + 1} de {pages}",
                                                 ui.BOLD, 15, ui.YELLOW)],
                                          anchor="top-left", offset=(60, 34), style_overrides={**CLEAR, "gap": 2})]
        self.tiles = []
        for k, p in enumerate(self.places[self.page * per:(self.page + 1) * per]):
            x = 60 + (k % PAGE_COLS) * (PHOTO_W + PHOTO_GAP)
            y = 128 + (k // PAGE_COLS) * (PHOTO_H + 54)
            if g.album.has(p.id):
                face = Image(ui.photo(g.album.photo_path(p.id), PHOTO_W, PHOTO_H),
                             style_overrides={"width": PHOTO_W, "height": PHOTO_H})
            else:
                face = Panel([Image(placemarks.icon(p.type, False, 28), style_overrides={"width": 28}),
                              _text(p.type.replace("_", " "), ui.BOLD, 13, ui.PAPER),
                              _text(p.municipio, ui.BOLD, 12, ui.GREY)],
                             style_overrides={"padding": "14 12", "gap": 4, "bg": "#3a3a3a",
                                              "width": PHOTO_W, "height": PHOTO_H})
            title = p.title if len(p.title) <= 26 else p.title[:25] + "…"
            tile = Panel([face, _text(title, ui.BOLD, 13, ui.PAPER)], anchor="top-left", offset=(x, y),
                         style_overrides={"padding": 3, "gap": 4, "bg": "#00000000", "border": 3,
                                          "border_color": "#00000000"})
            self.tiles.append(tile)
        self.panels += self.tiles
        self.panels.append(_hints(width, [("←↑↓→", "Elegir"), ("ENTER", "Ficha"), ("ESC", "Álbum")]))
        self.select(self.index)

    def select(self, k):
        per = PAGE_COLS * PAGE_ROWS
        self.index = k
        if k // per != self.page:  # another page: build it
            self.menu.show("province")
            return
        for j, tile in enumerate(self.tiles):
            on = j == k % per
            tile.style_overrides = {**tile.style_overrides, "border_color": ui.YELLOW if on else "#00000000"}

    def key(self, k):
        if k in ("left", "right", "up", "down"):
            self.select(album.grid_step(self.index, k, PAGE_COLS, len(self.places)))
        elif k == "enter" and self.places:
            self.menu.view_card(self.places[self.index], back="province")
        elif k == "back":
            self.menu.show("album")


KIND_NAMES = {"iglesia": "Iglesias", "fortaleza": "Fortalezas", "playa": "Playas y agua", "museo": "Museos y edificios",
              "monumento": "Monumentos", "plaza": "Plazas y parques", "teatro": "Teatros y estadios",
              "estacion": "Estaciones y obras", "naturaleza": "Naturaleza", "lugar": "Otros"}


OPTION_TEXT = {None: "Ninguno", True: "Sí", False: "No", "round_pod": "Redondos en el guardafango", "round": "Redondos",
               "rect": "Rectangulares", "almond": "Almendra", "bar": "Barra", "mesh": "Rejilla", "skin": "De carrocería",
               "chrome_bar": "Cromo", "blade": "Lámina", "rubber": "Goma", "pill": "Píldora", "fin": "En la aleta",
               "block": "Bloque", "spokes": "Rayos", "hubcap": "Tapacubos", "dome": "Domo", "twin": "Dobles",
               "over": "Sobre el motor", "forward": "Con trompa", "none": "Ninguna", "cargo_box": "Cajón",
               "bench": "Banco", "saddle": "Sillín", "long": "Largo", "pedal": "Pedales", "engine": "Motor",
               "hub": "Eléctrico", "delta": "Delta", "inline": "En línea", "tadpole": "Renacuajo"}


def _number(v: float, unit: str) -> str:
    return f"{v:.2f}".replace(".", ",") + (f" {unit}" if unit in ("m",) else "")


def option_text(key: str, v) -> str:
    """How the Garaje shows a knob's value."""
    if key in ("paint", "second"):
        return designs.colour_name(v)
    if isinstance(v, str) or v is None or isinstance(v, bool):
        return OPTION_TEXT.get(v, str(v))
    if isinstance(v, int):
        return str(v)
    return _number(float(v), "m")


def facts(design) -> list[str]:
    """The card's lines: the size and what the catalogue says of it."""
    p, pre = design.params(), design.preset
    if pre.family == "cycle":
        size = f"Batalla {_number(p.wheelbase, 'm')}"
    else:
        size = f"{_number(p.length, 'm')} de largo · {_number(p.height, 'm')} de alto"
    return [size, f"{pre.seats} plazas" if pre.seats > 1 else "1 plaza", f"Anda por: {pre.where}"]


class GarageScreen(Screen):
    """The Garaje's Salón: the family tabs, the family's vehicles as rows, a card on the right, and the chosen
    vehicle turning in the street behind. Up and down choose, Q and E change family, ENTER drives, TAB designs, a
    drag of the mouse turns the vehicle by hand, Escape goes back where the Garaje was opened from."""

    TABS: ClassVar[tuple] = (*designs.TABS, "Mis diseños")

    def build(self, width):
        from manifoldx.gui import Panel

        Row = _row_class()
        m = self.menu
        if m.current_before not in ("garage", "design"):
            self.back_to = "pause" if m.current_before == "pause" else "main"
            self.tab = m.chosen.preset.tab if m.chosen and m.chosen.name == "" else self.TABS[0]
            self.index = None
        self.items = m.garage_items(self.tab)
        if self.index is None:
            self.index = next((k for k, d in enumerate(self.items) if d == m.chosen), 0)
        self.index = min(self.index, max(len(self.items) - 1, 0))
        self.panels = [Panel([], anchor="top-left", style_overrides={"bg": ui.alpha(ui.INK, 0.72), "width": 560,
                                                                      "height": ui.DESIGN_H})]
        self.panels.append(Panel([_text("GARAJE", ui.TITLE, 50, ui.PAPER)], anchor="top-left", offset=(60, 40),
                                 style_overrides=CLEAR))
        tabs = []
        for tab in self.TABS:
            on = tab == self.tab
            tabs.append(Row([_text(tab.upper(), ui.BOLD, 14, ui.INK if on else ui.GREY)],
                            (lambda tab=tab: self.go_tab(tab)),
                            style_overrides={"padding": "5 10", "bg": ui.PAPER if on else ui.alpha(ui.INK, 0.8)}))
        self.panels.append(Panel(tabs, anchor="top-left", offset=(60, 104),
                                 style_overrides={**CLEAR, "direction": "h", "gap": 4}))
        self.frames, rows = [], []
        for k, d in enumerate(self.items):
            mark = Panel([], style_overrides={"bg": ui.RED, "width": 4, "height": 44, "padding": 0})
            r = Row([mark, Panel([_text((d.name or d.preset.name).upper(), ui.BOLD, 17, ui.PAPER)],
                                 style_overrides={**CLEAR, "padding": "11 0 0 14"}),
                     Panel([], style_overrides={**CLEAR, "flex": 1}),
                     Panel([_text(d.preset.era.upper(), ui.BOLD, 13, ui.GREY)],
                           style_overrides={**CLEAR, "padding": "13 14 0 0"})],
                    (lambda k=k: self.click(k)),
                    style_overrides={"direction": "h", "padding": 0, "gap": 0, "bg": ui.alpha(ui.INK, 0.82),
                                     "border": 2, "border_color": "#00000000", "width": 470, "height": 48})
            r.mark = mark
            self.frames.append(r)
            rows.append(r)
        if not rows:
            rows = [_text("Todavía no hay diseños: TAB en un vehículo abre el Taller.", ui.BOLD, 15, ui.PAPER)]
        self.panels.append(Panel(rows, anchor="top-left", offset=(60, 150), style_overrides={**CLEAR, "gap": 6}))
        self.name = _text("", ui.TITLE, 30, ui.PAPER)
        self.kind = _text("", ui.BOLD, 14, ui.GREY)
        self.lines = [_text("", ui.BOLD, 16, ui.PAPER) for _ in range(3)]
        self.status = _text("", ui.BOLD, 14, ui.YELLOW)
        self.card = Panel([Panel([], style_overrides={"bg": ui.YELLOW, "height": 4, "padding": 0}),
                           Panel([self.name, self.kind, *self.lines, self.status],
                                 style_overrides={**CLEAR, "padding": "12 16 16 16", "gap": 4})],
                          anchor="top-right", offset=(56, 150),
                          style_overrides={"padding": 0, "gap": 0, "bg": ui.alpha(ui.INK, 0.9), "width": 380})
        self.panels.append(self.card)
        keys = [("↑↓", "Vehículo"), ("Q E", "Familia"), ("ENTER", "Manejar"), ("TAB", "Diseñar")]
        if self.tab == "Mis diseños":
            keys.append(("SUPR", "Borrar"))
        self.panels.append(_hints(width, [*keys, ("ESC", "Volver")], "Arrastra para girar"))
        m.game.open_garage()
        self.select(self.index)

    def select(self, k: int) -> None:
        if not self.items:
            self.name.text, self.kind.text, self.status.text = "MIS DISEÑOS", "", ""
            for t in self.lines:
                t.text = ""
            return
        self.index = k % len(self.items)
        for j, r in enumerate(self.frames):
            on = j == self.index
            r.style_overrides = {**r.style_overrides, "border_color": ui.YELLOW if on else "#00000000"}
            r.mark.style_overrides = {**r.mark.style_overrides, "opacity": 1.0 if on else 0.0}
        d = self.items[self.index]
        self.menu.show_design(d)
        self.name.text = (d.name or d.preset.name).upper()
        self.kind.text = f"{d.preset.tab} · {d.preset.era}" + (f" · a partir de {d.preset.name}" if d.name else "")
        for t, line in zip(self.lines, facts(d), strict=True):
            t.text = line
        self.status.text = "" if d.preset.drivable else "Se maneja cuando tenga ciclista."
        ui.refit(self.card)

    def go_tab(self, tab: str) -> None:
        self.tab, self.index = tab, 0
        self.menu.show("garage")

    def click(self, k: int) -> None:
        self.select(k)

    def key(self, k):
        m = self.menu
        if k in ("up", "down") and self.items:
            self.select(self.index + (1 if k == "down" else -1))
        elif k in ("prev", "next"):
            j = self.TABS.index(self.tab) + (1 if k == "next" else -1)
            self.go_tab(self.TABS[j % len(self.TABS)])
        elif k == "enter" and self.items:
            d = self.items[self.index]
            if not d.preset.drivable:
                self.status.text = "Los ciclos se manejan cuando tengan ciclista."
                return
            m.drive_design(d, self.back_to)
        elif k == "design" and self.items:
            m.design = self.items[self.index]
            m.undo = []
            m.show("design")
        elif k == "delete" and self.tab == "Mis diseños" and self.items:
            garage.delete_design(self.items[self.index])
            m.designs = garage.load_designs()
            m.show("garage")
        elif k == "back":
            m.leave_garage(self.back_to)

    def tick(self, dt):
        self.menu.turn_vehicle(dt)


class KnobRow:
    """A row of the Taller: one knob of the design being edited, turned with left and right through designs.turn."""

    def __init__(self, menu, key: str, label: str):
        self.menu, self.key, self.label = menu, key, label

    @property
    def text(self) -> str:
        return option_text(self.key, self.menu.design.value(self.key))

    @property
    def help(self) -> str:
        m = self.menu
        if m.note:
            return m.note
        r = designs.street_ranges(m.design.preset.family)[self.key]
        if "choices" in r:
            return f"{len(r['choices'])} opciones: cambia con ← y →."
        lo, hi = designs.limits(m.design, self.key)
        unit = r["unit"] if r["unit"] == "m" else ""
        return (f"De {_number(r['min'], unit)} a {_number(r['max'], unit)}; con las otras medidas, entre "
                f"{_number(lo, unit)} y {_number(hi, unit)}.")

    def step(self, d: int) -> None:
        m = self.menu
        new, why = designs.turn(m.design, self.key, d)
        if new != m.design:
            m.undo.append(m.design)
            m.design, m.saved = new, False
        m.note = why


class DesignScreen(RowsScreen):
    """The Garaje's Taller: one group of the design's knobs as rows (Q and E change group), a card with its name
    and whether it is saved, and the vehicle rebuilt behind as the rows turn. R draws a random design, Z undoes,
    G saves to Mis diseños, ENTER drives it, Escape goes back to the Salón."""

    title = "Taller"
    hints: ClassVar[list] = [("↑↓", "Fila"), ("←→", "Cambiar"), ("Q E", "Grupo"), ("R", "Al azar"),
                             ("Z", "Deshacer"), ("G", "Guardar"), ("ENTER", "Manejar"), ("ESC", "Salón")]

    def rows(self):
        m = self.menu
        group = getattr(self, "group", designs.GROUPS[0])
        return [KnobRow(m, k, label) for k, label, _ in designs.knobs(m.design.preset.family, group)]

    def background(self, width):
        from manifoldx.gui import Panel

        if self.menu.current_before not in ("garage", "design"):
            self.group = designs.GROUPS[0]
        groups = [g for g in designs.GROUPS if designs.knobs(self.menu.design.preset.family, g)]
        self.group = getattr(self, "group", groups[0]) if getattr(self, "group", None) in groups else groups[0]
        self.groups = groups
        self.menu.game.open_garage()
        return [Panel([], anchor="top-left", style_overrides={"bg": ui.alpha(ui.INK, 0.72), "width": 580,
                                                               "height": ui.DESIGN_H})]

    def extra(self, width):
        from manifoldx.gui import Panel

        Row = _row_class()
        tabs = []
        for g in self.groups:
            on = g == self.group
            tabs.append(Row([_text(g.upper(), ui.BOLD, 14, ui.INK if on else ui.GREY)],
                            (lambda g=g: self.go_group(g)),
                            style_overrides={"padding": "5 10", "bg": ui.PAPER if on else ui.alpha(ui.INK, 0.8)}))
        self.name = _text("", ui.TITLE, 28, ui.PAPER)
        self.base = _text("", ui.BOLD, 14, ui.GREY)
        self.status = _text("", ui.BOLD, 14, ui.YELLOW)
        self.card = Panel([Panel([], style_overrides={"bg": ui.YELLOW, "height": 4, "padding": 0}),
                           Panel([self.name, self.base, self.status],
                                 style_overrides={**CLEAR, "padding": "12 16 16 16", "gap": 4})],
                          anchor="top-right", offset=(56, 130),
                          style_overrides={"padding": 0, "gap": 0, "bg": ui.alpha(ui.INK, 0.9), "width": 380})
        return [Panel(tabs, anchor="top-left", offset=(250, 58), style_overrides={**CLEAR, "direction": "h", "gap": 4}),
                self.card]

    def go_group(self, g: str) -> None:
        self.group, self.index = g, 0
        self.menu.note = ""
        self.menu.show("design")

    def changed(self):
        m = self.menu
        if not hasattr(self, "card"):
            return
        m.show_design(m.design)
        d = m.design
        self.name.text = (d.name or f"Mi {d.preset.name.lower()}").upper()
        self.base.text = f"A partir de: {d.preset.name}"
        self.status.text = "Guardado en Mis diseños" if m.saved else "Sin guardar"
        ui.refit(self.card)
        lines = textwrap.wrap(self.rows()[self.index].help, 62)[:3]
        for t, line in zip(self.help, lines + [""] * 3, strict=False):
            t.text = line

    def refresh(self) -> None:
        """Show every row's value again, after a change that was not one row's turn."""
        for row, (value, panel) in zip(self.rows(), self.values, strict=True):
            value.text = f"‹  {row.text.upper()}  ›"
            ui.refit(panel)
        self.changed()

    def key(self, k):
        m = self.menu
        if k in ("left", "right", "up", "down"):
            if k in ("up", "down"):
                m.note = ""
            super().key(k)
        elif k in ("prev", "next"):
            j = self.groups.index(self.group) + (1 if k == "next" else -1)
            self.go_group(self.groups[j % len(self.groups)])
        elif k == "random":
            m.undo.append(m.design)
            m.design, m.saved, m.note = m.random_design(m.design), False, ""
            self.refresh()
        elif k == "undo":
            if m.undo:
                m.design, m.saved, m.note = m.undo.pop(), False, ""
            else:
                m.note = "Nada que deshacer."
            self.refresh()
        elif k == "save":
            m.save_design()
            self.changed()
        elif k == "enter":
            if not m.design.preset.drivable:
                m.note = "Los ciclos se manejan cuando tengan ciclista."
                self.changed()
                return
            back = self.menu.screens["garage"].back_to if "garage" in self.menu.screens else "main"
            m.drive_design(m.design, back)
        elif k == "back":
            m.note = ""
            m.show("garage")

    def tick(self, dt):
        self.menu.turn_vehicle(dt)


SCREENS = {"main": MainScreen, "free": FreeScreen, "options": OptionsScreen, "pause": PauseScreen,
           "photo": PhotoScreen, "aim": AimScreen, "card": CardScreen, "map": MapScreen, "album": AlbumScreen, "province": ProvinceScreen, "garage": GarageScreen, "design": DesignScreen}
KEY_NAMES = set(KEYS) | {"l"}  # L, while driving, turns the driving help to its next level


class Menu:
    """The menus over a Game: which screen shows, the keys, and the moves between menus and driving."""

    def __init__(self, engine, game, free: FreeDrive, options: Options, screen: str | None = "main"):
        self.engine, self.game, self.free, self.options = engine, game, free, options
        self.screens, self.current, self.current_before, self._size = {}, None, None, None
        # The Garaje: what is driven (None: the sedan the game starts with), what the Taller edits, what the
        # game shows now and what it should show, and the player's saved designs.
        self.builder = garage.Builder()
        self.default_model = game.model
        self.chosen, self.design, self.target, self.shown = None, None, None, None
        self.province = ""  # the province whose album page shows
        self.card = None  # the last photographed place's card: {"place", "texture", "meta", "count"}
        self.undo, self.saved, self.note = [], False, ""
        self.designs = garage.load_designs()
        self.refresh_vehicles()
        engine.on("shutdown")(lambda _=None: self.builder.close())
        game.apply_options(options)
        self._first = screen
        if screen:
            game.to_menu()
        engine.on("startup")(self._startup)
        engine.on("map")(self._map_event)
        engine.system(self.update)

    def _map_event(self, p: dict) -> None:
        """Open the map (over the pause when driving) with its camera over (x, z) or (lat, lon), h metres up
        (as it was by default), turned yaw degrees from north (0 by default): for an agent checking the map."""
        g = self.game
        if self.current != "map":
            if self.current is None:
                self.pause()
            self.show("map")
        if g.map is None:
            return
        if "lat" in p:
            from contravia.geo import to_local

            x, z = to_local(float(p["lat"]), float(p["lon"]))
        else:
            x, z = float(p.get("x", g.map.x)), float(p.get("z", g.map.z))
        h = min(mapview.MAX_H, max(mapview.MIN_H, float(p.get("h", g.map.h))))
        g.flight = None  # no flight from the car: the camera is put there
        g.map = mapview.MapCam(float(x), float(z), h, math.radians(float(p.get("yaw", 0.0))))

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
        if self.game.curtain is not None:  # a mode starting under the curtain: Escape goes back to the main menu
            if key == "Escape":
                self.game.drop_curtain()
                if self.current == "map":
                    self.screens["map"].key("back")
                else:
                    self.to_menu()
            return
        if self.current is None:
            if key == "Escape" and self.game.mode == "drive":
                self.pause()
            elif key == "f" and self.game.mode == "drive" and self.game.start_aim():
                self.show("aim")
            elif key == "l" and self.game.mode == "drive":
                self.next_help()
            return
        if key in KEYS:
            self.screens[self.current].key(KEYS[key])

    def next_help(self) -> None:
        """Turn the «Ayuda al manejo» option to its next level, as choosing it in the options does: applied at
        once, saved, and named for a moment on the HUD."""
        row = self.options["help"]
        row.pick(next_level(row.value))
        self.game.apply_options(self.options)
        self.options.save()
        self.game.hud.note(f"Ayuda {row.value} · {row.text}".upper())

    def update(self, query, dt: float) -> None:
        """Read this frame's keys, follow a resized window and animate the screen."""
        inp = self.engine.input
        if self.game.curtain is not None:  # a mode starting under the curtain: only Escape counts (press)
            if inp.just_pressed("Escape"):
                self.press("Escape")
            return
        for k in KEY_NAMES:
            if inp.just_pressed(k):
                if self.current in ("photo", "map", "aim") and k in ("w", "s", "a", "d", "ArrowUp", "ArrowDown",
                                                            "ArrowLeft", "ArrowRight"):
                    continue  # the photo mode reads them as held keys
                if self.current is None and k not in ("Escape", "f", "l"):
                    continue
                self.press(k)
                break
        if self.current is None:
            return
        if self._size != (self.engine.w, self.engine.h):
            self.show(self.current)
        self.screens[self.current].tick(min(dt, 0.1))

    def start_free(self) -> None:
        """Drive with the free drive's rows, started under the loading curtain."""
        self.show(None)

        def start():
            self.chosen = self.free["vehicle"].value
            self._put(self.chosen)
            self.game.start(**self.free.start())

        self.game.raise_curtain(start)

    def open_map(self) -> None:
        """The map from the main menu, under the loading curtain till its first view is drawn."""
        self.show("map")
        self.game.raise_curtain()

    # The Garaje

    def refresh_vehicles(self) -> None:
        """The free drive's Vehículo row: the sedan, every drivable preset and every drivable saved design."""
        row = self.free["vehicle"]
        row.options = [("Sedán", None)] + [(p.name, designs.Design(k)) for k, p in fleet.catalogue().items()
                                           if p.drivable]
        row.options += [(d.name, d) for d in self.designs if d.preset.drivable]
        row.index = 0
        row.pick(self.chosen)

    def garage_items(self, tab: str) -> list:
        """The Salón's rows for a tab: its presets, or the saved designs."""
        if tab == "Mis diseños":
            return list(self.designs)
        return [designs.Design(k) for k, p in fleet.catalogue().items() if p.tab == tab]

    def _key(self, design) -> str:
        if design is None:
            return "default"
        return garage.key(design) + repr(sorted(design.palette().items()))

    def _put(self, design) -> None:
        """Show (and drive) `design` now, building it here if it has to be; None is the starting sedan."""
        if self._key(design) == self.shown:
            return
        if design is None:
            self.game.set_vehicle(self.default_model, family="moderno")
        else:
            self.game.set_vehicle(self.builder.now(design), design.palette(), family=design.base)
        self.shown = self._key(design)

    def show_design(self, design) -> None:
        """Turn the Garaje's camera round `design`, once it is built."""
        self.target = design

    def turn_vehicle(self, dt: float) -> None:
        """The Garaje's frame: the camera goes round (or follows a drag of the mouse), and the vehicle wanted
        replaces the one shown as soon as it is built."""
        g, inp = self.game, self.engine.input
        yaw, pitch, _ = g.photo or (g.car.yaw, 0.22, 9.0)
        if inp.is_mouse_pressed(1):
            yaw += inp.mouse_delta[0] * 0.01
        else:
            yaw += TURN * dt
        d = self.target
        if d is not None and self._key(d) != self.shown:
            try:
                vehicle = self.builder.want(d)
            except ValueError:
                vehicle, self.note = None, "Así no se puede construir: vuelve atrás con Z."
                self.target = None
            if vehicle is not None:
                g.set_vehicle(vehicle, d.palette(), family=d.base)
                self.shown = self._key(d)
        p = (d or self.chosen or designs.Design("ruso")).params()
        length = getattr(p, "length", 2.2)
        g.photo = (yaw, pitch, max(5.5, 1.5 * length + 2.5))

    def drive_design(self, design, back_to: str) -> None:
        """Drive `design`: from the pause, on from where the car stood; from the main menu, through the free
        drive's screen with it in the Vehículo row."""
        self._put(design)
        self.chosen = design
        self.refresh_vehicles()
        if back_to == "pause":
            self.resume()
        else:
            self.game.photo = None
            self.game.to_menu()
            self.show("free")

    def leave_garage(self, back_to: str) -> None:
        """Back from the Salón without driving: the vehicle driven before comes back."""
        self._put(self.chosen)
        self.target, self.game.photo = None, None
        if back_to == "pause":
            self.show("pause")
        else:
            self.to_menu()

    def random_design(self, design):
        """A design of the same preset with every knob drawn from its range, as the fleet would draw one."""
        rng = self.game._rng
        r = designs.street_ranges(design.preset.family)
        out = designs.Design(design.base, design.name)
        for key, _, _ in designs.knobs(design.preset.family):
            spec = r[key]
            if "choices" in spec:
                v = spec["choices"][int(rng.integers(len(spec["choices"])))]
            else:
                steps = round((spec["max"] - spec["min"]) / spec["step"])
                v = round(spec["min"] + int(rng.integers(steps + 1)) * spec["step"], 4)
            if key in ("paint", "second", "livery"):
                setattr(out, key, v)
            else:
                out.changes[key] = v
        return designs.fix(out)

    def save_design(self) -> None:
        """Keep the Taller's design in Mis diseños, naming it "Mi <vehículo> <n>" the first time."""
        from dataclasses import replace

        d = self.design
        if not d.name:
            d = replace(d, name=garage.new_name(d.preset.name, [x.name for x in self.designs]))
        garage.save_design(d)
        self.design, self.saved, self.note = d, True, ""
        self.designs = garage.load_designs()
        self.refresh_vehicles()

    def took(self, framing) -> None:
        """A good photo of the place aimed at: take it and show the place's card over the stopped game."""
        g = self.game
        place = g.aim["place"]
        texture, when, weather = g.take_photo()
        total = sum(p.province == place.province for p in g.places)
        self.card = {"place": place, "texture": texture, "meta": " · ".join(
            x for x in (place.type.replace("_", " "), place.municipio, when, weather) if x),
            "count": f"NUEVO · {g.album.count(place.province)} de {total} en {place.province}"}
        self.screens.pop("card", None)
        self.show("card")

    def view_card(self, place, back: str = "map") -> None:
        """A place's card from the map or the album: its photo if it has one, how many of its province the player
        has, and when the photo was taken. ENTER or Escape goes back to the screen `back`."""
        g = self.game
        total = sum(p.province == place.province for p in g.places)
        when = ()
        if g.album.has(place.id):
            e = g.album.entry(place.id)
            when = (e.get("game_time", ""), e.get("weather", ""))
            texture = ui.photo(g.album.photo_path(place.id), 480, 270)
        else:
            texture = ui.no_photo(480, 270)
        self.card = {"place": place, "texture": texture, "back": back, "meta": " · ".join(
            x for x in (place.type.replace("_", " "), place.municipio, *when) if x),
            "count": f"{g.album.count(place.province)} de {total} en {place.province}"}
        self.screens.pop("card", None)
        self.show("card")

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
