"""The first run's window, as docs/ux/contravia-ux.html draws it ("Primera vez"): "Construyendo Cuba" over
a dark poster panel, the four steps, a bar for the whole build, the time left and the disk it will take.

It runs before the game, in its own window: the game cannot open until the city exists. contravia.firstrun
does the work on a thread; this screen reads its progress every frame. At the end it says how long the build
took and how much it takes on disk, and Enter (or a click on "Jugar") closes the window for the game to open.
Escape stops: the build goes on from its last finished step the next time.
"""

import textwrap
import threading
import time
from functools import cache

from contravia import ui
from contravia.firstrun import STEPS, Build
from contravia.menu import CLEAR, _hints, _row_class, _text
from contravia.menu import KEYS as KEY_MAP

DIM = 0.25  # the poster panel's brightness behind the text
DISK_MB = 320  # what the built city takes, measured on the full Havana build
FRAME_W = 760  # the column the screen is read in
TOP = 150
BAR_H = 8
SUB = ("Tu computadora arma la ciudad a partir del mapa: calles, puentes, edificios y aceras. Se hace una "
       "sola vez; después juegas sin esperar.")
STOPPED = 3  # exit code when the player stops the build with Escape


@cache
def _dark_panel(name: str, k: float) -> str:
    """A poster panel at k of its brightness, as SVG markup that holds the JPEG (see ui.panel)."""
    import base64
    import io

    from PIL import Image as Pil
    from PIL import ImageEnhance

    img = ImageEnhance.Brightness(Pil.open(ui.PANELS / f"{name}.jpg").convert("RGB")).enhance(k)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    data = base64.b64encode(buf.getvalue()).decode()
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{img.width}" height="{img.height}">'
            f'<image width="{img.width}" height="{img.height}" href="data:image/jpeg;base64,{data}"/></svg>')


def _minutes(s: float) -> str:
    if s < 60:
        return f"{max(5, round(s / 5) * 5):.0f} s"
    return f"{round(s / 60):.0f} min"


class BuildScreen:
    """The build's screen: its root panels, rebuilt for the window's width, and refreshed from the build."""

    def __init__(self, host, build: Build):
        self.host, self.build = host, build
        self.panels, self._shown = [], None

    def make(self, width: float) -> None:
        from manifoldx.gui import Image, Panel

        Row = _row_class()
        self.width = width
        x = int(max(40, (width - FRAME_W) / 2))
        cover = ui.cover_uv(ui.panel_aspect("malecon"), width / ui.DESIGN_H)
        self.panels = [
            Panel([Image(_dark_panel("malecon", DIM), style_overrides={"width": int(width), "height": ui.DESIGN_H, "uv": cover})],
                  anchor="top-left", style_overrides={**CLEAR, "width": int(width), "height": ui.DESIGN_H}),
            Panel([Image(ui.logo(), style_overrides={"width": 300})], anchor="top-left", offset=(x, TOP - 70),
                  style_overrides=CLEAR),
        ]
        self.title = _text(" ", ui.TITLE, 50, ui.PAPER)
        self.sub = [_text(" ", ui.BOLD, 18, ui.PAPER) for _ in range(3)]
        self.panels.append(Panel([self.title], anchor="top-left", offset=(x, TOP), style_overrides=CLEAR))
        self.panels.append(Panel(self.sub, anchor="top-left", offset=(x, TOP + 70),
                                 style_overrides={**CLEAR, "gap": 0, "width": FRAME_W, "height": 66}))
        self.marks, self.steps = [], []
        cells = []
        for _ in STEPS:
            mark, name = _text(" ", ui.KEYS, 15, ui.PAPER), _text(" ", ui.BOLD, 16, ui.GREY)
            self.marks.append(mark), self.steps.append(name)
            cells.append(Panel([mark, name], style_overrides={**CLEAR, "direction": "h", "gap": 4, "padding": "0 22 0 0"}))
        self.ready = _text("LISTO", ui.BOLD, 16, ui.GREY)
        cells.append(Panel([self.ready], style_overrides=CLEAR))
        self.row = Panel(cells, anchor="top-left", offset=(x, TOP + 140), style_overrides={**CLEAR, "direction": "h"})
        self.panels.append(self.row)
        self.panels.append(Panel([Image(ui.bar(FRAME_W, BAR_H, "#333333"), style_overrides={"width": FRAME_W, "height": BAR_H})],
                                 anchor="top-left", offset=(x, TOP + 178), style_overrides={**CLEAR, "width": FRAME_W, "height": BAR_H}))
        self.fill_img = Image(ui.bar(FRAME_W, BAR_H, ui.YELLOW), style_overrides={"width": 1, "height": BAR_H})
        self.fill = Panel([self.fill_img], anchor="top-left", offset=(x, TOP + 178),
                          style_overrides={**CLEAR, "width": 1, "height": BAR_H})
        self.panels.append(self.fill)
        self.left, self.right = _text(" ", ui.BOLD, 16, ui.PAPER), _text(" ", ui.BOLD, 16, ui.PAPER)
        self.panels.append(Panel([self.left], anchor="top-left", offset=(x, TOP + 196),
                                 style_overrides={**CLEAR, "width": FRAME_W - 230, "height": 24}))
        self.panels.append(Panel([self.right], anchor="top-left", offset=(x + FRAME_W - 220, TOP + 196),
                                 style_overrides={**CLEAR, "width": 220, "height": 24}))
        self.note = _text(" ", ui.BOLD, 16, ui.GREY)
        self.panels.append(Panel([self.note], anchor="top-left", offset=(x, TOP + 240),
                                 style_overrides={**CLEAR, "width": FRAME_W, "height": 50}))
        self.play_t = _text(" ", ui.TITLE, 34, ui.INK)
        self.play = Row([self.play_t], self.host.finish, anchor="top-left", offset=(x, TOP + 300),
                        style_overrides={**CLEAR, "padding": "6 0 0 20", "width": 360, "height": 72})
        self.stripe = Panel([], anchor="top-left", offset=(x, TOP + 300), style_overrides={**CLEAR, "width": 1, "height": 1})
        self.panels += [self.stripe, self.play]
        self.hints = None
        self._shown = None
        self._making = True  # the window attaches every panel in order once they are made
        self.refresh()
        self._making = False

    def _state(self) -> str:
        b = self.build
        return "error" if b.error else ("done" if b.done else "running")

    def refresh(self) -> None:
        """Put the build's progress into the panels."""
        b, state = self.build, self._state()
        if state != self._shown:
            self._layout(state)
        for k, (mark, name) in enumerate(zip(self.marks, self.steps, strict=True)):
            label = STEPS[k][1].upper()
            if k < b.step or b.done:
                mark.text, name.text, fg = "✓", label, ui.PAPER
            elif k == b.step and state == "running":
                mark.text, name.text, fg = " ", f"{label} · {b.step_fraction:.0%}".replace("%", " %"), ui.YELLOW
            else:
                mark.text, name.text, fg = " ", label, ui.GREY
            name.style_overrides = {**name.style_overrides, "fg": fg}
        self.ready.style_overrides = {**self.ready.style_overrides, "fg": ui.YELLOW if b.done else ui.GREY}
        ui.refit(self.row)
        w = max(1, round(FRAME_W * b.overall))
        self.fill.style_overrides = {**self.fill.style_overrides, "width": w}
        self.fill_img.style_overrides = {**self.fill_img.style_overrides, "width": w}
        if state == "running":
            where = (f"{b.province.upper()} · {b.p + 1} DE {len(b.provinces)}" if b.p < len(b.provinces)
                     else "RESUMEN DE LA ISLA")
            self.left.text = f"{where} · {b.overall:.0%} EN TOTAL".replace("%", " %")
            self.right.text = f"quedan unos {_minutes(b.eta)}"

    def _layout(self, state: str) -> None:
        """The texts and hints that change with the build's state."""
        from manifoldx.gui import Image

        b = self.build
        self._shown = state
        if state == "done":
            self.title.text = "CUBA ESTÁ LISTA"
            self._say(" ")
            took = time.perf_counter() - b._t0
            self.left.text = f"Construida en {_minutes(took)} · {b.mb:.0f} MB en disco. No hace falta repetirlo."
            self.right.text = " "
            self.note.text = " "
            self.play_t.text = "JUGAR ›"
            self.stripe.children = [Image(ui.stripe(360, 72), style_overrides={"width": 360, "height": 72})]
            self.stripe.style_overrides = {**self.stripe.style_overrides, "width": 360, "height": 72}
            hints = [("ENTER", "Jugar")]
        elif state == "error":
            self.title.text = "ACTUALIZANDO CUBA" if b.updating else "CONSTRUYENDO CUBA"
            self._say(f"Algo falló: {b.error}")
            self.note.text = "Vuelve a abrir el juego: sigue en el paso donde iba."
            hints = [("ESC", "Salir")]
        else:
            self.title.text = "ACTUALIZANDO CUBA" if b.updating else "CONSTRUYENDO CUBA"
            self._say(SUB)
            self.note.text = "Si cierras el juego, la próxima vez sigue en el paso donde iba."
            hints = [("ESC", "Pausar y salir")]
        right = f"Ocupará unos {DISK_MB} MB en disco" if state == "running" else ""
        if self.hints is not None and self.hints in self.panels:
            self.host.replace(self.hints, None)
            self.panels.remove(self.hints)
        self.hints = _hints(self.width, hints, right)
        self.panels.append(self.hints)
        if not self._making:
            self.host.replace(None, self.hints)

    def _say(self, text: str) -> None:
        """The paragraph under the title, wrapped to the column's three lines."""
        lines = textwrap.wrap(text, 90)[:3]
        for k, t in enumerate(self.sub):
            t.text = lines[k] if k < len(lines) else " "

    def key(self, k: str) -> None:
        if k == "enter" and self._state() == "done":
            self.host.finish()
        elif k == "back":
            self.host.stop()


class BuildWindow:
    """The window the build screen lives in: runs the build on a thread, routes the keys, follows resizes."""

    def __init__(self, engine, build: Build):
        self.engine, self.build = engine, build
        self.screen = BuildScreen(self, build)
        self.code = None  # 0 played on, STOPPED stopped
        self._size, self._on = None, []
        self.thread = threading.Thread(target=build.run, args=(False,), daemon=True)
        engine.on("startup")(self._startup)
        engine.system(self.update)

    def _startup(self, _=None) -> None:
        self.thread.start()
        self._show()

    def _show(self) -> None:
        for p in list(self._on):
            self.replace(p, None)
        self._size = (self.engine.w, self.engine.h)
        self.screen.make(ui.DESIGN_H * self.engine.w / max(self.engine.h, 1))
        for p in self.screen.panels:
            if p not in self._on:
                self.replace(None, p)

    @staticmethod
    def _fit(p):
        return p if p.anchor != "top-left" and not any(hasattr(c, "children") for c in p.children) else ui.fit(p)

    def replace(self, old, new) -> None:
        """Take panel old off the window and put new on; either may be None."""
        g = self.engine.gui
        if old is not None and old in self._on:
            g.remove(old)
            self._on.remove(old)
        if new is not None and self._size is not None:
            g.append(self._fit(new))
            self._on.append(new)

    def update(self, query, dt: float) -> None:
        inp = self.engine.input
        for name, k in KEY_MAP.items():
            if inp.just_pressed(name):
                self.screen.key(k)
                break
        if self._size != (self.engine.w, self.engine.h):
            self._show()
        self.screen.refresh()

    def finish(self) -> None:
        self.code = 0
        self.engine.quit()

    def stop(self) -> None:
        self.code = STOPPED
        self.engine.quit()


def main(source, where=None, size=(1280, 720), shot: str | None = None) -> int:
    """Open the build window; return 0 when the city is built and the player goes on, STOPPED otherwise.

    shot renders offscreen instead, saving shot-<n>.png at each step and at the end, for checking the screen."""
    import manifoldx as mx

    engine = mx.Engine("Contravía", width=size[0], height=size[1])
    window = BuildWindow(engine, Build(source, where))
    if not shot:
        engine.run()
        return window.code if window.code is not None else STOPPED
    from PIL import Image

    seen = set()
    while True:
        img = engine.render_frame(supersample=1)
        b = window.build
        mark = "done" if b.done else ("error" if b.error else b.step)
        if mark not in seen and (b.done or b.error or b.step_fraction > 0.3):
            seen.add(mark)
            Image.fromarray(img).save(f"{shot}-{mark}.png")
        if b.done or b.error:
            for _ in range(3):
                img = engine.render_frame(supersample=1)
            Image.fromarray(img).save(f"{shot}-{mark}.png")
            break
        time.sleep(0.2)
    engine.close()
    return 0 if window.build.done else STOPPED
