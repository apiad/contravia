"""The HUD, in the four corners and nothing in the middle of the street, as docs/ux/contravia-ux.html draws it.

Top left: the game's time, the day and the weather's icon. Top right: a round minimap with a cream rim, the
scene itself seen from MINIMAP_HEIGHT metres straight above the car, rendered by the engine into a texture
(render_to_texture) when the car has moved MINIMAP_MOVE metres or turned MINIMAP_TURN degrees since the last
render, or while the cells around the car are still loading, at most every MINIMAP_EVERY frames and at least
every MINIMAP_IDLE. Its top is the way the car faces, so the red arrow in its middle
always points up and the streets turn around it; the N badge rides the rim where north is. Bottom left: the
name of the municipio when the car enters it, large, fading out after AREA_SHOWN seconds. Bottom right: the
speed, the gear in yellow and a rev counter in segments, from the car's own automatic gearbox, and the four
assists' lamps, lit while an assist corrects.

F1 shows or hides a panel of numbers in the top left corner for working on the game: frames per second and
the slowest frame lately, the streamed nodes drawn by level, what is building and uploading, the traffic, and
where the car is and how its own cell stands. The GUI scales with the window from a 720-pixel-high design.
"""

import math

import numpy as np

from contravia import ui

MINIMAP_SIZE = 256  # pixels of the texture
MINIMAP_SHOWN = 190  # pixels across on screen
MINIMAP_HEIGHT = 300.0  # metres above the car
MINIMAP_EVERY = 4  # fewest frames between two renders of the minimap
MINIMAP_MOVE = 2.0  # metres the car moves before the minimap renders again, about one pixel of it on screen
MINIMAP_TURN = 1.5  # degrees the car turns before the minimap renders again
MINIMAP_IDLE = 60  # most frames between two renders, so the traffic around a still car moves on
RIM = 8  # pixels of the minimap's cream rim
MARGIN = 24  # pixels between the HUD and the window's edges
STATS_LINES = 9  # lines of the F1 panel
STATS_EVERY = 10  # frames between two refreshes of its text
AREA_SHOWN = 3.0  # seconds the municipio's name stays before it fades
AREA_FADE = 0.8  # seconds it takes to fade
ADDRESS_H = 84  # pixels the address takes at the bottom left; the municipio's flash shows above it
REV_SEGMENTS = 10
DAY_NAMES = ("LUN", "MAR", "MIÉ", "JUE", "VIE", "SÁB", "DOM")
CLEAR = {"padding": 0, "bg": "#00000000"}


def bearing(yaw: float) -> float:
    """Degrees clockwise from north for a car's yaw: yaw 0 faces east (+x), and +z points south."""
    return math.degrees(math.atan2(math.cos(yaw), -math.sin(yaw))) % 360


def gearbox(car, idle: float, redline: float) -> tuple[str, float]:
    """The car's gear as the HUD writes it and how far up its revs are, 0 to 1: 'R' in reverse, 'N' standing
    still, else the gearbox's own gear; the revs run from idle to redline."""
    if abs(car.speed) * 3.6 < 1.0 and car.gear >= 0:
        return "N", 0.1
    name = "R" if car.gear < 0 else str(max(car.gear, 1))
    return name, float(min(1.0, max(0.0, (car.rpm - idle) / max(redline - idle, 1.0))))


LAMPS = (("steer", "DIR"), ("traction", "TRC"), ("abs", "ABS"), ("stability", "EST"))  # the assists' lamps


def lamps_lit(car) -> list[bool]:
    """Which assists' lamps are on: those that corrected in the car's last step."""
    lit = (car.telemetry or {}).get("lit", {})
    return [bool(lit.get(key, False)) for key, _ in LAMPS]


def speed_text(speed: float, units: str = "kmh") -> str:
    """The speed in m/s as the HUD's number: km/h, or mph with units='mph'."""
    return f"{abs(speed) * (3.6 if units == 'kmh' else 2.23694):.0f}"


def minimap_stale(last, pose, frames: int, loading: bool = False) -> bool:
    """Whether the minimap must render again for the car at pose = (x, z, yaw), `frames` frames after it
    last rendered at `last` (None before its first render); while loading, the cells landing around the car
    change it even when the car stands still."""
    if last is None or frames >= MINIMAP_IDLE:
        return True
    if frames < MINIMAP_EVERY:
        return False
    if loading:
        return True
    turn = abs((math.degrees(pose[2] - last[2]) + 180.0) % 360.0 - 180.0)
    return bool(math.hypot(pose[0] - last[0], pose[1] - last[1]) >= MINIMAP_MOVE or turn >= MINIMAP_TURN)


def north_offset(yaw: float, side: float, size: float) -> tuple[float, float]:
    """Where the N badge goes, as the offset of a top-right panel `size` across: on the minimap's rim, at the
    angle north shows at when the map's top is the way the car faces."""
    a = -math.radians(bearing(yaw))  # north turns the other way from the car
    r = side / 2 + RIM / 2
    dx, dy = r * math.sin(a), -r * math.cos(a)
    c = MARGIN + side / 2
    return (c - dx - size / 2, c + dy - size / 2)


class Hud:
    """The HUD's panels, built when the engine starts, shown while driving and updated by the drive."""

    def __init__(self, engine):
        from manifoldx.camera import Camera

        self.engine = engine
        self.camera = Camera(fov=60, near=10.0, far=MINIMAP_HEIGHT + 200.0)
        self.texture = None
        self.units, self.minimap_on, self.shown = "kmh", True, False
        self._frame = 0
        self._drawn, self._drawn_frame = None, 0  # the car's pose at the minimap's last render, and its frame
        self._loading = False
        self._area, self._area_t, self._area_a = "", 0.0, None
        self._address = None
        self._spot = ""  # the name of the place the photo hint shows, or "" with no hint
        self._clock_text, self._weather, self._lit, self._gear, self._speed = None, None, None, None, None
        self.panels, self.minimap = [], []
        self.stats_panel, self.stats_lines, self.stats_on = None, [], False
        engine.on("startup")(self._build)

    def _build(self, _=None) -> None:
        """The panels; the minimap's texture needs the engine's device, which exists from startup on."""
        from manifoldx.gui import Image, Panel, Text

        e = self.engine
        e.gui.design_height = ui.DESIGN_H
        self.texture = e.render_to_texture(self.camera, (MINIMAP_SIZE, MINIMAP_SIZE), live=False,
                                           background="#25331f")
        # Top left: the time, the day and the weather.
        self.weather_icon = Image(ui.icon("sol"), style_overrides={"width": 30, "height": 30})
        self.time_text = Text("10:00", style_overrides={"font": ui.TITLE, "font_size": 34, "fg": ui.PAPER})
        self.day_text = Text("MAR", style_overrides={"font": ui.BOLD, "font_size": 14, "fg": ui.PAPER})
        clock = Panel([self.weather_icon, self.time_text, self.day_text], anchor="top-left", offset=(MARGIN, MARGIN),
                      style_overrides={"direction": "h", "padding": "6 14", "gap": 10, "bg": ui.alpha(ui.INK, 0.85)})
        # Top right: the rim, the map, the car's arrow and the north badge.
        side = MINIMAP_SHOWN
        rim = Panel([Image(ui.ring(side + 2 * RIM, RIM), style_overrides={"width": side + 2 * RIM})],
                    anchor="top-right", offset=(MARGIN - RIM, MARGIN - RIM),
                    style_overrides={**CLEAR, "width": side + 2 * RIM, "height": side + 2 * RIM})
        disc = Panel([Image(self.texture, style_overrides={"width": side, "radius": side // 2})],
                     anchor="top-right", offset=(MARGIN, MARGIN), style_overrides={**CLEAR, "width": side, "height": side})
        car = Panel([Image(ui.arrow(22), style_overrides={"width": 22})], anchor="top-right",
                     offset=(MARGIN + side // 2 - 11, MARGIN + side // 2 - 11),
                     style_overrides={**CLEAR, "width": 22, "height": 22})
        self.north = Panel([Image(ui.north(26), style_overrides={"width": 26})], anchor="top-right",
                           offset=north_offset(0.0, side, 26), style_overrides={**CLEAR, "width": 26, "height": 26})
        self.minimap = [rim, disc, car, self.north]
        # Bottom left: the municipio, with the logo's red shadow, and the city under it.
        big = {"font": ui.TITLE, "font_size": 46}
        self.area_shadow = Text("", style_overrides={**big, "fg": ui.alpha(ui.RED, 0)})
        self.area_text = Text("", style_overrides={**big, "fg": ui.alpha(ui.PAPER, 0)})
        self.area_city = Text("LA HABANA", style_overrides={"font": ui.BOLD, "font_size": 14, "fg": ui.alpha(ui.INK, 0)})
        self.area_chip = Panel([self.area_city], style_overrides={"padding": "2 10", "bg": ui.alpha(ui.PAPER, 0)})
        area_back = Panel([self.area_shadow], anchor="bottom-left", offset=(MARGIN + 3, MARGIN + ADDRESS_H + 34 - 3),
                          style_overrides=CLEAR)
        area = Panel([self.area_text, self.area_chip], anchor="bottom-left", offset=(MARGIN, MARGIN + ADDRESS_H),
                     style_overrides={**CLEAR, "gap": 2})
        # Under it, always: the street with the same red shadow, the streets it is between, barrio and municipio.
        # The shadow is a copy of the whole column with only its street drawn, so both lay out alike.
        street = {"font": ui.TITLE, "font_size": 30}
        line = {"font": ui.BOLD, "font_size": 15}
        place = {"font": ui.BOLD, "font_size": 12}
        self.street_text = Text("", style_overrides={**street, "fg": ui.PAPER})
        self.line_text = Text("", style_overrides={**line, "fg": ui.PAPER})
        self.place_text = Text("", style_overrides={**place, "fg": ui.INK})
        self.street_shadow = Text("", style_overrides={**street, "fg": ui.RED})
        self.line_ghost = Text("", style_overrides={**line, "fg": ui.alpha(ui.PAPER, 0)})
        self.place_ghost = Text("", style_overrides={**place, "fg": ui.alpha(ui.INK, 0)})
        self.place_chip = Panel([self.place_text], style_overrides={"padding": "2 8", "bg": ui.PAPER})
        self.address_back = Panel([self.street_shadow, self.line_ghost,
                                   Panel([self.place_ghost], style_overrides={**CLEAR, "padding": "2 8"})],
                                  anchor="bottom-left", offset=(MARGIN + 2, MARGIN - 2), style_overrides={**CLEAR, "gap": 1})
        self.address = Panel([self.street_text, self.line_text, self.place_chip], anchor="bottom-left",
                             offset=(MARGIN, MARGIN), style_overrides={**CLEAR, "gap": 1})
        # Bottom right: the speed, the gear and the revs; an ink copy of the number behind it as its shadow.
        self.speed_shadow = Text("0", style_overrides={"font": ui.TITLE, "font_size": 76, "fg": ui.alpha(ui.INK, 0.8)})
        self.speed_text = Text("0", style_overrides={"font": ui.TITLE, "font_size": 76, "fg": ui.PAPER})
        self.gear_text = Text("N", style_overrides={"font": ui.TITLE, "font_size": 26, "fg": ui.INK})
        gear = Panel([self.gear_text], style_overrides={"padding": "2 9", "bg": ui.YELLOW})
        self.units_text = Text("KM/H", style_overrides={"font": ui.BOLD, "font_size": 15, "fg": ui.PAPER})
        self.revs = Image(ui.revs(0, REV_SEGMENTS), style_overrides={"width": 150, "height": 10})
        self.lamp_texts = [Text(label, style_overrides={"font": ui.BOLD, "font_size": 12, "fg": ui.alpha(ui.PAPER, 0.25)})
                           for _, label in LAMPS]
        self._lamps = [False] * len(LAMPS)
        self._number_panel = Panel([self.speed_text], anchor="bottom-right", offset=(MARGIN + 46, MARGIN + 34),
                                   style_overrides=CLEAR)
        speed_back = Panel([self.speed_shadow], anchor="bottom-right", offset=(MARGIN + 46 - 3, MARGIN + 34 - 3),
                           style_overrides=CLEAR)
        gear_panel = Panel([gear], anchor="bottom-right", offset=(MARGIN, MARGIN + 92), style_overrides=CLEAR)
        tail = Panel([self.units_text, self.revs], anchor="bottom-right", offset=(MARGIN, MARGIN),
                     style_overrides={**CLEAR, "gap": 4})
        lamps = Panel(self.lamp_texts, anchor="bottom-right", offset=(MARGIN + 52, MARGIN + 98),
                      style_overrides={**CLEAR, "gap": 6, "direction": "h"})
        # Bottom centre, while a place without a photo is in reach and in sight: its name and the key F.
        self.spot_text = Text(self._spot, style_overrides={"font": ui.BOLD, "font_size": 17, "fg": ui.PAPER})
        key = Panel([Text("F", style_overrides={"font": ui.KEYS, "font_size": 15, "fg": ui.INK})],
                    style_overrides={"padding": "1 8", "bg": ui.YELLOW})
        self.spot_panel = Panel([Image(ui.icon("foto"), style_overrides={"width": 22, "height": 22}), self.spot_text, key,
                                 Text("Foto", style_overrides={"font": ui.BOLD, "font_size": 15, "fg": ui.PAPER})],
                                anchor="bottom-center", offset=(0, MARGIN),
                                style_overrides={"direction": "h", "gap": 10, "padding": "6 14", "bg": ui.alpha(ui.INK, 0.85)})
        self.panels = [clock, area_back, area, self.address_back, self.address, speed_back, self._number_panel,
                       gear_panel, tail, lamps]
        self.area_panel = area
        self.stats_lines = [Text("", style_overrides={"font": ui.KEYS, "font_size": 16, "fg": "#e8e8e8"})
                            for _ in range(STATS_LINES)]
        self.stats_panel = Panel(self.stats_lines, anchor="top-left", offset=(MARGIN, MARGIN + 60),
                                 style_overrides={"padding": 10, "gap": 2, "bg": "#000000b0", "width": 460,
                                                  "height": 24 + 22 * STATS_LINES})
        for p in (clock, area, gear_panel, self.address_back, self.address, self.spot_panel):  # the others fit by their anchor
            ui.fit(p)
        if self.shown:
            self.shown = False
            self.show(True)

    def show(self, on: bool) -> None:
        """Put the HUD on the screen, or take it off for the menus and the photo mode."""
        if on == self.shown:
            return
        self.shown = on
        if not self.panels:  # not built yet: _build shows it
            return
        g = self.engine.gui
        for p in self.panels + (self.minimap if self.minimap_on else []):
            (g.append if on else g.remove)(p)
        if self.stats_on:
            (g.append if on else g.remove)(self.stats_panel)
        if self._spot:
            (g.append if on else g.remove)(self.spot_panel)

    def configure(self, units: str, minimap: bool) -> None:
        """km/h or mph, and the minimap shown or hidden; takes effect at once."""
        self.units = units
        if self.panels:
            self.units_text.text = "KM/H" if units == "kmh" else "MPH"
        if minimap != self.minimap_on:
            if self.shown and self.minimap:
                for p in self.minimap:
                    (self.engine.gui.append if minimap else self.engine.gui.remove)(p)
            self.minimap_on = minimap
            self._drawn = None

    def toggle_stats(self) -> None:
        """Show the F1 panel if hidden, hide it if shown."""
        if self.stats_panel is None:
            return
        self.stats_on = not self.stats_on
        if self.shown:
            (self.engine.gui.append if self.stats_on else self.engine.gui.remove)(self.stats_panel)

    def show_stats(self, lines) -> None:
        """Put these lines in the F1 panel, every STATS_EVERY frames while it is shown."""
        if not self.stats_on or self._frame % STATS_EVERY:
            return
        for text, line in zip(self.stats_lines, list(lines) + [""] * STATS_LINES, strict=False):
            text.text = line

    def set_time(self, when, weather: str | None) -> None:
        """The clock's time and day, and the weather state's icon ('clear', 'rain'...)."""
        from contravia.settings import WEATHER_ICON

        if not self.panels:
            return
        text = when.strftime("%H:%M")
        if text != self._clock_text:
            self._clock_text = text
            self.time_text.text = text
            self.day_text.text = DAY_NAMES[when.weekday()]
            ui.refit(self.panels[0])
        icon = WEATHER_ICON.get(weather, "sol")
        if icon == "sol" and not 6 <= when.hour < 19:
            icon = "luna"
        if icon != self._weather:
            self._weather = icon
            self.weather_icon.svg = ui.icon(icon)

    def set_area(self, name: str, dt: float) -> None:
        """The municipio the car is in: a new name shows at once, then fades after AREA_SHOWN seconds."""
        if not self.panels:
            return
        if name != self._area:
            self._area, self._area_t = name, 0.0
            self.area_text.text = self.area_shadow.text = name.upper()
            ui.refit(self.area_panel)
        else:
            self._area_t += dt
        a = 0.0 if not name else min(1.0, max(0.0, 1.0 - (self._area_t - AREA_SHOWN) / AREA_FADE))
        if a == self._area_a:
            return  # shown, or gone, as it was
        self._area_a = a
        self.area_text.style_overrides = {**self.area_text.style_overrides, "fg": ui.alpha(ui.PAPER, a)}
        self.area_shadow.style_overrides = {**self.area_shadow.style_overrides, "fg": ui.alpha(ui.RED, a)}
        self.area_city.style_overrides = {**self.area_city.style_overrides, "fg": ui.alpha(ui.INK, a)}
        self.area_chip.style_overrides = {**self.area_chip.style_overrides, "bg": ui.alpha(ui.PAPER, a)}

    def set_spot(self, name: str) -> None:
        """The photo hint at the bottom centre for the place `name`; "" takes it off."""
        if name == self._spot:
            return
        had, self._spot = self._spot, name
        if not self.panels:
            return
        self.spot_text.text = name
        ui.refit(self.spot_panel)
        if self.shown and bool(had) != bool(name):
            (self.engine.gui.append if name else self.engine.gui.remove)(self.spot_panel)

    def set_address(self, street: str, line: str, place: str) -> None:
        """The street the car is on, the line under it ("entre L y M") and "barrio · municipio"."""
        if not self.panels or (street, line, place) == self._address:
            return
        self._address = (street, line, place)
        self.street_text.text = self.street_shadow.text = street.upper()
        self.line_text.text = self.line_ghost.text = line
        self.place_text.text = self.place_ghost.text = place.upper()
        self.place_chip.style_overrides = {**self.place_chip.style_overrides, "bg": ui.PAPER if place else "#00000000"}
        ui.refit(self.address_back)
        ui.refit(self.address)

    def update(self, car, origin: np.ndarray, rpm_range: tuple[float, float] = (850.0, 6500.0),
               loading: bool = False) -> None:
        """Show the car's speed, gear and revs between rpm_range's idle and redline, light the assists' lamps,
        each written only when what it shows changed, and when the car has moved or turned enough, or the cells
        around it are loading (minimap_stale), render the minimap above it and turn the north badge with it;
        once loading ends it renders once more."""
        if not self.panels:
            return
        speed = speed_text(car.speed, self.units)
        if speed != self._speed:
            self._speed = speed
            self.speed_text.text = self.speed_shadow.text = speed
        for text, was, now in zip(self.lamp_texts, self._lamps, lamps_lit(car), strict=True):
            if was != now:
                text.style_overrides = {**text.style_overrides, "fg": ui.YELLOW if now else ui.alpha(ui.PAPER, 0.25)}
        self._lamps = lamps_lit(car)
        gear, revs = gearbox(car, *rpm_range)
        if gear != self._gear:
            self._gear = gear
            self.gear_text.text = gear
        lit = round(revs * REV_SEGMENTS)
        if lit != self._lit:
            self._lit = lit
            self.revs.svg = ui.revs(lit, REV_SEGMENTS)
        self._frame += 1
        pose = (car.x, car.z, car.yaw)
        if self._loading and not loading:  # the last cells landed after the last render
            self._drawn = None
        self._loading = loading
        if not self.minimap_on or not minimap_stale(self._drawn, pose, self._frame - self._drawn_frame, loading):
            return
        self._drawn, self._drawn_frame = pose, self._frame
        self.north.offset = north_offset(car.yaw, MINIMAP_SHOWN, 26)
        here = np.array([car.x, car.y, car.z]) - origin
        self.camera.up = np.array([math.cos(car.yaw), 0.0, math.sin(car.yaw)], np.float32)
        self.camera.set_pose(position=tuple(here + (0.0, MINIMAP_HEIGHT, 0.0)), target=tuple(here))
        self.engine.refresh_texture(self.texture)
