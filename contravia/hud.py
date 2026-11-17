"""The HUD: a round minimap in the top right corner, and the heading and speed in the bottom left.

The minimap is the scene itself seen from MINIMAP_HEIGHT metres straight above the car, rendered by the engine
into a texture (render_to_texture) every MINIMAP_EVERY frames and shown as a circle. Its top is the way the
car faces, so the arrow in its middle always points up and the streets turn around it. The heading names the
compass point the car faces and its bearing in degrees, clockwise from north. The GUI scales with the window
from a 720-pixel-high design.

F1 shows or hides a panel of numbers in the top left corner for working on the game: frames per second and
the slowest frame lately, the streamed nodes drawn by level, what is building and uploading, the traffic, and
where the car is and how its own cell stands.
"""

import math
from importlib.resources import files

import numpy as np

MINIMAP_SIZE = 256  # pixels of the texture
MINIMAP_SHOWN = 200  # pixels across on screen
MINIMAP_HEIGHT = 300.0  # metres above the car
MINIMAP_EVERY = 4  # frames between two renders of the minimap
MARGIN = 20  # pixels between the HUD and the window's edges
STATS_LINES = 9  # lines of the F1 panel
STATS_EVERY = 10  # frames between two refreshes of its text
POINTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
FONT = str(files("manifoldx.viz.assets").joinpath("DejaVuSansMono.ttf"))


def bearing(yaw: float) -> float:
    """Degrees clockwise from north for a car's yaw: yaw 0 faces east (+x), and +z points south."""
    return math.degrees(math.atan2(math.cos(yaw), -math.sin(yaw))) % 360


def heading_text(yaw: float, speed: float) -> str:
    """'NE 42°  36 km/h' for a car facing that way at that speed in m/s."""
    b = round(bearing(yaw)) % 360
    return f"{POINTS[round(b / 45) % 8]:2s} {b:3d}°  {abs(speed) * 3.6:3.0f} km/h"


class Hud:
    """The minimap and the heading line, built when the engine starts and updated by the drive."""

    def __init__(self, engine):
        from manifoldx.camera import Camera

        self.engine = engine
        self.camera = Camera(fov=60, near=10.0, far=MINIMAP_HEIGHT + 200.0)
        self.texture = None
        self.heading = None
        self._frame = 0
        self.stats_panel, self.stats_lines, self.stats_on = None, [], False
        engine.on("startup")(self._build)

    def _build(self, _=None) -> None:
        """The GUI's panels; the texture needs the engine's device, which exists from startup on."""
        from manifoldx.gui import Image, Panel, Text

        e = self.engine
        e.gui.design_height = 720
        self.texture = e.render_to_texture(self.camera, (MINIMAP_SIZE, MINIMAP_SIZE), live=False,
                                           background="#25331f")
        side, clear = MINIMAP_SHOWN, {"padding": 0, "bg": "#00000000"}
        e.gui.append(Panel([Image(self.texture, style_overrides={"width": side, "radius": side // 2})],
                           anchor="top-right", offset=(MARGIN, MARGIN),
                           style_overrides={**clear, "width": side, "height": side}))
        arrow = Text("▲", style_overrides={"font": FONT, "font_size": 24, "fg": "#ff453a"})
        e.gui.append(Panel([arrow], anchor="top-right", offset=(MARGIN + side // 2 - 9, MARGIN + side // 2 - 14),
                           style_overrides={**clear, "width": 18, "height": 28}))
        self.heading = Text(heading_text(0.0, 0.0), style_overrides={"font": FONT, "font_size": 26, "fg": "#ffffff"})
        e.gui.append(Panel([self.heading], anchor="bottom-left", offset=(MARGIN, MARGIN),
                           style_overrides={"padding": 10, "bg": "#000000a0", "width": 300, "height": 50}))
        self.stats_lines = [Text("", style_overrides={"font": FONT, "font_size": 16, "fg": "#e8e8e8"})
                            for _ in range(STATS_LINES)]
        self.stats_panel = Panel(self.stats_lines, anchor="top-left", offset=(MARGIN, MARGIN),
                                 style_overrides={"padding": 10, "gap": 2, "bg": "#000000b0", "width": 460,
                                                  "height": 24 + 22 * STATS_LINES})

    def toggle_stats(self) -> None:
        """Show the F1 panel if hidden, hide it if shown."""
        if self.stats_panel is None:
            return
        self.stats_on = not self.stats_on
        (self.engine.gui.append if self.stats_on else self.engine.gui.remove)(self.stats_panel)

    def show_stats(self, lines) -> None:
        """Put these lines in the F1 panel, every STATS_EVERY frames while it is shown."""
        if not self.stats_on or self._frame % STATS_EVERY:
            return
        for text, line in zip(self.stats_lines, list(lines) + [""] * STATS_LINES, strict=False):
            text.text = line

    def update(self, car, origin: np.ndarray) -> None:
        """Show the car's heading and speed, and every MINIMAP_EVERY frames render the minimap above it."""
        if self.heading is None:
            return
        self.heading.text = heading_text(car.yaw, car.speed)
        self._frame += 1
        if self._frame % MINIMAP_EVERY:
            return
        here = np.array([car.x, car.y, car.z]) - origin
        self.camera.up = np.array([math.cos(car.yaw), 0.0, math.sin(car.yaw)], np.float32)
        self.camera.set_pose(position=tuple(here + (0.0, MINIMAP_HEIGHT, 0.0)), target=tuple(here))
        self.engine.refresh_texture(self.texture)
