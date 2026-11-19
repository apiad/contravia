"""What the player chooses in the menus: the free drive's start (place, time, day, weather, traffic, clock
speed) and the options kept between runs (view distance, rain on screen, units, minimap).

Each row is a Choice: a label and a ring of (text, value) options that the left and right keys turn. The free
drive's rows hold what `contravia play` takes on its command line (--at, --time, --sky, --with-traffic,
--warp), so the flags fill the rows in and the menu can change them. The options are saved as JSON in the
user's configuration folder: $XDG_CONFIG_HOME/contravia (~/.config/contravia) on Linux, %APPDATA%\\contravia
on Windows, or $CONTRAVIA_CONFIG when it is set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from contravia.geo import to_local

# (name, district, lat, lon, poster panel or None)
PLACES = (
    ("Capitolio", "La Habana Vieja", 23.13528, -82.35972, "capitolio"),
    ("Malecón", "Centro Habana", 23.14390, -82.36400, "malecon"),
    ("Parque Central", "La Habana Vieja", 23.13780, -82.35770, None),
    ("Plaza de Armas", "La Habana Vieja", 23.14070, -82.35020, None),
    ("23 y L", "Plaza de la Revolución", 23.13960, -82.38300, None),
    ("Plaza de la Revolución", "Plaza de la Revolución", 23.12270, -82.38640, None),
)
RANDOM = "Al azar"
DAYS = (("Lunes", "mon"), ("Martes", "tue"), ("Miércoles", "wed"), ("Jueves", "thu"), ("Viernes", "fri"),
        ("Sábado", "sat"), ("Domingo", "sun"))
WEATHER = (("Como hoy", None), ("Sol", "clear"), ("Algunas nubes", "partly"), ("Nublado", "overcast"),
           ("Lluvia", "rain"), ("Tormenta", "storm"))
WEATHER_ICON = {None: "sol", "clear": "sol", "partly": "nubes", "overcast": "nubes", "rain": "lluvia",
                "storm": "tormenta"}
WARPS = (1.0, 10.0, 60.0)
VIEW = (("Cerca", (16.0, 64.0)), ("Media", (12.0, 48.0)), ("Lejos", (8.0, 32.0)))  # the streamer's tau range


@dataclass
class Choice:
    """One row of a menu: a label, its options as (text, value) and which one is chosen."""

    key: str
    label: str
    options: list[tuple[str, Any]]
    index: int = 0
    help: str = ""

    @property
    def text(self) -> str:
        return self.options[self.index][0]

    @property
    def value(self) -> Any:
        return self.options[self.index][1]

    def step(self, d: int) -> None:
        """Turn to the next option (d = 1) or the one before (d = -1), round the ring."""
        self.index = (self.index + d) % len(self.options)

    def pick(self, value: Any) -> None:
        """Choose the option with this value; an unknown value leaves the choice as it was."""
        for k, (_, v) in enumerate(self.options):
            if v == value:
                self.index = k


@dataclass
class FreeDrive:
    """The rows of the free drive's screen, filled in from the command line's flags."""

    rows: list[Choice] = field(default_factory=list)
    locked_assists: tuple = ()  # assists the game has not given the player yet: off at any realism

    @classmethod
    def from_flags(cls, at=(0.0, 0.0), when: str | None = None, force: str | None = None,
                   traffic: bool = False, warp: float = 1.0) -> FreeDrive:
        """The rows as the flags of contravia play set them: a start point other than the Capitolio becomes
        the first place, "17:30" or "Sun 08:00" set the hour and the day, force the weather."""
        places = [(name, ("place", name)) for name, *_ in PLACES] + [(RANDOM, ("random", None))]
        if tuple(at) != (0.0, 0.0):
            places.insert(0, ("Punto de partida", ("at", tuple(at))))
        hours = [(f"{h:02d}:{m:02d}", (h, m)) for h in range(24) for m in (0, 30)]
        rows = [
            Choice("place", "Lugar", places, help="Dónde arranca el carro: en la calle más cercana al lugar."),
            Choice("vehicle", "Vehículo", [("Sedán", None)],
                   help="El vehículo que manejas: cualquiera del Garaje, tus diseños también. En el Garaje los "
                        "ves girar y los cambias."),
            Choice("hour", "Hora", hours, help="La hora a la que arranca el reloj del juego."),
            Choice("day", "Día", list(DAYS), help="El día de la semana: el tráfico de un domingo no es el de un "
                                                  "martes."),
            Choice("weather", "Clima", list(WEATHER),
                   help="Como hoy sigue la cadena del clima de La Habana; las demás fijan el tiempo."),
            Choice("traffic", "Tráfico", [("Sin tráfico", False), ("Con tráfico", True)],
                   help="Los carros de la ciudad, según la hora y el día. Piden más a la computadora."),
            Choice("warp", "Velocidad del reloj", [(f"×{w:g}", w) for w in WARPS],
                   help="Cuántas veces más rápido que el reloj de verdad pasa el tiempo del juego."),
        ]
        fd = cls(rows)
        hm, day = "10:00", None
        now = datetime.now()
        if when and when != "now":
            parts = when.split()
            hm = parts[-1]
            day = parts[0][:3].lower() if len(parts) > 1 and "-" not in parts[0] else None
            if len(parts) > 1 and "-" in parts[0]:
                day = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")[datetime.fromisoformat(parts[0]).weekday()]
        elif when == "now":
            hm = f"{now.hour:02d}:{30 * (now.minute >= 30):02d}"
        h, m = (int(v) for v in hm.split(":"))
        fd["hour"].pick((h, 30 * (m >= 30)))
        fd["day"].pick(day or ("mon", "tue", "wed", "thu", "fri", "sat", "sun")[now.weekday()])
        fd["weather"].pick(force)
        fd["traffic"].pick(bool(traffic))
        fd["warp"].pick(float(warp))
        return fd

    def __getitem__(self, key: str) -> Choice:
        return next(r for r in self.rows if r.key == key)

    def place(self) -> tuple[str, str, str | None]:
        """The chosen place as (name, district, poster panel or None)."""
        kind, name = self["place"].value
        if kind == "place":
            p = next(p for p in PLACES if p[0] == name)
            return p[0], p[1], p[4]
        return self["place"].text, "La Habana", None

    def start(self, rng=None) -> dict:
        """What the game needs to start: `at` as local (x, z) or None for a random road, `when` as the
        clock's text ("tue 17:30"), `force` (a weather state or None), `traffic` and `warp`."""
        kind, v = self["place"].value
        if kind == "place":
            p = next(p for p in PLACES if p[0] == v)
            x, z = to_local(p[2], p[3])
            at = (float(x), float(z))
        elif kind == "at":
            at = v
        else:
            at = None
        h, m = self["hour"].value
        return {"at": at, "when": f"{self['day'].value} {h:02d}:{m:02d}", "force": self["weather"].value,
                "traffic": self["traffic"].value, "warp": self["warp"].value}


def config_dir() -> Path:
    """Where the options are kept: $CONTRAVIA_CONFIG, else the system's configuration folder."""
    if os.environ.get("CONTRAVIA_CONFIG"):
        return Path(os.environ["CONTRAVIA_CONFIG"])
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "contravia"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "contravia"


@dataclass
class Options:
    """The options screen's rows, saved between runs."""

    rows: list[Choice] = field(default_factory=list)
    locked_assists: tuple = ()  # assists the game has not given the player yet: off at any realism

    @classmethod
    def load(cls) -> Options:
        """The options as last saved, or the defaults; a value the file holds that no row knows is ignored."""
        rows = [
            Choice("view", "Distancia de vista", list(VIEW), 1,
                   help="Hasta dónde se dibuja la ciudad con detalle. Más lejos se ve mejor el horizonte, "
                        "pero pide más a la tarjeta de video."),
            Choice("rain", "Lluvia en pantalla", [("Sí", True), ("No", False)],
                   help="Las gotas que caen alrededor del carro cuando llueve."),
            Choice("units", "Unidades", [("km/h", "kmh"), ("mph", "mph")],
                   help="Cómo se muestra la velocidad."),
            Choice("minimap", "Minimapa", [("Sí", True), ("No", False)],
                   help="El mapa redondo arriba a la derecha, que gira con el carro."),
            Choice("realism", "Realismo",
                   [("0 · Fácil", 0)] + [(str(v), v) for v in range(10, 100, 10)] + [("100 · Realista", 100)], 3,
                   help="Cuánto te ayuda el carro: con 0 no se va de lado ni bloquea las ruedas; con 100 maneja "
                        "como uno de verdad, sin ayudas."),
        ]
        opts = cls(rows)
        path = config_dir() / "settings.json"
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            saved = {}
        for row in rows:
            if row.key in saved:
                row.pick(tuple(saved[row.key]) if isinstance(saved[row.key], list) else saved[row.key])
        opts.locked_assists = tuple(saved.get("locked_assists", ()))
        return opts

    def __getitem__(self, key: str) -> Choice:
        return next(r for r in self.rows if r.key == key)

    def save(self) -> None:
        """Write the chosen values to settings.json in the configuration folder."""
        path = config_dir() / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {r.key: r.value for r in self.rows}
        path.write_text(json.dumps({**values, "locked_assists": list(self.locked_assists)}, indent=1))
