"""The fleet's catalogue and the ranges its vehicles may take: what the Garaje shows, edits and builds.

Every vehicle is one of manifoldx's builders (car, bus, truck, cycle) with a parameter set: a preset is the set
measured from a reference drawing, with its colours and the name people in Havana give it (never a trademark). A
design is a preset with some of its parameters changed, each inside its range.

A knob is one parameter the Garaje edits. Its range is the street's (STREET below) inside the builder's own, which
manifoldx publishes as `Params.ranges()`: the street range spans every era of the family, because every era mixes.
Parameters also constrain each other (the wheelbase fits inside the length); manifoldx's `validate()` knows those
rules, so `limits()` finds the live interval of one knob by asking it, and `fix()` pulls a whole set back inside.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace

FAMILIES = ("car", "bus", "truck", "cycle")
TABS = ("Carros", "Guaguas", "Camiones", "Servicios", "Ciclos")
GROUPS = ("Forma", "Frente y cola", "Ruedas", "Pintura")

# The colours the Garaje offers, from the fleet's eras, faded as the street paints them: (name, sRGB).
COLOURS = (
    ("Turquesa", "#7fded6"), ("Crema", "#f6eed6"), ("Coral", "#f4a593"), ("Menta", "#a8e0c0"), ("Rosado", "#f2b6c6"),
    ("Amarillo pálido", "#f2e39a"), ("Azul marino", "#2b4a86"), ("Beige", "#d8c9a6"), ("Blanco", "#f2f3ef"),
    ("Ocre", "#c9932e"), ("Azul claro", "#8fb8d8"), ("Rojo", "#b8322b"), ("Plata", "#c4c7ca"), ("Gris", "#7d8185"),
    ("Negro", "#232427"), ("Azul", "#2f5fa8"), ("Verde", "#8ccf7a"), ("Naranja", "#d96a1f"),
)
COLOUR_NAMES = {h: n for n, h in COLOURS}

# Knob: (parameter, label, group). A parameter named "paint" or "second" is a colour of the palette, not of the
# builder's parameters; "livery" is one of LIVERIES.
KNOBS = {
    "car": (("length", "Largo", "Forma"), ("width", "Ancho", "Forma"), ("height", "Alto", "Forma"),
            ("wheelbase", "Batalla", "Forma"), ("fin_height", "Aletas", "Forma"),
            ("fender_crown", "Guardafangos", "Forma"), ("windscreen_wrap", "Parabrisas envolvente", "Forma"),
            ("headlamp", "Faros", "Frente y cola"), ("grille_style", "Parrilla", "Frente y cola"),
            ("bumper", "Parachoques", "Frente y cola"), ("taillamp", "Luces de atrás", "Frente y cola"),
            ("wheel_radius", "Radio de rueda", "Ruedas"), ("wheel", "Llantas", "Ruedas"),
            ("whitewall", "Banda blanca", "Ruedas"),
            ("paint", "Color", "Pintura"), ("second", "Segundo color", "Pintura"),
            ("roof_paint", "Techo de dos tonos", "Pintura"), ("black_pillars", "Parales negros", "Pintura"),
            ("livery", "Diseño", "Pintura"), ("lightbar", "Luces de techo", "Pintura")),
    "bus": (("length", "Largo", "Forma"), ("height", "Alto", "Forma"), ("wheelbase", "Batalla", "Forma"),
            ("front_overhang", "Voladizo delantero", "Forma"), ("corner_radius", "Esquinas", "Forma"),
            ("headlamp", "Faros", "Frente y cola"), ("taillamp", "Luces de atrás", "Frente y cola"),
            ("mirrors", "Espejos", "Frente y cola"),
            ("wheel_radius", "Radio de rueda", "Ruedas"), ("wheel", "Llantas", "Ruedas"),
            ("paint", "Color", "Pintura"), ("second", "Segundo color", "Pintura")),
    "truck": (("length", "Largo", "Forma"), ("height", "Alto", "Forma"), ("wheelbase", "Batalla", "Forma"),
              ("cab", "Cabina", "Forma"), ("axles", "Ejes", "Forma"),
              ("mirrors", "Espejos", "Frente y cola"), ("exhaust", "Escape alto", "Frente y cola"),
              ("beacons", "Luces de techo", "Frente y cola"),
              ("wheel_radius", "Radio de rueda", "Ruedas"), ("wheel", "Llantas", "Ruedas"),
              ("twin_rear", "Ruedas dobles", "Ruedas"),
              ("paint", "Color", "Pintura"), ("second", "Color de la carga", "Pintura")),
    "cycle": (("wheelbase", "Batalla", "Forma"), ("payload", "Carga", "Forma"), ("canopy", "Techo", "Forma"),
              ("seat_style", "Asiento", "Forma"), ("handlebar_width", "Timón", "Forma"),
              ("headlamp", "Faro", "Frente y cola"), ("tail_lamp", "Luz de atrás", "Frente y cola"),
              ("mudguards", "Guardafangos", "Frente y cola"),
              ("wheel_radius", "Radio de rueda", "Ruedas"),
              ("paint", "Color", "Pintura"), ("second", "Segundo color", "Pintura")),
}

# The street's range of each numeric knob, over every era of the family; the builder's own range clips it.
STREET = {
    "car": {"length": (3.5, 5.5), "width": (1.5, 2.0), "height": (1.35, 1.95), "wheelbase": (2.3, 3.2),
            "fin_height": (0.0, 0.3), "fender_crown": (0.0, 0.2), "windscreen_wrap": (0.0, 1.0),
            "wheel_radius": (0.27, 0.4), "whitewall": (0.0, 0.08)},
    "bus": {"length": (8.0, 15.0), "height": (2.8, 3.5), "wheelbase": (4.0, 8.0), "front_overhang": (1.5, 3.5),
            "corner_radius": (0.05, 0.6), "wheel_radius": (0.42, 0.58)},
    "truck": {"length": (5.5, 10.0), "height": (2.6, 4.0), "wheelbase": (3.0, 6.0), "wheel_radius": (0.4, 0.6)},
    "cycle": {"wheelbase": (0.9, 2.1), "handlebar_width": (0.4, 0.9), "wheel_radius": (0.2, 0.4)},
}

# A livery for a car: (name, paint areas as (u from the nose, height in metres) over the side, lightbar).
LIVERIES = {
    "Ninguno": ((), None),
    "Franja": ((((0.0, 0.62), (1.0, 0.62), (1.0, 0.70), (0.0, 0.70)),), None),
    "Policía": ((((0.0, 0.55), (1.0, 0.55), (1.0, 0.85), (0.0, 0.85)),), "bar"),
    "Taxi": ((((0.0, 0.66), (1.0, 0.66), (1.0, 0.74), (0.0, 0.74)),), "dome"),
}
# The palette material the second colour paints, per family.
SECOND = {"car": "roof", "bus": "green", "truck": "body", "cycle": "box"}


@dataclass(frozen=True)
class Preset:
    """A vehicle of the catalogue: its key, the name the game shows, its family, tab and era, its parameters and
    colours, and the facts its card shows. `drivable` is False for cycles until riders exist."""

    key: str
    name: str
    family: str
    tab: str
    era: str
    params: object
    palette: dict
    seats: int
    where: str
    livery: str = "Ninguno"
    drivable: bool = True


@dataclass
class Design:
    """A vehicle as the Garaje edits it: the preset it started from, the parameters changed, the two colours and
    the livery. `name` is what Mis diseños shows."""

    base: str
    name: str = ""
    changes: dict = field(default_factory=dict)
    paint: str | None = None
    second: str | None = None
    livery: str | None = None

    @property
    def preset(self) -> Preset:
        return catalogue()[self.base]

    def params(self):
        """The builder's parameters: the preset's with the changes, and the livery's paint areas for a car."""
        p = replace(self.preset.params, **self.changes)
        if self.preset.family == "car" and self.livery is not None:
            areas, bar = LIVERIES[self.livery]
            colour = self.second or self.preset.palette.get("roof", ("#1f3f8f",))[0]
            p = replace(p, livery=tuple((colour, list(a)) for a in areas), lightbar=self.changes.get("lightbar", bar))
        return p

    def palette(self) -> dict:
        """The palette overrides: the preset's, with the design's paint and second colour."""
        pal = dict(self.preset.palette)
        if self.paint:
            pal["paint"] = (self.paint, *pal.get("paint", ("", 0.45, 0.0))[1:])
        if self.second:
            pal[SECOND[self.preset.family]] = (self.second, 0.5, 0.0)
        return pal

    def value(self, key: str):
        """The current value of a knob."""
        if key == "paint":
            return self.paint or self.preset.palette.get("paint", ("#f2f3ef",))[0]
        if key == "second":
            return self.second or self.preset.palette.get(SECOND[self.preset.family], ("#f2f3ef",))[0]
        if key == "livery":
            return self.livery or self.preset.livery
        return getattr(self.params(), key)

    def to_json(self) -> dict:
        return {"base": self.base, "name": self.name, "changes": self.changes, "paint": self.paint,
                "second": self.second, "livery": self.livery}

    @classmethod
    def from_json(cls, d: dict) -> Design:
        return cls(d["base"], d.get("name", ""), dict(d.get("changes", {})), d.get("paint"), d.get("second"),
                   d.get("livery"))


def _sedan(L, width, track, wheels, r, hood, cowl_u, ws_u, top, roof_end_u, deck_u, deck, tail, belt, **kw):
    """A three-box sedan from the ratios measured on a side drawing: heights and stations as fractions of L."""
    from manifoldx.modeling.assets import CarParams

    wf, wr = wheels
    hc = kw.pop("headlamp_center", (0.36 * width, (hood - 0.07) * L))
    tc = kw.pop("taillamp_center", (0.38 * width, (deck - 0.065) * L))
    return CarParams(length=L, width=width, track=track, height=top * L, wheelbase=(wr - wf) * L,
                     front_overhang=wf * L, wheel_radius=r * L, nose_top=(hood - 0.035) * L, hood_front=hood * L,
                     nose_blunt=0.85, cowl_u=cowl_u, cowl_height=(hood + 0.012) * L, windscreen_u=ws_u,
                     roof_peak_u=(ws_u + roof_end_u) / 2, roof_end_u=roof_end_u, roof_end_drop=0.01 * L,
                     hatch_u=deck_u, hatch_height=deck * L, tail_top=tail * L, belt_front=belt * L,
                     belt_rear=belt * L, sill=0.06 * L, bumper_bottom_front=0.07 * L, bumper_bottom_rear=0.07 * L,
                     hood_crown=0.03, nose_half=0.9, tail_half=0.92, round_front=0.08, round_rear=0.08,
                     tumblehome=0.14, shoulder=0.98, a_pillar_foot=0.15, b_pillar_width=0.06, handles=True,
                     crease=30.0, headlamp_center=hc, taillamp_center=tc,
                     grille=kw.pop("grille", None) or (0.26 * width, (hood - 0.115) * L, (hood - 0.045) * L), **kw)


def _bus_livery():
    """The metroguagua's colours: green front corners and rear, a cyan wedge at the front of the roof line, an
    orange stripe under the windows."""
    side = (("green", [(0.0, 0.26), (0.055, 0.26), (0.055, 0.88), (0.0, 0.88)]),
            ("green", [(0.72, 0.26), (1.0, 0.26), (1.0, 1.14), (0.72, 1.14)]),
            ("green", [(0.72, 2.72), (1.0, 2.72), (1.0, 3.1), (0.72, 3.1)]),
            ("cyan", [(0.0, 2.72), (0.32, 2.72), (0.40, 3.1), (0.0, 3.1)]),
            ("stripe", [(0.10, 0.98), (0.72, 0.98), (0.72, 1.14), (0.10, 1.14)]))
    front = (("green", [(-1.02, 0.26), (1.02, 0.26), (1.02, 0.88), (-1.02, 0.88)]),
             ("cyan", [(-1.02, 3.0), (1.02, 3.0), (1.02, 3.09), (-1.02, 3.09)]))
    rear = (("green", [(-1.02, 0.26), (1.02, 0.26), (1.02, 1.0), (-1.02, 1.0)]),
            ("stripe", [(-1.02, 1.04), (1.02, 1.04), (1.02, 1.15), (-1.02, 1.15)]))
    roof = (("cyan", [(0.0, -1.27), (0.38, -1.27), (0.38, 1.27), (0.0, 1.27)]),
            ("green", [(0.72, -1.27), (1.0, -1.27), (1.0, 1.27), (0.72, 1.27)]))
    return {"livery": side, "front_livery": front, "rear_livery": rear, "roof_livery": roof}


@functools.cache
def catalogue() -> dict[str, Preset]:
    """Every preset by key, in the order the Garaje lists them."""
    from manifoldx.modeling import assets as a

    almendron = _sedan(4.5, 1.80, 1.48, (0.18, 0.78), 0.073, 0.278, 0.33, 0.40, 0.396, 0.70, 0.78, 0.275, 0.262,
                       0.255, headlamp="round_pod", headlamp_center=None, headlamp_size=(0.24, 0.24),
                       taillamp="fin", taillamp_center=None, fender_crown=0.12, fin_height=0.16, fin_start_u=0.72,
                       fin_width=0.20, windscreen_wrap=0.6,
                       side_trim=((0.78, 0.55, 0.95, "chrome"), (0.36, 0.2, 0.8, "chrome")),
                       black_pillars=True, roof_paint=True, roof_pillars=True, bumper="chrome_bar",
                       bumper_height=0.36, bumper_proud=0.06, overriders=True, grille_style="mesh",
                       grille_span="between_lamps", grille=(0.62, 0.42, 0.70), wheel="hubcap", whitewall=0.05)
    ruso = _sedan(4.13, 1.62, 1.36, (0.17, 0.77), 0.07, 0.272, 0.31, 0.37, 0.412, 0.74, 0.80, 0.29, 0.275, 0.25,
                  headlamp="rect", headlamp_size=(0.24, 0.14), taillamp="block", black_pillars=False,
                  bumper="chrome_bar", bumper_height=0.35, bumper_proud=0.055, grille_style="mesh",
                  grille_span="full_width", wheel="hubcap")
    moscovita = _sedan(4.25, 1.55, 1.27, (0.17, 0.76), 0.075, 0.28, 0.31, 0.37, 0.428, 0.75, 0.80, 0.30, 0.29,
                       0.265, headlamp="round_pod", headlamp_size=(0.16, 0.16), taillamp="block",
                       black_pillars=False, bumper="rubber", bumper_height=0.34, bumper_proud=0.05,
                       grille_style="mesh", wheel="hubcap")
    paint = lambda h, r=0.45, m=0.0: (h, r, m)
    presets = [
        Preset("almendron", "Almendrón", "car", "Carros", "Años 50", almendron,
               {"paint": paint("#7fded6"), "roof": paint("#f6eed6", 0.6)}, 6, "Rutas de almendrones, todo el día"),
        Preset("ruso", "Ruso", "car", "Carros", "Años 70-80", ruso, {"paint": paint("#2b4a86")}, 5,
               "Toda la ciudad"),
        Preset("moscovita", "Moscovita", "car", "Carros", "Años 70-80", moscovita, {"paint": paint("#f4a593")}, 5,
               "Toda la ciudad"),
        Preset("metroguagua", "Metroguagua", "bus", "Guaguas", "Moderno",
               a.BusParams(wheel="hubcap", **_bus_livery()),
               {"paint": paint("#f3efe2"), "green": paint("#8ccf7a", 0.5), "cyan": paint("#31b6d8", 0.5),
                "stripe": paint("#f0a63a", 0.5), "black": paint("#33373a", 0.5), "glass": paint("#c8d8dd", 0.05),
                "chrome": paint("#a8adb1", 0.3)}, 80, "Sus rutas y paradas"),
        Preset("camion", "Camión de los 80", "truck", "Camiones", "Años 70-80", a.cargo_truck_1980s(),
               {"paint": paint("#7fa8bd", 0.4, 0.1), "body": paint("#7fa8bd", 0.4, 0.1),
                "deck": paint("#8a5a33", 0.85)}, 2, "Avenidas y puerto"),
        Preset("caja", "Camión de caja", "truck", "Camiones", "Moderno", a.box_truck(),
               {"paint": paint("#f1f2f3", 0.35), "body": paint("#b9bdc2")}, 2, "El centro, de noche"),
        Preset("pipa", "Pipa", "truck", "Camiones", "Años 70-80", a.tanker(),
               {"paint": paint("#e7c33b", 0.35), "body": paint("#c9ccd0", 0.25, 0.9),
                "tank": paint("#c9ccd0", 0.25, 0.9)}, 2, "Avenidas, de noche"),
        Preset("volteo", "Volteo", "truck", "Camiones", "Años 70-80", a.dump_truck(),
               {"paint": paint("#d96a1f", 0.4), "body": paint("#d96a1f", 0.4)}, 2, "Obras y afueras"),
        Preset("patrulla", "Patrulla", "car", "Servicios", "Años 70-80",
               replace(ruso, livery=(("#1f3f8f", [(0.0, 0.55), (1.0, 0.55), (1.0, 0.85), (0.0, 0.85)]),),
                       lightbar="bar"),
               {"paint": paint("#f2f3ef", 0.35), "roof": paint("#1f3f8f", 0.5)}, 4, "Donde haya lío",
               livery="Policía"),
        Preset("bomberos", "Bomberos", "truck", "Servicios", "Años 70-80", a.fire_engine(),
               {"paint": paint("#c62828", 0.35), "body": paint("#c62828", 0.35),
                "locker": paint("#b9bdc2", 0.45, 0.6), "ladder": paint("#c9ccd0", 0.3, 0.8),
                "livery0": paint("#f4f4f2", 0.35)}, 4, "Desde su estación"),
        Preset("bicitaxi", "Bicitaxi", "cycle", "Ciclos", "Moderno", a.cycle_rickshaw(),
               {"paint": paint("#d6dbdc", 0.35, 0.1), "box": paint("#5fc4b7"), "canopy": paint("#efe4c6", 0.65),
                "spoke": paint("#d8dcdd", 0.25, 0.8), "chrome": paint("#d8dcdd", 0.2, 0.8),
                "saddle": paint("#2a2a2e", 0.8)}, 2, "Centro y Habana Vieja", drivable=False),
        Preset("triciclo", "Triciclo eléctrico", "cycle", "Ciclos", "Moderno", a.cargo_tricycle(),
               {"paint": paint("#ffe94f", 0.5), "box": paint("#ffe94f", 0.5), "canopy": paint("#ffe94f", 0.5),
                "seat": paint("#8fe0cf", 0.5), "saddle": paint("#8fe0cf", 0.5)}, 6, "Rutas cortas",
               drivable=False),
        Preset("moto", "Moto", "cycle", "Ciclos", "Años 70-80", a.motorcycle(),
               {"paint": paint("#b6f2e2", 0.4, 0.05), "saddle": paint("#17171a", 0.85),
                "engine": paint("#c9c9c4", 0.45, 0.2), "chrome": paint("#dfe2e3", 0.15, 0.9)}, 2,
               "Toda la ciudad", drivable=False),
        Preset("motoelectrica", "Moto eléctrica", "cycle", "Ciclos", "Moderno", a.electric_moto(),
               {"paint": paint("#ff7a62"), "saddle": paint("#bfe8d8", 0.6), "box": paint("#bfe8d8", 0.6),
                "chrome": paint("#c3c7c9", 0.35, 0.6), "rim": paint("#8e9295", 0.5, 0.6),
                "spoke": paint("#a9adaf", 0.5, 0.6)}, 2, "Toda la ciudad", drivable=False),
        Preset("bicicleta", "Bicicleta", "cycle", "Ciclos", "Moderno", a.bicycle(),
               {"paint": paint("#2f5fa8")}, 1, "Toda la ciudad", drivable=False),
    ]
    return {p.key: p for p in presets}


def knobs(family: str, group: str | None = None) -> list[tuple[str, str, str]]:
    """The family's knobs as (parameter, label, group), all or one group's."""
    return [k for k in KNOBS[family] if group is None or k[2] == group]


def ranges(family: str) -> dict[str, dict]:
    """Each knob's range: {"min", "max", "step", "unit"} for a number, {"choices"} for an option. A number's range
    is the street's clipped to the builder's; colours offer COLOURS and the car's livery LIVERIES."""
    params_cls = type(next(p for p in catalogue().values() if p.family == family).params)
    built = params_cls.ranges()
    out = {}
    for key, _, _ in KNOBS[family]:
        if key in ("paint", "second"):
            out[key] = {"choices": colours(family)}
        elif key == "livery":
            out[key] = {"choices": tuple(LIVERIES)}
        elif "choices" in built[key]:
            out[key] = {"choices": built[key]["choices"]}
        else:
            b = built[key]
            lo, hi = STREET[family].get(key, (b["min"], b["max"]))
            step = b.get("step") or 0.01
            out[key] = {"min": max(lo, b["min"]), "max": min(hi, b["max"]), "step": max(step, 0.01),
                        "unit": b.get("unit", "m")}
    return out


@functools.cache
def colours(family: str) -> tuple[str, ...]:
    """The colours a family may wear: COLOURS, then the factory colours of its presets that are not among them."""
    out = [h for _, h in COLOURS]
    for p in catalogue().values():
        if p.family == family:
            for mat in ("paint", SECOND[family]):
                if mat in p.palette and p.palette[mat][0] not in out:
                    out.append(p.palette[mat][0])
    return tuple(out)


def colour_name(hex_colour: str) -> str:
    """A colour's name on the Garaje's rows; a preset's own colour is "De fábrica"."""
    return COLOUR_NAMES.get(hex_colour, "De fábrica")


def valid(params) -> str | None:
    """None when the builder accepts these parameters, or its reason why not."""
    try:
        params.validate()
    except ValueError as e:
        return str(e)
    return None


def limits(design: Design, key: str) -> tuple[float, float]:
    """The interval a numeric knob can take with every other parameter as it is: its range, narrowed to where the
    builder's rules still hold, found by bisection from the current value at a tenth of the knob's step."""
    r = ranges(design.preset.family)[key]
    now = design.value(key)
    base = design.params()

    def ok(v):
        return valid(replace(base, **{key: v})) is None

    if not ok(now):
        return now, now
    tol = r["step"] / 10
    ends = []
    for far in (r["min"], r["max"]):
        if ok(far):
            ends.append(far)
            continue
        good, bad = now, far
        while abs(bad - good) > tol:
            mid = (good + bad) / 2
            good, bad = (mid, bad) if ok(mid) else (good, mid)
        ends.append(good)
    return ends[0], ends[1]


def fix(design: Design) -> Design:
    """The design pulled inside its ranges: each number clamped to its range, each option back to the preset's when
    unknown, then any change the builder still refuses dropped, the last first."""
    r = ranges(design.preset.family)
    changes = {}
    for key, v in design.changes.items():
        if key not in r:
            continue
        if "choices" in r[key]:
            if v in r[key]["choices"]:
                changes[key] = v
        else:
            changes[key] = min(r[key]["max"], max(r[key]["min"], float(v)))
    out = replace(design, changes=changes)
    if out.paint not in (None, *r["paint"]["choices"]):
        out.paint = None
    if out.second not in (None, *r["second"]["choices"]):
        out.second = None
    if out.livery not in (None, *LIVERIES):
        out.livery = None
    while out.changes and valid(out.params()) is not None:
        out.changes.pop(next(reversed(out.changes)))
    return out


def turn(design: Design, key: str, d: int) -> tuple[Design, str]:
    """The design with knob `key` turned one step up (d = 1) or down (d = -1), and a line for the player: empty
    when it moved, or why it did not (a range's end, or another measurement in the way)."""
    r = ranges(design.preset.family)[key]
    if "choices" in r:
        options = r["choices"]
        now = design.value(key)
        k = options.index(now) if now in options else 0
        for j in range(1, len(options) + 1):  # the next option the builder accepts
            v = options[(k + d * j) % len(options)]
            trial = _set(design, key, v)
            if valid(trial.params()) is None:
                return trial, ""
        return design, "Ninguna otra opción cabe con las demás medidas."
    span = r["max"] - r["min"]
    step = max(r["step"], round(span / 40 / r["step"]) * r["step"])  # 40 presses from end to end
    now = float(design.value(key))
    v = min(r["max"], max(r["min"], round((now + d * step) / r["step"]) * r["step"]))
    if v == now:
        return design, "Llegó al final de su rango."
    trial = _set(design, key, v)
    if valid(trial.params()) is not None:
        lo, hi = limits(design, key)
        edge = hi if d > 0 else lo
        if abs(edge - now) < r["step"] / 2:
            return design, "No cabe más con las otras medidas."
        trial = _set(design, key, edge)
    return trial, ""


def _set(design: Design, key: str, value) -> Design:
    if key in ("paint", "second", "livery"):
        return replace(design, **{key: value})
    return replace(design, changes={**design.changes, key: value})


def build(design: Design):
    """The design's Vehicle, from its family's builder."""
    from manifoldx.modeling import assets as a

    builder = {"car": a.build_car, "bus": a.build_bus, "truck": a.build_truck, "cycle": a.build_cycle}
    return builder[design.preset.family](design.params())

