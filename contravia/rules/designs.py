"""The designs a player edits in the Garaje: its knobs, their street ranges and the parameters a design builds.

A design is a catalogue preset of :mod:`contravia.rules.fleet` with some of its parameters changed, each inside
its range. A knob is one parameter the Garaje edits. Its range is the street's: the union, over every era of the
same builder, of the generator's ranges in the fleet's one table, clipped to the builder's own range, which
manifoldx publishes as ``Params.ranges()``. Every era mixes, so a slider spans the whole family rather than one
preset's era.

Parameters also constrain each other (the wheelbase fits inside the length), and manifoldx's ``validate()`` knows
those rules. :func:`limits` therefore finds the live interval of one knob by asking it, :func:`fix` pulls a whole
set back inside, and :func:`turn` moves one knob one step and reports why it stopped. Colours are the fleet's era
lists, faded as the street paints them, with the names the Garaje shows.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace

from contravia.rules import fleet

# The tabs the Salón browses by, in order.
TABS = ("Carros", "Guaguas", "Camiones", "Servicios", "Ciclos")

# The groups the Taller's rows are split into, in order.
GROUPS = ("Forma", "Frente y cola", "Ruedas", "Pintura")

# The colours the Garaje offers, built from the fleet's one table: (name, sRGB), each era in order and each
# colour faded as the street paints it. The Spanish names are the ones the Garaje has always shown.
COLOURS = tuple((name, fleet.faded(colour, era))
                for era, entries in fleet.ERA_COLOURS.items()
                for name, colour in entries)
COLOUR_NAMES = {hex_colour: name for name, hex_colour in COLOURS}

# Knob: (parameter, label, group). A parameter named "paint" or "second" is a colour of the palette, not one of
# the builder's parameters; "livery" is one of LIVERIES.
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

# A livery for a car: (name, paint areas as (u from the nose, height in metres) over the side, lightbar).
LIVERIES = {
    "Ninguno": ((), None),
    "Franja": ((((0.0, 0.62), (1.0, 0.62), (1.0, 0.70), (0.0, 0.70)),), None),
    "Policía": ((((0.0, 0.55), (1.0, 0.55), (1.0, 0.85), (0.0, 0.85)),), "bar"),
    "Taxi": ((((0.0, 0.66), (1.0, 0.66), (1.0, 0.74), (0.0, 0.74)),), "dome"),
}

# The palette material the second colour paints, per builder.
SECOND = {"car": "roof", "bus": "livery0", "truck": "body", "cycle": "box"}


def _street_union(family: str) -> dict[str, tuple[float, float]]:
    """The union, per parameter, of every fleet family's ranges for one builder (car, bus, truck or cycle)."""
    out: dict[str, tuple[float, float]] = {}
    for fam in fleet.FAMILIES.values():
        if fam.builder != family:
            continue
        for name, (low, high) in fleet.ranges(fam.name, fam.era).items():
            if name in out:
                out[name] = (min(out[name][0], low), max(out[name][1], high))
            else:
                out[name] = (low, high)
    return out


@functools.cache
def street_ranges(family: str) -> dict[str, dict]:
    """Each of a builder's knobs as ``{"min", "max", "step", "unit"}``, or ``{"choices"}`` for an option.

    A number's range is the union over every era of the family's generator ranges, clipped to manifoldx's own
    range for the parameter; its step and unit are manifoldx's. The options are manifoldx's choices, and the
    colours and car liveries the Garaje's own.
    """
    params_cls = fleet.PARAMS[family]
    built = params_cls.ranges()
    street = _street_union(family)
    out = {}
    for key, _, _ in KNOBS[family]:
        if key in ("paint", "second"):
            out[key] = {"choices": colours(family)}
        elif key == "livery":
            out[key] = {"choices": tuple(LIVERIES)}
        elif "choices" in built[key]:
            out[key] = {"choices": built[key]["choices"]}
        else:
            spec = built[key]
            low, high = street.get(key, (spec["min"], spec["max"]))
            step = spec.get("step") or 0.01
            out[key] = {"min": max(low, spec["min"]), "max": min(high, spec["max"]), "step": max(step, 0.01),
                        "unit": spec.get("unit", "m")}
    return out


def knobs(family: str, group: str | None = None) -> list[tuple[str, str, str]]:
    """The family's knobs as (parameter, label, group), all of them or one group's."""
    return [knob for knob in KNOBS[family] if group is None or knob[2] == group]


@functools.cache
def colours(family: str) -> tuple[str, ...]:
    """The colours a family may wear: the era lists, then the factory colours of its presets that are not among them."""
    out = [hex_colour for _, hex_colour in COLOURS]
    for preset in fleet.catalogue().values():
        if preset.family == family:
            for material in ("paint", SECOND[family]):
                if material in preset.palette and preset.palette[material][0] not in out:
                    out.append(preset.palette[material][0])
    return tuple(out)


def colour_name(hex_colour: str) -> str:
    """A colour's name on the Garaje's rows; a preset's own colour, which no era names, is "De fábrica"."""
    return COLOUR_NAMES.get(hex_colour, "De fábrica")


def valid(params) -> str | None:
    """None when the builder accepts these parameters, or its reason why not."""
    try:
        params.validate()
    except ValueError as e:
        return str(e)
    return None


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
    def preset(self) -> fleet.Preset:
        """The catalogue preset this design started from."""
        return fleet.catalogue()[self.base]

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
        """The current value of a knob; a colour the preset does not set falls back to the family's first swatch."""
        if key == "paint":
            return self.paint or self.preset.palette.get("paint", (colours(self.preset.family)[0],))[0]
        if key == "second":
            material = SECOND[self.preset.family]
            return self.second or self.preset.palette.get(material, (colours(self.preset.family)[0],))[0]
        if key == "livery":
            return self.livery or self.preset.livery
        return getattr(self.params(), key)

    def to_json(self) -> dict:
        """The design as the JSON file in the config folder keeps it."""
        return {"base": self.base, "name": self.name, "changes": self.changes, "paint": self.paint,
                "second": self.second, "livery": self.livery}

    @classmethod
    def from_json(cls, d: dict) -> Design:
        """The design a JSON file reads back."""
        return cls(d["base"], d.get("name", ""), dict(d.get("changes", {})), d.get("paint"), d.get("second"),
                   d.get("livery"))


def limits(design: Design, key: str) -> tuple[float, float]:
    """The interval a numeric knob can take with every other parameter as it is.

    It is the knob's range narrowed to where the builder's rules still hold, found by bisection from the current
    value at a tenth of the knob's step.
    """
    spec = street_ranges(design.preset.family)[key]
    now = design.value(key)
    base = design.params()

    def ok(v):
        return valid(replace(base, **{key: v})) is None

    if not ok(now):
        return now, now
    tol = spec["step"] / 10
    ends = []
    for far in (spec["min"], spec["max"]):
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
    """The design pulled inside its ranges.

    Each number is clamped to its range, each option is dropped when unknown, then any change the builder still
    refuses is dropped, the last first.
    """
    spec = street_ranges(design.preset.family)
    changes = {}
    for key, v in design.changes.items():
        if key not in spec:
            continue
        if "choices" in spec[key]:
            if v in spec[key]["choices"]:
                changes[key] = v
        else:
            changes[key] = min(spec[key]["max"], max(spec[key]["min"], float(v)))
    out = replace(design, changes=changes)
    if out.paint not in (None, *spec["paint"]["choices"]):
        out.paint = None
    if out.second not in (None, *spec["second"]["choices"]):
        out.second = None
    if out.livery not in (None, *LIVERIES):
        out.livery = None
    while out.changes and valid(out.params()) is not None:
        out.changes.pop(next(reversed(out.changes)))
    return out


def turn(design: Design, key: str, d: int) -> tuple[Design, str]:
    """The design with knob `key` turned one step up (d = 1) or down (d = -1), and a line for the player.

    The line is empty when the knob moved, or says why it did not: the range's end, or another measurement in the
    way.
    """
    spec = street_ranges(design.preset.family)[key]
    if "choices" in spec:
        options = spec["choices"]
        now = design.value(key)
        k = options.index(now) if now in options else 0
        for j in range(1, len(options) + 1):  # the next option the builder accepts
            v = options[(k + d * j) % len(options)]
            trial = _set(design, key, v)
            if valid(trial.params()) is None:
                return trial, ""
        return design, "Ninguna otra opción cabe con las demás medidas."
    span = spec["max"] - spec["min"]
    step = max(spec["step"], round(span / 40 / spec["step"]) * spec["step"])  # 40 presses from end to end
    now = float(design.value(key))
    v = min(spec["max"], max(spec["min"], round((now + d * step) / spec["step"]) * spec["step"]))
    if v == now:
        return design, "Llegó al final de su rango."
    trial = _set(design, key, v)
    if valid(trial.params()) is not None:
        lo, hi = limits(design, key)
        edge = hi if d > 0 else lo
        if abs(edge - now) < spec["step"] / 2:
            return design, "No cabe más con las otras medidas."
        trial = _set(design, key, edge)
    return trial, ""


def _set(design: Design, key: str, value) -> Design:
    """The design with one knob set, whether it is a colour or livery of the palette or a builder parameter."""
    if key in ("paint", "second", "livery"):
        return replace(design, **{key: value})
    return replace(design, changes={**design.changes, key: value})


def build(design: Design):
    """The design's Vehicle, from its family's builder."""
    return fleet.BUILDERS[design.preset.family](design.params())
