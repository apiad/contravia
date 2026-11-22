# The fleet {#the-fleet}

Havana's traffic is a museum on wheels: 1950s American sedans with fins, Soviet saloons, a few modern hatchbacks, long low-floor city buses, coaches between provinces, lorries with wooden flatbeds, electric tricycles, cycle rickshaws and bicycles. The game draws all of them, with no model file and no photograph, and lets the player take one into the Garaje and change it. This chapter is about how a vehicle is described so that the traffic, the review sheets, the physics and the player's own designs all agree.

## Kinds, families and variants

The fleet has three levels. A kind is what the traffic counts and what a player calls a sort of vehicle: there are `KINDS`, 10 of them, from `car` and `taxi` to `bike`, `police` and `fire`. A family is one shape inside a kind: the 1950s almendrón, the 1980s three-box saloon, the city bus. A variant is one parameter set of a family, with its colours, drawn from a seed.

`contravia/rules/fleet.py` holds `FAMILIES`, 23 of them, and between them 62 variants:

| Kind | Variants | Families |
| --- | --- | --- |
| car | 24 | almendron 8, ruso 4, moscovita 4, moderno 4, huevito 4 |
| taxi | 8 | compartido 4, estatal 4 |
| bus | 6 | metroguagua 2, guagua_vieja 2, interprovincial 2 |
| truck | 10 | camion_80, furgon, cisterna, volquete, pasajeros, 2 each |
| moto | 5 | moto_electrica 3, moto_dos_tiempos 2 |
| tricycle | 2 | triciclo_carga 1, triciclo_pasajeros 1 |
| bicitaxi | 2 | bicitaxi 2 |
| bike | 2 | bicicleta 2 |
| police | 2 | policia 2 |
| fire | 1 | bomberos 1 |

Every family wears an era, and the era decides its styling, its ranges and its colours: `1950s`, `80s`, `modern` and `services`, the last for a livery painted over another era's car.

```python {run}
from contravia.rules import fleet, traffic

counts = {kind: sum(f.variants for f in fleet.FAMILIES.values() if f.kind == kind) for kind in fleet.KINDS}
print(counts, "=", sum(counts.values()), "variants")
print(len(traffic.looks()), "looks once every variant is painted")
car = fleet.variant("car", "moderno", 0)
print(f"moderno 0: {car.params.length:.2f} m long, {len(car.colourways)} colourways, {car.colourways[0]['paint'][0]}")
```

```python {run continue echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

ERA = {"1950s": "#d62828", "80s": "#ffd400", "modern": "#7d7a70", "services": "#0b0b0b"}
rows = [f for kind in fleet.KINDS for f in fleet.FAMILIES.values() if f.kind == kind]
fig, ax = plt.subplots(figsize=(7.2, 6.6))
for y, fam in enumerate(rows):
    ranges = fleet.ranges(fam.name)
    ax.barh(y, fam.variants, height=0.66, color=ERA[fam.era], edgecolor="#0b0b0b", lw=0.5)
    low, high = ranges.get("length", ranges["wheelbase"])
    note = f"{low:.2f}–{high:.2f} m" + (" long" if "length" in ranges else " wheelbase")
    ax.text(fam.variants + 0.15, y, note, va="center", fontsize=6.3, color="#6f6b62")
ax.set_yticks(range(len(rows)), [f"{f.name} · {f.kind}" for f in rows], fontsize=7)
ax.invert_yaxis()
ax.set(xlabel="variants", xlim=(0, 11))
ax.grid(axis="x", color="#e4e2dc", lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(labelsize=7)
ax.legend(handles=[Patch(color=c, label=e) for e, c in ERA.items()], loc="lower right", frameon=False, fontsize=7)
ax.set_title(f"{len(fleet.FAMILIES)} families, {sum(f.variants for f in rows)} variants, by era", fontsize=9)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## A family and its preset

A family names the manifoldx builder that makes its body (`build_car`, `build_bus`, `build_truck` or `build_cycle`), a reference preset, the number of variants it draws and the ranges it draws them from. The preset is a function returning the builder's parameter object, and it is the only place those numbers exist: the traffic draws from it, the review sheet draws from it, the Garaje shows it as a preset, and the physics of [the car](#the-car) measures the built body against it to scale the mass.

The five car families are the almendrón, a long low sedan with chrome bumpers and fins; the ruso, an upright 1980s saloon; the moscovita, a softer 1970s saloon; the moderno, a three-box saloon or a hatchback, four draws in ten taking the hatchback body; and the huevito, a short tall hatchback. The three buses are a 12 m low-floor city bus, an older high-floor one and an intercity coach. The trucks are a flatbed, a box van, a tanker, a dump truck and a passenger truck with a canvas bed. The cycles are an electric moto, a two-stroke moto, a cargo tricycle, a passenger tricycle, a bicitaxi and a bicycle.

Three families have no shape of their own. The shared taxi is an almendrón with an amber sign; the state taxi and the police car are painted over the ruso and the moderno. Such a family lists its `bases` and carries the union of their ranges, but each of its variants still draws from one base family, because a box that mixes two eras holds measurements that do not build together.

## A variant is a draw

`fleet.variant(kind, family, k)` returns variant `k` of a family: its parameters and its colourways. The draw is a seeded random generator; the seed is the first eight bytes of a SHA-256 of the kind, the family, the base family it paints over, `k` and the attempt. Python's own `hash` is salted per process and would give a different fleet on every run, so it is not used.

The generator draws each numeric parameter of the family's ranges in the alphabetical order of the parameter's name, so the stream does not depend on how the table was written, and then picks the era's style options by their weights: a two-tone roof half the time on a 1950s car, whitewalls six in ten, a spoiler on one modern car in five. Two measurements are never drawn, because the preset or a style option fixes them: `STYLE_ONLY` names them, and they stay in the table so the Garaje can still offer their whole ladder. A style option that fixes a measurement overrides the draw, as the 80s fix their fins at zero. The drawn values go over the preset with `replace`.

Manifoldx's `validate()` then decides whether the parameter set builds: the wheelbase plus the front overhang fits inside the length, the wheelbase spans its two wheels, the sill sits below the belt and the belt below the roof, and the stations along the body keep their order from the cowl to the hatch. A draw it refuses is thrown away and drawn again from a derived seed, up to `ATTEMPTS`, 12, times. A family that keeps failing is a bug in its ranges, and the error says which family and variant, rather than hiding it.

The ranges narrow from the era to the family. A car's era box is 4.80 to 5.40 m long in the 1950s, 3.70 to 4.40 in the 80s and 4.30 to 4.70 in the modern era, and a family keeps its own share of it: the almendrón draws 4.90 to 5.40 m, the ruso 3.70 to 3.98, the huevito 3.60 to 4.00. Across families, a bus runs from 8.80 to 13.40 m and a truck from 6.60 to 10.00.

## Colourways

Every variant comes painted in three colourways (four for the bicitaxi, whose canopy has four colours). A car's colourway is a body colour from its era's six and a second colour for the roof or a stripe, picked so that the two differ by at least 0.45 in the sum of their channels; a two-tone that reads is the point. The era's paint is faded toward the sun-bleached dust of Havana before it is used: 30 % for the 1950s, 22 % for the 80s, none for a modern car. A bus is cream with a band in one of six accents, from the metroguaguas' green and cyan to a coach's blue and red. A truck takes its cab's era colour over a body of its trade: a wooden flatbed, a grey box, a silver tank. A police car is white with a blue band low on the flanks and blue front doors, and one of four blues on the roof bar; a fire engine is one of three reds.

The traffic paints every variant in every colourway ([Traffic](#traffic)): 62 variants give 188 looks, and the flow picks one for each car entering a street, weighted by the kind's share of the traffic and then by the family's share of its kind. That is why a street can hold two almendrones that are not the same car.

## The Garaje's ranges

`contravia/rules/designs.py` gives the player a design: a catalogue preset with some of its parameters changed, each inside a range that is guaranteed to build.

The catalogue has 15 presets in five tabs: `Carros`, `Guaguas`, `Camiones`, `Servicios` and `Ciclos`. Each carries its palette, its seats (80 for the metroguagua, 6 for the almendrón) and the phrase the Garaje shows about where it belongs. The cycles are listed but not drivable until riders exist.

A design's knobs are per builder: 20 for a car, 12 for a bus, 13 for a truck and 11 for a cycle, grouped as `Forma`, `Frente y cola`, `Ruedas` and `Pintura`. A numeric knob's range is the union, over every family of that builder, of the generator's ranges, clipped to the interval manifoldx publishes in `Params.ranges()`, with manifoldx's own step and unit. For a car that gives a length from 3.60 to 5.40 m in steps of 0.01, a width from 1.55 to 2.00, a height from 1.38 to 1.65, a wheelbase from 2.35 to 3.10 and a wheel radius from 0.27 to 0.36; a bus spans 8.80 to 13.40 m of length. Options come from manifoldx's choices, and the colours are the era lists plus the factory colours of the catalogue's presets.

Ranges alone are not enough, because the parameters constrain one another. `limits` finds a knob's live interval by asking `validate()` and bisecting from the current value down to a tenth of the step. `turn` moves a knob one press, a step of a fortieth of its span rounded to the grid, so a full slider takes about forty presses, and answers with a line when it cannot: `Llegó al final de su rango.` at the end of the range, `No cabe más con las otras medidas.` when another measurement is in the way. `fix` pulls a whole design back inside its ranges, dropping the changes the builder still refuses, the last one first; it runs on every design loaded from disk, so a design saved before a change of rules comes back legal.

A design is saved as JSON in the `garaje` folder of the user's configuration folder, one file per name, newest first. The file holds the preset it started from, the changes, the two colours and the livery; the palette is not geometry, so repainting a car costs no rebuild.

## Building one

`Variant.build` hands the parameters to the family's builder and gets a manifoldx vehicle back. Building takes seconds, longer than a frame, so `contravia/garage.py` builds on one worker thread and the screens ask it each frame whether their vehicle is ready; until it is, they keep drawing the one before. Built vehicles are cached by their design's key, the parameters plus the livery and its colour, and `CACHED`, 12, of them are kept, so going back to a preset or undoing a change costs nothing. The cache key leaves out the paint on purpose: a colour is a palette entry, not a shape.

`contravia fleet-sheet OUTDIR` renders the fleet for review, one PNG per kind with a row per variant in four views, offscreen, with the camera placed from each vehicle's own size so a bicycle and a coach read at the same size on the page. It renders through manifoldx's engine under the game's studio environment, so it needs a graphics card, and it is not drawn in this book:

<!-- shot: contravia fleet-sheet out --kind car | One sheet of the fleet: a row per variant of the almendrón, the ruso, the moscovita, the moderno and the huevito, in four views -->
