# Recipes {#recipes}

The island is a pipeline of layers, and a change costs what the layer that reads it costs. A rule read when a cell opens appears on the next save. A feature stored in a province file needs a command that rewrites the file. This chapter gives five recipes, each with the files it touches and what to run, and each ends on a view to take before and after the change, named in a shot comment. It closes with the checks that tell whether the change is clean.

## Which layer reads what

| The change lives in | Read when | To see it |
| --- | --- | --- |
| `contravia/regions.toml`, a rule by `municipio`, `place` or `polygon` | Every cell opens | A reload under `contravia dev`, or a new `contravia play` |
| `contravia/regions.toml`, a new rule by `tag` or `osm` id | Every cell opens, from `regions.src` | `contravia regions <pbf> --out data/cuba`, then a reload |
| `contravia/rules/species.py`, `treegrammar.py`, `trees.py` | Every cell opens | A reload |
| `contravia/rules/fleet.py`, `designs.py` | At start, and the traffic's variants while it runs | A reload, and `contravia fleet-sheet OUTDIR` to review the shapes |
| `contravia/rules/facades.py` and the `regions.toml` rule that names the new type | Every cell opens | A reload |
| The tags in a province's `building_attrs` | Every cell opens | `contravia building-tags <pbf> --data <province>.npz`, then a reload or a restart |
| `contravia/places.json` | At start | A reload, after saving a watched file to trigger it, or a restart |
| A province's streets, pavements, water, relief or cover | When the cell opens, from the province file | `contravia streets`, `contravia walks`, `contravia water` or `contravia cover`, then a reload or a restart; `contravia relief` also needs its streets and walks rebuilt after it |

The reload rules are those of [the development loop](#the-development-loop): a save to a watched file drops the app's modules and builds a new `Game` on the same state. A rebuilt province file and a data file such as `places.json` are not watched, so they need a save of a watched file or a restart.

## A region rule

A region is one `[[region]]` of `contravia/regions.toml`: a name, a shape (`where`), an optional priority and the groups of parameters it changes. `contravia/regions.py` merges four layers over each item a cell builds, later layers winning field by field and, inside a layer, the higher priority winning:

1. the defaults, `Params()`;
2. the `municipio` regions, matched by the municipio the building already carries;
3. the `tag` regions, whose polygons come from `regions.src`;
4. the named regions, by `barrio`, OSM id, `place` or `polygon`.

A field is replaced whole. A `types` table of one region is never mixed with another's weights; the winning table is used as it is. `where` names exactly one of six shapes: `municipio`, `barrio`, `osm`, `tag`, `place` (with a `radius` in metres) or `polygon` (three or more `[lat, lon]` points). Only `tag`, `osm` and `barrio` read polygons from `regions.src`, and `contravia regions` writes the first two; `municipio` matches a name, and `place` and `polygon` build their shape from `places.json` and from the coordinates.

The groups are `buildings`, `blocks`, `cemetery` and `audio`:

| Group | Keys | Values |
| --- | --- | --- |
| `buildings` | `types` | A weights table over `facades.TYPES`, or a list of cases tried in order |
| | `storeys` | `[low, high]`, over the type's own range |
| | `palette`, `wear` | Wall colours and how worn they are |
| | `portal` | Columns along main roads for eclectic and colonial buildings of two storeys or more |
| `blocks` | `fill` | `houses` (the default), `none`, or `cemetery` |
| | `parks` | `keep` (the default) or `drop` |
| `cemetery` | `pantheons` | The share, 0 to 1, of plots along the main lanes that a pantheon takes; 0.08 by default |
| | `palette` | The stone's colours |
| `audio` | `biome`, `music` | The sound's biome and the music's style here |

A case is `{ kind, above, below, types, storeys }`: the building's OSM kind, its footprint in m², strictly above and below the numbers given, and the types and storeys the case draws. The last case of a list has no condition and matches every building. A `municipio` region cannot set `parks`, because a laid park carries no municipio, and it cannot set `fill = "cemetery"`, because the cemetery fill lays out a polygon; both need a shape.

To add a rule for a municipio, add the block and reload:

```toml
[[region]]
name = "Regla, low houses"
where = { municipio = "Regla" }
priority = 5
[region.buildings]
types = { colonial = 0.6, suburban = 0.4 }
storeys = [1, 2]
```

To add a rule for a tag or an OSM id, name it in the file and write its polygon into the island:

```bash
contravia regions cuba-latest.osm.pbf --out data/cuba
```

That command writes `regions.src` (about 20 s on the reference machine), with every polygon the file names by `tag` or `osm`, every `landuse=cemetery`, and OSM's service roads and footways inside the cemeteries. The file's format is `regions.VERSION`, 1. Without `regions.src`, a `tag` or `osm` rule is left out; with the file present, a name it lacks stops the load and says which one.

The block lists the rules the shipped file resolves without a `regions.src`: its `municipio` rules, and any `polygon` rule, which needs no side file. The `Cementerios` and `Necrópolis de Colón` rules are tag and OSM rules, so they wait for the side file. A rule's rank is the tuple that decides which one wins: its layer, its priority and its place in the file, in that order.

```python {run}
from contravia.regions import Regions

book = Regions.load()
for region in book.regions:
    fields = ", ".join(sorted({k for k, v in region.fields.items() if v is not None}))
    print(f"{region.name:26} {region.shape:10} rank {region.rank}  {fields}")
print()
params = book.municipio("Guanabacoa")
print("Guanabacoa:", params.storeys, [(case.kind, case.types) for case in params.types])
```

<!-- shot: contravia play --place L0446 --screen drive --shot | Iglesia de Santo Domingo, Guanabacoa, the same street before and after a rule that changes the barrio's storeys -->

## A tree species

A tree has no stored look. Its species is drawn when its cell opens, from the zone's mix, and its form is the species' grammar. A new species therefore touches three files and no province data.

`contravia/rules/species.py` is the species' record: its name in `SPECIES`, its collision radius in `TRUNK_R`, its bark and leaf colours in `BARK` and `GREEN`, and its layer of small leaves in `FOLIAGE`. `season` reads `GREEN` for every species and has a branch for the flamboyán, the almendro and the ceiba; a species with no branch keeps a full crown all year and turns 25 % paler in March and April, which is what the laurel, the palms and the mangle do.

`contravia/rules/treegrammar.py` grows the form. `FORMS` holds one `Form` per species: a trunk height, radius, taper and lean, optional buttress or stilt roots, levels of branches (each a `count`, a `length`, a `pitch`, a `start` and a fork radius) and a crown of clusters with a size, a spread and a share of accent. A palm replaces the levels with a `Palm`: fronds round a crownshaft or a bunch of nuts. The form, the bark, the seasonal colours and the layer of small leaves are all looked up by species name; a missing entry raises the first time a cell draws the species.

`contravia/rules/trees.py` decides where it grows. `MIX` holds the five zones as weights over `SPECIES`; a species with no weight in any zone is never drawn. The weights need not sum to 100, since the table normalises them, but the shipped ones do.

| Zone | Mix |
| --- | --- |
| coast, within `COAST_REACH`, 300 m, of the sea | coco 50, almendro 35, laurel 15 |
| old centre | laurel 75, almendro 15, flamboyán 10 |
| Vedado | laurel 40, flamboyán 30, almendro 20, palma 10 |
| city park | laurel 45, flamboyán 25, palma 20, almendro 10 |
| outside the city | palma 45, flamboyán 20, laurel 20, almendro 10, ceiba 5 |

The recipe is: add the name and its four tables, add its `Form`, give it a weight in the zones where it should grow, and reload under `contravia dev`. The cells open again and grow it; no `contravia prepare` and no province rebuild. The scale, the turn and the seasonal shift are already seeded from each tree's position, and the same tree keeps them.

<!-- shot: contravia play --place L0411 --screen drive --shot | The Jardín Botánico Nacional, where a new species' form and foliage read at the distances of the three levels of detail -->

## A vehicle family

A family is one shape inside a kind: the 1950s almendrón, the 1980s saloon, the metroguagua. `contravia/rules/fleet.py` holds the ten kinds in `KINDS`, the four eras in `ERAS`, the builder and parameter class of each builder in `BUILDERS` and `PARAMS`, and the families in `FAMILIES`, keyed by name.

The recipe is one preset function and one `_car_family` call, added to `_add`:

```python
def moderno_dos() -> CarParams:
    return replace(moderno(), length=4.35, wheelbase=2.55, height=1.42)

_add(_car_family("car", "modern", "moderno_dos", moderno_dos, 8,
                 length=(4.15, 4.45), wheelbase=(2.45, 2.65),
                 track=(1.42, 1.52)))
```

The call carries the family's name, kind, era, builder, preset, variant count and its own narrower ranges over the era's. Those ranges are the one table: `fleet.ranges(family)` answers for a family, and `variant(kind, family, k)` draws the traffic's variants from it with a SHA-256 seed, so the same arguments give the same vehicle on every machine. The Garaje's sliders read the union of a builder's families through `designs.street_ranges`, clipped to manifoldx's own `Params.ranges()`, so one range changed in `fleet.py` moves the generator and the slider together.

```python {run echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from contravia.rules import fleet

families = ("almendron", "ruso", "moderno")
knobs = ("length", "width", "height", "wheelbase")
fig, ax = plt.subplots(figsize=(6.6, 3.0))
for row, family in enumerate(families):
    ranges = fleet.ranges(family)
    for k, knob in enumerate(knobs):
        low, high = ranges[knob]
        y = row * len(knobs) + k
        ax.plot([low, high], [y, y], color="#d62828", lw=5, solid_capstyle="round")
        ax.text(high + 0.06, y, f"{low:.2f} to {high:.2f} m", va="center", fontsize=7, color="#555")
ax.set_yticks([row * len(knobs) + k for row in range(len(families)) for k in range(len(knobs))],
              [f"{family}: {knob}" for family in families for knob in knobs], fontsize=7)
ax.set_xlabel("metres", fontsize=8, color="#0b0b0b")
ax.set_title("The ranges three car families draw from (fleet.ranges)", fontsize=9, color="#0b0b0b")
ax.set_xlim(0, 6.6)
ax.grid(axis="x", color="#ddd", lw=0.6)
ax.tick_params(axis="x", labelsize=7, colors="#555")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The traffic also needs a share for the new family in `FAMILY_SHARE` of `contravia/rules/traffic.py`, as a percent of its kind. A family with no share exists in the sheets and the Garaje but is never drawn on a street, because the share lookup gives it zero. Change it while playing with `{"families": {"moderno_dos": 10}}` in a `traffic_config` event, or in a traffic file.

Three constraints hold. The kind is in `KINDS`, the era in `ERAS`, and the builder in `BUILDERS`; every range lies inside the range manifoldx publishes for that parameter. A family that draws outside them fails `validate()` and says so. Names are never trademarks: they are the names people in Havana use.

Nothing in the province files changes. The traffic draws the new family on the next start, and the Garaje offers it under its kind. `contravia fleet-sheet OUTDIR --kind car` renders one review sheet, `sheet_car.png`, with a row per variant and four views of it, the front three-quarter, the left flank, the front and the rear, each cropped to what it draws.

<!-- shot: contravia fleet-sheet bench/out/fleet --kind car | The car sheet: a row per variant of each family, four views of each, on one page to review a new family against the others -->

## A building recipe

A building's plan is one of the `TYPES` of `contravia/rules/facades.py`, chosen from OSM's kind, levels and height, and from the `Params` of where it stands. A new recipe is one `Type` and a region rule that names it.

A `Type` carries the storeys it draws when OSM says nothing, the ground floor's and the other floors' heights in metres, the bay width, the parapet, the wall palette, the wear range and whether its walls along a main road stand on columns. The shipped types are `colonial`, `eclectic`, `vedado`, `tower`, `prefab`, `suburban` and `warehouse`; a tower draws 8 to 20 storeys, a prefab block 4 or 5.

```toml
[[region]]
name = "Alamar, a new block"
where = { municipio = "Habana del Este" }
priority = 30
[region.blocks]
fill = "none"
[region.buildings]
types = [{ kind = "apartments", above = 400, types = { tower = 0.3, prefab = 0.7 } },
         { types = { prefab = 1 } }]
```

The region's `types` table or case list decides when the new type is drawn; `storeys` can narrow the type's own range. `facades.plan` seeds every draw by the building's OSM id, so the same building keeps its type, height, colour and wear on every drive, and every level of detail reads that one plan.

If the recipe needs a tag the built data does not hold, `building_attrs` is the array to rewrite: kind, levels, height, use and municipio, in the order of `building_ids`. `contravia building-tags <pbf> --data data/cuba/la-habana.npz` adds it to a province prepared before and leaves the rest of the file as it is. A full `contravia prepare` writes it with everything else.

A reload shows the new type, because the plan is made when the cell opens; after `contravia building-tags`, trigger the reload with a watched save or restart. The portal's constants are the narrowest road it faces, `PORTAL_ROAD`, 8 m, the largest gap between a wall and the road's edge, `PORTAL_GAP`, 7 m, its depth, `PORTAL_DEPTH`, 3.5 m, and a column's side, `COLUMN`, 0.55 m.

<!-- shot: contravia play --place L0313 --screen drive --shot | A street by the Cathedral, where a new building type, its storeys and its portal read against the colonial fronts -->

## A collectible place

`contravia/places.json` is written outside the repository, from EcuRed and OpenStreetMap, and reviewed by hand. It is replaced whole, never edited in place. Each entry has an `id`, a `title`, a `type`, a `province`, a `municipio`, a `text`, a `photo_target`, its `lat` and `lon`, a `box` and its scores.

The box is the volume a photograph has to frame: a width `w` along an axis turned `yaw_deg` from east toward north, a depth `d` across it and a height `h`, all in metres. The `scores` are the review's own, with a `total` and a `tier`. `contravia/collect.py` turns each box into world metres by cell, finds the places near a point, and judges a framing with the constants `SMALLEST`, 0.30, `LARGEST`, 0.90, `CENTRE`, 0.60 and `INSIDE`, 0.80; a place is in reach closer than six times its largest side (`REACH_SIDES`) and never farther than `REACH_MAX`, 300 m.

A new entry needs nothing but the file, and then a restart or a reload triggered by a save to a watched file. `Places.load()` reads it at the start; the minimap and the map's «Lugares» layer draw it grey until it is photographed; the album counts it by province and kind. `contravia play --place ID` starts the car on a street that frames it, and `contravia play --place ID --screen card --shot` photographs it whatever the framing and saves the card. A real photo taken with F and SPACE is judged, and a good one saves `fotos/<id>.png` and `coleccion.json` in the config folder.

<!-- shot: contravia play --place <id> --screen card --shot | The card of a place just added: its title, its text and the photo the game saved, as the album pages through it -->

## After a change

Run the checks the layer can break. A change to a region, a building or a tree is a change to what cells draw, so the street surfaces and the walls they leave clear are the ones to check: `contravia streets-check --quick` over the province where you looked, then a full run, and `contravia walls-check` over Havana after a region rule that moves blocks or fillers. A change to the fleet is a change to the traffic's shapes and, if the handling changed, to the car: the tests in `tests/test_handling_targets.py` hold the families to the spec's targets, and `contravia fleet-sheet` shows the shapes. A new place is data: the place list has its own tests in `tests/test_collect.py`.

The chapters that hold the algorithms each recipe extends are [blocks and buildings from rules](#blocks-and-buildings), [parks and cemeteries](#parks-and-cemeteries) and [trees, grass and foliage](#trees-grass-foliage); [the fleet](#the-fleet) and [the car](#the-car) have their own chapters earlier in this part.

## The modules behind this

| Recipe | Module | What it owns |
| --- | --- | --- |
| A region rule | `contravia/regions.py`, `contravia/regions.toml`, `contravia/regions.src` | The layers, the shapes, the merged `Params` |
| A tree species | `contravia/rules/species.py`, `contravia/rules/treegrammar.py`, `contravia/rules/trees.py` | The colours, the grammar, the zone mixes |
| A vehicle family | `contravia/rules/fleet.py`, `contravia/rules/designs.py` | The kinds, the presets, the one table of ranges |
| A building recipe | `contravia/rules/facades.py`, `contravia/rules/buildings.py`, `contravia/buildingtags.py` | The types, the plan and the massing, the tags |
| A collectible place | `contravia/places.json`, `contravia/collect.py`, `contravia/album.py` | The list, the boxes and the framing, the collection |
