# The GPS {#the-gps}

The island is one thousand kilometres long and its roads are split into sixteen province files. A driver who opens the map, puts the crosshair on a street in Santiago and asks to be guided there needs one route, found in one search, that crosses every border on the way. The route also has to be the quickest by time rather than the shortest by distance, because the shortest way out of a Havana neighbourhood is through alleys, and it has to be spoken as a driver would say it: «Dobla a la derecha en Galiano».

## A node is an edge

A search over junctions cannot answer the question a driver asks, because the cost of a turn depends on which street you arrive from and which one you leave by. The router therefore searches over edges. Each directed edge of the road graph is a node of the search, and each allowed turn from one edge to the next is an arc. A one-way street is one node, a two-way street two, and a U-turn is not an arc.

The arc's cost is in seconds, and it has three parts. Driving the edge the arc enters costs its length over its free speed, times `1.5` on a service road, so a route avoids crossing a car park and only ends in one when that is the destination. The turn itself costs `3` s to the right and `8` s to the left past `30°` of angle, because a left turn crosses the oncoming traffic. The control at the end of the edge the arc leaves costs `10` s at a signal, `4` s at a stop and `2` s at a give-way sign, the average wait rather than the light's state now. A turn sharper than `150°` is not an arc at all.

The search is one `scipy.sparse.csgraph.dijkstra` over a sparse matrix of the arcs, from the car's edge to every edge of the destination's street. It runs on a daemon thread. The guide keeps the old route until the new one is in, so a reroute never blanks the panel mid-corner.

## One graph for the island

The provinces' road graphs are joined into one `RouteGraph`, kept in `route.npz` beside `index.npz`, about `42` MB for the whole island. An island edge id is the province's offset plus its local id, with the provinces in `geo.PROVINCES` order, so converting either way is one addition or one `searchsorted` over sixteen offsets. The graph keeps each edge's length, free speed, class, control, way, street name, headings at both ends and its polyline thinned to a point every `50` m, and for every arc its two island edges, its signed angle and its cost.

The provinces meet where they share an OSM node: a road that crosses a border is cut at a junction that both provinces' graphs hold, because both read the same node. Every edge ending there in one province gets an arc into every edge starting there in the other. A border crossing mapped with no shared node, where a dead end of one province lies within `15` m of a start of the other, is joined too. The two rules are what let a route from Havana to Santiago pass through Matanzas, Villa Clara and Camagüey without leaving the graph.

`route.npz` is written by the first run after the provinces, by `contravia route`, or on the router's thread the first time a route is asked for. It is rebuilt when a province file is newer than it or when `gps.ROUTE_VERSION` changes. It is written aside and moved into place whole, so a full disk never leaves a cut file that the next load would read.

## Finding the ends

The car's start is the edge nearest it within `300` m whose direction is within `90°` of its heading, or the nearest edge when none runs its way. When the game already knows where the car is on the road, it passes that edge and distance and no search of the streets is needed.

The destination is a point, so it resolves to the street nearest it. `ends_at` takes every edge within `10` m of the nearest one, both directions of a two-way street, and each is reached at the foot of the point on it. The route ends on whichever of them the search reaches first. A point farther than `300` m from every street has no route, and the game says «Aquí no hay calle cerca» before it asks.

The search also handles the awkward cases. A destination behind the car on the same one-way street is reached by going round the block: the route looks for the cheapest arc back into the start edge. A destination on the start edge ahead is the stretch between the two points.

## The instructions

A route is a list of edges. The instructions group them into steps. A junction where the route goes straight, keeps the same street name and has no other exit as straight is not a step. A junction where it turns is one, and so is one where it goes straight into a different named street with another exit as straight: that is a fork a driver must be told about.

| Angle from straight on | The words |
|---|---|
| under 30° | «Sigue recto» |
| 30° to 60° | «Dobla ligeramente a la derecha» or «a la izquierda» |
| 60° to 150° | «Dobla a la derecha» or «a la izquierda» |

A roundabout is a way whose edges are all one way and close a loop. Its steps merge into one: the router counts the exits until the route leaves it, the one taken included, and says «En la rotonda, toma la segunda salida hacia Paseo». The street name comes from the edge after the turn. A step more than `1 000` m ahead, on a named street, is said as «Sigue 2,4 km por Malecón» instead of naming the corner.

Distances are rounded the way the panel needs them: under `30` m the word is «Ahora», under `100` m it goes in steps of `10` m, under a kilometre in steps of `50` m, and past that in kilometres with one decimal and a comma. The second line of the panel gives the time left at the free speeds plus the controls' costs, the distance left and the hour of arrival, «12 min · 8,4 km · llegas 10:42». With the pilot driving it starts with the speed it wants.

## The guide

`Guide` follows the car each frame. The game locates the car once, on the road index, and hands the same place to the driving help and to the guide. The guide finds the car's edge in the route at or after the current step: found, the step advances and the distance along the edge gives the metres driven. The guide asks for a new route, and the count of reroutes goes up, when the car has been on an edge that is not a later edge of the route for `1` s, or off every road for `3` s while farther than `30` m from the route's line. The new route starts from where the car is.

The car arrives within `15` m of the end along the last edge, or within `20` m of the destination in a straight line. The panel says «Llegaste», with the place's name when the destination was one, for `4` s. A reroute that finds nothing says «No hay ruta» and tries again after `3` s. A new route shows «Ruta nueva» as a note for `1.5` s.

The guide draws three things. The arrow is a 1.6 m yellow slab with ink sides, floating `0.5` m over the car's roof and leaning `40°` toward the camera so it reads from the chase view. It points at the point on the route `clamp(2 s × speed, 20 m, 60 m)` ahead, and its heading eases toward that point with a time constant of `0.15` s. It is `55 %` opaque while the next step is more than `150` m away and solid within it, and it pulses once when the step changes. The band is a ribbon over the next `1 500` m of the route, `5` cm over the street, in the lane the driver should be in: the rightmost for a right turn and the leftmost from `80` m before a left, blended over `30` m. It is as wide as `0.8` of the lane, but never less than `2.5` minimap pixels or more than the carriageway, and it carries chevrons every `6` m. It is laid again every `200` m driven, on a thread, and swapped in whole; the minimap shows it because the minimap renders the street. The panel at the top of the screen gives the manoeuvre, the distance, the words and what is left.

Because the minimap and the map both render the street, the band needs no second overlay. On the map the route is drawn as a line over the roads with a cross where it ends, set again when the route or the map's scale changes.

## The pilot and the map

At level 0 of «Ayuda al manejo» the pilot asks the guide for the exit at each junction and follows the route, slowing for its turns and moving to the band's lane before them. A tap of the wheel within `40` m of a corner still picks the side, and the car leaves the route and the guide routes again. At the destination the pilot stops, and the stop line stays until it has stopped there. At levels 1 to 5 the driver steers and the guide only shows the way.

A destination is set from the map opened during a drive: ENTER on a point or a place offers «Ir con GPS», and «Quitar ruta» drops it. On a place, the destination is the street spot from which the place frames well, so the car stops where the photo can be taken. An agent sets one with `manifoldx emit gps x=... z=...`, or with `lat=` and `lon=`, and drops it with `manifoldx emit gps_clear`. `engine.state.gps` reports the destination, the metres and seconds left, the next step, how many times the route was found again and the message. The destination survives a hot reload in `engine.state`; the route is found again after it.

## A route on the fixture

The fixture city has a three by three grid. The block below asks the real router for the quickest way from its north-west corner to its south-east one, prints the steps in the words the panel uses, and draws the route. The same router on the built island is asked by `contravia route --from 23.1353,-82.3597 --to 20.0247,-75.8215`, which prints the instructions for Havana to Santiago, its length and its time.

```python {run}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from contravia.gps import RouteGraph, route, say, steps
from contravia.prepare import build_city, read_osm

city, _ = build_city(read_osm("../../tests/fixtures/traffic.osm"))
graph = city.graph
rg = RouteGraph.from_graphs([graph])
start = rg.start_at(0.0, 0.0, 0.0)
r = route(rg, start, rg.ends_at(200.0, 200.0))
prev = 0.0
for step in steps(rg, r):
    far, words = say(step, step.at - prev, None)
    print(f"{far:>6}  {words}")
    prev = step.at
print(f"{r.metres / 1000:.1f}".replace(".", ",") + f" km · {round(r.seconds / 60)} min")
```

```python {run continue echo=false}
fig, ax = plt.subplots(figsize=(5.2, 5.2))
poly, s0 = graph.a["g_poly"], graph.a["g_poly_start"]
for e in range(len(graph)):
    p = poly[s0[e]:s0[e + 1]]
    ax.plot(p[:, 0], p[:, 1], color="#c9c4b8", lw=2.2, solid_capstyle="round", zorder=1)
p = r.polyline(rg)
ax.plot(p[:, 0], p[:, 1], color="#ffd400", lw=7, solid_capstyle="round", zorder=2)
ax.plot(p[:, 0], p[:, 1], color="#0b0b0b", lw=0.9, zorder=3)
ax.plot(*p[0], "o", ms=9, color="#0b0b0b", zorder=4)
ax.annotate("start", p[0], xytext=(8, -18), textcoords="offset points", fontsize=9)
ax.plot(*p[-1], "X", ms=9, color="#d62828", zorder=4)
ax.annotate("destination", p[-1], xytext=(8, -18), textcoords="offset points", fontsize=9)
first = steps(rg, r)[0]
ax.set_title(f"{r.metres:.0f} m in {r.seconds:.0f} s over {len(r.edges)} edges", fontsize=10)
ax.text(0.02, 0.98, f"{first.at:.0f} m  {say(first, first.at, None)[1]}", transform=ax.transAxes,
        va="top", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="#f6f2e8", ec="#0b0b0b", lw=0.6))
ax.set_aspect("equal")
ax.set_axis_off()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

<!-- shot: contravia play --screen map --height 1200000 --at 0,0 --yaw 0 --shot, after manifoldx emit gps lat=20.0247 lon=-75.8215 | The route from Havana to Santiago, across the island -->

## The numbers

| Constant | Value | What it decides |
|---|---|---|
| `gps.RIGHT`, `LEFT` | 3 s, 8 s | What a turn past 30° costs |
| `gps.CONTROL_COST` | 0, 10, 4, 2 s | By `g_control`: none, signal, stop, give way |
| `gps.SERVICE` | 1.5 | How much longer a service road takes than its speed says |
| `gps.UTURN` | 150° | The sharpest turn that is not an arc |
| `gps.ROAD_NEAR` | 300 m | Farther than this, a point has no route |
| `gps.SAME_STREET` | 10 m | How far past the nearest edge still counts as its street |
| `gps.JOIN` | 15 m | A dead end this close to a start of another province is a junction |
| `gps.THIN` | 50 m | How often `route.npz` keeps a polyline point |
| `gps.OFF_ROUTE` | 1 s | On a wrong edge this long, route again |
| `gps.LOST`, `LOST_M` | 3 s, 30 m | Off every road this long and this far, route again |
| `gps.ARRIVE_ALONG`, `ARRIVE_NEAR` | 15 m, 20 m | When the car has arrived |
| `driving_help.STRAIGHT` | 30° | An exit within this of straight on is straight on |
| `gps.LOOK_TIME` | 2 s | The arrow points this much road ahead, within 20 and 60 m |
| `gps.BAND_AHEAD` | 1 500 m | How much of the route the band covers |
| `gps.BAND_EVERY` | 200 m | Driven before the band is laid again |

The modules are `contravia/gps.py` for the routing graph, the router, the steps and their words, the guide, the arrow and the band; `contravia/driving_help.py` for the pilot that follows a route; `contravia/hud.py` for the panel; and `contravia/menu.py` for «Ir con GPS». The graph it searches is the one the [road graph](#the-road-graph) chapter builds, and the streets it draws over are those of the [street surface](#the-street-surface).
