# The road graph {#the-road-graph}

OpenStreetMap stores a street as a way: an ordered list of nodes, with tags that say what the street is. A way is a good drawing instruction, and a poor network. Traffic needs to know which way a car may go, the GPS needs the cost of a turn, and the street surface needs every junction to be one place. `contravia prepare` therefore reads the ways once and turns them into a graph of junctions and directed edges, stored as flat arrays named `g_*` in the province file. Everything after that, from the street surface to the traffic and the route planner, works on the graph.

## What is read

Only the road classes the game can drive survive the read. `contravia/prepare.py` holds their drawn widths, and any `highway` value not in the table is dropped, so a footway or a railway never reaches the graph.

| `highway` | width (m) |
| --- | --- |
| motorway | 14 |
| trunk, primary | 10 |
| secondary | 8 |
| tertiary, unclassified, residential, living_street | 6 |
| service | 4 |

The `_link` variants share their class's width. For each drivable way the reader keeps the way's OSM id, its class, the node ids in order, the projected points, and the eight tags the graph reads: `oneway`, `lanes`, `maxspeed`, `junction`, `bridge`, `tunnel`, `layer` and `name`. It also keeps the nodes tagged `traffic_signals`, `stop` or `give_way`, and the restriction relations whose `via` is a node, because both change what a car may do at a junction. The municipio boundaries (`admin_level` 6), and the place areas and points that name a barrio, are read in the same pass.

## Junctions and edges

The first step is to decide what a junction is. The reader counts every node of every way, and counts a way's first and last node twice. A node with a count of two or more is a junction: the node where two ways meet, a node one way uses twice, or either end of a way. Each way is then cut at its junctions, so every piece runs from one junction to the next with no interruption in the middle.

Each piece becomes one edge per direction the way allows. A two-way street gives two edges, one each way, and a one-way street gives one. The rules come from `contravia/graph.py`:

| tags | directions |
| --- | --- |
| `oneway=-1` or `reverse` | backwards only |
| `oneway=yes`, `true` or `1` | forwards only |
| `oneway=no` | both ways |
| `junction=roundabout` or `circular`, or `highway=motorway` | forwards only |
| anything else | both ways |

A piece shorter than half a metre is dropped, since it has no direction worth keeping.

## Width and speed

The drawn width follows the `lanes` tag when there is one: each lane is `LANE` = 3.3 m and the drawn road adds a metre of edge. A one-way road is that wide; a two-way road is at least as wide as its class, and wider when its lanes need it. A one-way road without a `lanes` tag is one lane wide, or two lanes when its class is 8 m or wider.

Speed comes from `maxspeed`, with the first number taken and miles per hour converted. Without the tag a primary or secondary road is 50 km/h, a service road 25 km/h and every other class 40 km/h. The graph stores it in metres per second, as `g_vfree`.

## Levels

`graph.layer` reads the `bridge` and `tunnel` tags into a level. A bridge is at least level 1, a tunnel at most level -1, and OSM's `layer` value orders stacked structures. A `building_passage`, a `culvert` and an `avalanche_protector` are not tunnels the road goes down into, and an abandoned bridge is not a bridge: none of them leaves the ground. The level is not a height. It says what crosses what, and the street surface builds each level apart, as [Bridges and tunnels](#bridges-and-tunnels) explains.

## The part a car can use

A network is only drivable where a car can enter a street and leave it again. OSM has plenty of pieces that fail that: a car park entered by a one-way road with no way back is the common one, and an isolated piece of service road is another. The graph keeps the largest strongly connected part of the network, the set of junctions that can reach each other.

The search starts from the junction nearest the Capitolio, the projection's origin, and from about two dozen junctions spread over the province by distance from their median centre. Each start gives the junctions reachable from it and able to reach it; the largest such set wins, and the search stops early once it holds more than half of the candidate junctions. Edges with an end outside that set are dropped, so a car park entered by a one-way road with no way back disappears rather than becoming a trap.

## Province borders

The island is built one province at a time, and a road does not stop at a provincial line. `prepare` builds each province's graph over the province plus `GRAPH_MARGIN` = 2 000 m, inside the box the OSM reader already covers. Two masks decide the edges:

- an edge is kept when its midpoint is inside the province;
- the strongly connected search runs over the edges in the margin as well, so a road that continues into the next province still counts as leading somewhere.

The result is that a road crossing the border is cut at a junction both provinces' graphs hold, the same OSM node, and that node is where the GPS joins the two graphs. The margin does not keep the outside half of the road: only the edges with their midpoint inside the province are stored. A one-way piece that runs into the border therefore has no turn out of it in that province's graph. A traffic car drives off its end and the flow empties it. Using a shared node as the cut point is what lets the GPS change province without a gap.

## Turns

A turn is a pair of edges, one arriving at a junction and one leaving it. The graph stores every allowed pair in `g_turn_in` and `g_turn_out`, with a weight in `g_turn_w`: 1 for an ordinary street, 2 for a tertiary or secondary exit and 3 for a primary, trunk or motorway exit. Traffic uses the weight to prefer the bigger road.

Two rules remove turns. The U-turn back along the same piece of street is not offered when the junction has another way out, so a car turns around only at a dead end. An OSM restriction removes the turns it forbids: a `no_*` restriction drops the pairs it names, and an `only_*` restriction keeps only the turns it allows.

## What each edge carries

The arrays are flat and aligned, one row per edge, so a lookup along a street is an index into an array rather than a walk over objects.

| array | meaning |
| --- | --- |
| `g_from`, `g_to` | the junctions at the edge's ends, as indices into `g_junction` |
| `g_len` | length in metres |
| `g_poly`, `g_poly_start` | the centreline, sampled at least every `SAMPLE` = 10 m, so heights and traffic positions follow the road |
| `g_lanes` | lanes in this direction |
| `g_width` | drawn width in metres |
| `g_vfree` | free speed in metres per second |
| `g_class` | index into `CLASSES`, the nine drivable classes |
| `g_layer` | the structure level, 0 on the ground |
| `g_control` | 0 none, 1 signal, 2 stop, 3 give way, at the edge's end |
| `g_way` | the OSM way the edge came from |
| `g_junction`, `g_junction_node` | junction positions in metres, and their OSM node ids |
| `g_mid`, `g_cell` | the midpoint and the 200 m cell that holds it |
| `g_muni`, `g_muni_names` | the municipio of the midpoint, -1 outside every one |
| `g_built` | area of the buildings whose centre lies within `BUILT_RADIUS` = 100 m of the midpoint, in m² |
| `g_name`, `g_names`, `g_between` | the street's name and the names of the streets crossing it behind and ahead |
| `g_barrio`, `g_barrio_names` | the barrio of the midpoint |

A barrio comes from the smallest place area that holds the midpoint. When no area holds it, the nearest named place point takes it if the midpoint is within that kind's reach: 1 800 m for a suburb, 700 m for a neighbourhood or a quarter. The consejos populares, OSM's `admin_level` 9, fill what is left. `contravia/address.py` turns the street name, the crossing streets and the barrio into the address line the HUD shows.

One array is written later. `g_poly_y` holds the road's height at every polyline point, taken from the street surface once `contravia prepare` has built it, so the traffic drives at the same heights the wheels stand on.

## The graph of a small place

The test fixture is a 3 by 3 grid with a primary row, a one-way row and a service road into a car park. It shows the whole transformation in one picture. The left panel is what OSM holds, a centreline per way. The right panel is the graph: every piece cut at its junctions, one edge per direction, and the car park gone because no car could leave it.

```python {run echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

from contravia.geo import to_local
from contravia.graph import CLASSES
from contravia.prepare import build_city, read_osm

raw = read_osm("../../tests/fixtures/traffic.osm")
city, _ = build_city(raw)
g = city.graph
a = g.a
COLOUR = {"primary": "#d62828", "residential": "#3a3a3a", "service": "#9a9a9a"}

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.6))
for hw, pts in raw.roads:                       # the OSM ways as read: a centreline per way
    xz = to_local(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    left.plot(xz[:, 0], xz[:, 1], color=COLOUR[hw], lw=2.0)
    if hw == "service":
        left.annotate("car park:\nno way out, dropped", xy=xz[-1], xytext=(xz[-1, 0] + 10, xz[-1, 1] + 60),
                      fontsize=7, color="#777", arrowprops=dict(arrowstyle="->", color="#999"))
for e in range(len(g)):                          # the graph: one edge per direction, cut at junctions
    q = a["g_poly"][a["g_poly_start"][e]:a["g_poly_start"][e + 1]]
    cls = CLASSES[a["g_class"][e]]
    right.plot(q[:, 0], q[:, 1], color=COLOUR[cls], lw=1.0 + a["g_width"][e] / 5)
    k, u = len(q) // 2, q[len(q) // 2 + 1] - q[len(q) // 2]
    right.annotate("", xy=q[k] + u * 2.5, xytext=q[k], arrowprops=dict(arrowstyle="-|>", color=COLOUR[cls]))
right.plot(a["g_junction"][:, 0], a["g_junction"][:, 1], "o", ms=4, color="#0b0b0b")
for name, ax in (("OSM ways", left), ("the graph", right)):
    ax.set_title(name, fontsize=9, color="#0b0b0b")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()
fig.legend(handles=[Line2D([], [], color=c, lw=2.5, label=k) for k, c in COLOUR.items()], loc="lower center",
           ncol=3, frameon=False, fontsize=8)
print(f"{len(raw.roads)} ways become {len(g)} edges between {len(a['g_junction'])} junctions, "
      f"with {len(a['g_turn_in'])} turns")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The printed line counts what the picture shows: nine ways, twenty-two edges, nine junctions and thirty-four turns. The east-west row at the top is primary and red. The row below it is one-way east, so it has half the edges of the other streets, and the arrows on its pieces all point the same way. The car park's entry and loop exist in OSM and nowhere in the graph.

## Who uses it

`contravia prepare` builds the graph with each province, and `contravia build`, the first run, builds it from the shipped sources with the same calls. The traffic runs on the province's graph and starts over when the car crosses into another province. The GPS joins the sixteen provinces' graphs at their shared nodes into one routing graph, kept in `route.npz`. The street surface is built from the graph rather than from the raw ways, which is why two carriageways can become one avenue and why a junction is a single place. The modules are `contravia/graph.py` for the graph and `contravia/prepare.py` for the read and the province clip; `contravia/geo.py` holds the projection and the 200 m cell.
