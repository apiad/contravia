# Traffic {#traffic}

A Havana that is empty at eight in the morning is not Havana. The game needs cars that thicken on the avenues at the rush hour, thin out at three in the morning, queue behind a closed street and keep doing it while the player drives somewhere else. It also needs the cars near the player to be real cars, on their lanes, braking for each other, because those are the ones the player sees and hits.

One model cannot do both. A car that follows every rule costs a body and a frame; a hundred thousand of them do not fit. So traffic is two layers. Underneath, a flow layer counts vehicles on every street of the province and moves the counts once per simulated second. On top, a car layer drives real cars on the edges closest to the player. The rule that keeps them one city is simple: on the edges the car layer owns, the cars are the truth, and everywhere else the flow is.

## The flow layer

The flow layer is one float per directed edge: `n`, the vehicles on it. A step of one game second does four things, all with numpy over the graph's arrays.

What an edge can send is its demand: the vehicles that reach its end this second, `n · min(1, v_free · dt / length)`, capped by its capacity, `cap · dt`. The capacity is `0.5` vehicles per second per lane, and `0.3` on an edge that ends at a signal or a stop, where the queue leaves more slowly. What it holds at most is its jam, one vehicle per `7` m of lane.

Each edge's demand splits over its turns by the graph's turn weights, so a turn onto an avenue takes more than a turn into an alley. A receiving edge scales down everything coming to it by `min(1, room / want)`, where the room is `jam - n`. The junctions are first in, first out: when one of an edge's turns cannot take vehicles, the whole edge waits, at the pace of its most constrained turn. That is what makes a queue grow backwards from a closed street, as the car at its head blocks the ones behind.

Vehicles enter and leave on every edge, not at a few origins and destinations. Each edge has a source weight from the built area within `100` m of its midpoint and from its class: a residential street feeds `1.0`, a tertiary `0.6`, a primary `0.2`, a motorway `0.0`. The demand's target, the number of vehicles that should be moving at that hour, is spread over those weights and added where there is room. Vehicles are taken out at a rate set by `trip_minutes`, `15` by default, so a vehicle stays on the network about a quarter of an hour. The two municipios that `attract` names, Plaza de la Revolución at `1.5` and La Lisa at `1.3`, pull trips in during the morning peak, from `06:00` to `10:00`, and send them out again from `15:00` to `19:00`.

A step is a handful of passes over the province's edges. `FlowRunner` runs it in a daemon thread five times a second, one simulated second at a time, up to `600` steps per tick, so a laptop that was asleep does not freeze the game. The game starts by warming the flow for `30` simulated minutes before the clock's time, in coarse `20` s steps, and `contravia flowmap` warms it for `60` minutes.

## How many vehicles

There are no hourly traffic counts for Havana, so the level is an estimate built from published figures, and every number of it is a setting in `contravia/rules/traffic.py`. The defaults are a population of `1 814 207`, `2.3` trips per person per day, the modal split of the city's sustainable mobility plan (walking `46.2 %`, public transport `43.6 %`, private car `6.0 %`, motorbike `3.2 %`, bicycle `1.1 %`), `1.5` people per car, `1.2` per motorbike and `4.0` per shared taxi, `5 000` bus trips, and freight adding `25 %` on top. A weekday comes to `495 615` vehicle trips with the defaults.

The number that matters for the street is not the trips of a day but the vehicles moving at once. That is the trips starting in the hour, `hour_share`, times `trip_minutes / 60`, split over the ten kinds of the fleet by `mix`. The hour curves have their peaks where Havana reports them, at `7–9` and `16–18`; the rest of the shape is an assumption. Friday is `1.05` of a weekday, Saturday `0.80` and Sunday `0.60`. With the defaults, a weekday at `08:30` has about `10 500` vehicles moving, and at `03:00` about `430`. The mix is not only cars: `30 %` of the vehicles are motorbikes, `12 %` shared taxis and `7 %` tricycles, which is what a Havana street looks like.

```python {run echo=false}
import io
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from contravia.rules.traffic import TrafficConfig, moving, vehicle_trips

cfg = TrafficConfig()
monday = datetime(2026, 11, 9)  # a Monday
fig, ax = plt.subplots(figsize=(8, 3.4))
for offset, label, colour in ((0, "Mon–Thu", "#d62828"), (4, "Fri", "#e07a1f"),
                              (5, "Sat", "#8a8f3c"), (6, "Sun", "#31597a")):
    day = monday + timedelta(days=offset)
    y = [sum(moving(cfg, day.replace(hour=h, minute=30)).values()) for h in range(24)]
    ax.plot(range(24), y, lw=1.8, color=colour, label=label)
ax.set(xlabel="hour of the day", ylabel="vehicles moving")
ax.set_xticks(range(0, 24, 3))
ax.grid(color="#eeeeee", lw=0.6)
ax.legend(frameon=False, fontsize=9)
print("weekday trips:", round(sum(vehicle_trips(cfg, "weekday").values())))
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

Every setting can change while the game runs. A JSON file given to `contravia play --traffic my.json` names only the keys to change, and `manifoldx emit traffic_config population=3000000` does the same from outside. The flow layer reads the settings on every step, so the sources change at once and the street counts follow within about one trip time. `engine.state.traffic` reports the settings and what they give: trips per day, vehicles moving now by kind, the flow's total and the step time. Traffic is off in a plain `contravia play`; `--with-traffic` turns it on, and a `--traffic` file does too.

## The cars near the player

The car layer keeps each car as a row of arrays, never an object: the edge it is on, how far along it its front is, its lane, its speed, its look (which gives its kind), the edge it takes next, and how long it has waited at the junction. A look is a kind, a family, a variant and a colourway, drawn by a seeded hash of the edge and the minute, so the same street at the same minute always draws the same vehicles. The lengths and widths are per kind, from a `1.8` m bicycle to a `12` m bus.

Every frame each car finds the car ahead of it in its lane, or the stop line, or the player, and accelerates by the intelligent driver model: toward the edge's free speed, with a desired gap of `2` m plus `1.2` s of headway and a braking term that grows as the gap closes. The accelerations are `1.5` m/s² and `2.0` m/s². A car never advances into the car ahead: the step clips the move to the free gap.

At the end of an edge the car asks to enter the junction when it is within `LOOK`, `30` m, of it. At a signal it obeys the light: each signal has two groups, 30 s of green and 3 s of amber each, a cycle of `66` s, with the phase offset from the junction's OSM node. It goes on amber only when it can no longer stop. Anywhere else it waits for a car with priority: the bigger street ranks higher, a stop or give-way sign takes ten ranks off, and between equal streets the car from the right goes first. A car that has waited `8` s goes anyway, so a junction never locks up. A signal is obeyed however long the wait.

A car that is not the head of its lane changes lane by MOBIL: when the other lane lets it gain more than `0.4` m/s² after paying `0.3` of what the car behind loses, and never into a gap under `2` m or one that would make the car behind brake past `3` m/s². A car with a turn coming moves into the turn's lane and takes it anyway `30` m before the end. Decisions happen every `0.2` s, alternating sides, so two cars never take the same gap from both sides at once.

The drawn path is smooth. The sideways offset eases toward the lane's at `1.1` m/s, about three seconds a lane, and up to `7` m before the end of an edge and after the start of the next are joined by a quadratic curve, whose control point is where the two directions meet when they are not nearly parallel. A car neither jumps across a junction nor turns at every vertex of the street.

## Where the two layers meet

The car layer owns the edges whose cells lie within the radius of the player: `400` m at Alto and above, `300` m at Medio and `200` m at Bajo. Only when the player has moved `10` m is the circle worked out again, and at most `200` new edges are asked for in one update, nearest first, so a teleport does not stall a frame.

When an edge enters the circle, the flow hands over its count, its speed and its queue. The queued vehicles become stopped cars at the end of the edge, `7` m apart; the rest are spread along it at the edge's speed. Most streets hold a fraction of a vehicle, so the count is rounded at random and the difference is kept, which is why a block that averages `0.2` cars gets one about one time in five and the total stays exact. A vehicle the flow sends into an owned edge arrives as a car at its start when there is room. A car that drives onto an edge out of the circle is removed and adds one to that edge's count. A car that reaches the end of an edge with no turn out of it has reached the province's border, and it leaves the graph.

No vehicle is ever in both layers. `TrafficLayer.total` adds the flow's own edges, the cars, the fractions kept, the vehicles waiting at an edge's start and the ones in transit between the two layers, which is the property the coupling is tested on: drive the circle across a grid and back and the total does not change.

## Closing a street

`manifoldx emit close x=0 z=0 minutes=40` closes the edge nearest that point in local metres, or `edge=1234` names one. The closed edge keeps its cars where they are and takes nothing in. The junction before it sends only a tenth of its usual share into a blocked approach, `AVOID = 0.1`, so the rest of the traffic goes around. The queue grows backwards at the pace of the edge before it and drains over minutes when the street reopens.

In the car layer, the cars on the closed edge stop and stay stopped. When the edge leaves the circle their positions and looks are stored, and when it comes back they are put back, stopped, so a crash the player drove away from is still there on the way home. `engine.state.traffic` reports how many edges are closed. A hard hit by the player does not close an edge: the car that is hit leaves the layer as a rigid body in `contravia/crashes.py`, is pushed and spun, and a broken one becomes a wreck that the traffic stops behind or changes lane around until the player is `150` m away.

## Two layers on the fixture

The fixture city has twenty-two edges, so the picture below scales the demand down to about `15` vehicles at eight in the morning, the way the tests do. The colours are the ones `contravia flowmap` uses, from dark at no traffic to red at `600` vehicles per hour. At `03:00` the streets are nearly black; at `08:00` the centre column carries the most.

```python {run}
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from contravia.clock import FlowRunner
from contravia.flow import Flow
from contravia.flowmap import COLOURS, FULL, colour, per_hour
from contravia.prepare import build_city, read_osm
from contravia.rules.traffic import TrafficConfig, moving

city, _ = build_city(read_osm("../../tests/fixtures/traffic.osm"))
graph = city.graph
tuesday = datetime(2026, 11, 17, 8, 0)
cfg = TrafficConfig().updated(level=15 / sum(moving(TrafficConfig(), tuesday).values()))


class Fixed:
    def __init__(self, when):
        self.when = when

    def now(self):
        return self.when


views = {}
for hour in (3, 8):
    when = tuesday.replace(hour=hour)
    flow = Flow(graph, cfg)
    runner = FlowRunner(flow, Fixed(when))
    runner.warm_up(minutes=60)
    views[hour] = (runner.view, per_hour(runner.view, flow.length))
print({h: round(v[0].n.sum()) for h, v in views.items()})
```

```python {run continue echo=false}
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
poly, start = graph.a["g_poly"], graph.a["g_poly_start"]
for ax, (hour, (view, flow_hour)) in zip(axs, sorted(views.items())):
    rgb = colour(flow_hour / FULL) / 255
    for e in range(len(graph)):
        p = poly[start[e]:start[e + 1]]
        ax.plot(p[:, 0], p[:, 1], color=rgb[e], lw=4.5, solid_capstyle="round")
    ax.set_aspect("equal")
    ax.set_axis_off()
    shown = round(view.n.sum())
    ax.set_title(f"{hour:02d}:00 · {shown} vehicle" + ("s" if shown != 1 else ""), fontsize=10)
cmap = ListedColormap(COLOURS / 255)
fig.colorbar(ScalarMappable(norm=Normalize(0, FULL), cmap=cmap), ax=axs, fraction=0.025,
             label="vehicles per hour")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The same flow layer, at the same minute, is what `contravia flowmap` draws to a PNG without a window:

```bash
contravia flowmap traffic-08.png --time "Tue 08:00" --data data/cuba/la-habana.npz
```

<!-- shot: contravia flowmap traffic-08.png --time "Tue 08:00" --data data/cuba/la-habana.npz (then the same with traffic-03.png --time "Tue 03:00") | Havana's flow layer at the morning peak and at three in the morning -->

The file defaults to `la-habana.npz`, the time to `Tue 08:00`, and `--traffic my.json` takes another set of demand settings, so two settings compare as two pictures. `contravia bench --time "Tue 08:00"` adds the flow step time and the number of cars to its line.

## A province at a time

The traffic runs on the graph of the province the car is in, and it starts over when the car crosses into another. `Game._follow_province` stops the flow thread, clears the drawn cars and calls `_start_traffic` again on the new province's graph, which warms it for `30` minutes. The provinces meet at shared OSM nodes, so the cars a player leaves behind at the border are not the cars on the other side; the flow is a model of one province's streets, not of the island.

A data file built before layer 2 has no `g_*` arrays. The game plays with no traffic and says so: `no road graph in the data file: run contravia prepare again`.

## The numbers

| Constant | Value | What it decides |
|---|---|---|
| `flow.CAPACITY` | 0.5 veh/s per lane | What an edge can send per second |
| `flow.CAPACITY_CONTROLLED` | 0.3 veh/s per lane | The same at a signal or a stop |
| `flow.SPACING` | 7 m | One vehicle per this much lane in a jam |
| `flow.WALKING` | 1.4 m/s | The speed a jammed edge still reports |
| `flow.AVOID` | 0.1 | The share of the usual turn into a closed approach |
| `flow.MORNING`, `AFTERNOON` | 6–10, 15–19 | The hours attraction pulls trips in, then sends them out |
| `traffic.ACCEL`, `BRAKE` | 1.5, 2.0 m/s² | The intelligent driver model |
| `traffic.S0`, `HEADWAY` | 2 m, 1.2 s | The gap a car keeps |
| `traffic.LANE_WIDTH` | 3 m | The widest lane |
| `traffic.GREEN`, `AMBER` | 30 s, 3 s | A signal's phases; the cycle is 66 s |
| `traffic.PATIENCE` | 8 s | How long a car waits before it goes anyway |
| `traffic.LOOK` | 30 m | Where a car starts asking to enter a junction |
| `traffic.JUNCTION` | 7 m | The curve from one edge into the next |
| `traffic.MAX_ENTER` | 200 edges | New edges taken over per update |
| `rules.traffic.trip_minutes` | 15 min | How long a vehicle stays on the network |
| `rules.traffic.population` | 1 814 207 | Havana's people, the base of the estimate |
| `quality.traffic_radius` | 200–400 m | How far round the player the cars drive |

The modules are `contravia/flow.py` for the flow layer, `contravia/traffic.py` for the cars, `contravia/rules/traffic.py` for the demand, `contravia/clock.py` for the thread that keeps the flow up to the clock, `contravia/crashes.py` for the cars that become bodies, and `contravia/flowmap.py` for the picture. The graph they all read is the one the [road graph](#the-road-graph) chapter builds.
