# The development loop {#the-development-loop}

A rule change has to be seen on the island. Restarting the game for every edit would close the window, open the world index again and wait for the cells round the car, which turns a one-line change into a minute of waiting. `contravia dev` keeps one window, one sound process and one `engine.state` across saves, and rebuilds only the game.

## One save, one rebuild

`contravia dev` takes the options of `contravia play` and runs the app under manifoldx's hot reloader. The command is the one manifoldx documents:

```text
manifoldx run -m contravia.cli -- play [the options of contravia play]
```

Start it at the repository root: in module mode the reloader takes the folder it runs in as the app's folder, and watches the modules of that folder.

The reloader opens the window once and runs the app in it. It treats every module the app imported from the repository as a source file, so a save to any module under `contravia/` is a change. The game adds two more things to that list with `engine.watch`: `contravia/regions.toml` and the `*.src` files of the island folder. A save to the region rules or a rewritten side file therefore reloads like a source save. The data files in the package that are not modules, such as `places.json`, `towns.json` and `sky_data/`, are not watched.

The reloader checks its files every `poll`, 0.25 s. When one has changed it drops the app's modules from `sys.modules`, imports them again, runs the app, and gives the new engine the old window, the old GPU device and the same `engine.state`. The whole rebuild is a new `Game` on that state, not a new process.

A module is dropped, not patched. A function edited in `contravia/streets.py` is the new function; an object made from a class of that module before the save belongs to a class the interpreter no longer has. This is the rule behind everything below: `engine.state` may hold numbers, strings, lists, dicts, numpy arrays and objects of installed libraries, and never an instance of a `contravia` class.

<figure>
<svg viewBox="0 0 640 540" width="100%" role="img" aria-labelledby="loop-title" font-family="ui-sans-serif, system-ui, sans-serif">
<title id="loop-title">One save under contravia dev: the running game is stopped and a new one is built on the same state</title>
<defs>
  <marker id="loop-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#0b0b0b"/>
  </marker>
</defs>

<text x="14" y="24" font-size="11" font-weight="700" fill="#d62828" letter-spacing="1.4">ONE SAVE UNDER contravia dev</text>

<rect x="14" y="40" width="612" height="58" rx="3" fill="#fff8d6" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="28" y="61" font-size="12.5" font-weight="700" fill="#0b0b0b">01 · a file is saved</text>
<text x="28" y="79" font-family="ui-monospace, monospace" font-size="10.5" fill="#555">contravia/*.py · contravia/regions.toml · &lt;island&gt;/*.src</text>
<text x="28" y="93" font-size="9.5" fill="#777">any module the app imported from the repository, and the two paths engine.watch names</text>
<path d="M320 98 V124" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#loop-arrow)" fill="none"/>
<text x="332" y="115" font-size="9.5" fill="#777">within poll = 0.25 s</text>

<rect x="14" y="130" width="612" height="74" rx="3" fill="#f6f4ef" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="28" y="151" font-size="12.5" font-weight="700" fill="#0b0b0b">02 · HotReloader.reload</text>
<text x="28" y="169" font-size="10.5" fill="#555">drops every module of the app, imports them again and runs the app in the same window</text>
<text x="28" y="187" font-size="10" fill="#d62828">a broken save keeps the running version; a runtime error pauses until the next edit</text>
<path d="M320 204 V230" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#loop-arrow)" fill="none"/>

<rect x="14" y="236" width="612" height="74" rx="3" fill="#f6f4ef" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="28" y="257" font-size="12.5" font-weight="700" fill="#0b0b0b">03 · the old engine detaches</text>
<text x="28" y="275" font-size="10.5" fill="#555">fires shutdown, so Game._shutdown closes the streamer, the detail ring, forest, grass, graveyard and walls,</text>
<text x="28" y="291" font-size="10.5" fill="#555">and stops the flow's thread. The canvas, the GPU device and the sound process stay.</text>
<path d="M320 310 V336" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#loop-arrow)" fill="none"/>

<rect x="14" y="342" width="612" height="74" rx="3" fill="#f6f4ef" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="28" y="363" font-size="12.5" font-weight="700" fill="#0b0b0b">04 · a new Game on the same engine.state</text>
<text x="28" y="381" font-family="ui-monospace, monospace" font-size="10.5" fill="#555">World.load(data) · start_builders() · Game(engine, city, builders=...)</text>
<text x="28" y="397" font-size="10.5" fill="#555">the state carries the car, the GPS destination, the quality, the sound, the boat and the kept shelf</text>
<path d="M320 416 V442" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#loop-arrow)" fill="none"/>

<rect x="14" y="448" width="612" height="54" rx="3" fill="#fff8d6" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="28" y="469" font-size="12.5" font-weight="700" fill="#0b0b0b">05 · drive on from the same place</text>
<text x="28" y="487" font-size="10.5" fill="#555">Game.__init__ sees "car" in engine.state, resumes the pose, the physics and the camera, and keeps the drive</text>

<text x="14" y="524" font-size="10" fill="#777">A rebuilt province file is not watched: after contravia streets or contravia walks, save a watched file or restart.</text>
</svg>
<figcaption>A save drops the app's modules, stops the old game through its `shutdown` event, and builds a new `Game` on the state the old one left.</figcaption>
</figure>

## What survives: engine.state

`engine.state` is a `manifoldx.hot.State`, a dict whose keys read and write as attributes. It is the only thing manifoldx carries from one version of the code to the next. The game writes it every frame and again whenever a player changes an option.

| Key | Written by | What it carries |
| --- | --- | --- |
| `car` | `Game.saved_car`, at the end of every drive frame | The pose `[x, z, yaw, speed]`, the height `y`, the camera's eye and `driven.snapshot()`, the physics' own state |
| `craft` | `Game.board` | The kind, `boat` or `car`, and the boat's snapshot when one is steered |
| `gps_to` | `Game.gps_to` and `gps_clear` | The destination `[x, z, title]`, set again on the new guide |
| `quality` | `contravia/quality.py` | The preset's name, every knob, whether it is pinned, and the last error |
| `sound` | `Sound.start` | The live sound process, reused when it is alive |
| `kept` | `contravia/kept.py` | The shelf of results that take seconds to build |
| `teleport`, `gps`, `traffic`, `sky`, `audio`, `map`, `curtain` | The frame | Reports of the last frame, read by a person or an agent |

`Game.saved_car` runs at the end of the drive system, so the state holds the car as this frame left it. On the next start, `Game.__init__` tests `"car" in engine.state`. When it is there, `_resume` settles the car on the surface level nearest the saved `y`, places the physics and the camera, and restores the speed, the gear and the slide. A boat is restored from `craft` in the same way.

`quality` survives because the preset decides module-wide knobs in `forest`, `grass`, `rain` and `atmosphere`, and the new `Game` has to set them again; it reads the saved preset and every knob before it builds the scene. `gps_to` survives because the router's graph is expensive to rebuild and the destination is the player's; `_make_gps` asks for the route again on the new guide.

The reports are not inputs. The old game wrote them; the new game writes them again as it does those things, on its first frame for the ones a frame writes. Until then they are the last report of a game that no longer exists.

## The old game stops

A `Game` registers `_shutdown` on the `shutdown` event. The reloader calls the old engine's `_detach` before it runs the new app, and `_detach` fires `shutdown` at once. `_shutdown` closes the city streamer, the detail ring, the forest, the grass, the graveyard, the wall builder and the flow's thread. This is the only place the old game is stopped.

Without it, a save would leave the old builders turning cells into a window nobody reads, and the old traffic thread stepping a graph that is about to be replaced. Both would compete with the new game for the same cores.

The sound is the exception. `Sound.start` reads `engine.state.sound`; when it finds a live process whose controls still cover `CONTROLS`, it calls `audio.reload(MODULES)` and returns the same handle. The audio modules are imported again inside the sound process, so an edited engine graph, tyre model or music style is the one heard after the save, without a new process. The music process, which composes two bars ahead, is a child of the sound process and stays alive with it. A new control added to `CONTROLS` makes an older process incomplete, and then `Sound.start` closes it and starts a new one.

## Results that survive a save: kept()

Some results take seconds and depend only on their inputs: the player's car mesh, the traffic tables of a province's graph, the GPS's routing graph, the looks of the traffic's vehicles. `contravia/kept.py` keeps such a result in `engine.state` under a name, with a stamp of what it was built from. A later `kept(state, name, stamp, build)` with the same name and the same stamp returns the result; a different stamp calls `build()` again.

The stamp is the caller's. It includes `source(path)`, the modification time in nanoseconds of the file whose code builds the result, so a save to that file builds it afresh. The other part of a stamp says what the result depends on: the province's graph length, the route version, the manifoldx version.

| Kept result | Stamp | Built by |
| --- | --- | --- |
| `player car` | `"CarParams()"` | `_player_car`, a `CarParams` model cached on disk per manifoldx version |
| `traffic tables` | `source(traffic.__file__)` and the graph's length | `traffic.Cars(graph).tables()`, also baked beside the province's `.npz` |
| `traffic looks` | `_stamp()`, a hash of `vehicles.py`, `fleet.py` and manifoldx's version | The mesh of every vehicle look, shared by the traffic's pools |
| `gps route graph` | The world's source path and `gps.ROUTE_VERSION` | `RouteGraph.for_world`, the joined routing graph of about 42 MB |

The rule from the introduction applies here with force: a kept result is plain data or an object of an installed library. A `contravia` object would survive the save with its class gone, and the next frame would call a method that no longer exists.

```python {run}
from contravia.kept import kept, source

state, builds = {}, []
stamp = (source("../../contravia/kept.py"), "the traffic tables")

def build():
    builds.append(len(builds) + 1)
    return f"tables built from the code as it was, build {len(builds)}"

first = kept(state, "traffic tables", stamp, build)
again = kept(state, "traffic tables", stamp, build)
after_save = kept(state, "traffic tables", (*stamp, "a newer save"), build)
print(f"{len(builds)} builds for three reads: {(first, again, after_save)}")
```

## A script that builds a Game

A probe of your own can build `Game` directly: a custom camera, a measurement, a screenshot the game's own flags do not make. Two rules keep it working.

The first is that the code runs under `if __name__ == "__main__":`. The streamer starts its builders with `spawn`, and a spawned process imports the script again to find the work. A script without the guard fails with "start a new process before the current process has finished its bootstrapping phase".

The second is the order: `HAVANA_DATA` names the built island before any builder starts, and the builders are started before the world is loaded in the main process, so both read it at once. `contravia play` does exactly this in the `main` of `contravia/game.py`, and `start_builders` is the function it calls.

```python
import os

import manifoldx as mx

from contravia.game import Game, start_builders
from contravia.world import World

if __name__ == "__main__":
    data = "data/cuba"
    os.environ["HAVANA_DATA"] = os.path.abspath(data)
    builders = start_builders()
    engine = mx.Engine("probe", width=1280, height=720)
    game = Game(engine, World.load(data), builders=builders)
    # ... place the camera, render frames, read game.car and engine.state, save what you measured
    game._shutdown()  # close the streamers and the traffic thread before the process ends
```

For a picture of a screen the game already has, `contravia play --shot` is shorter and needs no script: `contravia play --screen garage --shot garage.png` renders the screen offscreen and exits. A script is for what no flag covers.

## Events into a running game

Under `contravia dev`, the app answers a small API. The command `manifoldx emit <event> '<json>'` posts a JSON payload to it, and the game handles the event at the start of the next frame. `manifoldx events` lists every event with its handler's docstring, which is the documentation of its payload.

| Event | Payload | What it does | Read it in |
| --- | --- | --- | --- |
| `teleport` | `{"lat": 20.02, "lon": -75.83}` or `{"x": 300.0, "z": 200.0}` | Puts the car on the nearest road, stopped | `state.teleport`, `state.pos` |
| `gps` | The same two forms, plus an optional `title` | Sets a destination and finds a route | `state.gps` |
| `gps_clear` | none | Drops the destination and the route | `state.gps` |
| `traffic_config` | `{"population": 3000000}` | Changes the demand settings the file names | `state.traffic` |
| `close` | `{"x": 300.0, "z": 200.0, "minutes": 40}` or `{"edge": 1234}` | Closes a street for a while | `state.traffic` |
| `quality` | `{"preset": "bajo"}` or `{"rain_pool": 6000}` | Plays at a preset or changes one knob | `state.quality` |
| `hlod` | `{"on": 0}` | Draws the level-1 summaries with their own colours | none |
| `sky_config` | `{"city_glow": 0}` | Changes a sky setting | `state.sky` |
| `weather` | `{"state": "storm", "minutes": 30}` | Holds a weather state | `state.sky` |
| `vehicle` | `{"kind": "boat"}` or `{"kind": "car"}` | Boards a boat from the nearest water, or returns to the car | `state.craft` |
| `map` | `{"x": 300.0, "z": 200.0, "h": 220000, "yaw": 30}` or `lat` and `lon` | Opens the map with its camera there | `state.map` |

The payload is JSON, not `key=value`: the server decodes the body with `json.loads` and answers 400 to anything else. A refused setting does not change the game; the handler writes the reason into the state key instead, under `error`.

The same server answers `manifoldx state [path]` with the whole state or one dotted key (`manifoldx state car.pose`), `manifoldx look` with a PNG of the next frame, `manifoldx metrics` with the last seconds of frame times under `--debug`, and `manifoldx log` with the app's log lines. These are how a person or an agent checks a screen without watching it. The dev server exists only under `manifoldx run`, so under `contravia play` these commands find no app to talk to.

## What the loop does not cover

A rebuilt province file is not watched. After `contravia streets`, `contravia walks` or another command that rewrites a province's `.npz`, save a watched file to force a reload, or restart `contravia dev`. The side files (`*.src`) are watched, so rewriting one reloads by itself. The reload reads the world index again and the new builders open the province files afresh.

A save that does not parse or import leaves the running version in place: the reloader prints the traceback and keeps the game that was already running. A save that runs but raises in a frame pauses the game until the next edit. Both keep the window, so the fix is visible in the same place.

## The modules behind this

| Piece | Module | What it owns |
| --- | --- | --- |
| The loop | `manifoldx.hot`, `contravia/cli.py` | The poll, the module drop, the app run, the `dev` subcommand |
| The game it rebuilds | `contravia/game.py` | `Game`, `saved_car`, `_resume`, `_shutdown`, `start_builders` |
| The state | `manifoldx.hot.State`, `contravia/quality.py`, `contravia/gps.py` | The dict and the keys the game keeps in it |
| Kept results | `contravia/kept.py`, `contravia/baked.py` | Stamps, the shelf, the on-disk cache |
| The sound that stays | `contravia/audio/__init__.py` | The process, `CONTROLS`, `MODULES`, `reload` |
| Events and reports | `manifoldx.devchannel`, `contravia/game.py`, `contravia/menu.py` | The API, the handlers, the state keys |
