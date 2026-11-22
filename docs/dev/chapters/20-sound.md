# Sound {#sound}

An engine that does not rise with the rpm, tyres that do not hiss past their grip and a city that sounds the same in a barrio and on the coast would make the car feel like a picture. The game could record all of that, but recordings bring licences, they make every design of a family sound alike, and they cannot follow a car the player edits in the Garaje. So every sound is synthesized while the game runs. The repository and the build hold no audio file.

The sound also has to survive the game's own garbage collector. Python's collector stops every thread of the game for a moment, and a click at every collection is worse than no sound. Sound therefore runs in its own process with a small heap, and the music in a second one, which renders seconds ahead so its own stalls are never heard.

## Two processes and five buses

The game writes, once per frame, a row of numbers per sound object: the player's car, the four loudest traffic cars, the weather, the biome and the listener. It also pushes one-shot events, a crash with its impulse, a horn, a menu click, onto a queue. `manifoldx.audio` keeps those numbers in shared memory and plays whatever the sound process builds from them.

The sound process builds a Euterpe graph per source and mixes them into five buses: `engine`, `effects`, `ambience`, `music` and `ui`. It is started once and lives as long as the interpreter. `contravia dev` keeps it across a hot reload and re-imports `contravia/audio/` in it, so an edit to an engine's sound is heard in about three seconds, like the rest of the game.

The music is a separate process. It reads three numbers, the style, the intensity and the date, composes two bars at a time, and keeps two bars rendered beyond what the sound process has played, with `2.5` s of tail. When the style changes it starts the new music at the old one's next bar line and crossfades with equal power over `3` s.

## The engine

An engine is one graph, built from the family's numbers in `contravia/rules/sound.py` and the vehicle's idle and peak-power rpm from its physics. The heart of it is a wavetable that holds one whole four-stroke cycle: as many decaying pulses as the engine has cylinders, each with its own amplitude from a seeded draw, so a V8 burbles and an inline four buzzes. The table plays once per cycle, at `rpm / (30 × strokes)` cycles per second, which puts its firing frequency at `rpm × cylinders / 120` for a four-stroke and `rpm / 60` for a two-stroke. Because the frequency comes from the table's playback rate rather than from a crossfade, the engine rises through the revs without a seam.

The pulses ring two resonances of the exhaust, a band-pass at each, and pass a low-pass whose cutoff opens with the throttle and with the rpm against the redline. Intake noise rises with the throttle and the rpm. A V8 and a Lada pop on a lift-off at high revs, a diesel clicks with every firing, and the metroguagua adds a turbo whine with load. All of it goes through a soft clipper, `Tanh`, so a hard rev does not clip the mix.

| Family | Cylinders | Resonances (Hz) | Brightness (Hz) | Character |
|---|---|---|---|---|
| almendron, compartido | 8 | 85, 260 | 900 | uneven firing, crackle on lift-off |
| ruso, moscovita | 4 | 170, 540 | 2 200 | brighter, a little rasp |
| moderno, estatal, policia | 4 | 150, 480 | 1 200 | even and muffled |
| huevito | 3 | 150, 480 | 1 200 | even and muffled |
| guagua_vieja, camion_80, furgon, cisterna, volquete, bomberos, interprovincial, pasajeros | 6 | 70, 210 | 1 400 | diesel knock with every firing |
| metroguagua | 6 | 70, 210 | 1 400 | diesel, softer, with a turbo |
| moto_dos_tiempos | 1, two-stroke | 300, 900 | 3 000 | thin and buzzy, fires every turn |
| moto_electrica, triciclo_carga, triciclo_pasajeros | none | | | a whine that follows speed |
| bicitaxi, bicicleta | none | | | chain clicks that follow pedalling |

A family with no pulses gets an electric whine that rises with speed, or, for a bicycle, clicks at the pedalling cadence. The player's engine is the family's graph at `2.4` times its level, about `−25` LUFS at a cruise, because it leads the mix.

```python {run echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from euterpe.sound import render
from PIL import Image

from contravia.audio.engine import cycle, engine
from contravia.rules.sound import FAMILY_SOUND

RATE = 48_000
snd = FAMILY_SOUND["almendron"]


def fixed(values):
    return lambda name: values.get(name, 0.0)


fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.6))
for family, colour in (("almendron", "#d62828"), ("ruso", "#0b0b0b")):
    row = FAMILY_SOUND[family]
    wave = cycle(row.cylinders, row.uneven, seed=1, size=512)
    left.plot(np.arange(len(wave)) / len(wave), wave, color=colour, lw=1.2,
              label=f"{family}: {row.cylinders} cylinders")
left.set(xlabel="one four-stroke cycle", ylabel="amplitude")
left.legend(fontsize=8, loc="upper right", frameon=True, framealpha=1.0, edgecolor="none", facecolor="#f6f2e8")

for k, rpm in enumerate((900, 1800, 2700, 3600, 4500)):
    x = render(engine(snd, 600, 4600, fixed({"car.rpm": rpm, "car.throttle": 0.6}), seed=1), 0.5)
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    band = (f > 20) & (f < 1200)
    base = k * 1.1
    right.fill_between(f[band], base, base + mag[band] / mag[band].max(), color="#d62828", alpha=0.75, lw=0)
    fire = rpm * snd.cylinders / 120.0
    right.plot([fire, fire], [base, base + 1], color="#0b0b0b", lw=0.8)
    right.text(1200, base + 0.3, f"{rpm} rpm · {fire:.0f} Hz", fontsize=7.5, ha="right")
right.set(xlim=(20, 1250), yticks=[], xlabel="frequency, Hz")
left.grid(color="#eeeeee", lw=0.6)
fig.suptitle("the almendrón's V8: one cycle table, and its spectrum at five engine speeds", fontsize=10)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The black ticks mark the firing frequency, `rpm × 8 / 120`. The harmonics above it are what the exhaust resonances and the low-pass shape, and they are why two engines at the same rpm do not sound alike.

## Tyres, wind, crashes and the horn

The tyre's roll is filtered pink noise, darker on asphalt and brighter and louder off it. The graph knows five surfaces, `0` on a street, `1` on a pavement, `2` on grass or earth, `3` on sand and `4` on rock, each with its own low-pass and level. The game reports only `0` and `2` today: `0` when the car is within reach of a street, or when there is no road index, and `2` otherwise, so a pavement sounds like asphalt and the sand and rock voices are never heard. The squeal is a wobbling band of noise near `1 050` Hz that rises with the worst wheel's slip past its tyre's peak, and it is duller on a wet road.

Wind is two bands of noise crossed by speed, and a cabin control, always `0` today because there is no cabin view, would muffle it. A crash needs an impulse of `300` N·s to be heard: a low thump, inharmonic metal from a modal model, and glass above `3 000` N·s, louder and longer with the impulse, up to four at once. A kerb or a pothole is a thump from how far the suspension moved in one frame.

The horn is on H. Its voice is the family's: two saws at `350` and `440` Hz for an almendrón, one saw at `420` for the other cars, an air horn at `220` and `277` with noise for a guagua or a truck, and a thin square at `600` for a motorbike or a tricycle. A bicycle rings a bell instead.

## The traffic

The traffic cars have no engines of their own; `contravia/traffic.py` moves them along lanes without a rpm. The four loudest at the listener get a cheap engine each, a pulse train through one fixed low-pass, fed by a virtual gearbox: a gear every `25` km/h for a car, `12` for a heavy vehicle and `20` for a motorbike, from idle to `70 %` of the redline within a gear, plus `15 %` of it while accelerating. The rest of the cars within `80` m are one bed that thickens with how many there are.

A car that has waited `4` s within `30` m of a stopped player honks once per wait, in its family's voice. A bus that comes to a stop lets go of its air brakes with a hiss. The four slots are chosen by distance, speed and weight, so a truck is heard further than a car and a motorbike nearer.

## Rain, thunder and the sea

Rain is a graph read from the weather's intensity, the car's wetness and its speed: a hiss of high noise, sparse drops, and spray from the tyres that grows with speed on a wet road. Thunder is drawn in the game process. In a storm a strike happens every `25` s on average, between `300` and `6 000` m away, and it is heard `distance / 343` seconds later: a fraction of a second for a near strike, up to seventeen seconds for a far one. A nearby strike is a sharp crack and a short rumble; a far one is all rumble.

A car that drives into water deeper than `0.7` m makes a splash, and for the `2.6` s it sinks the whole mix is low-passed to `400` Hz, with bubbles rising and the tyre and wind sounds muffled. It all comes back when the car stands on the shore.

## The biome, the ambience and the music

The sound of a place is a biome, and the biome is resolved from what the game already knows under the car. The sun below `−6°` makes it night. Sea within `300` m makes it coast. A built-up share of at least `0.5` in a dense municipio makes it city, and at least `0.15` a barrio; everything else is countryside. A region in `contravia/regions.toml` may name the biome or only the music; today the file gives Guantánamo changüí. A new biome takes over after `10` s in it, so a short cut through a park does not change the music.

| Biome | Music | Ambience |
|---|---|---|
| city | son montuno, 100 bpm | a murmur, distant guaguas, a dog |
| barrio | guaguancó, 106 bpm | a nearer street, a dog, a rooster at dawn |
| countryside | guajira, 180 eighth-notes a minute | birds, a rooster, a cow, a dog |
| coast | changüí, 118 bpm | waves and gulls |
| night | bolero, 72 bpm, minor | crickets, frogs, a tres from a far window |

Each biome has a continuous bed that crosses to the new one over `3` s, and a few one-shots scattered at random bearings around the listener, at their own mean intervals and inside their own hours. The city's dog barks every `40` s on average, the countryside's bird every `6` s between `05:30` and `19:00`, and the night's tres every `70` s.

The music is composed bar by bar from a seed of the style and the game's date, so the same day plays the same music. Each style writes a chord progression, a bass, a percussion pattern and a melody from its own rules, and chooses sections: the son opens with the tres alone and breaks to the percussion now and then, the guajira and the bolero rest their melody in an interlude. The menu plays the city's son at half intensity. A new day gives a new seed, so the music changes with the calendar without ever repeating a recorded loop.

## The mix

The options give five rows, «Volumen general», «Motor», «Efectos», «Ambiente» and «Música», from `0` to `10`; their defaults are `8`, `7`, `7`, `6` and `5`. A bus's gain is the general row over ten times its own row over ten, and the interface bus takes only the general row. Outside the drive the engine and the effects are silent and the rest plays on; the pause also lowers the music by `10` dB. A crash ducks the music by `6` dB for `1.5` s.

`contravia play --mute` starts the sound with the null backend, which pulls the blocks and drops them, and a shot never plays sound. `contravia play --audio-out drive.wav` records the mix to a WAV, with one WAV per bus beside it, which is how a bench run can be listened to and how a test checks for underruns. `engine.state.audio` reports the voices, the levels, the sound process's CPU and its underruns; `contravia bench` adds the underruns and the CPU of both processes to its line.

## The numbers

| Constant | Value | What it decides |
|---|---|---|
| `audio.BUSES` | 5 | engine, effects, ambience, music, ui |
| `rules.sound.FAMILY_SOUND` | one row per family | cylinders, strokes, resonances, brightness, character |
| firing frequency | rpm × cylinders / 120 | A four-stroke's pulses per second; rpm / 60 on a two-stroke |
| `car.SURFACES` | 500 to 5 000 Hz | The low-pass and level of each surface |
| `car.CRASH_MIN`, `GLASS_MIN` | 300, 3 000 N·s | When a hit is heard, and when it breaks glass |
| `traffic.SLOTS` | 4 | Traffic cars with an engine of their own |
| `traffic.BED_NEAR` | 80 m | Cars within this counted into the bed |
| `traffic.WAIT_HONK` | 4 s | A wait behind the player that earns a horn |
| `weather.MEAN_GAP` | 25 s | The average time between strikes in a storm |
| `weather.NEAR`, `FAR` | 300, 6 000 m | How close and how far a strike may be |
| `weather.SOUND` | 343 m/s | The speed that delays the thunder |
| `biome.HOLD` | 10 s | Time in a new biome before it is heard |
| `biome.NIGHT_SUN` | −6° | The sun's height that makes it night |
| `player.AHEAD` | 2 bars | Music kept rendered beyond what plays |
| `player.FADE` | 3 s | The equal-power crossfade between styles |

The modules are `contravia/audio/__init__.py` for the process, the controls, the buses and the events; `contravia/audio/engine.py`, `car.py`, `traffic.py`, `weather.py`, `ambience.py`, `biome.py` and `ui.py` for the sounds; `contravia/audio/music/` for the five styles and the composer; and `contravia/rules/sound.py` for the families' numbers. The rpm, the slip and the impulse they read come from the [car](#the-car) and the [traffic](#traffic).
