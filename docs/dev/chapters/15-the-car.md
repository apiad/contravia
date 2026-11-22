# The car {#the-car}

A 1950s sedan has to lean on its soft springs, hold a slide on low-grip tyres and take sixty metres to stop from a hundred; a bus has to feel heavy and roll in a tight corner; a keyboard has to be able to drive both. The car is also stepped every frame while the city streams in around it, so the work per frame has to stay small. This chapter is about the physics under the car, the numbers each family of [the fleet](#the-fleet) gives it, and the six levels of help that sit on top.

## Three layers

The car is three layers, and the physics is the middle one. The driver's hands and feet (`contravia/input.py`, the ramps of the keys) give four numbers: steer, throttle, brake and handbrake. The assists correct those four before they reach the physics. The physics turns them into forces and motion, fixed at 120 steps a second. The world answers with the ground's height under each wheel, from [the street surface](#the-street-surface), and the walls the car can reach.

The physics never learns that the assists exist. A car at level 0 and the same car at level 5 obey the same laws; only the numbers handed to the physics differ. That is what lets the parts be tested apart: the handling tests drive with no assists at all, and the driving-help tests check the corrections alone.

## A body on four ray wheels

`manifoldx.vehicles` keeps every vehicle as one row of arrays, so the same code steps the player's car and, one day, the traffic's. The body is a rigid body: position, orientation as a quaternion, velocity, angular velocity, mass, and a diagonal inertia from a solid box of the body's size.

Each wheel has a mount on the body. From the mount a ray goes down the body's own vertical, as long as the suspension's travel plus the wheel's radius. Where it touches the ground is the wheel. The compression is what the ray covers beyond the ground, and from it come the spring's force, stiffness times compression plus damping times the rate of compression, never negative, and the anti-roll bar of the axle, which pushes each wheel by the difference of compression with its pair. The wheel's load is that spring force, so the weight transfer under braking is not written anywhere: the body dips, the front springs compress and the front tyres load.

The ground is read three times per wheel, at the contact and at ±0.6 of the wheel's radius along the car, and the wheel rests on the highest of the three. A 15 cm kerb is therefore climbed rather than hit. The game hands the physics its street surface as the ground, and the heights are asked with the wheel's previous height as a hint, so a wheel under a bridge stays under the bridge.

The step is fixed at `Vehicles.DT`, 1/120 s. A frame longer than `MAX_FRAME`, 0.05 s, runs as 0.05 s, and the remainder of a frame is carried to the next, so the car behaves the same at 30 and at 144 frames a second. The drawn car is not the last step's pose but the interpolation between the last two (`pose`), which is why it moves smoothly whatever the frame rate. `Driven.step` in `contravia/car.py` feeds the game's frame to `fleet.advance` and reads the result back.

## Tyres that slide

The tyre is where the car's weight becomes grip. At the contact, the wheel's speed is split along the wheel and across it. Their ratio to the ground's speed gives the slip ratio and the tangent of the slip angle. Each is measured against its own peak, and the two are combined into one length, so a tyre that brakes hard has less grip left for turning: the force never passes the friction the load allows, which is what makes the handbrake slide the tail.

The force follows the magic formula, `sin(C·atan(B·s − E·(B·s − atan(B·s))))`, with three numbers of the tyre's own; it rises, peaks and falls to about 80 % of the peak past the slip. The family's `tyre_mu` scales the whole curve: 0.85 for the almendrón, the guagua and the camión, 0.9 for the Soviet saloons and the huevito, 1.0 for the moderno and the patrulla.

Below `V_MIN`, 3 m/s, the tyre adds damping that fades in as the car stops, and every slip relaxes over `RELAX`, 0.3 m, of rolling. Without those two a parked car creeps and shivers. Each wheel also spins with its own inertia, so the throttle can spin a wheel and the brake can lock one; a locked wheel has no spin left and slides whole.

## Engine, gearbox, brakes

The engine is a torque curve through three points: idle at 850 rpm with 0.6 of the torque at peak power, peak torque at 0.6 of the peak-power rpm with 1.3 of it, and peak power at the family's `peak_rpm`, which runs from 2 200 rpm in the guagua to 6 000 in the moderno. Off the throttle it brakes at 0.02 N·m per rpm.

The clutch slips below 1 200 rpm, which is how a car pulls away from rest. The gearbox is automatic: with the pedal down it shifts up at 0.9 of the peak-power rpm, and with a light pedal as early as 0.4 of it; it shifts down when the lower gear would turn under 0.75 of that speed; a shift cuts the drive for 0.3 s. The differential is open, so the unloaded wheel of an axle spins first. The families have three to five gears and drive the rear, the front or all wheels.

The brake is a torque over all four wheels with `brake_front` as the front axle's share, 0.55 for the almendrón, the huevito, the guagua and the camión, and 0.6 to 0.65 for the rest; the handbrake works the rear axle only. Steering is Ackermann geometry at up to 0.6 rad at the outer wheel, so the inner wheel turns more. Air drag is `½·ρ·Cd·A·v²` on an area of 0.85 of the body's width times its height, and rolling resistance is 0.015 per newton of load.

## The numbers per family

`contravia/rules/handling.py` holds one row per family. The masses and powers are the reference presets; the mass is scaled by the built body's volume against its preset's, so a longer Garaje design weighs more, and the anti-roll bars and brake torque scale with it. Everything else is fixed per family.

| Family | Mass (kg) | Power (kW) | Drive | Gears | Final drive | Peak power at |
| --- | --- | --- | --- | --- | --- | --- |
| almendron | 1 600 | 75 | rear | 3 | 4.1 | 4 400 rpm |
| ruso | 1 030 | 52 | rear | 4 | 4.1 | 5 600 rpm |
| moscovita | 1 080 | 55 | rear | 4 | 4.2 | 5 800 rpm |
| moderno | 1 250 | 85 | front | 5 | 3.9 | 6 000 rpm |
| huevito | 620 | 18 | rear | 4 | 4.9 | 5 000 rpm |
| patrulla | 1 250 | 100 | front | 5 | 3.9 | 6 000 rpm |
| guagua | 11 000 | 180 | rear | 4 | 5.3 | 2 200 rpm |
| camion | 8 000 | 130 | rear | 5 | 4.5 | 2 600 rpm |

| Family | Springs (Hz) | Damping | Anti-roll (N/m) | Tyres μ | Brakes (N·m) | CG height | Cd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| almendron | 1.1 / 1.1 | 0.25 | 5 000 / 2 000 | 0.85 | 3 800 | 0.38 | 0.45 |
| ruso | 1.3 / 1.3 | 0.30 | 6 000 / 2 500 | 0.90 | 4 000 | 0.38 | 0.45 |
| moscovita | 1.2 / 1.2 | 0.30 | 5 500 / 2 500 | 0.90 | 4 000 | 0.40 | 0.45 |
| moderno | 1.6 / 1.6 | 0.35 | 9 000 / 4 000 | 1.00 | 5 500 | 0.38 | 0.32 |
| huevito | 1.4 / 1.4 | 0.30 | 3 000 / 1 500 | 0.90 | 2 200 | 0.38 | 0.40 |
| patrulla | 1.6 / 1.6 | 0.35 | 9 000 / 4 000 | 1.00 | 5 500 | 0.38 | 0.32 |
| guagua | 1.0 / 1.0 | 0.35 | 90 000 / 60 000 | 0.85 | 40 000 | 0.55 | 0.70 |
| camion | 1.2 / 1.2 | 0.35 | 60 000 / 40 000 | 0.85 | 30 000 | 0.55 | 0.80 |

The centre of gravity sits 0.45 of the wheelbase behind the front axle, at the stated share of the body's height. The spring rate is not a number in the table: it is worked out from the corner's mass and the stated frequency, and the damper from a share of critical damping, so a heavier Garaje design keeps its springs soft or firm in the same way.

A bus, a truck, a fire engine and a police car do not have their own row: `handling_of` gives them the guagua's, the camión's or the patrulla's. A livery family, a shared taxi or a state one, drives as the car it paints over. The cycles raise an error, because nobody drives them yet.

## What the handling tests hold it to

`tests/test_handling_targets.py` drives every family on a flat ground with no assists and compares the result with the targets of the vehicle-physics spec. The spec's first guesses for the almendrón, the guagua and the camión were replaced by what the physics gave with the spec's own masses and powers: 14 s became 18, 40 s became 24, 35 s became 24.

| Family | Target | Margin |
| --- | --- | --- |
| almendrón, ruso, moscovita | 0 to 100 km/h in 18, 17 and 18 s | ± 3 s |
| moderno | 0 to 100 km/h in 11 s | ± 2 s |
| patrulla | 0 to 100 km/h in 9 s | ± 2 s |
| huevito | 0 to 60 km/h in 12 s | ± 3 s |
| guagua, camión | 0 to 70 km/h in 24 s | ± 6 s |
| moderno, on a 40 m circle | 0.8 to 0.95 g held sideways | |
| almendrón, on a 40 m circle | 0.65 to 0.8 g held sideways | |
| moderno, from 100 km/h with ABS | a stop in 38 to 48 m | |
| almendrón, from 100 km/h, no assists | a stop in 50 to 65 m | |

```python {run}
import numpy as np
from manifoldx.vehicles import ASSISTS, FlatGround, Vehicles

from contravia.rules import handling

NONE = np.zeros(4)
ABS = np.array([1.0 if a == "abs" else 0.0 for a in ASSISTS])


def fresh(family):
    """The family's car at rest on flat ground, its springs settled."""
    fleet, ground = Vehicles([handling.spec_for(handling._preset_of(family), family)]), FlatGround()
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    for _ in range(60):
        fleet.step(np.zeros((1, 4)), ground)
    return fleet, ground


def accelerate(family, kmh):
    """Seconds to kmh at full throttle, and the curve of speed against time."""
    fleet, ground = fresh(family)
    t, curve = 0.0, [(0.0, 0.0)]
    while float(fleet.telemetry(0)["speed"]) * 3.6 < kmh and t < 60:
        fleet.advance(np.array([[0.0, 1.0, 0.0, 0.0]]), ground, None, 1 / 30, NONE)
        t += 1 / 30
        curve.append((t, float(fleet.telemetry(0)["speed"]) * 3.6))
    return t, np.array(curve)
```

```python {run continue}
def braking(family, strengths):
    """Metres to stop from 100 km/h, and the curve of speed against distance."""
    fleet, ground = fresh(family)
    fleet.velocity[0], fleet.gear[0] = (100 / 3.6, 0.0, 0.0), 3
    fleet.omega[0] = 100 / 3.6 / fleet.p.radius[0]
    curve = []
    while float(fleet.telemetry(0)["speed"]) > 0.3:
        fleet.advance(np.array([[0.0, 0.0, 1.0, 0.0]]), ground, None, 1 / 60, strengths)
        curve.append((float(fleet.position[0, 0]), float(fleet.telemetry(0)["speed"]) * 3.6))
    return float(fleet.position[0, 0]), np.array(curve)


TARGET = {"almendron": 100, "ruso": 100, "moscovita": 100, "moderno": 100, "patrulla": 100,
          "huevito": 60, "guagua": 70, "camion": 70}
for family, kmh in TARGET.items():
    print(f"{family:9} 0 to {kmh} km/h in {accelerate(family, kmh)[0]:4.1f} s")
print(f"from 100 km/h: moderno with ABS {braking('moderno', ABS)[0]:.0f} m, "
      f"almendrón with none {braking('almendron', NONE)[0]:.0f} m")
```

```python {run continue echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.6))
for family, colour, style in (("almendron", "#d62828", "-"), ("ruso", "#ffd400", "-"), ("moderno", "#0b0b0b", "-"),
                              ("patrulla", "#2f5d8c", "-"), ("guagua", "#8a867b", "-"), ("camion", "#5f5c55", "--")):
    t, curve = accelerate(family, TARGET[family])
    left.plot(curve[:, 0], curve[:, 1], color=colour, ls=style, lw=1.4,
              label=f"{family}, {TARGET[family]} km/h in {t:.1f} s")
for family, strengths, colour, name in (("moderno", ABS, "#0b0b0b", "moderno, ABS"),
                                        ("almendron", NONE, "#d62828", "almendrón, no assists")):
    metres, curve = braking(family, strengths)
    right.plot(curve[:, 0], curve[:, 1], color=colour, lw=1.4, label=f"{name}: {metres:.0f} m")
left.set(xlabel="seconds", ylabel="km/h", title="Full throttle from rest")
right.set(xlabel="metres", ylabel="km/h", title="Braking from 100 km/h")
for ax in (left, right):
    ax.grid(color="#eeeeee", lw=0.6)
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    ax.tick_params(labelsize=7)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## Six levels of driving help

`contravia/driving_help.py` is the game's answer to a keyboard: six levels, from almost autopilot to a real car. Each level sets the strength of manifoldx's four chassis assists, steering, traction, anti-lock brakes and stability, and, from level 2 down, turns on the road assists, which read the streets.

| Level | Name in the game | Chassis assists | Road assists |
| --- | --- | --- | --- |
| 5 | Real | none | none |
| 4 | Chasis | traction, ABS and stability at 0.6; steering at 0 | none |
| 3 | Dirección asistida | all four at 1 | none |
| 2 | Mantener senda | all four at 1 | lane keeping |
| 1 | Freno de emergencia | all four at 1 | lane keeping, emergency brake |
| 0 | Casi piloto | all four at 1 | lane keeping, emergency brake, pilot |

The default level is 3. The choice is saved with the options; while driving, L turns it to the next level (5, 4, down to 0, then 5 again), applies it, saves it and names it on the HUD for a moment.

The steering assist narrows the wheel's lock with speed, down to a fifth of it at 45 m/s, and steers into a slide when the wheel is released. It never lets the front tyres past their peak slip angle. The traction assist cuts torque from a spinning driven wheel and recovers at 8 per second. The anti-lock takes 15 % of a locking wheel's brake away each step and gives it back at 3 per second once the wheel turns. The stability assist compares the yaw rate with the one the wheel asks for, limited by the grip: it brakes the front wheel outside the turn when the car yaws too much, the rear wheel inside it when too little, and cuts half the drive in that case.

Lane keeping and the pilot start from `Road.locate`, which finds the edge the car is on. The province's road graph is indexed on `CELL`, 16 m, cells, every segment cut into `PIECE`, 8 m, pieces filed by their midpoint, so a position reads the three-by-three cells round it. A car is on an edge when it is within `REACH`, 12 m, of its centreline and heading within 60° of its direction; the answer adds how far along the edge it is, how far right of the centreline, and which lane that falls in.

Lane keeping steers for a point `max(6 m, 0.9 s × speed)` ahead by pure pursuit, through the arc that passes it. It mixes in only while the driver's wheel is within `MIX`, 0.15, of centre, so a driver who steers keeps the car. A hold of the wheel past 0.5 for 0.3 s asks for the next lane on that side, and the offset eases toward it at `LANE_SPEED`, 1.1 m/s, about three seconds a lane. Below `MOVING`, 2 m/s, all of it stands down.

The emergency brake looks at the segments in the car's path, its width plus `MARGIN`, 0.3 m, each side and bent by the yaw rate. It brakes when stopping short would take more than `A_SOFT`, 3 m/s², and fully at `A_HARD`, 6 m/s², aiming `STOP_AT`, 2 m, short of the obstacle; creeping below 2 m/s it only holds a car within `HOLD`, 1.5 m. A swerve past 0.6 of the wheel, or the handbrake, stands the brake down at once.

The pilot, at level 0, drives with the intelligent driver model the traffic uses, toward the edge's free-flow speed, capped by a wanted speed the accelerator and brake move at `WANT_STEP`, 3 m/s², and slowed for a coming turn at 3 m/s² of lateral acceleration on an 8 m radius. It stops for a red light at the stop line, brakes for the nearest thing in its path, and holds at the destination until the accelerator is pressed. Within `TURN_AT`, 40 m, of a junction a tap of the wheel chooses the exit rather than steering, so the driver only indicates. An exit within 30° of straight is straight on; with the GPS routing, it moves to the turning lane 80 m before the turn and eases to the kerb 30 m before the destination. Pulling the handbrake hands the car back until the accelerator is pressed and released.

The game gives these three what they need once a frame: the car's place on the road, the walls it could reach (`brake_reach` looks `v²/6 + 12 m` ahead for the emergency brake), the traffic and crashed cars as moving segments with their speeds, and the province's signals. A province change builds a new road index and a new help, since the edge numbers are the province's own.

## The adapter

`contravia/car.py` keeps the game's view of all this and nothing of the physics. `CarState` is where the car is, with yaw 0 facing +x and a positive yaw turning toward +z, its speed, the ground under it, its pitch and roll, its sideways speed, its gear and its rpm, and the last step's telemetry. `Controls` is the four numbers the driver gives. `settle` puts a footprint of 2.6 by 1.53 m on four heights, for teleports and for the checks that need a car standing on its wheels rather than a simulation.

The adapter is also what survives a hot reload: `snapshot` takes the twelve arrays of the physics state as plain lists and `restore` puts them back, so a saved file rebuilt by `contravia dev` leaves the car where it was, mid-slide if it was sliding. `seen` returns the car where it is drawn, between two steps, which is what the chase camera follows; the camera is at its farthest at `CAMERA_FAST`, 45 m/s. `contravia play` hands the whole chain its inputs each frame, and the HUD reads the gear and the rpm from the physics rather than from a table.
