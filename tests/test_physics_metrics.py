import numpy as np
from manifoldx.vehicles import FlatGround, strengths_for

from contravia.car import CarState, Controls, Driven
from contravia.rules import fleet, handling

NONE = np.zeros((0, 2, 2), np.float32)
IDLE = np.zeros((1, 4), float)  # the second body's controls, as Crashes.controls() hands them over
SPEC = handling.spec_for(fleet.moderno(), "moderno")
HELP = strengths_for(30)


def fresh():
    """A Driven settled nowhere: its accumulator starts at 0, so the substep counts read off the frame length."""
    return Driven(SPEC, CarState(0.0, 0.0, 0.0, 0.0), FlatGround())


def test_substeps_start_at_zero_and_stay_zero_on_a_short_first_frame():
    d = fresh()
    assert d.substeps == 0 and d.state.telemetry["substeps"] == 0
    s = d.step(Controls(0, 0, 0, False), 1 / 240, NONE, HELP)
    assert s.telemetry["substeps"] == 0  # a frame shorter than the leftover accumulator runs nothing


def test_the_leftover_accumulator_carries_into_the_next_call():
    d = fresh()
    d.step(Controls(0, 0, 0, False), 1 / 240, NONE, HELP)
    s = d.step(Controls(0, 0, 0, False), 1 / 240, NONE, HELP)
    assert s.telemetry["substeps"] == 1  # 2/240 s adds up to one 120 Hz step
    whole = fresh()
    once = whole.step(Controls(0, 0, 0, False), 1 / 120, NONE, HELP)
    assert (s.x, s.z, s.speed) == (once.x, once.z, once.speed)  # two half frames simulate the same one step


def test_a_frame_of_60_and_30_fps_runs_2_and_4_steps():
    assert fresh().step(Controls(0, 0, 0, False), 1 / 60, NONE, HELP).telemetry["substeps"] == 2
    assert fresh().step(Controls(0, 0, 0, False), 1 / 30, NONE, HELP).telemetry["substeps"] == 4


def test_a_huge_frame_is_clamped_at_six_steps():
    d = fresh()
    s = d.step(Controls(0, 0, 0, False), 1.0, NONE, HELP)
    assert s.telemetry["substeps"] == 6  # MAX_FRAME is 0.05 s, six steps of 1/120


def test_bodies_count_the_car_and_the_crashed_traffic_that_joined():
    d = fresh()
    s = d.step(Controls(0, 0, 0, False), 1 / 60, NONE, HELP)
    assert s.telemetry["bodies"] == 1
    d.fleet.add(SPEC)  # a crashed traffic car joins, as Crashes does
    s = d.step(Controls(0, 0, 0, False), 1 / 60, NONE, HELP, others=IDLE)
    assert s.telemetry["bodies"] == 2 and s.telemetry["substeps"] == 2


def test_the_counters_ride_on_the_returned_car_state():
    d = fresh()
    s = d.run(Controls(0, 0, 0, False), 1.0, NONE, HELP)
    assert type(s.telemetry["substeps"]) is int and type(s.telemetry["bodies"]) is int
    assert s.telemetry["substeps"] == 2 and s.telemetry["bodies"] == 1
